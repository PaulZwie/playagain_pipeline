"""
TCP Prediction Server for Unity Integration.

Runs the gesture prediction pipeline and sends predictions
to connected clients (e.g., Unity game) over TCP.

Protocol: Newline-delimited JSON messages.

**Outgoing (Python → Unity):**
    Predictions:
        {"gesture": "fist", "gesture_id": 3, "confidence": 0.95, "probabilities": {...}, "timestamp": ...}
    Handshake (on connect):
        {"type": "handshake", "model_name": "...", "class_names": {...}, ...}

**Incoming (Unity → Python):**
    Game state (ground truth):
        {"type": "game_state", "gesture_requested": "fist", "ground_truth": true, "camera_blocking": true, "timestamp": ...}

Features:
    - Prediction smoothing (EMA + stability window) to prevent rapid gesture switching
    - Bidirectional communication for ground truth from Unity
    - Callbacks for prediction and game state events (used by GameRecorder)

Usage:
    From command line:
        python -m playagain_pipeline.prediction_server --model <model_name> [--host 127.0.0.1] [--port 5555]

    From GUI:
        The MainWindow can start/stop the server alongside prediction.
"""

import json
import queue
import socket
import threading
import time
import argparse
from pathlib import Path
from typing import Optional, Dict, List, Callable, Tuple

import numpy as np

from playagain_pipeline.config.config import get_default_config
from playagain_pipeline.devices.emg_device import DeviceManager, DeviceType, SyntheticEMGDevice
from playagain_pipeline.models.classifier import ModelManager, BaseClassifier


class PredictionSmoother:
    """
    Smooths gesture predictions using exponential moving average (EMA) and
    a time-based stability window.

    This prevents rapid switching between gesture classes caused by brief
    EMG signal similarities (e.g., tripod momentarily classified as fist).

    Algorithm:
        1. Apply EMA to the probability vector: ema = alpha * new + (1-alpha) * old
        2. Find the class with highest smoothed probability
        3. Only switch the confirmed prediction if the new class has been
           the EMA winner for at least `min_stable_ms` milliseconds

    Parameters:
        alpha (float): EMA weight for new observations. Range [0, 1].
            - Lower values = more smoothing / slower response
            - Higher values = less smoothing / faster response
            - Default 0.3 gives moderate smoothing
        min_stable_ms (float): Minimum time in milliseconds that a new
            prediction must be the EMA winner before it becomes the
            confirmed output. Default 150ms (barely perceptible lag).
    """

    def __init__(self, alpha: float = 0.3, min_stable_ms: float = 150.0):
        self._alpha = alpha
        self._min_stable_ms = min_stable_ms

        # EMA state
        self._ema_proba: Optional[np.ndarray] = None
        self._class_keys: Optional[List[str]] = None

        # Stability tracking
        self._last_ema_winner: Optional[str] = None
        self._winner_start_time: Optional[float] = None

        # Confirmed output (only changes after stability window passes)
        self._confirmed_gesture: str = "rest"
        self._confirmed_id: int = 0

    @property
    def alpha(self) -> float:
        return self._alpha

    @alpha.setter
    def alpha(self, value: float):
        self._alpha = max(0.01, min(1.0, value))

    @property
    def min_stable_ms(self) -> float:
        return self._min_stable_ms

    @min_stable_ms.setter
    def min_stable_ms(self, value: float):
        self._min_stable_ms = max(0.0, value)

    def smooth(self, gesture: str, gesture_id: int, confidence: float,
               probabilities: Dict[str, float]) -> Tuple[str, int, float, Dict[str, float]]:
        """
        Apply smoothing to a raw prediction.

        Args:
            gesture: Raw predicted gesture name.
            gesture_id: Raw predicted gesture class ID.
            confidence: Raw prediction confidence.
            probabilities: Raw per-class probability dict.

        Returns:
            Tuple of (smoothed_gesture, smoothed_id, smoothed_confidence, smoothed_probabilities).
        """
        keys = list(probabilities.keys())
        values = np.array([probabilities[k] for k in keys], dtype=np.float64)

        # Initialize or update EMA
        if self._ema_proba is None or len(self._ema_proba) != len(values):
            self._ema_proba = values.copy()
            self._class_keys = keys
        else:
            self._ema_proba = self._alpha * values + (1.0 - self._alpha) * self._ema_proba

        # Normalize EMA probabilities to sum to 1
        total = np.sum(self._ema_proba)
        if total > 0:
            self._ema_proba /= total

        # Find the EMA winner
        smoothed_idx = int(np.argmax(self._ema_proba))
        smoothed_gesture = keys[smoothed_idx]
        smoothed_confidence = float(self._ema_proba[smoothed_idx])

        # Stability tracking: has the winner changed?
        current_time_ms = time.time() * 1000.0

        if smoothed_gesture != self._last_ema_winner:
            # New winner — start stability timer
            self._last_ema_winner = smoothed_gesture
            self._winner_start_time = current_time_ms

        # Check if winner has been stable long enough
        if (self._winner_start_time is not None and
                current_time_ms - self._winner_start_time >= self._min_stable_ms):
            self._confirmed_gesture = smoothed_gesture
            self._confirmed_id = smoothed_idx

        # Build smoothed probability dict
        smoothed_proba = dict(zip(keys, [round(float(p), 4) for p in self._ema_proba]))

        return self._confirmed_gesture, self._confirmed_id, smoothed_confidence, smoothed_proba

    def reset(self):
        """Reset smoother state. Call when model changes."""
        self._ema_proba = None
        self._class_keys = None
        self._last_ema_winner = None
        self._winner_start_time = None
        self._confirmed_gesture = "rest"
        self._confirmed_id = 0


class PredictionServer:
    """
    TCP server that streams gesture predictions to connected clients.
    
    Integrates with the existing pipeline:
    - Loads a trained model
    - Connects to EMG device
    - Runs predictions in real-time
    - Broadcasts results to all connected TCP clients
    - Receives game state (ground truth) from Unity clients
    - Applies prediction smoothing to prevent rapid gesture switching
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 5555):
        self.host = host
        self.port = port
        
        self._server_socket: Optional[socket.socket] = None
        self._clients: List[socket.socket] = []
        self._clients_lock = threading.Lock()
        
        self._running = False
        self._accept_thread: Optional[threading.Thread] = None
        self._reader_threads: List[threading.Thread] = []
        
        # Pipeline components
        self._model: Optional[BaseClassifier] = None
        self._prediction_buffer: Optional[np.ndarray] = None
        self._prediction_buffer_lock = threading.Lock()
        self._prediction_window_ms = 200

        # Background prediction worker
        self._data_queue: queue.Queue = queue.Queue(maxsize=64)
        self._prediction_thread: Optional[threading.Thread] = None

        # Dedicated sender worker so network hiccups cannot block prediction
        self._send_queue: queue.Queue = queue.Queue(maxsize=8)
        self._sender_thread: Optional[threading.Thread] = None

        # Prediction smoothing
        self._smoother = PredictionSmoother(alpha=0.3, min_stable_ms=150.0)
        self._smoothing_enabled = True
        
        # Latest prediction state
        self._latest_gesture: str = "rest"
        self._latest_gesture_id: int = 0
        self._latest_confidence: float = 0.0
        self._latest_probabilities: Dict[str, float] = {}

        # Callbacks for external consumers (e.g., GameRecorder, GUI updates)
        self._prediction_callbacks: List[Callable] = []
        self._game_state_callbacks: List[Callable] = []

        # Pause flag: when True, EMG data is accepted but predictions are not run
        self._paused = False

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def is_paused(self) -> bool:
        return self._paused

    def pause(self):
        """Pause predictions without tearing down the TCP server."""
        self._paused = True
        print("[PredictionServer] Predictions paused")

    def resume(self):
        """Resume predictions after a pause."""
        self._paused = False
        print("[PredictionServer] Predictions resumed")

    @property
    def client_count(self) -> int:
        with self._clients_lock:
            return len(self._clients)

    def set_model(self, model: BaseClassifier):
        """Set the model to use for predictions."""
        self._model = model
        
        # Initialize prediction buffer with a reasonable default.
        # The actual channel count is determined when EMG data arrives via
        # on_emg_data, which will resize the buffer if needed.
        if model and model.metadata:
            window_samples = int(
                self._prediction_window_ms * model.metadata.sampling_rate / 1000
            )
            # Use num_channels from metadata but guard against corrupted values
            # (e.g. feature dimension stored instead of raw channels for
            # pre-extracted feature datasets). Cap at a reasonable max.
            num_ch = model.metadata.num_channels
            if num_ch > 256 or num_ch <= 0:
                num_ch = 32  # safe default
            self._prediction_buffer = np.zeros(
                (window_samples, num_ch)
            )
            print(f"[PredictionServer] Model set: {model.name}")
            print(f"[PredictionServer] Classes: {model.metadata.class_names}")
            print(f"[PredictionServer] Buffer: {window_samples} samples x {num_ch} channels")

        # Reset smoother when model changes
        self._smoother.reset()

    # ─── Smoothing Configuration ──────────────────────────────────────────

    def set_smoothing_enabled(self, enabled: bool):
        """Enable or disable prediction smoothing."""
        self._smoothing_enabled = enabled
        if not enabled:
            self._smoother.reset()
        print(f"[PredictionServer] Smoothing {'enabled' if enabled else 'disabled'}")

    def set_smoothing_params(self, alpha: float = None, min_stable_ms: float = None):
        """
        Update smoothing parameters.

        Args:
            alpha: EMA weight for new observations (0.01-1.0). Lower = smoother.
            min_stable_ms: Minimum stability window in ms before switching gesture.
        """
        if alpha is not None:
            self._smoother.alpha = alpha
        if min_stable_ms is not None:
            self._smoother.min_stable_ms = min_stable_ms
        print(f"[PredictionServer] Smoothing params: alpha={self._smoother.alpha:.2f}, "
              f"min_stable_ms={self._smoother.min_stable_ms:.0f}")

    @property
    def smoothing_enabled(self) -> bool:
        return self._smoothing_enabled

    @property
    def smoother(self) -> PredictionSmoother:
        return self._smoother

    # ─── Callback Registration ────────────────────────────────────────────

    def add_prediction_callback(self, callback: Callable):
        """
        Register a callback for prediction events.

        Callback signature: callback(gesture: str, gesture_id: int,
                                      confidence: float, probabilities: dict)

        Called after each prediction (smoothed if smoothing is enabled).
        Used by GameRecorder to track predictions for CSV recording.
        """
        self._prediction_callbacks.append(callback)

    def remove_prediction_callback(self, callback: Callable):
        """Remove a previously registered prediction callback."""
        try:
            self._prediction_callbacks.remove(callback)
        except ValueError:
            pass

    def add_game_state_callback(self, callback: Callable):
        """
        Register a callback for game state events from Unity.

        Callback signature: callback(ground_truth_active: bool,
                                      requested_gesture: str,
                                      camera_blocking: bool)

        Called when Unity sends a game_state message (ground truth update).
        Used by GameRecorder to track ground truth for CSV recording.
        """
        self._game_state_callbacks.append(callback)

    def remove_game_state_callback(self, callback: Callable):
        """Remove a previously registered game state callback."""
        try:
            self._game_state_callbacks.remove(callback)
        except ValueError:
            pass

    def start(self):
        """Start the TCP server and begin accepting connections."""
        if self._running:
            print("[PredictionServer] Already running")
            return
        
        try:
            self._server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._server_socket.bind((self.host, self.port))
            self._server_socket.listen(5)
            self._server_socket.settimeout(1.0)  # Allow periodic check for shutdown
            
            self._running = True

            # Drain any stale data in the queue
            while not self._data_queue.empty():
                try:
                    self._data_queue.get_nowait()
                except queue.Empty:
                    break

            # Start background prediction thread
            self._prediction_thread = threading.Thread(
                target=self._prediction_loop, daemon=True, name="PredictionServer-Predict"
            )
            self._prediction_thread.start()

            # Start sender thread
            self._sender_thread = threading.Thread(
                target=self._sender_loop, daemon=True, name="PredictionServer-Send"
            )
            self._sender_thread.start()

            # Start accept thread
            self._accept_thread = threading.Thread(
                target=self._accept_loop, daemon=True, name="PredictionServer-Accept"
            )
            self._accept_thread.start()
            
            print(f"[PredictionServer] Listening on {self.host}:{self.port}")
        except OSError as e:
            print(f"[PredictionServer] Failed to start: {e}")
            self._running = False

    def stop(self):
        """Stop the server and disconnect all clients."""
        if not self._running:
            return
        
        print("[PredictionServer] Stopping...")
        self._running = False
        
        # Close all client connections
        with self._clients_lock:
            for client in self._clients:
                try:
                    client.close()
                except OSError:
                    pass
            self._clients.clear()
        
        # Close server socket
        if self._server_socket:
            try:
                self._server_socket.close()
            except OSError:
                pass
            self._server_socket = None
        
        # Wake sender thread and wait for it
        try:
            self._send_queue.put_nowait(None)
        except queue.Full:
            pass

        if self._sender_thread:
            self._sender_thread.join(timeout=2.0)
            self._sender_thread = None

        # Wait for prediction thread
        if self._prediction_thread:
            self._prediction_thread.join(timeout=2.0)
            self._prediction_thread = None

        # Wait for accept thread
        if self._accept_thread:
            self._accept_thread.join(timeout=2.0)
            self._accept_thread = None

        # Wait for reader threads
        for thread in self._reader_threads:
            thread.join(timeout=1.0)
        self._reader_threads.clear()
        
        print("[PredictionServer] Stopped")

    def on_emg_data(self, data: np.ndarray):
        """
        Called when new EMG data arrives from the device.
        Enqueues data for the background prediction thread.
        This method is safe to call from the Qt main thread — it returns
        immediately without blocking.

        Args:
            data: EMG data array (samples, channels)
        """
        if not self._running or self._paused or self._model is None or self._prediction_buffer is None:
            return

        # Enqueue data for background processing (drop oldest if queue full)
        try:
            self._data_queue.put_nowait(data)
        except queue.Full:
            # Drop oldest frame to prevent unbounded memory growth
            try:
                self._data_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._data_queue.put_nowait(data)
            except queue.Full:
                pass

    def _prediction_loop(self):
        """
        Background thread: continuously processes queued EMG data,
        runs prediction, applies smoothing, broadcasts to Unity,
        and notifies callbacks.
        """
        while self._running:
            try:
                data = self._data_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if self._model is None or self._prediction_buffer is None:
                continue

            # Update rolling buffer (thread-safe)
            with self._prediction_buffer_lock:
                # Re-create buffer if channel count changed
                if data.shape[1] != self._prediction_buffer.shape[1]:
                    window_samples = self._prediction_buffer.shape[0]
                    self._prediction_buffer = np.zeros((window_samples, data.shape[1]))

                n_samples = min(data.shape[0], len(self._prediction_buffer))
                if n_samples <= 0:
                    continue
                if n_samples < len(self._prediction_buffer):
                    # In-place shift avoids np.roll allocation each frame.
                    self._prediction_buffer[:-n_samples] = self._prediction_buffer[n_samples:]
                self._prediction_buffer[-n_samples:] = data[:n_samples]
                buffer_snapshot = self._prediction_buffer.copy()

            # Run prediction
            try:
                X = buffer_snapshot[np.newaxis, :, :]
                # Hot path: infer both label and confidence from one probability pass.
                proba = self._model.predict_proba(X)[0]
                pred = int(np.argmax(proba))

                self._process_prediction(pred, proba)
            except Exception as e:
                if not hasattr(self, '_last_pred_error') or str(e) != self._last_pred_error:
                    self._last_pred_error = str(e)
                    print(f"[PredictionServer] Prediction error: {e}")

    def _process_prediction(self, pred, proba):
        """
        Process a raw prediction: resolve class name, apply smoothing,
        update state, notify callbacks, and broadcast to Unity.
        """
        try:
            # Get class name
            class_names = self._model.metadata.class_names
            gesture_name = class_names.get(int(pred), class_names.get(str(pred), f"class_{pred}"))
            
            # Get confidence
            try:
                confidence = float(proba[int(pred)])
            except (IndexError, KeyError):
                confidence = float(np.max(proba))
            
            # Build probabilities dict
            probabilities = {}
            for idx, p in enumerate(proba):
                name = class_names.get(idx, class_names.get(str(idx), f"class_{idx}"))
                probabilities[name] = round(float(p), 4)
            
            # Apply smoothing if enabled
            if self._smoothing_enabled:
                gesture_name, pred_id, confidence, probabilities = self._smoother.smooth(
                    gesture_name, int(pred), confidence, probabilities
                )
            else:
                pred_id = int(pred)
            
            # Update state
            self._latest_gesture = gesture_name
            self._latest_gesture_id = pred_id
            self._latest_confidence = confidence
            self._latest_probabilities = probabilities
            
            # Notify prediction callbacks (e.g., GameRecorder)
            for cb in self._prediction_callbacks:
                try:
                    cb(gesture_name, pred_id, confidence, probabilities)
                except Exception as e:
                    print(f"[PredictionServer] Prediction callback error: {e}")
            
            # Broadcast to connected clients (Unity)
            self._broadcast_prediction(
                gesture=gesture_name,
                gesture_id=pred_id,
                confidence=confidence,
                probabilities=probabilities
            )
        except Exception as e:
            # Don't crash on prediction errors but log them
            if not hasattr(self, '_last_proc_error') or str(e) != self._last_proc_error:
                self._last_proc_error = str(e)
                print(f"[PredictionServer] Processing error: {e}")

    def _broadcast_prediction(self, gesture: str, gesture_id: int,
                               confidence: float, probabilities: Dict[str, float]):
        """Queue prediction payload for asynchronous sending to connected clients."""
        message = {
            "gesture": gesture,
            "gesture_id": gesture_id,
            "confidence": round(confidence, 4),
            "probabilities": probabilities,
            "timestamp": time.time()
        }

        data = (json.dumps(message, separators=(",", ":")) + "\n").encode("utf-8")

        try:
            self._send_queue.put_nowait(data)
        except queue.Full:
            # Keep the newest payload; stale predictions are less useful.
            try:
                self._send_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._send_queue.put_nowait(data)
            except queue.Full:
                pass

    def _sender_loop(self):
        """Background thread: sends queued prediction payloads to all clients."""
        while self._running:
            try:
                payload = self._send_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if payload is None:
                break

            disconnected = []
            with self._clients_lock:
                for client in self._clients:
                    try:
                        client.sendall(payload)
                    except (OSError, BrokenPipeError, socket.timeout):
                        disconnected.append(client)

                for client in disconnected:
                    self._clients.remove(client)
                    try:
                        client.close()
                    except OSError:
                        pass
                    print(f"[PredictionServer] Client disconnected (send failed)")

    def _accept_loop(self):
        """Background thread: accept incoming client connections."""
        while self._running:
            try:
                client, addr = self._server_socket.accept()
                client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                # Configure once per client so sender loop stays lightweight.
                client.settimeout(0.05)
                
                with self._clients_lock:
                    self._clients.append(client)
                
                print(f"[PredictionServer] Client connected from {addr}")
                
                # Send a handshake message with model info
                if self._model and self._model.metadata:
                    handshake = {
                        "type": "handshake",
                        "model_name": self._model.name,
                        "model_type": self._model.metadata.model_type,
                        "class_names": self._model.metadata.class_names,
                        "num_classes": self._model.metadata.num_classes,
                        "smoothing_enabled": self._smoothing_enabled,
                        "timestamp": time.time()
                    }
                    line = json.dumps(handshake) + "\n"
                    try:
                        client.sendall(line.encode("utf-8"))
                    except OSError:
                        pass

                # Start a reader thread for this client to receive game state messages
                reader_thread = threading.Thread(
                    target=self._client_reader_loop,
                    args=(client, addr),
                    daemon=True,
                    name=f"PredictionServer-Reader-{addr[1]}"
                )
                reader_thread.start()
                self._reader_threads.append(reader_thread)
                
            except socket.timeout:
                continue
            except OSError:
                if self._running:
                    break

    def _client_reader_loop(self, client: socket.socket, addr):
        """
        Background thread: read incoming messages from a connected client.

        Handles game_state messages from Unity containing ground truth data.
        The TCP connection is full-duplex, so this reads from the same socket
        that _broadcast_prediction writes to.

        Message format (Unity → Python):
            {"type": "game_state", "gesture_requested": "fist",
             "ground_truth": true, "camera_blocking": true, "timestamp": 123.456}
        """
        reader = None
        try:
            # Create a file-like reader for line-delimited JSON
            reader = client.makefile("r", encoding="utf-8")

            while self._running:
                try:
                    line = reader.readline()
                    if not line:
                        # Client disconnected
                        break
                    line = line.strip()
                    if not line:
                        continue

                    # Parse incoming JSON message
                    try:
                        msg = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    msg_type = msg.get("type", "")

                    if msg_type == "game_state":
                        ground_truth = msg.get("ground_truth", False)
                        gesture_requested = msg.get("gesture_requested", "none")
                        camera_blocking = msg.get("camera_blocking", False)

                        # Notify game state callbacks (e.g., GameRecorder)
                        for cb in self._game_state_callbacks:
                            try:
                                cb(ground_truth, gesture_requested, camera_blocking)
                            except Exception as e:
                                print(f"[PredictionServer] Game state callback error: {e}")

                except (OSError, IOError):
                    if self._running:
                        break

        except Exception as e:
            if self._running:
                print(f"[PredictionServer] Reader error for {addr}: {e}")
        finally:
            # Clean up
            try:
                if reader is not None:
                    reader.close()
            except Exception:
                pass

            with self._clients_lock:
                if client in self._clients:
                    self._clients.remove(client)

            try:
                client.close()
            except OSError:
                pass

            print(f"[PredictionServer] Client {addr} reader stopped")


def run_standalone(model_name: str, host: str = "127.0.0.1", port: int = 5555,
                   device_type: str = "muovi"):
    """
    Run the prediction server as a standalone process.
    
    This connects to the EMG device, loads the model, and starts
    the TCP server - all without the GUI.
    
    Args:
        model_name: Name of the trained model to load
        host: TCP host to bind to
        port: TCP port to bind to
        device_type: EMG device type ("muovi" or "synthetic")
    """
    # Setup paths
    pipeline_dir = Path(__file__).parent
    data_dir = pipeline_dir / "data"
    
    # Load config
    config = get_default_config()
    
    # Initialize model manager and load model
    model_manager = ModelManager(data_dir / "models")
    
    print(f"[Standalone] Loading model: {model_name}")
    model = model_manager.load_model(model_name)
    print(f"[Standalone] Model loaded: {model.name} ({model.metadata.model_type})")
    print(f"[Standalone] Classes: {model.metadata.class_names}")
    
    # Initialize device
    device_mgr = DeviceManager()
    
    if device_type == "synthetic":
        device = SyntheticEMGDevice(
            num_channels=model.metadata.num_channels,
            sampling_rate=model.metadata.sampling_rate
        )
        device_mgr.set_device(device)
    else:
        dt = DeviceType.MUOVI_PLUS if device_type == "muovi_plus" else DeviceType.MUOVI
        device_mgr.create_device(dt)
    
    # Start prediction server
    server = PredictionServer(host=host, port=port)
    server.set_model(model)
    server.start()
    
    # Connect device data to server
    def on_data(data: np.ndarray):
        server.on_emg_data(data)
    
    device = device_mgr.device
    if device is None:
        print("[Standalone] ERROR: No device available")
        server.stop()
        return
    
    device.data_callback = on_data
    
    # Connect and start streaming
    print(f"[Standalone] Connecting to {device_type} device...")
    device.connect()
    device.start_streaming()
    print(f"[Standalone] Device streaming. Server ready on {host}:{port}")
    print(f"[Standalone] Press Ctrl+C to stop")
    
    # Wait for interrupt
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n[Standalone] Shutting down...")
    finally:
        device.stop_streaming()
        device.disconnect()
        server.stop()


def main():
    parser = argparse.ArgumentParser(
        description="Gesture Prediction TCP Server for Unity Integration"
    )
    parser.add_argument(
        "--model", required=True,
        help="Name of the trained model to load (e.g., 'random_forest_2026-02-12_09-59-54_3rep')"
    )
    parser.add_argument(
        "--host", default="127.0.0.1",
        help="TCP host to bind to (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port", type=int, default=5555,
        help="TCP port to bind to (default: 5555)"
    )
    parser.add_argument(
        "--device", default="muovi", choices=["muovi", "muovi_plus", "synthetic"],
        help="EMG device type (default: muovi)"
    )
    
    args = parser.parse_args()
    run_standalone(args.model, args.host, args.port, args.device)


if __name__ == "__main__":
    main()
