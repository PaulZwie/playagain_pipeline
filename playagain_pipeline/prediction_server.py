"""
TCP Prediction Server for Unity Integration.

Runs the gesture prediction pipeline and sends predictions
to connected clients (e.g., Unity game) over TCP.

Protocol: Newline-delimited JSON messages.
Each line is a JSON object:
    {"gesture": "fist", "gesture_id": 3, "confidence": 0.95, "probabilities": {...}, "timestamp": 1234567890.123}

Usage:
    From command line:
        python -m playagain_pipeline.prediction_server --model <model_name> [--host 127.0.0.1] [--port 5555]

    From GUI:
        The MainWindow can start/stop the server alongside prediction.
"""

import json
import socket
import threading
import time
import argparse
import sys
import signal
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np

from playagain_pipeline.config.config import get_default_config
from playagain_pipeline.core.data_manager import DataManager
from playagain_pipeline.devices.emg_device import DeviceManager
from playagain_pipeline.models.classifier import ModelManager, BaseClassifier


class PredictionServer:
    """
    TCP server that streams gesture predictions to connected clients.
    
    Integrates with the existing pipeline:
    - Loads a trained model
    - Connects to EMG device
    - Runs predictions in real-time
    - Broadcasts results to all connected TCP clients
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 5555):
        self.host = host
        self.port = port
        
        self._server_socket: Optional[socket.socket] = None
        self._clients: List[socket.socket] = []
        self._clients_lock = threading.Lock()
        
        self._running = False
        self._accept_thread: Optional[threading.Thread] = None
        
        # Pipeline components
        self._model: Optional[BaseClassifier] = None
        self._prediction_buffer: Optional[np.ndarray] = None
        self._prediction_window_ms = 200
        
        # Latest prediction state
        self._latest_gesture: str = "rest"
        self._latest_gesture_id: int = 0
        self._latest_confidence: float = 0.0
        self._latest_probabilities: Dict[str, float] = {}

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def client_count(self) -> int:
        with self._clients_lock:
            return len(self._clients)

    def set_model(self, model: BaseClassifier):
        """Set the model to use for predictions."""
        self._model = model
        
        # Initialize prediction buffer
        if model and model.metadata:
            window_samples = int(
                self._prediction_window_ms * model.metadata.sampling_rate / 1000
            )
            self._prediction_buffer = np.zeros(
                (window_samples, model.metadata.num_channels)
            )
            print(f"[PredictionServer] Model set: {model.name}")
            print(f"[PredictionServer] Classes: {model.metadata.class_names}")
            print(f"[PredictionServer] Buffer: {window_samples} samples x {model.metadata.num_channels} channels")

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
        
        # Wait for accept thread
        if self._accept_thread:
            self._accept_thread.join(timeout=2.0)
            self._accept_thread = None
        
        print("[PredictionServer] Stopped")

    def on_emg_data(self, data: np.ndarray):
        """
        Called when new EMG data arrives from the device.
        Updates the prediction buffer and triggers prediction.
        
        Args:
            data: EMG data array (samples, channels)
        """
        if not self._running or self._model is None or self._prediction_buffer is None:
            return
        
        # Update rolling buffer
        n_samples = min(data.shape[0], len(self._prediction_buffer))
        self._prediction_buffer = np.roll(self._prediction_buffer, -n_samples, axis=0)
        self._prediction_buffer[-n_samples:] = data[:n_samples]
        
        # Run prediction
        try:
            X = self._prediction_buffer[np.newaxis, :, :]
            pred = self._model.predict(X)[0]
            proba = self._model.predict_proba(X)[0]
            
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
            
            # Update state
            self._latest_gesture = gesture_name
            self._latest_gesture_id = int(pred)
            self._latest_confidence = confidence
            self._latest_probabilities = probabilities
            
            # Broadcast to clients
            self._broadcast_prediction(
                gesture=gesture_name,
                gesture_id=int(pred),
                confidence=confidence,
                probabilities=probabilities
            )
        except Exception as e:
            # Don't crash on prediction errors
            pass

    def _broadcast_prediction(self, gesture: str, gesture_id: int,
                               confidence: float, probabilities: Dict[str, float]):
        """Send prediction to all connected clients."""
        message = {
            "gesture": gesture,
            "gesture_id": gesture_id,
            "confidence": round(confidence, 4),
            "probabilities": probabilities,
            "timestamp": time.time()
        }
        
        line = json.dumps(message) + "\n"
        data = line.encode("utf-8")
        
        disconnected = []
        
        with self._clients_lock:
            for client in self._clients:
                try:
                    client.sendall(data)
                except (OSError, BrokenPipeError):
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
                        "timestamp": time.time()
                    }
                    line = json.dumps(handshake) + "\n"
                    try:
                        client.sendall(line.encode("utf-8"))
                    except OSError:
                        pass
                
            except socket.timeout:
                continue
            except OSError:
                if self._running:
                    break


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
    from playagain_pipeline.devices.emg_device import DeviceType
    
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
        from playagain_pipeline.devices.emg_device import SyntheticEMGDevice
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
