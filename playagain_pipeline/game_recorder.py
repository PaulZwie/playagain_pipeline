"""
Game Recording Module for PlayAgain Pipeline.

Records gameplay data including EMG signals, model predictions, and ground truth
from the Unity game into CSV files for later analysis.

The recorder captures three synchronized data streams:
1. Raw EMG data from the device (at device sample rate)
2. Model predictions (gesture, confidence, probabilities)
3. Ground truth from the Unity game (what gesture is requested, camera state)

Ground truth is sent from Unity back to Python through the bidirectional TCP
connection managed by PredictionServer.

Usage:
    recorder = GameRecorder(output_dir=Path("data"))
    recorder.set_class_names({0: "rest", 1: "fist", 2: "pinch", 3: "tripod"})
    recorder.start_recording(num_channels=32, subject_id="VP_01")
    # ... data flows through on_emg_data, on_prediction, on_game_state ...
    recorder.stop_recording()
"""

import csv
import json
import queue
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any

import numpy as np

from playagain_pipeline.utils.platform_utils import IS_WINDOWS


class GameRecorder:
    """
    Records synchronized EMG data, model predictions, and game ground truth to CSV.

    The recorder maintains the current prediction and ground truth state, which are
    updated asynchronously by callbacks. When EMG data arrives, each sample is recorded
    with the current prediction and ground truth state.

    CSV Format:
        Timestamp, PredictedGesture, PredictedGestureId, Confidence,
        Prob_<class1>, ..., Prob_<classN>,
        GroundTruthActive, RawGroundTruth, RequestedGesture, CameraBlocking,
        EMG_Ch0, ..., EMG_ChN

        Ground Truth Logic:
                GroundTruthActive is derived from Unity game-state signals using a
                small deterministic state machine.

                It is True only while a gesture is requested, and one of these is true:
                    - Unity raw ground_truth flag is true, or
                    - camera_blocking is true, or
                    - the request just started and we are inside a short pre-block grace window.

                The grace window makes labels robust when request and camera updates
                arrive slightly out of sync (e.g., user starts gesture just before block).
                RawGroundTruth stores the original Unity flag for reference.

    Thread Safety:
        - on_emg_data() is called from the device data thread
        - on_prediction() is called from the prediction pipeline
        - on_game_state() is called from the TCP reader thread
        All state updates use atomic assignments (Python GIL protected).
        Buffer flushing uses a lock to prevent concurrent writes.
    """

    def __init__(self, output_dir: Path):
        """
        Initialize the game recorder.

        Args:
            output_dir: Base directory for saving recordings.
                        Files are saved to output_dir/game_recordings/[subject_id]/
        """
        self._output_dir = Path(output_dir)
        self._num_channels = 32  # Set on start_recording
        self._sampling_rate = 2000  # Updated on start_recording

        # Recording state
        self._is_recording = False
        self._file = None
        self._writer = None
        self._start_time: Optional[float] = None
        self._sample_count = 0

        # Background writer thread and queue
        # Bounded queue prevents runaway RAM growth when disk stalls.
        self._write_queue: queue.Queue = queue.Queue(maxsize=256)
        self._writer_thread: Optional[threading.Thread] = None

        # Writer tuning (slower flush cadence helps Windows file I/O significantly)
        self._flush_interval_s = 0.35 if IS_WINDOWS else 0.2
        self._rows_per_flush = 2500 if IS_WINDOWS else 1500

        # Optional verbose ground-truth logging (disabled in hot path by default)
        self._debug_ground_truth = False
        self._last_gt_log_time = 0.0

        # Current prediction state (updated atomically via on_prediction)
        self._current_prediction = "rest"
        self._current_gesture_id = 0
        self._current_confidence = 0.0
        self._current_probabilities: Dict[str, float] = {}

        # Current ground truth state (updated atomically via on_game_state)
        self._ground_truth_active = False
        self._raw_ground_truth = False
        self._requested_gesture = "none"
        self._camera_blocking = False

        # GT timing state for robust camera-transition handling
        self._gt_pre_block_grace_s = 0.35
        self._gt_request_started_at_monotonic: Optional[float] = None

        # Class names for probability columns in CSV header
        self._class_names: List[str] = []

        # Model metadata for config.json
        self._model_metadata: Optional[Dict[str, Any]] = None

        # Current file path
        self._current_file_path: Optional[Path] = None

        # Current session directory (contains CSV + config.json)
        self._current_session_dir: Optional[Path] = None

        # Calibration/rotation metadata (set externally before start_recording)
        self._calibration_info: Optional[Dict[str, Any]] = None

        # Optional participant metadata (set externally before start_recording)
        self._participant_info: Optional[Dict[str, Any]] = None

    # ─── Properties ───────────────────────────────────────────────────────

    @property
    def is_recording(self) -> bool:
        """Whether the recorder is currently active."""
        return self._is_recording

    @property
    def sample_count(self) -> int:
        """Total number of samples recorded in current session."""
        return self._sample_count

    @property
    def current_file_path(self) -> Optional[Path]:
        """Path of the current recording file."""
        return self._current_file_path

    @property
    def duration(self) -> float:
        """Duration of current recording in seconds."""
        if self._start_time is None:
            return 0.0
        return time.time() - self._start_time

    # ─── Configuration ────────────────────────────────────────────────────

    def set_class_names(self, class_names):
        """
        Set class names for probability column headers.

        Args:
            class_names: Either a dict {label_id: name} or a list of names.
                         Example: {0: "rest", 1: "fist", 2: "pinch", 3: "tripod"}
        """
        if isinstance(class_names, dict):
            # Convert {0: "rest", 1: "fist", ...} to ["rest", "fist", ...]
            max_id = max(int(k) for k in class_names.keys()) if class_names else -1
            self._class_names = [
                class_names.get(i, class_names.get(str(i), f"class_{i}"))
                for i in range(max_id + 1)
            ]
        elif isinstance(class_names, list):
            self._class_names = list(class_names)

    def set_model_metadata(self, metadata):
        """
        Set model metadata to be saved alongside the recording.

        Args:
            metadata: A ModelMetadata object (with a .to_dict() method)
                      or a plain dict with model information.
        """
        if hasattr(metadata, 'to_dict'):
            self._model_metadata = metadata.to_dict()
        elif isinstance(metadata, dict):
            self._model_metadata = metadata
        else:
            self._model_metadata = None

    def set_calibration_info(self, rotation_offset: int, confidence: float,
                             channel_mapping: Optional[List[int]] = None):
        """
        Set calibration/rotation info to be stored with the game recording.

        This allows the game recording to carry the bracelet rotation offset
        that was active during the recording, so downstream analysis and
        training pipelines can apply proper channel alignment.

        Args:
            rotation_offset: Number of channels the bracelet is rotated from reference.
            confidence: Confidence of the rotation detection [0, 1].
            channel_mapping: Full channel mapping list (optional).
        """
        self._calibration_info = {
            "rotation_offset": rotation_offset,
            "rotation_confidence": round(confidence, 4),
            "channel_mapping": channel_mapping,
        }

    def set_participant_info(self, participant_info: Optional[Dict[str, Any]]):
        """Attach participant information to the recording config."""
        self._participant_info = dict(participant_info) if participant_info else None

    # ─── Recording Control ────────────────────────────────────────────────

    def start_recording(self, num_channels: int, session_name: Optional[str] = None,
                        subject_id: Optional[str] = None,
                        sampling_rate: int = 2000) -> Path:
        """
        Start a new game recording session.

        Args:
            num_channels: Number of EMG channels being recorded.
            session_name: Optional custom name for the recording file.
            subject_id: Optional subject identifier (creates subfolder).
            sampling_rate: Device sampling rate in Hz.

        Returns:
            Path to the created CSV file.
        """
        if self._is_recording:
            print("[GameRecorder] Already recording")
            return self._current_file_path

        self._num_channels = num_channels
        self._sampling_rate = sampling_rate

        # Create output directory — each recording gets its own folder
        game_data_dir = self._output_dir / "game_recordings"
        if subject_id:
            game_data_dir = game_data_dir / subject_id

        # Generate session folder name
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if session_name:
            folder_name = f"game_{session_name}_{timestamp}"
        else:
            folder_name = f"game_{timestamp}"

        self._current_session_dir = game_data_dir / folder_name
        self._current_session_dir.mkdir(parents=True, exist_ok=True)

        # CSV file inside session folder
        self._current_file_path = self._current_session_dir / "recording.csv"

        # Open file and write header
        # Large text buffering reduces frequent kernel writes during gameplay.
        self._file = open(self._current_file_path, "w", newline="", buffering=1024 * 1024)
        self._writer = csv.writer(self._file)

        # Build CSV header
        header = [
            "Timestamp",
            "PredictedGesture",
            "PredictedGestureId",
            "Confidence",
        ]

        # Probability columns (one per class)
        for name in self._class_names:
            header.append(f"Prob_{name}")

        header.extend([
            "GroundTruthActive",
            "RawGroundTruth",
            "RequestedGesture",
            "CameraBlocking",
        ])

        # EMG channel columns
        for i in range(self._num_channels):
            header.append(f"EMG_Ch{i}")

        self._writer.writerow(header)
        self._file.flush()

        # Reset state
        self._start_time = time.time()
        self._sample_count = 0

        # Reset ground truth (in case leftover from previous session)
        self._ground_truth_active = False
        self._raw_ground_truth = False
        self._requested_gesture = "none"
        self._camera_blocking = False
        self._gt_request_started_at_monotonic = None

        # Start background writer thread
        self._writer_thread = threading.Thread(
            target=self._writer_loop, daemon=True, name="GameRecorder-Writer"
        )

        self._is_recording = True
        self._writer_thread.start()

        # Save initial config.json with model + recording metadata
        self._save_config(subject_id=subject_id, session_name=session_name)

        print(f"[GameRecorder] Recording started: {self._current_session_dir}")
        print(f"[GameRecorder]   Channels: {num_channels}, Classes: {self._class_names}")
        return self._current_file_path

    def stop_recording(self) -> Optional[Path]:
        """
        Stop the current recording session.

        Returns:
            Path to the saved CSV file, or None if not recording.
        """
        if not self._is_recording:
            return None

        self._is_recording = False

        # Signal writer thread to stop and wait for it to drain
        if self._writer_thread is not None:
            # Ensure sentinel insertion even if queue is full.
            while True:
                try:
                    self._write_queue.put_nowait(None)  # sentinel
                    break
                except queue.Full:
                    try:
                        self._write_queue.get_nowait()
                    except queue.Empty:
                        break
            self._writer_thread.join(timeout=5.0)
            self._writer_thread = None

        # Close file
        if self._file:
            self._file.close()
            self._file = None
            self._writer = None

        duration = self.duration
        file_path = self._current_file_path

        # Update config.json with final recording stats
        self._finalize_config()

        print(f"[GameRecorder] Recording stopped: {self._sample_count} samples, "
              f"{duration:.1f}s, saved to {self._current_session_dir}")

        return file_path

    # ─── Data Callbacks ───────────────────────────────────────────────────

    def on_emg_data(self, data: np.ndarray):
        """
        Record EMG data with current prediction and ground truth state.

        Called every time new EMG data arrives from the device.
        Each sample in the data chunk gets a row in the CSV with the
        current prediction and ground truth state attached.

        Args:
            data: EMG data array with shape (samples, channels).
        """
        if not self._is_recording or self._writer is None:
            return

        n_samples = data.shape[0]
        if n_samples <= 0:
            return

        n_channels = min(data.shape[1], self._num_channels)
        current_time = time.time()
        base_timestamp = current_time - self._start_time

        # Snapshot current state (atomic reads under GIL)
        prediction = self._current_prediction
        gesture_id = self._current_gesture_id
        confidence = float(self._current_confidence)
        probabilities = dict(self._current_probabilities)
        gt_active = self._ground_truth_active
        raw_gt = getattr(self, '_raw_ground_truth', gt_active)
        req_gesture = self._requested_gesture
        cam_blocking = self._camera_blocking

        # Ground truth is derived only in on_game_state.
        # Keep on_emg_data a pure snapshot/write path for deterministic recording.

        # Build all rows in bulk, then enqueue as a single batch
        rows = []
        sr = float(self._sampling_rate)
        timestamps = base_timestamp + (np.arange(n_samples, dtype=np.float64) / sr)
        gt_flag = 1 if gt_active else 0
        raw_gt_flag = 1 if raw_gt else 0
        cam_flag = 1 if cam_blocking else 0
        prob_values = [float(probabilities.get(name, 0.0)) for name in self._class_names]
        emg_rows = data[:n_samples, :n_channels].tolist()
        pad = [0.0] * (self._num_channels - n_channels)

        for i, ts in enumerate(timestamps):
            row = [
                float(ts),
                prediction,
                gesture_id,
                confidence,
            ]
            row.extend(prob_values)
            row.extend([
                gt_flag,
                raw_gt_flag,
                req_gesture,
                cam_flag,
            ])
            row.extend(emg_rows[i])
            if pad:
                row.extend(pad)
            rows.append(row)

        self._sample_count += n_samples

        # Enqueue batch for background writer (non-blocking)
        try:
            self._write_queue.put_nowait(rows)
        except queue.Full:
            # Under extreme load, drop oldest batch to protect GUI
            try:
                self._write_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._write_queue.put_nowait(rows)
            except queue.Full:
                pass

    def on_prediction(self, gesture: str, gesture_id: int, confidence: float,
                      probabilities: Dict[str, float]):
        """
        Update current prediction state.

        Called by PredictionServer after each prediction (potentially smoothed).
        These values are attached to subsequent EMG samples in the CSV.

        Args:
            gesture: Predicted gesture name (e.g., "fist").
            gesture_id: Predicted gesture class ID.
            confidence: Prediction confidence [0, 1].
            probabilities: Per-class probability dict.
        """
        self._current_prediction = gesture
        self._current_gesture_id = gesture_id
        self._current_confidence = confidence
        self._current_probabilities = probabilities

    def on_game_state(self, ground_truth_active: bool, requested_gesture: str,
                      camera_blocking: bool):
        """
        Update current ground truth state from Unity game.

        Called when the PredictionServer receives a game_state message from Unity.

        The recorder computes GroundTruthActive from request-window semantics:
            - no requested gesture -> always False
            - requested gesture present -> True if raw_gt or camera_blocking
            - if request just started, also allow a short pre-block grace window

        This handles out-of-sync request/camera updates without coupling to model
        predictions and avoids false positives outside request windows.

        Args:
            ground_truth_active: Raw Unity flag (request active on Unity side).
            requested_gesture: Name of requested gesture (e.g., "fist" or "none").
            camera_blocking: True if camera is in interaction/feeding view.
        """
        self._raw_ground_truth = bool(ground_truth_active)
        self._camera_blocking = bool(camera_blocking)

        requested_norm = self._normalize_label(requested_gesture)
        has_requested_gesture = requested_norm not in ("", "none", "null")

        now = time.monotonic()
        previous_requested = self._normalize_label(self._requested_gesture)
        if has_requested_gesture and previous_requested != requested_norm:
            self._gt_request_started_at_monotonic = now
        elif not has_requested_gesture:
            self._gt_request_started_at_monotonic = None

        in_pre_block_grace = False
        if has_requested_gesture and self._gt_request_started_at_monotonic is not None:
            in_pre_block_grace = (now - self._gt_request_started_at_monotonic) <= self._gt_pre_block_grace_s

        self._ground_truth_active = bool(
            has_requested_gesture and (
                self._raw_ground_truth
                or self._camera_blocking
                or in_pre_block_grace
            )
        )
        self._requested_gesture = requested_norm if has_requested_gesture else "none"

        if self._is_recording and self._debug_ground_truth:
            now_log = time.monotonic()
            if now_log - self._last_gt_log_time >= 1.0:
                self._last_gt_log_time = now_log
                print(f"[GameRecorder] Ground truth update: "
                      f"reliable_active={self._ground_truth_active}, "
                      f"raw_gt={self._raw_ground_truth}, "
                      f"gesture={self._requested_gesture}, "
                      f"camera_blocking={self._camera_blocking}, "
                      f"pre_block_grace={in_pre_block_grace}")

    @staticmethod
    def _normalize_label(label: Any) -> str:
        """Normalize gesture labels to a comparable lowercase form."""
        text = str(label or "").strip().lower().replace("_", " ").replace("-", " ")
        return " ".join(text.split())

    # ─── Internal ─────────────────────────────────────────────────────────

    def _writer_loop(self):
        """
        Background thread: drains the write queue and flushes rows to CSV.
        Stops when it receives a None sentinel on the queue.
        """
        pending_rows = []
        last_flush = time.monotonic()

        while True:
            try:
                item = self._write_queue.get(timeout=0.1)
            except queue.Empty:
                item = None

            if item is None:
                # None can be either timeout or stop sentinel; only stop when recording ended.
                if not self._is_recording:
                    # Drain remaining queued batches before exit.
                    while True:
                        try:
                            remaining = self._write_queue.get_nowait()
                        except queue.Empty:
                            break
                        if remaining is not None:
                            pending_rows.extend(remaining)

                    if pending_rows and self._writer:
                        try:
                            self._writer.writerows(pending_rows)
                        except Exception as e:
                            print(f"[GameRecorder] Write error: {e}")

                    if self._file:
                        try:
                            self._file.flush()
                        except Exception:
                            pass
                    return
            else:
                pending_rows.extend(item)

            now = time.monotonic()
            should_flush = (
                pending_rows and (
                    len(pending_rows) >= self._rows_per_flush
                    or (now - last_flush) >= self._flush_interval_s
                )
            )

            if should_flush and self._writer:
                try:
                    self._writer.writerows(pending_rows)
                    self._file.flush()
                    pending_rows.clear()
                    last_flush = now
                except Exception as e:
                    print(f"[GameRecorder] Write error: {e}")

    def _save_config(self, subject_id: Optional[str] = None,
                     session_name: Optional[str] = None):
        """Save recording configuration and model metadata to config.json."""
        if self._current_session_dir is None:
            return

        config = {
            "recording": {
                "started_at": datetime.now().isoformat(),
                "subject_id": subject_id,
                "session_name": session_name,
                "num_channels": self._num_channels,
                "sampling_rate": self._sampling_rate,
                "class_names": self._class_names,
                "csv_file": "recording.csv",
            },
        }

        if self._model_metadata:
            # Remove training_history to keep config.json compact
            model_info = dict(self._model_metadata)
            model_info.pop("training_history", None)
            config["model"] = model_info

        if self._participant_info:
            config["participant"] = self._participant_info

        if self._calibration_info:
            config["calibration"] = self._calibration_info

        config_path = self._current_session_dir / "config.json"
        try:
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2, default=str)
        except Exception as e:
            print(f"[GameRecorder] Failed to save config.json: {e}")

    def _finalize_config(self):
        """Update config.json with final recording statistics."""
        if self._current_session_dir is None:
            return

        config_path = self._current_session_dir / "config.json"
        try:
            if config_path.exists():
                with open(config_path, "r") as f:
                    config = json.load(f)
            else:
                config = {}

            config["recording"]["finished_at"] = datetime.now().isoformat()
            config["recording"]["total_samples"] = self._sample_count
            config["recording"]["duration_seconds"] = round(self.duration, 2)

            with open(config_path, "w") as f:
                json.dump(config, f, indent=2, default=str)
        except Exception as e:
            print(f"[GameRecorder] Failed to finalize config.json: {e}")
