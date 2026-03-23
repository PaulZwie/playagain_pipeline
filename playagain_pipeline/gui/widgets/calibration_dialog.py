"""
Calibration Dialog for EMG device calibration.

Guides user through calibration process to determine electrode orientation.
"""

from typing import Optional, Dict
import numpy as np
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QProgressBar, QGroupBox, QTextEdit, QMessageBox
)
from PySide6.QtCore import Qt, Signal, Slot, QTimer
from datetime import datetime


class CalibrationDialog(QDialog):
    """
    Dialog for performing EMG device calibration.

    Guides the user through recording calibration gestures and
    computes the electrode rotation offset.

    Uses single finger movements for more granular EMG response patterns,
    which provides better electrode position discrimination.
    """

    calibration_complete = Signal(object)  # Emits CalibrationResult

    # waveOut is listed first: Barona López et al. (Sensors 2020, Appendix A)
    # tested all five gestures as sync signals and found waveOut produces the
    # sharpest azimuthal energy peak across subjects, making it the most
    # reliable primary synchronization gesture for MEC-based offset detection.
    # The remaining single-finger movements add spatial diversity to the
    # combined energy profile used as a calibration fallback.
    CALIBRATION_GESTURES = [
        ("waveout",      "Extend your wrist outward (bend the back of your hand upward)"),
        ("rest",         "Keep your hand completely relaxed, palm up"),
        ("index_flex",   "Bend your INDEX finger down toward your palm"),
        ("middle_flex",  "Bend your MIDDLE finger down toward your palm"),
        ("ring_flex",    "Bend your RING finger down toward your palm"),
        ("pinky_flex",   "Bend your PINKY finger down toward your palm"),
        ("thumb_flex",   "Bend your THUMB inward toward your palm"),
        ("fist",         "Close all fingers into a firm FIST"),
    ]

    def __init__(self, calibrator, device, parent=None):
        super().__init__(parent)
        self.calibrator = calibrator
        self.device = device
        self._calibration_result = None

        # Calibration data storage
        self._gesture_data: Dict[str, np.ndarray] = {}
        self._current_gesture_idx = 0
        self._is_recording = False
        self._recording_buffer = []
        self._recording_duration_s = 3.0
        self._countdown_s = 3

        # Timer for recording
        self._recording_timer = QTimer(self)
        self._recording_timer.timeout.connect(self._on_recording_tick)
        self._recording_elapsed = 0

        self._setup_ui()
        self.setWindowTitle("EMG Calibration")
        self.setMinimumSize(500, 400)

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Instructions
        instructions_group = QGroupBox("Instructions")
        instructions_layout = QVBoxLayout(instructions_group)

        self.instructions_label = QLabel(
            "This calibration determines the orientation of the electrode bracelet.\n\n"
            "The first gesture — wrist extension (waveOut) — is the primary sync signal.\n"
            "Research (Barona López et al., 2020) shows it produces the sharpest per-electrode\n"
            "energy peak across subjects, making it the most reliable gesture for MEC detection.\n\n"
            "The remaining single-finger movements provide spatial diversity for robustness.\n"
            "Hold each gesture steadily for 3 seconds.\n\n"
            "Click 'Start Calibration' when ready."
        )
        self.instructions_label.setWordWrap(True)
        instructions_layout.addWidget(self.instructions_label)
        layout.addWidget(instructions_group)

        # Current gesture display
        gesture_group = QGroupBox("Current Gesture")
        gesture_layout = QVBoxLayout(gesture_group)

        self.gesture_name_label = QLabel("Ready")
        self.gesture_name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font = self.gesture_name_label.font()
        font.setPointSize(24)
        font.setBold(True)
        self.gesture_name_label.setFont(font)
        gesture_layout.addWidget(self.gesture_name_label)

        self.gesture_instruction_label = QLabel("")
        self.gesture_instruction_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        gesture_layout.addWidget(self.gesture_instruction_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximum(int(self._recording_duration_s * 10))
        self.progress_bar.setValue(0)
        gesture_layout.addWidget(self.progress_bar)

        self.status_label = QLabel("Status: Waiting to start")
        gesture_layout.addWidget(self.status_label)

        layout.addWidget(gesture_group)

        # Log area
        log_group = QGroupBox("Calibration Log")
        log_layout = QVBoxLayout(log_group)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(100)
        log_layout.addWidget(self.log_text)
        layout.addWidget(log_group)

        # Buttons
        button_layout = QHBoxLayout()

        self.start_btn = QPushButton("Start Calibration")
        self.start_btn.clicked.connect(self._on_start_calibration)
        button_layout.addWidget(self.start_btn)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_btn)

        self.finish_btn = QPushButton("Finish")
        self.finish_btn.setEnabled(False)
        self.finish_btn.clicked.connect(self.accept)
        button_layout.addWidget(self.finish_btn)

        layout.addLayout(button_layout)

    def _log(self, message: str):
        """Add message to log."""
        self.log_text.append(message)

    @Slot()
    def _on_start_calibration(self):
        """Start the calibration process."""
        self.start_btn.setEnabled(False)
        self._current_gesture_idx = 0
        self._gesture_data.clear()
        self._log("Starting calibration...")
        self._start_next_gesture()

    def _start_next_gesture(self):
        """Start recording the next gesture."""
        if self._current_gesture_idx >= len(self.CALIBRATION_GESTURES):
            self._finish_calibration()
            return

        gesture_name, instruction = self.CALIBRATION_GESTURES[self._current_gesture_idx]

        self.gesture_name_label.setText(gesture_name.upper())
        self.gesture_instruction_label.setText(instruction)
        self.status_label.setText(f"Get ready... ({self._countdown_s}s)")
        self.progress_bar.setValue(0)

        # Countdown before recording
        self._countdown_remaining = self._countdown_s
        QTimer.singleShot(1000, self._countdown_tick)

    def _countdown_tick(self):
        """Handle countdown tick."""
        self._countdown_remaining -= 1
        if self._countdown_remaining > 0:
            self.status_label.setText(f"Get ready... ({self._countdown_remaining}s)")
            QTimer.singleShot(1000, self._countdown_tick)
        else:
            self._start_recording()

    def _start_recording(self):
        """Start recording EMG data."""
        gesture_name, _ = self.CALIBRATION_GESTURES[self._current_gesture_idx]
        self.status_label.setText(f"Recording {gesture_name}...")
        self._is_recording = True
        self._recording_buffer = []
        self._recording_elapsed = 0

        # Connect to device data
        if hasattr(self.device, 'data_ready'):
            self.device.data_ready.connect(self._on_data_received)

        # Start timer for progress updates
        self._recording_timer.start(100)  # 100ms ticks

    @Slot(object)
    def _on_data_received(self, data):
        """Handle incoming EMG data during recording."""
        if self._is_recording and data is not None:
            self._recording_buffer.append(data.copy())

    def _on_recording_tick(self):
        """Handle recording progress tick."""
        self._recording_elapsed += 0.1
        self.progress_bar.setValue(int(self._recording_elapsed * 10))

        if self._recording_elapsed >= self._recording_duration_s:
            self._stop_recording()

    def _stop_recording(self):
        """Stop recording and process data."""
        self._recording_timer.stop()
        self._is_recording = False

        # Disconnect from device
        if hasattr(self.device, 'data_ready'):
            try:
                self.device.data_ready.disconnect(self._on_data_received)
            except:
                pass

        gesture_name, _ = self.CALIBRATION_GESTURES[self._current_gesture_idx]

        if self._recording_buffer:
            # Concatenate all recorded data
            data = np.vstack(self._recording_buffer)
            self._gesture_data[gesture_name] = data
            self._log(f"Recorded {gesture_name}: {data.shape[0]} samples")
        else:
            self._log(f"No data recorded for {gesture_name}")

        # Move to next gesture
        self._current_gesture_idx += 1
        QTimer.singleShot(500, self._start_next_gesture)

    def _finish_calibration(self):
        """Finish calibration and compute result."""
        self.status_label.setText("Processing calibration...")
        self.gesture_name_label.setText("Processing")
        self.gesture_instruction_label.setText("Computing electrode orientation...")

        try:
            device_name = self.device.name if hasattr(self.device, 'name') else "unknown"
            no_reference = not self.calibrator.has_reference

            # calibrate() calls calibrate_from_data() internally, which:
            #   - runs compute_channel_energy() per gesture
            #   - runs _select_sync_pattern() to pick the best sync gesture
            #   - runs find_rotation_offset() (cross-correlation)
            #   - runs create_channel_mapping() with correct 32-ch topology
            #   - saves the result to disk
            # When there is no existing reference, save_as_reference=True makes
            # this session's patterns the reference for all future calibrations.
            self._calibration_result = self.calibrator.calibrate(
                calibration_data=self._gesture_data,
                device_name=device_name,
                save_as_reference=no_reference,
            )

            offset     = self._calibration_result.rotation_offset
            confidence = self._calibration_result.confidence

            self._log("Calibration complete!")
            self._log(f"  Sync gesture : {self._calibration_result.metadata.get('sync_gesture', 'n/a')}")
            self._log(f"  Rotation offset : {offset} channels")
            self._log(f"  Confidence : {confidence:.2%}")
            if no_reference:
                self._log("  Saved as reference calibration.")

            self.gesture_name_label.setText("Complete!")
            self.gesture_instruction_label.setText(
                f"Offset: {offset}  |  Confidence: {confidence:.2%}"
            )
            self.status_label.setText("Calibration successful")
            self.finish_btn.setEnabled(True)

        except Exception as e:
            self._log(f"Calibration failed: {e}")
            self.status_label.setText(f"Error: {e}")
            self.start_btn.setEnabled(True)

    def get_calibration_result(self):
        """Get the calibration result."""
        return self._calibration_result
