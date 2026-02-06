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
    """

    calibration_complete = Signal(object)  # Emits CalibrationResult

    CALIBRATION_GESTURES = [
        ("rest", "Keep your hand relaxed"),
        ("fist", "Make a tight fist"),
        ("extension", "Extend all fingers (open hand)"),
        ("flexion", "Flex your wrist down")
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
            "This calibration will determine the orientation of the electrode bracelet.\n"
            "You will be asked to perform several gestures. Hold each gesture for 3 seconds.\n\n"
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
            self._log(f"✓ Recorded {gesture_name}: {data.shape[0]} samples")
        else:
            self._log(f"✗ No data recorded for {gesture_name}")

        # Move to next gesture
        self._current_gesture_idx += 1
        QTimer.singleShot(500, self._start_next_gesture)

    def _finish_calibration(self):
        """Finish calibration and compute result."""
        self.status_label.setText("Processing calibration...")
        self.gesture_name_label.setText("Processing")
        self.gesture_instruction_label.setText("Computing electrode orientation...")

        try:
            # Compute activation patterns for each gesture
            patterns = {}
            for gesture_name, data in self._gesture_data.items():
                pattern = self.calibrator.processor.compute_activation_pattern(data)
                patterns[gesture_name] = pattern

            # If we have a reference, find the rotation offset
            if self.calibrator.has_reference:
                offset, confidence = self.calibrator.processor.find_rotation_offset(
                    patterns,
                    self.calibrator.processor.get_reference_calibration().reference_patterns
                )
            else:
                # No reference - this becomes the reference
                offset = 0
                confidence = 1.0

            # Create channel mapping
            num_channels = next(iter(self._gesture_data.values())).shape[1]
            channel_mapping = [(i + offset) % num_channels for i in range(num_channels)]

            # Create calibration result
            from playagain_pipeline.calibration.calibrator import CalibrationResult

            self._calibration_result = CalibrationResult(
                created_at=datetime.now(),
                device_name=self.device.name if hasattr(self.device, 'name') else "unknown",
                num_channels=num_channels,
                rotation_offset=offset,
                channel_mapping=channel_mapping,
                confidence=confidence,
                reference_patterns=patterns,
                metadata={"gesture_count": len(self._gesture_data)}
            )

            self._log(f"✓ Calibration complete!")
            self._log(f"  Rotation offset: {offset} channels")
            self._log(f"  Confidence: {confidence:.2%}")

            self.gesture_name_label.setText("Complete!")
            self.gesture_instruction_label.setText(f"Offset: {offset}, Confidence: {confidence:.2%}")
            self.status_label.setText("Calibration successful")
            self.finish_btn.setEnabled(True)

        except Exception as e:
            self._log(f"✗ Calibration failed: {e}")
            self.status_label.setText(f"Error: {e}")
            self.start_btn.setEnabled(True)

    def get_calibration_result(self):
        """Get the calibration result."""
        return self._calibration_result
