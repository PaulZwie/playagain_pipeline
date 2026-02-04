"""
Main application window for gesture recording pipeline.

Combines all components into a unified interface for:
- Recording training data
- Calibration
- Real-time prediction
"""

import sys
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QPushButton, QComboBox, QLineEdit,
    QLabel, QGroupBox, QFormLayout, QSpinBox,
    QMessageBox, QFileDialog, QTextEdit, QSplitter,
    QStatusBar, QToolBar, QListWidget, QListWidgetItem,
    QScrollArea, QCheckBox, QGridLayout, QApplication
)
from PySide6.QtCore import Qt, Slot, QTimer
from PySide6.QtGui import QFont
from PySide6.QtCore import Qt as QtCore

from playagain_pipeline.gui.widgets.emg_plot import EMGPlotWidget, EMGPlotWindow
from playagain_pipeline.gui.widgets.protocol_widget import ProtocolWidget
from playagain_pipeline.protocols.protocol import (
    RecordingProtocol, ProtocolConfig, ProtocolPhase,
    create_quick_protocol, create_standard_protocol, create_extended_protocol
)
from playagain_pipeline.core.gesture import (
    GestureSet, create_default_gesture_set
)
from playagain_pipeline.core.session import RecordingSession
from playagain_pipeline.core.data_manager import DataManager
from playagain_pipeline.devices.emg_device import (
    DeviceManager, DeviceType, SyntheticEMGDevice
)
from playagain_pipeline.calibration.calibrator import AutoCalibrator
from playagain_pipeline.models.classifier import ModelManager, BaseClassifier


class MainWindow(QMainWindow):
    """
    Main application window for gesture recording and prediction.
    """

    def __init__(self):
        super().__init__()

        self.setWindowTitle("EMG Gesture Recording Pipeline")
        self.setMinimumSize(1200, 800)

        # Data directory - use playagain_pipeline/data folder
        # Find the playagain_pipeline directory relative to this file
        self._pipeline_dir = Path(__file__).parent.parent
        self.data_dir = self._pipeline_dir / "data"
        self.data_dir.mkdir(exist_ok=True)

        # Initialize managers
        self.data_manager = DataManager(self.data_dir)
        self.device_manager = DeviceManager()
        self.calibrator = AutoCalibrator(
            self.data_dir / "calibrations",
            num_channels=32,
            sampling_rate=2000
        )
        self.model_manager = ModelManager(self.data_dir / "models")

        # Current session
        self._current_session: Optional[RecordingSession] = None
        self._current_protocol: Optional[RecordingProtocol] = None
        self._current_model: Optional[BaseClassifier] = None

        # Prediction state
        self._prediction_buffer: Optional[np.ndarray] = None
        self._prediction_window_ms = 200
        self._is_predicting = False

        # Plot window
        self._plot_window: Optional[EMGPlotWindow] = None

        # Setup UI
        self._setup_ui()
        self._setup_connections()

        # Status update timer
        self._status_timer = QTimer(self)
        self._status_timer.timeout.connect(self._update_status)
        self._status_timer.start(1000)

        self._log("Application started")

    def _setup_ui(self):
        """Setup the main user interface."""
        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        # Main splitter
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # Left panel - Controls
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        # Tab widget for different modes
        self.mode_tabs = QTabWidget()

        # Recording tab
        recording_tab = self._create_recording_tab()
        self.mode_tabs.addTab(recording_tab, "Recording")

        # Calibration tab
        calibration_tab = self._create_calibration_tab()
        self.mode_tabs.addTab(calibration_tab, "Calibration")

        # Training tab
        training_tab = self._create_training_tab()
        self.mode_tabs.addTab(training_tab, "Training")

        # Prediction tab
        prediction_tab = self._create_prediction_tab()
        self.mode_tabs.addTab(prediction_tab, "Prediction")

        left_layout.addWidget(self.mode_tabs)

        # Log area
        log_group = QGroupBox("Log")
        log_layout = QVBoxLayout(log_group)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(150)
        log_layout.addWidget(self.log_text)
        left_layout.addWidget(log_group)

        splitter.addWidget(left_panel)

        # Right panel - Visualization
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # Button to open plot window
        plot_btn_group = QGroupBox("EMG Visualization")
        plot_btn_layout = QVBoxLayout(plot_btn_group)

        self.open_plot_btn = QPushButton("Open Plot Window")
        self.open_plot_btn.clicked.connect(self._on_open_plot_window)
        self.open_plot_btn.setMinimumHeight(50)
        plot_btn_layout.addWidget(self.open_plot_btn)

        plot_info = QLabel("Opens a separate window with real-time EMG signals\nand channel controls for easy toggling.")
        plot_info.setWordWrap(True)
        plot_info.setStyleSheet("color: gray; font-size: 11px;")
        plot_btn_layout.addWidget(plot_info)

        right_layout.addWidget(plot_btn_group)

        # Protocol display
        self.protocol_widget = ProtocolWidget()
        right_layout.addWidget(self.protocol_widget, stretch=1)

        splitter.addWidget(right_panel)

        # Set splitter proportions
        splitter.setSizes([500, 700])

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

        # Device status label
        self.device_status_label = QLabel("Device: Not connected")
        self.status_bar.addPermanentWidget(self.device_status_label)

    def _create_recording_tab(self) -> QWidget:
        """Create the recording configuration tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Session settings
        session_group = QGroupBox("Session Settings")
        session_layout = QFormLayout(session_group)

        self.subject_id_edit = QLineEdit()
        self.subject_id_edit.setPlaceholderText("e.g., subject_01")
        session_layout.addRow("Subject ID:", self.subject_id_edit)

        self.session_notes_edit = QLineEdit()
        session_layout.addRow("Notes:", self.session_notes_edit)

        layout.addWidget(session_group)

        # Device settings
        device_group = QGroupBox("Device")
        device_layout = QFormLayout(device_group)

        self.device_combo = QComboBox()
        self.device_combo.addItems(["Synthetic", "Muovi", "Muovi Plus"])
        device_layout.addRow("Device:", self.device_combo)

        self.channels_spin = QSpinBox()
        self.channels_spin.setRange(1, 128)
        self.channels_spin.setValue(32)
        device_layout.addRow("Channels:", self.channels_spin)

        self.sampling_rate_spin = QSpinBox()
        self.sampling_rate_spin.setRange(100, 10000)
        self.sampling_rate_spin.setValue(2000)
        self.sampling_rate_spin.setSingleStep(100)
        device_layout.addRow("Sample Rate:", self.sampling_rate_spin)

        # Session replay options (for synthetic device)
        self.use_session_data_cb = QCheckBox("Use Session Data for Replay")
        self.use_session_data_cb.setChecked(False)
        self.use_session_data_cb.toggled.connect(self._on_session_data_toggled)
        device_layout.addRow(self.use_session_data_cb)

        self.session_subject_combo = QComboBox()
        self.session_subject_combo.setEnabled(False)
        self.session_subject_combo.currentTextChanged.connect(self._load_available_sessions)
        device_layout.addRow("Subject:", self.session_subject_combo)

        self.session_id_combo = QComboBox()
        self.session_id_combo.setEnabled(False)
        device_layout.addRow("Session:", self.session_id_combo)


        # Excluded channels with checkboxes
        # channels_group = QGroupBox("Channel Status (Uncheck to exclude)")
        # channels_layout = QVBoxLayout(channels_group)

        # Scrollable area for channel checkboxes
        self._channels_scroll = QScrollArea()
        self._channels_scroll.setWidgetResizable(True)
        self._channels_scroll.setMaximumHeight(150)

        self._channels_scroll_widget = QWidget()
        self._channels_grid_layout = QGridLayout(self._channels_scroll_widget)
        self._channels_grid_layout.setSpacing(5)

        self.channel_checks = []
        for i in range(32):  # Default 32 channels
            check = QCheckBox(f"Ch {i+1}")
            check.setChecked(True)
            self._channels_grid_layout.addWidget(check, i // 8, i % 8)
            self.channel_checks.append(check)

        # Add stretch to bottom of grid layout
        self._channels_grid_layout.setRowStretch(4, 1)
        self._channels_scroll.setWidget(self._channels_scroll_widget)
        # channels_layout.addWidget(self._channels_scroll)
        # device_layout.addRow(channels_group)

        device_btn_layout = QHBoxLayout()
        self.connect_btn = QPushButton("Connect")
        self.connect_btn.clicked.connect(self._on_connect_device)
        device_btn_layout.addWidget(self.connect_btn)

        self.disconnect_btn = QPushButton("Disconnect")
        self.disconnect_btn.setEnabled(False)
        self.disconnect_btn.clicked.connect(self._on_disconnect_device)
        device_btn_layout.addWidget(self.disconnect_btn)
        device_layout.addRow(device_btn_layout)

        layout.addWidget(device_group)

        # Gesture set selection
        gesture_group = QGroupBox("Gestures")
        gesture_layout = QFormLayout(gesture_group)

        self.gesture_set_combo = QComboBox()
        self.gesture_set_combo.addItems(["Default (4 gestures)", "Custom..."])
        gesture_layout.addRow("Gesture Set:", self.gesture_set_combo)

        layout.addWidget(gesture_group)

        # Protocol selection
        protocol_group = QGroupBox("Protocol")
        protocol_layout = QFormLayout(protocol_group)

        self.protocol_combo = QComboBox()
        self.protocol_combo.addItems(["Quick (3 reps)", "Standard (5 reps)", "Extended (10 reps)"])
        protocol_layout.addRow("Protocol:", self.protocol_combo)

        layout.addWidget(protocol_group)

        # Recording controls
        control_group = QGroupBox("Recording")
        control_layout = QHBoxLayout(control_group)

        self.start_recording_btn = QPushButton("Start Recording")
        self.start_recording_btn.clicked.connect(self._on_start_recording)
        control_layout.addWidget(self.start_recording_btn)

        self.stop_recording_btn = QPushButton("Stop Recording")
        self.stop_recording_btn.setEnabled(False)
        self.stop_recording_btn.clicked.connect(self._on_stop_recording)
        control_layout.addWidget(self.stop_recording_btn)

        layout.addWidget(control_group)

        layout.addStretch()
        return tab

    def _create_calibration_tab(self) -> QWidget:
        """Create the calibration tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Calibration status
        status_group = QGroupBox("Calibration Status")
        status_layout = QFormLayout(status_group)

        self.cal_status_label = QLabel("No calibration loaded")
        status_layout.addRow("Status:", self.cal_status_label)

        self.cal_confidence_label = QLabel("-")
        status_layout.addRow("Confidence:", self.cal_confidence_label)

        self.cal_rotation_label = QLabel("-")
        status_layout.addRow("Rotation Offset:", self.cal_rotation_label)

        layout.addWidget(status_group)

        # Calibration controls
        control_group = QGroupBox("Calibration")
        control_layout = QVBoxLayout(control_group)

        self.start_cal_btn = QPushButton("Start Calibration")
        self.start_cal_btn.clicked.connect(self._on_start_calibration)
        control_layout.addWidget(self.start_cal_btn)

        self.save_ref_btn = QPushButton("Save as Reference")
        self.save_ref_btn.setEnabled(False)
        self.save_ref_btn.clicked.connect(self._on_save_reference)
        control_layout.addWidget(self.save_ref_btn)

        self.load_cal_btn = QPushButton("Load Calibration...")
        self.load_cal_btn.clicked.connect(self._on_load_calibration)
        control_layout.addWidget(self.load_cal_btn)

        layout.addWidget(control_group)

        layout.addStretch()
        return tab

    def _create_training_tab(self) -> QWidget:
        """Create the model training tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Dataset selection
        dataset_group = QGroupBox("Dataset")
        dataset_layout = QVBoxLayout(dataset_group)

        self.refresh_datasets_btn = QPushButton("Refresh")
        self.refresh_datasets_btn.clicked.connect(self._refresh_datasets)
        dataset_layout.addWidget(self.refresh_datasets_btn)

        self.dataset_list = QListWidget()
        dataset_layout.addWidget(self.dataset_list)

        self.create_dataset_btn = QPushButton("Create Dataset from Sessions...")
        self.create_dataset_btn.clicked.connect(self._on_create_dataset)
        dataset_layout.addWidget(self.create_dataset_btn)

        layout.addWidget(dataset_group)

        # Model selection
        model_group = QGroupBox("Model")
        model_layout = QFormLayout(model_group)

        self.model_type_combo = QComboBox()
        self.model_type_combo.addItems(["SVM", "Random Forest", "LDA"])
        model_layout.addRow("Model Type:", self.model_type_combo)

        train_btn_layout = QHBoxLayout()
        self.train_btn = QPushButton("Train Model")
        self.train_btn.clicked.connect(self._on_train_model)
        train_btn_layout.addWidget(self.train_btn)
        model_layout.addRow(train_btn_layout)

        layout.addWidget(model_group)

        # Trained models
        models_group = QGroupBox("Trained Models")
        models_layout = QVBoxLayout(models_group)

        self.models_list = QListWidget()
        models_layout.addWidget(self.models_list)

        self.refresh_models_btn = QPushButton("Refresh")
        self.refresh_models_btn.clicked.connect(self._refresh_models)
        models_layout.addWidget(self.refresh_models_btn)

        layout.addWidget(models_group)

        layout.addStretch()
        return tab

    def _create_prediction_tab(self) -> QWidget:
        """Create the real-time prediction tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Model selection
        model_group = QGroupBox("Model")
        model_layout = QFormLayout(model_group)

        self.pred_model_combo = QComboBox()
        model_layout.addRow("Model:", self.pred_model_combo)

        self.load_model_btn = QPushButton("Load Model")
        self.load_model_btn.clicked.connect(self._on_load_model)
        model_layout.addRow(self.load_model_btn)

        layout.addWidget(model_group)

        # Prediction display
        pred_group = QGroupBox("Prediction")
        pred_layout = QVBoxLayout(pred_group)

        self.prediction_label = QLabel("No prediction")
        pred_font = QFont()
        pred_font.setPointSize(24)
        pred_font.setBold(True)
        self.prediction_label.setFont(pred_font)
        self.prediction_label.setAlignment(Qt.AlignCenter)
        pred_layout.addWidget(self.prediction_label)

        self.confidence_label = QLabel("Confidence: -")
        self.confidence_label.setAlignment(Qt.AlignCenter)
        pred_layout.addWidget(self.confidence_label)

        layout.addWidget(pred_group)

        # Prediction controls
        control_group = QGroupBox("Controls")
        control_layout = QHBoxLayout(control_group)

        self.start_pred_btn = QPushButton("Start Prediction")
        self.start_pred_btn.clicked.connect(self._on_start_prediction)
        control_layout.addWidget(self.start_pred_btn)

        self.stop_pred_btn = QPushButton("Stop Prediction")
        self.stop_pred_btn.setEnabled(False)
        self.stop_pred_btn.clicked.connect(self._on_stop_prediction)
        control_layout.addWidget(self.stop_pred_btn)

        layout.addWidget(control_group)

        layout.addStretch()
        return tab

    def _setup_connections(self):
        """Setup signal connections."""
        self.protocol_widget.step_started.connect(self._on_step_started)
        self.protocol_widget.step_completed.connect(self._on_step_completed)
        self.protocol_widget.protocol_completed.connect(self._on_protocol_completed)

    def _log(self, message: str):
        """Add message to log."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")

    @Slot()
    def _on_open_plot_window(self):
        """Open the plot window."""
        if self._plot_window is None or not self._plot_window.isVisible():
            device = self.device_manager.device
            num_channels = device.num_channels if device else 32
            sampling_rate = device.sampling_rate if device else 2000

            self._plot_window = EMGPlotWindow(
                num_channels=num_channels,
                sampling_rate=sampling_rate,
                parent=self
            )
            self._plot_window.closed.connect(self._on_plot_window_closed)
            self._plot_window.show()
            self._log("Opened plot window")
        else:
            self._plot_window.raise_()
            self._plot_window.activateWindow()

    @Slot()
    def _on_plot_window_closed(self):
        """Handle plot window close."""
        self._log("Plot window closed")
        self._plot_window = None

    def closeEvent(self, event):
        """Handle window close."""
        # Stop recording if active
        if self._current_session and self._current_session.is_recording:
            self._on_stop_recording()

        # Disconnect device
        self.device_manager.stop_and_disconnect()

        event.accept()

    def _update_status(self):
        """Update status bar."""
        device = self.device_manager.device
        if device and device.is_connected:
            status = "Connected" if not device.is_streaming else "Streaming"
            self.device_status_label.setText(
                f"Device: {status} ({device.num_channels}ch @ {device.sampling_rate}Hz)"
            )
        else:
            self.device_status_label.setText("Device: Not connected")

        # In playagain_pipeline/gui/main_window.py

    @Slot()
    def _on_connect_device(self):
        """
        Connect to the selected device with proper Muovi handshake handling.
        """
        # 1. Determine which device type is selected
        device_text = self.device_combo.currentText()
        if device_text == "Muovi":
            device_type = DeviceType.MUOVI
        elif device_text == "Muovi Plus":
            device_type = DeviceType.MUOVI_PLUS
        else:
            device_type = DeviceType.SYNTHETIC

        try:
            # 2. Create the device instance through the manager
            # This will clean up any existing device automatically
            kwargs = {
                "num_channels": self.channels_spin.value(),
                "sampling_rate": self.sampling_rate_spin.value()
            }

            # Add session replay parameters for synthetic device
            if device_type == DeviceType.SYNTHETIC and self.use_session_data_cb.isChecked():
                kwargs.update({
                    "use_session_data": True,
                    "session_subject_id": self.session_subject_combo.currentText(),
                    "session_id": self.session_id_combo.currentText(),
                    "data_dir": str(self.data_dir)
                })

            device = self.device_manager.create_device(device_type, **kwargs)

            # Update channel checkboxes based on device channel count
            self._update_channel_checkboxes(device.num_channels)

            # 3. Connect all signals BEFORE calling connect()
            # This is critical to catch the very first Muovi handshake packets
            device.data_ready.connect(self._on_data_received)
            device.connected.connect(self._on_device_connected)
            device.error.connect(self._on_device_error)

            # 4. Initiate connection based on device architecture
            if device_type in (DeviceType.MUOVI, DeviceType.MUOVI_PLUS):
                self._log(f"Starting {device_text} Server on port 54321...")
                self._log("Action Required: Please turn on your Muovi device now.")

                # Update UI to 'Waiting' state
                self.connect_btn.setEnabled(False)
                self.connect_btn.setText("Waiting for Device...")
                self.disconnect_btn.setEnabled(True)  # Allow user to cancel the server

                # Start the TCP/IP Listening Server
                device.connect()

            else:
                # Synthetic or other local devices connect immediately
                if device.connect():
                    self._log(f"Connected to {device_text}")
                    # Auto-start for synthetic
                    device.start_streaming()
                else:
                    self._log(f"Failed to initialize {device_text}")

        except ImportError as e:
            self._log(f"Dependency Error: {e}")
            QMessageBox.critical(self, "Missing Library",
                                 f"The device_interfaces library is required for real hardware.\nError: {e}")
        except Exception as e:
            self._log(f"Connection Error: {str(e)}")
            # Reset UI on hard failure
            self.connect_btn.setEnabled(True)
            self.connect_btn.setText("Connect")
            self.disconnect_btn.setEnabled(False)

    @Slot(bool)
    def _on_device_connected(self, connected: bool):
        if connected:
            self._log("Muovi Handshake Successful!")
            self.connect_btn.setText("Connect")
            self.connect_btn.setEnabled(False)
            self.disconnect_btn.setEnabled(True)

            # Start streaming immediately after connection signal
            self.device_manager.device.start_streaming()
        else:
            self._log("Device Disconnected.")
            self.connect_btn.setEnabled(True)
            self.disconnect_btn.setEnabled(False)

    @Slot(str)
    def _on_device_error(self, error: str):
        """Handle device errors."""
        self._log(f"Device error: {error}")

    @Slot()
    def _on_disconnect_device(self):
        """Disconnect from device."""
        self.device_manager.stop_and_disconnect()
        self.connect_btn.setEnabled(True)
        self.disconnect_btn.setEnabled(False)
        self._log("Disconnected from device")

    @Slot(np.ndarray)
    def _on_data_received(self, data: np.ndarray):
        """Handle incoming EMG data."""
        # Filter out excluded channels (keep only checked channels)
        excluded = self._get_excluded_channels()
        if excluded:
            # Create mask for channels to keep (inverse of excluded)
            keep_mask = np.ones(data.shape[1], dtype=bool)
            for ch_idx in excluded:
                if ch_idx < len(keep_mask):
                    keep_mask[ch_idx] = False
            data = data[:, keep_mask]

        # Update plot window if open
        if self._plot_window and self._plot_window.isVisible():
            # Update plot channel count if changed
            if data.shape[1] != self._plot_window.num_channels:
                self._plot_window.set_num_channels(data.shape[1])

            # Update plot
            self._plot_window.update_data(data)

        # Record if session is active
        if self._current_session and self._current_session.is_recording:
            self._current_session.add_data(data)

        # Predict if enabled
        if self._is_predicting and self._current_model:
            self._update_prediction(data)

    # Recording handlers
    @Slot()
    def _on_start_recording(self):
        """Start a new recording session."""
        if not self.subject_id_edit.text():
            QMessageBox.warning(self, "Warning", "Please enter a subject ID")
            return

        device = self.device_manager.device
        if not device or not device.is_connected:
            QMessageBox.warning(self, "Warning", "Please connect a device first")
            return

        # Create gesture set
        gesture_set = create_default_gesture_set()

        # Create protocol
        if "Quick" in self.protocol_combo.currentText():
            protocol_config = create_quick_protocol()
        elif "Extended" in self.protocol_combo.currentText():
            protocol_config = create_extended_protocol()
        else:
            protocol_config = create_standard_protocol()

        self._current_protocol = RecordingProtocol(gesture_set, protocol_config)

        # Create session
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Determine actual number of channels after exclusion
        excluded = self._get_excluded_channels()
        actual_channels = device.num_channels - len(excluded)

        self._current_session = RecordingSession(
            session_id=session_id,
            subject_id=self.subject_id_edit.text(),
            device_name=device.config.device_type.name,
            num_channels=actual_channels,
            sampling_rate=device.sampling_rate,
            gesture_set=gesture_set,
            protocol_name=protocol_config.name
        )
        self._current_session.metadata.notes = self.session_notes_edit.text()

        # Setup protocol widget
        self.protocol_widget.set_protocol(self._current_protocol)

        # Start recording
        self._current_session.start_recording()
        self.protocol_widget.start()

        self.start_recording_btn.setEnabled(False)
        self.stop_recording_btn.setEnabled(True)
        self._log(f"Started recording session: {session_id}")

    @Slot()
    def _on_stop_recording(self):
        """Stop the current recording."""
        if self._current_session:
            self._current_session.stop_recording()
            self.protocol_widget.stop()

            # Save session
            path = self.data_manager.save_session(self._current_session)
            self._log(f"Saved session to {path}")

            self._current_session = None

        self.start_recording_btn.setEnabled(True)
        self.stop_recording_btn.setEnabled(False)

    @Slot(object)
    def _on_step_completed(self, step):
        """Handle protocol step completion."""
        # End trial when HOLD phase completes
        if step.phase == ProtocolPhase.HOLD and step.is_recording:
            if self._current_session and step.gesture:
                self._current_session.end_trial()
                self._log(f"Trial recorded: {step.gesture.display_name}")

        # Update synthetic device gesture during CUE
        if step.phase == ProtocolPhase.CUE and step.gesture:
            device = self.device_manager.device
            if isinstance(device, SyntheticEMGDevice) and step.gesture:
                device.set_gesture(step.gesture.name)

    @Slot(object)
    def _on_step_started(self, step):
        """Handle protocol step starting (called from protocol widget)."""
        # Start trial at beginning of HOLD phase
        if step.phase == ProtocolPhase.HOLD and step.is_recording and step.gesture:
            if self._current_session:
                self._current_session.start_trial(step.gesture.name)

    @Slot()
    def _on_protocol_completed(self):
        """Handle protocol completion."""
        self._log("Protocol completed")
        self._on_stop_recording()
        QMessageBox.information(self, "Complete", "Recording protocol completed!")

    # Calibration handlers
    @Slot()
    def _on_start_calibration(self):
        """Start calibration protocol."""
        self._log("Calibration not yet implemented in UI")
        QMessageBox.information(
            self, "Info",
            "Calibration protocol coming soon!\n"
            "For now, use the API directly."
        )

    @Slot()
    def _on_save_reference(self):
        """Save current calibration as reference."""
        if self.calibrator.current_calibration:
            self.calibrator.save_as_reference(self.calibrator.current_calibration)
            self._log("Saved calibration as reference")

    @Slot()
    def _on_load_calibration(self):
        """Load a calibration file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Calibration",
            str(self.data_dir / "calibrations"),
            "JSON Files (*.json)"
        )
        if file_path:
            from playagain_pipeline.calibration.calibrator import CalibrationResult
            cal = CalibrationResult.load(Path(file_path))
            self.calibrator._current_calibration = cal
            self._update_calibration_display()
            self._log(f"Loaded calibration: {file_path}")

    def _update_calibration_display(self):
        """Update calibration status display."""
        cal = self.calibrator.current_calibration
        if cal:
            self.cal_status_label.setText("Loaded")
            self.cal_confidence_label.setText(f"{cal.confidence:.2%}")
            self.cal_rotation_label.setText(f"{cal.rotation_offset} channels")
            self.save_ref_btn.setEnabled(True)
        else:
            self.cal_status_label.setText("No calibration loaded")
            self.cal_confidence_label.setText("-")
            self.cal_rotation_label.setText("-")

    # Training handlers
    @Slot()
    def _refresh_datasets(self):
        """Refresh list of available datasets."""
        self.dataset_list.clear()
        for name in self.data_manager.list_datasets():
            self.dataset_list.addItem(name)

    @Slot()
    def _refresh_models(self):
        """Refresh list of trained models."""
        self.models_list.clear()
        self.pred_model_combo.clear()
        for name in self.model_manager.list_models():
            self.models_list.addItem(name)
            self.pred_model_combo.addItem(name)

    @Slot()
    def _on_create_dataset(self):
        """Create a dataset from recorded sessions with customizable parameters."""
        # Create a dialog for dataset creation parameters
        from PySide6.QtWidgets import QDialog, QFormLayout, QSpinBox, QCheckBox, QDialogButtonBox, QComboBox, QListWidget, QListWidgetItem

        dialog = QDialog(self)
        dialog.setWindowTitle("Create Dataset from Sessions")
        layout = QFormLayout(dialog)

        # Dataset name
        name_edit = QLineEdit("my_dataset")
        layout.addRow("Dataset Name:", name_edit)

        # Window parameters
        window_size_spin = QSpinBox()
        window_size_spin.setRange(50, 1000)
        window_size_spin.setValue(200)
        window_size_spin.setSuffix(" ms")
        layout.addRow("Window Size:", window_size_spin)

        stride_spin = QSpinBox()
        stride_spin.setRange(10, 500)
        stride_spin.setValue(50)
        stride_spin.setSuffix(" ms")
        layout.addRow("Stride:", stride_spin)

        # Include invalid trials
        include_invalid_cb = QCheckBox()
        include_invalid_cb.setChecked(False)
        layout.addRow("Include Invalid Trials:", include_invalid_cb)

        # Subject selection
        subject_list = QListWidget()
        subject_list.setSelectionMode(QListWidget.MultiSelection)
        subjects = self.data_manager.list_subjects()
        for subject in subjects:
            item = QListWidgetItem(subject)
            item.setSelected(True)  # Select all by default
            subject_list.addItem(item)
        layout.addRow("Subjects:", subject_list)

        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addRow(buttons)

        if dialog.exec() == QDialog.Accepted:
            try:
                # Get selected subjects
                selected_subjects = []
                for i in range(subject_list.count()):
                    if subject_list.item(i).isSelected():
                        selected_subjects.append(subject_list.item(i).text())

                if not selected_subjects:
                    QMessageBox.warning(self, "Warning", "Please select at least one subject")
                    return

                # Create dataset with custom parameters
                dataset = self.data_manager.create_dataset(
                    name=name_edit.text(),
                    subject_ids=selected_subjects,
                    window_size_ms=window_size_spin.value(),
                    window_stride_ms=stride_spin.value(),
                    include_invalid=include_invalid_cb.isChecked()
                )

                self.data_manager.save_dataset(dataset)
                self._log(f"Created dataset '{name_edit.text()}' with {dataset['metadata']['num_samples']} samples")
                self._refresh_datasets()

            except Exception as e:
                self._log(f"Error creating dataset: {e}")
                QMessageBox.critical(self, "Error", f"Failed to create dataset: {e}")

    @Slot()
    def _on_train_model(self):
        """Train a model on selected dataset."""
        selected = self.dataset_list.currentItem()
        if not selected:
            QMessageBox.warning(self, "Warning", "Please select a dataset")
            return

        try:
            # Load dataset
            dataset = self.data_manager.load_dataset(selected.text())

            # Create model
            model_type = self.model_type_combo.currentText().lower().replace(" ", "_")
            model = self.model_manager.create_model(model_type)

            # Train
            self._log(f"Training {model_type} model...")
            results = self.model_manager.train_model(model, dataset)

            self._log(
                f"Training complete - "
                f"Train acc: {results['training_accuracy']:.2%}, "
                f"Val acc: {results['validation_accuracy']:.2%}"
            )

            self._refresh_models()

        except Exception as e:
            self._log(f"Error training model: {e}")

    # Prediction handlers
    @Slot()
    def _on_load_model(self):
        """Load selected model for prediction."""
        model_name = self.pred_model_combo.currentText()
        if not model_name:
            return

        try:
            self._current_model = self.model_manager.load_model(model_name)
            self._log(f"Loaded model: {model_name}")

            # Initialize prediction buffer
            window_samples = int(
                self._prediction_window_ms *
                self._current_model.metadata.sampling_rate / 1000
            )
            self._prediction_buffer = np.zeros((
                window_samples,
                self._current_model.metadata.num_channels
            ))

        except Exception as e:
            self._log(f"Error loading model: {e}")

    @Slot()
    def _on_start_prediction(self):
        """Start real-time prediction."""
        if not self._current_model:
            QMessageBox.warning(self, "Warning", "Please load a model first")
            return

        device = self.device_manager.device
        if not device or not device.is_streaming:
            QMessageBox.warning(self, "Warning", "Please connect and start device")
            return

        self._is_predicting = True
        self.start_pred_btn.setEnabled(False)
        self.stop_pred_btn.setEnabled(True)
        self._log("Started prediction")

    @Slot()
    def _on_stop_prediction(self):
        """Stop real-time prediction."""
        self._is_predicting = False
        self.start_pred_btn.setEnabled(True)
        self.stop_pred_btn.setEnabled(False)
        self.prediction_label.setText("No prediction")
        self.confidence_label.setText("Confidence: -")
        self._log("Stopped prediction")

    def _update_prediction(self, data: np.ndarray):
        """Update prediction with new data."""
        if self._prediction_buffer is None or self._current_model is None:
            return

        # Roll buffer and add new data
        n_samples = min(data.shape[0], len(self._prediction_buffer))
        self._prediction_buffer = np.roll(self._prediction_buffer, -n_samples, axis=0)
        self._prediction_buffer[-n_samples:] = data[:n_samples]

        # Make prediction
        try:
            X = self._prediction_buffer[np.newaxis, :, :]
            pred = self._current_model.predict(X)[0]
            proba = self._current_model.predict_proba(X)[0]

            # Get class name
            class_names = self._current_model.metadata.class_names
            class_name = class_names.get(int(pred), f"Class {pred}")
            confidence = proba[pred]

            self.prediction_label.setText(class_name.replace("_", " ").title())
            self.confidence_label.setText(f"Confidence: {confidence:.1%}")

        except Exception as e:
            pass  # Silently handle prediction errors

    def _get_excluded_channels(self) -> list[int]:
        """Get list of excluded channel indices (0-based) from unchecked boxes."""
        excluded = []
        for i, check in enumerate(self.channel_checks):
            if not check.isChecked():
                excluded.append(i)
        return excluded

    def _update_channel_checkboxes(self, num_channels: int):
        """Update the number of channel checkboxes."""
        # Clear existing checkboxes
        for check in self.channel_checks:
            check.deleteLater()
        self.channel_checks.clear()

        # Clear grid layout
        while self._channels_grid_layout.count():
            item = self._channels_grid_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Create new checkboxes
        for i in range(num_channels):
            check = QCheckBox(f"Ch {i+1}")
            check.setChecked(True)
            self._channels_grid_layout.addWidget(check, i // 8, i % 8)
            self.channel_checks.append(check)

        # Add stretch to bottom
        self._channels_grid_layout.setRowStretch(max(4, (num_channels + 7) // 8), 1)

    def _on_session_data_toggled(self, checked: bool):
        """Handle toggling of session data checkbox."""
        enabled = checked
        self.session_subject_combo.setEnabled(enabled)
        self.session_id_combo.setEnabled(enabled)

        if enabled:
            # Load available subjects for session replay
            self.session_subject_combo.clear()
            self.session_subject_combo.addItems(self.data_manager.list_subjects())

            # Auto-select current subject if available
            current_subject = self.subject_id_edit.text()
            if current_subject and current_subject in [self.session_subject_combo.itemText(i) for i in range(self.session_subject_combo.count())]:
                self.session_subject_combo.setCurrentText(current_subject)

            # Load available sessions for the selected subject
            self._load_available_sessions()

    def _load_available_sessions(self):
        """Load available session IDs for the selected subject."""
        subject = self.session_subject_combo.currentText()
        if not subject:
            return

        # Clear existing session IDs
        self.session_id_combo.clear()

        # Load sessions for the selected subject
        sessions = self.data_manager.list_sessions(subject)
        self.session_id_combo.addItems(sessions)

        # Auto-select latest session if available
        if sessions:
            self.session_id_combo.setCurrentText(sessions[-1])

def main():
    """Run the application."""

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
