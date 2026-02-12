"""
Main application window for gesture recording pipeline.

Combines all components into a unified interface for:
- Recording training data
- Calibration
- Real-time prediction
"""

import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from PySide6.QtCore import Qt, Slot, QTimer, QThread, Signal, QMutex, QMutexLocker
from PySide6.QtGui import QFont, QAction
from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTabWidget, QPushButton, QComboBox,
                               QLineEdit, QLabel, QGroupBox, QFormLayout, QSpinBox, QMessageBox, QFileDialog, QTextEdit,
                               QSplitter, QStatusBar, QListWidget, QScrollArea, QCheckBox,
                               QGridLayout, QApplication, QMenuBar, QMenu, QDialog, QListWidgetItem)

from playagain_pipeline.calibration.calibrator import AutoCalibrator
from playagain_pipeline.config.config import get_default_config, PipelineConfig
from playagain_pipeline.core.data_manager import DataManager
from playagain_pipeline.core.gesture import (create_default_gesture_set)
from playagain_pipeline.core.session import RecordingSession
from playagain_pipeline.devices.emg_device import (DeviceManager, DeviceType, SyntheticEMGDevice)
from playagain_pipeline.gui.widgets.emg_plot import EMGPlotWindow
from playagain_pipeline.gui.widgets.protocol_widget import ProtocolWidget
from playagain_pipeline.models.classifier import ModelManager, BaseClassifier
from playagain_pipeline.protocols.protocol import (RecordingProtocol, ProtocolPhase,
                                                   create_quick_protocol, create_standard_protocol,
                                                   create_extended_protocol)


class PredictionWorker(QThread):
    """Background worker for model prediction to avoid blocking the GUI thread."""
    prediction_ready = Signal(object, object)  # pred_class, proba_array

    def __init__(self, parent=None):
        super().__init__(parent)
        self._model = None
        self._buffer = None
        self._new_data = False
        self._running = False
        self._mutex = QMutex()

    def set_model(self, model):
        self._model = model

    def update_buffer(self, buffer: np.ndarray):
        """Thread-safe buffer update (called from main thread)."""
        locker = QMutexLocker(self._mutex)
        self._buffer = buffer.copy()
        self._new_data = True

    def run(self):
        self._running = True
        while self._running:
            buffer = None
            self._mutex.lock()
            if self._new_data and self._buffer is not None:
                buffer = self._buffer.copy()
                self._new_data = False
            self._mutex.unlock()

            if buffer is not None and self._model is not None:
                try:
                    X = buffer[np.newaxis, :, :]
                    pred = self._model.predict(X)[0]
                    proba = self._model.predict_proba(X)[0]
                    self.prediction_ready.emit(pred, proba)
                except Exception:
                    pass

            self.msleep(100)  # Cap at ~10 predictions/sec

    def stop(self):
        self._running = False
        self.wait()


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
        self.calibrator = AutoCalibrator(self.data_dir / "calibrations", num_channels=32, sampling_rate=2000)
        self.model_manager = ModelManager(self.data_dir / "models")

        # Current session
        self._current_session: Optional[RecordingSession] = None
        self._current_protocol: Optional[RecordingProtocol] = None
        self._current_model: Optional[BaseClassifier] = None

        # Prediction state
        self._prediction_buffer: Optional[np.ndarray] = None
        self._prediction_window_ms = 200
        self._is_predicting = False
        self._prediction_worker: Optional[PredictionWorker] = None

        # Plot window
        self._plot_window: Optional[EMGPlotWindow] = None

        # Ground truth label for session replay
        self._current_ground_truth_label: Optional[str] = None

        # Setup UI
        self._setup_ui()
        self._setup_connections()

        # Status update timer
        self._status_timer = QTimer(self)
        self._status_timer.timeout.connect(self._update_status)
        self._status_timer.start(1000)

        # Load configuration
        self.config = get_default_config()
        # Try to load from file if exists
        config_path = self._pipeline_dir / "config.json"
        if config_path.exists():
            try:
                self.config = PipelineConfig.load(config_path)
                self._log("Loaded configuration from file")
            except Exception as e:
                self._log(f"Failed to load config, using defaults: {e}")

        self._log("Application started")

    def _setup_ui(self):
        """Setup the main user interface."""
        # Setup menu bar
        self._setup_menu_bar()

        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        # Main splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
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

        plot_info = QLabel(
            "Opens a separate window with real-time EMG signals\nand channel controls for easy toggling.")
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
            check = QCheckBox(f"Ch {i + 1}")
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

        # Information about calibration
        info_group = QGroupBox("About Calibration")
        info_layout = QVBoxLayout(info_group)

        info_label = QLabel(
            "Calibration determines the electrode bracelet orientation by analyzing\n"
            "EMG patterns from individual finger movements.\n\n"
            "Single finger gestures produce distinct, localized muscle activation\n"
            "patterns that help accurately detect electrode positioning."
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: #555; font-size: 11px;")
        info_layout.addWidget(info_label)
        layout.addWidget(info_group)

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
        control_group = QGroupBox("Calibration Actions")
        control_layout = QVBoxLayout(control_group)

        # Info about gestures used
        gesture_info = QLabel(
            "Gestures used: Rest, Index/Middle/Ring/Pinky/Thumb flex,\n"
            "Index extend, Wrist flex"
        )
        gesture_info.setStyleSheet("color: #666; font-size: 10px; font-style: italic;")
        control_layout.addWidget(gesture_info)

        self.start_cal_btn = QPushButton("Start Calibration")
        self.start_cal_btn.clicked.connect(self._on_start_calibration)
        self.start_cal_btn.setMinimumHeight(40)
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
        self.model_type_combo.addItems(["SVM", "Random Forest", "LDA", "CatBoost", "MLP", "CNN", "InceptionAttn"])
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
        self.prediction_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pred_layout.addWidget(self.prediction_label)

        self.confidence_label = QLabel("Confidence: -")
        self.confidence_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
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

    def _setup_menu_bar(self):
        """Setup the application menu bar."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("File")

        export_action = QAction("Export Dataset...", self)
        export_action.triggered.connect(self._on_export_dataset)
        file_menu.addAction(export_action)

        file_menu.addSeparator()

        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Settings menu
        settings_menu = menubar.addMenu("Settings")

        config_action = QAction("Configuration...", self)
        config_action.triggered.connect(self._on_open_config)
        settings_menu.addAction(config_action)

        # Tools menu
        tools_menu = menubar.addMenu("Tools")

        adv_training_action = QAction("Advanced Training...", self)
        adv_training_action.triggered.connect(self._on_advanced_training)
        tools_menu.addAction(adv_training_action)

        feature_editor_action = QAction("Feature Selection...", self)
        feature_editor_action.triggered.connect(self._on_feature_selection)
        tools_menu.addAction(feature_editor_action)

        tools_menu.addSeparator()

        bracelet_viz_action = QAction("Bracelet Visualization...", self)
        bracelet_viz_action.triggered.connect(self._on_bracelet_visualization)
        tools_menu.addAction(bracelet_viz_action)

        # Help menu
        help_menu = menubar.addMenu("Help")

        about_action = QAction("About", self)
        about_action.triggered.connect(self._on_about)
        help_menu.addAction(about_action)

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

            self._plot_window = EMGPlotWindow(num_channels=num_channels, sampling_rate=sampling_rate, parent=None)
            self._plot_window.closed.connect(self._on_plot_window_closed)
            self._plot_window.show()

            # If we have a current ground truth (session replay), set it
            if hasattr(self, '_current_ground_truth_label') and self._current_ground_truth_label:
                self._plot_window.set_ground_truth(self._current_ground_truth_label)

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
        # Stop prediction worker if running
        if self._prediction_worker:
            self._prediction_worker.stop()
            self._prediction_worker = None

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
            self.device_status_label.setText(f"Device: {status} ({device.num_channels}ch @ {device.sampling_rate}Hz)")
        else:
            self.device_status_label.setText("Device: Not connected")

    @Slot()
    def _on_export_dataset(self):
        """Export dataset to file."""
        selected = self.dataset_list.currentItem()
        if not selected:
            QMessageBox.warning(self, "Warning", "Please select a dataset to export")
            self.mode_tabs.setCurrentIndex(2)  # Switch to training tab
            return

        dataset_name = selected.text()
        try:
            # Get export path
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Export Dataset",
                str(Path.home() / f"{dataset_name}.npz"),
                "NumPy Archive (*.npz)"
            )

            if not file_path:
                return

            self._log(f"Exporting dataset '{dataset_name}'...")

            # Load dataset
            dataset = self.data_manager.load_dataset(dataset_name)

            # Save as npz
            np.savez_compressed(
                file_path,
                X=dataset["X"],
                y=dataset["y"],
                metadata=dataset["metadata"]
            )

            self._log(f"Dataset exported to: {file_path}")
            QMessageBox.information(self, "Success", f"Dataset exported successfully to:\n{file_path}")

        except Exception as e:
            self._log(f"Error exporting dataset: {e}")
            QMessageBox.critical(self, "Error", f"Failed to export dataset: {e}")

    @Slot()
    def _on_about(self):
        """Show about dialog."""
        QMessageBox.about(
            self,
            "About PlayAgain Pipeline",
            "<h3>PlayAgain EMG Pipeline</h3>"
            "<p>A comprehensive tool for EMG signal acquisition, processing, "
            "and gesture recognition model training.</p>"
            "<p>Version: 1.0.0</p>"
        )

    @Slot()
    def _on_open_config(self):
        """Open configuration dialog."""
        from playagain_pipeline.gui.widgets.config_dialog import ConfigurationDialog

        # Gather current config
        current_config = {
            "device": {
                "num_channels": self.channels_spin.value(),
                "sampling_rate": self.sampling_rate_spin.value()
            }
        }

        dialog = ConfigurationDialog(current_config, self)
        dialog.config_changed.connect(self._on_config_changed)

        if dialog.exec():
            # Apply changes if accepted
            self._on_config_changed(dialog.get_config())

    def _on_config_changed(self, config: dict):
        """Handle configuration changes."""
        self._log("Configuration updated")

        # Apply device settings
        if "device" in config:
            dev = config["device"]
            if "type" in dev:
                idx = self.device_combo.findText(dev["type"])
                if idx >= 0:
                    self.device_combo.setCurrentIndex(idx)
            if "num_channels" in dev:
                self.channels_spin.setValue(dev["num_channels"])
            if "sampling_rate" in dev:
                self.sampling_rate_spin.setValue(dev["sampling_rate"])

        # Apply recording settings
        if "recording" in config:
            rec = config["recording"]
            # TODO: Add recording duration controls to UI or use config values directly
            pass

        # Apply calibration settings
        if "calibration" in config:
            cal = config["calibration"]
            if self.calibrator and self.calibrator.current_calibration:
                self.calibrator.current_calibration.rotation_offset = cal.get("rotation_offset", 0)

    @Slot()
    def _on_advanced_training(self):
        """Open advanced training dialog."""
        try:
            # Get list of available datasets
            available_datasets = self.data_manager.list_datasets()
            if not available_datasets:
                QMessageBox.warning(self, "Warning", "No datasets available. Please create a dataset first.")
                self.mode_tabs.setCurrentIndex(2)  # Switch to training tab
                return

            from playagain_pipeline.gui.widgets.training_dialog import TrainingProgressDialog

            # Get available model types
            available_models = ["SVM", "CatBoost", "Random Forest", "LDA", "MLP", "CNN", "InceptionAttn"]

            # Create dialog without pre-selected dataset/model to enable selection
            dialog = TrainingProgressDialog(
                model_type=None,
                dataset=None,
                config=self.config,
                parent=self,
                available_datasets=available_datasets,
                available_models=available_models
            )

            # Switch to Training tab
            self.mode_tabs.setCurrentIndex(2)

            if dialog.exec():
                # Get trained model and save it
                model = dialog.get_trained_model()
                if model:
                    self.model_manager._current_model = model
                    model.save(self.model_manager.models_dir / model.name)
                    self._log(f"Model saved: {model.name}")
                    self._refresh_models()
        except Exception as e:
            self._log(f"Error in advanced training: {e}")

    @Slot()
    def _on_feature_selection(self):
        """Open feature selection dialog."""
        from playagain_pipeline.gui.widgets.feature_selection import FeatureSelectionDialog
        from playagain_pipeline.models.feature_pipeline import FeaturePipeline

        # In a real app, this would be stored in the model manager or config
        if not hasattr(self, '_feature_pipeline'):
            self._feature_pipeline = FeaturePipeline.create_default()

        dialog = FeatureSelectionDialog(self._feature_pipeline, self)
        if dialog.exec():
            self._log(f"Feature pipeline updated: {len(self._feature_pipeline.get_features())} features")

    @Slot()
    def _on_bracelet_visualization(self):
        """Open standalone bracelet visualization."""
        from playagain_pipeline.gui.widgets.config_dialog import BraceletVisualizationWidget

        # Create a simple dialog container
        dialog = QDialog(self)
        dialog.setWindowTitle("Bracelet Visualization")
        dialog.setMinimumSize(400, 400)

        layout = QVBoxLayout(dialog)

        viz = BraceletVisualizationWidget(num_electrodes=self.channels_spin.value())
        if self.calibrator and self.calibrator.current_calibration:
            viz.set_rotation(self.calibrator.current_calibration.rotation_offset)

        layout.addWidget(viz)

        # Add close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)

        dialog.exec()

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
            kwargs = {"num_channels": self.channels_spin.value(), "sampling_rate": self.sampling_rate_spin.value()}

            # Add session replay parameters for synthetic device
            if device_type == DeviceType.SYNTHETIC and self.use_session_data_cb.isChecked():
                kwargs.update({"use_session_data": True, "session_subject_id": self.session_subject_combo.currentText(),
                    "session_id": self.session_id_combo.currentText(), "data_dir": str(self.data_dir)})

            device = self.device_manager.create_device(device_type, **kwargs)

            # Update channel checkboxes based on device channel count
            self._update_channel_checkboxes(device.num_channels)

            # 3. Connect all signals BEFORE calling connect()
            # This is critical to catch the very first Muovi handshake packets
            device.data_ready.connect(self._on_data_received)
            device.connected.connect(self._on_device_connected)
            device.error.connect(self._on_device_error)

            # Connect ground truth signal for session replay visualization
            # This signal is emitted by SyntheticEMGDevice when replaying session data
            if hasattr(device, 'ground_truth_changed'):
                # Disconnect any previous connection to avoid duplicates
                try:
                    device.ground_truth_changed.disconnect(self._on_ground_truth_changed)
                except (TypeError, RuntimeError):
                    pass  # Not connected yet
                device.ground_truth_changed.connect(self._on_ground_truth_changed)
                self._log("Ground truth signal connected for session replay")

            # 4. Initiate connection based on device architecture
            if device_type in (DeviceType.MUOVI, DeviceType.MUOVI_PLUS):
                self._log(f"Starting {device_text} Server on port 54321...")
                self._log("Action Required: Please turn on your Muovi device now.")

                # Update UI to 'Waiting' state
                self.connect_btn.setEnabled(False)
                self.connect_btn.setText("Waiting for Device...")
                self.disconnect_btn.setEnabled(True)  # Allow user to cancel the server

                # Start the TCP/IP Listening Server
                device.connect_device()

            else:
                # Synthetic or other local devices connect immediately
                if device.connect_device():
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

    @Slot(str)
    def _on_ground_truth_changed(self, label: str):
        """Handle ground truth label change during session replay."""
        # Store the current ground truth for when plot window opens
        self._current_ground_truth_label = label

        # Forward to plot window if open
        if self._plot_window and self._plot_window.isVisible():
            self._plot_window.set_ground_truth(label)

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
            self._update_prediction_buffer(data)

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
        # Format: YYYY-MM-DD_HH:MM:SS_Nrep
        timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        n_rep = protocol_config.repetitions_per_gesture
        session_id = f"{timestamp}_{n_rep}rep"

        # Determine actual number of channels after exclusion
        excluded = self._get_excluded_channels()
        actual_channels = device.num_channels - len(excluded)

        self._current_session = RecordingSession(session_id=session_id, subject_id=self.subject_id_edit.text(),
            device_name=device.config.device_type.name, num_channels=actual_channels,
            sampling_rate=device.sampling_rate, gesture_set=gesture_set, protocol_name=protocol_config.name)
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
        if not self.device_manager.device or not self.device_manager.device.is_streaming:
            QMessageBox.warning(self, "Warning", "Please connect a device and start streaming first.")
            return

        # Show calibration dialog
        from playagain_pipeline.gui.widgets.calibration_dialog import CalibrationDialog

        dialog = CalibrationDialog(
            calibrator=self.calibrator,
            device=self.device_manager.device,
            parent=self
        )

        if dialog.exec():
            # Get the calibration result
            result = dialog.get_calibration_result()
            if result:
                self.calibrator._current_calibration = result
                self._update_calibration_display()
                self._log(f"✓ Calibration completed - Rotation offset: {result.rotation_offset} channels, "
                         f"Confidence: {result.confidence:.2%}")

    @Slot()
    def _on_save_reference(self):
        """Save current calibration as reference."""
        if self.calibrator.current_calibration:
            self.calibrator.save_as_reference(self.calibrator.current_calibration)
            self._log("Saved calibration as reference")

    @Slot()
    def _on_load_calibration(self):
        """Load a calibration file."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Calibration", str(self.data_dir / "calibrations"),
            "JSON Files (*.json)")
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
        from PySide6.QtWidgets import (QDialog, QFormLayout, QSpinBox, QCheckBox,
                                       QDialogButtonBox, QListWidget, QListWidgetItem,
                                       QTabWidget, QVBoxLayout as QVBoxLayoutDialog)

        dialog = QDialog(self)
        dialog.setWindowTitle("Create Dataset from Sessions")
        dialog.setMinimumSize(600, 500)
        main_layout = QVBoxLayoutDialog(dialog)

        # Tabs for different selection modes
        tabs = QTabWidget()
        main_layout.addWidget(tabs)

        # Tab 1: Select by subjects (all their sessions)
        subject_tab = QWidget()
        subject_layout = QFormLayout(subject_tab)

        subject_list = QListWidget()
        subject_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        subjects = self.data_manager.list_subjects()
        for subject in subjects:
            item = QListWidgetItem(subject)
            item.setSelected(True)  # Select all by default
            subject_list.addItem(item)
        subject_layout.addRow("Subjects:", subject_list)
        tabs.addTab(subject_tab, "Select by Subject")

        # Tab 2: Select specific sessions
        session_tab = QWidget()
        session_layout = QFormLayout(session_tab)

        session_list = QListWidget()
        session_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)

        # Populate with all sessions from all subjects
        for subject in subjects:
            sessions = self.data_manager.list_sessions(subject)
            for session_id in sessions:
                item = QListWidgetItem(f"{subject} / {session_id}")
                item.setData(Qt.ItemDataRole.UserRole, (subject, session_id))
                session_list.addItem(item)

        session_layout.addRow("Sessions:", session_list)
        tabs.addTab(session_tab, "Select Specific Sessions")

        # Common parameters layout
        params_layout = QFormLayout()

        # Dataset name
        name_edit = QLineEdit("my_dataset")
        params_layout.addRow("Dataset Name:", name_edit)

        # Setup auto-naming logic
        state = {"manual_edit": False}

        def on_name_edited(text):
            state["manual_edit"] = True
        name_edit.textEdited.connect(on_name_edited)

        def update_dataset_name():
            if state["manual_edit"]:
                return

            new_name = "my_dataset"

            if tabs.currentIndex() == 0:  # Subject tab
                selected_items = subject_list.selectedItems()
                if not selected_items:
                    new_name = "dataset_no_subject"
                elif len(selected_items) == 1:
                    new_name = f"{selected_items[0].text()}"
                else:
                    new_name = f"{len(selected_items)}_subjects"
            else:  # Session tab
                selected_items = session_list.selectedItems()
                if not selected_items:
                    new_name = "dataset_no_session"
                elif len(selected_items) == 1:
                    # Item text is "subject / session_id", data is (subject, session_id)
                    _, session_id = selected_items[0].data(Qt.ItemDataRole.UserRole)
                    new_name = f"{session_id}"
                else:
                    # Try to create a meaningful name if few sessions
                    if len(selected_items) <= 3:
                        names = []
                        for item in selected_items:
                            _, session_id = item.data(Qt.ItemDataRole.UserRole)
                            names.append(session_id)
                        new_name = f"{'_'.join(names)}"
                        # Truncate if too long
                        if len(new_name) > 80:
                            new_name = f"{len(selected_items)}_sessions"
                    else:
                        new_name = f"{len(selected_items)}_sessions"

            if name_edit.text() != new_name:
                name_edit.setText(new_name)

        # Connect signals for auto-naming
        subject_list.itemSelectionChanged.connect(update_dataset_name)
        session_list.itemSelectionChanged.connect(update_dataset_name)
        tabs.currentChanged.connect(update_dataset_name)

        # Trigger initial update
        update_dataset_name()

        # Window parameters
        window_size_spin = QSpinBox()
        window_size_spin.setRange(50, 1000)
        window_size_spin.setValue(200)
        window_size_spin.setSuffix(" ms")
        params_layout.addRow("Window Size:", window_size_spin)

        stride_spin = QSpinBox()
        stride_spin.setRange(10, 500)
        stride_spin.setValue(50)
        stride_spin.setSuffix(" ms")
        params_layout.addRow("Stride:", stride_spin)

        # Include invalid trials
        include_invalid_cb = QCheckBox()
        include_invalid_cb.setChecked(False)
        params_layout.addRow("Include Invalid Trials:", include_invalid_cb)

        main_layout.addLayout(params_layout)

        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        main_layout.addWidget(buttons)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            try:
                sessions_to_use = []

                # Check which tab was active
                if tabs.currentIndex() == 0:
                    # Subject selection mode - get all sessions for selected subjects
                    selected_subjects = []
                    for i in range(subject_list.count()):
                        if subject_list.item(i).isSelected():
                            selected_subjects.append(subject_list.item(i).text())

                    if not selected_subjects:
                        QMessageBox.warning(self, "Warning", "Please select at least one subject")
                        return

                    # Load all sessions for selected subjects
                    for subject_id in selected_subjects:
                        session_ids = self.data_manager.list_sessions(subject_id)
                        for session_id in session_ids:
                            session = self.data_manager.load_session(subject_id, session_id)
                            sessions_to_use.append(session)

                    self._log(f"Using {len(sessions_to_use)} sessions from {len(selected_subjects)} subject(s)")

                else:
                    # Specific session selection mode
                    for i in range(session_list.count()):
                        if session_list.item(i).isSelected():
                            subject_id, session_id = session_list.item(i).data(Qt.ItemDataRole.UserRole)
                            session = self.data_manager.load_session(subject_id, session_id)
                            sessions_to_use.append(session)

                    if not sessions_to_use:
                        QMessageBox.warning(self, "Warning", "Please select at least one session")
                        return

                    self._log(f"Using {len(sessions_to_use)} specifically selected session(s)")

                # Create dataset with selected sessions
                dataset = self.data_manager.create_dataset(
                    name=name_edit.text(),
                    sessions=sessions_to_use,
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
            dataset_name = selected.text()

            # Log dataset loading
            self._log(f"Loading dataset '{dataset_name}'...")
            dataset = self.data_manager.load_dataset(dataset_name)

            num_samples = dataset['metadata']['num_samples']
            num_classes = dataset['metadata']['num_classes']
            num_channels = dataset['metadata']['num_channels']
            window_samples = dataset['metadata']['window_samples']

            self._log(f"✓ Dataset loaded: {num_samples} samples, {num_classes} classes, "
                     f"{num_channels} channels, {window_samples} samples/window")

            # Create model
            model_type = self.model_type_combo.currentText().lower().replace(" ", "_")
            self._log(f"Creating {model_type} model...")

            # Build model name: type_datasetname
            safe_dataset_name = dataset_name.replace(":", "-").replace(" ", "_")
            safe_dataset_name = safe_dataset_name.replace("/", "_").replace("\\\\", "_")
            model_name = f"{model_type}_{safe_dataset_name}"

            model = self.model_manager.create_model(model_type, name=model_name)
            self._log(f"Model created: {model.name}")

            # Data splitting information
            from sklearn.model_selection import train_test_split
            X = dataset["X"]
            y = dataset["y"]
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, stratify=y, random_state=42
            )

            self._log(f"Splitting data: Train={len(X_train)} samples, "
                     f"Validation={len(X_val)} samples")

            # Train
            self._log(f"Starting training ({model_type})...")
            results = self.model_manager.train_model(model, dataset)

            # Log detailed results
            self._log(f"Training complete!")
            self._log(f"  • Training accuracy: {results['training_accuracy']:.2%}")
            self._log(f"  • Validation accuracy: {results['validation_accuracy']:.2%}")

            if 'training_time' in results:
                self._log(f"  • Training time: {results['training_time']:.2f}s")

            if 'feature_count' in results:
                self._log(f"  • Features extracted: {results['feature_count']}")

            self._refresh_models()
            self._log(f"Model saved successfully")

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
            window_samples = int(self._prediction_window_ms * self._current_model.metadata.sampling_rate / 1000)
            self._prediction_buffer = np.zeros((window_samples, self._current_model.metadata.num_channels))

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

        # Create and start prediction worker thread
        self._prediction_worker = PredictionWorker(self)
        self._prediction_worker.set_model(self._current_model)
        self._prediction_worker.prediction_ready.connect(self._on_prediction_ready)
        self._prediction_worker.start()

        self._is_predicting = True
        self.start_pred_btn.setEnabled(False)
        self.stop_pred_btn.setEnabled(True)
        self._log("Started prediction")

    @Slot()
    def _on_stop_prediction(self):
        """Stop real-time prediction."""
        self._is_predicting = False

        if self._prediction_worker:
            self._prediction_worker.stop()
            self._prediction_worker = None

        self.start_pred_btn.setEnabled(True)
        self.stop_pred_btn.setEnabled(False)
        self.prediction_label.setText("No prediction")
        self.confidence_label.setText("Confidence: -")
        self._log("Stopped prediction")

    def _update_prediction_buffer(self, data: np.ndarray):
        """Update the prediction buffer with new data and pass to worker."""
        if self._prediction_buffer is None or self._current_model is None:
            return

        # Roll buffer and add new data
        n_samples = min(data.shape[0], len(self._prediction_buffer))
        self._prediction_buffer = np.roll(self._prediction_buffer, -n_samples, axis=0)
        self._prediction_buffer[-n_samples:] = data[:n_samples]

        # Send updated buffer to worker thread
        if self._prediction_worker:
            self._prediction_worker.update_buffer(self._prediction_buffer)

    @Slot(object, object)
    def _on_prediction_ready(self, pred, proba):
        """Handle prediction result from worker thread (updates UI)."""
        if not self._is_predicting or self._current_model is None:
            return

        try:
            class_names = self._current_model.metadata.class_names
            class_name = class_names.get(int(pred), class_names.get(str(pred), f"Class {pred}"))

            try:
                confidence = proba[int(pred)]
            except (IndexError, KeyError):
                confidence = np.max(proba)

            display_name = class_name.replace("_", " ").title()
            self.prediction_label.setText(display_name)
            self.confidence_label.setText(f"Confidence: {confidence:.1%}")
        except Exception:
            pass

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
            check = QCheckBox(f"Ch {i + 1}")
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
            if current_subject and current_subject in [self.session_subject_combo.itemText(i) for i in
                                                       range(self.session_subject_combo.count())]:
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
