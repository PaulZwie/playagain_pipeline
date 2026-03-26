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

# Resolve local packages (device_interfaces, gui_custom_elements) via automatic
# sibling-directory search so hard-coded Mac paths are no longer needed.
from playagain_pipeline.utils.platform_utils import inject_local_packages, print_platform_info
inject_local_packages()
print_platform_info()

import numpy as np
from PySide6.QtCore import Qt, Slot, QTimer, QThread, Signal, QMutex, QMutexLocker
from PySide6.QtGui import QFont, QAction
from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTabWidget, QPushButton, QComboBox,
                               QLineEdit, QLabel, QGroupBox, QFormLayout, QSpinBox, QDoubleSpinBox, QMessageBox, QFileDialog, QTextEdit,
                               QSplitter, QStatusBar, QListWidget, QScrollArea, QCheckBox,
                               QGridLayout, QApplication, QDialog, QListWidgetItem)

from playagain_pipeline.calibration.calibrator import AutoCalibrator
from playagain_pipeline.config.config import get_default_config, PipelineConfig
from playagain_pipeline.core.data_manager import DataManager
from playagain_pipeline.core.gesture import GestureSet, create_default_gesture_set, create_single_gesture_set
from playagain_pipeline.core.session import RecordingSession
from playagain_pipeline.devices.emg_device import (DeviceManager, DeviceType, SyntheticEMGDevice)
from playagain_pipeline.gui.widgets.emg_plot import EMGPlotWidget
from playagain_pipeline.gui.widgets.performance_tab import PerformanceReviewTab
from playagain_pipeline.gui.widgets.protocol_widget import ProtocolWidget
from playagain_pipeline.models.classifier import ModelManager, BaseClassifier, apply_bad_channel_strategy
from playagain_pipeline.protocols.protocol import (
    RecordingProtocol,
    ProtocolPhase,
    create_quick_protocol,
    create_standard_protocol,
    create_extended_protocol,
    create_pinch_protocol,
    create_tripod_protocol,
    create_fist_protocol,
)
from playagain_pipeline.prediction_server import PredictionServer, PredictionSmoother
from playagain_pipeline.game_recorder import GameRecorder


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
                except Exception as e:
                    # Log prediction errors (throttled to avoid spam)
                    if not hasattr(self, '_last_error') or str(e) != self._last_error:
                        self._last_error = str(e)
                        print(f"[PredictionWorker] Prediction error: {e}")

            self.msleep(100)  # Cap at ~10 predictions/sec

    def stop(self):
        self._running = False
        self.wait()


class ParticipantInfoDialog(QDialog):
    """Modal dialog used to collect participant information."""

    def __init__(self, subject_id: str, participant_info: Optional[dict] = None, parent=None):
        super().__init__(parent)
        self.subject_id = subject_id
        self.participant_info = participant_info or {}
        self.setWindowTitle(f"Participant Info - {subject_id}")
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        form = QFormLayout()
        self.subject_label = QLabel(self.subject_id)
        form.addRow("Subject ID:", self.subject_label)

        self.full_name_edit = QLineEdit(self.participant_info.get("full_name", ""))
        self.full_name_edit.setPlaceholderText("Optional full name")
        form.addRow("Full Name:", self.full_name_edit)

        self.age_spin = QSpinBox()
        self.age_spin.setRange(0, 120)
        self.age_spin.setValue(int(self.participant_info.get("age", 0) or 0))
        self.age_spin.setSpecialValueText("Unknown")
        form.addRow("Age:", self.age_spin)

        self.handedness_combo = QComboBox()
        self.handedness_combo.addItems(["Unknown", "Left", "Right", "Ambidextrous"])
        handedness = self.participant_info.get("handedness", "Unknown")
        self.handedness_combo.setCurrentText(handedness if handedness in {"Unknown", "Left", "Right", "Ambidextrous"} else "Unknown")
        form.addRow("Handedness:", self.handedness_combo)

        self.notes_edit = QTextEdit(self.participant_info.get("notes", ""))
        self.notes_edit.setPlaceholderText("Optional notes about the participant")
        self.notes_edit.setMinimumHeight(90)
        form.addRow("Notes:", self.notes_edit)

        layout.addLayout(form)

        button_row = QHBoxLayout()
        button_row.addStretch()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_row.addWidget(cancel_btn)
        save_btn = QPushButton("Save")
        save_btn.clicked.connect(self.accept)
        button_row.addWidget(save_btn)
        layout.addLayout(button_row)

    def get_participant_info(self) -> dict:
        """Return the participant data entered by the user."""
        info = {
            "subject_id": self.subject_id,
            "full_name": self.full_name_edit.text().strip(),
            "age": int(self.age_spin.value()) if self.age_spin.value() > 0 else None,
            "handedness": self.handedness_combo.currentText(),
            "notes": self.notes_edit.toPlainText().strip(),
        }
        return {k: v for k, v in info.items() if v not in (None, "")}


class MainWindow(QMainWindow):
    """
    Main application window for gesture recording and prediction.
    """

    # Thread-safe signal for prediction updates from the server's background thread
    _server_prediction_signal = Signal(str, float)  # display_name, confidence

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
        self._recording_mode: str = "preset"
        self._manual_active: bool = False

        # Prediction state
        self._prediction_buffer: Optional[np.ndarray] = None
        self._prediction_window_ms = 200
        self._is_predicting = False
        self._prediction_worker: Optional[PredictionWorker] = None

        # Unity prediction server
        self._prediction_server: Optional[PredictionServer] = None

        # Game recorder for capturing gameplay data (EMG + predictions + ground truth)
        self._game_recorder: Optional[GameRecorder] = None

        # GUI-side prediction smoother (for display only; server has its own)
        self._gui_smoother: Optional[PredictionSmoother] = None

        # Plot widget
        self._plot_widget: Optional[EMGPlotWidget] = None

        # Ground truth label for session replay
        self._current_ground_truth_label: Optional[str] = None

        # Live quick calibration state
        self._live_cal_active = False
        self._live_cal_gestures: list = []
        self._live_cal_current_idx = 0
        self._live_cal_buffer: list = []        # accumulated EMG chunks for current gesture
        self._live_cal_collected: dict = {}     # gesture -> np.ndarray
        self._live_cal_timer: Optional[QTimer] = None
        self._live_cal_countdown = 0
        self._live_cal_remaining = 0

        # Setup UI
        self._setup_ui()
        self._setup_connections()

        # Connect thread-safe server prediction signal to UI update
        self._server_prediction_signal.connect(self._apply_server_prediction)

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

        if hasattr(self, "bad_ch_mode_combo"):
            preferred_mode = getattr(self.config.model, "bad_channel_mode", "interpolate")
            idx = self.bad_ch_mode_combo.findData(preferred_mode)
            if idx >= 0:
                self.bad_ch_mode_combo.setCurrentIndex(idx)

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

        # Performance Review tab
        self._performance_tab = PerformanceReviewTab(self.data_manager)
        self.mode_tabs.addTab(self._performance_tab, "Performance Review")

        left_layout.addWidget(self.mode_tabs)

        # Log area
        log_group = QGroupBox("Log")
        log_layout = QVBoxLayout(log_group)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(200)
        self.log_text.setMaximumHeight(300)
        self.log_text.setStyleSheet(
            "QTextEdit { font-family: monospace; font-size: 11px; "
            "background-color: #1e1e1e; color: #d4d4d4; "
            "border: 1px solid #444; border-radius: 3px; padding: 4px; }"
        )
        log_layout.addWidget(self.log_text)
        left_layout.addWidget(log_group)

        splitter.addWidget(left_panel)

        # Right panel - Visualization
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # EMG Visualization Widget
        plot_group = QGroupBox("EMG Visualization")
        plot_layout = QVBoxLayout(plot_group)

        # Initialize with default settings, will be updated when device connects
        self._plot_widget = EMGPlotWidget(num_channels=32, sampling_rate=2000)
        self._plot_widget.bad_channels_updated.connect(self._on_bad_channels_updated)
        plot_layout.addWidget(self._plot_widget)

        right_layout.addWidget(plot_group, stretch=3)

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

        # Create scroll area to prevent cramping on small screens
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)

        # Session settings
        session_group = QGroupBox("Session Settings")
        session_layout = QFormLayout(session_group)

        self.subject_id_edit = QLineEdit()
        self.subject_id_edit.setPlaceholderText("e.g., VP_01")
        session_layout.addRow("Subject ID:", self.subject_id_edit)

        participant_row = QHBoxLayout()
        self.participant_info_btn = QPushButton("Participant Info...")
        self.participant_info_btn.clicked.connect(self._on_edit_participant_info)
        participant_row.addWidget(self.participant_info_btn)

        self.participant_info_status_label = QLabel("No participant info saved yet")
        self.participant_info_status_label.setStyleSheet("color: #666; font-size: 10px;")
        participant_row.addWidget(self.participant_info_status_label)
        participant_row.addStretch()
        session_layout.addRow(participant_row)

        self.subject_id_edit.textChanged.connect(self._update_participant_info_status)
        self._update_participant_info_status(self.subject_id_edit.text())

        self.session_notes_edit = QLineEdit()
        session_layout.addRow("Notes:", self.session_notes_edit)

        scroll_layout.addWidget(session_group)

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

        # Bipolar Mode checkbox
        self.bipolar_mode_cb = QCheckBox("Bipolar Mode (Top - Bottom)")
        self.bipolar_mode_cb.setChecked(False)
        self.bipolar_mode_cb.setToolTip(
            "Subtracts the bottom row electrodes from the top row electrodes.\n"
            "Will halve the number of channels automatically."
        )
        self.bipolar_mode_cb.toggled.connect(self._on_bipolar_mode_toggled)
        device_layout.addRow(self.bipolar_mode_cb)

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

        # Note: channel enable/disable is done via the checkboxes in the EMG plot
        channel_note = QLabel("Channel Status: Use checkboxes in EMG plot to enable/disable channels")
        channel_note.setStyleSheet("color: #666; font-size: 10px; font-style: italic;")
        channel_note.setWordWrap(True)
        device_layout.addRow(channel_note)

        # Bad channel handling mode
        self.bad_ch_mode_combo = QComboBox()
        self.bad_ch_mode_combo.addItem("Interpolate (neighbor average)", "interpolate")
        self.bad_ch_mode_combo.addItem("Zero out bad channels", "zero")
        self.bad_ch_mode_combo.setToolTip(
            "Choose how excluded channels are handled for recording,\n"
            "dataset creation, and live prediction preprocessing."
        )
        self.bad_ch_mode_combo.currentIndexChanged.connect(self._on_bad_channel_mode_changed)
        device_layout.addRow("Bad Ch. Handling:", self.bad_ch_mode_combo)

        # Internal state for excluded channels (updated by plot widget signal)
        self._excluded_channels: list[int] = []

        device_btn_layout = QHBoxLayout()
        self.connect_btn = QPushButton("Connect")
        self.connect_btn.setFixedHeight(36)
        self.connect_btn.clicked.connect(self._on_connect_device)
        device_btn_layout.addWidget(self.connect_btn)

        self.disconnect_btn = QPushButton("Disconnect")
        self.disconnect_btn.setFixedHeight(36)
        self.disconnect_btn.setEnabled(False)
        self.disconnect_btn.clicked.connect(self._on_disconnect_device)
        device_btn_layout.addWidget(self.disconnect_btn)
        device_layout.addRow(device_btn_layout)

        scroll_layout.addWidget(device_group)

        # Gesture set selection
        gesture_group = QGroupBox("Gestures")
        gesture_layout = QFormLayout(gesture_group)

        self.gesture_set_combo = QComboBox()
        self.gesture_set_combo.addItems(["Auto (from protocol)"])
        self.gesture_set_combo.setEnabled(False)
        gesture_layout.addRow("Gesture Set:", self.gesture_set_combo)

        scroll_layout.addWidget(gesture_group)

        # Protocol selection
        protocol_group = QGroupBox("Protocol")
        protocol_layout = QFormLayout(protocol_group)

        self.recording_mode_combo = QComboBox()
        self.recording_mode_combo.addItems([
            "Preset Protocol",
            "Custom Protocol",
            "Manual Toggle",
        ])
        self.recording_mode_combo.currentIndexChanged.connect(self._on_recording_mode_changed)
        protocol_layout.addRow("Mode:", self.recording_mode_combo)

        self.protocol_combo = QComboBox()
        self.protocol_combo.addItems([
            "Quick (3 reps)",
            "Standard (5 reps)",
            "Extended (10 reps)",
            "Pinch (single gesture)",
            "Tripod (single gesture)",
            "Fist (single gesture)",
        ])
        protocol_layout.addRow("Protocol:", self.protocol_combo)

        self.custom_settings_group = QGroupBox("Custom Settings")
        custom_settings_layout = QVBoxLayout(self.custom_settings_group)

        self.custom_gesture_group = QGroupBox("Custom Gestures")
        custom_gesture_layout = QVBoxLayout(self.custom_gesture_group)
        self.custom_calibration_cb = QCheckBox("Calibration sync once (waveout)")
        self.custom_calibration_cb.setChecked(True)
        self.custom_fist_cb = QCheckBox("Fist")
        self.custom_fist_cb.setChecked(True)
        self.custom_tripod_cb = QCheckBox("Tripod")
        self.custom_tripod_cb.setChecked(True)
        self.custom_pinch_cb = QCheckBox("Pinch")
        self.custom_pinch_cb.setChecked(True)
        custom_gesture_layout.addWidget(self.custom_calibration_cb)
        custom_gesture_layout.addWidget(self.custom_fist_cb)
        custom_gesture_layout.addWidget(self.custom_tripod_cb)
        custom_gesture_layout.addWidget(self.custom_pinch_cb)
        custom_settings_layout.addWidget(self.custom_gesture_group)

        custom_timing_layout = QFormLayout()

        self.custom_hold_spin = QDoubleSpinBox()
        self.custom_hold_spin.setRange(0.5, 30.0)
        self.custom_hold_spin.setValue(8.0)
        self.custom_hold_spin.setSingleStep(0.5)
        self.custom_hold_spin.setSuffix(" s")
        custom_timing_layout.addRow("Gesture duration:", self.custom_hold_spin)

        self.custom_rest_spin = QDoubleSpinBox()
        self.custom_rest_spin.setRange(0.5, 30.0)
        self.custom_rest_spin.setValue(5.0)
        self.custom_rest_spin.setSingleStep(0.5)
        self.custom_rest_spin.setSuffix(" s")
        custom_timing_layout.addRow("Rest duration:", self.custom_rest_spin)

        self.custom_reps_spin = QSpinBox()
        self.custom_reps_spin.setRange(1, 100)
        self.custom_reps_spin.setValue(5)
        custom_timing_layout.addRow("Repetitions:", self.custom_reps_spin)
        custom_settings_layout.addLayout(custom_timing_layout)
        protocol_layout.addRow(self.custom_settings_group)

        self.manual_settings_group = QGroupBox("Manual Labeling")
        manual_settings_layout = QFormLayout(self.manual_settings_group)

        self.manual_gesture_combo = QComboBox()
        self.manual_gesture_combo.addItems(["fist", "tripod", "pinch"])
        self.manual_gesture_combo.currentTextChanged.connect(self._on_manual_gesture_selection_changed)
        manual_settings_layout.addRow("Gesture:", self.manual_gesture_combo)

        self.manual_toggle_btn = QPushButton("Mark Gesture Active")
        self.manual_toggle_btn.setCheckable(True)
        self.manual_toggle_btn.setEnabled(False)
        self.manual_toggle_btn.toggled.connect(self._on_manual_toggle_changed)
        manual_settings_layout.addRow("Ground Truth Toggle:", self.manual_toggle_btn)
        protocol_layout.addRow(self.manual_settings_group)

        scroll_layout.addWidget(protocol_group)

        # Recording controls
        control_group = QGroupBox("Recording")
        control_layout = QHBoxLayout(control_group)

        self.start_recording_btn = QPushButton("Start Recording")
        self.start_recording_btn.setFixedHeight(36)
        self.start_recording_btn.clicked.connect(self._on_start_recording)
        control_layout.addWidget(self.start_recording_btn)

        self.stop_recording_btn = QPushButton("Stop Recording")
        self.stop_recording_btn.setFixedHeight(36)
        self.stop_recording_btn.setEnabled(False)
        self.stop_recording_btn.clicked.connect(self._on_stop_recording)
        control_layout.addWidget(self.stop_recording_btn)

        scroll_layout.addWidget(control_group)

        scroll_layout.addStretch()

        scroll.setWidget(scroll_content)
        layout.addWidget(scroll)

        self._on_recording_mode_changed()

        return tab

    def _on_recording_mode_changed(self):
        """Update protocol controls based on selected recording mode."""
        mode_text = self.recording_mode_combo.currentText() if hasattr(self, "recording_mode_combo") else "Preset Protocol"
        self._recording_mode = "preset"
        if mode_text.startswith("Custom"):
            self._recording_mode = "custom"
        elif mode_text.startswith("Manual"):
            self._recording_mode = "manual"

        is_preset = self._recording_mode == "preset"
        is_custom = self._recording_mode == "custom"
        is_manual = self._recording_mode == "manual"

        self.protocol_combo.setVisible(is_preset)
        self.custom_settings_group.setVisible(is_custom)
        self.manual_settings_group.setVisible(is_manual)
        self._update_manual_combo_enabled_state()

    def _build_manual_gesture_set(self) -> GestureSet:
        """Create a manual-mode gesture set that supports switching targets mid-session."""
        base_set = create_default_gesture_set()
        manual_set = GestureSet(name="manual")
        manual_set.add_gesture(base_set.get_gesture("rest"))
        for gesture_name in ["fist", "tripod", "pinch"]:
            manual_set.add_gesture(base_set.get_gesture(gesture_name))
        return manual_set

    def _update_manual_combo_enabled_state(self):
        """Allow manual target changes only while manual mode is in rest."""
        if not hasattr(self, "manual_gesture_combo"):
            return

        mode_text = self.recording_mode_combo.currentText() if hasattr(self, "recording_mode_combo") else ""
        is_manual_mode = self._recording_mode == "manual" or mode_text.startswith("Manual")

        if not is_manual_mode:
            self.manual_gesture_combo.setEnabled(True)
            return

        session_active = self._current_session is not None and self._current_session.is_recording
        if not session_active:
            self.manual_gesture_combo.setEnabled(True)
            return

        self.manual_gesture_combo.setEnabled(not self._manual_active)

    @Slot(str)
    def _on_manual_gesture_selection_changed(self, gesture_name: str):
        """Log manual target changes while recording so the operator sees the next target."""
        if self._recording_mode == "manual" and self._current_session and self._current_session.is_recording and not self._manual_active:
            self._log(f"Next manual gesture set to: {gesture_name.strip().lower()}")

    def _build_custom_gesture_set(self) -> Optional[GestureSet]:
        """Build a custom gesture set from selected checkboxes."""
        selected_names = []
        if self.custom_fist_cb.isChecked():
            selected_names.append("fist")
        if self.custom_tripod_cb.isChecked():
            selected_names.append("tripod")
        if self.custom_pinch_cb.isChecked():
            selected_names.append("pinch")

        if not selected_names:
            return None

        base_set = create_default_gesture_set()
        custom_set = GestureSet(name="custom")
        custom_set.add_gesture(base_set.get_gesture("rest"))
        for name in selected_names:
            custom_set.add_gesture(base_set.get_gesture(name))
        return custom_set

    def _build_custom_protocol_config(self):
        """Create protocol config from custom timing and repetition controls."""
        from playagain_pipeline.protocols.protocol import ProtocolConfig

        return ProtocolConfig(
            name="custom",
            description="User-defined custom protocol",
            hold_time=float(self.custom_hold_spin.value()),
            rest_time=float(self.custom_rest_spin.value()),
            repetitions_per_gesture=int(self.custom_reps_spin.value()),
            randomize_order=False,
            include_calibration_sync=self.custom_calibration_cb.isChecked(),
        )

    def _start_manual_rest_trial(self):
        """Ensure manual mode is in rest state when no gesture is toggled."""
        if self._current_session:
            self._current_session.start_trial("rest")

    @Slot(bool)
    def _on_manual_toggle_changed(self, is_active: bool):
        """Toggle manual ground-truth between selected gesture and rest."""
        if not self._current_session:
            return

        gesture_name = self.manual_gesture_combo.currentText().strip().lower()
        if not gesture_name:
            return

        self._current_session.end_trial()
        if is_active:
            self._manual_active = True
            self._current_session.start_trial(gesture_name)
            self.manual_toggle_btn.setText("Switch to Rest")
            self.manual_gesture_combo.setEnabled(False)
            self._log(f"Manual label active: {gesture_name}")
        else:
            self._manual_active = False
            self._start_manual_rest_trial()
            self.manual_toggle_btn.setText("Mark Gesture Active")
            self.manual_gesture_combo.setEnabled(True)
            self._log("Manual label active: rest")

        self._update_manual_combo_enabled_state()

    def _create_calibration_tab(self) -> QWidget:
        """Create the calibration tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Create scroll area to prevent cramping on small screens
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)

        # Information about calibration
        info_group = QGroupBox("About Calibration")
        info_layout = QVBoxLayout(info_group)

        info_label = QLabel(
            "Calibration determines the electrode bracelet orientation by analyzing\n"
            "EMG activation patterns from your recorded sessions.\n\n"
            "No separate calibration recording is needed — just select a\n"
            "recording session and calibrate from its gesture data.\n\n"
            "Save a calibration as reference, then future sessions can detect\n"
            "how much the bracelet has rotated since the reference."
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: #555; font-size: 11px;")
        info_layout.addWidget(info_label)
        scroll_layout.addWidget(info_group)

        # Calibration status
        status_group = QGroupBox("Calibration Status")
        status_layout = QFormLayout(status_group)

        self.cal_status_label = QLabel("No calibration loaded")
        status_layout.addRow("Status:", self.cal_status_label)

        self.cal_confidence_label = QLabel("-")
        status_layout.addRow("Confidence:", self.cal_confidence_label)

        self.cal_rotation_label = QLabel("-")
        status_layout.addRow("Rotation Offset:", self.cal_rotation_label)

        # Sub-score breakdown (populated from calibration metadata)
        self.cal_subscores_label = QLabel("-")
        self.cal_subscores_label.setStyleSheet("color: #555; font-size: 10px;")
        self.cal_subscores_label.setWordWrap(True)
        status_layout.addRow("Score breakdown:", self.cal_subscores_label)

        scroll_layout.addWidget(status_group)

        # Calibrate from session
        session_cal_group = QGroupBox("Calibrate from Recording Session")
        session_cal_layout = QVBoxLayout(session_cal_group)

        session_cal_info = QLabel(
            "Select a recorded session to extract calibration patterns.\n"
            "Multiple trials of each gesture are averaged for robustness."
        )
        session_cal_info.setStyleSheet("color: #666; font-size: 10px; font-style: italic;")
        session_cal_layout.addWidget(session_cal_info)

        session_select_layout = QFormLayout()
        self.cal_subject_combo = QComboBox()
        self.cal_subject_combo.currentTextChanged.connect(self._on_cal_subject_changed)
        session_select_layout.addRow("Subject:", self.cal_subject_combo)

        self.cal_session_combo = QComboBox()
        session_select_layout.addRow("Session:", self.cal_session_combo)
        session_cal_layout.addLayout(session_select_layout)

        self.calibrate_from_session_btn = QPushButton("Calibrate from Session")
        self.calibrate_from_session_btn.clicked.connect(self._on_calibrate_from_session)
        self.calibrate_from_session_btn.setFixedHeight(36)
        session_cal_layout.addWidget(self.calibrate_from_session_btn)

        # Refresh button to reload subjects/sessions
        self.cal_refresh_btn = QPushButton("Refresh Sessions")
        self.cal_refresh_btn.setFixedHeight(36)
        self.cal_refresh_btn.clicked.connect(self._refresh_cal_sessions)
        session_cal_layout.addWidget(self.cal_refresh_btn)

        scroll_layout.addWidget(session_cal_group)

        # ── Live Quick Calibration (pretrained model workflow) ─────────────────
        live_cal_group = QGroupBox("Live Quick Calibration (for pretrained models)")
        live_cal_layout = QVBoxLayout(live_cal_group)

        live_cal_info = QLabel(
            "Detect bracelet rotation live without recording a session.\n"
            "Hold each gesture for ~3 seconds when prompted.\n"
            "Requires: device connected and streaming.\n\n"
            "Optimal workflow:\n"
            "  Connect Muovi → Live Calibrate → Load Model → Predict"
        )
        live_cal_info.setStyleSheet("color: #555; font-size: 10px;")
        live_cal_info.setWordWrap(True)
        live_cal_layout.addWidget(live_cal_info)

        # Gesture list for live calibration
        live_cal_gesture_layout = QFormLayout()
        self.live_cal_gestures_edit = QLineEdit("open_hand, fist, index_point, rest")
        self.live_cal_gestures_edit.setToolTip(
            "Comma-separated list of gesture names to use for live calibration.\n"
            "Must match the gestures the reference calibration was built from."
        )
        live_cal_gesture_layout.addRow("Gestures:", self.live_cal_gestures_edit)

        self.live_cal_duration_spin = QSpinBox()
        self.live_cal_duration_spin.setRange(1, 10)
        self.live_cal_duration_spin.setValue(3)
        self.live_cal_duration_spin.setSuffix(" s")
        self.live_cal_duration_spin.setToolTip("How long to hold each gesture")
        live_cal_gesture_layout.addRow("Hold duration:", self.live_cal_duration_spin)
        live_cal_layout.addLayout(live_cal_gesture_layout)

        # Progress label and start button
        self.live_cal_status_label = QLabel("Ready")
        self.live_cal_status_label.setStyleSheet("font-weight: bold; color: #333;")
        live_cal_layout.addWidget(self.live_cal_status_label)

        live_cal_btn_row = QHBoxLayout()
        self.start_live_cal_btn = QPushButton("▶  Start Live Calibration")
        self.start_live_cal_btn.setFixedHeight(36)
        self.start_live_cal_btn.clicked.connect(self._on_start_live_calibration)
        live_cal_btn_row.addWidget(self.start_live_cal_btn)

        self.cancel_live_cal_btn = QPushButton("✕  Cancel")
        self.cancel_live_cal_btn.setFixedHeight(36)
        self.cancel_live_cal_btn.setEnabled(False)
        self.cancel_live_cal_btn.clicked.connect(self._on_cancel_live_calibration)
        live_cal_btn_row.addWidget(self.cancel_live_cal_btn)
        live_cal_layout.addLayout(live_cal_btn_row)

        scroll_layout.addWidget(live_cal_group)
        control_group = QGroupBox("Calibration Actions")
        control_layout = QVBoxLayout(control_group)

        self.save_ref_btn = QPushButton("Save as Reference")
        self.save_ref_btn.setFixedHeight(36)
        self.save_ref_btn.setEnabled(False)
        self.save_ref_btn.clicked.connect(self._on_save_reference)
        control_layout.addWidget(self.save_ref_btn)

        self.save_ref_recompute_btn = QPushButton("Set as Reference && Recompute All Rotations")
        self.save_ref_recompute_btn.setFixedHeight(36)
        self.save_ref_recompute_btn.setEnabled(False)
        self.save_ref_recompute_btn.setToolTip(
            "Save this calibration as the new reference and re-detect\n"
            "bracelet rotation for ALL existing sessions relative to it."
        )
        self.save_ref_recompute_btn.clicked.connect(self._on_save_reference_and_recompute)
        control_layout.addWidget(self.save_ref_recompute_btn)

        self.load_cal_btn = QPushButton("Load Calibration...")
        self.load_cal_btn.setFixedHeight(36)
        self.load_cal_btn.clicked.connect(self._on_load_calibration)
        control_layout.addWidget(self.load_cal_btn)

        scroll_layout.addWidget(control_group)

        # Manual rotation override (use pretrained model without recording)
        manual_group = QGroupBox("Manual Rotation Override")
        manual_layout = QVBoxLayout(manual_group)

        manual_info = QLabel(
            "Set a rotation offset manually when using a pretrained model\n"
            "without recording. Set to 0 for no rotation correction.\n"
            "Recording sessions will still auto-detect rotation."
        )
        manual_info.setWordWrap(True)
        manual_info.setStyleSheet("color: #555; font-size: 10px;")
        manual_layout.addWidget(manual_info)

        manual_form = QFormLayout()
        self.manual_rotation_spin = QSpinBox()
        self.manual_rotation_spin.setRange(-16, 16)
        self.manual_rotation_spin.setValue(0)
        self.manual_rotation_spin.setSuffix(" ch")
        manual_form.addRow("Rotation Offset:", self.manual_rotation_spin)
        manual_layout.addLayout(manual_form)

        self.apply_manual_rot_btn = QPushButton("Apply Manual Rotation")
        self.apply_manual_rot_btn.setFixedHeight(36)
        self.apply_manual_rot_btn.clicked.connect(self._on_apply_manual_rotation)
        manual_layout.addWidget(self.apply_manual_rot_btn)

        self.clear_cal_btn = QPushButton("Clear Calibration (No Rotation)")
        self.clear_cal_btn.setFixedHeight(36)
        self.clear_cal_btn.clicked.connect(self._on_clear_calibration)
        manual_layout.addWidget(self.clear_cal_btn)

        scroll_layout.addWidget(manual_group)

        # Bracelet orientation graphic
        from playagain_pipeline.gui.widgets.bracelet_graphic import BraceletGraphicWidget
        bracelet_group = QGroupBox("Bracelet Orientation")
        bracelet_layout = QVBoxLayout(bracelet_group)
        self.bracelet_graphic = BraceletGraphicWidget(
            num_electrodes=self.channels_spin.value(),
            rotation_offset=0,
            signal_mode=("bipolar" if self.bipolar_mode_cb.isChecked() else "monopolar"),
        )
        # Set minimum height for bracelet graphic to ensure it doesn't get squashed
        self.bracelet_graphic.setMinimumHeight(200)
        bracelet_layout.addWidget(self.bracelet_graphic)
        scroll_layout.addWidget(bracelet_group)

        scroll_layout.addStretch()
        scroll.setWidget(scroll_content)
        layout.addWidget(scroll)

        # Populate session combos on first show
        QTimer.singleShot(100, self._refresh_cal_sessions)

        return tab

    def _create_training_tab(self) -> QWidget:
        """Create the model training tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Dataset selection
        dataset_group = QGroupBox("Dataset")
        dataset_layout = QVBoxLayout(dataset_group)

        self.refresh_datasets_btn = QPushButton("Refresh")
        self.refresh_datasets_btn.setFixedHeight(36)
        self.refresh_datasets_btn.clicked.connect(self._refresh_datasets)
        dataset_layout.addWidget(self.refresh_datasets_btn)

        self.dataset_list = QListWidget()
        self.dataset_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        dataset_layout.addWidget(self.dataset_list)

        self.create_dataset_btn = QPushButton("Create Dataset from Sessions...")
        self.create_dataset_btn.setFixedHeight(36)
        self.create_dataset_btn.clicked.connect(self._on_create_dataset)
        dataset_layout.addWidget(self.create_dataset_btn)

        self.delete_dataset_btn = QPushButton("Delete Selected Dataset(s)")
        self.delete_dataset_btn.setFixedHeight(36)
        self.delete_dataset_btn.clicked.connect(self._on_delete_dataset)
        self.delete_dataset_btn.setStyleSheet("QPushButton { color: red; }")
        dataset_layout.addWidget(self.delete_dataset_btn)

        layout.addWidget(dataset_group)

        # Model selection
        model_group = QGroupBox("Model")
        model_layout = QFormLayout(model_group)

        self.model_type_combo = QComboBox()
        self.model_type_combo.addItems(["SVM", "Random Forest", "LDA", "CatBoost", "MLP", "CNN", "AttentionNet", "MSTNet"])
        model_layout.addRow("Model Type:", self.model_type_combo)

        train_btn_layout = QHBoxLayout()
        self.train_btn = QPushButton("Train Model")
        self.train_btn.setFixedHeight(36)
        self.train_btn.clicked.connect(self._on_train_model)
        train_btn_layout.addWidget(self.train_btn)
        model_layout.addRow(train_btn_layout)

        layout.addWidget(model_group)

        # Trained models
        models_group = QGroupBox("Trained Models")
        models_layout = QVBoxLayout(models_group)

        self.models_list = QListWidget()
        self.models_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        models_layout.addWidget(self.models_list)

        self.refresh_models_btn = QPushButton("Refresh")
        self.refresh_models_btn.setFixedHeight(36)
        self.refresh_models_btn.clicked.connect(self._refresh_models)
        models_layout.addWidget(self.refresh_models_btn)

        self.delete_model_btn = QPushButton("Delete Selected Model(s)")
        self.delete_model_btn.setFixedHeight(36)
        self.delete_model_btn.clicked.connect(self._on_delete_model)
        self.delete_model_btn.setStyleSheet("QPushButton { color: red; }")
        models_layout.addWidget(self.delete_model_btn)

        layout.addWidget(models_group)

        layout.addStretch()
        return tab

    def _create_prediction_tab(self) -> QWidget:
        """Create the real-time prediction tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Wrap everything in a scroll area so it fits
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)

        # Model selection
        model_group = QGroupBox("Model")
        model_layout = QFormLayout(model_group)

        self.pred_model_combo = QComboBox()
        model_layout.addRow("Model:", self.pred_model_combo)

        self.load_model_btn = QPushButton("Load Model")
        self.load_model_btn.setFixedHeight(36)
        self.load_model_btn.clicked.connect(self._on_load_model)
        model_layout.addRow(self.load_model_btn)



        # Pretrained model workflow guide
        pretrained_info = QLabel(
            "Pretrained model workflow:\n"
            "  1. Connect device (Recording tab)\n"
            "  2. Calibrate or set manual rotation (Calibration tab)\n"
            "  3. Load a pretrained model here\n"
            "  4. Start Prediction — no recording needed"
        )
        pretrained_info.setStyleSheet("color: #555; font-size: 10px; background: #f5f5f5; "
                                      "padding: 4px; border-radius: 3px;")
        pretrained_info.setWordWrap(True)
        model_layout.addRow(pretrained_info)

        scroll_layout.addWidget(model_group)

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

        scroll_layout.addWidget(pred_group)

        # Prediction controls
        control_group = QGroupBox("Controls")
        control_layout = QHBoxLayout(control_group)

        self.start_pred_btn = QPushButton("Start Prediction")
        self.start_pred_btn.setFixedHeight(36)
        self.start_pred_btn.clicked.connect(self._on_start_prediction)
        control_layout.addWidget(self.start_pred_btn)

        self.stop_pred_btn = QPushButton("Stop Prediction")
        self.stop_pred_btn.setFixedHeight(36)
        self.stop_pred_btn.setEnabled(False)
        self.stop_pred_btn.clicked.connect(self._on_stop_prediction)
        control_layout.addWidget(self.stop_pred_btn)

        scroll_layout.addWidget(control_group)

        # ── Prediction Smoothing ──────────────────────────────────────────
        smoothing_group = QGroupBox("Prediction Smoothing")
        smoothing_layout = QFormLayout(smoothing_group)

        self.smoothing_enabled_cb = QCheckBox("Enable Smoothing")
        self.smoothing_enabled_cb.setChecked(True)
        self.smoothing_enabled_cb.toggled.connect(self._on_smoothing_toggled)
        smoothing_layout.addRow(self.smoothing_enabled_cb)

        self.smoothing_alpha_spin = QSpinBox()
        self.smoothing_alpha_spin.setRange(5, 100)
        self.smoothing_alpha_spin.setValue(30)
        self.smoothing_alpha_spin.setSuffix("%")
        self.smoothing_alpha_spin.setToolTip(
            "EMA weight for new predictions (lower = smoother, slower response)")
        self.smoothing_alpha_spin.valueChanged.connect(self._on_smoothing_params_changed)
        smoothing_layout.addRow("Alpha:", self.smoothing_alpha_spin)

        self.smoothing_stability_spin = QSpinBox()
        self.smoothing_stability_spin.setRange(0, 2000)
        self.smoothing_stability_spin.setValue(300)
        self.smoothing_stability_spin.setSuffix(" ms")
        self.smoothing_stability_spin.setToolTip(
            "Minimum time a new gesture must be predicted before switching (Grace Time)")
        self.smoothing_stability_spin.valueChanged.connect(self._on_smoothing_params_changed)
        smoothing_layout.addRow("Stability Window:", self.smoothing_stability_spin)

        smoothing_info = QLabel(
            "Smoothing prevents brief misclassifications from affecting\n"
            "the game. Increase stability for more robust predictions.")
        smoothing_info.setStyleSheet("color: gray; font-size: 10px;")
        smoothing_layout.addRow(smoothing_info)

        scroll_layout.addWidget(smoothing_group)

        # ── Unity TCP Server ──────────────────────────────────────────────
        server_group = QGroupBox("Unity TCP Server")
        server_layout = QVBoxLayout(server_group)

        server_config_layout = QHBoxLayout()
        server_config_layout.addWidget(QLabel("Host:"))
        self.server_host_edit = QLineEdit("127.0.0.1")
        self.server_host_edit.setMaximumWidth(120)
        server_config_layout.addWidget(self.server_host_edit)
        server_config_layout.addWidget(QLabel("Port:"))
        self.server_port_spin = QSpinBox()
        self.server_port_spin.setRange(1024, 65535)
        self.server_port_spin.setValue(5555)
        server_config_layout.addWidget(self.server_port_spin)
        server_layout.addLayout(server_config_layout)

        server_btn_layout = QHBoxLayout()
        self.start_server_btn = QPushButton("Start Server")
        self.start_server_btn.setFixedHeight(36)
        self.start_server_btn.clicked.connect(self._on_start_server)
        server_btn_layout.addWidget(self.start_server_btn)

        self.stop_server_btn = QPushButton("Stop Server")
        self.stop_server_btn.setFixedHeight(36)
        self.stop_server_btn.setEnabled(False)
        self.stop_server_btn.clicked.connect(self._on_stop_server)
        server_btn_layout.addWidget(self.stop_server_btn)
        server_layout.addLayout(server_btn_layout)

        self.server_status_label = QLabel("Server: Not running")
        server_layout.addWidget(self.server_status_label)

        scroll_layout.addWidget(server_group)

        # ── Game Recording ────────────────────────────────────────────────
        game_rec_group = QGroupBox("Game Recording")
        game_rec_layout = QVBoxLayout(game_rec_group)

        game_rec_info = QLabel(
            "Record EMG data, model predictions, and game ground truth\n"
            "(requested gesture, camera state) to CSV during gameplay.\n"
            "Requires Unity TCP Server to be running.")
        game_rec_info.setStyleSheet("color: #555; font-size: 10px;")
        game_rec_info.setWordWrap(True)
        game_rec_layout.addWidget(game_rec_info)

        game_rec_config = QFormLayout()
        self.game_rec_subject_edit = QLineEdit()
        self.game_rec_subject_edit.setPlaceholderText("e.g., VP_01")
        game_rec_config.addRow("Subject ID:", self.game_rec_subject_edit)

        self.game_rec_session_edit = QLineEdit()
        self.game_rec_session_edit.setPlaceholderText("Optional session name")
        game_rec_config.addRow("Session Name:", self.game_rec_session_edit)
        game_rec_layout.addLayout(game_rec_config)

        game_rec_btn_layout = QHBoxLayout()
        self.start_game_rec_btn = QPushButton("Start Game Recording")
        self.start_game_rec_btn.setFixedHeight(36)
        self.start_game_rec_btn.clicked.connect(self._on_start_game_recording)
        game_rec_btn_layout.addWidget(self.start_game_rec_btn)

        self.stop_game_rec_btn = QPushButton("Stop Game Recording")
        self.stop_game_rec_btn.setFixedHeight(36)
        self.stop_game_rec_btn.setEnabled(False)
        self.stop_game_rec_btn.clicked.connect(self._on_stop_game_recording)
        game_rec_btn_layout.addWidget(self.stop_game_rec_btn)
        game_rec_layout.addLayout(game_rec_btn_layout)

        self.game_rec_status_label = QLabel("Game Recording: Idle")
        self.game_rec_status_label.setStyleSheet("font-weight: bold;")
        game_rec_layout.addWidget(self.game_rec_status_label)

        scroll_layout.addWidget(game_rec_group)

        scroll_layout.addStretch()
        scroll.setWidget(scroll_content)
        layout.addWidget(scroll)
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

    def _update_participant_info_status(self, subject_id: Optional[str] = None):
        """Update the status label for the currently selected participant."""
        if not hasattr(self, "participant_info_status_label"):
            return

        subject = (subject_id or self.subject_id_edit.text()).strip()
        if not subject:
            self.participant_info_status_label.setText("No participant selected")
            return

        if self.data_manager.has_participant_info(subject):
            self.participant_info_status_label.setText(f"Info saved for {subject}")
        else:
            self.participant_info_status_label.setText(f"No info saved for {subject}")

    def _prompt_participant_info(self, subject_id: str, existing_info: Optional[dict] = None) -> Optional[dict]:
        """Open the participant info dialog and return the captured data."""
        dialog = ParticipantInfoDialog(subject_id, existing_info, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return None
        return dialog.get_participant_info()

    def _ensure_participant_info(self, subject_id: str, allow_create: bool = True) -> Optional[dict]:
        """Load existing participant info or prompt the user to create it."""
        subject_id = subject_id.strip()
        if not subject_id:
            return None

        existing = self.data_manager.load_participant_info(subject_id)
        if existing is not None:
            participant = existing.get("participant", existing)
            self._update_participant_info_status(subject_id)
            return participant

        if not allow_create:
            return None

        participant = self._prompt_participant_info(subject_id)
        if participant is None:
            return None

        self.data_manager.save_participant_info(subject_id, participant)
        self._update_participant_info_status(subject_id)
        return participant

    @Slot()
    def _on_edit_participant_info(self):
        """Open the participant info dialog for the current subject."""
        subject_id = self.subject_id_edit.text().strip()
        if not subject_id:
            subject_id = self.data_manager.get_next_subject_id()
            self.subject_id_edit.setText(subject_id)

        existing = self.data_manager.load_participant_info(subject_id)
        existing_participant = existing.get("participant", existing) if existing else None
        participant = self._prompt_participant_info(subject_id, existing_participant)
        if participant is None:
            return

        self.data_manager.save_participant_info(subject_id, participant)
        self._update_participant_info_status(subject_id)
        self._log(f"Saved participant info for {subject_id}")

    def _log(self, message: str):
        """Add message to log with color coding."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        # Color-code by content keywords for quick visual scanning
        if any(k in message.lower() for k in ("error", "failed", "critical")):
            color = "#f48771"   # red-ish
        elif any(k in message.lower() for k in ("warning", "mismatch", "skipped")):
            color = "#e5c07b"   # amber
        elif any(k in message.lower() for k in ("complete", "success", "loaded", "saved", "started", "connected")):
            color = "#98c379"   # green
        else:
            color = "#d4d4d4"   # default light grey
        ts_html = f'<span style="color:#858585;">[{timestamp}]</span>'
        msg_html = f'<span style="color:{color};">{message}</span>'
        self.log_text.append(f"{ts_html} {msg_html}")





    def closeEvent(self, event):
        """Handle window close."""
        # Stop game recording if active
        if self._game_recorder and self._game_recorder.is_recording:
            self._game_recorder.stop_recording()
            self._game_recorder = None

        # Stop prediction server if running
        if self._prediction_server:
            self._prediction_server.stop()
            self._prediction_server = None

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
            available_models = ["SVM", "CatBoost", "Random Forest", "LDA", "MLP", "CNN", "AttentionNet", "MSTNet"]

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
        from playagain_pipeline.gui.widgets.bracelet_graphic import BraceletGraphicWidget

        # Create a simple dialog container
        dialog = QDialog(self)
        dialog.setWindowTitle("Bracelet Visualization")
        dialog.setMinimumSize(450, 450)

        layout = QVBoxLayout(dialog)

        viz = BraceletGraphicWidget(
            num_electrodes=self.channels_spin.value(),
            rotation_offset=(
                self.calibrator.current_calibration.rotation_offset
                if self.calibrator and self.calibrator.current_calibration
                else 0
            ),
        )

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
            kwargs = {
                "num_channels": self.channels_spin.value(), 
                "sampling_rate": self.sampling_rate_spin.value(),
                "bipolar_mode": self.bipolar_mode_cb.isChecked()
            }

            # Add session replay parameters for synthetic device
            if device_type == DeviceType.SYNTHETIC and self.use_session_data_cb.isChecked():
                kwargs.update({"use_session_data": True, "session_subject_id": self.session_subject_combo.currentText(),
                    "session_id": self.session_id_combo.currentText(), "data_dir": str(self.data_dir)})

            device = self.device_manager.create_device(device_type, **kwargs)

            # Update calibrator to use the actual device channel count (adaptive)
            # This fixes the issue when bad channels are removed and fewer channels are available
            self.calibrator.processor.num_channels = device.num_channels
            self._on_bipolar_mode_toggled(self.bipolar_mode_cb.isChecked())

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

        # Forward to plot widget
        if self._plot_widget and self._plot_widget.isVisible():
            self._plot_widget.set_ground_truth(label)

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
        bad_channels = self._get_excluded_channels()

        # Always show ALL channels in the plot (including bad ones) so the
        # checkbox state is not reset by reconfiguring the plot widget.
        if self._plot_widget and self._plot_widget.isVisible():
            if data.shape[1] != self._plot_widget.num_channels:
                old_ch = self._plot_widget.num_channels
                self._plot_widget.set_num_channels(data.shape[1])
                self._log(f"Plot display reconfigured: {old_ch} -> {data.shape[1]} channels")
            self._plot_widget.update_data(data)

        downstream_data = apply_bad_channel_strategy(
            data,
            bad_channels,
            mode=self._get_bad_channel_mode(),
        )

        # Apply calibration channel reordering for prediction/server
        # (not for raw plot or session recording — those stay in physical order)
        calibrated_data = downstream_data
        if self.calibrator.current_calibration is not None:
            try:
                calibrated_data = self.calibrator.current_calibration.apply_to_data(downstream_data)
            except ValueError:
                calibrated_data = downstream_data  # Channel mismatch — skip

        # Record if session is active (bad-channel strategy already applied)
        if self._current_session and self._current_session.is_recording:
            self._current_session.add_data(downstream_data)

        # Feed calibrated data to Unity TCP server if running.
        # The server now runs prediction on its own background thread,
        # so this call returns immediately without blocking the GUI.
        server_active = (self._prediction_server is not None
                         and self._prediction_server.is_running)
        if server_active:
            self._prediction_server.on_emg_data(calibrated_data)

        # Only use the separate PredictionWorker when the server is NOT running
        # to avoid running inference twice on the same data.
        if self._is_predicting and self._current_model and not server_active:
            self._update_prediction_buffer(calibrated_data)

        # Feed raw data to game recorder (records physical channels)
        if self._game_recorder and self._game_recorder.is_recording:
            self._game_recorder.on_emg_data(data)

        # Accumulate raw data for live calibration (uses physical channel order)
        if self._live_cal_active:
            self._live_cal_buffer.append(data.copy())

    # Recording handlers
    @Slot()
    def _on_start_recording(self):
        """Start a new recording session."""
        device = self.device_manager.device
        if not device or not device.is_connected:
            QMessageBox.warning(self, "Warning", "Please connect a device first")
            return

        subject_id = self.subject_id_edit.text().strip()
        if not subject_id:
            subject_id = self.data_manager.get_next_subject_id()
            self.subject_id_edit.setText(subject_id)

        participant_info = self._ensure_participant_info(subject_id, allow_create=True)
        if participant_info is None and not self.data_manager.has_participant_info(subject_id):
            QMessageBox.warning(self, "Warning", "Participant info is required before starting a new participant.")
            return

        protocol_config = None
        self._current_protocol = None

        if self._recording_mode == "preset":
            selected_protocol = self.protocol_combo.currentText()
            if selected_protocol.startswith("Quick"):
                gesture_set = create_default_gesture_set()
                protocol_config = create_quick_protocol()
            elif selected_protocol.startswith("Standard"):
                gesture_set = create_default_gesture_set()
                protocol_config = create_standard_protocol()
            elif selected_protocol.startswith("Extended"):
                gesture_set = create_default_gesture_set()
                protocol_config = create_extended_protocol()
            elif selected_protocol.startswith("Pinch"):
                gesture_set = create_single_gesture_set("pinch")
                protocol_config = create_pinch_protocol()
            elif selected_protocol.startswith("Tripod"):
                gesture_set = create_single_gesture_set("tripod")
                protocol_config = create_tripod_protocol()
            elif selected_protocol.startswith("Fist"):
                gesture_set = create_single_gesture_set("fist")
                protocol_config = create_fist_protocol()
            else:
                QMessageBox.warning(self, "Warning", f"Unknown protocol selected: {selected_protocol}")
                return
            self._current_protocol = RecordingProtocol(gesture_set, protocol_config)
        elif self._recording_mode == "custom":
            gesture_set = self._build_custom_gesture_set()
            if gesture_set is None:
                QMessageBox.warning(self, "Warning", "Please select at least one active gesture for custom protocol.")
                return
            protocol_config = self._build_custom_protocol_config()
            self._current_protocol = RecordingProtocol(gesture_set, protocol_config)
        else:
            selected_gesture = self.manual_gesture_combo.currentText().strip().lower()
            if not selected_gesture:
                QMessageBox.warning(self, "Warning", "Please select a manual gesture.")
                return
            gesture_set = self._build_manual_gesture_set()

        # Create session
        # Format: YYYY-MM-DD_HH-MM-SS_Nrep (Windows-safe)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if protocol_config is not None:
            n_rep = protocol_config.repetitions_per_gesture
            session_id = f"{timestamp}_{n_rep}rep"
        else:
            session_id = f"{timestamp}_manual"

        # Determine actual number of channels (all channels kept; strategy applied in stream path)
        bad_channels = self._get_excluded_channels()
        actual_channels = device.num_channels

        self._current_session = RecordingSession(session_id=session_id, subject_id=subject_id,
            device_name=device.config.device_type.name, num_channels=actual_channels,
            sampling_rate=device.sampling_rate, gesture_set=gesture_set,
            protocol_name=(protocol_config.name if protocol_config else "manual"))
        self._current_session.metadata.notes = self.session_notes_edit.text()
        # Store bad channels in session metadata from the start
        self._current_session.metadata.bad_channels = bad_channels
        self._current_session.metadata.custom_metadata["bad_channel_mode"] = self._get_bad_channel_mode()
        signal_mode = "bipolar" if self.bipolar_mode_cb.isChecked() else "monopolar"
        self._current_session.metadata.custom_metadata["bipolar_mode"] = (signal_mode == "bipolar")
        self._current_session.metadata.custom_metadata["signal_mode"] = signal_mode
        self._current_session.metadata.custom_metadata["recording_mode"] = self._recording_mode
        if participant_info:
            self._current_session.metadata.custom_metadata["participant_info"] = participant_info

        if self._recording_mode == "custom":
            self._current_session.metadata.custom_metadata["custom_protocol"] = {
                "include_calibration_sync": self.custom_calibration_cb.isChecked(),
                "gestures": [
                    name for name, selected in {
                        "fist": self.custom_fist_cb.isChecked(),
                        "tripod": self.custom_tripod_cb.isChecked(),
                        "pinch": self.custom_pinch_cb.isChecked(),
                    }.items() if selected
                ],
                "hold_time": float(self.custom_hold_spin.value()),
                "rest_time": float(self.custom_rest_spin.value()),
                "repetitions": int(self.custom_reps_spin.value()),
            }
        if self._recording_mode == "manual":
            self._current_session.metadata.custom_metadata["manual_gesture"] = self.manual_gesture_combo.currentText().strip().lower()

        # Setup protocol widget
        if self._current_protocol is not None:
            self.protocol_widget.set_protocol(self._current_protocol)

        # Start recording
        self._current_session.start_recording()
        if self._current_protocol is not None:
            self.protocol_widget.start()
        else:
            self.protocol_widget.stop()
            self._manual_active = False
            self.manual_toggle_btn.blockSignals(True)
            self.manual_toggle_btn.setChecked(False)
            self.manual_toggle_btn.setText("Mark Gesture Active")
            self.manual_toggle_btn.blockSignals(False)
            self.manual_toggle_btn.setEnabled(True)
            self.manual_gesture_combo.setEnabled(True)
            self._start_manual_rest_trial()
            self._log("Manual recording started. Use the toggle button to mark gesture vs. rest.")

        self.start_recording_btn.setEnabled(False)
        self.stop_recording_btn.setEnabled(True)
        self.recording_mode_combo.setEnabled(False)
        self.protocol_combo.setEnabled(False)
        self.custom_settings_group.setEnabled(False)
        self.manual_settings_group.setEnabled(True)
        self._update_manual_combo_enabled_state()
        self._log(f"Started recording session: {session_id}")

    @Slot()
    def _on_stop_recording(self):
        """Stop the current recording."""
        if self._current_session:
            self._current_session.stop_recording()
            self.protocol_widget.stop()
            self._current_protocol = None
            self._manual_active = False
            if hasattr(self, "manual_toggle_btn"):
                self.manual_toggle_btn.blockSignals(True)
                self.manual_toggle_btn.setChecked(False)
                self.manual_toggle_btn.setText("Mark Gesture Active")
                self.manual_toggle_btn.blockSignals(False)
                self.manual_toggle_btn.setEnabled(False)

            # Store bad channels in session metadata
            bad_channels = self._get_excluded_channels()
            if bad_channels:
                self._current_session.metadata.bad_channels = bad_channels
                self._log(f"Marked {len(bad_channels)} bad channels: {bad_channels}")
            self._current_session.metadata.custom_metadata["bad_channel_mode"] = self._get_bad_channel_mode()
            signal_mode = "bipolar" if self.bipolar_mode_cb.isChecked() else "monopolar"
            self._current_session.metadata.custom_metadata["bipolar_mode"] = (signal_mode == "bipolar")
            self._current_session.metadata.custom_metadata["signal_mode"] = signal_mode

            # Auto-detect bracelet rotation before saving
            try:
                cal_result = self.calibrator.detect_session_rotation(
                    self._current_session, save_to_metadata=True
                )
                if cal_result is not None:
                    rot = cal_result.rotation_offset
                    conf = cal_result.confidence
                    self._log(f"Auto-detected rotation: {rot} channels "
                              f"(confidence: {conf:.0%})")
                    if not self.calibrator.has_reference:
                        self.calibrator.save_as_reference(cal_result)
                        self._log("Saved as reference calibration (first session)")
                else:
                    self._log("Rotation detection skipped (not enough gesture data)")
            except Exception as e:
                self._log(f"Rotation detection failed: {e}")

            # Save session (now includes rotation metadata)
            path = self.data_manager.save_session(self._current_session)
            self._log(f"Saved session to {path}")

            self._current_session = None

        self.start_recording_btn.setEnabled(True)
        self.stop_recording_btn.setEnabled(False)
        self.recording_mode_combo.setEnabled(True)
        self.protocol_combo.setEnabled(True)
        self.custom_settings_group.setEnabled(True)
        self._update_manual_combo_enabled_state()

    @Slot(object)
    def _on_step_completed(self, step):
        """Handle protocol step completion."""
        # End trial when HOLD phase completes
        if step.phase == ProtocolPhase.HOLD and step.is_recording:
            if self._current_session and step.gesture:
                self._current_session.end_trial()
                self._log(f"Trial recorded: {step.gesture.display_name}")

        # End rest trial when REST phase completes
        if step.phase == ProtocolPhase.REST and step.is_recording:
            if self._current_session:
                self._current_session.end_trial()
                self._log(f"Rest trial recorded (between gestures)")

        # End calibration-sync trial
        if step.phase == ProtocolPhase.CALIBRATION_SYNC and step.is_recording:
            if self._current_session:
                self._current_session.end_trial()
                self._log("Calibration sync (waveout) recorded")

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

        # Start rest trial at beginning of REST phase
        # This automatically uses the pause between gestures as rest training data
        if step.phase == ProtocolPhase.REST and step.is_recording:
            if self._current_session:
                self._current_session.start_trial("rest")

        # Start calibration-sync trial — bypasses gesture-set lookup, trial_type
        # keeps it out of model training while making it available to the calibrator
        if step.phase == ProtocolPhase.CALIBRATION_SYNC and step.is_recording:
            if self._current_session:
                self._current_session.start_trial(
                    "waveout_sync", trial_type="calibration_sync"
                )

    @Slot()
    def _on_protocol_completed(self):
        """Handle protocol completion."""
        self._log("Protocol completed")
        self._on_stop_recording()
        QMessageBox.information(self, "Complete", "Recording protocol completed!")

    # Calibration handlers
    def _refresh_cal_sessions(self):
        """Refresh the subject and session combos in the calibration tab."""
        self.cal_subject_combo.blockSignals(True)
        self.cal_subject_combo.clear()
        subjects = self.data_manager.list_subjects()
        self.cal_subject_combo.addItems(subjects)
        self.cal_subject_combo.blockSignals(False)

        # Trigger session reload for current subject
        if subjects:
            self._on_cal_subject_changed(self.cal_subject_combo.currentText())

    def _on_cal_subject_changed(self, subject: str):
        """Load sessions for the selected calibration subject."""
        self.cal_session_combo.clear()
        if not subject:
            return
        sessions = self.data_manager.list_sessions(subject)
        self.cal_session_combo.addItems(sessions)
        if sessions:
            self.cal_session_combo.setCurrentText(sessions[-1])

    @Slot()
    def _on_calibrate_from_session(self):
        """Calibrate from a normal recording session (no separate recording needed)."""
        subject = self.cal_subject_combo.currentText()
        session_id = self.cal_session_combo.currentText()

        if not subject or not session_id:
            QMessageBox.warning(self, "Warning",
                                "Please select a subject and session to calibrate from.\n"
                                "Record a session first if none are available.")
            return

        try:
            self._log(f"Loading session {subject}/{session_id} for calibration...")
            session = self.data_manager.load_session(subject, session_id)

            if not session.trials:
                QMessageBox.warning(self, "Warning",
                                    "Selected session has no recorded trials.\n"
                                    "Please select a session with gesture recordings.")
                return

            # Update calibrator to use the session's actual channel count (adaptive)
            # This ensures it works correctly even when bad channels were removed
            self.calibrator.processor.num_channels = session.metadata.num_channels

            # Show which gestures were found
            gesture_names = set(t.gesture_name for t in session.get_valid_trials())
            self._log(f"Found gestures: {', '.join(sorted(gesture_names))}")
            self._log(f"Total valid trials: {len(session.get_valid_trials())}")

            # Run calibration from the session data
            result = self.calibrator.calibrate_from_session(session)

            self._update_calibration_display()
            self._log(f"Calibration completed from session '{session_id}'")
            self._log(f"  Rotation offset: {result.rotation_offset} channels")
            self._log(f"  Confidence: {result.confidence:.2%}")

            # Show per-gesture confidence if available
            per_gesture = result.metadata.get("per_gesture_confidence", {})
            if per_gesture:
                for gesture, conf in sorted(per_gesture.items()):
                    self._log(f"  {gesture}: {conf:.2%}")

            # Check if reference was incompatible
            incompat = result.metadata.get("reference_incompatible")
            if incompat:
                self._log(f"  Warning - Reference was incompatible: {incompat}")
                self._log(f"  Saving this session as the new reference.")
                # Auto-save as new reference since the old one is useless
                self.calibrator.save_as_reference(result)
                QMessageBox.information(self, "New Reference Saved",
                    f"Calibration from session '{session_id}' complete!\n\n"
                    f"The previous reference calibration was incompatible:\n"
                    f"{incompat}\n\n"
                    f"This session has been saved as the new reference.\n"
                    f"Future sessions with the same gestures will detect\n"
                    f"bracelet rotation relative to this recording.")
            elif not self.calibrator.has_reference:
                self._log(f"  No reference found. Saving as reference.")
                self.calibrator.save_as_reference(result)
                QMessageBox.information(self, "Reference Saved",
                    f"Calibration from session '{session_id}' complete!\n\n"
                    f"No existing reference was found, so this session has\n"
                    f"been saved as the new reference calibration.\n\n"
                    f"Future sessions will detect bracelet rotation\n"
                    f"relative to this recording.")
            else:
                QMessageBox.information(self, "Calibration Complete",
                    f"Calibration from session '{session_id}' complete!\n\n"
                    f"Rotation offset: {result.rotation_offset} channels\n"
                    f"Confidence: {result.confidence:.2%}\n\n"
                    f"Save as reference to use for future sessions.")

        except Exception as e:
            self._log(f"Calibration error: {e}")
            QMessageBox.critical(self, "Error", f"Calibration failed: {e}")

    @Slot()
    def _on_save_reference(self):
        """Save current calibration as reference."""
        if self.calibrator.current_calibration:
            self.calibrator.save_as_reference(self.calibrator.current_calibration)
            self._log("Saved calibration as reference")

    @Slot()
    def _on_save_reference_and_recompute(self):
        """Save current calibration as reference and recompute all session rotations."""
        if self.calibrator.current_calibration:
            self._log("Setting new reference and recomputing all session rotations...")
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            try:
                from playagain_pipeline.calibration.calibrator import backfill_session_rotations
                self.calibrator.save_as_reference(
                    self.calibrator.current_calibration,
                    recompute_all=True,
                    data_dir=self.data_dir,
                )
                self._log("All session rotations recomputed relative to new reference")
                self._update_calibration_display()
            except Exception as e:
                self._log(f"Error recomputing rotations: {e}")
            finally:
                QApplication.restoreOverrideCursor()

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

    @Slot()
    def _on_apply_manual_rotation(self):
        """Apply a manually configured rotation offset (for pretrained model usage)."""
        from playagain_pipeline.calibration.calibrator import CalibrationResult
        offset = self.manual_rotation_spin.value()
        num_ch = self._effective_num_channels()
        signal_mode = "bipolar" if self.bipolar_mode_cb.isChecked() else "monopolar"
        mapping = [(i - offset) % num_ch for i in range(num_ch)]
        cal = CalibrationResult(
            created_at=datetime.now(),
            device_name="manual",
            num_channels=num_ch,
            rotation_offset=offset,
            channel_mapping=mapping,
            confidence=1.0,
            reference_patterns={},
            metadata={"source": "manual_override", "signal_mode": signal_mode}
        )
        self.calibrator._current_calibration = cal
        self._update_calibration_display()
        self._log(f"Applied manual rotation offset: {offset} channels")

    @Slot()
    def _on_clear_calibration(self):
        """Clear calibration so no rotation correction is applied."""
        self.calibrator._current_calibration = None
        self._update_calibration_display()
        self._log("Calibration cleared — no rotation correction active")

    # ── Live Quick Calibration ────────────────────────────────────────────────

    @Slot()
    def _on_start_live_calibration(self):
        """Start the live gesture-by-gesture calibration sequence."""
        device = self.device_manager.device
        if not device or not device.is_streaming:
            QMessageBox.warning(self, "Warning",
                                "Please connect and start the device before live calibration.")
            return

        # Update calibrator to use the actual device channel count (adaptive)
        # This ensures it works correctly even when bad channels are removed
        self.calibrator.processor.num_channels = device.num_channels

        gestures_text = self.live_cal_gestures_edit.text()
        gestures = [g.strip() for g in gestures_text.split(",") if g.strip()]
        if not gestures:
            QMessageBox.warning(self, "Warning", "Please enter at least one gesture name.")
            return

        self._live_cal_gestures = gestures
        self._live_cal_current_idx = 0
        self._live_cal_buffer = []
        self._live_cal_collected = {}
        self._live_cal_active = False   # will be set True when countdown reaches 0
        self._live_cal_countdown = 3    # 3-second countdown before first gesture

        self.start_live_cal_btn.setEnabled(False)
        self.cancel_live_cal_btn.setEnabled(True)

        # Start the per-second tick timer
        self._live_cal_timer = QTimer(self)
        self._live_cal_timer.timeout.connect(self._live_cal_tick)
        self._live_cal_timer.start(1000)

        self._live_cal_update_status(f"Get ready… starting in {self._live_cal_countdown}s")
        self._log("Live calibration started")

    # Gesture emoji lookup for live calibration visual prompts
    _GESTURE_EMOJIS = {
        "rest": "🖐🏻", "open_hand": "🖐🏻", "fist": "🤛🏻", "pinch": "👌🏻",
        "tripod": "🤌🏻", "index_point": "👆🏻", "thumb_out": "👍🏻",
        "pinky_out": "🤙🏻", "extension": "🖐🏻",
    }

    def _live_cal_tick(self):
        """Called every second to advance the live calibration sequence."""
        if self._live_cal_countdown > 0:
            self._live_cal_countdown -= 1
            if self._live_cal_countdown > 0:
                msg = f"Get ready… {self._live_cal_countdown}s"
                self._live_cal_update_status(msg)
                self._live_cal_show_gesture_prompt(
                    "GET READY",
                    self._live_cal_gestures[self._live_cal_current_idx],
                    f"Next gesture in {self._live_cal_countdown}s",
                    color="#FFA500"
                )
            else:
                # Countdown finished — begin collecting the current gesture
                self._live_cal_active = True
                self._live_cal_buffer = []
                hold = self.live_cal_duration_spin.value()
                gesture = self._live_cal_gestures[self._live_cal_current_idx]
                self._live_cal_remaining = hold
                self._live_cal_update_status(
                    f"RECORDING: Hold '{gesture}'  ({hold}s remaining)")
                self._live_cal_show_gesture_prompt(
                    "HOLD NOW",
                    gesture,
                    f"{hold}s remaining",
                    color="#F44336"
                )
            return

        # Already in collection phase — count down hold duration
        self._live_cal_remaining -= 1
        gesture = self._live_cal_gestures[self._live_cal_current_idx]

        if self._live_cal_remaining > 0:
            self._live_cal_update_status(
                f"RECORDING: Hold '{gesture}'  ({self._live_cal_remaining}s remaining)")
            self._live_cal_show_gesture_prompt(
                "HOLD NOW",
                gesture,
                f"{self._live_cal_remaining}s remaining",
                color="#F44336"
            )
        else:
            # Collection done for this gesture
            self._live_cal_active = False
            if self._live_cal_buffer:
                self._live_cal_collected[gesture] = np.concatenate(
                    self._live_cal_buffer, axis=0)
                self._log(f"  Collected '{gesture}': "
                          f"{self._live_cal_collected[gesture].shape[0]} samples")
            else:
                self._log(f"  No data received for '{gesture}' — skipped")

            self._live_cal_current_idx += 1

            if self._live_cal_current_idx >= len(self._live_cal_gestures):
                # All gestures done — run calibration
                self._live_cal_show_gesture_prompt(
                    "COMPUTING", "", "Analyzing patterns…", color="#2196F3"
                )
                self._live_cal_finish()
            else:
                # Prepare for next gesture with a short 2-second pause
                self._live_cal_countdown = 2
                next_gesture = self._live_cal_gestures[self._live_cal_current_idx]
                self._live_cal_update_status(
                    f"Done. Next: '{next_gesture}' in {self._live_cal_countdown}s...")
                self._live_cal_show_gesture_prompt(
                    "NEXT UP",
                    next_gesture,
                    f"Get ready… {self._live_cal_countdown}s",
                    color="#2196F3"
                )

    def _live_cal_finish(self):
        """Compute calibration from collected live gesture data."""
        self._live_cal_timer.stop()
        self._live_cal_timer = None
        self._live_cal_active = False

        self.start_live_cal_btn.setEnabled(True)
        self.cancel_live_cal_btn.setEnabled(False)

        if not self._live_cal_collected:
            self._live_cal_update_status("No data collected — calibration aborted")
            self.protocol_widget.gesture_display.clear()
            return

        self._live_cal_update_status("Computing calibration…")
        try:
            result = self.calibrator.calibrate(
                calibration_data=self._live_cal_collected,
                device_name=self.device_combo.currentText(),
                signal_mode=("bipolar" if self.bipolar_mode_cb.isChecked() else "monopolar"),
            )
            self.calibrator._current_calibration = result
            self._update_calibration_display()

            self._log(f"Live calibration complete!")
            self._log(f"  Rotation offset: {result.rotation_offset} channels")
            self._log(f"  Confidence: {result.confidence:.2%}")

            status = (f"✅ Done!  Rotation: {result.rotation_offset} ch  "
                      f"Confidence: {result.confidence:.0%}")
            self._live_cal_update_status(status)
            self._live_cal_show_gesture_prompt(
                "COMPLETE", "", f"Rotation: {result.rotation_offset} ch, "
                f"Confidence: {result.confidence:.0%}", color="#4CAF50"
            )

            QMessageBox.information(self, "Live Calibration Complete",
                f"Rotation offset: {result.rotation_offset} channels\n"
                f"Confidence: {result.confidence:.2%}\n\n"
                f"Calibration applied. You can now load a pretrained model\n"
                f"and start prediction.")
        except Exception as e:
            self._live_cal_update_status(f"Error: {e}")
            self._log(f"Live calibration error: {e}")

    @Slot()
    def _on_cancel_live_calibration(self):
        """Cancel an in-progress live calibration."""
        if self._live_cal_timer:
            self._live_cal_timer.stop()
            self._live_cal_timer = None
        self._live_cal_active = False
        self._live_cal_buffer = []
        self.start_live_cal_btn.setEnabled(True)
        self.cancel_live_cal_btn.setEnabled(False)
        self._live_cal_update_status("Cancelled")
        self.protocol_widget.gesture_display.clear()
        self._log("Live calibration cancelled")

    def _live_cal_update_status(self, text: str):
        """Update the live calibration status label safely."""
        if hasattr(self, 'live_cal_status_label'):
            self.live_cal_status_label.setText(text)

    def _live_cal_show_gesture_prompt(self, phase: str, gesture_name: str,
                                      detail: str, color: str = "#333"):
        """Show a large visual prompt in the protocol widget during live calibration.

        This makes the requested gesture clearly visible alongside the EMG plot,
        using the same gesture display area used during recording protocols.
        """
        display = self.protocol_widget.gesture_display

        # Phase label (GET READY / HOLD NOW / NEXT UP / etc.)
        display.phase_label.setText(phase)
        display.phase_label.setStyleSheet(f"color: {color};")

        # Gesture name (human-readable)
        pretty_name = gesture_name.replace("_", " ").title()
        display.gesture_label.setText(pretty_name)

        # Emoji for the gesture
        emoji = self._GESTURE_EMOJIS.get(gesture_name.lower(), "")
        if emoji:
            display.emoji_label.setText(emoji)
            display.emoji_label.setStyleSheet("")
        else:
            display.emoji_label.setText(pretty_name[:2].upper() if pretty_name else "")
            display.emoji_label.setStyleSheet(
                "background-color: #e0e0e0; border-radius: 10px; "
                "font-size: 48px; font-weight: bold; color: #666;"
            )

        # Detail text (countdown, instruction)
        display.description_label.setText(detail)

    def _update_calibration_display(self):
        """Update calibration status display."""
        cal = self.calibrator.current_calibration
        if cal:
            self.cal_status_label.setText("Loaded")
            self.cal_confidence_label.setText(f"{cal.confidence:.2%}")
            self.cal_rotation_label.setText(f"{cal.rotation_offset} channels")
            self.save_ref_btn.setEnabled(True)
            self.save_ref_recompute_btn.setEnabled(True)
            # Update bracelet graphic
            if hasattr(self, 'bracelet_graphic'):
                self.bracelet_graphic.set_num_electrodes(cal.num_channels)
                self.bracelet_graphic.set_signal_mode(cal.metadata.get("signal_mode", "monopolar"))
                self.bracelet_graphic.set_rotation_offset(cal.rotation_offset)
            # Show sub-score breakdown if available
            if hasattr(self, 'cal_subscores_label'):
                pg = cal.metadata.get("per_gesture_confidence", {})
                if pg:
                    margin = pg.get("__margin_score__", None)
                    agreement = pg.get("__agreement_score__", None)
                    corr = pg.get("__correlation_score__", None)
                    sharpness = pg.get("__sharpness__", None)
                    parts = []
                    if margin is not None:
                        parts.append(f"margin={margin:.0%}")
                    if sharpness is not None:
                        parts.append(f"sharpness={sharpness:.0%}")
                    if agreement is not None:
                        parts.append(f"agreement={agreement:.0%}")
                    if corr is not None:
                        parts.append(f"correlation={corr:.0%}")
                    self.cal_subscores_label.setText("  ".join(parts) if parts else "-")
                else:
                    self.cal_subscores_label.setText("-")
        else:
            self.cal_status_label.setText("No calibration loaded")
            self.cal_confidence_label.setText("-")
            self.cal_rotation_label.setText("-")
            if hasattr(self, 'cal_subscores_label'):
                self.cal_subscores_label.setText("-")
            self.save_ref_btn.setEnabled(False)
            self.save_ref_recompute_btn.setEnabled(False)
            if hasattr(self, 'bracelet_graphic'):
                self.bracelet_graphic.set_num_electrodes(self._effective_num_channels())
                self.bracelet_graphic.set_signal_mode(
                    "bipolar" if self.bipolar_mode_cb.isChecked() else "monopolar"
                )
                self.bracelet_graphic.set_rotation_offset(0)

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
    def _on_delete_dataset(self):
        """Delete the selected dataset(s)."""
        selected_items = self.dataset_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "Warning", "Please select one or more datasets to delete.")
            return

        names = [item.text() for item in selected_items]
        if len(names) == 1:
            msg = f"Are you sure you want to permanently delete the dataset '{names[0]}'?\n\nThis action cannot be undone."
        else:
            name_list = "\n".join(f"  • {n}" for n in names)
            msg = f"Are you sure you want to permanently delete {len(names)} datasets?\n\n{name_list}\n\nThis action cannot be undone."

        reply = QMessageBox.question(
            self, "Confirm Delete", msg,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            for dataset_name in names:
                try:
                    if self.data_manager.delete_dataset(dataset_name):
                        self._log(f"Dataset '{dataset_name}' deleted successfully.")
                    else:
                        QMessageBox.warning(self, "Warning", f"Dataset '{dataset_name}' not found.")
                except Exception as e:
                    self._log(f"Error deleting dataset '{dataset_name}': {e}")
            self._refresh_datasets()

    @Slot()
    def _on_delete_model(self):
        """Delete the selected model(s)."""
        selected_items = self.models_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "Warning", "Please select one or more models to delete.")
            return

        names = [item.text() for item in selected_items]
        if len(names) == 1:
            msg = f"Are you sure you want to permanently delete the model '{names[0]}'?\n\nThis action cannot be undone."
        else:
            name_list = "\n".join(f"  • {n}" for n in names)
            msg = f"Are you sure you want to permanently delete {len(names)} models?\n\n{name_list}\n\nThis action cannot be undone."

        reply = QMessageBox.question(
            self, "Confirm Delete", msg,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            for model_name in names:
                try:
                    # If this model is currently loaded, unload it
                    if (self._current_model is not None and
                            hasattr(self._current_model, 'name') and
                            self._current_model.name == model_name):
                        self._current_model = None
                        self.prediction_label.setText("No prediction")
                        self.confidence_label.setText("Confidence: -")
                        self._log("Unloaded current model before deletion.")

                    if self.model_manager.delete_model(model_name):
                        self._log(f"Model '{model_name}' deleted successfully.")
                    else:
                        QMessageBox.warning(self, "Warning", f"Model '{model_name}' not found.")
                except Exception as e:
                    self._log(f"Error deleting model '{model_name}': {e}")
            self._refresh_models()

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
        session_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)

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
            else: # Session tab
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

        bad_channel_mode_combo = QComboBox()
        bad_channel_mode_combo.addItem("Interpolate (neighbor average)", "interpolate")
        bad_channel_mode_combo.addItem("Zero out bad channels", "zero")
        current_bad_mode = self._get_bad_channel_mode()
        bad_mode_idx = bad_channel_mode_combo.findData(current_bad_mode)
        if bad_mode_idx >= 0:
            bad_channel_mode_combo.setCurrentIndex(bad_mode_idx)
        params_layout.addRow("Bad Ch. Handling:", bad_channel_mode_combo)

        # Per-session rotation alignment (recommended for multi-session datasets)
        per_session_rot_cb = QCheckBox()
        per_session_rot_cb.setChecked(True)  # Default ON — best for combining sessions
        per_session_rot_cb.setText(
            "Use each session's detected rotation offset to align channels"
        )
        per_session_rot_cb.setToolTip(
            "Recommended when combining recordings from different occasions.\n"
            "Each session's bracelet rotation is detected during recording and\n"
            "stored in its metadata. This option applies each session's own\n"
            "rotation correction so channels are aligned to a common reference."
        )
        params_layout.addRow("Per-Session Rotation:", per_session_rot_cb)

        # Apply single global calibration (alternative to per-session)
        apply_cal_cb = QCheckBox()
        has_cal = self.calibrator.current_calibration is not None
        apply_cal_cb.setChecked(False)  # Off by default when per-session is on
        apply_cal_cb.setEnabled(has_cal and not per_session_rot_cb.isChecked())
        cal_label_text = "Global Calibration:"
        if has_cal:
            cal = self.calibrator.current_calibration
            apply_cal_cb.setText(
                f"Rotate all sessions by {cal.rotation_offset} "
                f"(confidence: {cal.confidence:.0%})"
            )
        else:
            apply_cal_cb.setText("No calibration loaded")
        params_layout.addRow(cal_label_text, apply_cal_cb)

        # Mutual exclusion: per-session rotation disables global calibration
        def _on_per_session_toggled(checked):
            if checked:
                apply_cal_cb.setChecked(False)
                apply_cal_cb.setEnabled(False)
            else:
                apply_cal_cb.setEnabled(has_cal)
        per_session_rot_cb.toggled.connect(_on_per_session_toggled)

        # ── Feature extraction options ─────────────────────────────────────
        feature_group = QGroupBox("Feature Extraction (Optional)")
        feature_group_layout = QVBoxLayout(feature_group)

        extract_features_cb = QCheckBox("Pre-extract features at dataset creation time")
        extract_features_cb.setChecked(False)
        extract_features_cb.setToolTip(
            "If enabled, features are computed and stored in the dataset.\n"
            "Training will skip the feature extraction step, making it faster."
        )
        feature_group_layout.addWidget(extract_features_cb)

        from PySide6.QtWidgets import QRadioButton, QButtonGroup, QListWidget as QListWidgetD, QListWidgetItem as QListWidgetItemD
        feat_btn_group = QButtonGroup(dialog)
        feat_radio_default = QRadioButton("All Features (Default)")
        feat_radio_default.setChecked(True)
        feat_btn_group.addButton(feat_radio_default)
        feature_group_layout.addWidget(feat_radio_default)

        feat_radio_custom = QRadioButton("Custom Feature Selection")
        feat_btn_group.addButton(feat_radio_custom)
        feature_group_layout.addWidget(feat_radio_custom)

        feat_list = QListWidgetD()
        feat_list.setMaximumHeight(100)
        feat_list.setEnabled(False)
        from playagain_pipeline.models.feature_pipeline import get_registered_features
        for feat_name in sorted(get_registered_features().keys()):
            item = QListWidgetItemD(feat_name)
            item.setCheckState(Qt.CheckState.Checked)
            feat_list.addItem(item)
        feature_group_layout.addWidget(feat_list)

        def _on_feat_mode_changed():
            feat_list.setEnabled(feat_radio_custom.isChecked())
        feat_radio_custom.toggled.connect(lambda: _on_feat_mode_changed())
        extract_features_cb.toggled.connect(lambda checked: (
            feat_radio_default.setEnabled(checked),
            feat_radio_custom.setEnabled(checked),
            feat_list.setEnabled(checked and feat_radio_custom.isChecked()),
        ))
        feat_radio_default.setEnabled(False)
        feat_radio_custom.setEnabled(False)

        main_layout.addWidget(feature_group)

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
                # Determine rotation correction mode
                use_per_session = per_session_rot_cb.isChecked()
                cal_to_apply = None

                if use_per_session:
                    # Log per-session rotation offsets
                    sessions_with_rot = sum(
                        1 for s in sessions_to_use if s.metadata.rotation_offset != 0
                    )
                    self._log(f"Using per-session rotation alignment "
                              f"({sessions_with_rot}/{len(sessions_to_use)} sessions have non-zero rotation)")
                elif apply_cal_cb.isChecked() and self.calibrator.current_calibration:
                    cal_to_apply = self.calibrator.current_calibration
                    self._log(f"Applying global calibration (rotation offset: {cal_to_apply.rotation_offset})")

                # Build feature config if extraction is enabled
                ds_feature_config = None
                if extract_features_cb.isChecked():
                    if feat_radio_custom.isChecked():
                        selected_feats = []
                        for fi in range(feat_list.count()):
                            item = feat_list.item(fi)
                            if item.checkState() == Qt.CheckState.Checked:
                                selected_feats.append(item.text())
                        ds_feature_config = {"mode": "custom", "features": selected_feats}
                    else:
                        ds_feature_config = {"mode": "default", "features": []}
                    self._log(f"Pre-extracting features (mode: {ds_feature_config['mode']})")

                dataset = self.data_manager.create_dataset(
                    name=name_edit.text(),
                    sessions=sessions_to_use,
                    window_size_ms=window_size_spin.value(),
                    window_stride_ms=stride_spin.value(),
                    include_invalid=include_invalid_cb.isChecked(),
                    calibration=cal_to_apply,
                    use_per_session_rotation=use_per_session,
                    feature_config=ds_feature_config,
                    bad_channel_mode=bad_channel_mode_combo.currentData(),
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

            self._log(f"Dataset loaded: {num_samples} samples, {num_classes} classes, "
                     f"{num_channels} channels, {window_samples} samples/window")

            # Create model
            model_type_text = self.model_type_combo.currentText()
            # Convert UI display names to internal model type names
            if model_type_text == "AttentionNet":
                model_type = "attention_net"
            elif model_type_text == "MSTNet":
                model_type = "mstnet"
            else:
                model_type = model_type_text.lower().replace(" ", "_")
            self._log(f"Creating {model_type} model...")

            # Build model name: type_datasetname_timestamp
            safe_dataset_name = dataset_name.replace(":", "-").replace(" ", "_")
            safe_dataset_name = safe_dataset_name.replace("/", "_").replace("\\\\", "_")
            timestamp = datetime.now().strftime("%H%M%S")
            model_name = f"{model_type}_{safe_dataset_name}_{timestamp}"

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
            self._log(f"  Training accuracy: {results['training_accuracy']:.2%}")
            self._log(f"  Validation accuracy: {results['validation_accuracy']:.2%}")

            if 'training_time' in results:
                self._log(f"  Training time: {results['training_time']:.2f}s")

            if 'feature_count' in results:
                self._log(f"  Features extracted: {results['feature_count']}")

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

            model_bad_mode = getattr(self._current_model.metadata, "bad_channel_mode", None)
            if model_bad_mode in {"interpolate", "zero"} and hasattr(self, "bad_ch_mode_combo"):
                idx = self.bad_ch_mode_combo.findData(model_bad_mode)
                if idx >= 0:
                    self.bad_ch_mode_combo.setCurrentIndex(idx)

            # Initialize prediction buffer using actual device channel count.
            # Model metadata num_channels may be wrong for pre-extracted feature
            # datasets (it gets set to the feature dim instead of raw channels).
            # The buffer receives raw EMG data, so always use device channels.
            device = self.device_manager.device
            num_ch = device.num_channels if device else self.channels_spin.value()
            window_samples = int(self._prediction_window_ms * self._current_model.metadata.sampling_rate / 1000)
            self._prediction_buffer = np.zeros((window_samples, num_ch))

            # Warn if channel count differs from model's trained channel count
            model_ch = self._current_model.metadata.num_channels
            if model_ch > 0 and num_ch != model_ch:
                    self._log(
                        f"  Channel mismatch: device has {num_ch} ch, model trained on {model_ch} ch."
                    )

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

        # Create GUI-side smoother for display
        if self.smoothing_enabled_cb.isChecked():
            alpha = self.smoothing_alpha_spin.value() / 100.0
            stability = self.smoothing_stability_spin.value()
            self._gui_smoother = PredictionSmoother(alpha=alpha, min_stable_ms=stability)
        else:
            self._gui_smoother = None

        # Create and start prediction worker thread
        self._prediction_worker = PredictionWorker(self)
        self._prediction_worker.set_model(self._current_model)
        self._prediction_worker.prediction_ready.connect(self._on_prediction_ready)
        self._prediction_worker.start()

        # Resume prediction server if it was paused
        if self._prediction_server and self._prediction_server.is_running and self._prediction_server.is_paused:
            self._prediction_server.resume()

        self._is_predicting = True
        self.start_pred_btn.setEnabled(False)
        self.stop_pred_btn.setEnabled(True)
        self._log("Started prediction")

    @Slot()
    def _on_stop_prediction(self):
        """Stop real-time prediction."""
        # Immediately set flag to prevent any queued signals from updating display
        self._is_predicting = False

        if self._prediction_worker:
            # Disconnect the signal BEFORE stopping to prevent race conditions
            # where pending signals are still processed after stop() is called
            self._prediction_worker.prediction_ready.disconnect(self._on_prediction_ready)
            self._prediction_worker.stop()
            self._prediction_worker = None

        # Pause the prediction server if running — keeps TCP connection to Unity alive
        # but stops running inference on incoming EMG data
        if self._prediction_server and self._prediction_server.is_running:
            self._prediction_server.pause()

        # Clear GUI immediately (not waiting for signal disconnect to propagate)
        self.start_pred_btn.setEnabled(True)
        self.stop_pred_btn.setEnabled(False)
        self.prediction_label.setText("No prediction")
        self.confidence_label.setText("Confidence: -")
        self._log("Stopped prediction")

    @Slot()
    def _on_start_server(self):
        """Start the Unity TCP prediction server."""
        if not self._current_model:
            QMessageBox.warning(self, "Warning", "Please load a model first")
            return

        # Stop standalone PredictionWorker if running — the server does its own
        # predictions, so running both wastes CPU and can cause stalls.
        if self._prediction_worker:
            self._prediction_worker.stop()
            self._prediction_worker = None
            self._is_predicting = False
            self.start_pred_btn.setEnabled(True)
            self.stop_pred_btn.setEnabled(False)
            self._log("Standalone prediction stopped (server takes over)")

        host = self.server_host_edit.text().strip()
        port = self.server_port_spin.value()

        self._prediction_server = PredictionServer(host=host, port=port)
        self._prediction_server.set_model(self._current_model)

        # Register a callback so server predictions update the GUI labels.
        # The callback runs on the prediction worker thread, so we use
        # QTimer.singleShot to marshal the UI update to the main thread.
        self._prediction_server.add_prediction_callback(self._on_server_prediction)

        # Apply smoothing settings to server
        smoothing_on = self.smoothing_enabled_cb.isChecked()
        self._prediction_server.set_smoothing_enabled(smoothing_on)
        if smoothing_on:
            alpha = self.smoothing_alpha_spin.value() / 100.0
            stability = self.smoothing_stability_spin.value()
            self._prediction_server.set_smoothing_params(alpha=alpha, min_stable_ms=stability)

        self._prediction_server.start()

        self.start_server_btn.setEnabled(False)
        self.stop_server_btn.setEnabled(True)
        self.server_status_label.setText(f"Server: Running on {host}:{port}")
        self._log(f"Unity TCP server started on {host}:{port}")

    @Slot()
    def _on_stop_server(self):
        """Stop the Unity TCP prediction server."""
        # Stop game recording first if active (it depends on the server)
        if self._game_recorder and self._game_recorder.is_recording:
            self._on_stop_game_recording()

        if self._prediction_server:
            # Remove our GUI prediction callback before stopping
            self._prediction_server.remove_prediction_callback(self._on_server_prediction)
            self._prediction_server.stop()
            self._prediction_server = None

        self.start_server_btn.setEnabled(True)
        self.stop_server_btn.setEnabled(False)
        self.server_status_label.setText("Server: Not running")
        self._log("Unity TCP server stopped")

    def _update_prediction_buffer(self, data: np.ndarray):
        """Update the prediction buffer with new data and pass to worker."""
        if self._prediction_buffer is None or self._current_model is None:
            return

        # Guard: re-create buffer if channel count changed (e.g. device reconnect)
        if data.shape[1] != self._prediction_buffer.shape[1]:
            window_samples = self._prediction_buffer.shape[0]
            self._prediction_buffer = np.zeros((window_samples, data.shape[1]))

        # Roll buffer and add new data
        n_samples = min(data.shape[0], len(self._prediction_buffer))
        self._prediction_buffer = np.roll(self._prediction_buffer, -n_samples, axis=0)
        self._prediction_buffer[-n_samples:] = data[:n_samples]

        # Send updated buffer to worker thread
        if self._prediction_worker:
            self._prediction_worker.update_buffer(self._prediction_buffer)

    @Slot(object, object)
    def _on_prediction_ready(self, pred, proba):
        """Handle prediction result from worker thread (updates UI with optional smoothing)."""
        if not self._is_predicting or self._current_model is None:
            return

        try:
            class_names = self._current_model.metadata.class_names
            class_name = class_names.get(int(pred), class_names.get(str(pred), f"Class {pred}"))

            try:
                confidence = proba[int(pred)]
            except (IndexError, KeyError):
                confidence = np.max(proba)

            # Apply GUI-side smoothing if enabled
            if self._gui_smoother is not None:
                # Build probabilities dict for smoother
                prob_dict = {}
                for idx, p in enumerate(proba):
                    name = class_names.get(idx, class_names.get(str(idx), f"class_{idx}"))
                    prob_dict[name] = float(p)

                class_name, _, confidence, _ = self._gui_smoother.smooth(
                    class_name, int(pred), float(confidence), prob_dict
                )

            display_name = class_name.replace("_", " ").title()
            self.prediction_label.setText(display_name)
            self.confidence_label.setText(f"Confidence: {confidence:.1%}")
        except Exception:
            pass

    def _on_server_prediction(self, gesture: str, gesture_id: int,
                              confidence: float, probabilities: dict):
        """
        Callback from PredictionServer (runs on its background thread).
        Emits a Qt Signal to safely marshal the UI update onto the main thread.

        Throttled to ~20 Hz to prevent flooding the Qt event loop.
        """
        import time as _time
        now = _time.monotonic()
        # Throttle: skip update if less than 50 ms since last one
        if hasattr(self, '_last_server_pred_time') and now - self._last_server_pred_time < 0.05:
            return
        self._last_server_pred_time = now

        display_name = gesture.replace("_", " ").title()

        # Emit signal — Qt guarantees cross-thread delivery via event queue
        self._server_prediction_signal.emit(display_name, confidence)

    @Slot(str, float)
    def _apply_server_prediction(self, display_name: str, confidence: float):
        """Update prediction labels on the main thread."""
        # Guard against stale signals from server after it has been stopped
        if not self._prediction_server or not self._prediction_server.is_running:
            return

        self.prediction_label.setText(display_name)
        self.confidence_label.setText(f"Confidence: {confidence:.1%}")

    # ─── Smoothing Handlers ───────────────────────────────────────────────

    @Slot(bool)
    def _on_smoothing_toggled(self, enabled: bool):
        """Handle smoothing checkbox toggle."""
        # Update server smoother if running
        if self._prediction_server:
            self._prediction_server.set_smoothing_enabled(enabled)

        # Update GUI smoother
        if enabled:
            alpha = self.smoothing_alpha_spin.value() / 100.0
            stability = self.smoothing_stability_spin.value()
            self._gui_smoother = PredictionSmoother(alpha=alpha, min_stable_ms=stability)
        else:
            self._gui_smoother = None

        self._log(f"Smoothing {'enabled' if enabled else 'disabled'}")

    @Slot(int)
    def _on_smoothing_params_changed(self, _value=None):
        """Handle smoothing parameter changes."""
        alpha = self.smoothing_alpha_spin.value() / 100.0
        stability = self.smoothing_stability_spin.value()

        # Update server smoother
        if self._prediction_server:
            self._prediction_server.set_smoothing_params(alpha=alpha, min_stable_ms=stability)

        # Update GUI smoother
        if self._gui_smoother:
            self._gui_smoother.alpha = alpha
            self._gui_smoother.min_stable_ms = stability

    # ─── Game Recording Handlers ──────────────────────────────────────────

    @Slot()
    def _on_start_game_recording(self):
        """Start recording gameplay data (EMG + predictions + ground truth)."""
        if not self._prediction_server or not self._prediction_server.is_running:
            QMessageBox.warning(self, "Warning",
                                "Please start the Unity TCP Server first.\n"
                                "Game recording requires the server to receive ground truth from Unity.")
            return

        if not self._current_model:
            QMessageBox.warning(self, "Warning", "Please load a model first.")
            return

        device = self.device_manager.device
        if not device or not device.is_streaming:
            QMessageBox.warning(self, "Warning", "Please connect and start device first.")
            return

        # Create game recorder
        self._game_recorder = GameRecorder(output_dir=self.data_dir)

        # Set class names from model metadata
        if self._current_model.metadata and self._current_model.metadata.class_names:
            self._game_recorder.set_class_names(self._current_model.metadata.class_names)

        # Set model metadata for config.json
        if self._current_model.metadata:
            self._game_recorder.set_model_metadata(self._current_model.metadata)

        # Set calibration/rotation info for the game recording
        if self.calibrator.current_calibration:
            cal = self.calibrator.current_calibration
            self._game_recorder.set_calibration_info(
                rotation_offset=cal.rotation_offset,
                confidence=cal.confidence,
                channel_mapping=cal.channel_mapping
            )

        # Attach participant info for the gameplay recording
        subject_id = self.game_rec_subject_edit.text().strip() or self.subject_id_edit.text().strip()
        if not subject_id:
            subject_id = self.data_manager.get_next_subject_id()
        self.game_rec_subject_edit.setText(subject_id)

        participant_info = self._ensure_participant_info(subject_id, allow_create=True)
        if participant_info is None and not self.data_manager.has_participant_info(subject_id):
            QMessageBox.warning(self, "Warning", "Participant info is required before starting a new participant.")
            self._game_recorder = None
            return
        self._game_recorder.set_participant_info(participant_info)

        # Register callbacks with the prediction server
        self._prediction_server.add_prediction_callback(self._game_recorder.on_prediction)
        self._prediction_server.add_game_state_callback(self._game_recorder.on_game_state)

        # Determine number of channels (all channels kept, bad ones zeroed)
        num_channels = device.num_channels

        # Start recording
        subject_id = subject_id or None
        session_name = self.game_rec_session_edit.text().strip() or None

        file_path = self._game_recorder.start_recording(
            num_channels=num_channels,
            session_name=session_name,
            subject_id=subject_id,
            sampling_rate=device.sampling_rate
        )

        # Update UI
        self.start_game_rec_btn.setEnabled(False)
        self.stop_game_rec_btn.setEnabled(True)
        self.game_rec_status_label.setText("Game Recording: ACTIVE")
        self.game_rec_status_label.setStyleSheet("font-weight: bold; color: red;")
        self._log(f"Game recording started: {file_path}")

        # Start a timer to update recording stats in the UI
        if not hasattr(self, '_game_rec_timer'):
            self._game_rec_timer = QTimer(self)
            self._game_rec_timer.timeout.connect(self._update_game_rec_status)
        self._game_rec_timer.start(1000)

    @Slot()
    def _on_stop_game_recording(self):
        """Stop recording gameplay data."""
        if self._game_recorder and self._game_recorder.is_recording:
            file_path = self._game_recorder.stop_recording()

            # Remove callbacks from server
            if self._prediction_server:
                self._prediction_server.remove_prediction_callback(self._game_recorder.on_prediction)
                self._prediction_server.remove_game_state_callback(self._game_recorder.on_game_state)

            self._log(f"Game recording saved: {file_path}")

        self._game_recorder = None

        # Update UI
        self.start_game_rec_btn.setEnabled(True)
        self.stop_game_rec_btn.setEnabled(False)
        self.game_rec_status_label.setText("Game Recording: Idle")
        self.game_rec_status_label.setStyleSheet("font-weight: bold; color: black;")

        # Stop stats timer
        if hasattr(self, '_game_rec_timer'):
            self._game_rec_timer.stop()

    def _update_game_rec_status(self):
        """Update game recording status label with live stats."""
        if self._game_recorder and self._game_recorder.is_recording:
            samples = self._game_recorder.sample_count
            duration = self._game_recorder.duration
            self.game_rec_status_label.setText(
                f"Game Recording: ACTIVE — {samples:,} samples, {duration:.0f}s"
            )

    def _get_excluded_channels(self) -> list[int]:
        """Get list of excluded channel indices (0-based) from the EMG plot checkboxes."""
        return list(self._excluded_channels)

    def _get_bad_channel_mode(self) -> str:
        """Return configured bad-channel handling mode."""
        if hasattr(self, "bad_ch_mode_combo"):
            mode = self.bad_ch_mode_combo.currentData()
            if mode in {"interpolate", "zero"}:
                return mode
        return "interpolate"

    def _effective_num_channels(self) -> int:
        """Return currently active output channel count."""
        device = self.device_manager.device
        if device and device.is_connected:
            return int(device.num_channels)
        physical = int(self.channels_spin.value())
        if self.bipolar_mode_cb.isChecked() and physical % 2 == 0:
            if physical % 32 == 0:
                return (physical // 32) * 16
            return physical // 2
        return physical

    def _on_bipolar_mode_toggled(self, checked: bool):
        """Update bracelet visualization for selected signal mode."""
        if hasattr(self, "bracelet_graphic"):
            self.bracelet_graphic.set_num_electrodes(self._effective_num_channels())
            self.bracelet_graphic.set_signal_mode("bipolar" if checked else "monopolar")

    def _on_bad_channel_mode_changed(self, *_):
        """Persist selected bad-channel handling mode in runtime config."""
        if hasattr(self, "config") and hasattr(self.config, "model"):
            self.config.model.bad_channel_mode = self._get_bad_channel_mode()

    @Slot(np.ndarray)
    def _on_bad_channels_updated(self, lines_enabled: np.ndarray):
        """Handle channel toggle from the EMG plot's built-in checkboxes.

        Args:
            lines_enabled: Bool array — True means channel is enabled, False means bad/excluded.
        """
        self._excluded_channels = [i for i, enabled in enumerate(lines_enabled) if not enabled]

    def _update_channel_checkboxes(self, num_channels: int):
        """Reset excluded channels when device channel count changes."""
        self._excluded_channels = []

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
    """Entry point for the GUI application."""
    import sys
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
