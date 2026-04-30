"""
Main application window for gesture recording pipeline.

Combines all components into a unified interface for:
- Recording training data
- Calibration
- Real-time prediction
"""

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
                               QLineEdit, QLabel, QGroupBox, QFormLayout, QSpinBox, QDoubleSpinBox, QMessageBox,
                               QFileDialog, QTextEdit, QFrame,
                               QSplitter, QStatusBar, QListWidget, QScrollArea, QCheckBox, QSizePolicy,
                               QApplication, QDialog)

from playagain_pipeline.calibration.calibrator import AutoCalibrator
from playagain_pipeline.config.config import get_default_config, PipelineConfig
from playagain_pipeline.core.data_manager import DataManager
from playagain_pipeline.core.gesture import GestureSet, create_default_gesture_set, create_single_gesture_set
from playagain_pipeline.core.session import RecordingSession
from playagain_pipeline.devices.emg_device import (DeviceManager, DeviceType, SyntheticEMGDevice)
from playagain_pipeline.gui.widgets.emg_plot_panel import EMGPlotPanel
from playagain_pipeline.gui.widgets.validation_tab import ValidationTab
from playagain_pipeline.gui.widgets.protocol_popup import ProtocolPopup
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
from playagain_pipeline.gui.widgets.busy_overlay import run_blocking
from playagain_pipeline.gui.widgets.workflow_stepper import WorkflowStepper
from playagain_pipeline.game_recorder import GameRecorder
from playagain_pipeline.training_game_coordinator import (
    TrainingGameCoordinator, TrialSpec, build_default_schedule,
)
from playagain_pipeline.unity_launcher import UnityLauncher, UnityNotFoundError
from playagain_pipeline.gui.widgets.game_protocol_popup import GameProtocolPopup



# ---------------------------------------------------------------------------
# Tab index constants — single source of truth so the new ordering can be
# changed in one place without breaking signal/slot lookups elsewhere.
# ---------------------------------------------------------------------------
TAB_RECORD       = 0
TAB_TRAIN        = 1
TAB_PREDICT      = 2
TAB_CALIBRATION  = 3   # optional, moved out of the second slot on purpose
TAB_VALIDATION   = 4

# Workflow stepper indices (only the three primary stages get a step badge;
# Calibration & Validation are side-tabs accessible from the bar).
STEP_RECORD  = 0
STEP_TRAIN   = 1
STEP_PREDICT = 2


def _make_step_header(title: str, subtitle: str, accent: str = "#0284c7") -> QWidget:
    """
    Build a small banner placed at the top of each tab so the user always
    sees which step they are on and what to do next, without having to
    read every group-box label.
    """
    wrap = QFrame()
    wrap.setStyleSheet(
        f"QFrame {{"
        f"  background: #f1f5f9;"
        f"  border-left: 4px solid {accent};"
        f"  border-radius: 4px;"
        f"}}"
    )
    lay = QVBoxLayout(wrap)
    lay.setContentsMargins(10, 6, 10, 6)
    lay.setSpacing(2)

    title_lbl = QLabel(title)
    tf = QFont()
    tf.setBold(True)
    tf.setPointSize(12)
    title_lbl.setFont(tf)
    title_lbl.setStyleSheet(f"color: {accent}; background: transparent; border: none;")
    lay.addWidget(title_lbl)

    sub_lbl = QLabel(subtitle)
    sub_lbl.setWordWrap(True)
    sub_lbl.setStyleSheet("color: #475569; font-size: 11px; background: transparent; border: none;")
    lay.addWidget(sub_lbl)

    return wrap


class PredictionWorker(QThread):
    """Background worker for model prediction to avoid blocking the GUI thread."""
    prediction_ready = Signal(object, object)  # pred_class, proba_array
    # Surfaced from the worker thread so the main window can log the FIRST
    # error in the GUI log instead of having it disappear into stdout. If we
    # don't do this, a channel-count mismatch (the most common cause of
    # "Model loaded but no predictions") is completely invisible to the user.
    error_occurred = Signal(str)

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
                    # Use one model pass to reduce per-frame CPU load.
                    proba = self._model.predict_proba(X)[0]
                    pred = int(np.argmax(proba))
                    self.prediction_ready.emit(pred, proba)
                except Exception as e:
                    # Log prediction errors (throttled to avoid spam) AND
                    # forward the first occurrence to the GUI so the user
                    # actually finds out something is wrong.
                    if not hasattr(self, '_last_error') or str(e) != self._last_error:
                        self._last_error = str(e)
                        print(f"[PredictionWorker] Prediction error: {e}")
                        try:
                            self.error_occurred.emit(str(e))
                        except RuntimeError:
                            # Worker may be tearing down — emit on a deleted
                            # signal raises; safe to ignore.
                            pass

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
        self.handedness_combo.setCurrentText(
            handedness if handedness in {"Unknown", "Left", "Right", "Ambidextrous"} else "Unknown")
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
    _MUOVI_CONNECT_TIMEOUT_MS = 15_000

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
        #self._plot_widget: Optional[EMGPlotWidget] = None

        # Ground truth label for session replay
        self._current_ground_truth_label: Optional[str] = None
        self._awaiting_muovi_handshake = False

        # Quattrocento file-stream state (independent from DeviceManager device)
        self._q4_stream_active = False
        self._q4_stream_channels: Optional[int] = None

        # Live quick calibration state
        self._live_cal_active = False
        self._live_cal_gestures: list = []
        self._live_cal_current_idx = 0
        self._live_cal_buffer: list = []  # accumulated EMG chunks for current gesture
        self._live_cal_collected: dict = {}  # gesture -> np.ndarray
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
        # Vertical wrapper so we can put the workflow stepper above the
        # main horizontal splitter.
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # ── Workflow stepper banner ───────────────────────────────────────
        # Three primary stages — Calibration & Performance Review are
        # side-tabs that don't get their own badge here.
        self.workflow_stepper = WorkflowStepper([
            ("Record",            "Connect device, capture gestures"),
            ("Train & Evaluate",  "Build dataset, train & validate model"),
            ("Predict",           "Run live inference / Unity server"),
        ])
        self.workflow_stepper.step_clicked.connect(self._on_workflow_step_clicked)
        root_layout.addWidget(self.workflow_stepper)

        # Main horizontal area below the stepper
        main_area = QWidget()
        main_layout = QHBoxLayout(main_area)
        main_layout.setContentsMargins(6, 6, 6, 6)
        root_layout.addWidget(main_area, 1)

        # Main splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)

        # Left panel - Controls
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        # Tab widget for different modes
        self.mode_tabs = QTabWidget()
        self.mode_tabs.currentChanged.connect(self._on_mode_tab_changed)

        # ── New tab order ────────────────────────────────────────────────
        # 1. Record  →  2. Train & Evaluate  →  3. Predict
        # Calibration is moved to position 4 (optional), and Performance
        # Review stays at the end as the analysis tab.

        recording_tab = self._create_recording_tab()
        self.mode_tabs.addTab(recording_tab, "①  Record")
        self.mode_tabs.setTabToolTip(
            TAB_RECORD,
            "Step 1 — Connect a device and record gesture sessions for one subject."
        )

        training_tab = self._create_training_tab()
        self.mode_tabs.addTab(training_tab, "②  Train && Evaluate")
        self.mode_tabs.setTabToolTip(
            TAB_TRAIN,
            "Step 2 — Build datasets from recordings, train and validate models. "
            "Includes Quattrocento offline training."
        )

        prediction_tab = self._create_prediction_tab()
        self.mode_tabs.addTab(prediction_tab, "③  Predict")
        self.mode_tabs.setTabToolTip(
            TAB_PREDICT,
            "Step 3 — Run a trained model live on the device or stream "
            "predictions to Unity."
        )

        calibration_tab = self._create_calibration_tab()
        self.mode_tabs.addTab(calibration_tab, "Calibration  (optional)")
        self.mode_tabs.setTabToolTip(
            TAB_CALIBRATION,
            "Optional — only needed when reusing a pretrained model with a "
            "rotated bracelet, or to set a new reference orientation."
        )

        self._validation_tab = ValidationTab(self.data_manager)
        self.mode_tabs.addTab(self._validation_tab, "Validation")
        self.mode_tabs.setTabToolTip(
            TAB_VALIDATION,
            "Reproducible cross-validation across feature sets, models, and CV "
            "strategies. Replaces the old Performance Review tab — every run is "
            "saved to data/validation_runs/ with full config and environment."
        )
        self.mode_tabs.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        left_layout.addWidget(self.mode_tabs)

        # Log area
        log_group = QGroupBox("Log")
        log_layout = QVBoxLayout(log_group)

        # Create a horizontal layout for the title/clear button and the log text itself
        log_header_layout = QHBoxLayout()
        log_header_layout.addStretch()
        clear_log_btn = QPushButton("Clear")
        clear_log_btn.setFixedHeight(20)
        clear_log_btn.setFixedWidth(50)
        clear_log_btn.clicked.connect(self.log_text.clear if hasattr(self, 'log_text') else lambda: None)
        log_header_layout.addWidget(clear_log_btn)
        log_layout.addLayout(log_header_layout)

        self.log_text = QTextEdit()
        clear_log_btn.clicked.disconnect()
        clear_log_btn.clicked.connect(self.log_text.clear)

        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(200)
        self.log_text.setMaximumHeight(300)
        self.log_text.setStyleSheet(
            "QTextEdit {font-size: 11px; "
            "background-color: #1e1e1e; color: #d4d4d4; "
            "border: 1px solid #444; border-radius: 3px; padding: 4px; }"
        )
        log_layout.addWidget(self.log_text)
        left_layout.addWidget(log_group)
        left_layout.setStretch(0, 1)
        left_layout.setStretch(1, 0)

        splitter.addWidget(left_panel)

        # Right panel — fixed live EMG plot (always visible while the
        # app is running). ``self._plot_widget`` was already created in
        # ``_create_recording_tab`` as an EMGPlotPanel; we just host it
        # here in the main splitter rather than inside the Record tab.
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(self._plot_widget, stretch=1)

        splitter.addWidget(right_panel)

        # Gesture instructions are shown in a floating popup during
        # recording instead of occupying the sidebar. Existing calls
        # like ``self.protocol_widget.set_protocol(...)``, ``.start()``,
        # ``.stop()``, and ``.gesture_display.clear()`` all continue to
        # work — ProtocolPopup forwards them to its inner ProtocolWidget.
        self.protocol_widget = ProtocolPopup(parent=self)
        # Let the popup's red "Stop Recording" button drive the existing
        # stop handler, so the popup acts as a remote for the Record tab.
        self.protocol_widget.stop_requested.connect(self._on_stop_recording)

        # Set splitter proportions — left (controls + log) vs. right
        # (live plot). The plot benefits from a generous share.
        splitter.setSizes([700, 600])

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

        layout.addWidget(_make_step_header(
            "Step 1 — Record",
            "Set the subject ID, connect a device, choose a protocol, and capture gestures.",
        ))

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
        self.device_combo.addItems(["Synthetic", "Muovi", "Muovi Plus", "Quattrocento"])
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

        # Build the live plot panel here (where we already have the
        # channel/sampling-rate defaults handy) but do NOT add it to
        # the Record-tab form layout — ``_setup_ui`` re-parents it to
        # the main splitter's right panel so it stays visible across
        # every tab. A small breadcrumb replaces it in the form so
        # users wondering "where did the plot go?" can find it.
        self._plot_widget = EMGPlotPanel(num_channels=32, sampling_rate=2000)
        self._plot_widget.bad_channels_updated.connect(self._on_bad_channels_updated)

        _live_plot_note = QLabel("Shown in the right panel  →")
        _live_plot_note.setStyleSheet(
            "color: #64748b; font-size: 10px; font-style: italic;"
        )
        device_layout.addRow("Live plot:", _live_plot_note)

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
        self.start_recording_btn.setShortcut("Ctrl+R")
        self.start_recording_btn.clicked.connect(self._on_start_recording)
        control_layout.addWidget(self.start_recording_btn)

        self.stop_recording_btn = QPushButton("Stop Recording")
        self.stop_recording_btn.setFixedHeight(36)
        self.stop_recording_btn.setShortcut("Ctrl+S")
        self.stop_recording_btn.setEnabled(False)
        self.stop_recording_btn.clicked.connect(self._on_stop_recording)
        control_layout.addWidget(self.stop_recording_btn)

        scroll_layout.addWidget(control_group)

        # ── Training game (Unity integration) ─────────────────────────
        # Recording gesture data with children is easier if they see an
        # animal walk in and get fed each time they try a gesture. This
        # panel launches the Unity PlayAgain build and bridges it to the
        # current recording protocol.
        #
        # "Easy mode" is the critical part: the first time a child
        # records, there is no trained model yet, so the coordinator
        # just watches the RMS of the incoming EMG and broadcasts a
        # confidence=1.0 "prediction" to Unity whenever the child even
        # slightly tenses a muscle. The animal gets fed → positive
        # feedback → more data, less frustration.
        game_group = QGroupBox("Training Game (Unity) — easy mode")
        game_layout = QFormLayout(game_group)

        unity_row = QHBoxLayout()
        self.unity_path_label = QLabel()
        self.unity_path_label.setStyleSheet("color: #475569; font-size: 10px;")
        unity_row.addWidget(self.unity_path_label, 1)

        self.pick_unity_btn = QPushButton("Locate Unity…")
        self.pick_unity_btn.clicked.connect(self._on_pick_unity_exe)
        unity_row.addWidget(self.pick_unity_btn)
        game_layout.addRow("Unity build:", unity_row)

        self.game_easy_ratio = QDoubleSpinBox()
        self.game_easy_ratio.setRange(1.1, 4.0)
        self.game_easy_ratio.setSingleStep(0.1)
        self.game_easy_ratio.setValue(1.8)
        self.game_easy_ratio.setToolTip(
            "How many times above the resting RMS baseline counts as 'the "
            "child tried'. Lower = more forgiving (feeds the animal on "
            "weaker contractions). 1.8 is a gentle default for first "
            "sessions; raise toward 2.5 once the child is comfortable."
        )
        game_layout.addRow("Easy-mode sensitivity:", self.game_easy_ratio)

        self.game_reps_spin = QSpinBox()
        self.game_reps_spin.setRange(1, 30)
        self.game_reps_spin.setValue(3)
        self.game_reps_spin.setToolTip(
            "Number of trials per gesture. With 3 gestures and the "
            "default of 3 reps, the game runs 9 trials in total — "
            "exactly 3 fist, 3 pinch, and 3 tripod. Balanced mode "
            "guarantees this count even if Unity spawns animals in "
            "its own order."
        )
        game_layout.addRow("Reps per gesture:", self.game_reps_spin)

        game_btn_row = QHBoxLayout()
        self.start_game_btn = QPushButton("▶  Launch Training Game")
        self.start_game_btn.setFixedHeight(36)
        self.start_game_btn.setStyleSheet(
            "QPushButton {"
            "  background: #16a34a; color: white; border: none;"
            "  border-radius: 4px; font-weight: 600;"
            "}"
            "QPushButton:hover { background: #15803d; }"
            "QPushButton:disabled { background: #cbd5e1; color: #64748b; }"
        )
        self.start_game_btn.clicked.connect(self._on_start_training_game)
        game_btn_row.addWidget(self.start_game_btn)

        self.stop_game_btn = QPushButton("■  Stop")
        self.stop_game_btn.setFixedHeight(36)
        self.stop_game_btn.setEnabled(False)
        self.stop_game_btn.clicked.connect(self._on_stop_training_game)
        game_btn_row.addWidget(self.stop_game_btn)
        game_layout.addRow(game_btn_row)

        self.game_status_label = QLabel("Idle.")
        self.game_status_label.setStyleSheet(
            "color: #475569; font-size: 11px; padding: 4px;"
        )
        self.game_status_label.setWordWrap(True)
        game_layout.addRow(self.game_status_label)

        scroll_layout.addWidget(game_group)

        # Refresh the Unity-path label from whatever's saved in QSettings.
        # Safe to call now — UnityLauncher is a cheap constructor.
        self._refresh_unity_path_label()

        scroll_layout.addStretch()

        scroll.setWidget(scroll_content)
        layout.addWidget(scroll)

        self._on_recording_mode_changed()

        return tab

    def _on_recording_mode_changed(self):
        """Update protocol controls based on selected recording mode."""
        mode_text = self.recording_mode_combo.currentText() if hasattr(self,
                                                                       "recording_mode_combo") else "Preset Protocol"
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

        layout.addWidget(_make_step_header(
            "Optional — Calibration",
            "Only needed when reusing a pretrained model with a rotated electrode bracelet, "
            "or when establishing a new reference orientation. Skip this for first-time training.",
            accent="#a78bfa",
        ))

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
        """Create the improved model training tab."""
        tab = QWidget()
        outer = QVBoxLayout(tab)
        outer.setContentsMargins(6, 6, 6, 6)

        outer.addWidget(_make_step_header(
            "Step 2 — Train && Evaluate",
            "Build a dataset from your recordings, train a model, and (optionally) evaluate "
            "Quattrocento offline data using the same harness.",
        ))

        # Keep this step usable on smaller windows while preserving splitter layout.
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_lay = QVBoxLayout(scroll_content)
        scroll_lay.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.setChildrenCollapsible(False)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        scroll_lay.addWidget(splitter)

        # ── Top half: Dataset management ──────────────────────────────────────
        top_widget = QWidget()
        top_lay = QVBoxLayout(top_widget)
        top_lay.setContentsMargins(0, 0, 0, 0)

        ds_grp = QGroupBox("Datasets")
        ds_lay = QVBoxLayout(ds_grp)

        # Toolbar
        ds_toolbar = QHBoxLayout()
        self.refresh_datasets_btn = QPushButton("Refresh")
        self.refresh_datasets_btn.setFixedHeight(28)
        self.refresh_datasets_btn.clicked.connect(self._refresh_datasets)
        ds_toolbar.addWidget(self.refresh_datasets_btn)

        self.create_dataset_btn = QPushButton("＋  New Dataset…")
        self.create_dataset_btn.setFixedHeight(28)
        self.create_dataset_btn.setStyleSheet(
            "QPushButton{background:#1a3a1a;color:#22c55e;"
            "border:1px solid #22c55e;border-radius:5px;padding:4px 10px;}"
            "QPushButton:hover{background:#22c55e;color:#fff;}")
        self.create_dataset_btn.clicked.connect(self._on_create_dataset)
        ds_toolbar.addWidget(self.create_dataset_btn)

        self.delete_dataset_btn = QPushButton("Delete")
        self.delete_dataset_btn.setFixedHeight(28)
        self.delete_dataset_btn.setStyleSheet(
            "QPushButton{color:#ef4444;border-color:#ef4444;border-radius:5px;padding:4px 10px;}"
            "QPushButton:hover{background:#ef4444;color:#fff;}")
        self.delete_dataset_btn.clicked.connect(self._on_delete_dataset)
        ds_toolbar.addWidget(self.delete_dataset_btn)

        ds_toolbar.addStretch()

        # Dataset info label (updates on selection)
        self._ds_info_lbl = QLabel("Select a dataset to see info")
        self._ds_info_lbl.setStyleSheet(
            "color:#94a3b8;font-size:10px;font-style:italic;")
        ds_toolbar.addWidget(self._ds_info_lbl)
        ds_lay.addLayout(ds_toolbar)

        self.dataset_list = QListWidget()
        self.dataset_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.dataset_list.setAlternatingRowColors(True)
        self.dataset_list.setMinimumHeight(140)
        self.dataset_list.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.dataset_list.currentItemChanged.connect(self._on_dataset_selection_changed)
        ds_lay.addWidget(self.dataset_list)

        top_lay.addWidget(ds_grp)
        splitter.addWidget(top_widget)

        # ── Bottom half: Model training ───────────────────────────────────────
        bot_widget = QWidget()
        bot_lay = QVBoxLayout(bot_widget)
        bot_lay.setContentsMargins(0, 0, 0, 0)

        model_grp = QGroupBox("Model Training")
        model_outer = QVBoxLayout(model_grp)

        # Model type grid with annotations
        model_type_lbl = QLabel("Model type:")
        model_type_lbl.setStyleSheet("color:#94a3b8;font-size:10px;")
        model_outer.addWidget(model_type_lbl)

        model_type_row = QHBoxLayout()
        self.model_type_combo = QComboBox()

        _MODEL_ANNOTATIONS = [
            ("SVM", "Fast, linear/RBF. Great baseline for small datasets."),
            ("Random Forest", "Robust, handles noise well. Good for multi-class."),
            ("LDA", "Very fast, works well on linearly separable EMG data."),
            ("CatBoost", "Powerful gradient boosting. Often best on tabular features."),
            ("MLP", "Neural network. Best when you have 10k+ windows."),
            ("CNN", "Learns spatial EMG patterns. Requires raw windows."),
            ("AttentionNet", "Transformer attention. Strongest for temporal patterns."),
            ("MSTNet", "Multi-scale temporal net. State-of-the-art for HD-EMG."),
        ]
        for name, _ in _MODEL_ANNOTATIONS:
            self.model_type_combo.addItem(name)

        self._model_hint_lbl = QLabel("")
        self._model_hint_lbl.setStyleSheet(
            "color:#94a3b8;font-size:10px;font-style:italic;")
        self._model_hint_lbl.setWordWrap(True)

        def _update_model_hint(idx):
            if 0 <= idx < len(_MODEL_ANNOTATIONS):
                self._model_hint_lbl.setText(_MODEL_ANNOTATIONS[idx][1])

        self.model_type_combo.currentIndexChanged.connect(_update_model_hint)
        _update_model_hint(0)

        model_type_row.addWidget(self.model_type_combo, stretch=1)
        model_outer.addLayout(model_type_row)
        model_outer.addWidget(self._model_hint_lbl)

        # Train buttons
        train_btn_row = QHBoxLayout()

        self.train_btn = QPushButton("▶ Quick Train (80/20 split)")
        self.train_btn.setFixedHeight(36)
        self.train_btn.setStyleSheet(
            "QPushButton{background:#1a2e4a;color:#06b6d4;"
            "border:1px solid #06b6d4;border-radius:6px;"
            "font-weight:700;font-size:12px;padding:6px 18px;}"
            "QPushButton:hover{background:#06b6d4;color:#fff;}"
            "QPushButton:disabled{color:#4b5563;border-color:#2d2d40;background:#1e1e2e;}")
        self.train_btn.setShortcut("Ctrl+T")
        self.train_btn.setToolTip("Train on selected dataset with default 80/20 split (Ctrl+T)")
        self.train_btn.clicked.connect(self._on_train_model)
        train_btn_row.addWidget(self.train_btn)

        adv_train_btn = QPushButton("Advanced Training...")
        adv_train_btn.setFixedHeight(36)
        adv_train_btn.setToolTip("Open advanced training dialog with full CV, augmentation, etc.")
        adv_train_btn.clicked.connect(self._on_advanced_training)
        train_btn_row.addWidget(adv_train_btn)

        # Quattrocento training is no longer hidden in the Tools menu —
        # it lives here as a first-class evaluation entry point alongside
        # the Muovi-based local training.
        q4_train_btn = QPushButton("Quattrocento Training…")
        q4_train_btn.setFixedHeight(36)
        q4_train_btn.setToolTip(
            "Open the Quattrocento offline training dialog: load .otb+/.csv\n"
            "recordings, run k-fold or LOSO cross-validation, and compare\n"
            "model architectures on the same evaluation harness as Muovi data."
        )
        q4_train_btn.setStyleSheet(
            "QPushButton{background:#2a1a3a;color:#a78bfa;"
            "border:1px solid #a78bfa;border-radius:6px;"
            "font-weight:600;padding:4px 12px;}"
            "QPushButton:hover{background:#a78bfa;color:#fff;}")
        q4_train_btn.clicked.connect(self._on_quattrocento_training)
        train_btn_row.addWidget(q4_train_btn)

        model_outer.addLayout(train_btn_row)
        bot_lay.addWidget(model_grp)

        # ── Trained models ────────────────────────────────────────────────────
        saved_grp = QGroupBox("Saved Models")
        saved_lay = QVBoxLayout(saved_grp)

        models_toolbar = QHBoxLayout()
        self.refresh_models_btn = QPushButton("Refresh")
        self.refresh_models_btn.setFixedHeight(26)
        self.refresh_models_btn.clicked.connect(self._refresh_models)
        models_toolbar.addWidget(self.refresh_models_btn)

        self.delete_model_btn = QPushButton("Delete Selected")
        self.delete_model_btn.setFixedHeight(26)
        self.delete_model_btn.setStyleSheet(
            "QPushButton{color:#ef4444;border-color:#ef4444;}"
            "QPushButton:hover{background:#ef4444;color:#fff;}")
        self.delete_model_btn.clicked.connect(self._on_delete_model)
        models_toolbar.addWidget(self.delete_model_btn)
        models_toolbar.addStretch()
        saved_lay.addLayout(models_toolbar)

        self.models_list = QListWidget()
        self.models_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.models_list.setAlternatingRowColors(True)
        self.models_list.setMinimumHeight(140)
        self.models_list.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        saved_lay.addWidget(self.models_list)

        bot_lay.addWidget(saved_grp)
        splitter.addWidget(bot_widget)
        splitter.setSizes([280, 420])

        scroll_content.setMinimumHeight(720)
        scroll.setWidget(scroll_content)

        return tab

    def _on_dataset_selection_changed(self, current, previous):
        """Update dataset info label when selection changes."""
        if current is None:
            self._ds_info_lbl.setText("Select a dataset to see info")
            return
        name = current.text()
        try:
            meta = self.data_manager.load_dataset(name).get("metadata", {})
            n_samples = meta.get("num_samples", "?")
            n_classes = meta.get("num_classes", "?")
            n_channels = meta.get("num_channels", "?")
            sr = meta.get("sampling_rate", "?")
            ws_ms = meta.get("window_size_ms", "?")
            label_names = meta.get("label_names", {})
            classes_str = ", ".join(label_names.values()) if label_names else str(n_classes)
            self._ds_info_lbl.setText(
                f"{n_samples:,} windows  ·  {n_classes} classes  ·  "
                f"{n_channels}ch  ·  {sr}Hz  ·  {ws_ms}ms  |  {classes_str}"
            )
        except Exception:
            self._ds_info_lbl.setText(name)

    def _create_prediction_tab(self) -> QWidget:
        """Create the real-time prediction tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        layout.addWidget(_make_step_header(
            "Step 3 — Predict",
            "Load a trained model and run live inference. Optionally start the Unity TCP "
            "server to stream predictions to a game.",
        ))

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

        # ── Play Game with Model ─────────────────────────────────────────
        # One-click "load a model and play" experience. Mirrors the
        # Training Game panel on the Record tab (which uses RMS-based
        # easy mode for kids who don't have a model yet) but here the
        # game is driven by the *real* model output via the prediction
        # server. Everything the user needs is in this single group:
        #   • Locate Unity  — same launcher as the Record tab
        #   • Launch Game   — starts the TCP server + spawns Unity
        #   • Stop Game     — closes Unity, leaves the server running so
        #                     the user can keep iterating without
        #                     redoing every step
        play_group = QGroupBox("Play game with model")
        play_layout = QFormLayout(play_group)

        play_info = QLabel(
            "Launch the PlayAgain Unity game and let your trained model "
            "control it. One click starts the TCP server and opens the "
            "game — Unity will connect and receive live gesture "
            "predictions over the network."
        )
        play_info.setStyleSheet(
            "color: #555; font-size: 10px; background: #f5f5f5; "
            "padding: 4px; border-radius: 3px;"
        )
        play_info.setWordWrap(True)
        play_layout.addRow(play_info)

        # Unity build path picker — shares state with the Record tab via
        # the same UnityLauncher instance, so the user only ever has to
        # locate Unity once per machine.
        predict_unity_row = QHBoxLayout()
        self.predict_unity_path_label = QLabel()
        self.predict_unity_path_label.setStyleSheet("color: #475569; font-size: 10px;")
        predict_unity_row.addWidget(self.predict_unity_path_label, 1)

        self.predict_pick_unity_btn = QPushButton("Locate Unity…")
        self.predict_pick_unity_btn.clicked.connect(self._on_pick_unity_exe)
        predict_unity_row.addWidget(self.predict_pick_unity_btn)
        play_layout.addRow("Unity build:", predict_unity_row)

        play_btn_row = QHBoxLayout()
        self.start_play_game_btn = QPushButton("▶  Launch Game with Model")
        self.start_play_game_btn.setFixedHeight(36)
        # Disabled by default — only safe to click once a model is loaded.
        # _on_load_model() flips this on.
        self.start_play_game_btn.setEnabled(False)
        self.start_play_game_btn.setStyleSheet(
            "QPushButton {"
            "  background: #16a34a; color: white; border: none;"
            "  border-radius: 4px; font-weight: 600;"
            "}"
            "QPushButton:hover { background: #15803d; }"
            "QPushButton:disabled { background: #cbd5e1; color: #64748b; }"
        )
        self.start_play_game_btn.clicked.connect(self._on_launch_play_with_model)
        play_btn_row.addWidget(self.start_play_game_btn)

        self.stop_play_game_btn = QPushButton("■  Stop Game")
        self.stop_play_game_btn.setFixedHeight(36)
        self.stop_play_game_btn.setEnabled(False)
        self.stop_play_game_btn.clicked.connect(self._on_stop_play_with_model)
        play_btn_row.addWidget(self.stop_play_game_btn)
        play_layout.addRow(play_btn_row)

        self.play_game_status_label = QLabel("Not running.")
        self.play_game_status_label.setStyleSheet(
            "color: #475569; font-size: 11px; padding: 4px;"
        )
        self.play_game_status_label.setWordWrap(True)
        play_layout.addRow(self.play_game_status_label)

        scroll_layout.addWidget(play_group)

        # Initialise the path label from QSettings so it shows the right
        # state on first display (this tab is built before the user has
        # had a chance to click "Locate Unity…" on the Record tab).
        self._refresh_unity_path_label()

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

        q_train_action = QAction("Quattrocento Training…", self)
        q_train_action.triggered.connect(self._on_quattrocento_training)
        tools_menu.addAction(q_train_action)

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

        # Initialise the workflow stepper now that the tabs exist.
        self.workflow_stepper.set_current(STEP_RECORD)
        # If models / datasets already exist on disk, reflect that in the
        # stepper at startup so returning users see their progress.
        try:
            if self.data_manager.list_datasets():
                self.workflow_stepper.mark_done(STEP_RECORD)
            if any(self.data_dir.glob("models/*/metadata.json")):
                self.workflow_stepper.mark_done(STEP_TRAIN)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Workflow-stepper helpers
    # ------------------------------------------------------------------

    @Slot(int)
    def _on_workflow_step_clicked(self, step_idx: int):
        """Map a stepper click to the corresponding tab."""
        mapping = {
            STEP_RECORD:  TAB_RECORD,
            STEP_TRAIN:   TAB_TRAIN,
            STEP_PREDICT: TAB_PREDICT,
        }
        tab_idx = mapping.get(step_idx)
        if tab_idx is not None:
            self.mode_tabs.setCurrentIndex(tab_idx)

    @Slot(int)
    def _on_mode_tab_changed(self, tab_idx: int):
        """Sync the stepper highlight to whichever main tab is active."""
        reverse = {
            TAB_RECORD:  STEP_RECORD,
            TAB_TRAIN:   STEP_TRAIN,
            TAB_PREDICT: STEP_PREDICT,
        }
        step = reverse.get(tab_idx)
        if step is not None:
            self.workflow_stepper.set_current(step)

    def _stepper_mark(self, step_idx: int):
        """Convenience wrapper used by recording/training/prediction handlers."""
        try:
            self.workflow_stepper.mark_done(step_idx)
        except Exception:
            pass

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
        """Add message to log with color coding and always mirror to stdout."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        console_line = f"[{timestamp}] {message}"
        print(console_line, flush=True)

        # Keep a short transient cue in the status bar even in v2, where
        # the legacy QTextEdit log panel is not always visible.
        try:
            sb = self.statusBar()
            if sb is not None:
                sb.showMessage(message, 5000)
        except Exception:
            pass

        try:
            # Color-code by content keywords for quick visual scanning
            if any(k in message.lower() for k in ("error", "failed", "critical")):
                color = "#f48771"  # red-ish
            elif any(k in message.lower() for k in ("warning", "mismatch", "skipped")):
                color = "#e5c07b"  # amber
            elif any(k in message.lower() for k in ("complete", "success", "loaded", "saved", "started", "connected")):
                color = "#98c379"  # green
            else:
                color = "#d4d4d4"  # default light grey
            ts_html = f'<span style="color:#858585;">[{timestamp}]</span>'
            msg_html = f'<span style="color:{color};">{message}</span>'
            if hasattr(self, "log_text") and self.log_text is not None:
                self.log_text.append(f"{ts_html} {msg_html}")
                self.log_text.verticalScrollBar().setValue(
                    self.log_text.verticalScrollBar().maximum()
                )
        except RuntimeError as e:
            if "already deleted" in str(e):
                pass
            else:
                raise

    def closeEvent(self, event):
        """Handle window close."""
        # Stop game recording if active
        if self._game_recorder and self._game_recorder.is_recording:
            self._game_recorder.stop_recording()
            self._game_recorder = None

        # Stop play-with-model session if a Unity child process is still
        # alive — orphan processes are confusing and consume the audio
        # device.
        play_proc = getattr(self, "_play_unity_process", None)
        if play_proc is not None:
            try:
                self._unity_launcher().terminate(play_proc)
            except Exception:
                pass
            self._play_unity_process = None
        play_timer = getattr(self, "_play_status_timer", None)
        if play_timer is not None:
            try:
                play_timer.stop()
            except Exception:
                pass

        # Stop training-game Unity process the same way.
        train_proc = getattr(self, "_unity_process", None)
        if train_proc is not None:
            try:
                self._unity_launcher().terminate(train_proc)
            except Exception:
                pass
            self._unity_process = None

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
        """Update status bar with device, recording, and prediction state."""
        device = self.device_manager.device

        # Recording indicator
        is_recording = (self._current_session is not None and
                        self._current_session.is_recording)
        if is_recording:
            # Blink the indicator
            if not hasattr(self, "_rec_blink"):
                self._rec_blink = True
            self._rec_blink = not self._rec_blink
            blink = "🔴" if self._rec_blink else "⭕"
            rec_txt = f" {blink} RECORDING"
        else:
            rec_txt = ""

        if device and device.is_connected:
            streaming = device.is_streaming
            status = "Streaming" if streaming else "Connected"
            self.device_status_label.setText(
                f"Device: {status}  "
                f"{device.num_channels}ch @ {device.sampling_rate}Hz"
                f"{rec_txt}"
            )
        else:
            self.device_status_label.setText(f"Device: Not connected{rec_txt}")

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
    def _on_quattrocento_training(self):
        from playagain_pipeline.gui.widgets.quattrocento_training_dialog import (
            QuattrocentoTrainingDialog,
        )
        ui_theme = getattr(getattr(self, "config", None), "ui_theme", "bright")
        dialog = QuattrocentoTrainingDialog(
            model_manager=self.model_manager,
            data_dir=self.data_dir,
            parent=self,
            theme=ui_theme,
        )
        dialog.exec()
        self._refresh_models()

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
        elif device_text == "Quattrocento":
            # Ensure no previous device stream is still running.
            self.device_manager.stop_and_disconnect()
            # Quattrocento streaming: pick one NPY recording file.
            q4_file, _ = QFileDialog.getOpenFileName(
                self,
                "Select Quattrocento NPY recording",
                str(self.data_dir / "quattrocento"),
                "NumPy files (*.npy)",
            )
            if not q4_file:
                return
            self._start_quattrocento_stream(q4_file)
            return
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
                self._awaiting_muovi_handshake = True

                # Update UI to 'Waiting' state
                self.connect_btn.setEnabled(False)
                self.connect_btn.setText("Waiting for Device...")
                self.disconnect_btn.setEnabled(True)  # Allow user to cancel the server

                # Start the TCP/IP Listening Server
                device.connect_device()

                # If no handshake arrives, surface a clear diagnostic and
                # reset the button state so the user can retry.
                QTimer.singleShot(self._MUOVI_CONNECT_TIMEOUT_MS, self._on_muovi_connect_timeout)

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
            self._awaiting_muovi_handshake = False
            self._log("Muovi Handshake Successful!")
            self.connect_btn.setText("Connect")
            self.connect_btn.setEnabled(False)
            self.disconnect_btn.setEnabled(True)

            # Start streaming immediately after connection signal
            self.device_manager.device.start_streaming()
        else:
            self._awaiting_muovi_handshake = False
            self._log("Device Disconnected.")
            self.connect_btn.setEnabled(True)
            self.disconnect_btn.setEnabled(False)

    @Slot(str)
    def _on_device_error(self, error: str):
        """Handle device errors."""
        self._awaiting_muovi_handshake = False
        self._log(f"Device error: {error}")
        self.connect_btn.setEnabled(True)
        self.connect_btn.setText("Connect")
        self.disconnect_btn.setEnabled(False)

    @Slot()
    def _on_muovi_connect_timeout(self):
        """Warn if Muovi never completed its handshake after connect_device()."""
        if not self._awaiting_muovi_handshake:
            return
        dev = self.device_manager.device
        if dev is not None and dev.is_connected:
            self._awaiting_muovi_handshake = False
            return

        self._awaiting_muovi_handshake = False
        self._log(
            "Muovi handshake timeout. Check that the bracelet is powered on, "
            "Wi-Fi/network is reachable, and port 54321 is not already in use."
        )
        self.connect_btn.setEnabled(True)
        self.connect_btn.setText("Connect")
        self.disconnect_btn.setEnabled(False)

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
        """Disconnect from device (Muovi/Synthetic or Quattrocento stream)."""
        self._awaiting_muovi_handshake = False
        # Stop Quattrocento stream if active
        if hasattr(self, '_q4_worker') and self._q4_worker is not None:
            self._q4_worker.stop()
            self._q4_worker.wait(2000)
            self._q4_worker = None
            self._q4_stream_active = False
            self._q4_stream_channels = None
            self._log("Quattrocento stream stopped")
        self.device_manager.stop_and_disconnect()
        self.connect_btn.setEnabled(True)
        self.disconnect_btn.setEnabled(False)
        self._log("Disconnected from device")

    @Slot(np.ndarray)
    def _on_data_received(self, data: np.ndarray):
        """Handle incoming EMG data."""
        try:
            bad_channels = self._get_excluded_channels()

            # Always show ALL channels in the plot (including bad ones) so the
            # checkbox state is not reset by reconfiguring the plot widget.
            if self._plot_widget and self._plot_widget.isVisible():
                if data.shape[1] != self._plot_widget.num_channels:
                    old_ch = self._plot_widget.num_channels
                    self._plot_widget.set_num_channels(data.shape[1])
                    self._log(f"Plot display reconfigured: {old_ch} -> {data.shape[1]} channels")
                self._plot_widget.update_data(data)
        except RuntimeError as e:
            if "already deleted" in str(e):
                pass
            else:
                raise

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

        # Feed the training-game coordinator, if one is running. Uses
        # the *same* calibrated data the prediction server sees so the
        # RMS trigger stays consistent with whatever the model would
        # later see. No-op when no game session is active.
        coord = getattr(self, "_training_coordinator", None)
        if coord is not None and coord.is_running:
            coord.on_emg_data(calibrated_data)

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

    # ------------------------------------------------------------------
    # Training-game (Unity) handlers
    # ------------------------------------------------------------------

    def _unity_launcher(self) -> UnityLauncher:
        """Lazy accessor — one launcher per MainWindow."""
        if not hasattr(self, "_unity_launcher_inst") or self._unity_launcher_inst is None:
            self._unity_launcher_inst = UnityLauncher()
        return self._unity_launcher_inst

    def _refresh_unity_path_label(self) -> None:
        """Update the 'Unity build:' hint under the button group(s).

        Both the Record tab (training-game launcher) and the Predict tab
        (play-with-model launcher) show this path. We update every label
        we can find so the user never sees a stale value after picking a
        new build from either tab.
        """
        path = self._unity_launcher().saved_path()
        if path is None:
            text = "Not set — click 'Locate Unity…'"
            style = "color: #b45309; font-size: 10px;"
        else:
            text = str(path)
            style = "color: #475569; font-size: 10px;"
        for attr in ("unity_path_label", "predict_unity_path_label"):
            label = getattr(self, attr, None)
            if label is not None:
                try:
                    label.setText(text)
                    label.setStyleSheet(style)
                except RuntimeError:
                    # Widget was deleted (tab rebuild) — skip silently.
                    pass

    @Slot()
    def _on_pick_unity_exe(self) -> None:
        """Open a file picker for the Unity build and persist the choice."""
        launcher = self._unity_launcher()
        start_dir = str(launcher.saved_path().parent) if launcher.saved_path() else str(Path.home())
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Locate PlayAgain Unity executable",
            start_dir,
            launcher.file_dialog_filter(),
        )
        if not path:
            return
        try:
            launcher.remember_path(path)
            self._refresh_unity_path_label()
            self._log(f"Unity build remembered: {path}")
        except FileNotFoundError as e:
            QMessageBox.warning(self, "Invalid path", str(e))

    @Slot()
    def _on_start_training_game(self) -> None:
        """
        Wire up the training game end-to-end:
          1. Start (or reuse) a PredictionServer so Unity has something
             to connect to on port 5555.
          2. Launch the Unity executable.
          3. Build a trial schedule from the current gesture set.
          4. Construct a TrainingGameCoordinator and start it.
          5. Begin a RecordingSession so the data is saved.
        """
        # Guard: Unity exe configured?
        launcher = self._unity_launcher()
        if not launcher.has_saved_path():
            QMessageBox.information(
                self, "Unity path not set",
                "Please click 'Locate Unity…' and select your PlayAgain "
                "executable first.",
            )
            return

        # Guard: device connected?
        device = getattr(self.device_manager, "device", None)
        if device is None or not getattr(device, "is_connected", False):
            QMessageBox.warning(
                self, "No device connected",
                "Connect an EMG device (or the Synthetic device) before "
                "launching the training game — the game needs live data "
                "to trigger easy-mode feedings.",
            )
            return

        # ── 1. Prediction server ─────────────────────────────────────
        # Reuse the existing one if the user already has it running
        # (e.g. for live prediction); otherwise spin up a fresh one.
        server = getattr(self, "_prediction_server", None)
        if server is None:
            server = PredictionServer(host="127.0.0.1", port=5555)
            self._prediction_server = server
        if not server.is_running:
            server.start()
            self._log("Prediction server started on 127.0.0.1:5555 for Unity bridge.")

        # ── 2. Create (but do NOT start) a recording session ───────
        # The old flow called ``self._on_start_recording()`` here, which
        # began writing CSV samples immediately. But Unity needs a
        # few seconds to boot, then the child clicks through "Fuchs
        # Abenteuer" → Start — meaning the first ~5–10 seconds of every
        # CSV were recorded against the main menu, not gameplay.
        #
        # New flow: we prepare the session but leave it idle. The
        # coordinator will call ``session.start_recording()`` itself
        # the moment Unity sends ``game_level_started`` (see
        # TrainingGameCoordinator._begin_running). If the Unity build
        # is older and doesn't send that message, the coordinator's
        # 60-second fallback kicks in and starts the session anyway.
        if self._current_session is None:
            try:
                self._prepare_training_game_session()
            except Exception as e:
                QMessageBox.warning(self, "Could not prepare session",
                                    f"{e}\n\nPlay the game without saving?")

        session = self._current_session
        reps = int(self.game_reps_spin.value())
        if session is None:
            self._log("Training game running in preview mode — no session bound.")
            gestures_for_balance: list[str] = ["fist", "pinch", "tripod"]
        else:
            gestures_for_balance = [
                g.name for g in session.gesture_set.gestures
                if g.name and g.name.lower() != "rest"
            ]

        if not gestures_for_balance:
            QMessageBox.warning(self, "Empty schedule",
                                "The current gesture set has no non-rest gestures.")
            return

        # Expected total = exactly N reps × M gestures. Balanced mode
        # may run a few extra trials if Unity ignores the cued gesture
        # for a slot, but it stops as soon as every gesture has hit N
        # successful trials — which is the user-visible guarantee here.
        expected_total = len(gestures_for_balance) * reps

        # ── 3. Coordinator ──────────────────────────────────────────
        coord = TrainingGameCoordinator(prediction_server=server, parent=self)
        coord.set_trigger_ratio(float(self.game_easy_ratio.value()))
        # Balanced mode: the coordinator counts completions per gesture
        # (using Unity's own ``gesture_requested`` field — the gesture
        # the child actually saw) and keeps cuing the most under-
        # represented one until each target is met. Replaces the
        # static-schedule path used previously, which produced
        # imbalanced datasets whenever Unity didn't follow our cues.
        coord.set_balanced_mode(
            gestures=gestures_for_balance,
            reps_per_gesture=reps,
            hold_seconds=3.0,
            rest_seconds=2.0,
        )
        # take_ownership=True hands the session lifecycle to the
        # coordinator: it calls start_recording() when Unity's Level 1
        # signal arrives and stop_recording() on teardown.
        coord.bind_session(session, take_ownership=True)

        coord.trial_started.connect(self._on_game_trial_started)
        coord.trial_completed.connect(self._on_game_trial_completed)
        coord.trigger_fired.connect(self._on_game_trigger_fired)
        coord.all_complete.connect(self._on_game_all_complete)
        coord.rms_updated.connect(self._on_game_rms_updated)
        coord.state_changed.connect(self._on_game_state_changed)
        coord.game_level_started.connect(self._on_game_level_started)
        # New: per-gesture progress feed, used by _on_game_balance_progress
        # to render "fist 2/3, pinch 1/3, tripod 0/3" in the status label
        # so the user can see exactly why the game is still running.
        coord.balance_progress.connect(self._on_game_balance_progress)

        # Forget any progress from a previous run so the completion
        # summary doesn't reuse stale numbers if balance_progress hasn't
        # had time to fire yet.
        self._last_balance_progress = None

        self._training_coordinator = coord

        # ── 3b. Game Protocol Popup ────────────────────────────────
        # Floating window that mirrors the cue currently shown in
        # Unity AND exposes manual debug controls (Force Feed, Skip,
        # per-gesture buttons, synthetic-replay picker). Especially
        # important when testing without a real EMG device — the
        # popup's Force Feed button is the escape hatch when
        # easy-mode RMS doesn't trigger.
        #
        # We wrap creation in try/except so that an unexpected popup
        # error (e.g. Qt platform plugin issue) doesn't silently
        # swallow the rest of the training-game flow — without this
        # guard a popup failure would also block coord.start() and
        # the whole feature would appear broken with no log message.
        popup = None
        try:
            popup = GameProtocolPopup(parent=self)
            if session is not None:
                popup.set_gesture_set(session.gesture_set)
            popup.set_total_trials(expected_total)

            coord.trial_started.connect(popup.set_current_trial)
            coord.trial_completed.connect(lambda *_a: popup.clear_current_trial())
            coord.state_changed.connect(popup.set_state)
            coord.rms_updated.connect(popup.set_rms)
            coord.all_complete.connect(lambda: popup.set_state("complete"))

            popup.force_feed_requested.connect(coord.force_trigger)
            popup.skip_requested.connect(coord.skip_current_trial)
            popup.manual_gesture_requested.connect(coord.fire_manual_gesture)
            popup.replay_session_requested.connect(self._on_replay_session_picked)

            # Optional: replay-session dropdown for synthetic devices.
            device = getattr(self.device_manager, "device", None)
            if device is not None and "synthetic" in (
                getattr(device, "name", "") or ""
            ).lower():
                try:
                    from playagain_pipeline.core.data_manager import DataManager
                    dm = DataManager()
                    if hasattr(dm, "list_session_paths"):
                        replay_paths = dm.list_session_paths()
                    else:
                        sessions_dir = Path(getattr(dm, "sessions_dir",
                                                    "data/sessions"))
                        replay_paths = (
                            [p.parent for p in sessions_dir.rglob("metadata.json")]
                            if sessions_dir.exists() else []
                        )
                    popup.set_replay_options(list(replay_paths))
                except Exception:
                    self._log("Replay-picker discovery failed — proceeding without it.")

            popup.show()
            popup.raise_()
            popup.activateWindow()
            self._game_protocol_popup = popup
            self._log("Game protocol popup opened.")
        except Exception as e:
            self._log(f"Failed to open game protocol popup: {e!r}")
            import traceback
            traceback.print_exc()
            self._game_protocol_popup = None
            # Continue anyway — the game can still be played without
            # the popup, just with no manual controls.

        # ── 4. Launch Unity ─────────────────────────────────────────
        try:
            proc = launcher.launch(extra_args=["-screen-fullscreen", "0"])
            self._unity_process = proc
            self._log(f"Unity launched (PID {proc.pid}).")
        except UnityNotFoundError as e:
            QMessageBox.warning(self, "Unity not found", str(e))
            coord.stop()
            self._training_coordinator = None
            if popup is not None:
                popup.hide()
                popup.deleteLater()
                self._game_protocol_popup = None
            return

        # ── 5. Arm the coordinator (waits for Unity Level 1) ────────
        coord.start(wait_for_unity=True)
        self.start_game_btn.setEnabled(False)
        self.stop_game_btn.setEnabled(True)
        self.game_status_label.setText(
            f"Waiting for Unity… {expected_total} trials queued "
            f"({reps}× per gesture, balanced). Session starts when the "
            f"child hits Start in the game."
        )

    @Slot()
    def _on_stop_training_game(self) -> None:
        """Tear the game down in reverse order of startup."""
        coord = getattr(self, "_training_coordinator", None)
        if coord is not None:
            try:
                coord.stop()
            except Exception:
                pass
            self._training_coordinator = None

        # Close the protocol popup if it's still around. Its closeEvent
        # is overridden to just hide, so deleteLater is what actually
        # cleans the widget up.
        popup = getattr(self, "_game_protocol_popup", None)
        if popup is not None:
            try:
                popup.hide()
                popup.deleteLater()
            except Exception:
                pass
            self._game_protocol_popup = None

        # Politely ask Unity to quit — it can also be left open if the
        # clinician wants to inspect the final animal count.
        proc = getattr(self, "_unity_process", None)
        if proc is not None:
            try:
                self._unity_launcher().terminate(proc)
            except Exception:
                pass
            self._unity_process = None

        # If the session is still recording (coordinator didn't own it,
        # or the user hit Stop before Level 1 arrived), stop it now.
        if self._current_session is not None and self._current_session.is_recording:
            self._on_stop_recording()
        elif self._current_session is not None and not self._current_session.is_recording:
            # Coordinator already called stop_recording() internally —
            # but it never saves to disk. Do that now so early exits
            # don't silently discard collected data.
            try:
                if self._current_session.total_samples > 0:
                    # Auto-detect rotation before saving (mirrors _on_stop_recording)
                    try:
                        cal_result = self.calibrator.detect_session_rotation(
                            self._current_session, save_to_metadata=True
                        )
                        if cal_result is not None:
                            self._log(
                                f"Auto-detected rotation: {cal_result.rotation_offset} ch "
                                f"(confidence: {cal_result.confidence:.0%})"
                            )
                            if not self.calibrator.has_reference:
                                self.calibrator.save_as_reference(cal_result)
                    except Exception as e:
                        self._log(f"Rotation detection failed: {e}")
                    path = self.data_manager.save_session(self._current_session)
                    self._log(f"Training-game session saved to {path}")
                    self._stepper_mark(STEP_RECORD)
                else:
                    self._log("Training-game session had no data — not saved.")
            except Exception as save_err:
                self._log(f"Error saving training-game session: {save_err}")
            finally:
                self._current_session = None

        self.start_game_btn.setEnabled(True)
        self.stop_game_btn.setEnabled(False)
        self.game_status_label.setText("Stopped.")
        self._log("Training game stopped.")

    # ------------------------------------------------------------------
    # Play-with-model (Predict tab) handlers
    # ------------------------------------------------------------------
    #
    # The Record-tab "Launch Training Game" above runs Unity in
    # **easy mode**: an RMS detector fakes confidence-1.0 predictions so
    # children can collect their first dataset before any model exists.
    #
    # The handlers below are the **mirror image** for the Predict tab:
    # the user has a trained model, the prediction server runs real
    # inference on the live EMG, and Unity is just the playable
    # frontend. No coordinator, no popup — the game plays freely from
    # the model's output.
    # ------------------------------------------------------------------

    @Slot()
    def _on_launch_play_with_model(self) -> None:
        """One-click: start the prediction server with the loaded model
        and launch Unity. The user does not need to touch the standalone
        "Start Server" button — we handle it for them.
        """
        # 1. Model required.
        if not self._current_model:
            QMessageBox.warning(
                self, "No model loaded",
                "Please load a trained model first (use the 'Load Model' "
                "button in the Model section above)."
            )
            return

        # 2. Live EMG required — predictions need data flowing.
        device = getattr(self.device_manager, "device", None)
        stream_ready = bool(device and device.is_streaming) or bool(self._q4_stream_active)
        if not stream_ready:
            QMessageBox.warning(
                self, "No data stream",
                "Connect an EMG device (or start a Quattrocento file "
                "stream) on the Record tab first — the model needs live "
                "EMG to make predictions."
            )
            return

        # 3. Unity build configured?
        launcher = self._unity_launcher()
        if not launcher.has_saved_path():
            QMessageBox.information(
                self, "Unity path not set",
                "Please click 'Locate Unity…' and select your PlayAgain "
                "executable first."
            )
            return

        # 4. Smoke-test the model on the current buffer so a channel
        #    mismatch is caught BEFORE we spawn Unity. Saves the user
        #    from a "the game opened but nothing happens" experience.
        if self._prediction_buffer is None:
            # _on_load_model didn't run — guard anyway.
            QMessageBox.warning(
                self, "Buffer not initialised",
                "Re-load the model so the prediction buffer is set up."
            )
            return
        try:
            X_test = self._prediction_buffer[np.newaxis, :, :]
            _ = self._current_model.predict_proba(X_test)
        except Exception as e:
            md = self._current_model.metadata
            QMessageBox.critical(
                self, "Model cannot predict on this stream",
                f"The loaded model failed a quick test against the "
                f"current buffer.\n\n"
                f"Buffer:        {self._prediction_buffer.shape}\n"
                f"Model expects: {md.num_channels} ch @ {md.sampling_rate} Hz\n\n"
                f"Underlying error:\n{e}\n\n"
                f"Most common cause: the device's channel count does "
                f"not match the channel count the model was trained on."
            )
            self._log(f"Play-with-model smoke-test failed: {e}")
            return

        # 5. Start (or reuse) the TCP server with the loaded model.
        #    If the user already started it manually, don't disturb it.
        server = self._prediction_server
        if server is None or not server.is_running:
            try:
                # Delegate to the existing handler so smoothing settings,
                # callbacks, and buttons stay consistent with the manual
                # path. _on_start_server picks up the model + UI state
                # and flips the Start/Stop server buttons for us.
                self._on_start_server()
            except Exception as e:
                QMessageBox.warning(
                    self, "Server failed to start",
                    f"Could not start the TCP prediction server:\n{e}"
                )
                return
            server = self._prediction_server
            if server is None or not server.is_running:
                QMessageBox.warning(
                    self, "Server failed to start",
                    "The TCP prediction server did not come up — check "
                    "the log for details (port may already be in use)."
                )
                return
        else:
            # Server already up — make sure it isn't paused, otherwise
            # Unity will sit there receiving nothing.
            if getattr(server, "is_paused", False):
                server.resume()

        # 6. Launch Unity.
        try:
            proc = launcher.launch(extra_args=["-screen-fullscreen", "0"])
            self._play_unity_process = proc
            self._log(f"Unity launched (PID {proc.pid}) for play-with-model.")
        except UnityNotFoundError as e:
            QMessageBox.warning(self, "Unity not found", str(e))
            return

        # 7. Periodic status update — show whether Unity actually
        #    connected to the server. Without this the user has no way
        #    to tell the difference between "Unity is loading" and
        #    "Unity crashed before connecting".
        if not hasattr(self, "_play_status_timer") or self._play_status_timer is None:
            self._play_status_timer = QTimer(self)
            self._play_status_timer.timeout.connect(self._update_play_status_label)
        self._play_status_timer.start(1000)

        # 8. UI bookkeeping.
        self.start_play_game_btn.setEnabled(False)
        self.stop_play_game_btn.setEnabled(True)
        self.play_game_status_label.setText(
            f"Game launched. Waiting for Unity to connect to the server…"
        )

    @Slot()
    def _on_stop_play_with_model(self) -> None:
        """Stop the play-with-model session. Closes Unity but leaves the
        server running so the user can iterate (load a different model,
        click Launch again) without re-doing every step.
        """
        proc = getattr(self, "_play_unity_process", None)
        if proc is not None:
            try:
                self._unity_launcher().terminate(proc)
            except Exception as e:
                self._log(f"Could not cleanly terminate Unity: {e}")
            self._play_unity_process = None

        timer = getattr(self, "_play_status_timer", None)
        if timer is not None:
            try:
                timer.stop()
            except Exception:
                pass

        self.start_play_game_btn.setEnabled(self._current_model is not None)
        self.stop_play_game_btn.setEnabled(False)
        self.play_game_status_label.setText("Not running.")
        self._log("Play-with-model stopped.")

    @Slot()
    def _update_play_status_label(self) -> None:
        """Tick: refresh 'Game launched / N clients connected' text.

        Runs only while the play-with-model session is live. Stops the
        timer if the Unity process has died externally so the user gets
        a clear "the game crashed" indicator instead of a stuck label.
        """
        proc = getattr(self, "_play_unity_process", None)
        server = self._prediction_server

        # Unity died on its own (user closed the window, crashed, etc.)
        if proc is not None and proc.poll() is not None:
            self._play_unity_process = None
            timer = getattr(self, "_play_status_timer", None)
            if timer is not None:
                timer.stop()
            self.start_play_game_btn.setEnabled(self._current_model is not None)
            self.stop_play_game_btn.setEnabled(False)
            self.play_game_status_label.setText(
                "Game closed. Click 'Launch Game with Model' to play again."
            )
            self._log("Unity exited (play-with-model).")
            return

        if server is not None and server.is_running:
            n = server.client_count
            model_name = getattr(self._current_model, "name", "model")
            if n > 0:
                self.play_game_status_label.setText(
                    f"▶ Playing — {n} Unity client connected. Model: {model_name}."
                )
            else:
                self.play_game_status_label.setText(
                    "Game launched. Waiting for Unity to connect to the server…"
                )

    @Slot(str)
    def _on_replay_session_picked(self, path: str) -> None:
        """
        Forward a synthetic-replay-session pick from the popup to the
        active device. The popup just emits the chosen path; we try a
        few common method names on the device and log which one stuck.
        """
        device = getattr(self.device_manager, "device", None)
        if device is None:
            self._log("Replay request ignored — no device.")
            return
        for method_name in ("load_session", "set_replay_path",
                            "set_session_path", "set_source"):
            fn = getattr(device, method_name, None)
            if callable(fn):
                try:
                    fn(path)
                    self._log(f"Replay loaded ({method_name}): {path}")
                    return
                except Exception as e:
                    self._log(f"Replay load failed via {method_name}: {e}")
                    return
        self._log(
            f"Replay request: {path} — but the synthetic device exposes "
            "no recognised replay-loading method. Add load_session / "
            "set_replay_path / set_session_path / set_source."
        )

    # ── Coordinator callbacks — UI feedback ──────────────────────────

    @Slot(object, int)
    def _on_game_trial_started(self, spec: TrialSpec, index: int) -> None:
        total = len(getattr(self._training_coordinator, "_schedule", []))
        self.game_status_label.setText(
            f"Trial {index + 1}/{total} — target: {spec.gesture_name}"
        )
        self._log(f"[Game] Trial {index + 1}/{total}: {spec.gesture_name}")

    @Slot(object, int)
    def _on_game_trial_completed(self, spec: TrialSpec, index: int) -> None:
        self._log(f"[Game] Trial {index + 1} done ({spec.gesture_name}) ✓")

    @Slot(dict)
    def _on_game_balance_progress(self, progress: dict) -> None:
        """
        Per-gesture progress update from the coordinator's balanced
        mode. ``progress`` is ``{gesture: (completed, target)}``. Render
        a compact "fist 2/3 · pinch 1/3 · tripod 0/3" line beneath the
        current trial label so the user can see exactly why the game
        is still going (or that it's about to finish).
        """
        if not progress:
            return
        # Cache for the completion handler so we can give an honest
        # summary even when Unity's spawn order made full balance
        # impossible (counts will be uneven and the safety cap fires).
        self._last_balance_progress = dict(progress)
        # Sorted alphabetically for stable ordering across updates.
        parts = [
            f"{name} {done}/{target}"
            for name, (done, target) in sorted(progress.items())
        ]
        # Don't overwrite the "Trial X/N — target …" line — append on
        # the next line so both pieces of information are visible.
        existing = self.game_status_label.text().split("\n", 1)[0]
        self.game_status_label.setText(
            f"{existing}\nProgress: " + " · ".join(parts)
        )

    @Slot(str, float)
    def _on_game_trigger_fired(self, gesture: str, rms: float) -> None:
        """Easy-mode trigger fired → the animal is about to be fed."""
        self._log(f"[Game] Easy-mode trigger for {gesture} (RMS={rms:.4f})")

    @Slot()
    def _on_game_all_complete(self) -> None:
        """Schedule finished — stop everything cleanly. The summary is
        honest about whether balance was actually achieved (targets
        may not be hit if Unity's spawn order ignored our cues)."""
        progress = getattr(self, "_last_balance_progress", None)
        if progress:
            unmet = {
                g: (done, target)
                for g, (done, target) in progress.items()
                if done < target
            }
            if not unmet:
                self.game_status_label.setText(
                    "All trials complete — every gesture met its target ✓"
                )
                self._log("[Game] Balanced run complete — all targets met.")
            else:
                detail = ", ".join(
                    f"{g} {done}/{target}" for g, (done, target) in sorted(unmet.items())
                )
                self.game_status_label.setText(
                    f"Stopped — Unity spawn order didn't allow full balance.\n"
                    f"Missing: {detail}.  Consider re-running, or patch "
                    f"Unity to follow target_gesture cues."
                )
                self._log(
                    f"[Game] Balanced run ended without full balance: {detail}"
                )
        else:
            # Strict-schedule path or no progress emitted — fall back
            # to the original, simpler success line.
            self.game_status_label.setText("All trials complete — great job!")
            self._log("[Game] All trials complete.")
        # Delay the teardown slightly so the final animal animation
        # finishes on the Unity side before the process is killed.
        QTimer.singleShot(3000, self._on_stop_training_game)

    @Slot(float, float)
    def _on_game_rms_updated(self, rms: float, baseline: float) -> None:
        """Optional: could drive a live meter here. Currently just a pass."""
        pass

    @Slot(str)
    def _on_game_state_changed(self, state: str) -> None:
        """
        Surface coordinator state transitions in the status label.

        States:
          • waiting_for_unity — Unity launched, child is still in the menu
          • running           — Level 1 is live, cues are firing
          • stopped / idle    — session ended
        """
        if state == "waiting_for_unity":
            self.game_status_label.setText(
                "Waiting for Unity — the session starts when the child "
                "clicks Start in the main menu."
            )
        elif state == "running":
            total = len(getattr(self._training_coordinator, "_schedule", []))
            self.game_status_label.setText(
                f"Playing — {total} trials queued. "
                f"Easy-mode sensitivity: {self.game_easy_ratio.value():.1f}x"
            )

    @Slot()
    def _on_game_level_started(self) -> None:
        """Unity just reached Level 1 — log the transition."""
        self._log("[Game] Unity reached Level 1 — recording session started.")

    # ── Session preparation — no recording starts here! ─────────────

    def _prepare_training_game_session(self) -> None:
        """
        Build a RecordingSession in the 'ready but not recording' state.

        The coordinator will call ``start_recording()`` on it later,
        once Unity signals that Level 1 has begun. Until then the
        session just sits there with its gesture set configured and
        its metadata stamped — no samples are written to disk.

        This is a trimmed-down clone of the relevant parts of
        ``_on_start_recording`` — we deliberately skip the protocol /
        manual-mode branches because the training game is its own
        protocol source (the coordinator's schedule).
        """
        device = getattr(self.device_manager, "device", None)
        if device is None or not getattr(device, "is_connected", False):
            raise RuntimeError(
                "Device must be connected before preparing a training-game session."
            )

        subject_id = (self.subject_id_edit.text() or "VP_01").strip()
        reps = int(self.game_reps_spin.value())
        session_id = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{reps}rep"

        # Use the project's default gesture set — same as normal
        # recordings, so the dataset pipeline treats training-game
        # sessions identically to manually-recorded ones.
        gesture_set = create_default_gesture_set()

        self._current_session = RecordingSession(
            session_id=session_id,
            subject_id=subject_id,
            device_name=getattr(device, "name", self.device_combo.currentText()),
            num_channels=int(self.channels_spin.value()),
            sampling_rate=int(self.sampling_rate_spin.value()),
            gesture_set=gesture_set,
            protocol_name="training_game",
        )
        # Stamp some useful metadata so post-hoc analysis knows this
        # session came from the Unity training game.
        self._current_session.metadata.custom_metadata["source"] = "training_game"
        self._current_session.metadata.custom_metadata["easy_mode_sensitivity"] = \
            float(self.game_easy_ratio.value())
        self._log(
            f"Training-game session prepared ({subject_id}/{session_id}) — "
            "waiting for Unity Level 1 before recording starts."
        )

    # ------------------------------------------------------------------

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
                                                 device_name=device.config.device_type.name,
                                                 num_channels=actual_channels,
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
            self._current_session.metadata.custom_metadata[
                "manual_gesture"] = self.manual_gesture_combo.currentText().strip().lower()

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
            # Recording finished → first workflow step is complete.
            self._stepper_mark(STEP_RECORD)
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

            # Save session (now includes rotation metadata) — do it on main thread
            # (saving is usually fast for recording sizes, but show feedback)
            try:
                path = self.data_manager.save_session(self._current_session)
                self._log(f"Saved session to {path}")
            except Exception as save_err:
                self._log(f"Error saving session: {save_err}")

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
        try:
            self.cal_subject_combo.blockSignals(True)
            self.cal_subject_combo.clear()
            subjects = self.data_manager.list_subjects()
            self.cal_subject_combo.addItems(subjects)
            self.cal_subject_combo.blockSignals(False)

            # Trigger session reload for current subject
            if subjects:
                self._on_cal_subject_changed(self.cal_subject_combo.currentText())
        except RuntimeError as e:
            if "already deleted" in str(e):
                pass
            else:
                raise

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

        self.calibrate_from_session_btn.setEnabled(False)
        self._log(f"Loading session {subject}/{session_id} for calibration...")

        def _do_cal():
            session = self.data_manager.load_session(subject, session_id)
            if not session.trials:
                raise ValueError("Selected session has no recorded trials.")
            self.calibrator.processor.num_channels = session.metadata.num_channels
            result = self.calibrator.calibrate_from_session(session)
            return session, result

        def _cal_done(payload):
            self.calibrate_from_session_btn.setEnabled(True)
            session, result = payload
            try:

                gesture_names = set(t.gesture_name for t in session.get_valid_trials())
                self._log(f"Found gestures: {', '.join(sorted(gesture_names))}")
                self._log(f"Total valid trials: {len(session.get_valid_trials())}")
                self._update_calibration_display()
                self._log(f"Calibration completed from session '{session_id}'")
                self._log(f"  Rotation offset: {result.rotation_offset} channels")
                self._log(f"  Confidence: {result.confidence:.2%}")

                per_gesture = result.metadata.get("per_gesture_confidence", {})
                if per_gesture:
                    for gesture, conf in sorted(per_gesture.items()):
                        self._log(f"  {gesture}: {conf:.2%}")

                incompat = result.metadata.get("reference_incompatible")
                if incompat:
                    self._log(f"  Warning - Reference was incompatible: {incompat}")
                    self._log(f"  Saving this session as the new reference.")
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
                self._log(f"Calibration display error: {e}")

        def _cal_err(tb):
            self.calibrate_from_session_btn.setEnabled(True)
            self._log(f"Calibration error:\n{tb}")
            QMessageBox.critical(self, "Error", f"Calibration failed.\nSee log for details.")

        run_blocking(self, _do_cal, _cal_done, _cal_err,
                     label="Calibrating from session…")

    @Slot()
    def _on_save_reference(self):
        """Save current calibration as reference."""
        if self.calibrator.current_calibration:
            self.calibrator.save_as_reference(self.calibrator.current_calibration)
            self._log("Saved calibration as reference")

    @Slot()
    def _on_save_reference_and_recompute(self):
        """Save current calibration as reference and recompute all session rotations."""
        if not self.calibrator.current_calibration:
            return
        cal_snapshot = self.calibrator.current_calibration
        data_dir_snap = self.data_dir
        self._log("Setting new reference and recomputing all session rotations...")

        def _do_recompute():
            self.calibrator.save_as_reference(
                cal_snapshot,
                recompute_all=True,
                data_dir=data_dir_snap,
            )
            return True

        def _done(_):
            self._log("All session rotations recomputed relative to new reference")
            self._update_calibration_display()

        def _err(tb):
            self._log(f"Error recomputing rotations:\n{tb}")

        run_blocking(self, _do_recompute, _done, _err,
                     label="Recomputing all session rotations…")

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
        self._live_cal_active = False  # will be set True when countdown reaches 0
        self._live_cal_countdown = 3  # 3-second countdown before first gesture

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
            else:  # Session tab
                selected_items = session_list.selectedItems()
                if not selected_items:
                    new_name = "dataset_no_session"
                elif len(selected_items) == 1:
                    # Item text is "subject / session_id", data is (subject, session_id).
                    # Always prefix the subject so the dataset (and the model derived
                    # from it) is identifiable later without having to open metadata.
                    subject, session_id = selected_items[0].data(Qt.ItemDataRole.UserRole)
                    new_name = f"{subject}_{session_id}"
                else:
                    # Multi-session selection: collect distinct subjects so the
                    # generated name still tells the user *whose* data it is.
                    subjects_in_sel = []
                    seen = set()
                    for item in selected_items:
                        subj, _sid = item.data(Qt.ItemDataRole.UserRole)
                        if subj not in seen:
                            seen.add(subj)
                            subjects_in_sel.append(subj)

                    subject_prefix = (
                        subjects_in_sel[0] if len(subjects_in_sel) == 1
                        else f"{len(subjects_in_sel)}subjects"
                    )

                    if len(selected_items) <= 3:
                        names = []
                        for item in selected_items:
                            _, session_id = item.data(Qt.ItemDataRole.UserRole)
                            names.append(session_id)
                        new_name = f"{subject_prefix}_{'_'.join(names)}"
                        # Truncate if too long
                        if len(new_name) > 80:
                            new_name = f"{subject_prefix}_{len(selected_items)}_sessions"
                    else:
                        new_name = f"{subject_prefix}_{len(selected_items)}_sessions"

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

        from PySide6.QtWidgets import QRadioButton, QButtonGroup, QListWidget as QListWidgetD, \
            QListWidgetItem as QListWidgetItemD
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

                ds_name      = name_edit.text()
                ds_sessions  = sessions_to_use
                ds_ws        = window_size_spin.value()
                ds_stride    = stride_spin.value()
                ds_invalid   = include_invalid_cb.isChecked()
                ds_cal       = cal_to_apply
                ds_per_sess  = use_per_session
                ds_feat_cfg  = ds_feature_config
                ds_bad_mode  = bad_channel_mode_combo.currentData()

                def _create():
                    ds = self.data_manager.create_dataset(
                        name=ds_name, sessions=ds_sessions,
                        window_size_ms=ds_ws, window_stride_ms=ds_stride,
                        include_invalid=ds_invalid, calibration=ds_cal,
                        use_per_session_rotation=ds_per_sess,
                        feature_config=ds_feat_cfg, bad_channel_mode=ds_bad_mode,
                    )
                    self.data_manager.save_dataset(ds)
                    return ds

                def _ds_done(ds):
                    self._log(f"Created dataset '{ds_name}' with {ds['metadata']['num_samples']} samples")
                    self._refresh_datasets()

                def _ds_err(tb):
                    self._log(f"Error creating dataset:\n{tb}")
                    from PySide6.QtWidgets import QMessageBox
                    QMessageBox.critical(self, "Error", f"Failed to create dataset.\nSee log for details.")

                run_blocking(self, _create, _ds_done, _ds_err,
                             label=f"Creating dataset '{ds_name}'…")

            except Exception as e:
                self._log(f"Error preparing dataset creation: {e}")
                QMessageBox.critical(self, "Error", f"Failed to create dataset: {e}")

    @Slot()
    def _on_train_model(self):
        """Train a model on selected dataset (non-blocking)."""
        selected = self.dataset_list.currentItem()
        if not selected:
            QMessageBox.warning(self, "Warning", "Please select a dataset")
            return

        dataset_name = selected.text()
        model_type_text = self.model_type_combo.currentText()
        if model_type_text == "AttentionNet":
            model_type = "attention_net"
        elif model_type_text == "MSTNet":
            model_type = "mstnet"
        else:
            model_type = model_type_text.lower().replace(" ", "_")

        safe_dataset_name = dataset_name.replace(":", "-").replace(" ", "_").replace("/", "_")
        timestamp = datetime.now().strftime("%H%M%S")
        model_name = f"{model_type}_{safe_dataset_name}_{timestamp}"

        self._log(f"Loading dataset '{dataset_name}' and starting training ({model_type})...")
        self.train_btn.setEnabled(False)

        def _do_train():
            dataset = self.data_manager.load_dataset(dataset_name)
            model = self.model_manager.create_model(model_type, name=model_name)
            results = self.model_manager.train_model(model, dataset)
            return model, results

        def _train_done(payload):
            self.train_btn.setEnabled(True)
            model, results = payload
            self._log(f"Training complete!  acc={results.get('training_accuracy', 0):.2%}  "
                      f"val={results.get('validation_accuracy', 0):.2%}")
            if 'training_time' in results:
                self._log(f"  Training time: {results['training_time']:.2f}s")
            self._refresh_models()
            self._log(f"Model saved: {model_name}")
            # A trained model now exists → Train step is complete.
            self._stepper_mark(STEP_TRAIN)

        def _train_err(tb):
            self.train_btn.setEnabled(True)
            self._log(f"Training error:\n{tb}")

        run_blocking(self, _do_train, _train_done, _train_err,
                     label=f"Training {model_type}…")

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
            # A model is loaded → Predict step is ready / done.
            self._stepper_mark(STEP_PREDICT)

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
            if self._q4_stream_active and self._q4_stream_channels is not None:
                num_ch = int(self._q4_stream_channels)
            else:
                num_ch = device.num_channels if device else self.channels_spin.value()
            window_samples = int(self._prediction_window_ms * self._current_model.metadata.sampling_rate / 1000)
            self._prediction_buffer = np.zeros((window_samples, num_ch))

            # Warn if channel count differs from model's trained channel count.
            # Reuse the same wording in _on_start_prediction so the user
            # sees the warning at load time AND right before predicting.
            md = self._current_model.metadata
            classes = md.class_names if isinstance(md.class_names, dict) else {}
            self._log(
                f"  Model expects: {md.num_channels} ch @ {md.sampling_rate} Hz, "
                f"{len(classes)} classes ({', '.join(str(v) for v in classes.values())})"
            )
            self._log(
                f"  Prediction buffer initialised: {window_samples} samples × {num_ch} channels"
            )
            model_ch = md.num_channels
            if model_ch > 0 and num_ch != model_ch:
                self._log(
                    f"  ⚠ Channel mismatch: device has {num_ch} ch, model "
                    f"trained on {model_ch} ch — predictions will likely "
                    f"fail until channels match. Reconnect the device or "
                    f"retrain on this device's channel count."
                )

            # Re-enable launch buttons that were greyed out before a model
            # was available — most importantly the "Launch Game with Model"
            # one-click flow on this tab.
            if hasattr(self, "start_play_game_btn"):
                self.start_play_game_btn.setEnabled(True)

        except Exception as e:
            self._log(f"Error loading model: {e}")

    @Slot()
    def _on_start_prediction(self):
        """Start real-time prediction."""
        if not self._current_model:
            QMessageBox.warning(self, "Warning", "Please load a model first")
            return

        device = self.device_manager.device
        stream_ready = bool(device and device.is_streaming) or bool(self._q4_stream_active)
        if not stream_ready:
            QMessageBox.warning(self, "Warning", "Please connect and start device")
            return

        # ── Smoke-test the model BEFORE we start the worker thread ───────
        # The PredictionWorker swallows exceptions on its background loop
        # (it only printed to stdout previously, which the user never
        # sees). If the model can't predict on the current buffer shape
        # — almost always a channel-count mismatch caused by training on
        # one device and predicting on another — we want to fail loudly
        # and explain WHY, not silently leave the label at "No prediction".
        if self._prediction_buffer is None:
            QMessageBox.warning(
                self, "Warning",
                "Prediction buffer is not initialised. Load the model again."
            )
            return
        try:
            X_test = self._prediction_buffer[np.newaxis, :, :]
            _ = self._current_model.predict_proba(X_test)
        except Exception as e:
            md = self._current_model.metadata
            QMessageBox.critical(
                self, "Model cannot predict on this stream",
                f"The loaded model failed a quick test on the current buffer.\n\n"
                f"Buffer shape:   {self._prediction_buffer.shape} (samples × channels)\n"
                f"Model expects:  {md.num_channels} ch @ {md.sampling_rate} Hz\n\n"
                f"Underlying error:\n{e}\n\n"
                f"Most common cause: the device's channel count does not "
                f"match the channel count the model was trained on. "
                f"Reconnect the matching device, or retrain a model on "
                f"this device's data."
            )
            self._log(f"Prediction smoke-test failed: {e}")
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
        # Surface background-thread errors to the user — without this,
        # silent failures leave the label stuck on "No prediction" with
        # no indication why.
        self._prediction_worker.error_occurred.connect(self._on_prediction_worker_error)
        self._prediction_worker.start()

        # Resume prediction server if it was paused
        if self._prediction_server and self._prediction_server.is_running and self._prediction_server.is_paused:
            self._prediction_server.resume()

        self._is_predicting = True
        self.start_pred_btn.setEnabled(False)
        self.stop_pred_btn.setEnabled(True)
        self._log("Started prediction")

    @Slot(str)
    def _on_prediction_worker_error(self, message: str):
        """
        Background worker hit an exception. Show it in the log so the
        user can see WHY predictions stopped flowing instead of staring
        at a frozen label. Throttled by the worker itself.
        """
        self._log(f"Prediction error: {message}")
        # Don't auto-stop the worker — the same exception may be a one-off
        # caused by a transient device hiccup. The user can stop manually.

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
        # Surface server-side prediction errors to the GUI log too. The
        # callback runs on the server's background thread, so we cannot
        # touch widgets directly — _on_server_prediction_error reschedules
        # onto the main thread.
        self._prediction_server.add_error_callback(self._on_server_prediction_error)

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

    def _on_server_prediction_error(self, message: str):
        """
        Callback from PredictionServer (runs on its background thread)
        for prediction-loop errors. Marshals onto the main thread so we
        can safely touch the log widget.
        """
        # QTimer.singleShot(0, ...) is the standard Qt pattern for
        # "do this on the main thread next event loop tick".
        try:
            QTimer.singleShot(
                0, lambda m=message: self._log(f"Server prediction error: {m}")
            )
        except Exception:
            # Last-ditch fallback if Qt is shutting down.
            print(f"[Server] Prediction error: {message}")

    @Slot()
    def _on_stop_server(self):
        """Stop the Unity TCP prediction server."""
        # Stop game recording first if active (it depends on the server)
        if self._game_recorder and self._game_recorder.is_recording:
            self._on_stop_game_recording()

        if self._prediction_server:
            # Remove our GUI prediction callback before stopping
            self._prediction_server.remove_prediction_callback(self._on_server_prediction)
            # Same for the error callback added in _on_start_server.
            try:
                self._prediction_server.remove_error_callback(self._on_server_prediction_error)
            except AttributeError:
                # Older PredictionServer versions don't have the method —
                # safe to ignore, the server is being torn down anyway.
                pass
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
        # In-place shift avoids np.roll allocation each frame (reduces UI lag).
        self._prediction_buffer[:-n_samples] = self._prediction_buffer[n_samples:]
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

    def _apply_app_theme(self):
        """Apply a consistent, modern dark-adaptive theme to the whole app."""
        self.setStyleSheet("""
            QMainWindow, QDialog {
                background: #1e1e2e;
                color: #e2e8f0;
            }
            QTabWidget::pane {
                border: 1px solid #3f3f5c;
                border-radius: 6px;
                background: #2a2a3e;
            }
            QTabBar::tab {
                background: #1e1e2e;
                color: #94a3b8;
                padding: 6px 14px;
                border: 1px solid #3f3f5c;
                border-bottom: none;
                border-top-left-radius: 5px;
                border-top-right-radius: 5px;
                margin-right: 2px;
                font-size: 11px;
            }
            QTabBar::tab:selected {
                background: #2a2a3e;
                color: #06b6d4;
                border-bottom: 2px solid #06b6d4;
            }
            QGroupBox {
                background: #2a2a3e;
                border: 1px solid #3f3f5c;
                border-radius: 7px;
                font-weight: 600;
                color: #e2e8f0;
                padding-top: 14px;
                margin-top: 5px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                color: #06b6d4;
                font-size: 11px;
            }
            QPushButton {
                background: #313145;
                color: #e2e8f0;
                border: 1px solid #3f3f5c;
                border-radius: 5px;
                padding: 5px 12px;
                font-size: 12px;
            }
            QPushButton:hover { border-color: #06b6d4; color: #06b6d4; }
            QPushButton:pressed { background: #3f3f5c; }
            QPushButton:disabled { color: #4b5563; border-color: #2d2d40; }
            QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox, QTextEdit {
                background: #1e1e2e;
                color: #e2e8f0;
                border: 1px solid #3f3f5c;
                border-radius: 4px;
                padding: 3px 6px;
                selection-background-color: #7c3aed;
            }
            QComboBox:focus, QLineEdit:focus,
            QSpinBox:focus, QDoubleSpinBox:focus {
                border-color: #7c3aed;
            }
            QComboBox::drop-down { border: none; }
            QListWidget, QTableWidget {
                background: #1e1e2e;
                color: #e2e8f0;
                border: 1px solid #3f3f5c;
                border-radius: 5px;
                alternate-background-color: #252538;
                selection-background-color: #7c3aed;
            }
            QProgressBar {
                background: #1e1e2e;
                border: 1px solid #3f3f5c;
                border-radius: 4px;
                height: 12px;
                text-align: center;
                font-size: 10px;
                color: #e2e8f0;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                    stop:0 #7c3aed, stop:1 #06b6d4);
                border-radius: 3px;
            }
            QCheckBox, QRadioButton { color: #e2e8f0; spacing: 6px; }
            QCheckBox::indicator, QRadioButton::indicator {
                border: 1px solid #3f3f5c;
                background: #1e1e2e;
                border-radius: 3px;
                width: 13px; height: 13px;
            }
            QCheckBox::indicator:checked, QRadioButton::indicator:checked {
                background: #7c3aed; border-color: #7c3aed;
            }
            QLabel { color: #e2e8f0; }
            QHeaderView::section {
                background: #2a2a3e; color: #94a3b8;
                border: none; padding: 4px 8px; font-size: 10px;
            }
            QStatusBar { background: #13131f; color: #94a3b8; font-size: 11px; }
            QMenuBar { background: #13131f; color: #e2e8f0; }
            QMenuBar::item:selected { background: #2a2a3e; }
            QMenu { background: #2a2a3e; border: 1px solid #3f3f5c; color: #e2e8f0; }
            QMenu::item:selected { background: #7c3aed; }
            QScrollBar:vertical {
                background: #1e1e2e; width: 8px; margin: 0;
            }
            QScrollBar::handle:vertical {
                background: #3f3f5c; border-radius: 4px; min-height: 20px;
            }
            QSplitter::handle { background: #3f3f5c; }
        """)


    # ── Quattrocento streaming ────────────────────────────────────────────────

    def _start_quattrocento_stream(self, source_path: str) -> None:
        """Start streaming Quattrocento data from one NPY file (or folder fallback)."""
        from PySide6.QtCore import QThread, Signal as QSignal

        class _Q4Worker(QThread):
            chunk_ready = QSignal(object, str)  # (ndarray, gesture_name)
            finished_all = QSignal()
            error = QSignal(str)

            def __init__(self, source, side, chunk_ms, parent=None):
                super().__init__(parent)
                self._source = source
                self._side = side
                self._chunk_ms = chunk_ms
                self._stop = False

            def stop(self):
                self._stop = True

            @staticmethod
            def _infer_subject(file_path: Path) -> str:
                for part in file_path.parts:
                    if part.upper().startswith("VP_"):
                        return part
                return "quattro_file"

            def _stream_signal(self, signal, gesture: str, sampling_rate: float):
                import numpy as np
                chunk_smp = max(1, int(float(sampling_rate) * self._chunk_ms / 1000.0))
                pos = 0
                while pos < len(signal):
                    if self._stop:
                        return
                    chunk = np.asarray(signal[pos:pos + chunk_smp], dtype=np.float32)
                    if chunk.size > 0:
                        self.chunk_ready.emit(chunk, gesture)
                    pos += chunk_smp
                    self.msleep(int(self._chunk_ms))

            def _stream_single_file(self, file_path: Path):
                from playagain_pipeline.gui.widgets.quattrocento_loader import (
                    QuattrocentoFileLoader,
                    TriggerSegment,
                    detect_segments,
                )

                side = "right" if file_path.name.lower().endswith("_right.npy") else "left"
                rec = QuattrocentoFileLoader.load_recording(
                    file_path,
                    subject_id=self._infer_subject(file_path),
                )
                if self._side in {"left", "right"} and side != self._side:
                    raise ValueError(
                        f"Selected file side '{side}' does not match requested side '{self._side}'."
                    )

                signal, trigger, ref_rms = QuattrocentoFileLoader.load_raw_data(rec)
                segs, _ = detect_segments(
                    signal,
                    trigger,
                    ref_rms,
                    rec.sampling_rate,
                    onset_delay_ms=150.0,
                )
                if not segs:
                    segs = [TriggerSegment(0, 0, len(signal), len(signal), "full", signal)]

                for seg in segs:
                    if self._stop:
                        break
                    self._stream_signal(seg.signal, rec.gesture, rec.sampling_rate)

            def run(self):
                try:
                    source = Path(self._source).expanduser()
                    if source.is_file():
                        self._stream_single_file(source)
                    elif source.is_dir():
                        # Backward-compatible fallback for callers that pass a folder.
                        from playagain_pipeline.gui.widgets.quattrocento_loader import (
                            QuattrocentoStreamAdapter,
                        )
                        adapter = QuattrocentoStreamAdapter(
                            source,
                            side=self._side,
                            chunk_ms=self._chunk_ms,
                            use_trigger_segments=True,
                            onset_delay_ms=150.0,
                        )
                        adapter.scan()
                        for chunk, gesture, _ in adapter.stream():
                            if self._stop:
                                break
                            self.chunk_ready.emit(chunk, gesture)
                            self.msleep(int(self._chunk_ms))
                    else:
                        raise FileNotFoundError(f"Quattrocento source not found: {source}")
                    self.finished_all.emit()
                except Exception as e:
                    self.error.emit(str(e))

        if hasattr(self, "_q4_worker") and self._q4_worker is not None:
            self._q4_worker.stop()
            self._q4_worker.wait(2000)

        src_name = Path(source_path).name.lower()
        side = "right" if src_name.endswith("_right.npy") else "left"
        chunk_ms = 20.0
        num_ch = self.channels_spin.value()

        self._q4_worker = _Q4Worker(source_path, side, chunk_ms, parent=self)
        self._q4_worker.chunk_ready.connect(self._on_q4_chunk)
        self._q4_worker.finished_all.connect(self._on_q4_finished)
        self._q4_worker.error.connect(lambda e: self._log(f"Quattrocento stream error: {e}"))
        self._q4_worker.start()

        self._q4_stream_active = True
        self._q4_stream_channels = None
        try:
            src = Path(source_path)
            if src.is_file() and src.suffix.lower() == ".npy":
                preview = np.load(str(src), mmap_mode="r")
                if preview.ndim == 2:
                    self._q4_stream_channels = int(max(1, preview.shape[1] - 4))
                del preview
        except Exception:
            # Non-fatal: first chunk will still set channel count.
            self._q4_stream_channels = None

        self.connect_btn.setEnabled(False)
        self.disconnect_btn.setEnabled(True)
        self._log(f"Quattrocento streaming started from: {source_path}  (side={side})")

    @Slot(object, str)
    def _on_q4_chunk(self, chunk, gesture: str) -> None:
        """Handle a streaming chunk from Quattrocento adapter."""
        import numpy as np
        if not isinstance(chunk, np.ndarray) or chunk.size == 0:
            return
        if chunk.ndim == 1:
            chunk = chunk.reshape(-1, 1)
        self._q4_stream_channels = int(chunk.shape[1])

        # If model is already loaded, keep prediction buffer in sync with stream channels.
        if self._current_model is not None and self._prediction_buffer is not None:
            if self._prediction_buffer.shape[1] != chunk.shape[1]:
                window_samples = self._prediction_buffer.shape[0]
                self._prediction_buffer = np.zeros((window_samples, chunk.shape[1]), dtype=np.float32)

        # Update ground truth label display
        if gesture != self._current_ground_truth_label:
            self._on_ground_truth_changed(gesture)
        # Feed into standard data path
        self._on_data_received(chunk)

    @Slot()
    def _on_q4_finished(self) -> None:
        self._log("Quattrocento: all files streamed.")
        self._q4_stream_active = False
        self.connect_btn.setEnabled(True)
        self.disconnect_btn.setEnabled(False)

def main():
    """Entry point for the GUI application."""
    import sys
    from PySide6.QtWidgets import QApplication
    from playagain_pipeline.gui.widgets.main_window_v2 import MainWindowV2

    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindowV2()
    window.show()
    sys.exit(app.exec())