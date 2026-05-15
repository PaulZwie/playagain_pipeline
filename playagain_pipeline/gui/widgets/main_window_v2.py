"""
main_window_v2.py
─────────────────
Redesigned main window shell.

This file is intentionally small. It wires together the three new
drop-in widgets (StatusStrip, HomeTab, QuickstartWizard) with the
*existing* tab-content factories from the old MainWindow. None of the
heavy lifting is rewritten — we reuse every ``_create_*_tab`` method
and every dialog. The redesign is structural, not algorithmic.

Usage
─────
    # After installing the three widget files alongside this one:
    from playagain_pipeline.gui.main_window_v2 import MainWindowV2
    app = QApplication(sys.argv)
    win = MainWindowV2()
    win.show()
    sys.exit(app.exec())

Or from the existing launcher, gate on an env var:

    ui_version = os.environ.get("PLAYAGAIN_UI", "v1")
    if ui_version == "v2":
        from ....main_window_v2 import MainWindowV2 as MainWindow
    else:
        from ....main_window   import MainWindow

Implementation note
───────────────────
This skeleton assumes the existing MainWindow's `_create_recording_tab`,
`_create_training_tab`, `_create_prediction_tab`, and
`_create_calibration_tab` methods are available. The cleanest way to
share them is to factor a mixin out of the existing class. For now,
``MainWindowV2`` subclasses ``MainWindow`` — inheriting all the tab
factory methods, all the signal handlers, and all the data-manager
plumbing, and overriding only ``_setup_ui`` to build the new shell.

That's a one-file-change migration. Once the v2 shell is stable, the
v1 ``_setup_ui`` can be deleted from MainWindow (or MainWindow made
abstract) and the two can be cleanly separated.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, Slot, QSettings, QTimer
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QMainWindow, QMenu, QMessageBox, QSplitter, QTabWidget, QToolButton,
    QVBoxLayout, QWidget,
)

# New widgets
from playagain_pipeline.gui.widgets.status_strip       import StatusStrip
from playagain_pipeline.gui.widgets.home_tab           import (
    HomeTab, TARGET_RECORD, TARGET_TRAIN, TARGET_USE, TARGET_TOOLS,
)
from playagain_pipeline.gui.widgets.quickstart_wizard  import QuickstartWizard

from playagain_pipeline.gui.main_window import MainWindow

from playagain_pipeline.gui.widgets.evaluation_tab import EvaluationTab

from playagain_pipeline.gui.widgets.thesis_report_dialog import (
        ThesisReportDialog,
    )


log = logging.getLogger(__name__)


# Tab indices in the new IA — kept as constants so handlers don't
# hard-code numbers.
TAB_HOME    = 0
TAB_RECORD  = 1
TAB_TRAIN   = 2
TAB_USE     = 3


class MainWindowV2(MainWindow):
    """
    Inherits everything from MainWindow and replaces the shell with
    the redesigned layout (Home tab + status strip + Tools menu).

    Because this is a subclass, every signal/slot defined on
    MainWindow is still connected — the tab contents behave exactly
    as before. Only the navigation chrome changes.
    """

    def __init__(self):
        # MainWindow.__init__ does a lot of setup (data manager, config
        # loading, logging). We let it run completely, then rebuild the
        # central widget from scratch.
        super().__init__()
        self._v2_rebuild_shell()

        # First-run check happens after the shell is built so the
        # wizard has a parent window to attach to.
        QTimer.singleShot(0, self._v2_maybe_show_quickstart)

    # ------------------------------------------------------------------
    # Shell construction
    # ------------------------------------------------------------------

    def _v2_rebuild_shell(self) -> None:
        """
        Replace the v1 central widget with the new Home + 3-tab v2 layout.

        ┌─────────────────────────────────────────────────────────────┐
        │ LIFETIME ORDERING — read this before touching this method.  │
        │                                                             │
        │ super().__init__() → _setup_ui() builds a v1 central widget │
        │ that owns (as Qt children):                                 │
        │   • self.workflow_stepper                                   │
        │   • self._plot_widget                                       │
        │   • self.mode_tabs  (contains the 3 content tabs)          │
        │                                                             │
        │ setCentralWidget(new_widget) schedules the OLD central      │
        │ widget for deletion.  Any Qt child that has NOT been        │
        │ re-parented BEFORE that call will be destroyed, and any     │
        │ Python attribute still pointing at it becomes a dangling    │
        │ wrapper — touching it causes SIGBUS.                        │
        │                                                             │
        │ Therefore: RESCUE before REPLACE.                           │
        │   1. Re-parent every widget we want to keep to a temporary  │
        │      invisible widget (rescuer) so Qt drops their ownership │
        │      from the old central widget.                           │
        │   2. Call setCentralWidget — old widget + orphaned children │
        │      are safely deleted; rescued widgets survive.           │
        │   3. Adopt rescued widgets into the new layout.             │
        │                                                             │
        │ Also: NEVER call _create_recording_tab() / _create_         │
        │ training_tab() / _create_prediction_tab() here — those      │
        │ methods have side-effects (they reassign self._plot_widget, │
        │ self.subject_id_edit, etc.) and create duplicate widgets    │
        │ with broken signal connections.  Instead pull the already-  │
        │ constructed tab widgets straight out of self.mode_tabs.     │
        └─────────────────────────────────────────────────────────────┘
        """
        self.setWindowTitle("EMG Pipeline")

        # ── Step 1: Rescue widgets we want to keep ───────────────────
        # Re-parent to a temporary off-screen widget so setCentralWidget
        # cannot delete them.  The rescuer stays alive (referenced by
        # self) long enough for the subsequent addWidget calls.
        rescuer = QWidget()          # temporary safe harbor
        rescuer.hide()

        stepper      = getattr(self, "workflow_stepper", None)
        plot_widget  = getattr(self, "_plot_widget",     None)
        mode_tabs_v1 = getattr(self, "mode_tabs",        None)

        # Pull the three content tabs out of the v1 QTabWidget before
        # it gets orphaned.  removeTab() detaches the widget from the
        # tab-bar but keeps it alive; re-parent to rescuer for safety.
        tab_record = tab_train = tab_use = None
        if mode_tabs_v1 is not None:
            # v1 tab order: 0=Record, 1=Train, 2=Predict, 3=Calibration, 4=Validation
            # We grab by position; labels are tab-bar cosmetics only.
            try:
                tab_record = mode_tabs_v1.widget(0)
                if tab_record:
                    mode_tabs_v1.removeTab(0)
                    tab_record.setParent(rescuer)
            except Exception:
                tab_record = None
            try:
                # After removing index-0, old index-1 is now index-0
                tab_train = mode_tabs_v1.widget(0)
                if tab_train:
                    mode_tabs_v1.removeTab(0)
                    tab_train.setParent(rescuer)
            except Exception:
                tab_train = None
            try:
                tab_use = mode_tabs_v1.widget(0)
                if tab_use:
                    mode_tabs_v1.removeTab(0)
                    tab_use.setParent(rescuer)
            except Exception:
                tab_use = None

        if stepper is not None:
            stepper.setParent(rescuer)
        if plot_widget is not None:
            plot_widget.setParent(rescuer)

        # ── Step 2: Replace the central widget ───────────────────────
        # The old central widget (and everything left in it) is now
        # safely destroyable.
        central = QWidget()
        self.setCentralWidget(central)          # old central widget dies here
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Step 3: Adopt rescued widgets into the new layout ─────────

        # Status strip
        self._v2_status = StatusStrip(app_name="EMG Pipeline", parent=central)
        self._v2_status.device_pill_clicked.connect(self._on_connect_device)
        self._v2_status.subject_pill_clicked.connect(self._on_edit_participant_info)
        self._v2_status.model_pill_clicked.connect(
            lambda: self._v2_tabs.setCurrentIndex(TAB_USE)
        )
        self._v2_status.help_requested.connect(self._on_about)
        root.addWidget(self._v2_status)

        # Workflow stepper
        if stepper is not None:
            stepper.setParent(central)
            root.addWidget(stepper)
            try:
                stepper.step_clicked.disconnect()
            except Exception:
                pass
            stepper.step_clicked.connect(self._v2_on_workflow_step_clicked)

        # ── Main tab widget ──────────────────────────────────────────
        self._v2_tabs = QTabWidget()
        self._v2_tabs.currentChanged.connect(self._v2_on_tab_changed)

        # Home tab — new
        self._v2_home = HomeTab(
            data_dir=Path(self.data_manager.data_dir),
            parent=self,
        )
        self._v2_home.jump_to_target.connect(self._v2_jump_to)
        self._v2_home.quickstart_requested.connect(self._v2_show_quickstart)
        self._v2_tabs.addTab(self._v2_home, "  Home  ")

        # Reuse the already-created tab widgets from the v1 mode_tabs.
        # Fall back to creating fresh ones only if extraction failed
        # (e.g. running against a future MainWindow that changed the
        # tab layout — graceful degradation beats a hard crash).
        def _adopt(widget, factory_fn, label):
            """Re-parent widget into v2_tabs, or rebuild via factory."""
            if widget is not None:
                widget.setParent(self._v2_tabs)
                self._v2_tabs.addTab(widget, label)
            else:
                try:
                    self._v2_tabs.addTab(factory_fn(), label)
                except Exception as exc:
                    log.error("Could not build tab %r: %s", label, exc)

        _adopt(tab_record, self._create_recording_tab, "  Record new data  ")
        _adopt(tab_train,  self._create_training_tab,  "  Train a model  ")
        _adopt(tab_use,    self._create_prediction_tab, "  Use a model live  ")

        self._v2_tabs.setTabToolTip(TAB_HOME,   "Landing page — recent activity and quick actions")
        self._v2_tabs.setTabToolTip(TAB_RECORD, "Connect a device and capture gesture sessions")
        self._v2_tabs.setTabToolTip(TAB_TRAIN,  "Build a dataset and train a classifier")
        self._v2_tabs.setTabToolTip(TAB_USE,    "Run a trained model live on the EMG stream")

        # ── Plot splitter ────────────────────────────────────────────
        if plot_widget is not None:
            v2_splitter = QSplitter(Qt.Orientation.Horizontal, parent=central)
            v2_splitter.addWidget(self._v2_tabs)

            plot_host = QWidget(central)
            plot_host_layout = QVBoxLayout(plot_host)
            plot_host_layout.setContentsMargins(0, 0, 0, 0)
            plot_widget.setParent(plot_host)
            plot_host_layout.addWidget(plot_widget)
            v2_splitter.addWidget(plot_host)

            v2_splitter.setStretchFactor(0, 3)
            v2_splitter.setStretchFactor(1, 2)
            v2_splitter.setSizes([900, 540])
            root.addWidget(v2_splitter, 1)
        else:
            root.addWidget(self._v2_tabs, 1)

        # rescuer goes out of scope here — any remaining v1 widgets it
        # was holding are cleaned up by Python/Qt GC.  The widgets we
        # adopted above are now owned by the new layout tree.

        # ── Tools menu ───────────────────────────────────────────────
        # Attached to a toolbar button in the top-right of the tab bar.
        tools_btn = QToolButton()
        tools_btn.setText("Tools ▾")
        tools_btn.setToolTip("Advanced actions — validation, calibration, "
                             "dataset management, and more")
        tools_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        tools_btn.setStyleSheet(
            "QToolButton {"
            "  border: none; padding: 6px 12px; font-size: 11px; "
            "  color: #475569;"
            "}"
            "QToolButton:hover { color: #0284c7; }"
            "QToolButton::menu-indicator { image: none; }"
        )
        tools_btn.setMenu(self._v2_build_tools_menu())
        self._v2_tabs.setCornerWidget(tools_btn, Qt.Corner.TopRightCorner)

        # Log area — keep a reference from the parent class if it
        # exists, otherwise skip. Some setups don't need a global log.
        # (The existing MainWindow stashes it as self.log_text.)
        # Intentionally not re-parented here; the Home tab + per-tab
        # controls carry most of the signaling.

        # Initial status strip population from data manager state
        self._v2_refresh_status_strip()

    # ------------------------------------------------------------------
    # Tools menu
    # ------------------------------------------------------------------

    def _v2_build_tools_menu(self) -> QMenu:
        menu = QMenu(self)

        act_validate = QAction("Validate models…", self)
        act_validate.triggered.connect(self._v2_open_evaluation)
        menu.addAction(act_validate)

        act_thesis = QAction("Build thesis report…", self)
        act_thesis.setToolTip(
            "Create the validation runs Chapter 6 depends on and generate "
            "every table and figure referenced from chapters 6, 7 and 8."
        )
        act_thesis.triggered.connect(self._v2_open_thesis_report)
        menu.addAction(act_thesis)

        act_calibrate = QAction("Calibrate bracelet…", self)
        # Reuse the existing handler — in v1 this was a tab that the
        # user opened via self.mode_tabs.setCurrentIndex(...). Here we
        # route through the existing calibration dialog or the
        # calibration tab if it's easier to keep as a popout.
        act_calibrate.triggered.connect(self._v2_open_calibration)
        menu.addAction(act_calibrate)

        menu.addSeparator()

        act_quattrocento = QAction("Quattrocento training…", self)
        if hasattr(self, "_on_quattrocento_training"):
            act_quattrocento.triggered.connect(self._on_quattrocento_training)
        menu.addAction(act_quattrocento)

        act_features = QAction("Feature selection…", self)
        if hasattr(self, "_on_feature_selection"):
            act_features.triggered.connect(self._on_feature_selection)
        menu.addAction(act_features)

        act_bracelet = QAction("Bracelet visualisation…", self)
        if hasattr(self, "_on_bracelet_visualization"):
            act_bracelet.triggered.connect(self._on_bracelet_visualization)
        menu.addAction(act_bracelet)

        menu.addSeparator()

        act_participant = QAction("Edit participant info…", self)
        if hasattr(self, "_on_edit_participant_info"):
            act_participant.triggered.connect(self._on_edit_participant_info)
        menu.addAction(act_participant)

        act_config = QAction("Open configuration…", self)
        if hasattr(self, "_on_open_config"):
            act_config.triggered.connect(self._on_open_config)
        menu.addAction(act_config)

        menu.addSeparator()

        act_quickstart = QAction("Show quickstart again", self)
        act_quickstart.triggered.connect(self._v2_show_quickstart)
        menu.addAction(act_quickstart)

        act_about = QAction("About", self)
        if hasattr(self, "_on_about"):
            act_about.triggered.connect(self._on_about)
        menu.addAction(act_about)

        return menu

    # ------------------------------------------------------------------
    # Jump/routing handlers
    # ------------------------------------------------------------------

    @Slot(str)
    def _v2_jump_to(self, target: str) -> None:
        """Central router for Home card clicks and Recent-activity rows."""
        if target == TARGET_RECORD:
            self._v2_tabs.setCurrentIndex(TAB_RECORD)
        elif target == TARGET_TRAIN:
            self._v2_tabs.setCurrentIndex(TAB_TRAIN)
        elif target == TARGET_USE:
            self._v2_tabs.setCurrentIndex(TAB_USE)
        elif target == TARGET_TOOLS:
            self._v2_open_validation()
        else:
            log.warning("Unknown jump target: %s", target)

    @Slot(int)
    def _v2_on_workflow_step_clicked(self, step_idx: int) -> None:
        """Stepper is 0: Record, 1: Train, 2: Use. Map to new tab indices."""
        if step_idx == 0:
            self._v2_tabs.setCurrentIndex(TAB_RECORD)
        elif step_idx == 1:
            self._v2_tabs.setCurrentIndex(TAB_TRAIN)
        elif step_idx == 2:
            self._v2_tabs.setCurrentIndex(TAB_USE)

    @Slot(int)
    def _v2_on_tab_changed(self, tab_idx: int) -> None:
        """Update the workflow stepper as tabs change."""
        if not hasattr(self, "workflow_stepper") or self.workflow_stepper is None:
            return
        mapping = {
            TAB_RECORD: 0,
            TAB_TRAIN:  1,
            TAB_USE:    2,
        }
        if tab_idx in mapping:
            self.workflow_stepper.set_current(mapping[tab_idx])
        # On Home tab, clear the current highlight so the stepper
        # looks "at rest".
        else:
            self.workflow_stepper.reset()

        # Re-scan Home tab's recent activity when user comes back to it
        if tab_idx == TAB_HOME:
            self._v2_home.refresh()

    # ------------------------------------------------------------------
    # Tools handlers
    # ------------------------------------------------------------------

    def _v2_open_evaluation(self) -> None:
        # The merged EvaluationTab covers four modes — sessions, game
        # recordings, Unity recordings and cross-validation — so the
        # window title and default size reflect the broader scope. The
        # widget itself is constructed exactly the same way as before;
        # the constructor signature is unchanged, so the only thing
        # that moves here is the chrome around it.
        if not hasattr(self, "_v2_evaluation_window") or self._v2_evaluation_window is None:
            self._v2_evaluation_window = QMainWindow(self)
            self._v2_evaluation_window.setWindowTitle("Evaluation & Cross-validation")
            self._v2_evaluation_window.resize(1400, 900)
            self._v2_evaluation_window.setCentralWidget(EvaluationTab(self.data_manager))
        self._v2_evaluation_window.show()
        self._v2_evaluation_window.raise_()
        self._v2_evaluation_window.activateWindow()

    def _v2_open_thesis_report(self) -> None:
        '''Open the thesis-report builder dialog.'''
        # The dialog is non-modal so the user can keep the main window
        # visible while a long-running validation suite is queued. We
        # cache it on self so reopening doesn't lose state.
        if not hasattr(self, "_v2_thesis_dialog") or self._v2_thesis_dialog is None:
            data_dir = Path(self.data_manager.data_dir)
            self._v2_thesis_dialog = ThesisReportDialog(
                data_dir=data_dir, parent=self,
            )
        self._v2_thesis_dialog.show()
        self._v2_thesis_dialog.raise_()
        self._v2_thesis_dialog.activateWindow()

    def _v2_open_calibration(self) -> None:
        """
        Open CalibrationDialog directly.

        The v1 inline calibration panel (_on_start_live_calibration,
        live_cal_gestures_edit, etc.) is a child of self.mode_tabs which
        no longer exists in the v2 shell — calling _on_start_live_calibration
        would SIGBUS on the first .text() access to a deleted QLineEdit.
        CalibrationDialog provides Live / From Session / Validate / Corpus
        Audit in a standalone window and is the correct entry point here.
        """
        try:
            from playagain_pipeline.gui.widgets.calibration_dialog import CalibrationDialog

            calibrator = getattr(self, "calibrator", None) or getattr(self, "_calibrator", None)
            if calibrator is None:
                QMessageBox.warning(self, "Calibration unavailable",
                                    "No calibrator attached to this window.")
                return

            # Always resolve device from device_manager at call time.
            dm = getattr(self, "device_manager", None)
            device = (getattr(dm, "device", None) if dm is not None
                      else getattr(self, "_device", None))

            data_manager = (getattr(self, "data_manager", None)
                            or getattr(self, "_data_manager", None))

            dlg = CalibrationDialog(
                calibrator=calibrator,
                device=device,
                data_manager=data_manager,
                parent=self,
            )
            dlg.calibration_complete.connect(self._v2_on_calibration_complete)
            dlg.exec()

        except Exception as e:
            log.exception("Could not open calibration dialog")
            QMessageBox.warning(self, "Calibration unavailable",
                                f"Could not open the calibration dialog:\n{e}")

    @Slot(object)
    def _v2_on_calibration_complete(self, result) -> None:
        """Apply a CalibrationResult emitted by CalibrationDialog."""
        if result is None:
            return
        cal = getattr(self, "calibrator", None) or getattr(self, "_calibrator", None)
        if cal is not None:
            cal._current_calibration = result
        if hasattr(self, "_update_calibration_display"):
            try:
                self._update_calibration_display()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Status strip population
    # ------------------------------------------------------------------

    @staticmethod
    def _v2_widget_alive(w) -> bool:
        """True if w is a non-None Qt widget whose C++ object still exists."""
        if w is None:
            return False
        try:
            w.isVisible()
            return True
        except RuntimeError:
            return False

    def _v2_refresh_status_strip(self) -> None:
        """Pull current device/subject/model state into the strip."""
        # Device — always read from device_manager for freshness
        dm = getattr(self, "device_manager", None)
        device = getattr(dm, "device", None) if dm is not None else getattr(self, "_device", None)
        if device is not None and getattr(device, "is_connected", False):
            self._v2_status.set_device_status("connected", getattr(device, "name", "Device"))
        else:
            self._v2_status.set_device_status("disconnected")

        # Subject — prefer scalar attr; only fall back to widgets if they're alive
        subject = getattr(self, "_current_subject", None)
        if not subject:
            for attr_name in ("subject_combo", "subject_input", "_subject_id",
                              "subject_id_edit"):
                w = getattr(self, attr_name, None)
                if not self._v2_widget_alive(w):
                    continue
                try:
                    subject = w.currentText() if hasattr(w, "currentText") else w.text()
                except Exception:
                    subject = None
                if subject:
                    break
        self._v2_status.set_subject(subject)

        # Active model
        model = getattr(self, "_current_model", None)
        if model is not None:
            name = getattr(model, "name", None) or getattr(model, "_name", None)
            self._v2_status.set_active_model(name, ready=True)
        else:
            self._v2_status.set_active_model(None)

    # Hooks the parent class can call to keep status strip in sync.
    # These are safe to call even if the strip doesn't exist yet.

    def notify_device_state(self, state: str, name: Optional[str] = None) -> None:
        if hasattr(self, "_v2_status"):
            self._v2_status.set_device_status(state, name)

    def notify_subject_changed(self, subject_id: Optional[str]) -> None:
        if hasattr(self, "_v2_status"):
            self._v2_status.set_subject(subject_id)
        if hasattr(self, "_v2_home"):
            self._v2_home.refresh()

    def notify_model_changed(self, model_name: Optional[str], ready: bool = False) -> None:
        if hasattr(self, "_v2_status"):
            self._v2_status.set_active_model(model_name, ready=ready)
        if hasattr(self, "_v2_home"):
            self._v2_home.refresh()

    def notify_session_recorded(self) -> None:
        """Call after a session has been written to disk."""
        if hasattr(self, "_v2_home"):
            self._v2_home.refresh()
        if hasattr(self, "workflow_stepper"):
            self.workflow_stepper.mark_done(0)

    def notify_model_trained(self) -> None:
        if hasattr(self, "_v2_home"):
            self._v2_home.refresh()
        if hasattr(self, "workflow_stepper"):
            self.workflow_stepper.mark_done(1)

    # ------------------------------------------------------------------
    # First-run quickstart
    # ------------------------------------------------------------------

    def _v2_maybe_show_quickstart(self) -> None:
        settings = QSettings("playagain", "pipeline")
        if settings.value("quickstart_completed", False, type=bool):
            return
        # Also bail out if the data dir already has sessions — the
        # user has clearly used the app before, even if the settings
        # key is missing (e.g. fresh install pointed at existing data).
        if hasattr(self, "_v2_home") and not self._v2_home._is_first_run:
            return
        self._v2_show_quickstart()

    @Slot()
    def _v2_show_quickstart(self) -> None:
        dlg = QuickstartWizard(parent=self)
        subject = getattr(self, "_current_subject", None) or "VP_01"
        dlg.set_default_subject(subject)

        dlg.subject_chosen.connect(self._v2_on_quickstart_subject)
        dlg.connect_device_requested.connect(self._on_connect_device)
        dlg.jump_to_record_requested.connect(
            lambda: self._v2_tabs.setCurrentIndex(TAB_RECORD)
        )

        dlg.exec()

        settings = QSettings("playagain", "pipeline")
        settings.setValue("quickstart_completed", True)
        if hasattr(self, "_v2_home"):
            self._v2_home.mark_no_longer_first_run()

    @Slot(str)
    def _v2_on_quickstart_subject(self, subject_id: str) -> None:
        """Record the subject id picked during the wizard."""
        self._current_subject = subject_id
        self.notify_subject_changed(subject_id)

        # If the parent class has a subject combo, push it in there too.
        for attr_name in ("subject_combo", "subject_input"):
            w = getattr(self, attr_name, None)
            if w is None:
                continue
            try:
                if hasattr(w, "setCurrentText"):
                    w.setCurrentText(subject_id)
                elif hasattr(w, "setText"):
                    w.setText(subject_id)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Parent overrides — keep existing behaviour but also refresh v2
    # ------------------------------------------------------------------

    def _on_device_connected(self, connected: bool):  # type: ignore[override]
        # Let the parent run its full signal path, then update the strip.
        super_fn = getattr(super(), "_on_device_connected", None)
        if callable(super_fn):
            try:
                super_fn(connected)
            except Exception as e:
                log.warning("parent _on_device_connected raised: %s", e)
        state = "connected" if connected else "disconnected"
        device_name = getattr(getattr(self, "_device", None), "name", None)
        self.notify_device_state(state, device_name)