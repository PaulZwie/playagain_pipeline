"""
calibration_dialog.py
─────────────────────
Standalone calibration dialog — three tabs in one window.

Tab 1: Live Calibration
    Record gestures on the connected device right now.  Useful for
    the "pretrained-model" workflow where no session exists yet.

Tab 2: From Session
    Pick any previously-recorded session and derive the rotation offset
    from its gesture trials.  Same path as the inline calibration tab
    (calibrator.calibrate_from_session()).  Works without a device.

Tab 3: Validate
    Run the free checks (self-consistency + symmetry) after any
    calibration.  Optionally pick a labelled session for the
    held-out classification accuracy check.

Usage
─────
    dlg = CalibrationDialog(
        calibrator=self.calibrator,
        device=self.device_manager.device,   # may be None
        data_manager=self.data_manager,      # may be None (disables Tab 2)
        parent=self,
    )
    if dlg.exec():
        result = dlg.get_calibration_result()
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from PySide6.QtCore import Qt, Signal, Slot, QTimer
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_GESTURE_EMOJIS: Dict[str, str] = {
    "rest": "🖐",      "open_hand": "🖐",   "fist": "✊",
    "pinch": "👌",     "tripod": "🤌",      "index_point": "☝",
    "thumb_out": "👍", "waveout": "🤚",
    "index_flex": "☝", "middle_flex": "🖕", "ring_flex": "💍",
    "pinky_flex": "🤙", "thumb_flex": "👍",
}


def _info(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setWordWrap(True)
    lbl.setStyleSheet("color: #555; font-size: 10px;")
    return lbl


# ---------------------------------------------------------------------------
# CalibrationDialog
# ---------------------------------------------------------------------------

class CalibrationDialog(QDialog):
    """Tabbed calibration dialog.  Live, From Session, and Validate in one place."""

    calibration_complete = Signal(object)  # CalibrationResult

    # Standard 8-gesture live-calibration protocol
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

    def __init__(
        self,
        calibrator,
        device=None,
        data_manager=None,
        parent=None,
    ):
        super().__init__(parent)
        self.calibrator   = calibrator
        self.device       = device
        self.data_manager = data_manager

        self._calibration_result = None
        # Gesture arrays captured during live or extracted from a session,
        # kept for the validator.
        self._gesture_data: Dict[str, np.ndarray] = {}

        # Live-recording state machine
        self._recording_timer    = QTimer(self)
        self._recording_duration = 3.0
        self._live_idx           = 0
        self._live_countdown     = 0
        self._live_remaining     = 0.0
        self._is_recording       = False
        self._live_buffer: list  = []

        self.setWindowTitle("EMG Calibration")
        self.setMinimumSize(560, 560)
        self.resize(640, 620)
        self._setup_ui()
        QTimer.singleShot(0, self._refresh_sessions)

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)

        self._tabs = QTabWidget()
        root.addWidget(self._tabs, 1)

        self._tabs.addTab(self._build_live_tab(),     "① Live")
        self._tabs.addTab(self._build_session_tab(),  "② From Session")
        self._tabs.addTab(self._build_validate_tab(), "③ Validate")

        log_box = QGroupBox("Log")
        ll = QVBoxLayout(log_box)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFixedHeight(90)
        ll.addWidget(self.log_text)
        root.addWidget(log_box)

        brow = QHBoxLayout()
        brow.addStretch()
        self.finish_btn = QPushButton("Apply && Close")
        self.finish_btn.setToolTip("Apply the last successful calibration and close.")
        self.finish_btn.setEnabled(False)
        self.finish_btn.setStyleSheet(
            "QPushButton { background:#16a34a; color:white; border-radius:4px; "
            "font-weight:600; padding:4px 16px; } "
            "QPushButton:hover { background:#15803d; } "
            "QPushButton:disabled { background:#cbd5e1; color:#64748b; }"
        )
        self.finish_btn.clicked.connect(self.accept)
        brow.addWidget(self.finish_btn)

        cancel_btn = QPushButton("Close without applying")
        cancel_btn.clicked.connect(self.reject)
        brow.addWidget(cancel_btn)
        root.addLayout(brow)

        # Disable session tab when no data_manager is available
        if self.data_manager is None:
            self._tabs.setTabEnabled(1, False)
            self._tabs.setTabToolTip(
                1, "Session-based calibration requires a DataManager."
            )
            # Point the user to the live tab directly
            self._tabs.setCurrentIndex(0)

    # ── Tab 1: Live ──────────────────────────────────────────────────

    def _build_live_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)

        lay.addWidget(_info(
            "Record gestures live with the connected device.\n"
            "waveOut (first gesture) is the primary sync signal "
            "(Barona López et al., 2020). Device must be streaming."
        ))

        self.live_gesture_lbl = QLabel("—")
        self.live_gesture_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        f = self.live_gesture_lbl.font()
        f.setPointSize(28); f.setBold(True)
        self.live_gesture_lbl.setFont(f)
        lay.addWidget(self.live_gesture_lbl)

        self.live_instruction_lbl = QLabel("")
        self.live_instruction_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.live_instruction_lbl.setWordWrap(True)
        lay.addWidget(self.live_instruction_lbl)

        self.live_progress = QProgressBar()
        self.live_progress.setMaximum(int(self._recording_duration * 10))
        self.live_progress.setValue(0)
        lay.addWidget(self.live_progress)

        self.live_status_lbl = QLabel("Ready — click Start when device is streaming.")
        self.live_status_lbl.setStyleSheet("font-weight:bold;")
        lay.addWidget(self.live_status_lbl)

        brow = QHBoxLayout()
        self.live_start_btn = QPushButton("▶  Start Live Calibration")
        self.live_start_btn.setFixedHeight(34)
        self.live_start_btn.clicked.connect(self._on_live_start)
        brow.addWidget(self.live_start_btn)

        self.live_cancel_btn = QPushButton("✕  Cancel")
        self.live_cancel_btn.setFixedHeight(34)
        self.live_cancel_btn.setEnabled(False)
        self.live_cancel_btn.clicked.connect(self._on_live_cancel)
        brow.addWidget(self.live_cancel_btn)
        lay.addLayout(brow)

        lay.addStretch()
        return w

    # ── Tab 2: From Session ──────────────────────────────────────────

    def _build_session_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)

        lay.addWidget(_info(
            "Derive the rotation offset from a previously-recorded session.\n"
            "No device needed — works entirely from saved EMG data.\n\n"
            "If the session has dedicated calibration_sync trials (waveOut), "
            "those are used first. Otherwise all gesture trials are used."
        ))

        form = QFormLayout()
        self.ses_subject_combo = QComboBox()
        self.ses_subject_combo.currentTextChanged.connect(self._on_subject_changed)
        form.addRow("Subject:", self.ses_subject_combo)

        self.ses_session_combo = QComboBox()
        form.addRow("Session:", self.ses_session_combo)
        lay.addLayout(form)

        refresh_btn = QPushButton("⟳  Refresh list")
        refresh_btn.clicked.connect(self._refresh_sessions)
        lay.addWidget(refresh_btn)

        self.ses_cal_btn = QPushButton("Calibrate from this session")
        self.ses_cal_btn.setFixedHeight(36)
        self.ses_cal_btn.setStyleSheet(
            "QPushButton { background:#0284c7; color:white; border-radius:4px; "
            "font-weight:600; } QPushButton:hover { background:#0369a1; } "
            "QPushButton:disabled { background:#cbd5e1; color:#64748b; }"
        )
        self.ses_cal_btn.clicked.connect(self._on_calibrate_from_session)
        lay.addWidget(self.ses_cal_btn)

        self.ses_status_lbl = QLabel("")
        lay.addWidget(self.ses_status_lbl)
        lay.addStretch()
        return w

    # ── Tab 3: Validate ──────────────────────────────────────────────

    def _build_validate_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)

        lay.addWidget(_info(
            "Three independent checks verify the calibration quality.\n\n"
            "Free (no extra data): Self-consistency + Symmetry.\n"
            "  • Self-consistency: re-run xcorr — drift > 1 ch = unreliable.\n"
            "  • Symmetry: primary peak ≥ 1.5× runner-up — catches bracelet-on-backwards.\n\n"
            "Held-out (needs a labelled session): Gesture discrimination accuracy ≥ 70%."
        ))

        self.val_free_btn = QPushButton("Run free checks (self-consistency + symmetry)")
        self.val_free_btn.setFixedHeight(34)
        self.val_free_btn.clicked.connect(self._on_run_free_checks)
        lay.addWidget(self.val_free_btn)

        self.val_ho_btn = QPushButton("Run held-out check (pick a session)…")
        self.val_ho_btn.setFixedHeight(34)
        self.val_ho_btn.clicked.connect(self._on_run_held_out_check)
        lay.addWidget(self.val_ho_btn)

        self.val_result_lbl = QLabel("No results yet.")
        self.val_result_lbl.setWordWrap(True)
        self.val_result_lbl.setStyleSheet(
            "background:#f8fafc; border:1px solid #e2e8f0; "
            "padding:6px; border-radius:4px; font-family:monospace;"
        )
        self.val_result_lbl.setMinimumHeight(100)
        lay.addWidget(self.val_result_lbl)

        lay.addStretch()
        return w

    # ------------------------------------------------------------------
    # Live calibration
    # ------------------------------------------------------------------

    @Slot()
    def _on_live_start(self) -> None:
        if self.device is None or not getattr(self.device, "is_streaming", False):
            QMessageBox.warning(
                self, "Device not streaming",
                "Please connect a device and start streaming before live calibration."
            )
            return

        self._live_idx      = 0
        self._live_countdown = 3
        self._gesture_data  = {}
        self._live_buffer   = []
        self._is_recording  = False

        self.live_start_btn.setEnabled(False)
        self.live_cancel_btn.setEnabled(True)
        self.live_progress.setValue(0)
        self._live_set("GET READY",
                       f"First: {self.CALIBRATION_GESTURES[0][0]}",
                       f"Starting in {self._live_countdown}s")

        # Use 1-second ticks for the countdown phase
        self._recording_timer.timeout.disconnect() if self._recording_timer.receivers(
            self._recording_timer.timeout) else None
        self._recording_timer.timeout.connect(self._live_countdown_tick)
        self._recording_timer.start(1000)

    @Slot()
    def _on_live_cancel(self) -> None:
        self._recording_timer.stop()
        try:
            self._recording_timer.timeout.disconnect()
        except Exception:
            pass
        self._is_recording = False
        self._live_buffer  = []
        self._live_disconnect()
        self.live_start_btn.setEnabled(True)
        self.live_cancel_btn.setEnabled(False)
        self._live_set("Cancelled", "", "")

    def _live_countdown_tick(self) -> None:
        self._live_countdown -= 1
        if self._live_countdown > 0:
            g = self.CALIBRATION_GESTURES[self._live_idx][0]
            self._live_set("GET READY", f"Next: {g}", f"Starting in {self._live_countdown}s")
        else:
            # Switch to 100 ms recording ticks
            self._recording_timer.stop()
            try:
                self._recording_timer.timeout.disconnect()
            except Exception:
                pass
            self._is_recording   = True
            self._live_buffer    = []
            self._live_remaining = self._recording_duration
            g, instr = self.CALIBRATION_GESTURES[self._live_idx]
            emoji = _GESTURE_EMOJIS.get(g, "")
            self._live_set(f"{emoji} {g.upper()}" if emoji else g.upper(),
                           instr, f"Hold… {self._live_remaining:.0f}s")
            self.live_progress.setValue(int(self._recording_duration * 10))
            self._live_connect()
            self._recording_timer.timeout.connect(self._live_record_tick)
            self._recording_timer.start(100)

    def _live_record_tick(self) -> None:
        self._live_remaining -= 0.1
        self.live_progress.setValue(max(0, int(self._live_remaining * 10)))
        if self._live_remaining > 0:
            return

        # Done recording this gesture
        self._recording_timer.stop()
        try:
            self._recording_timer.timeout.disconnect()
        except Exception:
            pass
        self._is_recording = False
        self._live_disconnect()

        g, _ = self.CALIBRATION_GESTURES[self._live_idx]
        if self._live_buffer:
            arr = np.vstack(self._live_buffer)
            self._gesture_data[g] = arr
            self._log(f"✓ '{g}': {arr.shape[0]} samples")
        else:
            self._log(f"⚠ No data for '{g}' — skipped")

        self._live_idx += 1
        if self._live_idx >= len(self.CALIBRATION_GESTURES):
            self._live_finish()
            return

        # Pause then countdown for next gesture
        self._live_countdown = 2
        nxt = self.CALIBRATION_GESTURES[self._live_idx][0]
        self._live_set("DONE", f"Next: {nxt}", f"Pause {self._live_countdown}s")
        self.live_progress.setValue(0)
        self._recording_timer.timeout.connect(self._live_countdown_tick)
        self._recording_timer.start(1000)

    def _live_finish(self) -> None:
        self._recording_timer.stop()
        try:
            self._recording_timer.timeout.disconnect()
        except Exception:
            pass
        self.live_start_btn.setEnabled(True)
        self.live_cancel_btn.setEnabled(False)

        if not self._gesture_data:
            self._live_set("Failed", "No data collected.", "")
            return

        self._live_set("COMPUTING", "Analysing patterns…", "")
        try:
            device_name = getattr(self.device, "name", "unknown")
            result = self.calibrator.calibrate(
                calibration_data=self._gesture_data,
                device_name=device_name,
            )
            self._calibration_result = result
            self.finish_btn.setEnabled(True)
            self.calibration_complete.emit(result)
            self._log(f"Live calibration complete!")
            self._log(f"  Offset: {result.rotation_offset} ch  "
                      f"Confidence: {result.confidence:.2%}")
            self._live_set(
                "COMPLETE ✓",
                f"Offset {result.rotation_offset} ch  |  "
                f"Confidence {result.confidence:.0%}",
                "Click 'Apply & Close' or check the Validate tab.",
            )
            self._run_free_checks_internal()
        except Exception as e:
            self._log(f"Live calibration error: {e}")
            self._live_set("ERROR", str(e), "")

    def _live_set(self, title: str, instruction: str, detail: str) -> None:
        self.live_gesture_lbl.setText(title)
        self.live_instruction_lbl.setText(instruction)
        self.live_status_lbl.setText(detail)

    def _live_connect(self) -> None:
        if self.device and hasattr(self.device, "data_ready"):
            try:
                self.device.data_ready.connect(self._on_emg_data)
            except Exception:
                pass

    def _live_disconnect(self) -> None:
        if self.device and hasattr(self.device, "data_ready"):
            try:
                self.device.data_ready.disconnect(self._on_emg_data)
            except Exception:
                pass

    @Slot(object)
    def _on_emg_data(self, data) -> None:
        if self._is_recording and data is not None:
            self._live_buffer.append(np.asarray(data).copy())

    # ------------------------------------------------------------------
    # Session-based calibration
    # ------------------------------------------------------------------

    def _refresh_sessions(self) -> None:
        if self.data_manager is None:
            return
        try:
            self.ses_subject_combo.blockSignals(True)
            self.ses_subject_combo.clear()
            subjects = self.data_manager.list_subjects()
            self.ses_subject_combo.addItems(subjects)
            self.ses_subject_combo.blockSignals(False)
            if subjects:
                self._on_subject_changed(self.ses_subject_combo.currentText())
        except Exception as e:
            self._log(f"Could not list subjects: {e}")

    @Slot(str)
    def _on_subject_changed(self, subject: str) -> None:
        self.ses_session_combo.clear()
        if not subject or self.data_manager is None:
            return
        try:
            sessions = self.data_manager.list_sessions(subject)
            self.ses_session_combo.addItems(sessions)
            if sessions:
                self.ses_session_combo.setCurrentText(sessions[-1])
        except Exception as e:
            self._log(f"Could not list sessions: {e}")

    @Slot()
    def _on_calibrate_from_session(self) -> None:
        subject    = self.ses_subject_combo.currentText()
        session_id = self.ses_session_combo.currentText()
        if not subject or not session_id:
            QMessageBox.warning(
                self, "Nothing selected",
                "Select a subject and session.\n"
                "If the list is empty, record a session first."
            )
            return

        self.ses_cal_btn.setEnabled(False)
        self.ses_status_lbl.setText("Loading session…")
        self._log(f"Loading {subject}/{session_id}…")

        try:
            session = self.data_manager.load_session(subject, session_id)
        except Exception as e:
            self._log(f"Load error: {e}")
            self.ses_cal_btn.setEnabled(True)
            self.ses_status_lbl.setText("Load failed.")
            QMessageBox.critical(self, "Load error", str(e))
            return

        if not getattr(session, "trials", None):
            self.ses_cal_btn.setEnabled(True)
            self.ses_status_lbl.setText("No trials in session.")
            QMessageBox.warning(
                self, "No trials",
                f"Session '{session_id}' has no recorded trials."
            )
            return

        self.ses_status_lbl.setText("Calibrating…")
        try:
            num_ch = getattr(getattr(session, "metadata", None),
                             "num_channels", None)
            if num_ch:
                self.calibrator.processor.num_channels = num_ch
            result = self.calibrator.calibrate_from_session(session)
        except Exception as e:
            self._log(f"Calibration error: {e}")
            self.ses_cal_btn.setEnabled(True)
            self.ses_status_lbl.setText("Calibration failed.")
            QMessageBox.critical(self, "Calibration failed", str(e))
            return

        self._calibration_result = result
        self.ses_cal_btn.setEnabled(True)
        self.finish_btn.setEnabled(True)
        self.calibration_complete.emit(result)
        self.ses_status_lbl.setText(
            f"✓ Offset: {result.rotation_offset} ch  "
            f"Confidence: {result.confidence:.2%}"
        )

        # Log details
        valid_trials = (session.get_valid_trials()
                        if hasattr(session, "get_valid_trials") else [])
        if valid_trials:
            gestures = {t.gesture_name for t in valid_trials}
            self._log(f"  Gestures: {', '.join(sorted(gestures))}")
            self._log(f"  Valid trials: {len(valid_trials)}")
        self._log(f"  Offset: {result.rotation_offset} ch  "
                  f"Confidence: {result.confidence:.2%}")

        # Auto-save as reference when none exists
        saved_new_ref = False
        if not self.calibrator.has_reference:
            self.calibrator.save_as_reference(result)
            self._log("  No reference existed — saved as reference.")
            saved_new_ref = True

        QMessageBox.information(
            self, "Calibration complete",
            f"{subject} / {session_id}\n"
            f"Offset: {result.rotation_offset} ch\n"
            f"Confidence: {result.confidence:.2%}"
            + ("\n\nSaved as new reference calibration." if saved_new_ref else "")
            + "\n\nClick 'Apply & Close' to use this calibration."
        )

        # Extract gesture arrays for the validator
        self._gesture_data = {}
        try:
            all_data = session.get_data()
            for trial in valid_trials:
                chunk = all_data[trial.start_sample:trial.end_sample]
                if chunk.shape[0] >= 10:
                    self._gesture_data.setdefault(trial.gesture_name, [])
                    self._gesture_data[trial.gesture_name].append(chunk)
            for g, chunks in list(self._gesture_data.items()):
                self._gesture_data[g] = np.vstack(chunks)
        except Exception:
            pass

        self._run_free_checks_internal()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _get_validator_class(self):
        try:
            from playagain_pipeline.calibration.calibration_validation import (
                CalibrationValidator,
            )
            return CalibrationValidator
        except ImportError:
            return None

    @Slot()
    def _on_run_free_checks(self) -> None:
        if self._calibration_result is None:
            QMessageBox.information(
                self, "No calibration yet",
                "Run a calibration first (Live or From Session tab)."
            )
            return
        if not self._gesture_data:
            QMessageBox.information(
                self, "No gesture data",
                "The free checks need the gesture data captured during calibration. "
                "Re-run calibration, then validate."
            )
            return
        self._run_free_checks_internal()

    def _run_free_checks_internal(self) -> None:
        ValidatorClass = self._get_validator_class()
        if ValidatorClass is None or not self._gesture_data:
            return
        try:
            report = ValidatorClass(self.calibrator).run_all(
                gesture_data=self._gesture_data
            )
        except Exception as e:
            self._log(f"Validation error: {e}")
            return

        self._log("Validation checks:")
        for check in report.checks:
            self._log(check.line())
        verdict = "✓ Looks good." if report.is_acceptable else "⚠ One or more checks failed."
        self._log(f"  → {verdict}")

        lines = [
            f"Offset: {report.rotation_offset}  Confidence: {report.confidence:.0%}",
            verdict, "",
        ] + [c.line() for c in report.checks]
        self.val_result_lbl.setText("\n".join(lines))

    @Slot()
    def _on_run_held_out_check(self) -> None:
        if self._calibration_result is None:
            QMessageBox.information(
                self, "No calibration yet",
                "Run a calibration first, then validate against a session."
            )
            return
        ValidatorClass = self._get_validator_class()
        if ValidatorClass is None:
            QMessageBox.warning(
                self, "Module missing",
                "calibration_validation.py not found. "
                "Add it to playagain_pipeline/calibration/."
            )
            return
        if self.data_manager is None:
            QMessageBox.warning(
                self, "No data manager",
                "Held-out validation needs a DataManager. "
                "Open the dialog from the main window."
            )
            return

        # Mini session picker
        picker = QDialog(self)
        picker.setWindowTitle("Pick held-out session")
        picker.resize(320, 140)
        pl = QVBoxLayout(picker)
        pf = QFormLayout()
        subj_combo = QComboBox()
        sess_combo = QComboBox()
        try:
            subj_combo.addItems(self.data_manager.list_subjects())
        except Exception:
            pass

        def _upd(s: str) -> None:
            sess_combo.clear()
            if not s:
                return
            try:
                sessions = self.data_manager.list_sessions(s)
                sess_combo.addItems(sessions)
                if sessions:
                    sess_combo.setCurrentText(sessions[-1])
            except Exception:
                pass

        subj_combo.currentTextChanged.connect(_upd)
        if subj_combo.count():
            _upd(subj_combo.currentText())
        pf.addRow("Subject:", subj_combo)
        pf.addRow("Session:", sess_combo)
        pl.addLayout(pf)
        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(picker.accept)
        btns.rejected.connect(picker.reject)
        pl.addWidget(btns)

        if not picker.exec():
            return

        subject    = subj_combo.currentText()
        session_id = sess_combo.currentText()
        if not subject or not session_id:
            return

        try:
            session = self.data_manager.load_session(subject, session_id)
        except Exception as e:
            QMessageBox.critical(self, "Load error", str(e))
            return

        try:
            report = ValidatorClass(self.calibrator).run_all(
                gesture_data=self._gesture_data,
                held_out_session=session,
            )
        except Exception as e:
            QMessageBox.warning(self, "Validation failed", str(e))
            return

        self._log(f"Held-out validation vs {session_id}:")
        for check in report.checks:
            self._log(check.line())
        for g, acc in sorted(report.per_gesture_accuracy.items()):
            self._log(f"  {g}: {acc:.2%}")

        lines = [
            f"Offset: {report.rotation_offset}  Confidence: {report.confidence:.0%}",
            f"{'✓ Acceptable' if report.is_acceptable else '⚠ Failed'}",
            f"Held-out: {session_id}", "",
        ] + [c.line() for c in report.checks]
        if report.per_gesture_accuracy:
            lines += ["", "Per-gesture accuracy:"]
            lines += [f"  {g}: {acc:.2%}"
                      for g, acc in sorted(report.per_gesture_accuracy.items())]
        self.val_result_lbl.setText("\n".join(lines))
        QMessageBox.information(self, "Held-out validation", report.summary())

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _log(self, msg: str) -> None:
        self.log_text.append(msg)

    def get_calibration_result(self):
        return self._calibration_result

    def closeEvent(self, event) -> None:
        self._recording_timer.stop()
        self._live_disconnect()
        self._is_recording = False
        super().closeEvent(event)