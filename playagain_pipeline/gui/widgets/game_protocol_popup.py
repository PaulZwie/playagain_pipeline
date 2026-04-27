"""
game_protocol_popup.py
──────────────────────
Floating popup window that mirrors the gesture cue currently active
in the Unity training game and exposes manual controls for debugging
without a real EMG signal.

What it shows
─────────────
  • Current target gesture (emoji + name + description), updated
    whenever ``TrainingGameCoordinator`` advances a trial.
  • Trial progress: "Trial 3 / 15".
  • Coordinator state pill: ``waiting for Unity`` / ``running`` /
    ``complete`` / ``stopped``.
  • A live RMS meter showing the latest measurement against the
    rolling baseline — handy for tuning ``easy_mode_sensitivity``
    visually.

What it lets you do
───────────────────
  • **Force Feed Animal.** Fires the current trial's easy-mode
    broadcast immediately, ignoring the RMS check. Solves the
    "the synthetic replay never crosses threshold so the animal
    never gets fed" problem you hit during dry-run testing.
  • **Skip.** Ends the current trial as invalid and advances to
    the next one — useful when a recording goes off the rails.
  • **Manual gesture buttons.** One button per gesture in the
    set ("rest", "fist", "pinch", "tripod"). Clicking one
    broadcasts a confidence-1.0 prediction for that gesture
    *independent of the schedule*. Lets you sanity-check the
    Unity-side TargetGestureAnimalMapper without waiting for a
    real trigger.
  • **Replay session picker (optional).** When the active device
    is the synthetic replay device, the popup shows a dropdown
    of available recorded sessions so you can pick which one to
    play back. The actual replay engine lives elsewhere — this
    widget just emits the chosen path.

Usage
─────
    popup = GameProtocolPopup(parent=main_window)
    popup.set_gesture_set(session.gesture_set)
    popup.set_total_trials(len(schedule))

    coord.trial_started.connect(lambda spec, idx:
        popup.set_current_trial(spec, idx))
    coord.state_changed.connect(popup.set_state)
    coord.rms_updated.connect(popup.set_rms)
    coord.all_complete.connect(lambda: popup.set_state("complete"))

    popup.force_feed_requested.connect(coord.force_trigger)
    popup.skip_requested.connect(coord.skip_current_trial)
    popup.manual_gesture_requested.connect(coord.fire_manual_gesture)
    popup.replay_session_requested.connect(self._on_replay_pick)

    popup.show()
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QCloseEvent, QFont
from PySide6.QtWidgets import (
    QComboBox, QDialog, QFrame, QHBoxLayout, QLabel, QProgressBar,
    QPushButton, QSizePolicy, QVBoxLayout, QWidget,
)

# Imports kept lazy at the top — Gesture / TrialSpec are referenced
# only as type hints, so they fall through ``object`` typing if the
# pipeline isn't installed (e.g. someone imports this widget standalone).
try:
    from playagain_pipeline.core.gesture import Gesture, GestureSet
except Exception:  # pragma: no cover
    Gesture = object  # type: ignore[assignment,misc]
    GestureSet = object  # type: ignore[assignment,misc]

try:
    from playagain_pipeline.training_game_coordinator import TrialSpec
except Exception:  # pragma: no cover
    TrialSpec = object  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# Public popup
# ---------------------------------------------------------------------------

class GameProtocolPopup(QDialog):
    """Synced gesture cue + manual control panel for the training game."""

    # Signals routed to the coordinator / main window
    force_feed_requested      = Signal()
    skip_requested            = Signal()
    manual_gesture_requested  = Signal(str)   # gesture name
    replay_session_requested  = Signal(str)   # path of selected session

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Training Game Protocol")
        self.setModal(False)
        # Don't pin on top — clinicians frequently alt-tab to inspect
        # the EMG plot or other windows while the game is running.
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, False)
        self.resize(540, 660)

        self._gesture_set: Optional[GestureSet] = None
        self._total_trials: int = 0

        self._build_ui()
        # Initialise to the idle visual so an opened-but-not-started
        # popup doesn't show stale cue content.
        self.set_state("idle")
        self.clear_current_trial()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Header ──────────────────────────────────────────────────
        header = QFrame()
        header.setObjectName("GameProtocolHeader")
        header.setStyleSheet(
            "#GameProtocolHeader {"
            "  background: #0f172a; border-bottom: 1px solid #1e293b;"
            "}"
        )
        header.setFixedHeight(38)
        h = QHBoxLayout(header)
        h.setContentsMargins(12, 0, 12, 0)

        self._state_dot = QLabel("●")
        self._state_dot.setStyleSheet("color: #94a3b8; font-size: 14px;")
        h.addWidget(self._state_dot)

        self._state_label = QLabel("Idle")
        self._state_label.setStyleSheet(
            "color: #e2e8f0; font-weight: 600; font-size: 12px;"
        )
        h.addWidget(self._state_label)
        h.addStretch(1)

        self._progress_label = QLabel("")
        self._progress_label.setStyleSheet("color: #94a3b8; font-size: 11px;")
        h.addWidget(self._progress_label)
        root.addWidget(header)

        # ── Gesture display ─────────────────────────────────────────
        gesture_box = QFrame()
        gesture_box.setStyleSheet(
            "QFrame { background: #f8fafc; border-bottom: 1px solid #e2e8f0; }"
        )
        gb = QVBoxLayout(gesture_box)
        gb.setContentsMargins(20, 18, 20, 18)
        gb.setSpacing(6)
        gb.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._emoji_label = QLabel("")
        self._emoji_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        emoji_font = QFont(); emoji_font.setPointSize(72)
        self._emoji_label.setFont(emoji_font)
        gb.addWidget(self._emoji_label)

        self._gesture_name_label = QLabel("—")
        self._gesture_name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        name_font = QFont(); name_font.setPointSize(20); name_font.setBold(True)
        self._gesture_name_label.setFont(name_font)
        self._gesture_name_label.setStyleSheet("color: #0f172a;")
        gb.addWidget(self._gesture_name_label)

        self._gesture_desc_label = QLabel("")
        self._gesture_desc_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._gesture_desc_label.setWordWrap(True)
        self._gesture_desc_label.setStyleSheet("color: #475569; font-size: 11px;")
        gb.addWidget(self._gesture_desc_label)

        root.addWidget(gesture_box)

        # ── RMS meter ───────────────────────────────────────────────
        rms_box = QFrame()
        rms_box.setStyleSheet("QFrame { background: white; padding: 6px; }")
        rms_lay = QVBoxLayout(rms_box)
        rms_lay.setContentsMargins(20, 8, 20, 8)
        rms_lay.setSpacing(2)

        rms_top = QHBoxLayout()
        rms_label = QLabel("Live RMS vs baseline:")
        rms_label.setStyleSheet("color: #475569; font-size: 10px;")
        rms_top.addWidget(rms_label)
        rms_top.addStretch(1)
        self._rms_value_label = QLabel("—")
        self._rms_value_label.setStyleSheet(
            "color: #0f172a; font-size: 10px; font-family: monospace;"
        )
        rms_top.addWidget(self._rms_value_label)
        rms_lay.addLayout(rms_top)

        # 0..200 means "0..2× baseline" — the trigger ratio default is
        # 1.8, so the bar visually crosses ~90% when it would fire.
        self._rms_bar = QProgressBar()
        self._rms_bar.setMinimum(0)
        self._rms_bar.setMaximum(200)
        self._rms_bar.setValue(0)
        self._rms_bar.setTextVisible(False)
        self._rms_bar.setFixedHeight(8)
        self._rms_bar.setStyleSheet(
            "QProgressBar { border: 1px solid #cbd5e1; border-radius: 4px; "
            "  background: #f1f5f9; }"
            "QProgressBar::chunk { background: #06b6d4; border-radius: 3px; }"
        )
        rms_lay.addWidget(self._rms_bar)
        root.addWidget(rms_box)

        # ── Primary action buttons ──────────────────────────────────
        primary = QFrame()
        primary.setStyleSheet("QFrame { background: white; }")
        p_lay = QHBoxLayout(primary)
        p_lay.setContentsMargins(20, 10, 20, 10)
        p_lay.setSpacing(10)

        self._force_feed_btn = QPushButton("🍎  Force Feed Animal")
        self._force_feed_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._force_feed_btn.setFixedHeight(36)
        self._force_feed_btn.setStyleSheet(
            "QPushButton { background: #16a34a; color: white; border: none;"
            "  border-radius: 4px; padding: 0 14px; font-weight: 600; }"
            "QPushButton:hover { background: #15803d; }"
            "QPushButton:disabled { background: #cbd5e1; color: #64748b; }"
        )
        self._force_feed_btn.clicked.connect(self.force_feed_requested.emit)
        p_lay.addWidget(self._force_feed_btn, 2)

        self._skip_btn = QPushButton("⏭  Skip")
        self._skip_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._skip_btn.setFixedHeight(36)
        self._skip_btn.setStyleSheet(
            "QPushButton { background: white; color: #475569;"
            "  border: 1px solid #cbd5e1; border-radius: 4px;"
            "  padding: 0 14px; font-weight: 600; }"
            "QPushButton:hover { border-color: #0284c7; color: #0284c7; }"
        )
        self._skip_btn.clicked.connect(self.skip_requested.emit)
        p_lay.addWidget(self._skip_btn, 1)
        root.addWidget(primary)

        # ── Manual gesture trigger row ──────────────────────────────
        manual = QFrame()
        manual.setStyleSheet(
            "QFrame { background: #f8fafc; border-top: 1px solid #e2e8f0; }"
        )
        m_outer = QVBoxLayout(manual)
        m_outer.setContentsMargins(20, 10, 20, 10)
        m_outer.setSpacing(6)

        m_caption = QLabel(
            "Manual gesture triggers (debug — independent of schedule):"
        )
        m_caption.setStyleSheet("color: #475569; font-size: 10px;")
        m_outer.addWidget(m_caption)

        # Filled in once set_gesture_set() is called.
        self._manual_btn_row = QHBoxLayout()
        self._manual_btn_row.setSpacing(6)
        m_outer.addLayout(self._manual_btn_row)
        root.addWidget(manual)

        # ── Replay picker (only visible if populated) ───────────────
        self._replay_box = QFrame()
        self._replay_box.setStyleSheet(
            "QFrame { background: white; border-top: 1px solid #e2e8f0; }"
        )
        r_lay = QHBoxLayout(self._replay_box)
        r_lay.setContentsMargins(20, 10, 20, 10)
        r_lay.setSpacing(8)
        r_caption = QLabel("Synthetic replay:")
        r_caption.setStyleSheet("color: #475569; font-size: 11px;")
        r_lay.addWidget(r_caption)

        self._replay_combo = QComboBox()
        self._replay_combo.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed,
        )
        r_lay.addWidget(self._replay_combo, 1)

        self._replay_play_btn = QPushButton("▶  Play")
        self._replay_play_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._replay_play_btn.setFixedHeight(28)
        self._replay_play_btn.clicked.connect(self._on_replay_play_clicked)
        r_lay.addWidget(self._replay_play_btn)
        root.addWidget(self._replay_box)
        # Hidden until set_replay_options gives us at least one entry.
        self._replay_box.hide()

        # ── Footer hint ─────────────────────────────────────────────
        footer = QFrame()
        footer.setStyleSheet(
            "QFrame { background: #f8fafc; border-top: 1px solid #e2e8f0; }"
        )
        f_lay = QVBoxLayout(footer)
        f_lay.setContentsMargins(20, 8, 20, 8)

        hint = QLabel(
            "Closing this window does not stop the game. "
            "Use the pipeline's Stop button to end the session."
        )
        hint.setStyleSheet("color: #94a3b8; font-size: 10px; font-style: italic;")
        hint.setWordWrap(True)
        f_lay.addWidget(hint)
        root.addWidget(footer)

    # ------------------------------------------------------------------
    # Configuration — call before / during start()
    # ------------------------------------------------------------------

    def set_gesture_set(self, gesture_set: "GestureSet") -> None:
        """
        Install the gesture set so the popup can render emoji and
        descriptions, and so it knows what manual-trigger buttons to
        show. Calling this twice rebuilds the manual-trigger row.
        """
        self._gesture_set = gesture_set

        # Clear existing manual-trigger buttons
        while self._manual_btn_row.count():
            item = self._manual_btn_row.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

        if gesture_set is None:
            return

        for gesture in gesture_set.gestures:
            btn = self._build_manual_button(gesture)
            self._manual_btn_row.addWidget(btn)
        self._manual_btn_row.addStretch(1)

    def _build_manual_button(self, gesture: "Gesture") -> QPushButton:
        """Per-gesture debug button — fires fire_manual_gesture(name)."""
        emoji = getattr(gesture, "emoji", None) or ""
        name = getattr(gesture, "display_name", None) or gesture.name
        btn = QPushButton(f"{emoji}  {name}".strip())
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setFixedHeight(28)
        btn.setStyleSheet(
            "QPushButton { background: white; color: #0f172a;"
            "  border: 1px solid #cbd5e1; border-radius: 14px;"
            "  padding: 0 12px; font-size: 11px; }"
            "QPushButton:hover { background: #06b6d4; color: white;"
            "                    border-color: #06b6d4; }"
        )
        btn.clicked.connect(
            lambda _checked=False, n=gesture.name: self.manual_gesture_requested.emit(n)
        )
        return btn

    def set_total_trials(self, total: int) -> None:
        self._total_trials = max(0, int(total))
        self._refresh_progress_label()

    def set_replay_options(self, paths: List[Path]) -> None:
        """
        Populate the replay-session dropdown. Pass an empty list to
        hide the picker (e.g. when the active device is not
        synthetic).
        """
        self._replay_combo.clear()
        if not paths:
            self._replay_box.hide()
            return
        for p in paths:
            self._replay_combo.addItem(p.name, str(p))
        self._replay_box.show()

    # ------------------------------------------------------------------
    # Live updates — wire to coordinator signals
    # ------------------------------------------------------------------

    @Slot(object, int)
    def set_current_trial(self, spec, index: int) -> None:
        """Update the cue display with the in-flight trial."""
        # Trial spec carries the gesture name; emoji/description come
        # from the gesture set for richer presentation.
        gesture_name = getattr(spec, "gesture_name", str(spec))
        gesture = None
        if self._gesture_set is not None:
            gesture = self._gesture_set.get_gesture(gesture_name)

        self._emoji_label.setText(getattr(gesture, "emoji", "") or "🎯")
        display = getattr(gesture, "display_name", None) or gesture_name.upper()
        self._gesture_name_label.setText(display)
        self._gesture_desc_label.setText(
            getattr(gesture, "description", "") if gesture else ""
        )

        # Refresh trial counter
        self._current_index = index
        self._refresh_progress_label()
        self._force_feed_btn.setEnabled(True)
        self._skip_btn.setEnabled(True)

    @Slot()
    def clear_current_trial(self) -> None:
        """Reset the cue to a neutral state (between trials / idle)."""
        self._emoji_label.setText("")
        self._gesture_name_label.setText("—")
        self._gesture_desc_label.setText("")
        self._force_feed_btn.setEnabled(False)
        self._skip_btn.setEnabled(False)

    @Slot(str)
    def set_state(self, state: str) -> None:
        """
        Update the header pill. Recognised states:
          ``idle``, ``waiting_for_unity``, ``calibrating``, ``running``,
          ``complete``, ``stopped``.
        """
        # Map state name -> (dot color, label text)
        spec = {
            "idle":              ("#94a3b8", "Idle"),
            "waiting_for_unity": ("#f59e0b", "Waiting for Unity…"),
            "calibrating":       ("#a855f7", "Calibrating baseline"),
            "running":           ("#16a34a", "Running"),
            "complete":          ("#0284c7", "Complete"),
            "stopped":           ("#94a3b8", "Stopped"),
        }.get(state, ("#94a3b8", state))
        color, label = spec
        self._state_dot.setStyleSheet(f"color: {color}; font-size: 14px;")
        self._state_label.setText(label)

        if state == "calibrating":
            # Override the gesture display with an explicit rest cue —
            # this is the moment where the child needs to hold still
            # so the detector can learn the noise floor.
            self._emoji_label.setText("🖐")
            self._gesture_name_label.setText("REST")
            self._gesture_desc_label.setText(
                "Keep your hand relaxed for a moment — "
                "we're learning your resting EMG."
            )
            # No trial yet, so Force Feed / Skip are not meaningful.
            self._force_feed_btn.setEnabled(False)
            self._skip_btn.setEnabled(False)
            return

        # Disable Force Feed / Skip outside the running window — they
        # only make sense during an active trial.
        is_running = (state == "running")
        if not is_running:
            self.clear_current_trial()
        # Manual buttons stay enabled in any state — they're for poking
        # Unity directly and that's useful even before / after a session.

    @Slot(float, float)
    def set_rms(self, rms: float, baseline: float) -> None:
        """Update the live RMS meter."""
        # Render as percentage of baseline (0–200%). 100% = at baseline,
        # ~180% = at the default trigger threshold, 200% = saturated.
        if baseline > 1e-12:
            ratio = rms / baseline
        else:
            ratio = 0.0
        bar_value = max(0, min(200, int(ratio * 100)))
        self._rms_bar.setValue(bar_value)
        self._rms_value_label.setText(
            f"{rms:.4f}   /   baseline {baseline:.4f}   ({ratio:.2f}×)"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _refresh_progress_label(self) -> None:
        idx = getattr(self, "_current_index", -1)
        if self._total_trials <= 0:
            self._progress_label.setText("")
        elif idx < 0:
            self._progress_label.setText(f"0 / {self._total_trials}")
        else:
            self._progress_label.setText(f"Trial {idx + 1} / {self._total_trials}")

    @Slot()
    def _on_replay_play_clicked(self) -> None:
        path = self._replay_combo.currentData()
        if path:
            self.replay_session_requested.emit(str(path))

    # ------------------------------------------------------------------
    # Qt overrides
    # ------------------------------------------------------------------

    def closeEvent(self, event: QCloseEvent) -> None:
        """Hide instead of destroying — main window owns the lifecycle."""
        event.ignore()
        self.hide()