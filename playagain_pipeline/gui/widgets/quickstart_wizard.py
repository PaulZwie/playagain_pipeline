"""
quickstart_wizard.py
────────────────────
A very small, optional first-run wizard.

It does exactly four things:

  1. Welcomes the user and explains the three-step workflow.
  2. Asks for a subject id (with a sensible default).
  3. Asks if a device is currently plugged in, and if so, helps connect.
  4. Hands off to the Record tab.

Deliberate non-features:

  • No device auto-scan — the user's main window already knows how
    to connect; the wizard just surfaces that button. Everything else
    stays the same so the wizard doesn't become a second control
    surface with its own bugs.
  • No data entry beyond the subject id. The wizard is a
    self-contained introduction, not a replacement for participant
    info, settings, or configuration.

Integration
───────────

    dlg = QuickstartWizard(parent=self)
    dlg.set_default_subject("VP_01")
    dlg.connect_device_requested.connect(self._on_connect_device)
    dlg.subject_chosen.connect(self._on_set_subject)
    dlg.finished.connect(self._on_quickstart_finished)
    dlg.exec()

The wizard never touches the data manager directly; it emits signals
and lets the main window act.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QFont, QPixmap
from PySide6.QtWidgets import (
    QDialog, QFrame, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QStackedWidget, QVBoxLayout, QWidget,
)


# ---------------------------------------------------------------------------
# Step container helper
# ---------------------------------------------------------------------------

class _Step(QWidget):
    """A single step panel with optional title, subtitle, and free body."""

    def __init__(
        self,
        title: str,
        subtitle: str = "",
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self._lay = QVBoxLayout(self)
        self._lay.setContentsMargins(32, 28, 32, 12)
        self._lay.setSpacing(14)

        t = QLabel(title)
        tf = QFont(); tf.setBold(True); tf.setPointSize(16)
        t.setFont(tf)
        t.setStyleSheet("color: #0f172a;")
        self._lay.addWidget(t)

        if subtitle:
            s = QLabel(subtitle)
            s.setWordWrap(True)
            s.setStyleSheet("color: #475569; font-size: 11px;")
            self._lay.addWidget(s)

    def add_body(self, w: QWidget) -> None:
        self._lay.addWidget(w)

    def add_stretch(self) -> None:
        self._lay.addStretch(1)


# ---------------------------------------------------------------------------
# Public wizard
# ---------------------------------------------------------------------------

class QuickstartWizard(QDialog):
    """
    Modal wizard shown on first run (or from Help menu).

    Signals:
      subject_chosen(str)          — user confirmed subject id
      connect_device_requested()   — user clicked "Connect a device"
      jump_to_record_requested()   — user wants to start recording now
    """

    subject_chosen             = Signal(str)
    connect_device_requested   = Signal()
    jump_to_record_requested   = Signal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Welcome")
        self.setMinimumSize(620, 460)
        self.setModal(True)

        self._current_step = 0
        self._total_steps  = 3
        self._default_subject = "VP_01"

        self._setup_ui()

    # ------------------------------------------------------------------

    def set_default_subject(self, subject_id: str) -> None:
        """Prefill the subject input on step 2."""
        self._default_subject = subject_id or "VP_01"
        if self._subject_input is not None:
            self._subject_input.setText(self._default_subject)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        self.setStyleSheet(
            "QDialog { background: #f8fafc; }"
            "QPushButton { padding: 8px 18px; border-radius: 6px; font-size: 11px; }"
            "QPushButton[role='primary'] {"
            "  background: #0284c7; color: white; border: none; font-weight: 600;"
            "}"
            "QPushButton[role='primary']:hover { background: #0369a1; }"
            "QPushButton[role='secondary'] {"
            "  background: white; color: #0f172a; border: 1px solid #cbd5e1;"
            "}"
            "QPushButton[role='secondary']:hover { border-color: #0284c7; color: #0284c7; }"
            "QPushButton[role='ghost'] {"
            "  background: transparent; color: #64748b; border: none;"
            "}"
            "QPushButton[role='ghost']:hover { color: #0284c7; }"
            "QLineEdit {"
            "  padding: 8px 10px; border: 1px solid #cbd5e1;"
            "  border-radius: 6px; background: white; font-size: 12px;"
            "}"
            "QLineEdit:focus { border-color: #0284c7; }"
        )

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Step indicator at top
        self._indicator_host = QFrame()
        self._indicator_host.setStyleSheet(
            "QFrame { background: white; border-bottom: 1px solid #e2e8f0; }"
        )
        ind_lay = QHBoxLayout(self._indicator_host)
        ind_lay.setContentsMargins(20, 10, 20, 10)
        self._step_dots: list[QLabel] = []
        for i in range(self._total_steps):
            dot = QLabel()
            dot.setFixedSize(10, 10)
            dot.setStyleSheet(
                "background: #cbd5e1; border-radius: 5px;"
            )
            ind_lay.addWidget(dot)
            self._step_dots.append(dot)
        ind_lay.addStretch()
        self._step_label = QLabel("")
        self._step_label.setStyleSheet("color: #64748b; font-size: 10px;")
        ind_lay.addWidget(self._step_label)
        root.addWidget(self._indicator_host)

        # Stacked step content
        self._stack = QStackedWidget()
        self._subject_input: Optional[QLineEdit] = None
        self._stack.addWidget(self._build_step_welcome())
        self._stack.addWidget(self._build_step_subject())
        self._stack.addWidget(self._build_step_device())
        root.addWidget(self._stack, 1)

        # Navigation footer
        nav = QFrame()
        nav.setStyleSheet(
            "QFrame { background: white; border-top: 1px solid #e2e8f0; }"
        )
        nav_lay = QHBoxLayout(nav)
        nav_lay.setContentsMargins(20, 12, 20, 12)

        self._skip_btn = QPushButton("Skip for now")
        self._skip_btn.setProperty("role", "ghost")
        self._skip_btn.clicked.connect(self.reject)
        nav_lay.addWidget(self._skip_btn)

        nav_lay.addStretch()

        self._back_btn = QPushButton("‹  Back")
        self._back_btn.setProperty("role", "secondary")
        self._back_btn.clicked.connect(self._go_back)
        nav_lay.addWidget(self._back_btn)

        self._next_btn = QPushButton("Next  ›")
        self._next_btn.setProperty("role", "primary")
        self._next_btn.clicked.connect(self._go_next)
        nav_lay.addWidget(self._next_btn)

        root.addWidget(nav)

        self._update_indicator()
        self._update_nav_state()

    # ------------------------------------------------------------------
    # Step 1 — welcome
    # ------------------------------------------------------------------

    def _build_step_welcome(self) -> QWidget:
        step = _Step(
            "Welcome to the EMG pipeline",
            "This quickstart takes about 60 seconds. It walks you through "
            "the three things you'll do most often — record, train, and "
            "use a model live. You can skip at any time.",
        )

        # A simple visual showing the three-step flow
        flow = QFrame()
        flow.setStyleSheet(
            "QFrame { background: white; border: 1px solid #e2e8f0; "
            "  border-radius: 10px; }"
        )
        flow_lay = QVBoxLayout(flow)
        flow_lay.setContentsMargins(18, 18, 18, 18)
        flow_lay.setSpacing(12)

        for num, title, desc in [
            ("1", "Record",
             "Connect a device and capture a few minutes of EMG while "
             "performing gestures on cue."),
            ("2", "Train",
             "Build a dataset from your recordings and train a model "
             "(classical or deep)."),
            ("3", "Use",
             "Run the model live on the EMG stream or pipe its predictions "
             "into Unity."),
        ]:
            row = QHBoxLayout()
            row.setSpacing(12)
            badge = QLabel(num)
            badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
            badge.setFixedSize(28, 28)
            bf = QFont(); bf.setBold(True); bf.setPointSize(11)
            badge.setFont(bf)
            badge.setStyleSheet(
                "background: #0284c7; color: white; border-radius: 14px;"
            )
            row.addWidget(badge, 0, Qt.AlignmentFlag.AlignTop)

            col = QVBoxLayout()
            col.setSpacing(0)
            t = QLabel(title)
            tf = QFont(); tf.setBold(True); tf.setPointSize(12)
            t.setFont(tf)
            t.setStyleSheet("color: #0f172a;")
            col.addWidget(t)
            d = QLabel(desc)
            d.setWordWrap(True)
            d.setStyleSheet("color: #475569; font-size: 10px;")
            col.addWidget(d)
            row.addLayout(col, 1)

            flow_lay.addLayout(row)

        step.add_body(flow)
        step.add_stretch()
        return step

    # ------------------------------------------------------------------
    # Step 2 — subject id
    # ------------------------------------------------------------------

    def _build_step_subject(self) -> QWidget:
        step = _Step(
            "Who are you recording?",
            "Each recording is tagged with a subject id so you can group "
            "them later. Use something like 'VP_01' for research subjects, "
            "or your own name for testing.",
        )

        self._subject_input = QLineEdit()
        self._subject_input.setText(self._default_subject)
        self._subject_input.setPlaceholderText("e.g. VP_01")
        self._subject_input.textChanged.connect(self._update_nav_state)
        step.add_body(self._subject_input)

        hint = QLabel(
            "You can change this later from the top status bar — "
            "just click the Subject pill."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #94a3b8; font-size: 10px; font-style: italic;")
        step.add_body(hint)

        step.add_stretch()
        return step

    # ------------------------------------------------------------------
    # Step 3 — device
    # ------------------------------------------------------------------

    def _build_step_device(self) -> QWidget:
        step = _Step(
            "Do you have a device plugged in?",
            "You can connect it now or skip this — the app is fine to "
            "explore without hardware. Previously recorded sessions are "
            "available for training either way.",
        )

        btn_row = QHBoxLayout()
        btn_row.setSpacing(12)

        connect_btn = QPushButton("  Connect a device  ")
        connect_btn.setProperty("role", "secondary")
        connect_btn.clicked.connect(self._on_connect_clicked)
        btn_row.addWidget(connect_btn)

        skip_btn = QPushButton("  Skip — I'll do it later  ")
        skip_btn.setProperty("role", "ghost")
        skip_btn.clicked.connect(self._finish_wizard)
        btn_row.addWidget(skip_btn)

        btn_row.addStretch()
        wrap = QWidget()
        wrap.setLayout(btn_row)
        step.add_body(wrap)

        step.add_stretch()

        note = QLabel(
            "ℹ  Nothing is saved until you click Next. Your subject id "
            f"('{self._default_subject}') will be remembered across sessions."
        )
        note.setWordWrap(True)
        note.setStyleSheet(
            "color: #0284c7; font-size: 10px; "
            "background: #eff6ff; padding: 10px; border-radius: 6px;"
        )
        step.add_body(note)

        return step

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def _update_indicator(self) -> None:
        for i, dot in enumerate(self._step_dots):
            if i < self._current_step:
                dot.setStyleSheet("background: #16a34a; border-radius: 5px;")
            elif i == self._current_step:
                dot.setStyleSheet("background: #0284c7; border-radius: 5px;")
            else:
                dot.setStyleSheet("background: #cbd5e1; border-radius: 5px;")
        self._step_label.setText(
            f"Step {self._current_step + 1} of {self._total_steps}"
        )

    def _update_nav_state(self) -> None:
        self._back_btn.setEnabled(self._current_step > 0)
        # Subject step must have non-empty text to proceed.
        if self._current_step == 1 and self._subject_input is not None:
            text = self._subject_input.text().strip()
            self._next_btn.setEnabled(bool(text))
        else:
            self._next_btn.setEnabled(True)

        # Last step → next becomes "Finish"
        if self._current_step == self._total_steps - 1:
            self._next_btn.setText("Finish  ✓")
        else:
            self._next_btn.setText("Next  ›")

    @Slot()
    def _go_next(self) -> None:
        if self._current_step == 1 and self._subject_input is not None:
            subject_id = self._subject_input.text().strip()
            if subject_id:
                self.subject_chosen.emit(subject_id)

        if self._current_step >= self._total_steps - 1:
            self._finish_wizard()
            return

        self._current_step += 1
        self._stack.setCurrentIndex(self._current_step)
        self._update_indicator()
        self._update_nav_state()

    @Slot()
    def _go_back(self) -> None:
        if self._current_step <= 0:
            return
        self._current_step -= 1
        self._stack.setCurrentIndex(self._current_step)
        self._update_indicator()
        self._update_nav_state()

    @Slot()
    def _on_connect_clicked(self) -> None:
        self.connect_device_requested.emit()
        # Don't auto-advance — the main window may show a connect
        # dialog, and the user can come back and click Finish when
        # they're ready.

    def _finish_wizard(self) -> None:
        self.jump_to_record_requested.emit()
        self.accept()
