"""
workflow_stepper.py
───────────────────
A compact horizontal step-indicator banner for the main window.

It shows the user which stage of the EMG pipeline they are in and which
stages have already been completed. Each step is clickable — clicking
jumps the main tab widget to the corresponding tab.

States
──────
    pending  : grey, hollow circle
    current  : highlighted accent colour, filled circle, bold label
    done     : green, check-mark glyph

Usage
─────
    stepper = WorkflowStepper([
        ("Record",            "Connect a device and record gestures"),
        ("Train & Evaluate",  "Build a dataset and train a model"),
        ("Predict",           "Run the model live or with Unity"),
    ])
    layout.addWidget(stepper)

    stepper.set_current(0)
    stepper.mark_done(0)            # tick step 1
    stepper.set_current(1)          # advance highlight to step 2

    stepper.step_clicked.connect(my_tab_widget.setCurrentIndex)
"""

from __future__ import annotations

from typing import List, Tuple, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel, QFrame, QSizePolicy,
)


# ---------------------------------------------------------------------------
# Single-step circle + label
# ---------------------------------------------------------------------------

class _StepBadge(QFrame):
    """A clickable circle-with-number plus a title and subtitle."""

    clicked = Signal(int)

    _STATE_COLORS = {
        # state name -> (border, fill, text, glyph)
        "pending": ("#c7d2fe", "#ffffff", "#6b7280", ""),
        "current": ("#0284c7", "#0284c7", "#ffffff", ""),
        "done":    ("#16a34a", "#16a34a", "#ffffff", "✓"),
    }

    def __init__(self, index: int, number_text: str, title: str, subtitle: str,
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._index = index
        self._number_text = number_text
        self._state = "pending"

        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFrameShape(QFrame.Shape.NoFrame)

        outer = QHBoxLayout(self)
        outer.setContentsMargins(6, 4, 6, 4)
        outer.setSpacing(8)

        # Circle badge
        self._circle = QLabel(number_text)
        self._circle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._circle.setFixedSize(30, 30)
        f = QFont()
        f.setBold(True)
        f.setPointSize(11)
        self._circle.setFont(f)
        outer.addWidget(self._circle, 0, Qt.AlignmentFlag.AlignVCenter)

        # Title + subtitle (stacked)
        text_col = QVBoxLayout()
        text_col.setSpacing(0)
        text_col.setContentsMargins(0, 0, 0, 0)

        self._title = QLabel(title)
        title_font = QFont()
        title_font.setPointSize(11)
        title_font.setBold(True)
        self._title.setFont(title_font)
        text_col.addWidget(self._title)

        self._subtitle = QLabel(subtitle)
        sub_font = QFont()
        sub_font.setPointSize(9)
        self._subtitle.setFont(sub_font)
        self._subtitle.setStyleSheet("color: #6b7280;")
        text_col.addWidget(self._subtitle)

        outer.addLayout(text_col, 1)

        self._apply_state()

    # ------------------------------------------------------------------

    def state(self) -> str:
        return self._state

    def set_state(self, state: str):
        if state not in self._STATE_COLORS:
            return
        self._state = state
        self._apply_state()

    def _apply_state(self):
        border, fill, text_color, glyph = self._STATE_COLORS[self._state]
        glyph_text = glyph if glyph else self._number_text
        self._circle.setText(glyph_text)
        self._circle.setStyleSheet(
            f"QLabel {{"
            f"  border: 2px solid {border};"
            f"  background: {fill};"
            f"  color: {text_color};"
            f"  border-radius: 15px;"
            f"}}"
        )

        # Title color and weight echo the state
        if self._state == "current":
            self._title.setStyleSheet("color: #0284c7;")
        elif self._state == "done":
            self._title.setStyleSheet("color: #16a34a;")
        else:
            self._title.setStyleSheet("color: #374151;")

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self._index)
            event.accept()
        else:
            super().mousePressEvent(event)


# ---------------------------------------------------------------------------
# Connector chevron between badges
# ---------------------------------------------------------------------------

class _Connector(QLabel):
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__("›", parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        f = QFont()
        f.setPointSize(20)
        self.setFont(f)
        self.setStyleSheet("color: #c7d2fe;")
        self.setFixedWidth(16)


# ---------------------------------------------------------------------------
# Public widget
# ---------------------------------------------------------------------------

class WorkflowStepper(QWidget):
    """
    Horizontal step indicator banner.

    Pass a list of (title, subtitle) tuples in workflow order. Use
    set_current(idx) to highlight the active step, mark_done(idx) to tick
    a step as completed, and connect to step_clicked(int) to jump tabs.
    """

    step_clicked = Signal(int)

    def __init__(self, steps: List[Tuple[str, str]], parent: Optional[QWidget] = None):
        super().__init__(parent)

        self._badges: List[_StepBadge] = []
        self._current_idx: Optional[int] = None
        self._done: set[int] = set()

        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        wrap = QHBoxLayout(self)
        wrap.setContentsMargins(10, 6, 10, 6)
        wrap.setSpacing(2)

        for i, (title, subtitle) in enumerate(steps):
            badge = _StepBadge(i, str(i + 1), title, subtitle, parent=self)
            badge.clicked.connect(self.step_clicked.emit)
            self._badges.append(badge)
            wrap.addWidget(badge, 1)
            if i < len(steps) - 1:
                wrap.addWidget(_Connector(self), 0)

        self.setStyleSheet(
            "WorkflowStepper {"
            "  background: #f8fafc;"
            "  border-bottom: 1px solid #e2e8f0;"
            "}"
        )

    # ------------------------------------------------------------------

    def set_current(self, idx: int):
        """Highlight ``idx`` as the active step."""
        if not (0 <= idx < len(self._badges)):
            return
        self._current_idx = idx
        self._refresh()

    def mark_done(self, idx: int):
        """Mark step ``idx`` as completed (green check)."""
        if not (0 <= idx < len(self._badges)):
            return
        self._done.add(idx)
        self._refresh()

    def clear_done(self, idx: int):
        """Un-mark a previously-done step."""
        self._done.discard(idx)
        self._refresh()

    def reset(self):
        """Reset all steps to pending and forget the current one."""
        self._done.clear()
        self._current_idx = None
        self._refresh()

    # ------------------------------------------------------------------

    def _refresh(self):
        for i, badge in enumerate(self._badges):
            if i in self._done and i != self._current_idx:
                badge.set_state("done")
            elif i == self._current_idx:
                badge.set_state("current")
            else:
                badge.set_state("pending")
