"""
status_strip.py
───────────────
A thin, always-visible status strip for the redesigned main window.

The strip sits between the title bar and the tab widget. It surfaces
the three pieces of context that a user needs to orient themselves
at any moment:

  • device status  — are we connected? to what?
  • subject        — who's the current participant?
  • active model   — which model would Use Model tab run right now?

The strip is stateless — it's a pure view. The main window owns the
actual state and pushes updates in via the set_* methods. The strip
emits signals when the user clicks a pill so the window can route to
the appropriate tab / dialog.

Why not just use QStatusBar?
  Qt's QStatusBar is at the *bottom* of the window and doesn't
  accept rich content well. For a novice-facing UI the persistent
  orientation belongs at the top where the eye lands first.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPalette
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QSizePolicy, QWidget,
)


# ---------------------------------------------------------------------------
# A single status "pill"
# ---------------------------------------------------------------------------

class _Pill(QFrame):
    """
    A rounded-corner status widget with a coloured leading dot, a
    short label on top, and a primary value below. Clickable.
    """

    clicked = Signal()

    _DOT_COLORS = {
        "green":   "#16a34a",
        "amber":   "#f59e0b",
        "red":     "#dc2626",
        "grey":    "#94a3b8",
        "blue":    "#0284c7",
        "muted":   "#cbd5e1",
    }

    def __init__(self, label: str, value: str = "—", dot: str = "grey",
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._label_text = label
        self._value_text = value
        self._dot_color = dot
        self._clickable = True

        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)

        outer = QHBoxLayout(self)
        outer.setContentsMargins(10, 4, 12, 4)
        outer.setSpacing(8)

        # Dot — drawn by the widget itself via paintEvent so it keeps
        # its perfect circle even when the pill stretches.
        self._dot_size = 10
        dot_holder = QWidget()
        dot_holder.setFixedSize(self._dot_size, self._dot_size)
        outer.addWidget(dot_holder, 0, Qt.AlignmentFlag.AlignVCenter)
        self._dot_holder = dot_holder

        text_col = QWidget()
        text_lay = QHBoxLayout(text_col)
        text_lay.setContentsMargins(0, 0, 0, 0)
        text_lay.setSpacing(6)

        self._label = QLabel(label)
        f_label = QFont()
        f_label.setPointSize(9)
        self._label.setFont(f_label)
        self._label.setStyleSheet("color: #64748b;")
        text_lay.addWidget(self._label)

        self._value = QLabel(value)
        f_value = QFont()
        f_value.setPointSize(10)
        f_value.setBold(True)
        self._value.setFont(f_value)
        self._value.setStyleSheet("color: #0f172a;")
        text_lay.addWidget(self._value, 1)

        outer.addWidget(text_col, 1)

        self._apply_pill_style()
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    # ------------------------------------------------------------------

    def set_value(self, value: str) -> None:
        self._value_text = value
        self._value.setText(value)

    def set_dot(self, color_name: str) -> None:
        if color_name not in self._DOT_COLORS:
            color_name = "grey"
        self._dot_color = color_name
        self._dot_holder.update()

    def set_clickable(self, enabled: bool) -> None:
        self._clickable = enabled
        self.setCursor(
            Qt.CursorShape.PointingHandCursor if enabled
            else Qt.CursorShape.ArrowCursor
        )

    # ------------------------------------------------------------------

    def _apply_pill_style(self) -> None:
        self.setStyleSheet(
            "_Pill {"
            "  background: #f1f5f9;"
            "  border: 1px solid #e2e8f0;"
            "  border-radius: 12px;"
            "}"
            "_Pill:hover {"
            "  background: #e0f2fe;"
            "  border: 1px solid #7dd3fc;"
            "}"
        )

    def paintEvent(self, event):
        super().paintEvent(event)
        # Draw the coloured dot in the dot holder we reserved space for.
        dot_center = self._dot_holder.mapTo(
            self,
            self._dot_holder.rect().center(),
        )
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(self._DOT_COLORS.get(self._dot_color, "#94a3b8")))
        p.drawEllipse(
            dot_center.x() - self._dot_size // 2,
            dot_center.y() - self._dot_size // 2,
            self._dot_size, self._dot_size,
        )

    def mousePressEvent(self, event):
        if self._clickable and event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
            event.accept()
        else:
            super().mousePressEvent(event)


# ---------------------------------------------------------------------------
# Public strip
# ---------------------------------------------------------------------------

class StatusStrip(QWidget):
    """
    Persistent top status bar.

    Signals:
      device_pill_clicked  — user wants to connect / configure the device
      subject_pill_clicked — user wants to edit participant info
      model_pill_clicked   — user wants to switch to the Use Model tab
    """

    device_pill_clicked  = Signal()
    subject_pill_clicked = Signal()
    model_pill_clicked   = Signal()

    def __init__(self, app_name: str = "EMG Pipeline",
                 parent: Optional[QWidget] = None):
        super().__init__(parent)

        self.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed,
        )
        self.setAutoFillBackground(True)
        pal = self.palette()
        pal.setColor(QPalette.ColorRole.Window, QColor("#f8fafc"))
        self.setPalette(pal)
        self.setStyleSheet(
            "StatusStrip {"
            "  border-bottom: 1px solid #e2e8f0;"
            "}"
        )

        lay = QHBoxLayout(self)
        lay.setContentsMargins(12, 6, 12, 6)
        lay.setSpacing(10)

        # App name
        app_label = QLabel(app_name)
        app_font = QFont()
        app_font.setBold(True)
        app_font.setPointSize(11)
        app_label.setFont(app_font)
        app_label.setStyleSheet("color: #0284c7;")
        lay.addWidget(app_label)

        lay.addSpacing(16)

        # Pills
        self._device_pill = _Pill("Device", "Not connected", "grey")
        self._device_pill.clicked.connect(self.device_pill_clicked.emit)
        lay.addWidget(self._device_pill)

        self._subject_pill = _Pill("Subject", "—", "grey")
        self._subject_pill.clicked.connect(self.subject_pill_clicked.emit)
        lay.addWidget(self._subject_pill)

        self._model_pill = _Pill("Model", "None loaded", "grey")
        self._model_pill.clicked.connect(self.model_pill_clicked.emit)
        lay.addWidget(self._model_pill)

        lay.addStretch(1)

        # Trailing "help" link — one-line cheatsheet, keeps the user
        # oriented without opening a menu.
        self._help_label = QLabel(
            "<a href='#help' style='color:#64748b;text-decoration:none;'>"
            "Help</a>"
        )
        self._help_label.setTextFormat(Qt.TextFormat.RichText)
        self._help_label.setOpenExternalLinks(False)
        self._help_label.linkActivated.connect(
            lambda *_: self.help_requested.emit()
        )
        lay.addWidget(self._help_label)

    # Extra signal for the help link; declared after __init__ so
    # documentation order reads naturally.
    help_requested = Signal()

    # ------------------------------------------------------------------
    # Public API — called by the main window whenever state changes.
    # ------------------------------------------------------------------

    def set_device_status(
        self,
        state: str = "disconnected",
        name: Optional[str] = None,
    ) -> None:
        """
        Update the device pill.

        state ∈ {"connected", "connecting", "disconnected", "error"}
        name  — device name (shown when connected) or friendly label
        """
        state = (state or "disconnected").lower()
        dot_map = {
            "connected":    "green",
            "connecting":   "amber",
            "disconnected": "grey",
            "error":        "red",
        }
        label_map = {
            "connected":    name or "Connected",
            "connecting":   "Connecting…",
            "disconnected": "Not connected",
            "error":        name or "Error",
        }
        self._device_pill.set_dot(dot_map.get(state, "grey"))
        self._device_pill.set_value(label_map.get(state, "Not connected"))

    def set_subject(self, subject_id: Optional[str]) -> None:
        """Update the subject pill. Pass None / empty string to clear."""
        if subject_id:
            self._subject_pill.set_value(subject_id)
            self._subject_pill.set_dot("blue")
        else:
            self._subject_pill.set_value("—")
            self._subject_pill.set_dot("grey")

    def set_active_model(
        self,
        model_name: Optional[str],
        ready: bool = False,
    ) -> None:
        """
        Update the active-model pill.

        model_name — e.g. "lda_pinch_vs_rest"; None clears
        ready      — True if loaded into memory (green), False if just
                     known by name (blue)
        """
        if not model_name:
            self._model_pill.set_value("None loaded")
            self._model_pill.set_dot("grey")
            return
        self._model_pill.set_value(model_name)
        self._model_pill.set_dot("green" if ready else "blue")
