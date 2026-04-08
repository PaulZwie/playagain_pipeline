"""
busy_overlay.py
───────────────
A reusable semi-transparent loading overlay + a thread runner helper.

Usage
─────
# Show overlay over any widget while a blocking function runs on a thread:
from playagain_pipeline.gui.widgets.busy_overlay import run_blocking

def my_slow_work():           # runs on background thread – NO Qt calls here
    result = heavy_compute()
    return result

run_blocking(
    parent_widget   = self,          # overlay shown over this widget
    fn              = my_slow_work,  # the blocking function
    on_done         = lambda r: self._handle_result(r),   # called on main thread
    on_error        = lambda e: QMessageBox.critical(self, "Error", str(e)),
    label           = "Creating dataset…",
)

BusyOverlay can also be used standalone:
    overlay = BusyOverlay(self, "Loading…")
    overlay.show()
    ...
    overlay.hide()
"""
from __future__ import annotations

import math
import traceback
from typing import Any, Callable, Optional

from PySide6.QtCore import (
    QEasingCurve, QPropertyAnimation, QRect, QRectF,
    QThread, QTimer, Qt, Signal, Slot,
)
from PySide6.QtGui import QBrush, QColor, QPainter, QPen, QFont
from PySide6.QtWidgets import QWidget


# ---------------------------------------------------------------------------
# Spinner canvas
# ---------------------------------------------------------------------------

class _SpinnerWidget(QWidget):
    """A lightweight arc-spinner drawn with QPainter — no external assets."""

    def __init__(self, parent: Optional[QWidget] = None, size: int = 56):
        super().__init__(parent)
        self._size   = size
        self._angle  = 0
        self._timer  = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(16)   # ~60 fps
        self.setFixedSize(size, size)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

    def _tick(self):
        self._angle = (self._angle + 6) % 360
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        s = self._size
        margin = 5
        rect = QRectF(margin, margin, s - 2 * margin, s - 2 * margin)

        # Track ring
        pen = QPen(QColor(63, 63, 92), 4, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawArc(rect, 0, 360 * 16)

        # Spinning arc
        pen.setColor(QColor(6, 182, 212))   # cyan accent
        pen.setWidth(4)
        p.setPen(pen)
        start_angle = (self._angle) * 16
        span_angle  = 100 * 16             # 100° arc
        p.drawArc(rect, start_angle, span_angle)


# ---------------------------------------------------------------------------
# Overlay
# ---------------------------------------------------------------------------

class BusyOverlay(QWidget):
    """
    Semi-transparent dark overlay with a spinner + status label.
    Place it over any parent widget.
    """

    def __init__(self, parent: QWidget, label: str = "Working…"):
        super().__init__(parent)
        self._label = label

        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, False)
        self.setAutoFillBackground(False)

        # Intercept all mouse/key events so the user can't click through
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self._spinner = _SpinnerWidget(self, size=52)
        self._resize_to_parent()
        self.hide()

    # ------------------------------------------------------------------

    def set_label(self, text: str):
        self._label = text
        self.update()

    def show_over(self, text: Optional[str] = None):
        if text:
            self._label = text
        self._resize_to_parent()
        self.raise_()
        self.show()
        self.update()

    def _resize_to_parent(self):
        if self.parent():
            self.setGeometry(self.parent().rect())
        cx = self.width() // 2
        cy = self.height() // 2
        self._spinner.move(cx - self._spinner.width() // 2,
                           cy - self._spinner.height() // 2 - 18)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._resize_to_parent()

    # ------------------------------------------------------------------

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Dark scrim
        p.fillRect(self.rect(), QColor(15, 15, 30, 190))

        # Card behind text
        cx = self.width()  // 2
        cy = self.height() // 2
        card_w, card_h = 220, 84
        card_rect = QRect(cx - card_w // 2, cy - card_h // 2, card_w, card_h)
        p.setBrush(QBrush(QColor(42, 42, 62)))
        p.setPen(QPen(QColor(63, 63, 92), 1))
        p.drawRoundedRect(card_rect, 10, 10)

        # Label text
        font = QFont()
        font.setPointSize(11)
        font.setBold(True)
        p.setFont(font)
        p.setPen(QColor(226, 232, 240))
        text_rect = QRect(cx - card_w // 2, cy + 12, card_w, 32)
        p.drawText(text_rect, Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
                   self._label)

    # Eat all mouse/key events while visible so UI stays frozen
    def mousePressEvent(self, e):   e.accept()
    def mouseReleaseEvent(self, e): e.accept()
    def keyPressEvent(self, e):     e.accept()


# ---------------------------------------------------------------------------
# Background worker thread
# ---------------------------------------------------------------------------

class _Worker(QThread):
    succeeded = Signal(object)   # result
    failed    = Signal(str)      # traceback string

    def __init__(self, fn: Callable, *args, **kwargs):
        super().__init__()
        self._fn     = fn
        self._args   = args
        self._kwargs = kwargs

    def run(self):
        try:
            result = self._fn(*self._args, **self._kwargs)
            self.succeeded.emit(result)
        except Exception:
            self.failed.emit(traceback.format_exc())


# ---------------------------------------------------------------------------
# Public helper
# ---------------------------------------------------------------------------

def run_blocking(
    parent_widget: QWidget,
    fn: Callable[[], Any],
    on_done: Callable[[Any], None],
    on_error: Optional[Callable[[str], None]] = None,
    label: str = "Working…",
) -> _Worker:
    """
    Run *fn* on a background thread while showing a BusyOverlay over
    *parent_widget*.  Calls *on_done(result)* or *on_error(traceback_str)*
    on the main thread when complete.

    Returns the _Worker (QThread) so the caller can keep a reference to
    prevent premature garbage collection.
    """
    # Create or reuse overlay
    overlay = getattr(parent_widget, "_busy_overlay", None)
    if overlay is None or not isinstance(overlay, BusyOverlay):
        overlay = BusyOverlay(parent_widget)
        parent_widget._busy_overlay = overlay    # type: ignore[attr-defined]

    overlay.show_over(label)

    worker = _Worker(fn)

    def _done(result):
        overlay.hide()
        on_done(result)

    def _err(tb_str):
        overlay.hide()
        if on_error:
            on_error(tb_str)

    worker.succeeded.connect(_done)
    worker.failed.connect(_err)
    # Keep reference alive on the parent so Python GC doesn't collect it
    parent_widget._busy_worker = worker          # type: ignore[attr-defined]
    worker.start()
    return worker
