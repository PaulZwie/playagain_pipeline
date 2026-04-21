"""
emg_plot_panel.py
─────────────────
A fixed sidebar panel that embeds the live EMG plot directly in the
main window's right-side area.

Context
───────
Previously the plot was opened via a floating button (see
``emg_plot_popout.py``). That button was convenient for quick
inspection but made it easy to forget the plot existed — users would
record for minutes without noticing a dead channel or clipping.

This panel puts the plot back in the main window where it belongs:
always visible (unless the user collapses it), always updating when
the device is connected, and always surfacing the bad-channel
checkboxes the recording pipeline relies on.

Drop-in compatibility
─────────────────────
The panel exposes the same public surface as ``EMGPlotPopoutButton``,
so swapping classes in ``main_window.py`` requires no other changes:

    bad_channels_updated : Signal(np.ndarray)
    num_channels         : int property
    update_data(data)    : forward a chunk to the plot
    set_num_channels(n)  : change channel count
    set_ground_truth(s)  : set the current gesture label
    clear()              : reset the buffer

Collapsed state
───────────────
Clicking the ``–`` in the header (or the narrow strip, once collapsed)
shrinks the panel to a 28-px-wide bookmark. While collapsed,
``update_data()`` is a no-op — the hot path in the main window's
``_on_data_received`` stays cheap even if a user hides the plot to
focus on the tabs.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QColor, QFont, QPainter
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QSizePolicy, QToolButton,
    QVBoxLayout, QWidget,
)

# Use the same inner plot the popout used so the visual is identical to
# what the user saw before. Fall back to the non-reworked module if the
# reworked one doesn't export ``EMGPlotWidget`` (some environments only
# expose ``EMGPlotWindow`` from the reworked module).
try:
    from playagain_pipeline.gui.widgets.emg_plot_reworked import EMGPlotWidget
except Exception:  # pragma: no cover — environment-dependent fallback
    from playagain_pipeline.gui.widgets.emg_plot import EMGPlotWidget


# ---------------------------------------------------------------------------
# Header with title + collapse/expand toggle
# ---------------------------------------------------------------------------

class _PanelHeader(QFrame):
    """Compact title bar sitting at the top of the plot panel."""

    toggle_clicked = Signal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("EMGPlotPanelHeader")
        self.setStyleSheet(
            "#EMGPlotPanelHeader {"
            "  background: #0f172a;"
            "  border-top-left-radius: 6px;"
            "  border-top-right-radius: 6px;"
            "}"
        )
        self.setFixedHeight(30)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(10, 0, 6, 0)
        lay.setSpacing(6)

        dot = QLabel("●")
        dot.setStyleSheet("color: #06b6d4; font-size: 12px;")
        lay.addWidget(dot)

        title = QLabel("Live EMG")
        tf = QFont(); tf.setBold(True); tf.setPointSize(10)
        title.setFont(tf)
        title.setStyleSheet("color: #e2e8f0;")
        lay.addWidget(title)

        lay.addStretch(1)

        self._toggle = QToolButton()
        self._toggle.setText("–")
        self._toggle.setToolTip("Collapse the live plot")
        self._toggle.setCursor(Qt.CursorShape.PointingHandCursor)
        self._toggle.setStyleSheet(
            "QToolButton {"
            "  color: #94a3b8; border: none; background: transparent;"
            "  font-size: 16px; font-weight: 700; padding: 0 8px;"
            "}"
            "QToolButton:hover { color: #06b6d4; }"
        )
        self._toggle.clicked.connect(self.toggle_clicked.emit)
        lay.addWidget(self._toggle)

    def set_collapsed(self, collapsed: bool) -> None:
        self._toggle.setText("+" if collapsed else "–")
        self._toggle.setToolTip(
            "Expand the live plot" if collapsed else "Collapse the live plot"
        )


# ---------------------------------------------------------------------------
# Narrow strip shown while the panel is collapsed
# ---------------------------------------------------------------------------

class _CollapsedStrip(QFrame):
    """
    28-px-wide vertical bookmark. Clicking it expands the panel. The
    label is drawn rotated so the user can still read "LIVE EMG"
    without a tooltip.
    """

    clicked = Signal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setFixedWidth(28)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet(
            "_CollapsedStrip {"
            "  background: #0f172a; border-radius: 4px;"
            "}"
            "_CollapsedStrip:hover {"
            "  background: #1e293b;"
            "}"
        )
        self.setToolTip("Click to expand the live EMG plot")

    def paintEvent(self, event):
        super().paintEvent(event)
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(QColor("#06b6d4"))
        f = p.font()
        f.setPointSize(10)
        f.setBold(True)
        p.setFont(f)
        # Rotate so the text reads bottom-to-top along the strip.
        p.translate(self.width() / 2, self.height() / 2)
        p.rotate(-90)
        # Centre roughly — this is a label, not a layout problem.
        p.drawText(-self.height() // 2 + 12, 4, "▶  LIVE EMG")

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
            event.accept()
        else:
            super().mousePressEvent(event)


# ---------------------------------------------------------------------------
# Public panel
# ---------------------------------------------------------------------------

class EMGPlotPanel(QWidget):
    """
    Fixed sidebar panel hosting ``EMGPlotWidget``.

    Parameters
    ----------
    num_channels : int
        Initial channel count. Updated automatically by
        ``set_num_channels`` as the device reports its real layout.
    sampling_rate : int
        Sampling rate forwarded to the inner plot.
    display_seconds : float
        Default time window shown on the plot.
    parent : QWidget
        Parent widget. Normally the right panel of the main splitter.
    """

    # Same signal shape as the original EMGPlotWidget / popout button
    # so the main window's existing slot connects unchanged.
    bad_channels_updated = Signal(np.ndarray)

    def __init__(
        self,
        num_channels: int = 32,
        sampling_rate: int = 2000,
        display_seconds: float = 10.0,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self._num_channels     = int(num_channels)
        self._sampling_rate    = int(sampling_rate)
        self._display_seconds  = float(display_seconds)
        self._collapsed        = False

        self.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding,
        )
        self.setMinimumWidth(320)

        # Root vertical layout — holds either the expanded tree or the
        # collapsed strip (only one is visible at a time).
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(0)

        # ── Expanded container ─────────────────────────────────────────
        self._expanded = QWidget()
        exp_lay = QVBoxLayout(self._expanded)
        exp_lay.setContentsMargins(0, 0, 0, 0)
        exp_lay.setSpacing(0)

        self._header = _PanelHeader(self._expanded)
        self._header.toggle_clicked.connect(self._on_toggle_clicked)
        exp_lay.addWidget(self._header)

        # The plot itself — created once, reused across collapse cycles
        # so we don't lose the user's bad-channel selection each time.
        self._plot = EMGPlotWidget(
            num_channels=self._num_channels,
            sampling_rate=self._sampling_rate,
            display_seconds=self._display_seconds,
            parent=self._expanded,
        )
        self._plot.bad_channels_updated.connect(self.bad_channels_updated.emit)

        # Bordered host gives the plot a clean "panel" silhouette
        # that matches the header above it.
        plot_host = QFrame()
        plot_host.setObjectName("EMGPlotHost")
        plot_host.setStyleSheet(
            "#EMGPlotHost {"
            "  background: #0b1220;"
            "  border: 1px solid #1e293b; border-top: none;"
            "  border-bottom-left-radius: 6px;"
            "  border-bottom-right-radius: 6px;"
            "}"
        )
        host_lay = QVBoxLayout(plot_host)
        host_lay.setContentsMargins(6, 6, 6, 6)
        host_lay.addWidget(self._plot)
        exp_lay.addWidget(plot_host, 1)

        root.addWidget(self._expanded, 1)

        # ── Collapsed variant ──────────────────────────────────────────
        self._strip = _CollapsedStrip(self)
        self._strip.clicked.connect(self._on_toggle_clicked)
        self._strip.hide()
        root.addWidget(self._strip, 0)

    # ------------------------------------------------------------------
    # Collapse / expand
    # ------------------------------------------------------------------

    @Slot()
    def _on_toggle_clicked(self) -> None:
        self.set_collapsed(not self._collapsed)

    def set_collapsed(self, collapsed: bool) -> None:
        """Programmatically collapse or expand the panel."""
        if collapsed == self._collapsed:
            return
        self._collapsed = collapsed
        if collapsed:
            self._expanded.hide()
            self._strip.show()
            self.setMinimumWidth(28)
            self.setMaximumWidth(32)
        else:
            self._strip.hide()
            self._expanded.show()
            self.setMinimumWidth(320)
            self.setMaximumWidth(16777215)  # Qt's QWIDGETSIZE_MAX
        self._header.set_collapsed(collapsed)

    def is_collapsed(self) -> bool:
        return self._collapsed

    # ------------------------------------------------------------------
    # Drop-in API — matches EMGPlotPopoutButton so main_window.py swaps
    # in one line.
    # ------------------------------------------------------------------

    @property
    def num_channels(self) -> int:
        return self._num_channels

    def update_data(self, data: np.ndarray) -> None:
        """
        Push a chunk of samples to the plot.

        No-op when the panel is collapsed so the data-reception hot
        path stays as cheap as it was with the popout button. A few
        seconds of history may be lost while collapsed; that's a fair
        trade for zero CPU cost.
        """
        if self._collapsed:
            return
        try:
            self._plot.update_data(data)
        except RuntimeError:
            # Widget destroyed mid-update (rare — e.g. app shutdown).
            pass

    def set_num_channels(self, n: int) -> None:
        n = int(n)
        if n == self._num_channels:
            return
        self._num_channels = n
        try:
            self._plot.set_num_channels(n)
        except RuntimeError:
            pass

    def set_ground_truth(self, label: str) -> None:
        try:
            self._plot.set_ground_truth(label)
        except RuntimeError:
            pass

    def clear(self) -> None:
        try:
            self._plot.clear()
        except RuntimeError:
            pass
