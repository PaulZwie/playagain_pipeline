"""
emg_plot.py
───────────
Real-time multi-channel EMG visualization with per-channel "good / bad"
toggles.

Drop-in API-compatible replacement for the original ``EMGPlotWidget``.
The public contract matches the previous implementation exactly:

    widget = EMGPlotWidget(num_channels=32, sampling_rate=2000)
    widget.bad_channels_updated.connect(callback)   # emits np.ndarray (bool)
    widget.update_data(data)                        # (samples, channels)
    widget.set_num_channels(n)
    widget.set_ground_truth("Fist")
    widget.clear()

Internally:
  * Render backend is ``pyqtgraph`` (already a dependency of
    performance_tab and training_dialog — no new third-party deps).
  * Each channel is a separate ``PlotDataItem`` in a single plot,
    stacked vertically with a per-channel DC offset so the traces
    don't overlap.
  * A small checkbox column on the right of the plot lets the user
    toggle each channel good/bad. Disabling a channel greys its
    trace, fades its label, and includes the channel index in the
    ``bad_channels_updated`` signal the next time anything changes.
  * Data buffering is mutex-guarded so a producer thread
    (device → `update_data`) can't race with the 30 Hz render timer.
  * Heavy-traffic plots are downsampled via pyqtgraph's built-in
    ``setDownsampling(auto=True)`` so 2 kHz × 32 ch × 10 s stays
    interactive on a laptop.

Why this replaces the vispy version
────────────────────────────────────
The previous widget depends on the local ``gui_custom_elements.vispy``
package. That package carries the interactive channel-checkbox strip
alongside the plot. Environments without ``gui_custom_elements``
installed (and there are several — CI, collaborators, a fresh clone)
can't import the old module, which breaks the main window at startup.
This rewrite uses only pyqtgraph + PySide6 so the widget works
anywhere the rest of the app works.
"""
from __future__ import annotations

import numpy as np
from typing import List, Optional

from PySide6.QtCore import Qt, QMutex, QMutexLocker, QTimer, Signal, Slot
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QCheckBox, QFrame, QHBoxLayout, QLabel, QMainWindow, QPushButton,
    QScrollArea, QSizePolicy, QSpinBox, QVBoxLayout, QWidget,
)

import pyqtgraph as pg


# ---------------------------------------------------------------------------
# Visual constants
# ---------------------------------------------------------------------------

# Dark background to match the rest of the app — contrasts well with the
# warm-yellow trace colour, which is kept uniform across channels
# because users navigate by the numeric label on the Y axis, not colour.
_BG          = "#1e1e2e"
_FG          = "#e2e8f0"
_GRID        = "#2a2a3e"
_AXIS        = "#94a3b8"
_TRACE_GOOD  = "#38bdf8"     # cyan — visible on the dark background
_TRACE_BAD   = "#3f3f55"     # muted grey-blue for disabled channels
_SEPARATOR   = "#313145"
_HIGHLIGHT   = "#f59e0b"     # amber — gesture-label highlight


# ---------------------------------------------------------------------------
# A single channel row (checkbox + label)
# ---------------------------------------------------------------------------

class _ChannelRow(QFrame):
    """Checkbox + 'CH_N' label for one channel in the side panel."""

    toggled = Signal(int, bool)   # (channel_index, is_enabled)

    def __init__(self, index: int, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._index = index

        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setFixedHeight(18)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(4, 0, 4, 0)
        lay.setSpacing(4)

        self._check = QCheckBox()
        self._check.setChecked(True)
        self._check.toggled.connect(self._on_toggled)
        lay.addWidget(self._check)

        self._label = QLabel(f"CH {index + 1}")
        f = QFont()
        f.setPointSize(9)
        f.setFamily("Menlo")   # monospace keeps column width stable
        self._label.setFont(f)
        self._label.setStyleSheet(f"color: {_FG};")
        lay.addWidget(self._label, 1)

    # ------------------------------------------------------------------

    def is_enabled(self) -> bool:
        return self._check.isChecked()

    def set_enabled(self, enabled: bool, *, silent: bool = False) -> None:
        if silent:
            self._check.blockSignals(True)
        self._check.setChecked(enabled)
        self._update_label_style(enabled)
        if silent:
            self._check.blockSignals(False)

    def _on_toggled(self, checked: bool) -> None:
        self._update_label_style(checked)
        self.toggled.emit(self._index, checked)

    def _update_label_style(self, enabled: bool) -> None:
        if enabled:
            self._label.setStyleSheet(f"color: {_FG};")
        else:
            self._label.setStyleSheet(f"color: {_TRACE_BAD}; "
                                      f"text-decoration: line-through;")


# ---------------------------------------------------------------------------
# Main widget
# ---------------------------------------------------------------------------

class EMGPlotWidget(QWidget):
    """
    Real-time EMG plot with per-channel good / bad toggles.

    Signal:
      bad_channels_updated(np.ndarray)
        Emitted with a length-``num_channels`` boolean array whenever
        the user toggles a channel. ``True`` means the channel is
        enabled; ``False`` means it's marked bad.

    The ``bad_channels_updated`` signal name matches the old vispy
    widget's API exactly so the main window's existing
    ``_on_bad_channels_updated(lines_enabled)`` slot wires up unchanged.
    """

    bad_channels_updated = Signal(np.ndarray)

    def __init__(
        self,
        num_channels: int = 32,
        sampling_rate: int = 2000,
        display_seconds: float = 10.0,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)

        self.num_channels     = int(num_channels)
        self.sampling_rate    = int(sampling_rate)
        self.display_seconds  = float(display_seconds)
        self.display_samples  = int(display_seconds * sampling_rate)

        # Circular data buffer — producer thread writes, render timer reads.
        self._data_buffer = np.zeros(
            (self.display_samples, self.num_channels), dtype=np.float32,
        )
        self._buffer_index = 0
        self._buffer_full  = False
        self._data_mutex   = QMutex()

        # Per-channel state
        self._enabled: List[bool] = [True] * self.num_channels
        self._curves: List[pg.PlotDataItem] = []
        self._rows:   List[_ChannelRow]      = []

        # Autoscale state — recomputed once per second from current buffer
        self._y_spacing = 1.0            # distance between trace baselines
        self._y_scale   = 1.0            # per-trace vertical gain
        self._autoscale_enabled = True

        # Build UI *first* (creates the plot widget, side panel, etc.),
        # then create curves that live inside it.
        self._setup_ui()
        self._rebuild_curves_and_rows()

        # ~30 FPS render timer, decoupled from producer
        self._render_timer = QTimer(self)
        self._render_timer.setInterval(33)
        self._render_timer.timeout.connect(self._on_render_tick)
        self._render_timer.start()

        # Less-frequent autoscale timer — 1 Hz is plenty, autoscale
        # per frame would jitter visibly.
        self._autoscale_timer = QTimer(self)
        self._autoscale_timer.setInterval(1000)
        self._autoscale_timer.timeout.connect(self._recompute_autoscale)
        self._autoscale_timer.start()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        self.setStyleSheet(
            f"EMGPlotWidget {{ background: {_BG}; }}"
        )

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # ── Control strip (top) ───────────────────────────────────────
        controls = QFrame()
        controls.setStyleSheet(
            f"QFrame {{ background: {_BG}; "
            f"  border-bottom: 1px solid {_SEPARATOR}; }}"
        )
        cl = QHBoxLayout(controls)
        cl.setContentsMargins(8, 4, 8, 4)
        cl.setSpacing(10)

        cl.addWidget(_make_label("Display (s):"))
        self.display_spin = QSpinBox()
        self.display_spin.setRange(1, 30)
        self.display_spin.setValue(int(self.display_seconds))
        self.display_spin.valueChanged.connect(self._on_display_time_changed)
        self.display_spin.setFixedWidth(60)
        cl.addWidget(self.display_spin)

        self.autoscale_check = QCheckBox("Auto-scale Y")
        self.autoscale_check.setChecked(True)
        self.autoscale_check.toggled.connect(self._on_autoscale_toggled)
        self.autoscale_check.setStyleSheet(f"color: {_FG};")
        cl.addWidget(self.autoscale_check)

        cl.addSpacing(12)
        cl.addWidget(_make_label("Channels:", muted=True))

        self._mark_all_good_btn = QPushButton("All good")
        self._mark_all_good_btn.setStyleSheet(_button_qss())
        self._mark_all_good_btn.clicked.connect(self._mark_all_good)
        cl.addWidget(self._mark_all_good_btn)

        self._invert_btn = QPushButton("Invert")
        self._invert_btn.setStyleSheet(_button_qss())
        self._invert_btn.clicked.connect(self._invert_selection)
        cl.addWidget(self._invert_btn)

        cl.addStretch(1)

        self._bad_count_label = QLabel("0 bad")
        self._bad_count_label.setStyleSheet(f"color: {_AXIS}; font-size: 10px;")
        cl.addWidget(self._bad_count_label)

        outer.addWidget(controls)

        # ── Plot + channel panel (middle, stretchy) ──────────────────
        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)

        # Plot
        self.plot_widget = pg.PlotWidget(background=_BG)
        self.plot_widget.showGrid(x=True, y=False, alpha=0.15)
        self.plot_widget.setMouseEnabled(x=False, y=False)
        self.plot_widget.hideButtons()
        self.plot_widget.setLabel("bottom", "Time (s)",
                                  color=_AXIS, size="9pt")
        self.plot_widget.getPlotItem().setDownsampling(auto=True)
        self.plot_widget.getPlotItem().setClipToView(True)
        self.plot_widget.setXRange(-self.display_seconds, 0, padding=0)
        self.plot_widget.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding,
        )

        # Axis styling — subtle grey ticks against the dark bg
        for axis_name in ("left", "bottom"):
            ax = self.plot_widget.getAxis(axis_name)
            ax.setPen(pg.mkPen(_AXIS, width=1))
            ax.setTextPen(pg.mkPen(_AXIS))

        body.addWidget(self.plot_widget, 1)

        # Side panel with channel checkboxes — scrollable to handle
        # 64+ channel devices without eating vertical space.
        self._channel_panel_scroll = QScrollArea()
        self._channel_panel_scroll.setWidgetResizable(True)
        self._channel_panel_scroll.setFixedWidth(108)
        self._channel_panel_scroll.setStyleSheet(
            f"QScrollArea {{ border: none; border-left: 1px solid {_SEPARATOR}; "
            f"  background: {_BG}; }}"
        )
        self._channel_host = QWidget()
        self._channel_host.setStyleSheet(f"background: {_BG};")
        self._channel_layout = QVBoxLayout(self._channel_host)
        self._channel_layout.setContentsMargins(4, 6, 4, 6)
        self._channel_layout.setSpacing(1)
        self._channel_layout.addStretch(1)
        self._channel_panel_scroll.setWidget(self._channel_host)
        body.addWidget(self._channel_panel_scroll)

        body_wrap = QWidget()
        body_wrap.setLayout(body)
        outer.addWidget(body_wrap, 1)

        # ── Ground-truth label (bottom) ──────────────────────────────
        self._ground_truth_label = QLabel("Ground Truth: —")
        self._ground_truth_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._ground_truth_label.setFixedHeight(28)
        self._ground_truth_label.setStyleSheet(_gt_qss("idle"))
        outer.addWidget(self._ground_truth_label)

    # ------------------------------------------------------------------
    # Channel-row / curve rebuild (called on init + set_num_channels)
    # ------------------------------------------------------------------

    def _rebuild_curves_and_rows(self) -> None:
        """Tear down existing curves + rows and recreate for `num_channels`."""
        pi = self.plot_widget.getPlotItem()

        # Drop old curves
        for curve in self._curves:
            pi.removeItem(curve)
        self._curves.clear()

        # Drop old rows
        while self._channel_layout.count() > 1:
            item = self._channel_layout.takeAt(0)
            if item and item.widget():
                item.widget().deleteLater()
        self._rows.clear()
        self._enabled = [True] * self.num_channels

        # Create curves — all use the same pen so colour isn't overloaded;
        # disabled state is indicated by pen swap, not hue.
        for ch in range(self.num_channels):
            curve = pi.plot(
                x=np.array([], dtype=np.float32),
                y=np.array([], dtype=np.float32),
                pen=pg.mkPen(_TRACE_GOOD, width=1),
                skipFiniteCheck=True,
            )
            self._curves.append(curve)

        # Create rows
        for ch in range(self.num_channels):
            row = _ChannelRow(ch)
            row.toggled.connect(self._on_channel_toggled)
            self._channel_layout.insertWidget(
                self._channel_layout.count() - 1, row,
            )
            self._rows.append(row)

        # Y-axis ticks matching channel indices
        self._update_y_axis_ticks()
        self._update_bad_count_label()

    def _update_y_axis_ticks(self) -> None:
        """Label each channel's baseline with its index on the Y axis."""
        ax = self.plot_widget.getAxis("left")
        ticks = [
            ((self.num_channels - 1 - ch) * self._y_spacing, f"CH {ch + 1}")
            for ch in range(self.num_channels)
        ]
        ax.setTicks([ticks])
        lo = -self._y_spacing
        hi = self.num_channels * self._y_spacing
        self.plot_widget.setYRange(lo, hi, padding=0.02)

    # ------------------------------------------------------------------
    # Producer-side API (thread-safe)
    # ------------------------------------------------------------------

    def update_data(self, data: np.ndarray) -> None:
        """
        Push new EMG samples into the plot. Safe to call from any thread.

        Args:
            data: ``(n_samples, n_channels)`` or ``(n_samples,)`` for
                  single-channel streams. Channel count mismatch is
                  tolerated — missing channels get zeros, extras are
                  dropped.
        """
        if data is None or data.size == 0:
            return
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        if data.shape[1] != self.num_channels:
            # Tolerate mismatch: either pad with zeros or truncate.
            # This mirrors the old widget's behaviour exactly — keeping
            # it prevents a single out-of-shape batch from crashing the
            # producer when the device is briefly mis-configured.
            min_ch = min(data.shape[1], self.num_channels)
            if min_ch == 0:
                return
            padded = np.zeros(
                (data.shape[0], self.num_channels), dtype=np.float32,
            )
            padded[:, :min_ch] = data[:, :min_ch]
            data = padded

        self._add_data_to_buffer(data.astype(np.float32, copy=False))

    def _add_data_to_buffer(self, data: np.ndarray) -> None:
        with QMutexLocker(self._data_mutex):
            n = data.shape[0]
            if n >= self.display_samples:
                # Replace whole buffer — newest display_samples are kept.
                self._data_buffer[:] = data[-self.display_samples:, :]
                self._buffer_index = 0
                self._buffer_full  = True
                return
            end = (self._buffer_index + n) % self.display_samples
            if end < self._buffer_index:
                first = self.display_samples - self._buffer_index
                self._data_buffer[self._buffer_index:] = data[:first]
                self._data_buffer[:end] = data[first:]
                self._buffer_full = True
            else:
                self._data_buffer[self._buffer_index:end] = data
                if end == 0:
                    self._buffer_full = True
            self._buffer_index = end

    # ------------------------------------------------------------------
    # Render loop
    # ------------------------------------------------------------------

    def _on_render_tick(self) -> None:
        with QMutexLocker(self._data_mutex):
            ordered = self._get_ordered_data()

        n = ordered.shape[0]
        if n == 0:
            return

        # X axis shows the last `display_seconds` of time, ending at 0.
        # Render only the data we have — the axis is clamped so short
        # buffers render flush to the right.
        x = (np.arange(n, dtype=np.float32) - (self.display_samples - 1)) \
            / self.sampling_rate

        for ch in range(self.num_channels):
            curve = self._curves[ch]
            base = (self.num_channels - 1 - ch) * self._y_spacing
            
            if not self._enabled[ch]:
                # Disabled: show a flat ghost line at the channel's baseline
                # rather than hiding the row entirely. This gives the user
                # a visual confirmation of where the bad channel lives.
                curve.setData(
                    x=x,
                    y=np.full(n, base, dtype=np.float32),
                    pen=pg.mkPen(_TRACE_BAD, width=1, style=Qt.PenStyle.DashLine),
                )
                continue

            y = ordered[:, ch] * self._y_scale + base
            curve.setData(x=x, y=y,
                          pen=pg.mkPen(_TRACE_GOOD, width=1))

    def _get_ordered_data(self) -> np.ndarray:
        if not self._buffer_full:
            return self._data_buffer[:self._buffer_index]
        return np.concatenate(
            (self._data_buffer[self._buffer_index:],
             self._data_buffer[:self._buffer_index]),
            axis=0,
        )

    # ------------------------------------------------------------------
    # Autoscale
    # ------------------------------------------------------------------

    def _recompute_autoscale(self) -> None:
        if not self._autoscale_enabled:
            return
        with QMutexLocker(self._data_mutex):
            ordered = self._get_ordered_data()
        if ordered.size == 0:
            return

        # Only consider enabled channels — bad channels are often full-
        # scale noise that would squash the scale for everyone else.
        enabled_mask = np.asarray(self._enabled, dtype=bool)
        if not enabled_mask.any():
            return
        sub = ordered[:, enabled_mask]

        # Robust spread: 95th percentile of |x| across time, mean across
        # enabled channels. Resistant to a single spike from a twitchy
        # electrode.
        per_ch = np.percentile(np.abs(sub), 95, axis=0)
        spread = float(np.mean(per_ch)) if per_ch.size else 0.0
        if spread <= 1e-9:
            spread = 1.0

        # Target: each trace occupies ~80% of its row's vertical slot.
        # Row size stays fixed at 1.0 in data units; scale brings the
        # amplitude to ~0.4.
        new_scale = 0.4 / spread

        # Exponential smoothing → avoid jumpy re-scales when a transient
        # spike sweeps through.
        self._y_scale = 0.7 * self._y_scale + 0.3 * new_scale

    # ------------------------------------------------------------------
    # Ground-truth label
    # ------------------------------------------------------------------

    @Slot(str)
    def set_ground_truth(self, label: str) -> None:
        """Update the inline 'Ground Truth: X' banner below the plot."""
        text = label if label else "—"
        self._ground_truth_label.setText(f"Ground Truth: {text}")
        if label in (None, "", "—", "-", "Unknown"):
            style = "idle"
        elif label == "Rest":
            style = "rest"
        else:
            style = "active"
        self._ground_truth_label.setStyleSheet(_gt_qss(style))

    # ------------------------------------------------------------------
    # Channel toggling
    # ------------------------------------------------------------------

    @Slot(int, bool)
    def _on_channel_toggled(self, index: int, enabled: bool) -> None:
        if not (0 <= index < self.num_channels):
            return
        self._enabled[index] = enabled
        self._update_bad_count_label()
        self._emit_bad_channels()

    def _emit_bad_channels(self) -> None:
        arr = np.asarray(self._enabled, dtype=bool)
        self.bad_channels_updated.emit(arr)

    def _update_bad_count_label(self) -> None:
        n_bad = sum(1 for e in self._enabled if not e)
        if n_bad == 0:
            self._bad_count_label.setText("0 bad")
            self._bad_count_label.setStyleSheet(
                f"color: {_AXIS}; font-size: 10px;"
            )
        else:
            self._bad_count_label.setText(f"{n_bad} bad")
            self._bad_count_label.setStyleSheet(
                f"color: {_HIGHLIGHT}; font-size: 10px; font-weight: 600;"
            )

    @Slot()
    def _mark_all_good(self) -> None:
        for i, row in enumerate(self._rows):
            row.set_enabled(True, silent=True)
            self._enabled[i] = True
        self._update_bad_count_label()
        self._emit_bad_channels()

    @Slot()
    def _invert_selection(self) -> None:
        for i, row in enumerate(self._rows):
            new_state = not self._enabled[i]
            row.set_enabled(new_state, silent=True)
            self._enabled[i] = new_state
        self._update_bad_count_label()
        self._emit_bad_channels()

    # ------------------------------------------------------------------
    # Public setters
    # ------------------------------------------------------------------

    def set_num_channels(self, n: int) -> None:
        """Reconfigure the plot for ``n`` channels. Clears the buffer."""
        n = int(n)
        if n == self.num_channels:
            return
        was_running = self._render_timer.isActive()
        if was_running:
            self._render_timer.stop()

        self.num_channels = n
        with QMutexLocker(self._data_mutex):
            self._data_buffer = np.zeros(
                (self.display_samples, n), dtype=np.float32,
            )
            self._buffer_index = 0
            self._buffer_full  = False

        self._rebuild_curves_and_rows()

        if was_running:
            self._render_timer.start()

        self._emit_bad_channels()

    def set_channel_visible(self, channel: int, visible: bool) -> None:
        """Compatibility shim — delegates to toggle-bad-channel."""
        if 0 <= channel < self.num_channels:
            self._rows[channel].set_enabled(visible)
            # Row emits toggled → _on_channel_toggled, which emits the signal.

    def clear(self) -> None:
        """Zero out the buffer and blank all traces."""
        with QMutexLocker(self._data_mutex):
            self._data_buffer[:] = 0.0
            self._buffer_index = 0
            self._buffer_full  = False
        self._on_render_tick()

    def get_bad_channels(self) -> List[int]:
        """Return the list of 0-based bad channel indices."""
        return [i for i, e in enumerate(self._enabled) if not e]

    def set_bad_channels(self, bad_indices: List[int]) -> None:
        """Programmatically mark a set of channels as bad."""
        bad_set = set(int(i) for i in bad_indices)
        for i, row in enumerate(self._rows):
            row.set_enabled(i not in bad_set, silent=True)
            self._enabled[i] = i not in bad_set
        self._update_bad_count_label()
        self._emit_bad_channels()

    # ------------------------------------------------------------------
    # Control handlers
    # ------------------------------------------------------------------

    @Slot(int)
    def _on_display_time_changed(self, seconds: int) -> None:
        self.display_seconds = float(seconds)
        new_samples = int(seconds * self.sampling_rate)

        with QMutexLocker(self._data_mutex):
            old = self._get_ordered_data()
            new_buf = np.zeros(
                (new_samples, self.num_channels), dtype=np.float32,
            )
            keep = min(len(old), new_samples)
            if keep > 0:
                new_buf[-keep:] = old[-keep:]
            self._data_buffer  = new_buf
            self._display_samples_old = self.display_samples
            self.display_samples = new_samples
            self._buffer_index   = 0 if keep < new_samples else new_samples % new_samples
            self._buffer_full    = keep >= new_samples

        self.plot_widget.setXRange(-seconds, 0, padding=0)

    @Slot(bool)
    def _on_autoscale_toggled(self, enabled: bool) -> None:
        self._autoscale_enabled = enabled


# ---------------------------------------------------------------------------
# Standalone window — matches the old EMGPlotWindow surface
# ---------------------------------------------------------------------------

class EMGPlotWindow(QMainWindow):
    """
    Plot in a standalone window. The window owns an EMGPlotWidget and
    forwards its public API, so any existing code that constructs
    ``EMGPlotWindow(...)`` and calls ``.update_data(...)`` keeps
    working.
    """

    closed = Signal()

    def __init__(
        self,
        num_channels: int = 32,
        sampling_rate: int = 2000,
        display_seconds: float = 10.0,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("EMG Signal Visualization")
        self.setMinimumSize(900, 600)
        self.setStyleSheet(f"QMainWindow {{ background: {_BG}; }}")

        self._widget = EMGPlotWidget(
            num_channels=num_channels,
            sampling_rate=sampling_rate,
            display_seconds=display_seconds,
            parent=self,
        )
        self.setCentralWidget(self._widget)

    # Delegate methods so tests and callers that passed the old
    # EMGPlotWindow object continue to work verbatim.
    def update_data(self, data):    self._widget.update_data(data)
    def set_num_channels(self, n):  self._widget.set_num_channels(n)
    def set_ground_truth(self, l):  self._widget.set_ground_truth(l)
    def clear(self):                self._widget.clear()

    @property
    def bad_channels_updated(self):
        return self._widget.bad_channels_updated

    def closeEvent(self, event):
        self.closed.emit()
        super().closeEvent(event)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _make_label(text: str, muted: bool = False) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet(
        f"color: {_AXIS if muted else _FG}; font-size: 10px;"
    )
    return lbl


def _button_qss() -> str:
    return (
        f"QPushButton {{"
        f"  background: {_SEPARATOR}; color: {_FG}; "
        f"  border: 1px solid {_SEPARATOR}; padding: 3px 10px; "
        f"  border-radius: 4px; font-size: 10px;"
        f"}}"
        f"QPushButton:hover {{ border-color: {_TRACE_GOOD}; color: {_TRACE_GOOD}; }}"
    )


def _gt_qss(state: str) -> str:
    common = "font-size: 12px; font-weight: 700; padding: 4px; border-radius: 0;"
    if state == "rest":
        return f"{common} background: #164e63; color: #a5f3fc;"
    if state == "active":
        return f"{common} background: #064e3b; color: #86efac;"
    return f"{common} background: {_SEPARATOR}; color: {_AXIS};"
