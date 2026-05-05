"""
emg_plot_popout.py
──────────────────
A small drop-in helper that exposes the EMG plot in a floating window.

Why this exists
───────────────
The v1 main window embedded ``EMGPlotWidget`` in the right panel of a
splitter. The v2 shell replaces the central widget, which deletes the
right panel — leaving ``self._plot_widget`` dangling and unseen.

Rather than reshape either shell, this module offers a single widget
that drops into the Recording tab:

    self._plot_widget = EMGPlotPopoutButton(num_channels=32, sampling_rate=2000)
    self._plot_widget.bad_channels_updated.connect(self._on_bad_channels_updated)
    device_layout.addRow(self._plot_widget)   # or anywhere else in the tab

It keeps the **same attribute name** and the **same public methods** as
the old widget (``update_data``, ``set_num_channels``, ``set_ground_truth``,
``clear``, ``isVisible``) so every existing call site in
``_on_data_received`` / ``_on_ground_truth_changed`` works unchanged.
The only visible change is a button — click it to pop the plot into a
floating window. Click again or close the window to hide it.

Reverting
─────────
Replace this with the original embedded ``EMGPlotWidget`` whenever you
want. No other file needs to change.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal, Slot
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QPushButton, QSizePolicy, QWidget

from playagain_pipeline.gui.widgets.emg_plot_reworked import EMGPlotWindow


class EMGPlotPopoutButton(QPushButton):
    """
    Drop-in replacement for the inline ``EMGPlotWidget`` that instead
    offers the plot in a floating window.

    Matches the subset of ``EMGPlotWidget`` API the main window actually
    uses, so swapping it in requires changing only the constructor line.

    Parameters
    ----------
    num_channels : int
        Initial channel count. Updated automatically by ``set_num_channels``.
    sampling_rate : int
        Sampling rate forwarded to the popup window.
    display_seconds : float
        Default time window shown in the popup.
    parent : QWidget
        Parent widget. The popup window is deliberately parentless (top-
        level) so the user can place it on a second monitor independent
        of the main window's size.
    """

    # Same signal shape as the original EMGPlotWidget so the main
    # window's existing `_on_bad_channels_updated` slot connects
    # unchanged.
    bad_channels_updated = Signal(np.ndarray)

    def __init__(
        self,
        num_channels: int = 32,
        sampling_rate: int = 2000,
        display_seconds: float = 10.0,
        parent: Optional[QWidget] = None,
    ):
        super().__init__("  📊  Show live EMG plot  ", parent)

        self._num_channels     = int(num_channels)
        self._sampling_rate    = int(sampling_rate)
        self._display_seconds  = float(display_seconds)
        self._window: Optional[EMGPlotWindow] = None

        # Cache the most recent "set_num_channels"/ground-truth calls
        # made *before* the window is opened, so the popup starts out
        # in the right state the first time it's shown.
        self._pending_ground_truth: Optional[str] = None

        # Stored bad-channel indices so the user's previous selection
        # survives close-and-reopen. Starts empty (all good).
        self._bad_channel_indices: list[int] = []

        # Styling — make it look like an obvious affordance, not a
        # generic dialog button.
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setSizePolicy(
            QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed,
        )
        self.setFixedHeight(30)
        self.setStyleSheet(
            "QPushButton {"
            "  background: #0f172a; color: #06b6d4; "
            "  border: 1px solid #06b6d4; border-radius: 6px;"
            "  padding: 4px 14px; font-size: 11px; font-weight: 600;"
            "}"
            "QPushButton:hover {"
            "  background: #06b6d4; color: #ffffff;"
            "}"
            "QPushButton:pressed {"
            "  background: #0891b2; color: #ffffff;"
            "}"
        )
        self.setToolTip(
            "Open the live EMG plot in a separate window.\n"
            "Use the checkboxes next to each trace to mark channels "
            "as bad (interpolated / zeroed per your preference)."
        )

        self.clicked.connect(self._toggle_window)

    # ------------------------------------------------------------------
    # Window lifecycle
    # ------------------------------------------------------------------

    @Slot()
    def _toggle_window(self) -> None:
        """Open the plot window, or raise and focus it if already open."""
        if self._window is None:
            self._open_window()
        elif self._window.isVisible():
            # Already visible — bring it forward rather than hiding,
            # so a user clicking twice doesn't play hide-and-seek.
            self._window.raise_()
            self._window.activateWindow()
        else:
            self._window.show()
            self._window.raise_()
            self._window.activateWindow()

    def _open_window(self) -> None:
        """Create the popup, restore state, and wire signals."""
        self._window = EMGPlotWindow(
            num_channels=self._num_channels,
            sampling_rate=self._sampling_rate,
            display_seconds=self._display_seconds,
            parent=None,
        )
        # Forward the bad-channels signal so the main window's existing
        # slot (which writes to session metadata) sees changes.
        self._window.bad_channels_updated.connect(
            self._on_bad_channels_updated_from_window
        )

        # Watch for the window closing so we can reset our reference —
        # otherwise the next button click would try to call methods on
        # a deleted widget.
        self._window.closed.connect(self._on_window_closed)

        # Restore state the user accumulated before opening.
        if self._bad_channel_indices:
            try:
                # EMGPlotWindow exposes set_bad_channels via the
                # underlying widget — access it directly.
                self._window._widget.set_bad_channels(
                    self._bad_channel_indices,
                )
            except Exception:
                pass

        if self._pending_ground_truth is not None:
            self._window.set_ground_truth(self._pending_ground_truth)

        self._window.show()
        self._window.raise_()
        self._window.activateWindow()
        self.setText("  📊  Hide live EMG plot  ")

    @Slot()
    def _on_window_closed(self) -> None:
        """User closed the window — let it be garbage-collected."""
        self._window = None
        self.setText("  📊  Show live EMG plot  ")

    @Slot(np.ndarray)
    def _on_bad_channels_updated_from_window(self, arr: np.ndarray) -> None:
        """Cache the selection, then re-emit on our own signal."""
        self._bad_channel_indices = [
            i for i, enabled in enumerate(arr) if not bool(enabled)
        ]
        self.bad_channels_updated.emit(arr)

    # ------------------------------------------------------------------
    # Public surface — mirrors the old EMGPlotWidget for drop-in compat
    # ------------------------------------------------------------------

    @property
    def num_channels(self) -> int:
        """Current channel count (kept in sync with the popup)."""
        return self._num_channels

    def update_data(self, data: np.ndarray) -> None:
        """Forward a batch of samples to the popup if it's open.

        No-op when the popup is closed — keeps the main window's
        ``_on_data_received`` hot path cheap when the user isn't looking
        at the plot.
        """
        if self._window is None:
            return
        try:
            self._window.update_data(data)
        except RuntimeError:
            # Window was destroyed between the isVisible check and this
            # call (rare but possible during close).
            self._window = None

    def set_num_channels(self, n: int) -> None:
        """Track channel-count changes from the device."""
        n = int(n)
        if n == self._num_channels:
            return
        self._num_channels = n
        if self._window is not None:
            try:
                self._window.set_num_channels(n)
            except RuntimeError:
                self._window = None

    def set_ground_truth(self, label: str) -> None:
        """Forward the ground-truth label to the popup."""
        self._pending_ground_truth = label
        if self._window is not None:
            try:
                self._window.set_ground_truth(label)
            except RuntimeError:
                self._window = None

    def clear(self) -> None:
        """Reset the popup buffer if it's open."""
        if self._window is not None:
            try:
                self._window.clear()
            except RuntimeError:
                self._window = None

    def isVisible(self) -> bool:  # noqa: N802  (match Qt casing)
        """
        Report True only when the plot is actually being shown.

        The main window's data handler uses this as a gate:

            if self._plot_widget and self._plot_widget.isVisible():
                self._plot_widget.update_data(data)

        Returning False when the popup is closed keeps that gate
        behaving the same as it did with the embedded widget.
        """
        return self._window is not None and self._window.isVisible()
