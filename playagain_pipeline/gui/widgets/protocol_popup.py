"""
protocol_popup.py
─────────────────
A floating popup window that hosts the gesture-instruction display
during recording.

Why this exists
───────────────
The original layout pinned ``ProtocolWidget`` to the right side of the
main splitter, competing with the live EMG plot for horizontal space.
Now that the plot owns the sidebar (see ``emg_plot_panel.py``), the
protocol display moves into a modeless dialog that appears when a
recording starts and disappears when it stops.

Drop-in compatibility
─────────────────────
``ProtocolPopup`` wraps a ``ProtocolWidget`` and forwards the exact
public surface the main window uses:

    gesture_display                   — forwarded attribute
    set_protocol(protocol)            — forwarded method
    start() / stop() / pause() / resume()
    step_started, step_completed, protocol_completed  — re-emitted

This means ``main_window.py`` keeps every call it had before —
``self.protocol_widget.set_protocol(…)``, ``.start()``, ``.stop()``,
``.gesture_display.clear()`` — with only the creation line changed.

Show / hide behaviour
─────────────────────
    start()   →  show and raise the window
    stop()    →  hide the window
    close X   →  hide only; never destroy. The main window stays in
                 charge of recording state.

If the user dismisses the window mid-recording the gesture timer keeps
running under the hood, and the window re-opens itself on the next
``step_started`` so the user doesn't silently miss a cue.
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QDialog, QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget,
)

from playagain_pipeline.gui.widgets.protocol_widget import ProtocolWidget
from playagain_pipeline.protocols.protocol import (
    ProtocolStep, RecordingProtocol,
)


class ProtocolPopup(QDialog):
    """Modeless popup wrapping a ``ProtocolWidget`` for use during recording."""

    # Re-emitted from the inner ProtocolWidget so main_window connects
    # to ``protocol_widget.step_started`` exactly as before. We use
    # ``object`` rather than the concrete class for signal signatures
    # to avoid any QMetaType registration surprises across Qt builds.
    step_started       = Signal(object)   # ProtocolStep
    step_completed     = Signal(object)   # ProtocolStep
    protocol_completed = Signal()

    # Emitted when the user clicks "Stop Recording" on the popup itself.
    # Optional — if the main window connects this to its existing stop
    # handler, the popup becomes a remote control for the Record tab.
    stop_requested = Signal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Recording Protocol")
        self.setModal(False)
        # Plain dialog — no always-on-top so the user can alt-tab freely.
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, False)

        # Roughly matches the old right-panel footprint so the content
        # inside ProtocolWidget (emoji + big phase label) doesn't squish.
        self.resize(560, 620)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Header ──────────────────────────────────────────────────
        header = QFrame()
        header.setObjectName("ProtocolPopupHeader")
        header.setStyleSheet(
            "#ProtocolPopupHeader {"
            "  background: #0f172a;"
            "  border-bottom: 1px solid #1e293b;"
            "}"
        )
        header.setFixedHeight(34)
        h_lay = QHBoxLayout(header)
        h_lay.setContentsMargins(12, 0, 12, 0)

        # Red recording dot makes it obvious this window means business.
        dot = QLabel("●")
        dot.setStyleSheet("color: #ef4444; font-size: 13px;")
        h_lay.addWidget(dot)

        title = QLabel("Recording — follow the gesture cues")
        title.setStyleSheet("color: #e2e8f0; font-weight: 600; font-size: 11px;")
        h_lay.addWidget(title)
        h_lay.addStretch(1)

        root.addWidget(header)

        # ── Inner ProtocolWidget ────────────────────────────────────
        self._inner = ProtocolWidget(parent=self)
        self._inner.step_started.connect(self._on_inner_step_started)
        self._inner.step_completed.connect(self.step_completed.emit)
        self._inner.protocol_completed.connect(self._on_inner_protocol_completed)
        root.addWidget(self._inner, 1)

        # ── Footer with a convenience Stop button ───────────────────
        footer = QFrame()
        footer.setStyleSheet(
            "QFrame { background: #f8fafc; border-top: 1px solid #e2e8f0; }"
        )
        footer.setFixedHeight(52)
        f_lay = QHBoxLayout(footer)
        f_lay.setContentsMargins(12, 6, 12, 6)

        hint = QLabel(
            "You can close this window — recording keeps running. "
            "It will reopen on the next gesture cue."
        )
        hint.setStyleSheet("color: #64748b; font-size: 10px;")
        hint.setWordWrap(True)
        f_lay.addWidget(hint, 1)

        self._stop_btn = QPushButton("Stop Recording")
        self._stop_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._stop_btn.setStyleSheet(
            "QPushButton {"
            "  background: #dc2626; color: white; border: none;"
            "  border-radius: 4px; padding: 6px 14px; font-weight: 600;"
            "}"
            "QPushButton:hover { background: #b91c1c; }"
            "QPushButton:pressed { background: #991b1b; }"
        )
        self._stop_btn.clicked.connect(self._on_stop_clicked)
        f_lay.addWidget(self._stop_btn)

        root.addWidget(footer)

    # ------------------------------------------------------------------
    # Forwarded surface — mirrors ProtocolWidget so main_window.py
    # doesn't need to be aware that the widget is in a popup.
    # ------------------------------------------------------------------

    @property
    def gesture_display(self):
        """The inner widget's gesture_display (used by main_window)."""
        return self._inner.gesture_display

    @property
    def progress_widget(self):
        return self._inner.progress_widget

    @property
    def is_running(self) -> bool:
        return self._inner.is_running

    @property
    def current_step(self) -> Optional[ProtocolStep]:
        return self._inner.current_step

    def set_protocol(self, protocol: RecordingProtocol) -> None:
        self._inner.set_protocol(protocol)

    def start(self) -> None:
        """Start the protocol and show the popup."""
        self._inner.start()
        self._show_and_raise()

    def stop(self) -> None:
        """Stop the protocol and hide the popup."""
        self._inner.stop()
        self.hide()

    def pause(self) -> None:
        self._inner.pause()

    def resume(self) -> None:
        self._inner.resume()
        self._show_and_raise()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _show_and_raise(self) -> None:
        """Show and bring the popup to the front of the window stack."""
        if not self.isVisible():
            self._position_near_parent()
        self.show()
        self.raise_()
        self.activateWindow()

    def _position_near_parent(self) -> None:
        """Place the popup near — but not covering — the main window."""
        parent = self.parentWidget()
        if parent is None:
            return
        parent_rect = parent.geometry()
        # Prefer top-right of the parent so controls stay uncovered.
        x = parent_rect.x() + parent_rect.width() - self.width() - 40
        y = parent_rect.y() + 80
        # Clamp for tiny/offscreen parents.
        x = max(parent_rect.x() + 20, x)
        y = max(parent_rect.y() + 20, y)
        self.move(x, y)

    @Slot()
    def _on_stop_clicked(self) -> None:
        """User hit Stop inside the popup — forward the intent."""
        self.stop_requested.emit()
        # Fallback: if nothing listens, at least stop the inner widget
        # so it doesn't keep advancing steps with stale state.
        if self._inner.is_running:
            self._inner.stop()
        self.hide()

    @Slot(object)
    def _on_inner_step_started(self, step: ProtocolStep) -> None:
        # If the user dismissed the popup but recording is still live,
        # bring it back on the next cue so they don't silently miss it.
        if not self.isVisible() and self._inner.is_running:
            self._show_and_raise()
        self.step_started.emit(step)

    @Slot()
    def _on_inner_protocol_completed(self) -> None:
        self.protocol_completed.emit()
        # Hide so the user's desktop isn't left with a stale popup.
        # The main window's _on_protocol_completed handler still runs.
        self.hide()

    # ------------------------------------------------------------------
    # Qt overrides
    # ------------------------------------------------------------------

    def closeEvent(self, event: QCloseEvent) -> None:
        """
        Closing the window only hides it — the main window owns
        recording state, so destroying the popup here would leave
        dangling references.
        """
        event.ignore()
        self.hide()
