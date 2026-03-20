"""
Protocol display widget for showing gesture instructions.

Displays the current gesture to perform, timing information, and progress.
"""

from typing import Optional
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QProgressBar, QFrame, QSizePolicy
)
from PySide6.QtCore import Qt, Signal, Slot, QTimer
from PySide6.QtGui import QFont, QPixmap

from playagain_pipeline.protocols.protocol import (
    RecordingProtocol, ProtocolStep, ProtocolPhase
)


class GestureDisplayWidget(QWidget):
    """
    Widget displaying the current gesture instruction.

    Shows:
    - Gesture name and description
    - Visual cue (image if available)
    - Phase indicator (rest, prepare, hold, release)
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        """Setup the user interface."""
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Phase indicator
        self.phase_label = QLabel("READY")
        self.phase_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        phase_font = QFont()
        phase_font.setPointSize(24)
        phase_font.setBold(True)
        self.phase_label.setFont(phase_font)
        layout.addWidget(self.phase_label)

        # Gesture name
        self.gesture_label = QLabel("")
        self.gesture_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        gesture_font = QFont()
        gesture_font.setPointSize(36)
        gesture_font.setBold(True)
        self.gesture_label.setFont(gesture_font)
        layout.addWidget(self.gesture_label)

        # Emoji display
        self.emoji_label = QLabel()
        self.emoji_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        emoji_font = QFont()
        emoji_font.setPointSize(96)  # Large emoji
        self.emoji_label.setFont(emoji_font)
        self.emoji_label.setMinimumSize(200, 200)
        layout.addWidget(self.emoji_label)

        # Description
        self.description_label = QLabel("")
        self.description_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.description_label.setWordWrap(True)
        desc_font = QFont()
        desc_font.setPointSize(14)
        self.description_label.setFont(desc_font)
        layout.addWidget(self.description_label)

        # Phase colors
        self._phase_colors = {
            ProtocolPhase.PREPARATION:      "#FFA500",  # Orange
            ProtocolPhase.REST:             "#4CAF50",  # Green
            ProtocolPhase.CUE:              "#2196F3",  # Blue
            ProtocolPhase.HOLD:             "#F44336",  # Red
            ProtocolPhase.RELEASE:          "#9C27B0",  # Purple
            ProtocolPhase.FEEDBACK:         "#00BCD4",  # Cyan
            ProtocolPhase.COMPLETE:         "#4CAF50",  # Green
            ProtocolPhase.CALIBRATION_SYNC: "#FF6F00",  # Deep amber — visually distinct
        }

    def update_step(self, step: ProtocolStep):
        """Update display for a protocol step."""
        # Update phase
        phase_text = step.phase.name.replace("_", " ")
        self.phase_label.setText(phase_text)

        color = self._phase_colors.get(step.phase, "#000000")
        self.phase_label.setStyleSheet(f"color: {color};")

        # Update gesture info
        if step.message:
            self.gesture_label.setText(step.message)
            self.description_label.setText("")
        elif step.gesture:
            self.gesture_label.setText(step.gesture.display_name)
            self.description_label.setText(step.gesture.description)
        else:
            self.gesture_label.setText("")
            self.description_label.setText("")

        # Display emoji or image
        if step.gesture and step.gesture.emoji:
            # Show emoji
            self.emoji_label.setText(step.gesture.emoji)
            self.emoji_label.setStyleSheet("")
        elif step.gesture and step.gesture.image_path:
            # Load image if available
            pixmap = QPixmap(step.gesture.image_path)
            if not pixmap.isNull():
                scaled_pixmap = pixmap.scaled(200, 200, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                # Convert pixmap to text label (not ideal but works)
                self.emoji_label.setPixmap(scaled_pixmap)
                self.emoji_label.setText("")
            else:
                self._set_placeholder_emoji(step.gesture.display_name)
        else:
            self._set_placeholder_emoji("")

    def _set_placeholder_emoji(self, text: str):
        """Set a placeholder for the gesture visualization."""
        self.emoji_label.setText(text[:2].upper() if text else "")
        self.emoji_label.setStyleSheet(
            "background-color: #e0e0e0; border-radius: 10px; "
            "font-size: 48px; font-weight: bold; color: #666;"
        )

    def clear(self):
        """Clear the display."""
        self.phase_label.setText("READY")
        self.phase_label.setStyleSheet("")
        self.gesture_label.setText("")
        self.description_label.setText("")
        self._set_placeholder_emoji("")


class ProtocolProgressWidget(QWidget):
    """
    Widget showing protocol progress and timing.
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._setup_ui()

        # Timer for countdown
        self._countdown_timer = QTimer(self)
        self._countdown_timer.timeout.connect(self._update_countdown)
        self._remaining_time = 0.0
        self._step_duration = 0.0

    def _setup_ui(self):
        """Setup the user interface."""
        layout = QVBoxLayout(self)

        # Overall progress
        overall_layout = QHBoxLayout()
        overall_layout.addWidget(QLabel("Overall Progress:"))
        self.overall_progress = QProgressBar()
        self.overall_progress.setRange(0, 100)
        self.overall_progress.setValue(0)
        overall_layout.addWidget(self.overall_progress)
        self.progress_label = QLabel("0%")
        overall_layout.addWidget(self.progress_label)
        layout.addLayout(overall_layout)

        # Trial progress
        trial_layout = QHBoxLayout()
        trial_layout.addWidget(QLabel("Trial:"))
        self.trial_label = QLabel("0 / 0")
        trial_layout.addWidget(self.trial_label)
        trial_layout.addStretch()
        trial_layout.addWidget(QLabel("Time:"))
        self.time_label = QLabel("0:00 / 0:00")
        trial_layout.addWidget(self.time_label)
        layout.addLayout(trial_layout)

        # Step countdown
        countdown_layout = QHBoxLayout()
        countdown_layout.addWidget(QLabel("Step:"))
        self.step_progress = QProgressBar()
        self.step_progress.setRange(0, 100)
        self.step_progress.setValue(0)
        countdown_layout.addWidget(self.step_progress)
        self.countdown_label = QLabel("0.0s")
        self.countdown_label.setMinimumWidth(50)
        countdown_layout.addWidget(self.countdown_label)
        layout.addLayout(countdown_layout)

    def set_protocol(self, protocol: RecordingProtocol):
        """Set the protocol to display progress for."""
        total_trials = sum(
            1 for step in protocol.steps
            if step.phase == ProtocolPhase.HOLD
        )
        total_duration = protocol.total_duration

        self.trial_label.setText(f"0 / {total_trials}")
        self.time_label.setText(
            f"0:00 / {int(total_duration // 60)}:{int(total_duration % 60):02d}"
        )

    def update_progress(self, protocol: RecordingProtocol):
        """Update progress display."""
        # Overall progress
        progress_pct = int(protocol.progress * 100)
        self.overall_progress.setValue(progress_pct)
        self.progress_label.setText(f"{progress_pct}%")

        # Trial count
        completed_trials = sum(
            1 for step in protocol.steps[:protocol.current_step_index]
            if step.phase == ProtocolPhase.HOLD
        )
        total_trials = sum(
            1 for step in protocol.steps
            if step.phase == ProtocolPhase.HOLD
        )
        self.trial_label.setText(f"{completed_trials} / {total_trials}")

        # Time
        elapsed = protocol.elapsed_duration
        total = protocol.total_duration
        self.time_label.setText(
            f"{int(elapsed // 60)}:{int(elapsed % 60):02d} / "
            f"{int(total // 60)}:{int(total % 60):02d}"
        )

    def start_step_countdown(self, duration: float):
        """Start countdown for current step."""
        self._step_duration = duration
        self._remaining_time = duration
        self.step_progress.setValue(100)
        self._countdown_timer.start(100)  # Update every 100ms

    def stop_countdown(self):
        """Stop the countdown timer."""
        self._countdown_timer.stop()

    def _update_countdown(self):
        """Update countdown display."""
        self._remaining_time -= 0.1
        if self._remaining_time <= 0:
            self._remaining_time = 0
            self._countdown_timer.stop()

        # Update progress bar
        if self._step_duration > 0:
            progress = int((self._remaining_time / self._step_duration) * 100)
            self.step_progress.setValue(progress)

        self.countdown_label.setText(f"{self._remaining_time:.1f}s")


class ProtocolWidget(QWidget):
    """
    Combined widget for protocol display and control.
    """

    # Signals
    step_started = Signal(ProtocolStep)
    step_completed = Signal(ProtocolStep)
    protocol_completed = Signal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self._protocol: Optional[RecordingProtocol] = None
        self._is_running = False

        # Timer for step advancement
        self._step_timer = QTimer(self)
        self._step_timer.timeout.connect(self._advance_step)

        self._setup_ui()

    def _setup_ui(self):
        """Setup the user interface."""
        layout = QVBoxLayout(self)

        # Gesture display
        self.gesture_display = GestureDisplayWidget()
        layout.addWidget(self.gesture_display, stretch=2)

        # Separator
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        layout.addWidget(line)

        # Progress display
        self.progress_widget = ProtocolProgressWidget()
        layout.addWidget(self.progress_widget)

    def set_protocol(self, protocol: RecordingProtocol):
        """Set the protocol to execute."""
        self._protocol = protocol
        self.progress_widget.set_protocol(protocol)

        if protocol.current_step:
            self.gesture_display.update_step(protocol.current_step)

    def start(self):
        """Start protocol execution."""
        if self._protocol is None:
            return

        self._is_running = True
        self._protocol.reset()

        if self._protocol.current_step:
            self._execute_current_step()

    def stop(self):
        """Stop protocol execution."""
        self._is_running = False
        self._step_timer.stop()
        self.progress_widget.stop_countdown()

    def pause(self):
        """Pause protocol execution."""
        self._step_timer.stop()
        self.progress_widget.stop_countdown()

    def resume(self):
        """Resume protocol execution."""
        if self._protocol and self._protocol.current_step:
            self._execute_current_step()

    @property
    def is_running(self) -> bool:
        return self._is_running

    @property
    def current_step(self) -> Optional[ProtocolStep]:
        return self._protocol.current_step if self._protocol else None

    def _execute_current_step(self):
        """Execute the current protocol step."""
        step = self._protocol.current_step
        if step is None:
            return

        # Emit step started signal
        self.step_started.emit(step)

        # Update display
        self.gesture_display.update_step(step)
        self.progress_widget.update_progress(self._protocol)

        # Start countdown
        if step.duration > 0:
            self.progress_widget.start_step_countdown(step.duration)
            self._step_timer.start(int(step.duration * 1000))
        else:
            # Zero duration step (e.g., complete) - emit immediately
            self.step_completed.emit(step)
            if step.phase == ProtocolPhase.COMPLETE:
                self._is_running = False
                self.protocol_completed.emit()

    def _advance_step(self):
        """Advance to the next protocol step."""
        self._step_timer.stop()
        self.progress_widget.stop_countdown()

        current = self._protocol.current_step
        if current:
            self.step_completed.emit(current)

        # Advance to next step
        next_step = self._protocol.advance()

        if next_step is None or self._protocol.is_complete:
            self._is_running = False
            self.protocol_completed.emit()
            return

        # Execute next step
        self._execute_current_step()
