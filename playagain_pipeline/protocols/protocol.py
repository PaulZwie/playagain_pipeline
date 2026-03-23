"""
Training protocols for gesture recording.

Protocols define the structure and timing of recording sessions,
including which gestures to record, how many repetitions, and timing.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Dict, Any, Callable, Iterator
import random
import json
from pathlib import Path

from playagain_pipeline.core.gesture import GestureSet, Gesture
from playagain_pipeline.config.config import get_default_config


class ProtocolPhase(Enum):
    """Phases within a protocol."""
    PREPARATION = auto()       # Get ready phase
    REST = auto()              # Rest between trials
    CUE = auto()               # Show gesture cue
    HOLD = auto()              # Hold the gesture
    RELEASE = auto()           # Release the gesture
    FEEDBACK = auto()          # Show feedback
    COMPLETE = auto()          # Protocol complete
    CALIBRATION_SYNC = auto()  # Waveout sync gesture at session start


@dataclass
class ProtocolStep:
    """A single step in the protocol."""
    phase: ProtocolPhase
    gesture: Optional[Gesture]
    duration: float  # seconds
    message: str = ""
    trial_index: int = 0
    repetition_index: int = 0
    is_recording: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "phase": self.phase.name,
            "gesture_name": self.gesture.name if self.gesture else None,
            "duration": self.duration,
            "message": self.message,
            "trial_index": self.trial_index,
            "repetition_index": self.repetition_index,
            "is_recording": self.is_recording
        }


@dataclass
class ProtocolConfig:
    """Configuration for a recording protocol."""
    name: str
    description: str = ""

    # Timing parameters (in seconds)
    preparation_time: float = 3.0
    cue_time: float = 1.0
    hold_time: float = 8.0
    release_time: float = 0.5
    rest_time: float = 5.0

    # Repetition parameters
    repetitions_per_gesture: int = 3
    randomize_order: bool = True

    # Recording parameters
    record_during_cue: bool = False
    record_during_release: bool = False

    # Optional callbacks (not serialized)
    pre_trial_callback: Optional[Callable] = field(default=None, repr=False)
    post_trial_callback: Optional[Callable] = field(default=None, repr=False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "preparation_time": self.preparation_time,
            "cue_time": self.cue_time,
            "hold_time": self.hold_time,
            "release_time": self.release_time,
            "rest_time": self.rest_time,
            "repetitions_per_gesture": self.repetitions_per_gesture,
            "randomize_order": self.randomize_order,
            "record_during_cue": self.record_during_cue,
            "record_during_release": self.record_during_release
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProtocolConfig":
        return cls(**data)

    def save(self, path: Path) -> None:
        """Save protocol configuration to JSON."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "ProtocolConfig":
        """Load protocol configuration from JSON."""
        with open(path, 'r') as f:
            return cls.from_dict(json.load(f))


class RecordingProtocol:
    """
    Manages the execution of a recording protocol.

    Generates a sequence of steps for the user to follow during recording.
    """

    def __init__(self, gesture_set: GestureSet, config: ProtocolConfig):
        """
        Initialize the protocol.

        Args:
            gesture_set: Set of gestures to record
            config: Protocol configuration
        """
        self.gesture_set = gesture_set
        self.config = config
        self._steps: List[ProtocolStep] = []
        self._current_step_index: int = 0
        self._build_protocol()

    def _build_protocol(self) -> None:
        """Build the sequence of protocol steps."""
        self._steps = []

        # Initial preparation
        self._steps.append(ProtocolStep(
            phase=ProtocolPhase.PREPARATION,
            gesture=None,
            duration=self.config.preparation_time,
            message="Bereit? Wir starten gleich!",
            is_recording=False
        ))

        # Add a short rest before the calibration sync so the first cue does not
        # start abruptly. Keep startup duration close to previous behavior by
        # taking this time from the calibration-sync step.
        pre_sync_rest = min(1.0, max(0.0, self.config.rest_time * 0.3))
        sync_duration = self.config.rest_time - pre_sync_rest

        if sync_duration <= 0:
            pre_sync_rest = 0.0
            sync_duration = self.config.rest_time

        if pre_sync_rest > 0:
            self._steps.append(ProtocolStep(
                phase=ProtocolPhase.REST,
                gesture=None,
                duration=pre_sync_rest,
                message="Rest. Calibration starts next.",
                is_recording=False,
            ))

        # Calibration sync gesture (waveout) — recorded once per session.
        # This is NOT part of the gesture set and will NOT be used for model
        # training. It gives the rotation-detection calibrator a clean,
        # high-quality waveout signal regardless of which gesture set the
        # session was recorded with.
        self._steps.append(ProtocolStep(
            phase=ProtocolPhase.CALIBRATION_SYNC,
            gesture=None,
            duration=sync_duration,
            message="Calibration sync: WAVE OUT now (move wrist clearly outward)",
            is_recording=True,
        ))

        # Get the gesture sequence
        gestures = list(self.gesture_set.gestures)
        if self.config.randomize_order:
            random.shuffle(gestures)

        # Find the rest gesture in the set (for labeling rest phases)
        rest_gesture = None
        for g in gestures:
            if g.name == "rest":
                rest_gesture = g
                break

        # Filter out the rest gesture from the active gesture sequence
        # (rest data is now automatically collected between gestures)
        active_gestures = [g for g in gestures if g.name != "rest"]
        if not active_gestures:
            active_gestures = gestures  # Fallback if no non-rest gestures

        # Repeat the sequence
        for rep in range(self.config.repetitions_per_gesture):
            for i, gesture in enumerate(active_gestures):
                # Hold phase — perform the gesture
                self._steps.append(ProtocolStep(
                    phase=ProtocolPhase.HOLD,
                    gesture=gesture,
                    duration=self.config.hold_time,
                    message=f"{gesture.display_name}",
                    trial_index=rep * len(active_gestures) + i,
                    repetition_index=rep,
                    is_recording=True
                ))

                # Check if there's a next gesture (not last gesture of last repetition)
                is_last_gesture = (i == len(active_gestures) - 1) and (rep == self.config.repetitions_per_gesture - 1)

                if not is_last_gesture:
                    # Determine next gesture
                    next_i = (i + 1) % len(active_gestures)
                    if next_i == 0 and rep + 1 < self.config.repetitions_per_gesture:
                        next_gesture = active_gestures[0]  # Next repetition starts with first
                    else:
                        next_gesture = active_gestures[next_i]

                    # Pause phase — also recorded as rest data
                    # The REST phase is now recorded as a "rest" trial automatically,
                    # so users don't need to perform "rest" as a separate gesture.
                    self._steps.append(ProtocolStep(
                        phase=ProtocolPhase.REST,
                        gesture=next_gesture,  # Pass next gesture for visualization
                        duration=self.config.rest_time,
                        message=f"Rest. Next: {next_gesture.display_name}",
                        trial_index=rep * len(active_gestures) + i,
                        repetition_index=rep,
                        is_recording=True  # Now recorded as rest data
                    ))


        # Final completion
        self._steps.append(ProtocolStep(
            phase=ProtocolPhase.COMPLETE,
            gesture=None,
            duration=0,
            message="Recording complete! Thank you.",
            is_recording=False
        ))

    @property
    def steps(self) -> List[ProtocolStep]:
        """Get all protocol steps."""
        return self._steps

    @property
    def current_step(self) -> Optional[ProtocolStep]:
        """Get the current step."""
        if 0 <= self._current_step_index < len(self._steps):
            return self._steps[self._current_step_index]
        return None

    @property
    def current_step_index(self) -> int:
        """Get the current step index."""
        return self._current_step_index

    @property
    def total_steps(self) -> int:
        """Get total number of steps."""
        return len(self._steps)

    @property
    def progress(self) -> float:
        """Get progress as a fraction (0-1)."""
        return self._current_step_index / max(1, len(self._steps) - 1)

    @property
    def is_complete(self) -> bool:
        """Check if protocol is complete."""
        return self._current_step_index >= len(self._steps)

    @property
    def total_duration(self) -> float:
        """Get estimated total duration in seconds."""
        return sum(step.duration for step in self._steps)

    @property
    def elapsed_duration(self) -> float:
        """Get elapsed duration based on completed steps."""
        return sum(step.duration for step in self._steps[:self._current_step_index])

    def reset(self) -> None:
        """Reset the protocol to the beginning."""
        self._current_step_index = 0
        if self.config.randomize_order:
            self._build_protocol()

    def advance(self) -> Optional[ProtocolStep]:
        """Advance to the next step."""
        self._current_step_index += 1
        return self.current_step

    def get_step(self, index: int) -> Optional[ProtocolStep]:
        """Get a specific step by index."""
        if 0 <= index < len(self._steps):
            return self._steps[index]
        return None

    def __iter__(self) -> Iterator[ProtocolStep]:
        """Iterate through all steps."""
        return iter(self._steps)

    def __len__(self) -> int:
        return len(self._steps)


# Pre-defined protocol configurations
def create_quick_protocol() -> ProtocolConfig:
    """Create a quick protocol for testing."""
    settings = get_default_config().protocol
    return ProtocolConfig(
        name="quick",
        description="Quick protocol for testing",
        preparation_time=settings.quick_preparation_time,
        cue_time=settings.quick_cue_time,
        hold_time=settings.quick_hold_time,
        release_time=settings.quick_release_time,
        rest_time=settings.quick_rest_time,
        repetitions_per_gesture=settings.quick_repetitions,
        randomize_order=settings.quick_randomize
    )


def create_standard_protocol() -> ProtocolConfig:
    """Create a standard recording protocol."""
    settings = get_default_config().protocol
    return ProtocolConfig(
        name="standard",
        description="Standard protocol",
        preparation_time=settings.std_preparation_time,
        cue_time=settings.std_cue_time,
        hold_time=settings.std_hold_time,
        release_time=settings.std_release_time,
        rest_time=settings.std_rest_time,
        repetitions_per_gesture=settings.std_repetitions,
        randomize_order=settings.std_randomize
    )


def create_extended_protocol() -> ProtocolConfig:
    """Create an extended protocol for thorough data collection."""
    # Extended protocol uses standard settings but doubled repetitions (or we can add specific config if needed)
    # For now, let's just keep it hardcoded or maybe use standard settings x2?
    # The user specifically asked for standard repetition number and time.
    # Let's leave extended as is but maybe base time on standard?
    # But for consistency, let's use the standard times at least.

    settings = get_default_config().protocol
    return ProtocolConfig(
        name="extended",
        description="Extended protocol (10 repetitions)",
        preparation_time=settings.long_preparation_time,
        cue_time=settings.long_cue_time,
        hold_time=settings.long_hold_time,
        release_time=settings.long_release_time,
        rest_time=settings.long_rest_time,
        repetitions_per_gesture=settings.long_repetitions,
        randomize_order=settings.long_randomize
    )


def create_calibration_protocol() -> ProtocolConfig:
    """Create a protocol specifically for calibration."""
    settings = get_default_config().protocol
    return ProtocolConfig(
        name="calibration",
        description="Calibration protocol",
        preparation_time=settings.cal_preparation_time,
        cue_time=settings.cal_cue_time,
        hold_time=settings.cal_hold_time,
        release_time=settings.cal_release_time,
        rest_time=settings.cal_rest_time,
        repetitions_per_gesture=settings.cal_repetitions,
        randomize_order=settings.cal_randomize
    )
