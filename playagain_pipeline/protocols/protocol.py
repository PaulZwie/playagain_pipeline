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


class ProtocolPhase(Enum):
    """Phases within a protocol."""
    PREPARATION = auto()  # Get ready phase
    REST = auto()         # Rest between trials
    CUE = auto()          # Show gesture cue
    HOLD = auto()         # Hold the gesture
    RELEASE = auto()      # Release the gesture
    FEEDBACK = auto()     # Show feedback
    COMPLETE = auto()     # Protocol complete


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
    hold_time: float = 3.0
    release_time: float = 0.5
    rest_time: float = 2.0

    # Repetition parameters
    repetitions_per_gesture: int = 5
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
            duration=3.0,
            message="Get ready! We will start soon.",
            is_recording=False
        ))

        # Get the gesture sequence
        gestures = list(self.gesture_set.gestures)
        if self.config.randomize_order:
            random.shuffle(gestures)

        # Repeat the sequence
        for rep in range(self.config.repetitions_per_gesture):
            for i, gesture in enumerate(gestures):
                # Hold phase (5s)
                self._steps.append(ProtocolStep(
                    phase=ProtocolPhase.HOLD,
                    gesture=gesture,
                    duration=5.0,
                    message=f"Perform: {gesture.display_name}",
                    trial_index=rep * len(gestures) + i,
                    repetition_index=rep,
                    is_recording=True
                ))

                # Check if there's a next gesture (not last gesture of last repetition)
                is_last_gesture = (i == len(gestures) - 1) and (rep == self.config.repetitions_per_gesture - 1)

                if not is_last_gesture:
                    # Determine next gesture
                    next_i = (i + 1) % len(gestures)
                    if next_i == 0 and rep + 1 < self.config.repetitions_per_gesture:
                        next_gesture = gestures[0]  # Next repetition starts with first
                    else:
                        next_gesture = gestures[next_i]

                    # Pause phase (3s) - show next gesture
                    self._steps.append(ProtocolStep(
                        phase=ProtocolPhase.REST,
                        gesture=next_gesture,  # Pass next gesture for visualization
                        duration=3.0,
                        message=f"Pause. Next: {next_gesture.display_name}",
                        trial_index=rep * len(gestures) + i,
                        repetition_index=rep,
                        is_recording=False
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
    return ProtocolConfig(
        name="quick",
        description="Quick protocol for testing (3 repetitions, short holds)",
        preparation_time=2.0,
        cue_time=0.5,
        hold_time=2.0,
        release_time=0.3,
        rest_time=1.0,
        repetitions_per_gesture=3,
        randomize_order=True
    )


def create_standard_protocol() -> ProtocolConfig:
    """Create a standard recording protocol."""
    return ProtocolConfig(
        name="standard",
        description="Standard protocol (5 repetitions, 3s holds)",
        preparation_time=3.0,
        cue_time=1.0,
        hold_time=3.0,
        release_time=0.5,
        rest_time=2.0,
        repetitions_per_gesture=5,
        randomize_order=True
    )


def create_extended_protocol() -> ProtocolConfig:
    """Create an extended protocol for thorough data collection."""
    return ProtocolConfig(
        name="extended",
        description="Extended protocol (10 repetitions, 4s holds)",
        preparation_time=5.0,
        cue_time=1.5,
        hold_time=4.0,
        release_time=0.5,
        rest_time=3.0,
        repetitions_per_gesture=10,
        randomize_order=True
    )


def create_calibration_protocol() -> ProtocolConfig:
    """Create a protocol specifically for calibration."""
    return ProtocolConfig(
        name="calibration",
        description="Calibration protocol (3 repetitions, 2s holds)",
        preparation_time=2.0,
        cue_time=0.5,
        hold_time=2.0,
        release_time=0.3,
        rest_time=1.5,
        repetitions_per_gesture=3,
        randomize_order=False  # Keep order consistent for calibration
    )
