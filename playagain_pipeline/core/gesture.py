"""
Gesture definitions and management.

This module provides a flexible system for defining and managing gestures.
Gestures can be easily added, modified, or removed for different experiments.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, List, Dict, Any
from pathlib import Path
import json


class GestureCategory(Enum):
    """Categories of gestures for organization and filtering."""
    REST = auto()
    FINGER = auto()
    HAND = auto()
    WRIST = auto()
    GRIP = auto()
    CUSTOM = auto()


@dataclass
class Gesture:
    """
    Represents a single gesture that can be performed and recorded.

    Attributes:
        name: Unique identifier for the gesture
        display_name: Human-readable name for display in UI
        description: Detailed description of how to perform the gesture
        category: Category of the gesture for organization
        image_path: Optional path to an image showing the gesture
        emoji: Optional emoji for visual representation (e.g., "🤛🏻" for fist)
        label_id: Numeric label for ML classification (auto-assigned if not provided)
        duration_hint: Suggested duration for holding the gesture (seconds)
        metadata: Additional custom metadata for the gesture
    """
    name: str
    display_name: str
    description: str = ""
    category: GestureCategory = GestureCategory.CUSTOM
    image_path: Optional[str] = None
    emoji: Optional[str] = None
    label_id: Optional[int] = None
    duration_hint: float = 5.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.label_id is None:
            # Will be assigned by GestureSet
            pass

    def to_dict(self) -> Dict[str, Any]:
        """Convert gesture to dictionary for serialization."""
        return {
            "name": self.name,
            "display_name": self.display_name,
            "description": self.description,
            "category": self.category.name,
            "image_path": self.image_path,
            "emoji": self.emoji,
            "label_id": self.label_id,
            "duration_hint": self.duration_hint,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Gesture":
        """Create gesture from dictionary."""
        data = data.copy()
        data["category"] = GestureCategory[data["category"]]
        return cls(**data)


class GestureSet:
    """
    A collection of gestures for a specific experiment or recording session.

    Provides methods for managing gestures and ensuring consistent label assignments.
    """

    def __init__(self, name: str = "default"):
        self.name = name
        self._gestures: Dict[str, Gesture] = {}
        self._label_counter = 0

    def add_gesture(self, gesture: Gesture) -> None:
        """Add a gesture to the set, assigning a label_id if not set."""
        if gesture.label_id is None:
            gesture.label_id = self._label_counter
            self._label_counter += 1
        else:
            self._label_counter = max(self._label_counter, gesture.label_id + 1)

        self._gestures[gesture.name] = gesture

    def remove_gesture(self, name: str) -> Optional[Gesture]:
        """Remove a gesture from the set by name."""
        return self._gestures.pop(name, None)

    def get_gesture(self, name: str) -> Optional[Gesture]:
        """Get a gesture by name."""
        return self._gestures.get(name)

    def get_gesture_by_label(self, label_id: int) -> Optional[Gesture]:
        """Get a gesture by its label_id."""
        for gesture in self._gestures.values():
            if gesture.label_id == label_id:
                return gesture
        return None

    @property
    def gestures(self) -> List[Gesture]:
        """Get all gestures in the set, sorted by label_id."""
        return sorted(self._gestures.values(), key=lambda g: g.label_id)

    @property
    def gesture_names(self) -> List[str]:
        """Get all gesture names."""
        return [g.name for g in self.gestures]

    @property
    def num_gestures(self) -> int:
        """Get the number of gestures in the set."""
        return len(self._gestures)

    def to_dict(self) -> Dict[str, Any]:
        """Convert gesture set to dictionary for serialization."""
        return {
            "name": self.name,
            "gestures": [g.to_dict() for g in self.gestures]
        }

    def save(self, path: Path) -> None:
        """Save gesture set to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GestureSet":
        """Create gesture set from dictionary."""
        gesture_set = cls(name=data["name"])
        for g_data in data["gestures"]:
            gesture_set.add_gesture(Gesture.from_dict(g_data))
        return gesture_set

    @classmethod
    def load(cls, path: Path) -> "GestureSet":
        """Load gesture set from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)

    def __iter__(self):
        return iter(self.gestures)

    def __len__(self):
        return self.num_gestures


# Simple prompt timings (seconds)
TASK_DURATION: float = 5.0
PAUSE_DURATION: float = 8.0


def format_prompt(current: Gesture, next_gesture: Optional[Gesture] = None) -> str:
    """Return a minimal prompt showing only the current task and the next task.

    Example:
        Current: Fist
        Hold: 5s
        Next: Open Hand
    """
    next_text = f"Next: {next_gesture.display_name}" if next_gesture else "Next: —"
    return f"Current: {current.display_name}\nHold: {int(TASK_DURATION)}s\n{next_text}"


def format_pause_prompt(next_gesture: Optional[Gesture] = None) -> str:
    """Return a minimal pause prompt that shows what comes next and pause duration.

    Example:
        Pause
        Next: Open Hand
        Resuming in 8s
    """
    next_text = f"Next: {next_gesture.display_name}" if next_gesture else "Next: —"
    return f"Pause\n{next_text}\nResuming in {int(PAUSE_DURATION)}s"


# Pre-defined gesture sets
def create_default_gesture_set() -> GestureSet:
    """
    Create a default gesture set with common hand gestures.
    """
    gesture_set = GestureSet(name="default")

    # Rest (baseline/neutral) - Label 0
    gesture_set.add_gesture(Gesture(
        name="rest",
        display_name="Rest",
        description="Relax your hand completely in a neutral position.",
        category=GestureCategory.REST,
        emoji="🖐🏻",
        duration_hint=2.0
    ))

    # Fist - Label 1
    gesture_set.add_gesture(Gesture(
        name="fist",
        display_name="Fist",
        description="Close your hand into a tight fist with thumb over fingers.",
        category=GestureCategory.GRIP,
        emoji="🤛🏻",
        duration_hint=3.0
    ))

    # Index finger to thumb (pinch)
    gesture_set.add_gesture(Gesture(
        name="pinch",
        display_name="Pinch",
        description="Touch your index finger tip to your thumb tip, keeping other fingers extended.",
        category=GestureCategory.FINGER,
        emoji="👌🏻",
        duration_hint=3.0
    ))

    # Three fingers to thumb (tripod)
    gesture_set.add_gesture(Gesture(
        name="tripod",
        display_name="Tripod",
        description="Touch index, middle, and ring finger tips to your thumb tip.",
        category=GestureCategory.FINGER,
        emoji="🤌🏻",
        duration_hint=3.0
    ))

    return gesture_set


def create_calibration_gesture_set() -> GestureSet:
    """
    Create a gesture set specifically for calibration.
    These gestures are chosen to activate different muscle groups
    for electrode orientation detection.
    """
    gesture_set = GestureSet(name="calibration")

    gesture_set.add_gesture(Gesture(
        name="cal_rest",
        display_name="Rest",
        description="Relax your hand completely.",
        category=GestureCategory.REST,
        duration_hint=2.0
    ))

    gesture_set.add_gesture(Gesture(
        name="cal_fist",
        display_name="Strong Fist",
        description="Make a tight fist and squeeze firmly.",
        category=GestureCategory.GRIP,
        duration_hint=2.0
    ))

    gesture_set.add_gesture(Gesture(
        name="cal_extend",
        display_name="Full Extension",
        description="Extend and spread all fingers as wide as possible.",
        category=GestureCategory.HAND,
        duration_hint=2.0
    ))

    gesture_set.add_gesture(Gesture(
        name="cal_thumb_out",
        display_name="Thumb Out",
        description="Move your thumb outward away from your hand.",
        category=GestureCategory.FINGER,
        duration_hint=2.0
    ))

    gesture_set.add_gesture(Gesture(
        name="cal_pinky_out",
        display_name="Pinky Out",
        description="Extend only your pinky finger.",
        category=GestureCategory.FINGER,
        duration_hint=2.0
    ))

    return gesture_set
