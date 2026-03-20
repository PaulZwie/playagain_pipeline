"""
Recording session management.

This module handles individual recording sessions, including metadata,
timestamps, and data organization.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
import json
import numpy as np

from playagain_pipeline.core.gesture import GestureSet


@dataclass
class RecordingMetadata:
    """Metadata for a recording session."""
    session_id: str
    subject_id: str
    created_at: datetime
    device_name: str
    num_channels: int
    sampling_rate: int
    gesture_set_name: str
    protocol_name: str
    calibration_applied: bool = False
    channel_mapping: Optional[List[int]] = None
    rotation_offset: int = 0  # Detected bracelet rotation (channels shifted from reference)
    rotation_confidence: float = 0.0  # Confidence of the rotation detection [0, 1]
    notes: str = ""
    bad_channels: List[int] = field(default_factory=list)  # Indices of bad channels (0-based)
    custom_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "subject_id": self.subject_id,
            "created_at": self.created_at.isoformat(),
            "device_name": self.device_name,
            "num_channels": self.num_channels,
            "sampling_rate": self.sampling_rate,
            "gesture_set_name": self.gesture_set_name,
            "protocol_name": self.protocol_name,
            "calibration_applied": self.calibration_applied,
            "channel_mapping": self.channel_mapping,
            "rotation_offset": self.rotation_offset,
            "rotation_confidence": self.rotation_confidence,
            "notes": self.notes,
            "bad_channels": self.bad_channels,
            "custom_metadata": self.custom_metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RecordingMetadata":
        data = data.copy()
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        # Ensure backward compatibility for older sessions without bad_channels
        if "bad_channels" not in data:
            data["bad_channels"] = []
        return cls(**data)


@dataclass
class RecordingTrial:
    """A single trial within a recording session."""
    trial_id: int
    gesture_name: str
    gesture_label: int
    start_sample: int
    end_sample: int
    start_time: float
    end_time: float
    is_valid: bool = True
    notes: str = ""
    # "gesture"          — normal training trial (default; used by all older sessions)
    # "calibration_sync" — waveout sync gesture; excluded from model training,
    #                      used only by the rotation-detection calibrator
    trial_type: str = "gesture"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trial_id": self.trial_id,
            "gesture_name": self.gesture_name,
            "gesture_label": self.gesture_label,
            "start_sample": self.start_sample,
            "end_sample": self.end_sample,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "is_valid": self.is_valid,
            "notes": self.notes,
            "trial_type": self.trial_type,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RecordingTrial":
        # trial_type was added later; older session JSON files won't have it.
        # Passing data as **kwargs lets the dataclass default ("gesture") fill in
        # for any missing key, so older sessions load transparently.
        return cls(**data)


class RecordingSession:
    """
    Manages a complete recording session with EMG data and trial annotations.

    A session contains:
    - Raw EMG data (samples x channels)
    - Trial annotations (start/end times for each gesture)
    - Metadata about the recording
    """

    def __init__(
        self,
        session_id: str,
        subject_id: str,
        device_name: str,
        num_channels: int,
        sampling_rate: int,
        gesture_set: GestureSet,
        protocol_name: str = "default"
    ):
        self.metadata = RecordingMetadata(
            session_id=session_id,
            subject_id=subject_id,
            created_at=datetime.now(),
            device_name=device_name,
            num_channels=num_channels,
            sampling_rate=sampling_rate,
            gesture_set_name=gesture_set.name,
            protocol_name=protocol_name
        )

        self.gesture_set = gesture_set
        self.trials: List[RecordingTrial] = []
        self._data_chunks: List[np.ndarray] = []
        self._current_sample: int = 0
        self._is_recording: bool = False
        self._current_trial_start: Optional[int] = None
        self._current_trial_gesture: Optional[str] = None
        self._current_trial_type: str = "gesture"   # tracks type for the in-progress trial
        self._source_dir: Optional[Path] = None  # Set when loaded from disk

    @property
    def is_recording(self) -> bool:
        return self._is_recording

    @property
    def total_samples(self) -> int:
        return self._current_sample

    @property
    def duration_seconds(self) -> float:
        return self._current_sample / self.metadata.sampling_rate

    def start_recording(self) -> None:
        """Start the recording session."""
        self._is_recording = True
        self.metadata.created_at = datetime.now()

    def stop_recording(self) -> None:
        """Stop the recording session."""
        self._is_recording = False
        # End any active trial
        if self._current_trial_start is not None:
            self.end_trial()

    def add_data(self, data: np.ndarray) -> None:
        """
        Add EMG data to the session.

        Args:
            data: EMG data array of shape (samples, channels)
        """
        if not self._is_recording:
            return

        if data.ndim == 1:
            data = data.reshape(-1, 1)

        self._data_chunks.append(data.copy())
        self._current_sample += data.shape[0]

    def start_trial(self, gesture_name: str, trial_type: str = "gesture") -> None:
        """
        Mark the start of a new trial.

        Args:
            gesture_name: Name of the gesture being performed.
            trial_type:   "gesture" (default) for normal training trials.
                          "calibration_sync" for the waveout sync gesture
                          recorded at the start of each session.  Calibration-sync
                          trials are *not* required to exist in the gesture set —
                          they bypass the gesture lookup and store gesture_label=-1.
        """
        if self._current_trial_start is not None:
            # End previous trial first
            self.end_trial()

        if trial_type != "calibration_sync":
            # Normal gesture: must exist in the gesture set
            gesture = self.gesture_set.get_gesture(gesture_name)
            if gesture is None:
                raise ValueError(f"Unknown gesture: {gesture_name}")

        self._current_trial_start = self._current_sample
        self._current_trial_gesture = gesture_name
        self._current_trial_type = trial_type

    def end_trial(self, is_valid: bool = True, notes: str = "") -> None:
        """
        Mark the end of the current trial.

        Args:
            is_valid: Whether this trial should be used for training
            notes: Any notes about the trial
        """
        if self._current_trial_start is None:
            return

        trial_type = self._current_trial_type

        if trial_type == "calibration_sync":
            # Calibration-sync trials are not in the gesture set.
            # Use gesture_label = -1 as a sentinel so they are never confused
            # with a real gesture class.
            gesture_label = -1
        else:
            gesture = self.gesture_set.get_gesture(self._current_trial_gesture)
            gesture_label = gesture.label_id

        trial = RecordingTrial(
            trial_id=len(self.trials),
            gesture_name=self._current_trial_gesture,
            gesture_label=gesture_label,
            start_sample=self._current_trial_start,
            end_sample=self._current_sample,
            start_time=self._current_trial_start / self.metadata.sampling_rate,
            end_time=self._current_sample / self.metadata.sampling_rate,
            is_valid=is_valid,
            notes=notes,
            trial_type=trial_type,
        )

        self.trials.append(trial)
        self._current_trial_start = None
        self._current_trial_gesture = None
        self._current_trial_type = "gesture"

    def get_data(self) -> np.ndarray:
        """
        Get all recorded data as a single array.

        Returns:
            EMG data array of shape (total_samples, num_channels)
        """
        if not self._data_chunks:
            return np.array([]).reshape(0, self.metadata.num_channels)
        return np.vstack(self._data_chunks)

    def get_trial_data(self, trial: RecordingTrial) -> np.ndarray:
        """
        Get data for a specific trial.

        Args:
            trial: The trial to get data for

        Returns:
            EMG data for the trial
        """
        all_data = self.get_data()
        return all_data[trial.start_sample:trial.end_sample]

    def get_valid_trials(self) -> List[RecordingTrial]:
        """
        Get valid gesture trials only.

        Calibration-sync trials (trial_type == "calibration_sync") are always
        excluded — they exist for rotation detection, not model training.
        Old sessions without a trial_type field default to "gesture" on load,
        so this is fully backward-compatible.
        """
        return [t for t in self.trials
                if t.is_valid and t.trial_type == "gesture"]

    def get_calibration_trials(self) -> List[RecordingTrial]:
        """
        Get calibration-sync trials recorded at the start of the session.

        These contain the waveout sync gesture used by AutoCalibrator for
        rotation detection.  Returns an empty list for older sessions that
        were recorded before calibration-sync trials were introduced.
        """
        return [t for t in self.trials
                if t.trial_type == "calibration_sync"]

    def save(self, directory: Path) -> None:
        """
        Save the session to disk.

        Saves:
        - metadata.json: Session metadata and trial information
        - gesture_set.json: Gesture definitions
        - data.npy: Raw EMG data (binary format)
        - data.csv: Raw EMG data (CSV format for easy viewing)
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        # Save metadata and trials
        session_info = {
            "metadata": self.metadata.to_dict(),
            "trials": [t.to_dict() for t in self.trials]
        }
        with open(directory / "metadata.json", 'w') as f:
            json.dump(session_info, f, indent=2)

        # Save gesture set
        self.gesture_set.save(directory / "gesture_set.json")

        # Save data in binary format (fast, compact)
        data = self.get_data()
        np.save(directory / "data.npy", data)

        # Also save as CSV (human-readable)
        csv_path = directory / "data.csv"
        header = ",".join([f"CH_{i+1}" for i in range(self.metadata.num_channels)])
        np.savetxt(csv_path, data, delimiter=",", header=header, comments='')

    @classmethod
    def load(cls, directory: Path) -> "RecordingSession":
        """
        Load a session from disk.

        Args:
            directory: Directory containing the saved session

        Returns:
            Loaded RecordingSession
        """
        directory = Path(directory)

        # Load metadata
        with open(directory / "metadata.json", 'r') as f:
            session_info = json.load(f)

        metadata = RecordingMetadata.from_dict(session_info["metadata"])

        # Load gesture set
        gesture_set = GestureSet.load(directory / "gesture_set.json")

        # Create session
        session = cls(
            session_id=metadata.session_id,
            subject_id=metadata.subject_id,
            device_name=metadata.device_name,
            num_channels=metadata.num_channels,
            sampling_rate=metadata.sampling_rate,
            gesture_set=gesture_set,
            protocol_name=metadata.protocol_name
        )
        session.metadata = metadata

        # Load trials
        session.trials = [
            RecordingTrial.from_dict(t) for t in session_info["trials"]
        ]

        # Load data — use memory-mapping by default so we only read
        # pages that are actually accessed.  The mmap is read-only; any
        # write (e.g. zeroing bad channels) triggers a copy-on-write via
        # numpy slicing, so callers stay safe.
        data = np.load(directory / "data.npy", mmap_mode="r")
        session._data_chunks = [data]
        session._current_sample = data.shape[0]
        session._source_dir = directory  # remember where we came from

        return session
