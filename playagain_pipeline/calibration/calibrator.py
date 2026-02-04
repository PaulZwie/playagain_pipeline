"""
Calibration system for electrode orientation detection.

This module provides algorithms to detect the orientation of EMG electrodes
and create channel mappings that correct for different bracelet positions.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
import json
import numpy as np
from scipy import signal
from scipy.stats import pearsonr
from datetime import datetime


@dataclass
class CalibrationResult:
    """Result of a calibration procedure."""
    created_at: datetime
    device_name: str
    num_channels: int
    rotation_offset: int  # Channel offset for rotation correction
    channel_mapping: List[int]  # Corrected channel order
    confidence: float  # Confidence score of the calibration
    reference_patterns: Dict[str, np.ndarray]  # Patterns used for calibration
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "created_at": self.created_at.isoformat(),
            "device_name": self.device_name,
            "num_channels": self.num_channels,
            "rotation_offset": self.rotation_offset,
            "channel_mapping": self.channel_mapping,
            "confidence": self.confidence,
            "reference_patterns": {
                k: v.tolist() for k, v in self.reference_patterns.items()
            },
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CalibrationResult":
        data = data.copy()
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        data["reference_patterns"] = {
            k: np.array(v) for k, v in data["reference_patterns"].items()
        }
        return cls(**data)

    def save(self, path: Path) -> None:
        """Save calibration result to file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "CalibrationResult":
        """Load calibration result from file."""
        with open(path, 'r') as f:
            return cls.from_dict(json.load(f))

    def apply_to_data(self, data: np.ndarray) -> np.ndarray:
        """
        Apply the calibration to EMG data.

        Args:
            data: EMG data of shape (samples, channels)

        Returns:
            Reordered data with corrected channel mapping
        """
        if data.shape[1] != self.num_channels:
            raise ValueError(
                f"Data has {data.shape[1]} channels, expected {self.num_channels}"
            )
        return data[:, self.channel_mapping]


class CalibrationProcessor:
    """
    Processes calibration data to determine electrode orientation.

    Uses activation patterns from specific gestures to identify
    which muscles are under which electrodes, allowing correction
    for different bracelet orientations.
    """

    def __init__(self, num_channels: int = 32, sampling_rate: int = 2000):
        """
        Initialize the calibration processor.

        Args:
            num_channels: Number of EMG channels
            sampling_rate: Sampling rate in Hz
        """
        self.num_channels = num_channels
        self.sampling_rate = sampling_rate
        self._reference_calibration: Optional[CalibrationResult] = None

    def compute_rms_envelope(
        self,
        data: np.ndarray,
        window_ms: int = 100
    ) -> np.ndarray:
        """
        Compute RMS envelope of EMG signals.

        Args:
            data: EMG data (samples, channels)
            window_ms: Window size in milliseconds

        Returns:
            RMS envelope (samples, channels)
        """
        window_samples = int(window_ms * self.sampling_rate / 1000)
        window_samples = max(1, window_samples)

        # Compute RMS using convolution
        squared = data ** 2
        window = np.ones(window_samples) / window_samples

        envelope = np.zeros_like(data)
        for ch in range(data.shape[1]):
            envelope[:, ch] = np.sqrt(
                np.convolve(squared[:, ch], window, mode='same')
            )

        return envelope

    def compute_activation_pattern(
        self,
        data: np.ndarray,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Compute the spatial activation pattern across channels.

        Args:
            data: EMG data for a single gesture (samples, channels)
            normalize: Whether to normalize the pattern

        Returns:
            Activation pattern (num_channels,)
        """
        # Compute RMS envelope
        envelope = self.compute_rms_envelope(data)

        # Get mean activation per channel
        pattern = np.mean(envelope, axis=0)

        if normalize and np.max(pattern) > 0:
            pattern = pattern / np.max(pattern)

        return pattern

    def find_rotation_offset(
        self,
        current_patterns: Dict[str, np.ndarray],
        reference_patterns: Dict[str, np.ndarray]
    ) -> Tuple[int, float]:
        """
        Find the rotation offset that best aligns current patterns with reference.

        Uses circular cross-correlation to find the optimal rotation.

        Args:
            current_patterns: Current activation patterns by gesture
            reference_patterns: Reference patterns to match against

        Returns:
            Tuple of (rotation_offset, confidence_score)
        """
        # Find common gestures
        common_gestures = set(current_patterns.keys()) & set(reference_patterns.keys())
        if not common_gestures:
            raise ValueError("No common gestures between current and reference")

        best_offset = 0
        best_score = -1

        # Try each possible rotation
        for offset in range(self.num_channels):
            total_score = 0

            for gesture in common_gestures:
                current = current_patterns[gesture]
                reference = reference_patterns[gesture]

                # Apply rotation to current pattern
                rotated = np.roll(current, offset)

                # Compute correlation
                corr, _ = pearsonr(rotated, reference)
                total_score += corr

            avg_score = total_score / len(common_gestures)

            if avg_score > best_score:
                best_score = avg_score
                best_offset = offset

        return best_offset, best_score

    def create_channel_mapping(self, rotation_offset: int) -> List[int]:
        """
        Create a channel mapping based on rotation offset.

        Args:
            rotation_offset: Number of channels to rotate

        Returns:
            List of channel indices in corrected order
        """
        return [(i - rotation_offset) % self.num_channels
                for i in range(self.num_channels)]

    def calibrate_from_data(
        self,
        calibration_data: Dict[str, np.ndarray],
        device_name: str = "unknown",
        reference_result: Optional[CalibrationResult] = None
    ) -> CalibrationResult:
        """
        Perform calibration from recorded gesture data.

        Args:
            calibration_data: Dictionary mapping gesture names to EMG data arrays
            device_name: Name of the device
            reference_result: Optional reference calibration to match against

        Returns:
            CalibrationResult with channel mapping
        """
        # Compute activation patterns for each gesture
        current_patterns = {}
        for gesture_name, data in calibration_data.items():
            current_patterns[gesture_name] = self.compute_activation_pattern(data)

        if reference_result is not None:
            # Find rotation relative to reference
            offset, confidence = self.find_rotation_offset(
                current_patterns,
                reference_result.reference_patterns
            )
        else:
            # No reference - use identity mapping
            offset = 0
            confidence = 1.0

        channel_mapping = self.create_channel_mapping(offset)

        result = CalibrationResult(
            created_at=datetime.now(),
            device_name=device_name,
            num_channels=self.num_channels,
            rotation_offset=offset,
            channel_mapping=channel_mapping,
            confidence=confidence,
            reference_patterns=current_patterns
        )

        return result

    def set_reference_calibration(self, result: CalibrationResult) -> None:
        """Set a calibration result as the reference for future calibrations."""
        self._reference_calibration = result

    def get_reference_calibration(self) -> Optional[CalibrationResult]:
        """Get the current reference calibration."""
        return self._reference_calibration


class AutoCalibrator:
    """
    High-level interface for automatic calibration.

    Handles the complete calibration workflow including:
    - Running calibration protocol
    - Processing recorded data
    - Storing and loading calibration results
    """

    def __init__(
        self,
        data_dir: Path,
        num_channels: int = 32,
        sampling_rate: int = 2000
    ):
        """
        Initialize the auto-calibrator.

        Args:
            data_dir: Directory for storing calibration data
            num_channels: Number of EMG channels
            sampling_rate: Sampling rate in Hz
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.processor = CalibrationProcessor(num_channels, sampling_rate)
        self._current_calibration: Optional[CalibrationResult] = None

        # Try to load existing reference calibration
        self._load_reference()

    def _load_reference(self) -> None:
        """Load existing reference calibration if available."""
        ref_path = self.data_dir / "reference_calibration.json"
        if ref_path.exists():
            try:
                ref = CalibrationResult.load(ref_path)
                self.processor.set_reference_calibration(ref)
            except Exception as e:
                print(f"Warning: Could not load reference calibration: {e}")

    def save_as_reference(self, result: CalibrationResult) -> None:
        """Save a calibration result as the reference."""
        ref_path = self.data_dir / "reference_calibration.json"
        result.save(ref_path)
        self.processor.set_reference_calibration(result)

    def calibrate(
        self,
        calibration_data: Dict[str, np.ndarray],
        device_name: str = "unknown",
        save_as_reference: bool = False
    ) -> CalibrationResult:
        """
        Perform calibration from gesture data.

        Args:
            calibration_data: Dictionary mapping gesture names to EMG data
            device_name: Name of the device
            save_as_reference: Whether to save this as the new reference

        Returns:
            CalibrationResult
        """
        result = self.processor.calibrate_from_data(
            calibration_data,
            device_name,
            self.processor.get_reference_calibration()
        )

        self._current_calibration = result

        # Save calibration
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result.save(self.data_dir / f"calibration_{timestamp}.json")

        if save_as_reference:
            self.save_as_reference(result)

        return result

    @property
    def current_calibration(self) -> Optional[CalibrationResult]:
        """Get the current calibration result."""
        return self._current_calibration

    @property
    def has_reference(self) -> bool:
        """Check if a reference calibration is available."""
        return self.processor.get_reference_calibration() is not None

    def apply_calibration(
        self,
        data: np.ndarray,
        calibration: Optional[CalibrationResult] = None
    ) -> np.ndarray:
        """
        Apply calibration to EMG data.

        Args:
            data: EMG data (samples, channels)
            calibration: Calibration to apply (uses current if None)

        Returns:
            Calibrated data with corrected channel order
        """
        if calibration is None:
            calibration = self._current_calibration

        if calibration is None:
            # No calibration - return data unchanged
            return data

        return calibration.apply_to_data(data)
