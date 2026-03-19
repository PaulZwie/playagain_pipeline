"""
Calibration system for electrode orientation detection.

This module provides robust algorithms to detect the orientation of EMG
electrodes and create channel mappings that correct for different bracelet
positions.

Uses the Maximum Energy Channel (MEC) method based on:
  Barona López et al., "An Energy-Based Method for Orientation Correction
  of EMG Bracelet Sensors in Hand Gesture Recognition Systems",
  Sensors 2020, 20, 6327.

The algorithm works as follows:
  1. A synchronization gesture is recorded (e.g. fist or waveOut).
  2. Per-channel energy is computed using the EMG energy formula:
       E_ch = sum_{i=2}^{L} |x_i * |x_i| - x_{i-1} * |x_{i-1}||
  3. The channel with the maximum average energy (MEC) is identified.
  4. Channels are circularly rearranged so the MEC becomes channel 0.

The MEC from the reference recording is stored. Subsequent recordings
find *their* MEC, and the rotation offset is the circular distance
between the two MECs.

Calibration can be performed directly from normal recording sessions —
no separate calibration recording is required.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple, Union
from pathlib import Path
import json
import numpy as np
from scipy import signal as scipy_signal
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
            data: EMG data of shape (samples, channels) or (windows, samples, channels)

        Returns:
            Reordered data with corrected channel mapping
        """
        if data.ndim == 2:
            if data.shape[1] != self.num_channels:
                raise ValueError(
                    f"Data has {data.shape[1]} channels, expected {self.num_channels}"
                )
            return data[:, self.channel_mapping]
        elif data.ndim == 3:
            if data.shape[2] != self.num_channels:
                raise ValueError(
                    f"Data has {data.shape[2]} channels, expected {self.num_channels}"
                )
            return data[:, :, self.channel_mapping]
        else:
            raise ValueError(f"Expected 2D or 3D array, got {data.ndim}D")


class CalibrationProcessor:
    """
    Processes calibration data to determine electrode orientation using the
    Maximum Energy Channel (MEC) method.

    The energy of each channel is computed using the formula from
    Barona López et al. (Sensors 2020):
        E_ch = sum_{i=2}^{L} |x_i * |x_i| - x_{i-1} * |x_{i-1}||

    The channel with the highest average energy across repetitions of a
    synchronization gesture is the MEC. The rotation offset is the circular
    distance between the current MEC and the reference MEC.

    Features:
    - Bandpass filtering to remove noise and motion artifacts
    - Energy computation per channel following the paper's formula
    - Multiple repetitions averaged with outlier rejection
    - Confidence scoring based on energy peak prominence
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

        # Design bandpass filter coefficients (20-450 Hz)
        nyquist = sampling_rate / 2.0
        low = min(20.0 / nyquist, 0.95)
        high = min(450.0 / nyquist, 0.95)
        if low < high:
            self._bp_b, self._bp_a = scipy_signal.butter(4, [low, high], btype='band')
        else:
            self._bp_b, self._bp_a = None, None

        # Design notch filter at 50 Hz (powerline interference)
        if sampling_rate > 100:
            notch_freq = 50.0 / nyquist
            if 0 < notch_freq < 1.0:
                self._notch_b, self._notch_a = scipy_signal.iirnotch(notch_freq, Q=30)
            else:
                self._notch_b, self._notch_a = None, None
        else:
            self._notch_b, self._notch_a = None, None

    def _bandpass_filter(self, data: np.ndarray) -> np.ndarray:
        """Apply bandpass and notch filters to EMG data.

        Args:
            data: EMG data (samples, channels)

        Returns:
            Filtered data (samples, channels)
        """
        filtered = data.copy()
        if data.shape[0] < 50:
            return filtered

        # Apply notch filter first (powerline removal)
        if self._notch_b is not None:
            try:
                filtered = scipy_signal.filtfilt(
                    self._notch_b, self._notch_a, filtered, axis=0
                )
            except ValueError:
                pass

        # Apply bandpass filter
        if self._bp_b is not None:
            try:
                filtered = scipy_signal.filtfilt(
                    self._bp_b, self._bp_a, filtered, axis=0
                )
            except ValueError:
                pass

        return filtered

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

    def compute_channel_energy(self, data: np.ndarray) -> np.ndarray:
        """
        Compute per-channel energy using the formula from Barona López et al.

        E_ch = sum_{i=2}^{L} |x_i * |x_i| - x_{i-1} * |x_{i-1}||

        Args:
            data: EMG data for a single gesture trial (samples, channels)

        Returns:
            Energy vector (num_channels,)
        """
        filtered = self._bandpass_filter(data)

        # Trim 15% from start and end to remove transition artifacts
        trim = max(1, int(filtered.shape[0] * 0.15))
        trimmed = filtered[trim:-trim] if filtered.shape[0] > 2 * trim + 20 else filtered

        n_channels = min(trimmed.shape[1], self.num_channels)
        energy = np.zeros(self.num_channels)
        for ch in range(n_channels):
            x = trimmed[:, ch]
            # E = sum |x_i * |x_i| - x_{i-1} * |x_{i-1}||
            term = x * np.abs(x)
            energy[ch] = np.sum(np.abs(term[1:] - term[:-1]))

        return energy

    def compute_energy_pattern(
        self,
        calibration_data: Dict[str, Union[np.ndarray, List[np.ndarray]]],
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Compute a combined energy pattern across all gestures.

        For each gesture, the per-channel energy is computed and averaged
        across multiple trials (with outlier rejection). The overall energy
        pattern sums contributions from all active (non-rest) gestures.

        Args:
            calibration_data: Gesture name -> EMG data or list of EMG data arrays

        Returns:
            (combined_energy, per_gesture_energy)
            combined_energy: (num_channels,) sum across gestures
            per_gesture_energy: dict of gesture -> (num_channels,)
        """
        per_gesture: Dict[str, np.ndarray] = {}

        for gesture_name, data in calibration_data.items():
            if isinstance(data, list):
                trial_energies = [self.compute_channel_energy(d) for d in data]
                per_gesture[gesture_name] = self._average_with_outlier_rejection(trial_energies)
            else:
                per_gesture[gesture_name] = self.compute_channel_energy(data)

        # Combine: sum active gesture energies (exclude rest)
        active_gestures = {g: e for g, e in per_gesture.items() if "rest" not in g.lower()}
        if not active_gestures:
            active_gestures = per_gesture

        combined = np.zeros(self.num_channels)
        for e in active_gestures.values():
            combined += e

        return combined, per_gesture

    def _average_with_outlier_rejection(
        self,
        energies: List[np.ndarray],
        threshold: float = 1.5,
    ) -> np.ndarray:
        """
        Average multiple energy vectors with outlier rejection.

        Uses median-based outlier detection: a trial whose total energy
        deviates by more than threshold * IQR from the median is discarded.

        Args:
            energies: List of per-channel energy arrays
            threshold: IQR multiplier for outlier detection

        Returns:
            Averaged energy vector
        """
        if len(energies) <= 1:
            return energies[0] if energies else np.zeros(self.num_channels)

        arr = np.array(energies)
        if len(energies) <= 2:
            return np.mean(arr, axis=0)

        # Compute total energy per trial for outlier detection
        totals = np.sum(arr, axis=1)
        q1 = np.percentile(totals, 25)
        q3 = np.percentile(totals, 75)
        iqr = q3 - q1
        lower = q1 - threshold * iqr
        upper = q3 + threshold * iqr
        mask = (totals >= lower) & (totals <= upper)
        if not mask.any():
            mask = np.ones(len(energies), dtype=bool)

        return np.mean(arr[mask], axis=0)

    def find_rotation_offset(
        self,
        current_energy: np.ndarray,
        reference_energy: np.ndarray,
    ) -> Tuple[int, float]:
        """
        Find the rotation offset by comparing the MEC of the current and
        reference energy patterns.

        The Maximum Energy Channel (MEC) is identified for both patterns.
        The rotation offset is the circular distance between them.

        For 32-channel devices (2-ring layout), energies are summed across rows
        to create a robust 16-channel angular profile. This prevents noise
        from shifting the peak between rings (e.g. index 0 vs 16).

        Confidence is based on how prominent the energy peak is—a sharp
        peak in a single channel means high confidence; a flat distribution
        means low confidence.

        Args:
            current_energy: Current per-channel energy (num_channels,)
            reference_energy: Reference per-channel energy (num_channels,)

        Returns:
            (rotation_offset, confidence)
        """
        n = self.num_channels

        # --- Enhanced logic for 32-channel (2-ring) devices ---
        if n == 32:
            # Fold 32 channels into 16 angular sectors by summing rings.
            # Sector i = Channel i (Inner) + Channel i+16 (Outer)
            # This makes the detection insensitive to proximal/distal shifts.
            current_folded = current_energy[:16] + current_energy[16:]
            reference_folded = reference_energy[:16] + reference_energy[16:]
            
            # Find MEC on the folded profile (0..15)
            # We use circular cross-correlation to find the best shift,
            # which is more robust than simple argmax difference.
            # (Though paper uses MEC, cross-corr is consistent with "max energy alignment")
            # Sticking to MEC as per paper request:
            curr_mec = int(np.argmax(current_folded))
            ref_mec = int(np.argmax(reference_folded))
            
            rotation_offset = (curr_mec - ref_mec) % 16
            
            # Re-calculate confidence on the cleaner 16-channel profile
            confidence = self._compute_energy_confidence(current_folded)
            
            # Note: We return offset in 0..15. The channel mapping logic
            # handles this as a column shift. Row shifts (flipping) are ignored.
            return rotation_offset, confidence

        # --- Standard logic for single-ring (e.g. 8-channel) devices ---
        current_mec = int(np.argmax(current_energy))
        reference_mec = int(np.argmax(reference_energy))

        # Rotation offset: how many channels the bracelet has shifted
        rotation_offset = (current_mec - reference_mec) % n

        # Confidence scoring based on peak prominence
        # A good sync gesture produces a clear energy peak in one channel.
        confidence = self._compute_energy_confidence(current_energy)

        return rotation_offset, confidence

    def _compute_energy_confidence(self, energy: np.ndarray) -> float:
        """
        Compute confidence from the energy distribution.

        Confidence is high when one channel clearly dominates (sharp peak),
        and low when energy is spread uniformly.

        Uses two metrics:
        1. Peak-to-second ratio: how much larger the MEC is vs the runner-up
        2. Peak fraction: what fraction of total energy is in the MEC

        Returns:
            Confidence in [0, 1]
        """
        total = np.sum(energy)
        if total < 1e-12:
            return 0.0

        sorted_e = np.sort(energy)[::-1]
        peak = sorted_e[0]
        second = sorted_e[1] if len(sorted_e) > 1 else 0.0

        # Peak-to-second ratio: ratio 1.0 → 0%, 1.5 → 50%, 2.0+ → 100%
        if second > 1e-12:
            ratio = peak / second
            ratio_score = float(np.clip((ratio - 1.0) / 1.0, 0.0, 1.0))
        else:
            ratio_score = 1.0

        # Peak fraction: for n channels, uniform → 1/n; peaked → close to 1
        peak_frac = peak / total
        expected_uniform = 1.0 / max(len(energy), 1)
        frac_score = float(np.clip(
            (peak_frac - expected_uniform) / (1.0 - expected_uniform),
            0.0, 1.0
        ))

        # Blend: fraction is more reliable
        return 0.4 * ratio_score + 0.6 * frac_score

    def create_channel_mapping(self, rotation_offset: int) -> List[int]:
        """
        Create a channel mapping based on rotation offset.

        Channels are rearranged so that channel[rotation_offset] becomes
        channel[0] in the new order.

        For 32-channel devices (2x16 layout), the mapping respects the 
        physical electrode topology where rows wraps independently.
        (Indices 0-15 and 16-31).

        Args:
            rotation_offset: Number of channels the bracelet has rotated

        Returns:
            List of channel indices in corrected order
        """
        if self.num_channels == 32:
            # Handle split-ring topology for 2x16 electrode layout
            # Row 1: 0-15 -> Wraps 15 to 0
            # Row 2: 16-31 -> Wraps 31 to 16
            
            row_shift = rotation_offset // 16
            col_shift = rotation_offset % 16
            
            mapping = []
            for i in range(self.num_channels):
                r = i // 16
                c = i % 16
                
                # Apply shift respecting the 2-row topology
                target_r = (r + row_shift) % 2
                target_c = (c + col_shift) % 16
                
                mapping.append(target_r * 16 + target_c)
            return mapping

        # Default ring topology for other channel counts
        return [(i + rotation_offset) % self.num_channels
                for i in range(self.num_channels)]

    def calibrate_from_data(
        self,
        calibration_data: Dict[str, Union[np.ndarray, List[np.ndarray]]],
        device_name: str = "unknown",
        reference_result: Optional[CalibrationResult] = None,
        plot_path: Optional[Path] = None,
    ) -> CalibrationResult:
        """
        Perform calibration from recorded gesture data using the MEC method.

        Args:
            calibration_data: Dictionary mapping gesture names to EMG data arrays.
                Values can be a single array or a list of arrays (multiple trials)
                which will be averaged with outlier rejection.
            device_name: Name of the device
            reference_result: Optional reference calibration to match against

        Returns:
            CalibrationResult with channel mapping
        """
        combined_energy, per_gesture_energy = self.compute_energy_pattern(calibration_data)

        # Store energy patterns as reference_patterns (serialised as dict of arrays)
        energy_patterns = {g: e for g, e in per_gesture_energy.items()}
        energy_patterns["__combined__"] = combined_energy

        per_gesture_confidence = {}
        for g, e in per_gesture_energy.items():
            per_gesture_confidence[g] = self._compute_energy_confidence(e)

        # Select the best sync pattern (e.g. waveOut or Fist)
        ref_patterns_map = reference_result.reference_patterns if reference_result else None
        
        sync_energy, ref_sync_energy, sync_gesture = self._select_sync_pattern(
            energy_patterns, ref_patterns_map
        )
        
        # Calculate offset using the selected patterns
        if ref_sync_energy is not None and len(ref_sync_energy) == self.num_channels:
            offset, confidence = self.find_rotation_offset(sync_energy, ref_sync_energy)
        else:
            offset = 0
            # If creating a new reference, confidence is how "peaked" the selected gesture is
            confidence = self._compute_energy_confidence(sync_energy)

        channel_mapping = self.create_channel_mapping(offset)

        # --- Debug plot for energy profile alignment ---
        if plot_path is not None:
             # Use the actual sync patterns for plotting, not the combined ones
             # unless combined was selected.
            self._plot_energy_profile(sync_energy, ref_sync_energy, offset, confidence, plot_path)

        return CalibrationResult(
            created_at=datetime.now(),
            device_name=device_name,
            num_channels=self.num_channels,
            rotation_offset=offset,
            channel_mapping=channel_mapping,
            confidence=confidence,
            reference_patterns=energy_patterns,
            metadata={
                "method": "maximum_energy_channel",
                "sync_gesture": sync_gesture,  # Record which gesture was used
                "per_gesture_confidence": per_gesture_confidence,
                "num_gestures": len(calibration_data),
                "has_reference": reference_result is not None,
                "mec_channel": int(np.argmax(sync_energy)),
            },
        )

    def _select_sync_pattern(
        self,
        patterns: Dict[str, np.ndarray],
        reference_patterns: Optional[Dict[str, np.ndarray]] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray], str]:
        """
        Select the best gesture pattern for synchronization.

        Prioritizes gestures recommended by Barona López et al. (Sensors 2020):
        waveOut > Fist > Open > waveIn > Pinch > Tripod.

        If a reference is provided, selects the highest priority gesture
        available in *both* sets. If no common gesture, falls back to
        "__combined__".

        Returns:
            (current_pattern, reference_pattern, gesture_name)
        """
        # Normalized priority list (lowercase for matching)
        priority = ["waveout", "fist", "open", "wavein", "pinch", "tripod"]

        # Helper to find case-insensitive match
        def find_match(info_dict, target):
            for key in info_dict:
                if key.lower() == target:
                    return key
            return None

        # 1. If we have a reference, try to find the best common gesture
        if reference_patterns:
            for p in priority:
                curr_key = find_match(patterns, p)
                ref_key = find_match(reference_patterns, p)

                if curr_key and ref_key:
                    return patterns[curr_key], reference_patterns[ref_key], curr_key

            # Fallback: use __combined__ if both have it (they should)
            if "__combined__" in patterns and "__combined__" in reference_patterns:
                return patterns["__combined__"], reference_patterns["__combined__"], "Combined (Fallback)"

        # 2. No reference (or creating one): pick best available from current
        for p in priority:
            key = find_match(patterns, p)
            if key:
                return patterns[key], None, key

        # 3. Last resort: Combined
        return patterns.get("__combined__", np.zeros(self.num_channels)), None, "Combined"

    def set_reference_calibration(self, result: Optional[CalibrationResult]) -> None:
        """Set or clear the reference calibration for future calibrations."""
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

    def save_as_reference(self, result: CalibrationResult,
                          recompute_all: bool = False,
                          data_dir: Optional[Path] = None) -> None:
        """Save a calibration result as the reference.

        Args:
            result: The calibration result to use as the new reference.
            recompute_all: If True, re-detect rotations for all existing sessions
                           relative to this new reference.
            data_dir: Root data directory (needed when recompute_all=True).
        """
        # When saving as reference the result itself becomes offset=0
        # (it IS the reference). Store original patterns but reset offset.
        ref_result = CalibrationResult(
            created_at=result.created_at,
            device_name=result.device_name,
            num_channels=result.num_channels,
            rotation_offset=0,
            channel_mapping=list(range(result.num_channels)),
            confidence=1.0,
            reference_patterns=result.reference_patterns,
            metadata={**result.metadata, "is_reference": True},
        )
        ref_path = self.data_dir / "reference_calibration.json"
        ref_result.save(ref_path)
        self.processor.set_reference_calibration(ref_result)

        if recompute_all and data_dir is not None:
            backfill_session_rotations(
                data_dir,
                calibrations_dir=self.data_dir,
                num_channels=self.processor.num_channels,
                sampling_rate=self.processor.sampling_rate,
                force=True,
            )

    def calibrate(
        self,
        calibration_data: Dict[str, Any],
        device_name: str = "unknown",
        save_as_reference: bool = False
    ) -> CalibrationResult:
        """
        Perform calibration from gesture data.

        Args:
            calibration_data: Dictionary mapping gesture names to EMG data.
                Values can be single arrays or lists of arrays (multiple trials).
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

    def calibrate_from_session(self, session, plot_path: Optional[Path] = None) -> CalibrationResult:
        """
        Perform calibration directly from a normal recording session.

        Extracts per-gesture trial data from the session and uses it to
        compute calibration patterns. Multiple trials of the same gesture
        are averaged with outlier rejection for robustness.

        This eliminates the need for a separate calibration recording.

        Args:
            session: A RecordingSession with annotated trials
            plot_path: Optional path to save a debug plot of the energy profile

        Returns:
            CalibrationResult with rotation offset and channel mapping
        """
        # Group trial data by gesture name
        gesture_trials: Dict[str, List[np.ndarray]] = {}
        all_data = session.get_data()

        valid_trials = session.get_valid_trials()
        if not valid_trials:
            raise ValueError("Session has no valid trials to calibrate from")

        for trial in valid_trials:
            trial_data = all_data[trial.start_sample:trial.end_sample]
            if trial_data.shape[0] < 10:  # Skip very short trials
                continue

            if trial.gesture_name not in gesture_trials:
                gesture_trials[trial.gesture_name] = []
            gesture_trials[trial.gesture_name].append(trial_data)

        if not gesture_trials:
            raise ValueError("No usable trial data found in session")

        # Pass as lists of arrays — the processor handles multi-trial averaging
        result = self.processor.calibrate_from_data(
            gesture_trials,
            device_name=session.metadata.device_name,
            reference_result=self.processor.get_reference_calibration(),
            plot_path=plot_path
        )

        self._current_calibration = result

        # Save calibration
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result.save(self.data_dir / f"calibration_{timestamp}.json")

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
            data: EMG data (samples, channels) or (windows, samples, channels)
            calibration: Calibration to apply (uses current if None)

        Returns:
            Calibrated data with corrected channel order
        """
        if calibration is None:
            calibration = self._current_calibration

        if calibration is None:
            # No calibration — return data unchanged
            return data

        return calibration.apply_to_data(data)

    def detect_session_rotation(self, session, save_to_metadata: bool = True, save_plot: bool = False) -> Optional[CalibrationResult]:
        """
        Detect bracelet rotation for a session and optionally write it to the session metadata.

        This is designed to be called automatically after each recording session
        is saved. It computes the rotation offset relative to the reference
        calibration and stores it in the session's metadata so that:
        - The rotation info is persisted alongside the recording data
        - Dataset creation can use per-session rotation for alignment
        - Models can be trained on data from multiple bracelet positions

        If no reference calibration exists, the first session with enough gesture
        data is automatically saved as the reference.

        Args:
            session: A RecordingSession with annotated trials
            save_to_metadata: If True, writes rotation_offset and rotation_confidence
                              to the session's metadata fields
            save_plot: If True, saves a debug plot in the session folder

        Returns:
            CalibrationResult if detection succeeded, None otherwise
        """
        valid_trials = session.get_valid_trials()
        if not valid_trials:
            return None

        # Need at least some non-rest gesture data for meaningful rotation detection
        non_rest = [t for t in valid_trials if "rest" not in t.gesture_name.lower()]
        if not non_rest:
            return None

        plot_path = None
        if save_plot:
            # We assume session has a path or we can construct one.
            # session doesn't store its own path? DataManager loads it from path.
            # But here we might not know the path easily if it's new.
            # However, detect_session_rotation is often called with a session object.
            # Let's save it to calibration dir by session ID if possible.
            try:
                # Assuming session_id is unique
                plot_filename = f"rotation_plot_{session.metadata.session_id}.png"
                plot_path = self.data_dir / "plots" / plot_filename
            except Exception:
                pass

        try:
            result = self.calibrate_from_session(session, plot_path=plot_path)
        except (ValueError, Exception) as e:
            print(f"[AutoCalibrator] Could not detect rotation for session "
                  f"'{session.metadata.session_id}': {e}")
            return None

        if save_to_metadata:
            session.metadata.rotation_offset = result.rotation_offset
            session.metadata.rotation_confidence = result.confidence
            session.metadata.calibration_applied = True
            session.metadata.channel_mapping = result.channel_mapping

        return result

def backfill_session_rotations(data_dir: Path, calibrations_dir: Optional[Path] = None,
                                num_channels: int = 32, sampling_rate: int = 2000,
                                force: bool = False, save_plots: bool = False) -> Dict[str, Dict[str, Any]]:
    """
    Backfill rotation offsets for all existing recording sessions.

    Scans all sessions, detects their bracelet rotation relative to the
    reference calibration, and writes the rotation_offset and rotation_confidence
    into each session's metadata.json. Sessions that already have a non-zero
    rotation_offset are skipped unless force=True.

    The first session (chronologically) with enough gesture data is used as the
    reference if no reference calibration exists yet.

    This is a one-time migration utility — new sessions get rotation info
    automatically during recording.

    Args:
        data_dir: Root data directory (containing sessions/, calibrations/ etc.)
        calibrations_dir: Directory for calibration files (default: data_dir/calibrations)
        num_channels: Number of EMG channels
        sampling_rate: Sampling rate in Hz
        force: If True, re-detect rotation even for sessions that already have it
        save_plots: If True, generate debug plots for each session

    Returns:
        Dictionary mapping session_id to {"rotation_offset": int, "confidence": float, "status": str}
    """
    from playagain_pipeline.core.data_manager import DataManager

    data_dir = Path(data_dir)
    if calibrations_dir is None:
        calibrations_dir = data_dir / "calibrations"

    dm = DataManager(data_dir)
    calibrator = AutoCalibrator(calibrations_dir, num_channels, sampling_rate)

    # Check if existing reference has compatible pattern format.
    # Energy-based patterns store per-gesture energy vectors (num_channels each)
    # plus a "__combined__" key. Detect stale multi-feature patterns.
    if calibrator.has_reference:
        ref = calibrator.processor.get_reference_calibration()
        combined = ref.reference_patterns.get("__combined__")
        if combined is None or len(combined) != num_channels:
            print(f"[Backfill] Reference pattern format outdated or incompatible. "
                  f"Will regenerate from first session.")
            calibrator.processor.set_reference_calibration(None)

    results = {}

    for subject in dm.list_subjects():
        for session_id in dm.list_sessions(subject):
            try:
                session = dm.load_session(subject, session_id)
            except Exception as e:
                results[f"{subject}/{session_id}"] = {
                    "status": f"load_error: {e}",
                    "rotation_offset": 0,
                    "confidence": 0.0
                }
                continue

            # Skip if already has rotation data (unless force)
            # Use confidence > 0 as indicator that rotation was computed,
            # since offset=0 can be a valid computed result
            already_computed = (session.metadata.rotation_offset != 0 or
                                session.metadata.rotation_confidence > 0)
            if not force and already_computed:
                results[f"{subject}/{session_id}"] = {
                    "status": "already_has_rotation",
                    "rotation_offset": session.metadata.rotation_offset,
                    "confidence": session.metadata.rotation_confidence
                }
                continue

            # Detect rotation
            cal_result = calibrator.detect_session_rotation(session, save_to_metadata=True, save_plot=save_plots)

            if cal_result is not None:
                # If no reference exists yet, save first successful calibration as reference
                if not calibrator.has_reference:
                    calibrator.save_as_reference(cal_result)
                    print(f"[Backfill] Saved '{subject}/{session_id}' as reference calibration")

                # Re-save session metadata with rotation info
                session_path = dm.get_session_path(subject, session_id)
                session.save(session_path)

                results[f"{subject}/{session_id}"] = {
                    "status": "detected",
                    "rotation_offset": cal_result.rotation_offset,
                    "confidence": round(cal_result.confidence, 4)
                }
                print(f"[Backfill] {subject}/{session_id}: "
                      f"rotation={cal_result.rotation_offset}, "
                      f"confidence={cal_result.confidence:.2%}")
            else:
                results[f"{subject}/{session_id}"] = {
                    "status": "detection_failed",
                    "rotation_offset": 0,
                    "confidence": 0.0
                }
                print(f"[Backfill] {subject}/{session_id}: detection failed (not enough data)")

    return results
