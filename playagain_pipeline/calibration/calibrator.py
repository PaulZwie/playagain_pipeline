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
import re


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

    # ------------------------------------------------------------------
    # Energy pre-processing helpers
    # ------------------------------------------------------------------

    def _normalize_energy(self, energy: np.ndarray) -> np.ndarray:
        """
        L2-normalise an energy vector so that inter-session amplitude
        differences do not bias the MEC location.

        Without normalisation a session recorded with a stronger contraction
        would shift the raw argmax toward a spuriously dominant channel
        simply because the overall signal was louder that day, not because
        the bracelet was aligned differently.
        """
        norm = np.linalg.norm(energy)
        if norm < 1e-12:
            return energy.copy()
        return energy / norm

    def _smooth_energy_circular(
        self, energy: np.ndarray, sigma: float = 0.8
    ) -> np.ndarray:
        """
        Apply a circular Gaussian smoothing pass to the energy ring.

        A single noisy electrode can steal the argmax from the true
        anatomical peak.  A light smooth (sigma ≈ 1 channel) blends each
        channel with its immediate neighbours while preserving the dominant
        spatial mode.  The padding is circular so the smoothing wraps
        correctly across the ring boundary (channel 0 ↔ channel N−1).

        Args:
            energy: Per-channel energy vector (num_channels,)
            sigma:  Gaussian sigma in channel units

        Returns:
            Smoothed energy vector of the same length
        """
        from scipy.ndimage import gaussian_filter1d
        n = len(energy)
        if n < 3 or sigma <= 0:
            return energy.copy()
        pad = min(int(4 * sigma) + 1, n)
        extended = np.concatenate([energy[-pad:], energy, energy[:pad]])
        smoothed = gaussian_filter1d(extended, sigma=sigma)
        return smoothed[pad: pad + n]

    def _compute_xcorr_confidence(self, xcorr: np.ndarray) -> float:
        """
        Confidence of the rotation estimate derived from a circular
        cross-correlation vector.

        A sharp, isolated peak in xcorr means the two profiles align
        clearly at exactly one shift — high confidence.  A broad or flat
        xcorr means any shift is almost equally plausible — low confidence.

        Metric: z-score of the best shift  = (peak − mean) / (std + ε),
        mapped to [0, 1] via clipping at z = 5.  This is more sensitive to
        sharpness of the alignment than a peak-to-second ratio because it
        accounts for the full spread of the distribution.

        Args:
            xcorr: Real-valued circular cross-correlation (any length)

        Returns:
            Confidence in [0, 1]
        """
        if len(xcorr) < 2:
            return 0.0
        peak = float(np.max(xcorr))
        mu   = float(np.mean(xcorr))
        std  = float(np.std(xcorr))
        if std < 1e-12:
            return 0.0          # completely flat → no information
        z = (peak - mu) / std
        return float(np.clip(z / 5.0, 0.0, 1.0))

    # ------------------------------------------------------------------

    def find_rotation_offset(
        self,
        current_energy: np.ndarray,
        reference_energy: np.ndarray,
    ) -> Tuple[int, float]:
        """
        Find the rotation offset using circular cross-correlation of the
        L2-normalised, smoothed energy profiles.

        **Why cross-correlation instead of argmax difference?**

        The original MEC paper (Barona López et al., Sensors 2020) compares
        the two peak channels:
            offset = argmax(current) − argmax(reference)
        This is fast but brittle: a single noisy electrode can shift the
        argmax by ±1 (or more), causing a systematic mis-correction.

        Circular cross-correlation instead maximises the inner product
        between the two profiles at every possible circular shift,
        effectively using *all* channels as corroborating evidence.  The
        shift that best aligns the whole ring wins — identical in spirit to
        the paper's MEC goal but far more robust.

        Algorithm:
          1. For 32-ch devices, fold into 16 angular sectors (row-sum),
             making detection insensitive to proximal/distal row shifts.
          2. L2-normalise both profiles to remove inter-session gain bias.
          3. Lightly smooth with a circular Gaussian (suppresses single-
             channel impulse noise before peak-finding).
          4. Compute circular cross-correlation via FFT.
          5. The lag at max xcorr is the rotation offset.
          6. Confidence is the z-score of the xcorr peak (measures how
             isolated and unambiguous the winning shift is).

        Args:
            current_energy:   Current per-channel energy (num_channels,)
            reference_energy: Reference per-channel energy (num_channels,)

        Returns:
            (rotation_offset, confidence)
        """
        n = self.num_channels

        # Step 1 — fold 32-ch into 16 angular sectors
        if n == 32:
            # Sector i = inner-ring channel i  +  outer-ring channel i+16
            # This collapses the 2-row topology to a pure azimuthal profile.
            current_profile   = current_energy[:16]  + current_energy[16:]
            reference_profile = reference_energy[:16] + reference_energy[16:]
        else:
            current_profile   = current_energy.copy()
            reference_profile = reference_energy.copy()

        # Step 2 — L2-normalise: equalise inter-session amplitude
        curr_norm = self._normalize_energy(current_profile)
        ref_norm  = self._normalize_energy(reference_profile)

        # Step 3 — circular Gaussian smooth: suppress single-channel spikes
        curr_smooth = self._smooth_energy_circular(curr_norm,  sigma=0.8)
        ref_smooth  = self._smooth_energy_circular(ref_norm,   sigma=0.8)

        # Step 4 — circular cross-correlation via FFT
        # xcorr[lag] = Σ_k  curr[k] · ref[(k − lag) mod N]
        # Peaks at the lag equal to the true rotation offset.
        xcorr = np.real(
            np.fft.ifft(
                np.fft.fft(curr_smooth) * np.conj(np.fft.fft(ref_smooth))
            )
        )

        # Step 5 — the winning lag is the rotation offset
        rotation_offset = int(np.argmax(xcorr))

        # Step 6 — confidence from xcorr peak z-score
        confidence = self._compute_xcorr_confidence(xcorr)

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
        signal_mode: str = "monopolar",
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
        reference_incompatible = None
        ref_patterns_map = None
        if reference_result is not None:
            ref_mode = str(reference_result.metadata.get("signal_mode", "monopolar")).lower()
            cur_mode = str(signal_mode or "monopolar").lower()
            if reference_result.num_channels != self.num_channels:
                reference_incompatible = (
                    f"channel mismatch: reference={reference_result.num_channels}, "
                    f"current={self.num_channels}"
                )
            elif ref_mode != cur_mode:
                reference_incompatible = (
                    f"signal mode mismatch: reference={ref_mode}, current={cur_mode}"
                )
            else:
                ref_patterns_map = reference_result.reference_patterns
        
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
                "has_reference": ref_patterns_map is not None,
                "signal_mode": str(signal_mode or "monopolar").lower(),
                "reference_incompatible": reference_incompatible,
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

        Prioritises gestures recommended by Barona López et al. (Sensors 2020,
        Appendix A): waveOut > Fist > Open > waveIn > Pinch > Tripod.

        Matching uses *substring* search (case-insensitive) so gesture keys
        like "cal_waveout", "waveout_sync", or "WAVEOUT" all resolve to the
        correct priority tier without requiring exact naming conventions.

        Selection logic:
          1. If a reference exists, return the highest-priority gesture that
             appears in *both* the current and reference pattern dictionaries.
          2. If no common priority gesture is found, fall back to "__combined__"
             (present in both by construction).
          3. If there is no reference, pick the highest-priority gesture in the
             current set.
          4. If no priority token matches at all, pick the gesture whose energy
             pattern has the highest confidence (sharpest spatial peak) — a
             sharper profile is a better sync signal regardless of its name.

        Returns:
            (current_pattern, reference_pattern_or_None, gesture_name_used)
        """
        priority = ["waveout", "fist", "open", "wavein", "pinch", "tripod"]

        def find_match(d: dict, token: str) -> Optional[str]:
            """Return the first key that contains *token* as a substring."""
            for key in d:
                if token in key.lower():
                    return key
            return None

        # 1. Best common gesture when a reference exists
        if reference_patterns:
            for p in priority:
                curr_key = find_match(patterns, p)
                ref_key  = find_match(reference_patterns, p)
                if curr_key and ref_key:
                    return patterns[curr_key], reference_patterns[ref_key], curr_key

            # Fallback: combined pseudo-key (always present)
            if "__combined__" in patterns and "__combined__" in reference_patterns:
                return (
                    patterns["__combined__"],
                    reference_patterns["__combined__"],
                    "Combined (fallback)",
                )

        # 2. No reference — best priority gesture in the current set
        for p in priority:
            key = find_match(patterns, p)
            if key:
                return patterns[key], None, key

        # 3. No priority match at all — pick sharpest non-rest pattern
        active = {
            k: v for k, v in patterns.items()
            if k != "__combined__" and "rest" not in k.lower()
        }
        if active:
            best_key = max(
                active,
                key=lambda k: self._compute_energy_confidence(active[k]),
            )
            return active[best_key], None, f"{best_key} (auto-selected)"

        # 4. Last resort
        return patterns.get("__combined__", np.zeros(self.num_channels)), None, "Combined"

    def _plot_energy_profile(
        self,
        sync_energy: np.ndarray,
        ref_sync_energy: Optional[np.ndarray],
        offset: int,
        confidence: float,
        plot_path: Path,
    ) -> None:
        """Save a lightweight diagnostic plot for sync-energy alignment."""
        import matplotlib.pyplot as plt

        plot_path = Path(plot_path)
        plot_path.parent.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(8, 4))
        x = np.arange(len(sync_energy))
        ax.plot(x, sync_energy, label="current", linewidth=2)

        if ref_sync_energy is not None and len(ref_sync_energy) == len(sync_energy):
            ax.plot(x, ref_sync_energy, label="reference", linewidth=1.5, alpha=0.8)

        ax.set_title(f"Calibration Energy Profile (offset={offset}, conf={confidence:.2%})")
        ax.set_xlabel("Channel")
        ax.set_ylabel("Energy")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)

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

        # Try to load existing monopolar reference calibration.
        self._load_reference_for_mode("monopolar")

    def _load_reference(self) -> None:
        """Load existing monopolar reference calibration (legacy compatibility)."""
        self._load_reference_for_mode("monopolar")

    @staticmethod
    def _normalize_signal_mode(signal_mode: Optional[str]) -> str:
        mode = str(signal_mode or "monopolar").strip().lower()
        return "bipolar" if mode == "bipolar" else "monopolar"

    @staticmethod
    def _sanitize_name_component(value: Any) -> str:
        safe = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "-", str(value or "")).strip().rstrip(". ")
        return safe or "unnamed"

    def _build_calibration_path(
        self,
        *,
        session_name: Optional[str],
        signal_mode: str,
        created_at: Optional[datetime] = None,
    ) -> Path:
        mode = self._normalize_signal_mode(signal_mode)
        if session_name:
            stem = f"calibration_{self._sanitize_name_component(session_name)}"
            if mode == "bipolar":
                stem = f"{stem}_{mode}"
        else:
            stamp = (created_at or datetime.now()).strftime("%Y%m%d_%H%M%S")
            stem = f"calibration_{stamp}"

        candidate = self.data_dir / f"{stem}.json"
        if not candidate.exists():
            return candidate

        idx = 2
        while True:
            collision = self.data_dir / f"{stem}_{idx}.json"
            if not collision.exists():
                return collision
            idx += 1

    def _reference_path(self, signal_mode: str) -> Path:
        mode = self._normalize_signal_mode(signal_mode)
        return self.data_dir / f"reference_calibration_{mode}.json"

    def _extract_session_signal_mode(self, session) -> str:
        custom = getattr(session.metadata, "custom_metadata", {}) or {}
        if isinstance(custom, dict):
            mode = custom.get("signal_mode")
            if isinstance(mode, str):
                return self._normalize_signal_mode(mode)
            if bool(custom.get("bipolar_mode", False)):
                return "bipolar"
        return "monopolar"

    def _load_reference_for_mode(self, signal_mode: str) -> None:
        """Load existing reference calibration for a specific signal mode."""
        mode = self._normalize_signal_mode(signal_mode)
        ref_path = self._reference_path(mode)

        if mode == "monopolar" and not ref_path.exists():
            legacy = self.data_dir / "reference_calibration.json"
            if legacy.exists():
                ref_path = legacy

        if not ref_path.exists():
            self.processor.set_reference_calibration(None)
            return

        try:
            ref = CalibrationResult.load(ref_path)
            ref_mode = self._normalize_signal_mode(ref.metadata.get("signal_mode", "monopolar"))
            if ref.num_channels != self.processor.num_channels or ref_mode != mode:
                self.processor.set_reference_calibration(None)
                return
            self.processor.set_reference_calibration(ref)
        except Exception as e:
            print(f"Warning: Could not load reference calibration ({mode}): {e}")
            self.processor.set_reference_calibration(None)

    def save_as_reference(self, result: CalibrationResult,
                          recompute_all: bool = False,
                          data_dir: Optional[Path] = None,
                          signal_mode: Optional[str] = None) -> None:
        """Save a calibration result as the reference.

        Args:
            result: The calibration result to use as the new reference.
            recompute_all: If True, re-detect rotations for all existing sessions
                           relative to this new reference.
            data_dir: Root data directory (needed when recompute_all=True).
        """
        mode = self._normalize_signal_mode(
            signal_mode or result.metadata.get("signal_mode", "monopolar")
        )

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
            metadata={**result.metadata, "is_reference": True, "signal_mode": mode},
        )
        ref_path = self._reference_path(mode)
        ref_result.save(ref_path)
        self.processor.set_reference_calibration(ref_result)

        if mode == "monopolar":
            legacy_path = self.data_dir / "reference_calibration.json"
            ref_result.save(legacy_path)

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
        save_as_reference: bool = False,
        signal_mode: str = "monopolar",
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
        mode = self._normalize_signal_mode(signal_mode)
        self._load_reference_for_mode(mode)

        result = self.processor.calibrate_from_data(
            calibration_data,
            device_name,
            self.processor.get_reference_calibration(),
            signal_mode=mode,
        )

        self._current_calibration = result

        # Save calibration
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result.save(self.data_dir / f"calibration_{timestamp}.json")

        if save_as_reference:
            self.save_as_reference(result, signal_mode=mode)

        return result

    def calibrate_from_session(self, session, plot_path: Optional[Path] = None) -> CalibrationResult:
        """
        Perform calibration directly from a normal recording session.

        **Preferred path (new sessions):** if the session contains one or more
        `calibration_sync` trials (the waveout gesture recorded at the very start
        of each session by the protocol), those trials are used exclusively as the
        synchronisation gesture.  This gives the calibrator the cleanest possible
        waveout signal — the paper's recommended sync gesture — without mixing in
        other gesture data.

        **Fallback path (older sessions):** if no calibration_sync trials exist,
        all valid gesture trials are used as before, and `_select_sync_pattern`
        picks the best available gesture.  This preserves full backward
        compatibility with sessions recorded before this feature was introduced.

        Args:
            session: A RecordingSession with annotated trials
            plot_path: Optional path to save a debug plot of the energy profile

        Returns:
            CalibrationResult with rotation offset and channel mapping
        """
        all_data = session.get_data()
        signal_mode = self._extract_session_signal_mode(session)
        session_name = getattr(session.metadata, "session_id", None)
        subject_id = getattr(session.metadata, "subject_id", None)
        self._load_reference_for_mode(signal_mode)

        # ── Try dedicated calibration-sync trials first ──────────────────────
        cal_trials = []
        if hasattr(session, "get_calibration_trials"):
            cal_trials = session.get_calibration_trials()

        if cal_trials:
            # Build a single-gesture dict so _select_sync_pattern
            # immediately picks this waveout recording.
            sync_chunks = []
            for trial in cal_trials:
                chunk = all_data[trial.start_sample:trial.end_sample]
                if chunk.shape[0] >= 10:
                    sync_chunks.append(chunk)

            if sync_chunks:
                calibration_data = {"waveout": sync_chunks}

                # Also include valid gesture trials so the combined energy
                # pattern is still informative for sessions that use it.
                valid_trials = session.get_valid_trials()
                for trial in valid_trials:
                    chunk = all_data[trial.start_sample:trial.end_sample]
                    if chunk.shape[0] < 10:
                        continue
                    name = trial.gesture_name
                    calibration_data.setdefault(name, []).append(chunk)

                result = self.processor.calibrate_from_data(
                    calibration_data,
                    device_name=session.metadata.device_name,
                    reference_result=self.processor.get_reference_calibration(),
                    signal_mode=signal_mode,
                    plot_path=plot_path,
                )
                self._current_calibration = result
                result.metadata.update({
                    "signal_mode": signal_mode,
                    "source_session_id": session_name,
                    "source_subject_id": subject_id,
                })
                result.save(
                    self._build_calibration_path(
                        session_name=session_name,
                        signal_mode=signal_mode,
                        created_at=result.created_at,
                    )
                )
                return result

        # ── Fallback: use all valid gesture trials (older sessions) ──────────
        gesture_trials: Dict[str, List[np.ndarray]] = {}
        valid_trials = session.get_valid_trials()
        if not valid_trials:
            raise ValueError("Session has no valid trials to calibrate from")

        for trial in valid_trials:
            trial_data = all_data[trial.start_sample:trial.end_sample]
            if trial_data.shape[0] < 10:
                continue
            gesture_trials.setdefault(trial.gesture_name, []).append(trial_data)

        if not gesture_trials:
            raise ValueError("No usable trial data found in session")

        result = self.processor.calibrate_from_data(
            gesture_trials,
            device_name=session.metadata.device_name,
            reference_result=self.processor.get_reference_calibration(),
            signal_mode=signal_mode,
            plot_path=plot_path,
        )
        self._current_calibration = result
        result.metadata.update({
            "signal_mode": signal_mode,
            "source_session_id": session_name,
            "source_subject_id": subject_id,
        })
        result.save(
            self._build_calibration_path(
                session_name=session_name,
                signal_mode=signal_mode,
                created_at=result.created_at,
            )
        )
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

                # Re-save metadata to the directory this session was loaded from.
                # This preserves legacy folder names (e.g. ':' timestamps) and
                # avoids creating duplicate sanitised folders.
                session_path = getattr(session, "_source_dir", None) or dm.get_session_path(subject, session_id)
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
