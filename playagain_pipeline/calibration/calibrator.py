"""
Calibration system for electrode orientation detection.

This module provides robust algorithms to detect the orientation of EMG
electrodes and create channel mappings that correct for different bracelet
positions.

Calibration can be performed directly from normal recording sessions —
no separate calibration recording is required. The system extracts
per-gesture activation patterns from the annotated trial data and uses
circular cross-correlation against a reference to find the rotation offset.

Improvements:
- Works directly from normal recording sessions (no separate calibration step)
- Bandpass filtering (20-450 Hz) for cleaner EMG signals
- Multi-feature spatial patterns (RMS + MAV + spectral power)
- Multiple trial repetitions averaged with outlier rejection
- Confidence scoring with margin-based validation
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple, Union
from pathlib import Path
import json
import numpy as np
from scipy import signal as scipy_signal
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
    Processes calibration data to determine electrode orientation.

    Uses robust multi-feature activation patterns from specific gestures
    to identify which muscles are under which electrodes, allowing
    correction for different bracelet orientations.

    Features:
    - Bandpass filtering to remove noise and motion artifacts
    - Multi-feature patterns (RMS + MAV + spectral) for richer representation
    - Multiple repetitions with outlier rejection for robust averaging
    - Confidence scoring per gesture and overall
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

    def compute_activation_pattern(
        self,
        data: np.ndarray,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Compute a robust spatial activation pattern from EMG trial data.

        Uses multiple complementary features to build a rich spatial fingerprint
        that is robust to amplitude variation but preserves channel-to-channel
        differences critical for rotation detection.

        Args:
            data: EMG data for a single gesture (samples, channels)
            normalize: Whether to normalize the pattern

        Returns:
            Activation pattern (6 * num_channels,)
            Features: [RMS | MAV | waveform_length | log_power | spectral_entropy | peak_frequency]
        """
        # Apply bandpass filtering
        filtered = self._bandpass_filter(data)

        # Trim 15% from start and end to remove transition artifacts
        trim = max(1, int(filtered.shape[0] * 0.15))
        trimmed = filtered[trim:-trim] if filtered.shape[0] > 2 * trim + 20 else filtered

        # Feature 1: RMS activation per channel
        rms_pattern = np.sqrt(np.mean(trimmed ** 2, axis=0))

        # Feature 2: MAV (Mean Absolute Value) per channel
        mav_pattern = np.mean(np.abs(trimmed), axis=0)

        # Feature 3: Waveform Length per channel (captures signal complexity)
        wl_pattern = np.mean(np.abs(np.diff(trimmed, axis=0)), axis=0)

        # Feature 4: Log power — compresses dynamic range while preserving ratios
        log_power = np.log1p(rms_pattern ** 2)

        # Feature 5: Spectral entropy per channel — captures frequency distribution
        # shape. Different muscles have different frequency characteristics, making
        # this a strong discriminator for rotation detection.
        spectral_entropy = np.zeros(self.num_channels)
        for ch in range(min(trimmed.shape[1], self.num_channels)):
            if trimmed.shape[0] >= 32:
                freqs = np.abs(np.fft.rfft(trimmed[:, ch]))
                freqs = freqs[1:]  # Remove DC
                psd = freqs ** 2
                psd_norm = psd / (np.sum(psd) + 1e-12)
                psd_norm = psd_norm[psd_norm > 1e-12]
                spectral_entropy[ch] = -np.sum(psd_norm * np.log2(psd_norm + 1e-12))
            else:
                spectral_entropy[ch] = 0.0

        # Feature 6: Peak frequency per channel (normalized to [0, 1])
        peak_freq = np.zeros(self.num_channels)
        for ch in range(min(trimmed.shape[1], self.num_channels)):
            if trimmed.shape[0] >= 32:
                freqs = np.abs(np.fft.rfft(trimmed[:, ch]))
                freqs = freqs[1:]  # Remove DC
                if len(freqs) > 0:
                    peak_freq[ch] = np.argmax(freqs) / (len(freqs) + 1e-12)

        # Combine features into a single pattern vector
        pattern = np.concatenate([rms_pattern, mav_pattern, wl_pattern,
                                  log_power, spectral_entropy, peak_freq])

        if normalize and np.max(pattern) > 0:
            # Hybrid normalization per feature group:
            # 1. Rank-based normalization (robust to amplitude differences)
            # 2. Z-score normalization (preserves relative magnitude info)
            # Final pattern = 0.7 * rank + 0.3 * zscore (both scaled to [0,1])
            n = self.num_channels
            for start in range(0, len(pattern), n):
                group = pattern[start:start + n]
                # Rank-based: argsort of argsort gives ranks
                ranks = np.argsort(np.argsort(group)).astype(float)
                if n > 1:
                    ranks /= (n - 1)

                # Z-score scaled to [0, 1] via min-max
                std = np.std(group)
                if std > 1e-12:
                    zscore = (group - np.mean(group)) / std
                    zmin, zmax = zscore.min(), zscore.max()
                    if zmax - zmin > 1e-12:
                        zscore_01 = (zscore - zmin) / (zmax - zmin)
                    else:
                        zscore_01 = np.full_like(zscore, 0.5)
                else:
                    zscore_01 = np.full_like(group, 0.5)

                pattern[start:start + n] = 0.7 * ranks + 0.3 * zscore_01

        return pattern

    def average_patterns_with_outlier_rejection(
        self,
        patterns: List[np.ndarray],
        threshold: float = 1.5
    ) -> np.ndarray:
        """
        Average multiple pattern repetitions with outlier rejection.

        Uses median-based outlier detection: discard any repetition whose
        correlation with the median is below (Q1 - threshold * IQR).

        Args:
            patterns: List of pattern arrays from multiple repetitions
            threshold: IQR multiplier for outlier detection

        Returns:
            Averaged pattern with outliers removed
        """
        if len(patterns) <= 1:
            return patterns[0] if patterns else np.zeros(1)

        patterns_arr = np.array(patterns)

        if len(patterns) <= 2:
            return np.mean(patterns_arr, axis=0)

        # Compute median pattern as reference
        median_pattern = np.median(patterns_arr, axis=0)

        # Compute correlation of each repetition with the median
        correlations = []
        for p in patterns:
            if np.std(p) > 0 and np.std(median_pattern) > 0:
                corr, _ = pearsonr(p, median_pattern)
                correlations.append(corr)
            else:
                correlations.append(0.0)

        correlations = np.array(correlations)

        # Outlier detection using IQR
        q1 = np.percentile(correlations, 25)
        q3 = np.percentile(correlations, 75)
        iqr = q3 - q1
        lower_bound = q1 - threshold * iqr

        inlier_mask = correlations >= lower_bound
        if not inlier_mask.any():
            inlier_mask = np.ones(len(patterns), dtype=bool)

        return np.mean(patterns_arr[inlier_mask], axis=0)

    def _check_reference_compatible(
        self,
        current_patterns: Dict[str, np.ndarray],
        reference_patterns: Dict[str, np.ndarray]
    ) -> Tuple[bool, str]:
        """
        Check if reference calibration is compatible with current patterns.

        Returns:
            (is_compatible, reason) — reason describes why not compatible
        """
        common = set(current_patterns.keys()) & set(reference_patterns.keys())
        active = {g for g in common if "rest" not in g.lower()}
        usable = active if active else common

        if not usable:
            return False, (f"No common gestures. "
                          f"Current: {sorted(current_patterns.keys())}. "
                          f"Reference: {sorted(reference_patterns.keys())}.")

        # Check pattern length compatibility
        for g in usable:
            if len(current_patterns[g]) != len(reference_patterns[g]):
                return False, (f"Pattern length mismatch for '{g}': "
                              f"current={len(current_patterns[g])}, "
                              f"reference={len(reference_patterns[g])}. "
                              f"Reference may be from an older calibration format.")
        return True, ""

    def _fft_circular_cross_correlation(
        self,
        current: np.ndarray,
        reference: np.ndarray
    ) -> np.ndarray:
        """
        Compute circular cross-correlation using FFT for all rotation offsets.

        This is O(N log N) instead of O(N^2) and numerically more stable
        than brute-force Pearson at each offset.

        Args:
            current: 1D pattern array (one feature group, num_channels long)
            reference: 1D pattern array (same length)

        Returns:
            Cross-correlation scores for each offset (normalized to [-1, 1])
        """
        # Zero-mean normalization for correlation
        c = current - np.mean(current)
        r = reference - np.mean(reference)

        norm = np.sqrt(np.sum(c ** 2) * np.sum(r ** 2))
        if norm < 1e-12:
            return np.zeros(len(current))

        # FFT-based circular cross-correlation: corr = IFFT(FFT(r)* × FFT(c))
        cross_corr = np.real(np.fft.ifft(np.conj(np.fft.fft(r)) * np.fft.fft(c)))
        return cross_corr / norm

    def _compute_gesture_discriminability(
        self,
        pattern: np.ndarray
    ) -> float:
        """
        Score how spatially discriminative a gesture's activation pattern is.

        Gestures with uniform activation across channels (like rest) are poor
        for rotation detection. Gestures with strong spatial peaks are better.

        Uses the "peakiness" of the rank distribution: if only a few channels
        dominate (high variance in the rank-normalised pattern), the gesture
        is a good landmark for rotation.

        Returns a weight in [0, 1] — higher means more useful for rotation.
        """
        n = self.num_channels
        n_features = len(pattern) // n
        if n_features < 1:
            n_features = 1

        scores = []
        for fg in range(n_features):
            group = pattern[fg * n:(fg + 1) * n]
            if len(group) == 0:
                continue

            # For rank-normalised patterns (values in [0, 1]):
            # A perfectly uniform pattern has std ≈ 0.29 (uniform on [0,1])
            # A peaked pattern (one dominant channel) has higher std
            # A flat/rest pattern where all channels are similar has lower std
            #
            # But since all rank patterns have the same std (they're permutations),
            # we instead look at the top-k concentration: what fraction of the
            # total "activation" is in the top 25% of channels?
            sorted_vals = np.sort(group)[::-1]
            top_k = max(1, n // 4)
            top_sum = np.sum(sorted_vals[:top_k])
            total_sum = np.sum(sorted_vals) + 1e-12
            concentration = top_sum / total_sum

            # For a uniform distribution, top 25% has 25% of the total → ratio ≈ 0.25
            # For a highly peaked pattern, top 25% might have 60-80% → ratio ≈ 0.6-0.8
            # Map [0.25, 0.65] → [0, 1]
            score = np.clip((concentration - 0.25) / 0.4, 0.0, 1.0)
            scores.append(score)

        if not scores:
            return 0.0

        return float(np.mean(scores))

    def find_rotation_offset(
        self,
        current_patterns: Dict[str, np.ndarray],
        reference_patterns: Dict[str, np.ndarray]
    ) -> Tuple[int, float, Dict[str, float]]:
        """
        Find the rotation offset that best aligns current patterns with reference.

        Uses FFT-based circular cross-correlation on rank-normalised multi-feature
        patterns. Each gesture is weighted by spatial discriminability. The final
        offset is chosen by weighted accumulation across all gestures and feature
        groups.

        Confidence is based on:
        1. The margin ratio between the best and second-best offset scores
        2. Per-gesture agreement (how many gestures independently pick the same offset)
        3. Mean Spearman correlation at the chosen offset

        Args:
            current_patterns: Current activation patterns by gesture
            reference_patterns: Reference patterns to match against

        Returns:
            Tuple of (rotation_offset, overall_confidence, per_gesture_confidence)
        """
        compatible, reason = self._check_reference_compatible(
            current_patterns, reference_patterns)
        if not compatible:
            raise ValueError(f"Incompatible reference: {reason}")

        common_gestures = set(current_patterns.keys()) & set(reference_patterns.keys())
        active_gestures = {g for g in common_gestures if "rest" not in g.lower()}
        if not active_gestures:
            active_gestures = common_gestures
        if not active_gestures:
            raise ValueError("No common gestures between current and reference")

        first_pattern = next(iter(current_patterns.values()))
        n_features_per_channel = len(first_pattern) // self.num_channels
        if n_features_per_channel < 1:
            n_features_per_channel = 1

        n = self.num_channels

        # Accumulate weighted cross-correlation scores
        offset_scores = np.zeros(n)
        total_weight = 0.0

        # Per-gesture best offsets for consistency checking
        per_gesture_best = {}  # gesture -> (best_offset, weight)

        for gesture in active_gestures:
            current = current_patterns[gesture]
            reference = reference_patterns[gesture]
            if len(current) != len(reference):
                continue

            weight = self._compute_gesture_discriminability(reference)
            weight = max(weight, 0.05)

            gesture_scores = np.zeros(n)
            gesture_weight = 0.0

            for fg in range(n_features_per_channel):
                s = fg * n
                e = s + n
                ccorr = self._fft_circular_cross_correlation(
                    current[s:e], reference[s:e]
                )
                gesture_scores += ccorr * weight
                gesture_weight += weight

            if gesture_weight > 0:
                gesture_scores /= gesture_weight

            offset_scores += gesture_scores * gesture_weight
            total_weight += gesture_weight

            per_gesture_best[gesture] = (
                int(np.argmax(gesture_scores)),
                gesture_weight
            )

        if total_weight > 0:
            offset_scores /= total_weight

        best_offset = int(np.argmax(offset_scores))

        # ── Confidence scoring ──────────────────────────────────────────

        # 1. Margin ratio: how much better is the best vs second-best?
        sorted_scores = np.sort(offset_scores)[::-1]
        if len(sorted_scores) > 1 and sorted_scores[0] > 1e-12:
            margin_ratio = sorted_scores[0] / max(sorted_scores[1], 1e-12)
            # Tighter range so moderate margins score well:
            # ratio 1.0 → 0%, 1.03 → 30%, 1.06 → 60%, 1.10+ → 100%
            margin_score = float(np.clip((margin_ratio - 1.0) / 0.10, 0.0, 1.0))
        else:
            margin_score = 0.5

        # 1b. Peak sharpness bonus: how concentrated is the score around the best offset?
        #     If the best offset dominates the score distribution, boost confidence.
        if len(offset_scores) > 1:
            scores_shifted = offset_scores - np.min(offset_scores)
            total_score = np.sum(scores_shifted) + 1e-12
            peak_fraction = float(scores_shifted[best_offset] / total_score)
            # For n channels, uniform → peak_fraction ≈ 1/n; peaked → close to 1.0
            expected_uniform = 1.0 / max(len(offset_scores), 1)
            sharpness = float(np.clip((peak_fraction - expected_uniform) / (1.0 - expected_uniform), 0.0, 1.0))
        else:
            sharpness = 1.0

        # Blend margin and sharpness: sharpness is more reliable
        margin_score = 0.4 * margin_score + 0.6 * sharpness

        # 2. Per-gesture agreement: weighted fraction of gestures picking
        #    the same offset (±1), weighted by gesture discriminability
        if per_gesture_best:
            weighted_agree = 0.0
            weighted_near_agree = 0.0
            total_w = 0.0
            for g, (g_offset, g_weight) in per_gesture_best.items():
                total_w += g_weight
                if g_offset == best_offset:
                    weighted_agree += g_weight
                    weighted_near_agree += g_weight
                else:
                    circ_dist = min(abs(g_offset - best_offset),
                                    n - abs(g_offset - best_offset))
                    if circ_dist <= 1:
                        weighted_near_agree += g_weight

            if total_w > 0:
                exact_agreement = weighted_agree / total_w
                near_agreement = weighted_near_agree / total_w
            else:
                exact_agreement = 0.0
                near_agreement = 0.0
            agreement_score = 0.7 * exact_agreement + 0.3 * near_agreement
        else:
            agreement_score = 0.0

        # 3. Mean per-feature-group Pearson correlation at best offset
        #    (more spatially sensitive than whole-vector Spearman)
        per_gesture_confidence = {}
        pearson_scores = []

        for gesture in common_gestures:
            current = current_patterns[gesture]
            reference = reference_patterns[gesture]
            if len(current) != len(reference):
                per_gesture_confidence[gesture] = 0.0
                continue

            # Roll each feature group and compute per-group Pearson
            group_corrs = []
            for fg in range(n_features_per_channel):
                s = fg * n
                e = s + n
                rolled = np.roll(current[s:e], best_offset)
                ref_group = reference[s:e]
                if np.std(rolled) > 0 and np.std(ref_group) > 0:
                    corr, _ = pearsonr(rolled, ref_group)
                    if not np.isnan(corr):
                        group_corrs.append(corr)

            gesture_corr = float(np.mean(group_corrs)) if group_corrs else 0.0
            per_gesture_confidence[gesture] = gesture_corr
            if gesture in active_gestures:
                pearson_scores.append(gesture_corr)

        mean_corr = float(np.mean(pearson_scores)) if pearson_scores else 0.0
        # Lower floor (0.1) and gentler slope so moderate correlations still score reasonably:
        # corr 0.1 → 0%, 0.4 → ~38%, 0.6 → ~63%, 0.8 → ~88%, 1.0 → 100%
        corr_score = float(np.clip((mean_corr - 0.1) / 0.8, 0.0, 1.0))

        # Final confidence: correlation is the strongest indicator, give it most weight
        confidence = 0.20 * margin_score + 0.30 * agreement_score + 0.50 * corr_score
        confidence = max(0.0, min(1.0, confidence))

        # Store sub-scores for diagnostics
        per_gesture_confidence["__margin_score__"] = round(margin_score, 4)
        per_gesture_confidence["__agreement_score__"] = round(agreement_score, 4)
        per_gesture_confidence["__correlation_score__"] = round(corr_score, 4)
        per_gesture_confidence["__sharpness__"] = round(sharpness, 4)

        return best_offset, confidence, per_gesture_confidence

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
        calibration_data: Dict[str, Union[np.ndarray, List[np.ndarray]]],
        device_name: str = "unknown",
        reference_result: Optional[CalibrationResult] = None
    ) -> CalibrationResult:
        """
        Perform calibration from recorded gesture data.

        Args:
            calibration_data: Dictionary mapping gesture names to EMG data arrays.
                Values can be a single array or a list of arrays (multiple trials)
                which will be averaged with outlier rejection.
            device_name: Name of the device
            reference_result: Optional reference calibration to match against

        Returns:
            CalibrationResult with channel mapping
        """
        # Compute activation patterns for each gesture
        current_patterns = {}
        per_gesture_confidence = {}

        for gesture_name, data in calibration_data.items():
            if isinstance(data, list):
                # Multiple trials — compute pattern per trial, average with rejection
                rep_patterns = [self.compute_activation_pattern(d) for d in data]
                current_patterns[gesture_name] = (
                    self.average_patterns_with_outlier_rejection(rep_patterns)
                )
            else:
                current_patterns[gesture_name] = self.compute_activation_pattern(data)

        incompatible_reason = None
        if reference_result is not None:
            # Check compatibility first
            compatible, reason = self._check_reference_compatible(
                current_patterns, reference_result.reference_patterns
            )
            if compatible:
                offset, confidence, per_gesture_confidence = self.find_rotation_offset(
                    current_patterns,
                    reference_result.reference_patterns
                )
            else:
                # Reference exists but is incompatible — treat as no reference
                incompatible_reason = reason
                offset = 0
                confidence = 1.0
        else:
            # No reference — use identity mapping
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
            reference_patterns=current_patterns,
            metadata={
                "per_gesture_confidence": per_gesture_confidence,
                "num_gestures": len(calibration_data),
                "has_reference": reference_result is not None,
                "reference_incompatible": incompatible_reason,
                "confidence_breakdown": {
                    "margin_score": per_gesture_confidence.get("__margin_score__", None),
                    "agreement_score": per_gesture_confidence.get("__agreement_score__", None),
                    "correlation_score": per_gesture_confidence.get("__correlation_score__", None),
                } if reference_result is not None else {},
            }
        )

        return result

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

    def calibrate_from_session(self, session) -> CalibrationResult:
        """
        Perform calibration directly from a normal recording session.

        Extracts per-gesture trial data from the session and uses it to
        compute calibration patterns. Multiple trials of the same gesture
        are averaged with outlier rejection for robustness.

        This eliminates the need for a separate calibration recording.

        Args:
            session: A RecordingSession with annotated trials

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
            reference_result=self.processor.get_reference_calibration()
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

    def detect_session_rotation(self, session, save_to_metadata: bool = True) -> Optional[CalibrationResult]:
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

        try:
            result = self.calibrate_from_session(session)
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
                                force: bool = False) -> Dict[str, Dict[str, Any]]:
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
    # If not (e.g. after a code update changed the feature count),
    # discard it so the first processed session becomes the new reference.
    if calibrator.has_reference:
        ref = calibrator.processor.get_reference_calibration()
        expected_len = 6 * num_channels  # Current format: RMS + MAV + WL + log_power + spectral_entropy + peak_freq
        sample_pattern = next(iter(ref.reference_patterns.values()), None)
        if sample_pattern is not None and len(sample_pattern) != expected_len:
            print(f"[Backfill] Reference pattern format outdated "
                  f"({len(sample_pattern)} vs expected {expected_len}). "
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
            cal_result = calibrator.detect_session_rotation(session, save_to_metadata=True)

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
