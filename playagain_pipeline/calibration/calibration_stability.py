"""
calibration/calibration_stability.py
────────────────────────────────────
Compute a *real* confidence number for a rotation calibration —
"how often do my trials agree on the same offset?" — instead of the
legacy peak-prominence metric which is dominated by per-subject muscle
anatomy.

The legacy ``CalibrationProcessor._compute_xcorr_confidence`` returns
the z-score of the xcorr peak, scaled into [0, 1]. That's a measure
of how angularly sharp the user's sync-gesture activation is, not a
measure of whether the offset is right. Two consequences:

* Subjects with broad activation patterns get low "confidence" even
  when every session of theirs lands on the same offset within ±0.
* Subjects with sharp activation get high "confidence" even when the
  detected peak could be on the wrong channel due to a single
  dominant electrode.

The stability metric in this module fixes both:

    stability = fraction of trials whose individually-detected offset
                matches the modal offset across all trials

It is a property of the **data**, not of the user's anatomy. A
session with 3 trials all voting for lag = 3 gets stability = 1.0;
a session with [3, 4, 3] gets stability = 0.67; a session with
[2, 5, 7] gets stability = 0.33.

The peak-prominence z-score is still computed and returned as a
diagnostic — it remains useful for spotting completely flat xcorr
spectra (broken electrodes, dead sync gesture).

This module is deliberately self-contained: it doesn't import the
calibrator class. The xcorr / fold / smooth helpers mirror
``CalibrationProcessor`` so retroactive recompute and live computation
agree to the bit.

Public API
──────────
* ``StabilityResult``                  — dataclass with all numbers
* ``compute_stability_metrics``        — main entry point
* ``aggregate_energy_per_trial``       — exported for reuse / tests
* ``per_trial_offsets``                — exported for reuse / tests
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class StabilityResult:
    """
    Everything we know about a single session's rotation calibration.

    Attributes
    ----------
    offset
        The offset the calibrator would report (mode of per-trial
        offsets; equal to the offset of the averaged energy in well-
        behaved sessions).
    stability
        Fraction of trials whose own offset matches ``offset``. The
        primary "confidence" replacement, in [0, 1].
    peak_prominence
        Legacy metric: z-score of the xcorr peak of the *averaged*
        energy profile, mapped into [0, 1] by clipping at z = 5.
        Useful as a sanity check, but not a confidence number.
    top2_ratio
        Ratio of the winning xcorr lag's value to the runner-up's.
        > 1 means the winner is unambiguous; ≈ 1 means the choice
        could have gone either way. Not bounded to [0, 1].
    n_trials_used
        How many sync-gesture trials contributed.
    per_trial_offsets
        The offset detected on each trial individually.
    bootstrap_offsets
        Optional bootstrap distribution of offsets if the trial count
        was large enough; empty list otherwise.
    """
    offset:           int
    stability:        float
    peak_prominence:  float
    top2_ratio:       float
    n_trials_used:    int
    per_trial_offsets:  List[int]      = field(default_factory=list)
    bootstrap_offsets:  List[int]      = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Trial-level pipeline (mirrors CalibrationProcessor)
# ---------------------------------------------------------------------------

def _bandpass_filter(
    data: np.ndarray, sampling_rate: float,
    low: float = 20.0, high: float = 450.0,
) -> np.ndarray:
    """Mirror of ``CalibrationProcessor._bandpass_filter``.

    A 4th-order Butterworth band-pass via SciPy if available; otherwise
    a no-op (the raw signal is good enough for an angular energy
    profile, just slightly noisier).
    """
    try:
        from scipy.signal import butter, sosfiltfilt
        ny = sampling_rate * 0.5
        if ny <= 0:
            return data
        wn = (max(low / ny, 1e-3), min(high / ny, 0.9999))
        if wn[0] >= wn[1]:
            return data
        sos = butter(4, wn, btype="band", output="sos")
        return sosfiltfilt(sos, data, axis=0)
    except Exception:
        return data


def _channel_energy(data: np.ndarray, num_channels: int) -> np.ndarray:
    """Per-channel energy with the Barona-Lopez formula.

    E_ch = Σ_i |x_i·|x_i| − x_{i-1}·|x_{i-1}||

    Matches ``CalibrationProcessor.compute_channel_energy`` modulo the
    band-pass filtering, which the caller is responsible for.
    """
    trim = max(1, int(data.shape[0] * 0.15))
    if data.shape[0] > 2 * trim + 20:
        data = data[trim:-trim]
    n_ch = min(data.shape[1], num_channels)
    energy = np.zeros(num_channels, dtype=np.float64)
    for ch in range(n_ch):
        x    = data[:, ch].astype(np.float64)
        term = x * np.abs(x)
        energy[ch] = float(np.sum(np.abs(np.diff(term))))
    return energy


def _normalize_l2(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v.copy()


def _smooth_circular(v: np.ndarray, sigma: float = 0.8) -> np.ndarray:
    n = v.size
    if n < 3 or sigma <= 0:
        return v.copy()
    pad = min(int(4 * sigma) + 1, n)
    ext = np.concatenate([v[-pad:], v, v[:pad]])
    try:
        from scipy.ndimage import gaussian_filter1d
        sm = gaussian_filter1d(ext, sigma=sigma)
    except Exception:
        # Lightweight Gaussian if scipy is unavailable
        kernel_half = max(1, int(3 * sigma))
        ks = np.arange(-kernel_half, kernel_half + 1)
        kernel = np.exp(-(ks ** 2) / (2.0 * sigma ** 2))
        kernel = kernel / kernel.sum()
        sm = np.convolve(ext, kernel, mode="same")
    return sm[pad: pad + n]


def _angular_profile(energy: np.ndarray) -> np.ndarray:
    """Fold a 32-channel pattern into 16 angular sectors (see calibrator)."""
    if energy.size == 32:
        return energy[:16] + energy[16:]
    return energy.copy()


def _xcorr_circular(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Real circular cross-correlation via FFT."""
    return np.real(np.fft.ifft(np.fft.fft(a) * np.conj(np.fft.fft(b))))


def _offset_from_energy(
    cur_energy: np.ndarray, ref_energy: np.ndarray,
) -> Tuple[int, np.ndarray]:
    """Run one xcorr-based offset detection.

    Returns ``(offset, xcorr_vector)`` so the caller can derive
    additional diagnostics (peak prominence, top-2 ratio).
    """
    cur = _smooth_circular(_normalize_l2(_angular_profile(cur_energy)))
    ref = _smooth_circular(_normalize_l2(_angular_profile(ref_energy)))
    xc  = _xcorr_circular(cur, ref)
    return int(np.argmax(xc)), xc


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def _peak_prominence_z(xc: np.ndarray) -> float:
    """Legacy z-score / 5 metric, clipped to [0, 1].

    Reproduces ``CalibrationProcessor._compute_xcorr_confidence`` exactly so
    sessions recomputed by this module match the legacy field bit-for-bit
    when the trial-set is identical.
    """
    if xc.size < 2:
        return 0.0
    std = float(np.std(xc))
    if std < 1e-12:
        return 0.0
    z = (float(np.max(xc)) - float(np.mean(xc))) / std
    return float(np.clip(z / 5.0, 0.0, 1.0))


def _top2_ratio(xc: np.ndarray) -> float:
    """Ratio peak / runner-up. Unambiguous detection → much greater than 1."""
    if xc.size < 2:
        return 1.0
    s = np.sort(xc)[::-1]
    second = float(s[1])
    if abs(second) < 1e-12:
        return float("inf")
    return float(s[0] / second)


# ---------------------------------------------------------------------------
# Helpers exported for reuse
# ---------------------------------------------------------------------------

def aggregate_energy_per_trial(
    trials: Sequence[np.ndarray],
    *,
    num_channels: int,
    sampling_rate: float,
) -> List[np.ndarray]:
    """Compute one energy vector per trial.

    ``trials`` is a list of ``(n_samples, n_channels)`` arrays — the raw
    EMG chunks for one gesture. Empty trials are silently skipped.
    """
    out: List[np.ndarray] = []
    for t in trials:
        if t is None or t.size == 0 or t.shape[0] < 10:
            continue
        filtered = _bandpass_filter(t, sampling_rate)
        out.append(_channel_energy(filtered, num_channels))
    return out


def per_trial_offsets(
    trial_energies: Sequence[np.ndarray],
    ref_energy: np.ndarray,
) -> List[int]:
    """One detected offset per trial. Useful for plotting jitter."""
    return [_offset_from_energy(e, ref_energy)[0] for e in trial_energies]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compute_stability_metrics(
    trials: Sequence[np.ndarray],
    ref_energy: np.ndarray,
    *,
    num_channels: int,
    sampling_rate: float,
    bootstrap_n: int = 0,
    rng: Optional[np.random.Generator] = None,
) -> StabilityResult:
    """
    Compute the full bundle of calibration-stability statistics for one
    session's sync-gesture trials.

    Parameters
    ----------
    trials
        List of ``(n_samples, n_channels)`` arrays — the raw EMG chunks
        for the sync gesture in one session. The same trials the
        calibrator already averages.
    ref_energy
        Per-channel energy of the *reference* sync gesture, shape
        ``(num_channels,)``. Read from the saved reference calibration's
        ``reference_patterns``.
    num_channels
        Number of channels on the device (32 for the bracelet — folded
        into 16 angular sectors before xcorr, just like the calibrator).
    sampling_rate
        EMG sampling rate in Hz (used only for the optional band-pass).
    bootstrap_n
        If > 0 and there are at least 3 trials, additionally compute a
        bootstrap distribution of offsets for richer plots. Pure
        diagnostic; the primary ``stability`` value never depends on it.
    rng
        Optional NumPy generator for reproducible bootstrap.

    Returns
    -------
    StabilityResult
    """
    trial_energies = aggregate_energy_per_trial(
        trials, num_channels=num_channels, sampling_rate=sampling_rate,
    )

    n_used = len(trial_energies)
    if n_used == 0 or ref_energy is None or ref_energy.size == 0:
        return StabilityResult(
            offset=0, stability=0.0, peak_prominence=0.0,
            top2_ratio=1.0, n_trials_used=n_used,
        )

    # Per-trial offsets — the basis of stability.
    p_off = [_offset_from_energy(e, ref_energy)[0] for e in trial_energies]
    vals, counts = np.unique(p_off, return_counts=True)
    mode_idx = int(np.argmax(counts))
    offset   = int(vals[mode_idx])
    stability = float(counts[mode_idx]) / float(n_used)

    # Peak prominence + top-2 from the *averaged* energy — matches the
    # calibrator's report-time numbers exactly when n_used == 1.
    avg_energy = np.mean(np.stack(trial_energies, axis=0), axis=0)
    _, avg_xc  = _offset_from_energy(avg_energy, ref_energy)
    prominence = _peak_prominence_z(avg_xc)
    top2       = _top2_ratio(avg_xc)

    # Optional bootstrap, gated on enough trials to be meaningful.
    boot_offsets: List[int] = []
    if bootstrap_n > 0 and n_used >= 3:
        rng = rng if rng is not None else np.random.default_rng(0)
        stacked = np.stack(trial_energies, axis=0)
        for _ in range(int(bootstrap_n)):
            idx = rng.integers(0, n_used, size=n_used)
            mean_e = stacked[idx].mean(axis=0)
            boot_offsets.append(_offset_from_energy(mean_e, ref_energy)[0])

    return StabilityResult(
        offset=offset,
        stability=stability,
        peak_prominence=prominence,
        top2_ratio=top2,
        n_trials_used=n_used,
        per_trial_offsets=list(p_off),
        bootstrap_offsets=boot_offsets,
    )
