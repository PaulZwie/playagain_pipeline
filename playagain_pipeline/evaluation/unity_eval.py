"""
evaluation/unity_eval.py
────────────────────────
Evaluate Unity recordings under the RMS-threshold model.

Two source formats:

1. **Raw Unity CSV** — has explicit ``RMS`` and ``GroundTruthActive``
   columns. We use those directly.
2. **Converted Unity session** — has raw EMG channels and trials with
   ``rest`` / ``fist`` labels. We compute RMS in a sliding window and
   build the binary label from the trials.

For both we:
- sweep RMS thresholds
- pick an optimal threshold (F1-max by default)
- report binary metrics at the chosen threshold (accuracy, precision,
  recall/sensitivity, specificity, F1) plus AUROC across the sweep
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .core import (
    EvaluationMode, EvaluationResult, RecordingDescriptor, RecordingKind,
)
from .loaders import load_session_data
from .metrics import (
    auroc_binary, fill_binary_metrics,
    pick_optimal_threshold, threshold_sweep,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

@dataclass
class UnityEvalSettings:
    """Tunables for the Unity RMS-threshold evaluator."""
    # Active class label in the converted-session format. The default
    # gesture set assigns ``fist=1``; tweak if the user has imported a
    # multi-gesture Unity recording with a different "active" label.
    active_label: int = 1

    # Window length used to recompute RMS from raw EMG when the source
    # is a converted session (samples are inferred from sampling_rate).
    rms_window_ms:   int = 100
    rms_stride_ms:   int = 25

    # Threshold sweep resolution.
    n_thresholds: int = 200

    # Objective for picking the optimal threshold:  "f1" | "youden" | "accuracy"
    objective: str = "f1"

    # Optional override — if not None, force this threshold instead of
    # picking one. Useful for "what does threshold=X actually give us".
    fixed_threshold: Optional[float] = None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def evaluate_unity(
    recordings: List[RecordingDescriptor],
    settings: UnityEvalSettings,
) -> EvaluationResult:
    """Evaluate one or more Unity recordings as a single pooled binary task."""
    if not recordings:
        raise ValueError("evaluate_unity needs at least one recording")
    for r in recordings:
        if r.kind != RecordingKind.UNITY:
            raise ValueError(f"Expected UNITY descriptors, got {r.kind} ({r.label})")

    # Pool RMS + ground truth across all recordings
    all_rms:   List[np.ndarray] = []
    all_truth: List[np.ndarray] = []
    notes:     List[str]        = []

    for rec in recordings:
        try:
            rms, truth, src = _extract_rms_and_truth(rec, settings)
        except Exception as exc:
            log.exception("Could not load %s", rec.path)
            notes.append(f"⚠ Skipped {rec.label}: {exc}")
            continue
        if rms.size == 0 or truth.size == 0:
            notes.append(f"⚠ {rec.label}: empty after extraction")
            continue
        notes.append(f"• {rec.label}: {rms.size} samples ({src})")
        all_rms.append(rms)
        all_truth.append(truth)

    if not all_rms:
        raise RuntimeError("No usable Unity recordings — see notes for details.")

    rms_pooled   = np.concatenate(all_rms)
    truth_pooled = np.concatenate(all_truth)

    # Threshold sweep
    sweep = threshold_sweep(rms_pooled, truth_pooled, n_thresholds=settings.n_thresholds)

    # Pick a threshold (fixed override or argmax of objective)
    if settings.fixed_threshold is not None:
        chosen_t = float(settings.fixed_threshold)
        chosen_pt = min(sweep, key=lambda p: abs(p.threshold - chosen_t)) if sweep else None
    else:
        chosen_pt = pick_optimal_threshold(sweep, objective=settings.objective)
        chosen_t = chosen_pt.threshold if chosen_pt else float("nan")

    # Build the result
    result = EvaluationResult(
        title=_make_title(recordings),
        mode=EvaluationMode.RMS_THRESHOLD,
        kind=RecordingKind.UNITY,
        recordings=list(recordings),
        settings={
            "active_label":   settings.active_label,
            "rms_window_ms":  settings.rms_window_ms,
            "rms_stride_ms":  settings.rms_stride_ms,
            "n_thresholds":   settings.n_thresholds,
            "objective":      settings.objective,
            "fixed_threshold": settings.fixed_threshold,
        },
    )
    for line in notes:
        result.add_note(line)

    # Binary metrics at the chosen threshold
    if chosen_pt is not None:
        y_pred = (rms_pooled > chosen_pt.threshold).astype(np.int64)
        fill_binary_metrics(
            result, truth_pooled, y_pred,
            label_names=("rest", "active"),
        )
        result.chosen_threshold = float(chosen_pt.threshold)
    else:
        result.add_note("Threshold sweep returned no points.")

    result.threshold_sweep = sweep
    result.auroc = auroc_binary(rms_pooled, truth_pooled)

    # Tiny preview timeline (downsampled if huge) for the UI to plot
    result.timeline = _build_timeline(rms_pooled, truth_pooled, chosen_pt.threshold if chosen_pt else None)

    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_rms_and_truth(
    rec: RecordingDescriptor,
    settings: UnityEvalSettings,
) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Return (rms_per_sample, ground_truth_active, source_label).

    The two arrays are aligned: same length, one entry per sample
    (or per RMS window when computed from raw EMG).
    """
    fmt = rec.meta.get("format")
    if fmt == "csv" or rec.path.suffix.lower() == ".csv":
        return _from_raw_csv(Path(rec.path))
    elif fmt == "session" or rec.path.is_dir():
        return _from_converted_session(rec, settings)
    else:
        # Fallback: try CSV first, then session
        if rec.path.suffix.lower() == ".csv":
            return _from_raw_csv(Path(rec.path))
        return _from_converted_session(rec, settings)


def _from_raw_csv(csv_path: Path) -> Tuple[np.ndarray, np.ndarray, str]:
    """Pull RMS and GroundTruthActive directly from a raw Unity CSV."""
    import pandas as pd
    df = pd.read_csv(csv_path)
    if "RMS" not in df.columns or "GroundTruthActive" not in df.columns:
        raise ValueError(f"{csv_path.name} is not a Unity raw CSV (missing RMS / GroundTruthActive)")
    rms   = df["RMS"].to_numpy(dtype=np.float64)
    truth = (df["GroundTruthActive"].to_numpy(dtype=np.int64) > 0).astype(np.int64)
    return rms, truth, "logged RMS"


def _from_converted_session(
    rec: RecordingDescriptor,
    settings: UnityEvalSettings,
) -> Tuple[np.ndarray, np.ndarray, str]:
    """Compute RMS from raw EMG and build ground truth from trials."""
    sess_dir = Path(rec.path)
    # Build a SESSION-flavoured descriptor so we can reuse load_session_data
    sess_desc = RecordingDescriptor(
        kind=RecordingKind.SESSION,
        subject_id=rec.subject_id,
        session_id=rec.session_id,
        path=sess_dir,
        label=rec.label,
        meta=rec.meta,
    )
    data, meta, trials = load_session_data(sess_desc)

    fs = int(meta.get("sampling_rate") or 2000)
    win = max(1, int(round(settings.rms_window_ms * fs / 1000.0)))
    stride = max(1, int(round(settings.rms_stride_ms * fs / 1000.0)))

    # Per-sample sample-level ground truth from trials.
    n = data.shape[0]
    truth_sample = np.zeros(n, dtype=np.int64)
    for t in trials:
        if not t.get("is_valid", True):
            continue
        if int(t.get("gesture_label", -1)) == int(settings.active_label):
            s, e = int(t.get("start_sample", 0)), int(t.get("end_sample", 0))
            s = max(0, min(s, n)); e = max(0, min(e, n))
            truth_sample[s:e] = 1

    # Sliding RMS over a single channel? No — we want a global activation
    # signal, so use the mean across channels of the per-window RMS.
    rms_win, idx = _windowed_rms_mean(data, win, stride)

    # Truth per window: majority active over the window
    truth_win = np.zeros(rms_win.size, dtype=np.int64)
    for k, start in enumerate(idx):
        end = min(start + win, n)
        if (truth_sample[start:end].sum() * 2) >= (end - start):
            truth_win[k] = 1

    return rms_win, truth_win, f"computed RMS  ({settings.rms_window_ms} ms window)"


def _windowed_rms_mean(data: np.ndarray, win: int, stride: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the average per-window RMS across all channels.

    Returns (rms_values, window_start_indices).
    """
    if data.ndim == 1:
        data = data[:, None]
    n = data.shape[0]
    if n < win:
        return np.array([]), np.array([], dtype=np.int64)

    # Build window-start indices
    starts = np.arange(0, n - win + 1, stride, dtype=np.int64)
    if starts.size == 0:
        return np.array([]), np.array([], dtype=np.int64)

    # RMS per channel per window, then mean across channels.
    out = np.empty(starts.size, dtype=np.float64)
    sq = data.astype(np.float64) ** 2
    for k, s in enumerate(starts):
        e = s + win
        # mean(sq) over (window x channel) → scalar; sqrt → RMS
        out[k] = float(np.sqrt(np.mean(sq[s:e])))
    return out, starts


def _build_timeline(
    rms: np.ndarray,
    truth: np.ndarray,
    threshold: Optional[float],
    *,
    max_points: int = 4000,
) -> Dict[str, np.ndarray]:
    """Downsample (rms, truth) to ≤``max_points`` for plotting in the UI."""
    n = rms.size
    if n == 0:
        return {"x": np.array([]), "rms": np.array([]), "truth": np.array([])}
    if n <= max_points:
        idx = np.arange(n)
    else:
        idx = np.linspace(0, n - 1, max_points).astype(np.int64)
    out = {
        "x":    idx.astype(np.float64),
        "rms":  rms[idx].astype(np.float64),
        "truth": truth[idx].astype(np.int64),
    }
    if threshold is not None:
        out["threshold"] = np.full_like(out["rms"], threshold)
    return out


def _make_title(recs: List[RecordingDescriptor]) -> str:
    if len(recs) == 1:
        return f"Unity · {recs[0].label}"
    subjects = sorted({r.subject_id for r in recs})
    return f"Unity · {len(recs)} recordings ({', '.join(subjects)})"