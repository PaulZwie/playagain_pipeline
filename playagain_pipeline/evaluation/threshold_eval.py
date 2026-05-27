"""
evaluation/threshold_eval.py
════════════════════════════
Evaluate **threshold-driven gameplay** — the original Unity scheme that
fires on a single RMS threshold instead of a multi-class gesture model.

This evaluator complements ``unity_eval.py``. ``unity_eval`` treats a
Unity recording as a binary-classification benchmark for the EMG signal
itself (sweep thresholds, pick the F1 winner, report signal quality).
This module answers a different, user-facing question:

    *How well did the threshold-based game actually work for the user
    at the time of recording?*

— and additionally:

    *How much performance was left on the table by the threshold
    calibration choice?*

That second question is the quantitative motivation for replacing
threshold-driven gameplay with the gesture-recognition model.

Three perspectives per recording
────────────────────────────────
1. **As-recorded** — uses the ``GestureActive`` column the game wrote
   while running. This is what the user actually experienced. When
   the live threshold was wrong, this number is poor *regardless of
   signal quality*.

2. **Profile-threshold** — re-derive predictions from RMS using the
   threshold that ``profile.json`` claims was active at recording
   time. Often equivalent to (1); occasionally differs when the live
   threshold and the profile desynced (e.g. the game wrote a value
   it never actually applied).

3. **Optimal-threshold** — F1-optimal threshold chosen post-hoc per
   recording. The upper bound the RMS signal could have achieved
   with a perfectly-tuned threshold. AUROC is reported alongside as
   a threshold-free measure of signal quality.

Why thresholds need extra care
──────────────────────────────
Profiles in the wild misbehave in two specific ways:

  * **Wrong scale.** A profile may say ``currentThreshold = 0.05``
    against a recording whose RMS never exceeds ``0.004``. The
    as-recorded gameplay then has F1 = 0 — but the underlying
    EMG has AUC ≈ 0.82. A naive "as-recorded only" table would
    conclude the EMG is broken, which is the wrong conclusion.
    ``suspect_threshold`` flags this when the active threshold is
    above the session's max RMS times a safety ratio.

  * **Off-by-history.** ``currentThreshold`` reflects the most
    recent save — which may have happened *after* the recording.
    The right field is the entry in ``thresholdHistory`` whose
    timestamp is the latest still ≤ the recording start. The
    ``_profile_threshold_for`` helper does that lookup.

For the uploaded example (VP_03, 2026-02-16 12:17:32):
    history @ 12:17:21  → threshold 0.25  (active at recording start)
    history @ 12:18:02  → threshold 0.05  (saved AFTER recording)
    currentThreshold    = 0.05

The correct as-recorded threshold for this CSV is 0.25, not 0.05.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Settings
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ThresholdEvalSettings:
    """Tunables for the threshold-gameplay evaluator."""

    # Threshold-sweep resolution. 200 points covers 4-5 orders of
    # magnitude in log space; finer is overkill given RMS quantisation.
    n_thresholds: int = 200

    # Objective for the "optimal" row.  "f1" | "youden" | "accuracy"
    objective: str = "f1"

    # If a profile threshold is greater than this multiple of the
    # session's max RMS, mark it ``suspect``. A profile threshold
    # well above the session's max RMS cannot possibly trigger —
    # the most common manifestation of a desynced profile.
    threshold_sanity_ratio: float = 1.5

    # Optional global override for the profile threshold — ignored
    # when None. Useful for "what if every session had used this
    # one threshold?"
    fixed_profile_threshold: Optional[float] = None


# ═══════════════════════════════════════════════════════════════════════════
# Per-recording row
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ThresholdRow:
    """One row of the threshold-gameplay per-recording table."""

    subject_id: str
    session_id: str
    path:       str               # source CSV path

    n_frames:   int
    duration_s: float

    # RMS distribution summary (helps users sanity-check threshold scale)
    rms_max:    float
    rms_p50:    float
    rms_p95:    float

    # Thresholds (NaN when unavailable)
    profile_threshold: float
    optimal_threshold: float

    # As-recorded metrics — from the GestureActive column the game logged
    asrec_accuracy:       float
    asrec_precision:      float
    asrec_recall:         float
    asrec_f1:             float
    asrec_pos_rate:       float    # fraction of frames where GestureActive=1
    asrec_pos_rate_truth: float    # fraction of frames where GroundTruth=1

    # Profile-threshold metrics
    profile_accuracy:  float
    profile_precision: float
    profile_recall:    float
    profile_f1:        float

    # Optimal-threshold metrics
    opt_accuracy:      float
    opt_precision:     float
    opt_recall:        float
    opt_f1:            float

    # Signal-quality summary (threshold-free)
    auc: float

    # Flags
    suspect_threshold: bool = False
    notes:             str  = ""

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class ThresholdReport:
    rows:     List[ThresholdRow] = field(default_factory=list)
    sweep:    List[Dict[str, float]] = field(default_factory=list)
    pooled:   Dict[str, Dict[str, float]] = field(default_factory=dict)
    settings: Dict[str, Any] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════
# CSV reader — both old and new schemas
# ═══════════════════════════════════════════════════════════════════════════

_RMS_COLS            = ("RMS",)
_GESTURE_ACTIVE_COLS = ("GestureActive",)
_GROUND_TRUTH_COLS   = ("GroundTruthActive", "GroundTruth")   # new, old
_INPUT_SOURCE_COLS   = ("InputSource",)


def _read_unity_csv(
    path: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Read a Unity gameplay CSV.

    Returns
    -------
    rms        : float64 (n,) — per-frame RMS the game saw
    asrec_pred : int64   (n,) — per-frame GestureActive
                 (−1 sentinel when the column is absent)
    truth      : int64   (n,) — per-frame GroundTruth(Active)
    duration_s : float — span of Timestamp column
    """
    import pandas as pd                                 # lazy: pandas is heavy

    df = pd.read_csv(path)

    def _pick(cands: Tuple[str, ...]) -> Optional[str]:
        for c in cands:
            if c in df.columns:
                return c
        return None

    rms_col   = _pick(_RMS_COLS)
    pred_col  = _pick(_GESTURE_ACTIVE_COLS)
    truth_col = _pick(_GROUND_TRUTH_COLS)
    src_col   = _pick(_INPUT_SOURCE_COLS)

    if rms_col is None:
        raise ValueError(f"{path.name}: no RMS column")
    if truth_col is None:
        raise ValueError(f"{path.name}: no GroundTruth(Active) column")

    # Drop non-EMG rows (Markers etc.). They have no associated EMG
    # measurement and would distort per-frame metrics.
    if src_col is not None:
        df = df[df[src_col] == "EMG"].reset_index(drop=True)

    rms   = df[rms_col].to_numpy(dtype=np.float64)
    truth = (df[truth_col].to_numpy(dtype=np.float64) > 0.5).astype(np.int64)

    if pred_col is None:
        # Older recordings may lack a logged prediction column.
        # We can still evaluate profile + optimal; mark the
        # as-recorded row NaN.
        asrec = np.full(rms.size, -1, dtype=np.int64)
    else:
        asrec = (df[pred_col].to_numpy(dtype=np.float64) > 0.5).astype(np.int64)

    if "Timestamp" in df.columns and df["Timestamp"].notna().any():
        ts = df["Timestamp"].to_numpy(dtype=np.float64)
        duration_s = float(ts[-1] - ts[0])
    else:
        duration_s = float("nan")

    return rms, asrec, truth, duration_s


# ═══════════════════════════════════════════════════════════════════════════
# Profile handling
# ═══════════════════════════════════════════════════════════════════════════

def _parse_iso_ish(s: str) -> Optional[datetime]:
    """Tolerant timestamp parser — returns None instead of raising."""
    if not s:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f",
                "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f"):
        try:
            return datetime.strptime(s.strip(), fmt)
        except (ValueError, AttributeError):
            continue
    return None


def _profile_threshold_for(
    profile: Dict[str, Any],
    recording_started_at: Optional[datetime],
) -> Tuple[float, str]:
    """
    Pick the threshold from ``profile`` that was active when the
    recording started.

    Selection order:
      1. If a recording start timestamp is provided, the
         ``thresholdHistory`` entry whose timestamp is the latest
         still ≤ recording start. This is the *only* honest choice
         when the profile has been edited after the recording.
      2. ``currentThreshold`` — most recent save (may have happened
         after the recording, in which case (1) above is preferred).
      3. NaN if neither is present.
    """
    history = profile.get("thresholdHistory") or []
    if recording_started_at is not None and history:
        matched: Optional[Dict[str, Any]] = None
        matched_ts: Optional[datetime] = None
        for entry in history:
            ts = _parse_iso_ish(entry.get("timestamp", ""))
            if ts is None or ts > recording_started_at:
                continue
            if matched_ts is None or ts > matched_ts:
                matched, matched_ts = entry, ts
        if matched is not None:
            return (
                float(matched.get("threshold", float("nan"))),
                f"profile history @ {matched.get('timestamp')}",
            )

    cur = profile.get("currentThreshold")
    if cur is not None:
        return float(cur), "profile currentThreshold"
    return float("nan"), "no profile threshold"


def _load_profile(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    """Robust profile reader; returns None on missing/invalid."""
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:                            # noqa: BLE001
        log.warning("Could not read profile %s: %s", p, exc)
        return None


def _find_profile_path(csv_path: Path) -> Optional[Path]:
    """
    Look for ``profile.json`` near ``csv_path``. Unity writes profiles
    one or two levels above the recording (``Users/<id>/profile.json``,
    with recordings under ``Users/<id>/RecordedData/...``). Walk up
    a handful of levels.
    """
    p = csv_path.parent
    for _ in range(4):
        cand = p / "profile.json"
        if cand.exists():
            return cand
        if p.parent == p:
            break
        p = p.parent
    return None


# Filename timestamp parser: ``emg_2026-02-16_12-17-32.csv`` →
# datetime(2026, 2, 16, 12, 17, 32).
_FNAME_TS_RE = re.compile(
    r"(\d{4})-(\d{2})-(\d{2})[_-](\d{2})-(\d{2})-(\d{2})"
)


def _recording_started_at(csv_path: Path) -> Optional[datetime]:
    """Best-effort recording-start extraction from the filename."""
    m = _FNAME_TS_RE.search(csv_path.stem)
    if not m:
        return None
    try:
        y, mo, d, h, mi, s = map(int, m.groups())
        return datetime(y, mo, d, h, mi, s)
    except ValueError:
        return None


# ═══════════════════════════════════════════════════════════════════════════
# Binary metrics
# ═══════════════════════════════════════════════════════════════════════════

def _binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Accuracy / precision / recall / F1 for binary {0,1} arrays."""
    if y_true.size == 0:
        return {"accuracy": float("nan"), "precision": float("nan"),
                "recall":   float("nan"), "f1":        float("nan")}
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    acc = (tp + tn) / max(1, y_true.size)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    f1   = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


def _auc_from_score(score: np.ndarray, truth: np.ndarray) -> float:
    """AUROC by Mann–Whitney U with proper tie handling. Avoids sklearn."""
    truth = truth.astype(np.int64)
    if truth.sum() == 0 or truth.sum() == truth.size:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(order) + 1)
    s = score[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            avg = ranks[order[i:j+1]].mean()
            ranks[order[i:j+1]] = avg
        i = j + 1
    n_pos = int(truth.sum())
    n_neg = int(truth.size - n_pos)
    rank_sum_pos = float(ranks[truth == 1].sum())
    u = rank_sum_pos - n_pos * (n_pos + 1) / 2.0
    return u / (n_pos * n_neg)


def _threshold_sweep(
    rms: np.ndarray,
    truth: np.ndarray,
    n_points: int,
) -> List[Dict[str, float]]:
    """
    Build a log-spaced sweep over thresholds between the 1st and 99.5th
    RMS percentiles. Each row has threshold + acc/precision/recall/F1
    + TPR/FPR for ROC curve construction.
    """
    if rms.size == 0:
        return []
    rms_pos = rms[rms > 0]
    if rms_pos.size == 0:
        return []
    lo = float(np.quantile(rms_pos, 0.01))
    hi = float(np.quantile(rms_pos, 0.995))
    if not np.isfinite(lo) or lo <= 0:
        lo = max(float(rms_pos.min()), 1e-12)
    if hi <= lo:
        hi = lo * 10.0
    thresholds = np.geomspace(lo, hi, num=max(2, n_points))
    out: List[Dict[str, float]] = []
    n_pos = max(1, int(truth.sum()))
    n_neg = max(1, int(truth.size - truth.sum()))
    for t in thresholds:
        pred = (rms > t).astype(np.int64)
        m = _binary_metrics(truth, pred)
        tp = int(np.sum((pred == 1) & (truth == 1)))
        fp = int(np.sum((pred == 1) & (truth == 0)))
        out.append({
            "threshold": float(t),
            "accuracy":  m["accuracy"],
            "precision": m["precision"],
            "recall":    m["recall"],
            "f1":        m["f1"],
            "tpr":       tp / n_pos,
            "fpr":       fp / n_neg,
        })
    return out


def _pick_threshold(
    sweep: List[Dict[str, float]], objective: str
) -> Dict[str, float]:
    """Return the sweep row that maximises ``objective``."""
    if not sweep:
        return {}
    if objective == "youden":
        key = lambda r: r["tpr"] - r["fpr"]
    elif objective == "accuracy":
        key = lambda r: r["accuracy"]
    else:
        key = lambda r: r["f1"]
    return max(sweep, key=key)


# ═══════════════════════════════════════════════════════════════════════════
# Public entry point
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_threshold_gameplay(
    csv_paths: List[Path],
    settings: Optional[ThresholdEvalSettings] = None,
    *,
    profile_path: Optional[Path] = None,
    recording_started_at: Optional[Dict[Path, datetime]] = None,
) -> ThresholdReport:
    """
    Score one or more Unity threshold-gameplay CSVs.

    Parameters
    ----------
    csv_paths : List[Path]
        Recordings to evaluate. Each must have at least ``RMS`` and
        either ``GroundTruthActive`` or ``GroundTruth``.
    settings : ThresholdEvalSettings, optional
        Defaults are reasonable.
    profile_path : Path, optional
        Single ``profile.json`` applied to every recording. When
        omitted, this function searches for ``profile.json`` next to
        each CSV (walks up four directory levels).
    recording_started_at : dict[Path → datetime], optional
        Per-recording start times for selecting the right threshold-
        history entry. When omitted, the filename's
        ``YYYY-MM-DD_HH-MM-SS`` timestamp is parsed; failing that,
        ``currentThreshold`` is used.

    Returns
    -------
    ThresholdReport
    """
    s = settings or ThresholdEvalSettings()
    started_at = dict(recording_started_at or {})

    rows: List[ThresholdRow] = []
    pooled_rms:   List[np.ndarray] = []
    pooled_truth: List[np.ndarray] = []

    explicit_profile = _load_profile(profile_path)

    for csv in csv_paths:
        csv = Path(csv)
        try:
            rms, asrec, truth, duration = _read_unity_csv(csv)
        except Exception as exc:                        # noqa: BLE001
            log.warning("Could not read %s: %s", csv, exc)
            continue

        if rms.size == 0:
            continue

        # Resolve profile + when-the-recording-started
        profile = explicit_profile if explicit_profile is not None \
            else _load_profile(_find_profile_path(csv))
        rec_started = started_at.get(csv) or _recording_started_at(csv)

        if s.fixed_profile_threshold is not None:
            thr_profile, thr_src = (
                float(s.fixed_profile_threshold), "fixed override",
            )
        elif profile is not None:
            thr_profile, thr_src = _profile_threshold_for(profile, rec_started)
        else:
            thr_profile, thr_src = float("nan"), "no profile"

        # As-recorded metrics
        if (asrec >= 0).all():
            asrec_m = _binary_metrics(truth, asrec)
            asrec_pos_rate = float(asrec.mean())
        else:
            asrec_m = {"accuracy": float("nan"), "precision": float("nan"),
                       "recall":   float("nan"), "f1":        float("nan")}
            asrec_pos_rate = float("nan")

        # Profile-threshold metrics
        if np.isfinite(thr_profile):
            pred_p = (rms > thr_profile).astype(np.int64)
            profile_m = _binary_metrics(truth, pred_p)
        else:
            profile_m = {"accuracy": float("nan"), "precision": float("nan"),
                         "recall":   float("nan"), "f1":        float("nan")}

        # Optimal threshold via sweep
        sweep = _threshold_sweep(rms, truth, n_points=s.n_thresholds)
        best  = _pick_threshold(sweep, s.objective)
        thr_opt = best.get("threshold", float("nan"))
        opt_m = {"accuracy":  best.get("accuracy", float("nan")),
                 "precision": best.get("precision", float("nan")),
                 "recall":    best.get("recall", float("nan")),
                 "f1":        best.get("f1", float("nan"))}

        # Plausibility check on the profile threshold
        rms_max = float(rms.max())
        suspect = (
            np.isfinite(thr_profile)
            and rms_max > 0
            and thr_profile > s.threshold_sanity_ratio * rms_max
        )

        note_bits: List[str] = [thr_src]
        if suspect:
            note_bits.append(
                f"⚠ profile threshold {thr_profile:.6g} > {rms_max:.6g} "
                f"(RMS max); session never triggered as recorded"
            )
        notes = "; ".join(note_bits)

        subject_id = _infer_subject_id(csv)
        session_id = csv.stem

        rows.append(ThresholdRow(
            subject_id=subject_id,
            session_id=session_id,
            path=str(csv),
            n_frames=int(rms.size),
            duration_s=float(duration),
            profile_threshold=float(thr_profile),
            optimal_threshold=float(thr_opt),
            rms_max=rms_max,
            rms_p50=float(np.quantile(rms, 0.5)),
            rms_p95=float(np.quantile(rms, 0.95)),
            asrec_accuracy=float(asrec_m["accuracy"]),
            asrec_precision=float(asrec_m["precision"]),
            asrec_recall=float(asrec_m["recall"]),
            asrec_f1=float(asrec_m["f1"]),
            asrec_pos_rate=float(asrec_pos_rate),
            asrec_pos_rate_truth=float(truth.mean()),
            profile_accuracy=float(profile_m["accuracy"]),
            profile_precision=float(profile_m["precision"]),
            profile_recall=float(profile_m["recall"]),
            profile_f1=float(profile_m["f1"]),
            opt_accuracy=float(opt_m["accuracy"]),
            opt_precision=float(opt_m["precision"]),
            opt_recall=float(opt_m["recall"]),
            opt_f1=float(opt_m["f1"]),
            auc=_auc_from_score(rms, truth),
            suspect_threshold=bool(suspect),
            notes=notes,
        ))
        pooled_rms.append(rms)
        pooled_truth.append(truth)

    # Pooled sweep + summary
    pooled: Dict[str, Dict[str, float]] = {}
    sweep_pooled: List[Dict[str, float]] = []
    if pooled_rms:
        rms_all   = np.concatenate(pooled_rms)
        truth_all = np.concatenate(pooled_truth)
        sweep_pooled = _threshold_sweep(rms_all, truth_all, n_points=s.n_thresholds)

        # Frame-weighted pooled means across per-recording rows.
        # We do this rather than re-concatenate prediction streams
        # because each "perspective" uses a different prediction
        # stream (asrec / profile / optimal) and re-deriving them
        # is redundant disk work.
        def _pooled_mean(field_name: str) -> float:
            num = den = 0.0
            for r in rows:
                v = getattr(r, field_name)
                if v is not None and np.isfinite(v):
                    num += float(v) * r.n_frames
                    den += r.n_frames
            return float(num / den) if den else float("nan")

        for prefix, label in [("asrec",   "asrec"),
                              ("profile", "profile"),
                              ("opt",     "optimal")]:
            pooled[label] = {
                "accuracy":  _pooled_mean(f"{prefix}_accuracy"),
                "precision": _pooled_mean(f"{prefix}_precision"),
                "recall":    _pooled_mean(f"{prefix}_recall"),
                "f1":        _pooled_mean(f"{prefix}_f1"),
            }
        pooled["auc_pooled"] = {"value": _auc_from_score(rms_all, truth_all)}

    return ThresholdReport(
        rows=rows,
        sweep=sweep_pooled,
        pooled=pooled,
        settings={
            "n_thresholds":            s.n_thresholds,
            "objective":               s.objective,
            "threshold_sanity_ratio":  s.threshold_sanity_ratio,
            "fixed_profile_threshold": s.fixed_profile_threshold,
        },
    )


def _infer_subject_id(csv_path: Path) -> str:
    """
    Best-effort subject extraction from a Unity recording path.

    Unity layout: ``.../<subject>/RecordedData/<session>.csv``. Walk
    parents looking for a ``VP_``-prefix or ``SUBJECT*``; fall back
    to the immediate parent name. Skip well-known intermediate dirs.
    """
    intermediate = {"RecordedData", "recorded_data", "recordings",
                    "Users", "users"}
    for part in reversed(csv_path.parts[:-1]):
        if part in intermediate:
            continue
        if part.startswith("VP_") or part.upper().startswith("SUBJECT"):
            return part
    parent = csv_path.parent.name
    return parent or "unknown"


# ═══════════════════════════════════════════════════════════════════════════
# Discovery
# ═══════════════════════════════════════════════════════════════════════════

def discover_threshold_gameplay(root: Path) -> List[Path]:
    """
    Walk ``root`` and return every CSV that looks like a Unity
    threshold-gameplay recording (has ``RMS`` and ``GroundTruth(Active)``).

    Cheap: reads only the header line of each CSV.
    """
    root = Path(root)
    out: List[Path] = []
    if not root.exists():
        return out
    for p in root.rglob("*.csv"):
        try:
            with open(p, "r", encoding="utf-8") as f:
                header = f.readline()
        except Exception:                               # noqa: BLE001
            continue
        cols = {c.strip().lower() for c in header.split(",")}
        if "rms" not in cols:
            continue
        if not (cols & {"groundtruth", "groundtruthactive"}):
            continue
        out.append(p)
    return sorted(out)
