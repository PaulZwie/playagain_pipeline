"""
evaluation/game_eval.py
───────────────────────
Evaluate game-recording CSVs.

Game recordings (the output of :class:`playagain_pipeline.game_recorder.
GameRecorder`) are unique because they contain BOTH the raw EMG and
the predictions the live model produced at recording time, including
per-class probabilities. That makes them especially valuable: we can
ask "did the live system actually do its job?" without re-running
anything.

Two evaluation modes:

* **Logged predictions** (default) — use the ``PredictedGesture`` /
  ``PredictedGestureId`` columns directly. Compares them to one of:
    - ``RawGroundTruth``    (multi-class — preferred when available)
    - ``RequestedGesture``  (multi-class fallback if RawGroundTruth is empty)
    - ``GroundTruthActive`` (binary — collapses everything to active/rest)

* **Replay model** — re-run a saved model over the EMG channels in the
  CSV and compare its output to the ground truth. Useful for asking
  "would model B have done better than model A on the same recording?"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .core import (
    EvaluationMode, EvaluationResult, RecordingDescriptor, RecordingKind,
)
from .loaders import GameRecording, load_game_csv
from .metrics import fill_binary_metrics, fill_classification_metrics

log = logging.getLogger(__name__)


# Ground-truth source labels exposed in the UI
TRUTH_RAW       = "raw_ground_truth"
TRUTH_REQUESTED = "requested_gesture"
TRUTH_ACTIVE    = "ground_truth_active"


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

@dataclass
class GameEvalSettings:
    """Tunables for the game-recording evaluator."""
    # "logged" or "replay"
    mode: str = "logged"

    # Which column to treat as ground truth
    truth_source: str = TRUTH_RAW

    # When True, drop frames where ground truth is "no active gesture"
    # (i.e. transitions / between-trial periods) so the metric isn't
    # diluted by undecidable samples. Recommended.
    drop_inactive_truth_frames: bool = True

    # Optional: when present, only include frames where Confidence
    # exceeds this threshold. None = include all.
    min_confidence: Optional[float] = None

    # ── Replay-mode only ────────────────────────────────────────────
    model_name:       Optional[str] = None
    window_size_ms:   Optional[int] = None
    window_stride_ms: int           = 50

    # ── Output options ──────────────────────────────────────────────
    per_recording_breakdown: bool = True

    # Optional ``subject_id -> cohort`` resolver. When supplied, every
    # per-recording breakdown row is tagged with the participant's
    # cohort ("H" / "I" / "?") so the UI can separate healthy and
    # impaired participants without re-deriving the grouping. Kept as a
    # plain callable so this module needs no dependency on the
    # validation package's ParticipantGroups — the caller (GUI or the
    # thesis pipeline) passes whatever resolver it has. None disables
    # the tagging entirely.
    group_of: Optional[Callable[[str], str]] = None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def evaluate_games(
    data_dir: Path,
    recordings: List[RecordingDescriptor],
    settings: GameEvalSettings,
    *,
    progress: Optional[Callable[[float, str], None]] = None,
) -> EvaluationResult:
    """Evaluate one or more game-recording CSVs as a pooled run."""
    if not recordings:
        raise ValueError("evaluate_games needs at least one recording")
    for r in recordings:
        if r.kind != RecordingKind.GAME:
            raise ValueError(f"Expected GAME descriptors, got {r.kind} ({r.label})")

    if settings.mode == "logged":
        return _evaluate_logged(recordings, settings, progress=progress)
    elif settings.mode == "replay":
        if not settings.model_name:
            raise ValueError("Replay mode requires settings.model_name")
        return _evaluate_replay(data_dir, recordings, settings, progress=progress)
    else:
        raise ValueError(f"Unknown mode: {settings.mode!r}")


# ---------------------------------------------------------------------------
# Logged-predictions mode
# ---------------------------------------------------------------------------

def _evaluate_logged(
    recordings: List[RecordingDescriptor],
    settings: GameEvalSettings,
    *,
    progress: Optional[Callable[[float, str], None]] = None,
) -> EvaluationResult:
    pooled_y_true:  List[np.ndarray] = []
    pooled_y_pred:  List[np.ndarray] = []
    pooled_proba:   List[np.ndarray] = []
    label_names:    Dict[int, str]   = {}
    per_rec:        List[Dict[str, Any]] = []
    notes:          List[str]        = []
    binary_truth = (settings.truth_source == TRUTH_ACTIVE)

    for i, rec in enumerate(recordings):
        if progress: progress(0.05 + 0.85 * i / len(recordings),
                              f"Reading {rec.label}…")

        try:
            game = load_game_csv(rec)
        except Exception as exc:
            log.exception("Failed to load %s", rec.path)
            notes.append(f"⚠ Skipped {rec.label}: {exc}")
            continue

        try:
            y_true, y_pred, y_proba, names_for_rec = _logged_arrays_from_game(
                game, settings,
            )
        except Exception as exc:
            log.exception("Failed to extract arrays from %s", rec.path)
            notes.append(f"⚠ {rec.label}: {exc}")
            continue

        if y_true.size == 0:
            notes.append(f"⚠ {rec.label}: 0 frames after filtering")
            continue

        for k, v in names_for_rec.items():
            label_names.setdefault(k, v)

        pooled_y_true.append(y_true)
        pooled_y_pred.append(y_pred)
        if y_proba is not None:
            pooled_proba.append(y_proba)

        if settings.per_recording_breakdown:
            acc = float(np.mean(y_pred == y_true)) if y_true.size else 0.0
            rec_row = {
                "subject_id":   rec.subject_id,
                "session_id":   rec.session_id,
                "n":            int(y_true.size),
                "accuracy":     acc,
                "model":        rec.meta.get("model_name"),
                "per_class_f1": _per_class_f1(y_true, y_pred),
            }
            if settings.group_of is not None:
                try:
                    rec_row["group"] = settings.group_of(rec.subject_id)
                except Exception:
                    rec_row["group"] = "?"
            per_rec.append(rec_row)

        notes.append(
            f"• {rec.label}: {y_true.size} frames → "
            f"acc={float(np.mean(y_pred == y_true)):.3f}"
        )

    if not pooled_y_true:
        raise RuntimeError("No usable game recordings — see notes for details.")

    y_true  = np.concatenate(pooled_y_true)
    y_pred  = np.concatenate(pooled_y_pred)
    y_proba = np.concatenate(pooled_proba) if (
        pooled_proba and len(pooled_proba) == len(pooled_y_true)
    ) else None

    result = EvaluationResult(
        title=_make_title(recordings, "logged predictions"),
        mode=EvaluationMode.LOGGED,
        kind=RecordingKind.GAME,
        recordings=list(recordings),
        model_name=_pick_model_name(recordings),
        settings={
            "truth_source":   settings.truth_source,
            "drop_inactive":  settings.drop_inactive_truth_frames,
            "min_confidence": settings.min_confidence,
            "per_recording":  per_rec,
        },
    )
    if binary_truth:
        fill_binary_metrics(result, y_true, y_pred, label_names=("rest", "active"))
    else:
        fill_classification_metrics(result, y_true, y_pred,
                                    label_names=label_names, y_proba=y_proba)
    for line in notes:
        result.add_note(line)
    if progress: progress(1.0, "Done.")
    return result


def _logged_arrays_from_game(
    game: GameRecording,
    settings: GameEvalSettings,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Dict[int, str]]:
    """
    Pull (y_true, y_pred, y_proba, label_names) out of a GameRecording
    according to the chosen ground-truth column and filters.
    """
    df = game.df
    if "PredictedGestureId" not in df.columns:
        raise ValueError("Recording is missing 'PredictedGestureId'")

    y_pred_all = df["PredictedGestureId"].to_numpy(dtype=np.int64)

    # Build label_names from the recording's class_names.
    name_map: Dict[int, str] = {}
    for i, name in enumerate(game.class_names):
        name_map[int(i)] = str(name)

    # Resolve ground truth
    if settings.truth_source == TRUTH_ACTIVE:
        if "GroundTruthActive" not in df.columns:
            raise ValueError("Recording is missing 'GroundTruthActive'")
        y_true_all = (df["GroundTruthActive"].to_numpy(dtype=np.int64) > 0).astype(np.int64)
        # Collapse predictions to binary (rest=0 vs anything-else=1).
        # Convention: class 0 == 'rest' in the default gesture set.
        rest_id = _find_rest_id(game.class_names)
        y_pred_all = np.where(y_pred_all == rest_id, 0, 1).astype(np.int64)
        keep = np.ones_like(y_true_all, dtype=bool)
        names_used = {0: "rest", 1: "active"}
    else:
        if settings.truth_source == TRUTH_RAW and "RawGroundTruth" in df.columns:
            y_true_all = df["RawGroundTruth"].to_numpy(dtype=np.int64)
            keep = np.ones_like(y_true_all, dtype=bool)
        elif "RequestedGesture" in df.columns:
            req = df["RequestedGesture"].astype(str).str.strip().str.lower()
            y_true_all = np.array([_lookup(name, game.class_names) for name in req],
                                   dtype=np.int64)
            # Drop "none"/"-1"/empty as undecidable
            keep = np.array([_is_active_truth(name) for name in req])
        else:
            raise ValueError("No usable ground-truth column "
                             "(need RawGroundTruth or RequestedGesture)")
        names_used = name_map

        if settings.drop_inactive_truth_frames:
            # Frames where the game wasn't actively asking for a gesture
            # have RawGroundTruth == 0 *and* GroundTruthActive == 0 — but
            # rest (class 0) is a legitimate label, so we use
            # GroundTruthActive when present to disambiguate.
            if "GroundTruthActive" in df.columns:
                active = df["GroundTruthActive"].to_numpy(dtype=np.int64) > 0
                # Active=1 means a gesture is being held; if we DO have
                # rest as a real class we want to keep both. Practical rule:
                # keep frames where (active) or (RawGroundTruth was logged
                # explicitly as 0 with a non-empty RequestedGesture).
                keep = keep & active
            else:
                keep = keep & (y_true_all >= 0)

    if settings.min_confidence is not None and "Confidence" in df.columns:
        conf_mask = df["Confidence"].to_numpy(dtype=np.float64) >= float(settings.min_confidence)
        keep = keep & conf_mask

    y_true = y_true_all[keep]
    y_pred = y_pred_all[keep]
    y_proba = None

    # If we have per-class Prob_ columns AND we're doing multi-class,
    # keep the probability matrix for confidence/calibration metrics.
    if settings.truth_source != TRUTH_ACTIVE and game.prob_columns:
        proba = game.prob_matrix()
        if proba.shape[0] == len(keep):
            y_proba = proba[keep]

    # Drop labels that are negative sentinels (-1) — happens when the
    # CSV uses -1 for "not asked".
    valid = y_true >= 0
    if not np.all(valid):
        y_true = y_true[valid]; y_pred = y_pred[valid]
        if y_proba is not None: y_proba = y_proba[valid]

    return y_true, y_pred, y_proba, names_used


# ---------------------------------------------------------------------------
# Replay mode
# ---------------------------------------------------------------------------

def _evaluate_replay(
    data_dir: Path,
    recordings: List[RecordingDescriptor],
    settings: GameEvalSettings,
    *,
    progress: Optional[Callable[[float, str], None]] = None,
) -> EvaluationResult:
    """Re-run a saved model on the EMG channels of each game CSV."""
    from playagain_pipeline.models.classifier import (
        ModelManager, EMGFeatureExtractor,
    )

    mm = ModelManager(Path(data_dir) / "models")
    if progress: progress(0.02, f"Loading model {settings.model_name}…")
    model = mm.load_model(settings.model_name)
    md = model.metadata
    win_ms        = int(settings.window_size_ms or getattr(md, "window_size_ms", 200))
    win_stride_ms = int(settings.window_stride_ms or 50)
    feature_config = getattr(md, "feature_config", None)
    extractor = EMGFeatureExtractor(feature_config) if (
        feature_config and feature_config.get("mode") not in (None, "raw")
    ) else None

    pooled_y_true:  List[np.ndarray] = []
    pooled_y_pred:  List[np.ndarray] = []
    pooled_proba:   List[np.ndarray] = []
    label_names = _label_names_from_metadata(md)
    per_rec:        List[Dict[str, Any]] = []
    notes:          List[str]        = []
    binary_truth = (settings.truth_source == TRUTH_ACTIVE)

    for i, rec in enumerate(recordings):
        if progress: progress(0.05 + 0.85 * i / len(recordings),
                              f"Replaying {rec.label}…")
        try:
            game = load_game_csv(rec)
        except Exception as exc:
            notes.append(f"⚠ Skipped {rec.label}: {exc}")
            continue

        emg = game.emg_matrix()
        if emg.size == 0:
            notes.append(f"⚠ {rec.label}: no EMG columns")
            continue

        fs = game.sampling_rate or int(getattr(md, "sampling_rate", 2000))
        win = max(1, int(round(win_ms * fs / 1000.0)))
        stride = max(1, int(round(win_stride_ms * fs / 1000.0)))

        if emg.shape[0] < win:
            notes.append(f"⚠ {rec.label}: too short for window {win}")
            continue

        starts = np.arange(0, emg.shape[0] - win + 1, stride, dtype=np.int64)
        if starts.size == 0:
            notes.append(f"⚠ {rec.label}: no windows")
            continue

        X = np.stack([emg[s:s + win] for s in starts]).astype(np.float32)
        if extractor is not None:
            X = extractor.extract_features(X)

        try:
            y_pred = np.asarray(model.predict(X)).astype(np.int64)
        except Exception as exc:
            notes.append(f"⚠ Model failed on {rec.label}: {exc}")
            continue
        try:
            y_proba = np.asarray(model.predict_proba(X))
        except Exception:
            y_proba = None

        # Build ground truth aligned to window centres
        y_true, keep = _truth_for_windows(game, starts + win // 2, settings,
                                           binary=binary_truth)
        if keep.sum() == 0:
            notes.append(f"⚠ {rec.label}: no labelled frames")
            continue
        y_pred = y_pred[keep]
        if y_proba is not None: y_proba = y_proba[keep]
        y_true = y_true[keep]

        if binary_truth:
            rest_id = _find_rest_id(game.class_names)
            y_pred = np.where(y_pred == rest_id, 0, 1).astype(np.int64)

        pooled_y_true.append(y_true)
        pooled_y_pred.append(y_pred)
        if y_proba is not None: pooled_proba.append(y_proba)

        if settings.per_recording_breakdown:
            rec_row = {
                "subject_id":   rec.subject_id,
                "session_id":   rec.session_id,
                "n":            int(y_true.size),
                "accuracy":     float(np.mean(y_pred == y_true)) if y_true.size else 0.0,
                "per_class_f1": _per_class_f1(y_true, y_pred),
            }
            if settings.group_of is not None:
                try:
                    rec_row["group"] = settings.group_of(rec.subject_id)
                except Exception:
                    rec_row["group"] = "?"
            per_rec.append(rec_row)
        notes.append(f"• {rec.label}: {y_true.size} frames → acc="
                     f"{float(np.mean(y_pred == y_true)):.3f}")

    if not pooled_y_true:
        raise RuntimeError("Replay produced no usable predictions.")

    y_true  = np.concatenate(pooled_y_true)
    y_pred  = np.concatenate(pooled_y_pred)
    y_proba = np.concatenate(pooled_proba) if (
        pooled_proba and len(pooled_proba) == len(pooled_y_true)
    ) else None

    result = EvaluationResult(
        title=_make_title(recordings, settings.model_name or "(replay)"),
        mode=EvaluationMode.MODEL_INFERENCE,
        kind=RecordingKind.GAME,
        recordings=list(recordings),
        model_name=settings.model_name,
        settings={
            "truth_source":     settings.truth_source,
            "window_size_ms":   win_ms,
            "window_stride_ms": win_stride_ms,
            "feature_config":   feature_config,
            "per_recording":    per_rec,
        },
    )
    if binary_truth:
        fill_binary_metrics(result, y_true, y_pred, label_names=("rest", "active"))
    else:
        fill_classification_metrics(result, y_true, y_pred,
                                    label_names=label_names, y_proba=y_proba)
    for line in notes:
        result.add_note(line)
    if progress: progress(1.0, "Done.")
    return result


def _truth_for_windows(
    game: GameRecording,
    centre_indices: np.ndarray,
    settings: GameEvalSettings,
    *,
    binary: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """Pull a label per window centre from the game CSV."""
    df = game.df
    n = len(df)
    centres = np.clip(centre_indices, 0, n - 1).astype(np.int64)
    keep = np.ones(centres.size, dtype=bool)

    if binary:
        if "GroundTruthActive" not in df.columns:
            return np.zeros(centres.size, dtype=np.int64), np.zeros(centres.size, dtype=bool)
        y = (df["GroundTruthActive"].to_numpy(dtype=np.int64)[centres] > 0).astype(np.int64)
        return y, keep

    if settings.truth_source == TRUTH_RAW and "RawGroundTruth" in df.columns:
        y = df["RawGroundTruth"].to_numpy(dtype=np.int64)[centres]
        if settings.drop_inactive_truth_frames and "GroundTruthActive" in df.columns:
            active = df["GroundTruthActive"].to_numpy(dtype=np.int64)[centres] > 0
            keep = keep & active
        keep = keep & (y >= 0)
        return y, keep

    if "RequestedGesture" in df.columns:
        req = df["RequestedGesture"].astype(str).str.strip().str.lower().to_numpy()
        req = req[centres]
        y = np.array([_lookup(name, game.class_names) for name in req], dtype=np.int64)
        keep = np.array([_is_active_truth(name) for name in req])
        return y, keep

    return np.zeros(centres.size, dtype=np.int64), np.zeros(centres.size, dtype=bool)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_rest_id(class_names: List[str]) -> int:
    """Best-effort: find which class id corresponds to 'rest'. Default 0."""
    for i, name in enumerate(class_names):
        if str(name).strip().lower() == "rest":
            return int(i)
    return 0


def _lookup(name: str, class_names: List[str]) -> int:
    """Return the index of ``name`` in ``class_names`` (case-insensitive). -1 if missing."""
    target = str(name).strip().lower()
    for i, n in enumerate(class_names):
        if str(n).strip().lower() == target:
            return int(i)
    return -1


def _is_active_truth(req: str) -> bool:
    """Return True if RequestedGesture string represents a real gesture."""
    if not req:
        return False
    req = req.strip().lower()
    return req not in {"none", "-1", "", "nan"}


def _pick_model_name(recordings: List[RecordingDescriptor]) -> Optional[str]:
    """If all recordings agree on a model name, return it. Else None."""
    names = {r.meta.get("model_name") for r in recordings if r.meta.get("model_name")}
    return next(iter(names)) if len(names) == 1 else None


def _label_names_from_metadata(md: Any) -> Dict[int, str]:
    out: Dict[int, str] = {}
    raw = getattr(md, "class_names", None) or {}
    if isinstance(raw, dict):
        for k, v in raw.items():
            try: out[int(k)] = str(v)
            except (TypeError, ValueError): continue
    elif isinstance(raw, (list, tuple)):
        for i, n in enumerate(raw):
            out[int(i)] = str(n)
    return out


def _make_title(recs: List[RecordingDescriptor], model_or_mode: str) -> str:
    if len(recs) == 1:
        return f"Game · {recs[0].label}  ·  {model_or_mode}"
    subjects = sorted({r.subject_id for r in recs})
    return f"Game · {len(recs)} recordings ({', '.join(subjects)})  ·  {model_or_mode}"


def _per_class_f1(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[int, float]:
    """Per-class F1 for one recording, keyed by integer label."""
    from sklearn.metrics import f1_score
    if y_true.size == 0:
        return {}
    labels = sorted(np.unique(np.concatenate([y_true, y_pred])).tolist())
    if not labels:
        return {}
    f1s = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return {int(l): float(f) for l, f in zip(labels, f1s)}