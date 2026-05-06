"""
evaluation/session_eval.py
──────────────────────────
Evaluate a trained model on one or more recording sessions.

Pipeline:

1. Load each session via the project's :class:`RecordingSession`
   (we do this through the ``DataManager`` so calibration / bad-channel
   logic stays consistent with training).
2. Cut sliding windows that match the model's window/stride.
3. Optionally pre-extract features using the model's recorded
   ``feature_config``.
4. ``model.predict`` (and ``predict_proba`` if available).
5. Aggregate metrics with :mod:`evaluation.metrics`.

We import the project's ``ModelManager`` / ``EMGFeatureExtractor`` /
``apply_bad_channel_strategy`` lazily so this module is cheap to
import in a worker thread that hasn't constructed Qt/torch yet.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .core import (
    EvaluationMode, EvaluationResult, RecordingDescriptor, RecordingKind,
)
from .loaders import load_session_data, load_session_gesture_set
from .metrics import fill_classification_metrics

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

@dataclass
class SessionEvalSettings:
    """Tunables for the session evaluator."""
    model_name: str

    # Window slicing — defaults to the model's metadata if zero/None.
    window_size_ms:    Optional[int] = None
    window_stride_ms:  int           = 50

    # Whether to apply the per-session rotation (matches default dataset
    # creation behaviour). Bad-channel handling reuses the strategy
    # baked into the session metadata.
    apply_rotation:    bool = True
    bad_channel_mode:  str  = "interpolate"   # "interpolate" | "zero"

    # Skip trials whose gesture_label is < 0 (calibration sync, etc.)
    include_invalid:   bool = False

    # Optional: report metrics per-session in addition to the pooled set.
    per_session_breakdown: bool = True


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def evaluate_sessions(
    data_dir: Path,
    recordings: List[RecordingDescriptor],
    settings: SessionEvalSettings,
    *,
    progress: Optional[Callable[[float, str], None]] = None,
) -> EvaluationResult:
    """Run a saved model over sessions and return a unified result."""
    if not recordings:
        raise ValueError("evaluate_sessions needs at least one recording")
    for r in recordings:
        if r.kind != RecordingKind.SESSION:
            raise ValueError(f"Expected SESSION descriptors, got {r.kind} ({r.label})")

    # ── Lazy imports — keep the module light ───────────────────────────
    from playagain_pipeline.models.classifier import (
        ModelManager, EMGFeatureExtractor, apply_bad_channel_strategy,
    )

    mm = ModelManager(Path(data_dir) / "models")
    if progress: progress(0.02, f"Loading model {settings.model_name}…")
    model = mm.load_model(settings.model_name)

    md = model.metadata
    win_ms = int(settings.window_size_ms or getattr(md, "window_size_ms", 200))
    win_stride_ms = int(settings.window_stride_ms or 50)
    sampling_rate = int(getattr(md, "sampling_rate", 2000))
    feature_config = getattr(md, "feature_config", None)
    label_names = _label_names_from_metadata(md)

    extractor = EMGFeatureExtractor(feature_config) if (
        feature_config and feature_config.get("mode") not in (None, "raw")
    ) else None

    # ── Per-session collection ─────────────────────────────────────────
    pooled_y_true: List[np.ndarray] = []
    pooled_y_pred: List[np.ndarray] = []
    pooled_proba:  List[np.ndarray] = []
    per_session_metrics: List[Dict[str, Any]] = []
    notes: List[str] = []

    for i, rec in enumerate(recordings):
        if progress: progress(0.05 + 0.85 * i / max(len(recordings), 1),
                              f"Evaluating {rec.label}…")

        try:
            X, y, sess_label_names = _windows_for_session(
                rec, win_ms, win_stride_ms, sampling_rate,
                bad_channel_mode=settings.bad_channel_mode,
                apply_rotation=settings.apply_rotation,
                include_invalid=settings.include_invalid,
                feature_extractor=extractor,
                bad_channel_apply_fn=apply_bad_channel_strategy,
            )
        except Exception as exc:
            log.exception("Could not build windows for %s", rec.path)
            notes.append(f"⚠ Skipped {rec.label}: {exc}")
            continue

        if X.shape[0] == 0:
            notes.append(f"⚠ {rec.label}: 0 windows after slicing")
            continue

        try:
            y_pred = np.asarray(model.predict(X))
        except Exception as exc:
            log.exception("model.predict failed on %s", rec.path)
            notes.append(f"⚠ Model failed on {rec.label}: {exc}")
            continue

        try:
            y_proba = np.asarray(model.predict_proba(X))
        except Exception:
            y_proba = None

        # Merge any newly-discovered label names from this session
        for k, v in sess_label_names.items():
            label_names.setdefault(k, v)

        pooled_y_true.append(y)
        pooled_y_pred.append(y_pred)
        if y_proba is not None:
            pooled_proba.append(y_proba)

        if settings.per_session_breakdown:
            per_session_metrics.append(_per_session_summary(rec, y, y_pred))

        notes.append(
            f"• {rec.label}: {X.shape[0]} windows → "
            f"acc={float(np.mean(y_pred == y)):.3f}"
        )

    if not pooled_y_true:
        raise RuntimeError("No usable sessions — see notes for details.")

    y_true = np.concatenate(pooled_y_true)
    y_pred = np.concatenate(pooled_y_pred)
    y_proba = np.concatenate(pooled_proba) if pooled_proba and len(pooled_proba) == len(pooled_y_true) else None

    # ── Build the result ───────────────────────────────────────────────
    result = EvaluationResult(
        title=_make_title(recordings, settings.model_name),
        mode=EvaluationMode.MODEL_INFERENCE,
        kind=RecordingKind.SESSION,
        recordings=list(recordings),
        model_name=settings.model_name,
        settings={
            "window_size_ms":   win_ms,
            "window_stride_ms": win_stride_ms,
            "apply_rotation":   settings.apply_rotation,
            "bad_channel_mode": settings.bad_channel_mode,
            "include_invalid":  settings.include_invalid,
            "feature_config":   feature_config,
            "per_session": per_session_metrics,
        },
    )
    fill_classification_metrics(result, y_true, y_pred,
                                label_names=label_names, y_proba=y_proba)
    for line in notes:
        result.add_note(line)
    if progress: progress(1.0, "Done.")
    return result


# ---------------------------------------------------------------------------
# Per-feature comparison (diagnostic)
# ---------------------------------------------------------------------------

def evaluate_features_lda(
    data_dir: Path,
    recordings: List[RecordingDescriptor],
    feature_names: Sequence[str],
    *,
    window_size_ms: int = 200,
    window_stride_ms: int = 50,
    test_ratio: float = 0.25,
    random_state: int = 42,
    progress: Optional[Callable[[float, str], None]] = None,
) -> EvaluationResult:
    """
    Score each feature individually with a held-out LDA classifier and
    return the per-feature accuracy in ``result.per_feature``.

    Useful as a quick "which feature carries the gesture signal best"
    sanity check. Does NOT replace proper feature ablation experiments.
    """
    if not recordings:
        raise ValueError("evaluate_features_lda needs at least one recording")
    feature_names = list(feature_names)
    if not feature_names:
        raise ValueError("Provide at least one feature to evaluate")

    from playagain_pipeline.models.classifier import (
        EMGFeatureExtractor, apply_bad_channel_strategy,
    )
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    # Slice raw windows once (shared across features)
    if progress: progress(0.05, "Slicing windows…")

    pooled_X: List[np.ndarray] = []
    pooled_y: List[np.ndarray] = []
    label_names: Dict[int, str] = {}
    for i, rec in enumerate(recordings):
        if progress: progress(0.05 + 0.25 * i / len(recordings),
                              f"Loading {rec.label}…")
        try:
            X, y, names = _windows_for_session(
                rec, window_size_ms, window_stride_ms,
                sampling_rate=None,             # use session's own rate
                bad_channel_mode="interpolate",
                apply_rotation=True,
                include_invalid=False,
                feature_extractor=None,         # raw windows
                bad_channel_apply_fn=apply_bad_channel_strategy,
            )
        except Exception as exc:
            log.warning("skip %s: %s", rec.label, exc)
            continue
        if X.shape[0] == 0:
            continue
        pooled_X.append(X); pooled_y.append(y)
        for k, v in names.items():
            label_names.setdefault(k, v)

    if not pooled_X:
        raise RuntimeError("No data after slicing windows.")

    X = np.concatenate(pooled_X)         # (windows, samples, channels)
    y = np.concatenate(pooled_y)
    if len(np.unique(y)) < 2:
        raise RuntimeError("Need ≥2 classes to fit an LDA.")

    # One feature at a time
    per_feature: Dict[str, float] = {}
    for j, fname in enumerate(feature_names):
        if progress: progress(0.35 + 0.6 * j / len(feature_names),
                              f"Feature: {fname}")
        try:
            extractor = EMGFeatureExtractor({"mode": "selected", "features": [fname]})
            Xf = extractor.extract_features(X)
            Xtr, Xte, ytr, yte = train_test_split(
                Xf, y, test_size=test_ratio,
                stratify=y, random_state=random_state,
            )
            clf = LinearDiscriminantAnalysis()
            clf.fit(Xtr, ytr)
            per_feature[fname] = float(accuracy_score(yte, clf.predict(Xte)))
        except Exception as exc:
            log.warning("feature %s failed: %s", fname, exc)
            per_feature[fname] = float("nan")

    result = EvaluationResult(
        title=f"Feature ranking · {len(recordings)} session(s) · LDA",
        mode=EvaluationMode.MODEL_INFERENCE,
        kind=RecordingKind.SESSION,
        recordings=list(recordings),
        model_name="LDA (per-feature)",
        settings={
            "window_size_ms":   window_size_ms,
            "window_stride_ms": window_stride_ms,
            "test_ratio":       test_ratio,
            "random_state":     random_state,
            "features":         feature_names,
        },
        n_samples=int(X.shape[0]),
        per_feature=per_feature,
    )
    if progress: progress(1.0, "Done.")
    return result


# ---------------------------------------------------------------------------
# Window builder
# ---------------------------------------------------------------------------

def _windows_for_session(
    rec: RecordingDescriptor,
    window_size_ms: int,
    window_stride_ms: int,
    sampling_rate: Optional[int],
    *,
    bad_channel_mode: str,
    apply_rotation: bool,
    include_invalid: bool,
    feature_extractor,            # Optional[EMGFeatureExtractor]
    bad_channel_apply_fn,         # apply_bad_channel_strategy
) -> Tuple[np.ndarray, np.ndarray, Dict[int, str]]:
    """
    Build (X, y) windows from one session, mirroring the logic in
    DataManager.create_dataset so metrics match training-time numbers.
    """
    data, meta, trials = load_session_data(rec)
    fs = int(sampling_rate or meta.get("sampling_rate") or 2000)

    win = max(1, int(round(window_size_ms * fs / 1000.0)))
    stride = max(1, int(round(window_stride_ms * fs / 1000.0)))

    # Bad-channel strategy
    bad_channels = list(meta.get("bad_channels") or [])
    if bad_channels:
        data = bad_channel_apply_fn(data, bad_channels, mode=bad_channel_mode)

    # Per-session rotation: prefer stored channel_mapping
    if apply_rotation:
        mapping = meta.get("channel_mapping")
        rot     = int(meta.get("rotation_offset") or 0)
        if mapping and len(mapping) == data.shape[1]:
            data = data[:, list(mapping)]
        elif rot != 0:
            n_ch = int(meta.get("num_channels") or data.shape[1])
            data = data[:, [(i + rot) % n_ch for i in range(n_ch)]]

    # Slice
    Xs: List[np.ndarray] = []
    ys: List[int]        = []
    label_names = load_session_gesture_set(rec)
    for t in trials:
        if not include_invalid and not t.get("is_valid", True):
            continue
        if t.get("trial_type", "gesture") != "gesture":
            continue       # skip calibration_sync etc.
        s, e = int(t.get("start_sample", 0)), int(t.get("end_sample", 0))
        if e <= s + win:
            continue
        lbl = int(t.get("gesture_label", -1))
        if lbl < 0:
            continue
        for start in range(s, e - win + 1, stride):
            Xs.append(data[start:start + win])
            ys.append(lbl)

    if not Xs:
        return np.empty((0, 0, 0), dtype=np.float32), np.empty(0, dtype=np.int64), label_names

    X = np.stack(Xs).astype(np.float32, copy=False)
    y = np.asarray(ys, dtype=np.int64)

    if feature_extractor is not None:
        X = feature_extractor.extract_features(X)

    return X, y, label_names


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _label_names_from_metadata(md: Any) -> Dict[int, str]:
    """Pull a {label_id: display_name} dict out of a model's metadata."""
    out: Dict[int, str] = {}
    raw = getattr(md, "class_names", None) or {}
    if isinstance(raw, dict):
        for k, v in raw.items():
            try:
                out[int(k)] = str(v)
            except (TypeError, ValueError):
                continue
    elif isinstance(raw, (list, tuple)):
        for i, name in enumerate(raw):
            out[int(i)] = str(name)
    return out


def _per_session_summary(
    rec: RecordingDescriptor,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, Any]:
    from sklearn.metrics import f1_score
    correct = int(np.sum(y_pred == y_true))
    n = int(y_true.size)
    # Per-class F1 — keyed by integer label so the UI can join with the
    # global label_names map. Empty dict if y_true is empty.
    pcf1: Dict[int, float] = {}
    if n > 0:
        labels = sorted(np.unique(np.concatenate([y_true, y_pred])).tolist())
        if labels:
            f1s = f1_score(y_true, y_pred, labels=labels,
                           average=None, zero_division=0)
            pcf1 = {int(l): float(f) for l, f in zip(labels, f1s)}
    return {
        "subject_id":   rec.subject_id,
        "session_id":   rec.session_id,
        "n":            n,
        "correct":      correct,
        "accuracy":     correct / n if n else 0.0,
        "per_class_f1": pcf1,
    }


def _make_title(recs: List[RecordingDescriptor], model_name: str) -> str:
    subjects = sorted({r.subject_id for r in recs})
    if len(recs) == 1:
        return f"{model_name}  ·  {recs[0].label}"
    return f"{model_name}  ·  {len(recs)} sessions ({', '.join(subjects)})"