"""
evaluation/intra_session_eval.py
────────────────────────────────
Standalone **intra-session** validation — deliberately *outside* the
main experiment pipeline (``runner.py`` / ``cv_strategies.py``).

Why this module exists
──────────────────────
The everyday question this answers is the optimistic ceiling:

    "If I fit a model on the *first* part of one recording and test it
     on the *held-out tail* of that **same** recording — same electrode
     placement, same day, same skin prep — how well does it do?"

That number is the upper bound the cross-session / LOSO results in the
main pipeline are measured against. It is not a deployment number; it
is a sanity ceiling. Keeping it in its own module means you can point
it at "the models that did best in the real experiments" and get a
per-session ceiling without spinning up an ``ExperimentConfig``,
``SessionCorpus``, or the whole ``ValidationRunner``.

How the split is done (and why it differs from runner.within_session)
─────────────────────────────────────────────────────────────────────
``runner._materialise_fold`` implements ``within_session`` as a single
global temporal cut over *all* concatenated windows. Its own comment
notes the trap: because windows are stored trial-by-trial, the tail
"often contains only the last few gesture classes," so the held-out
test set can be missing classes entirely.

This module avoids that by splitting **per class, at the trial level**:

  * group each session's valid gesture trials by class,
  * order each class's trials in time (by ``start_sample``),
  * hold out that class's temporally-last trials as test, the rest train.

Consequences, all of them deliberate:
  * every class with ≥2 trials appears in *both* train and test;
  * the split is still **temporal** (test is the later repetitions),
    so it respects the anti-leakage philosophy in ``cv_strategies``;
  * whole trials go to one side, so no two windows from the *same*
    trial ever straddle the cut — the single biggest source of inflated
    within-session accuracy is closed off.

A class with only one trial can't be split at the trial level; the
``singleton_policy`` setting decides what happens (default: a temporal
window split *inside* that one trial so the class is still scored, with
a note recording that it was done).

Two run modes
─────────────
``refit=True``  (default, the real intra-session protocol)
    Train a *fresh* model — same recipe as the saved model you picked,
    or an explicit ``model_type`` — on the train section, score the
    held-out section.

``refit=False`` (quick look)
    Skip training entirely: load the saved model as-is and score it on
    the held-out section only. The train section is unused. This is the
    "what does my deployed model do on the tail of this recording"
    sanity check, *not* an intra-session ceiling — a note is added so
    the distinction survives into the report.

Output
──────
Every entry point returns the package-standard :class:`EvaluationResult`
(via :func:`evaluation.metrics.fill_classification_metrics`) so the
existing plots / thesis reports render it unchanged. Pooled runs also
stash a per-session breakdown under ``result.settings["per_session"]``,
exactly like :func:`evaluation.session_eval.evaluate_sessions`.

Heavy ML imports (``ModelManager`` / ``EMGFeatureExtractor`` /
``apply_bad_channel_strategy``) are deferred into the functions that
need them so this module stays cheap to import from a worker thread.
"""

from __future__ import annotations

import logging
import random as _random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from .core import (
    EvaluationMode, EvaluationResult, RecordingDescriptor, RecordingKind,
)
from .loaders import load_session_data, load_session_gesture_set
from .metrics import fill_classification_metrics

log = logging.getLogger(__name__)


# Deep models that need raw 3-D windows (N, C, T) and crash on 2-D
# feature tensors — kept in sync with runner._RAW_WINDOW_MODELS.
_RAW_WINDOW_MODELS = frozenset({"cnn", "attention_net", "mstnet"})


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

@dataclass
class IntraSessionSettings:
    """Tunables for the standalone intra-session evaluator.

    Pick the model recipe with exactly one of:
      * ``model_name``  — recover the recipe (type, feature_config,
        window size, hyper-params, class names) from a saved model;
      * ``model_type`` (+ optional ``feature_config`` / ``model_params``)
        — specify it explicitly.

    Providing ``model_name`` is the "select the best performing model"
    path: point at the saved artefact that won your real experiments and
    this re-fits the *same recipe* within each session.
    """
    # ── Model recipe (choose one) ──────────────────────────────────────
    model_name:    Optional[str]            = None
    model_type:    Optional[str]            = None
    feature_config: Optional[Dict[str, Any]] = None
    model_params:  Dict[str, Any]           = field(default_factory=dict)

    # ── Run mode ───────────────────────────────────────────────────────
    # True  → fit a fresh model on the train section (real intra-session).
    # False → load the saved model and only score the test section.
    refit: bool = True

    # ── Windowing (falls back to saved-model metadata, then defaults) ──
    window_size_ms:   Optional[int] = None
    window_stride_ms: int           = 50

    # ── Intra-session split ────────────────────────────────────────────
    test_fraction:             float = 0.25
    split_unit:                str   = "trial"   # "trial" (recommended) | "window"
    min_test_trials_per_class: int   = 1
    # What to do with a class that has a single trial (can't trial-split):
    #   "within_trial" → temporal window split inside that trial (default)
    #   "train_only"   → put it all in train; class is left unscored
    singleton_policy: str = "within_trial"

    # ── Preprocessing (mirror dataset-creation defaults) ───────────────
    apply_rotation:   bool = True
    bad_channel_mode: str  = "interpolate"       # "interpolate" | "zero"
    include_invalid:  bool = False

    # ── Determinism / reporting ────────────────────────────────────────
    seed:                  int  = 42
    per_session_breakdown: bool = True

    # ------------------------------------------------------------------

    def resolved_model_label(self) -> str:
        if self.model_name:
            return self.model_name
        if self.model_type:
            return str(self.model_type)
        return "?"


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def evaluate_intra_session(
    data_dir: Path,
    recordings: List[RecordingDescriptor],
    settings: IntraSessionSettings,
    *,
    progress: Optional[Callable[[float, str], None]] = None,
) -> EvaluationResult:
    """
    Run one model recipe through an intra-session split on each session
    and return a single **pooled** :class:`EvaluationResult`.

    Each session is split independently (per-class temporal trial
    hold-out); a fresh model is fit per session when ``refit`` is True.
    Test-set predictions are pooled across sessions for the headline
    metrics, and a per-session breakdown lands in
    ``result.settings["per_session"]``.
    """
    if not recordings:
        raise ValueError("evaluate_intra_session needs at least one recording")
    for r in recordings:
        if r.kind != RecordingKind.SESSION:
            raise ValueError(
                f"Expected SESSION descriptors, got {r.kind} ({r.label})"
            )
    if not settings.model_name and not settings.model_type:
        raise ValueError(
            "IntraSessionSettings needs either model_name (recover recipe "
            "from a saved model) or model_type (explicit)."
        )

    # Lazy, heavy imports.
    from playagain_pipeline.models.classifier import (   # noqa: WPS433
        ModelManager, EMGFeatureExtractor, apply_bad_channel_strategy,
    )

    mm = ModelManager(Path(data_dir) / "models")
    recipe = _resolve_recipe(mm, settings)
    # Build the feature extractor once and stash it on the recipe so the
    # per-session split builder can apply it to each train/test half.
    recipe["_extractor_obj"] = _make_extractor(recipe, EMGFeatureExtractor)

    pooled_y_true: List[np.ndarray] = []
    pooled_y_pred: List[np.ndarray] = []
    pooled_proba:  List[np.ndarray] = []
    per_session_metrics: List[Dict[str, Any]] = []
    notes: List[str] = []
    label_names: Dict[int, str] = dict(recipe["label_names"])
    inf_ms_samples: List[float] = []

    n = max(len(recordings), 1)
    for i, rec in enumerate(recordings):
        if progress:
            progress(0.05 + 0.9 * i / n, f"Intra-session · {rec.label}…")

        # Per-session deterministic seed so repeated runs match.
        _seed_everything(_session_seed(settings.seed, rec))

        try:
            split = _build_intra_session_split(
                rec, recipe, settings,
                bad_channel_apply_fn=apply_bad_channel_strategy,
            )
        except Exception as exc:                              # noqa: BLE001
            log.exception("Intra-session split failed for %s", rec.path)
            notes.append(f"⚠ Skipped {rec.label}: {exc}")
            continue

        for line in split.notes:
            notes.append(f"  ({rec.label}) {line}")

        if split.X_test.shape[0] == 0:
            notes.append(f"⚠ {rec.label}: empty test section after split")
            continue
        if settings.refit and split.X_train.shape[0] == 0:
            notes.append(f"⚠ {rec.label}: empty train section after split")
            continue

        try:
            model, y_pred, y_proba, inf_ms = _fit_and_predict(
                mm, recipe, settings, split, data_dir,
                seed=_session_seed(settings.seed, rec),
            )
        except Exception as exc:                              # noqa: BLE001
            log.exception("Intra-session model step failed on %s", rec.path)
            notes.append(f"⚠ Model failed on {rec.label}: {exc}")
            continue

        for k, v in split.label_names.items():
            label_names.setdefault(k, v)

        pooled_y_true.append(split.y_test)
        pooled_y_pred.append(y_pred)
        if y_proba is not None:
            pooled_proba.append(y_proba)
        if inf_ms is not None:
            inf_ms_samples.append(inf_ms)

        if settings.per_session_breakdown:
            per_session_metrics.append(
                _per_session_summary(rec, split, y_pred)
            )

        notes.append(
            f"• {rec.label}: train={split.X_train.shape[0]} "
            f"test={split.X_test.shape[0]} windows → "
            f"acc={float(np.mean(y_pred == split.y_test)):.3f}"
        )

    if not pooled_y_true:
        raise RuntimeError("No usable sessions — see notes for details.")

    y_true = np.concatenate(pooled_y_true)
    y_pred = np.concatenate(pooled_y_pred)
    y_proba = (
        np.concatenate(pooled_proba)
        if pooled_proba and len(pooled_proba) == len(pooled_y_true)
        else None
    )

    result = EvaluationResult(
        title=_make_title(recordings, settings.resolved_model_label(), settings.refit),
        mode=EvaluationMode.MODEL_INFERENCE,
        kind=RecordingKind.SESSION,
        recordings=list(recordings),
        model_name=settings.resolved_model_label(),
        settings={
            "evaluation":        "intra_session",
            "refit":             settings.refit,
            "model_type":        recipe["model_type"],
            "window_size_ms":    recipe["window_size_ms"],
            "window_stride_ms":  settings.window_stride_ms,
            "feature_config":    recipe["feature_config"],
            "test_fraction":     settings.test_fraction,
            "split_unit":        settings.split_unit,
            "singleton_policy":  settings.singleton_policy,
            "apply_rotation":    settings.apply_rotation,
            "bad_channel_mode":  settings.bad_channel_mode,
            "include_invalid":   settings.include_invalid,
            "seed":              settings.seed,
            "per_session":       per_session_metrics,
        },
    )
    if not settings.refit:
        result.add_note(
            "refit=False — saved model scored on the held-out section only; "
            "this is NOT an intra-session ceiling (no within-session fit)."
        )
    fill_classification_metrics(
        result, y_true, y_pred, label_names=label_names, y_proba=y_proba,
    )
    if inf_ms_samples:
        result.inference_ms_per_window = float(np.mean(inf_ms_samples))
    for line in notes:
        result.add_note(line)
    if progress:
        progress(1.0, "Done.")
    return result


def evaluate_intra_session_per_recording(
    data_dir: Path,
    recordings: List[RecordingDescriptor],
    settings: IntraSessionSettings,
    *,
    progress: Optional[Callable[[float, str], None]] = None,
) -> List[EvaluationResult]:
    """
    Like :func:`evaluate_intra_session` but return **one result per
    session** instead of a pooled one — handy when you want a row per
    recording in a thesis table rather than a single aggregate.

    Sessions that error out are skipped (a logged warning is emitted);
    the returned list only contains sessions that produced metrics.
    """
    out: List[EvaluationResult] = []
    n = max(len(recordings), 1)
    for i, rec in enumerate(recordings):
        if progress:
            progress(i / n, f"Intra-session · {rec.label}…")
        try:
            res = evaluate_intra_session(
                data_dir, [rec], settings, progress=None,
            )
        except Exception as exc:                              # noqa: BLE001
            log.warning("Intra-session skipped %s: %s", rec.label, exc)
            continue
        out.append(res)
    if progress:
        progress(1.0, "Done.")
    return out


def run_intra_session_suite(
    data_dir: Path,
    recordings: List[RecordingDescriptor],
    model_names: Sequence[str],
    *,
    base_settings: Optional[IntraSessionSettings] = None,
    progress: Optional[Callable[[float, str], None]] = None,
) -> Dict[str, EvaluationResult]:
    """
    Run intra-session validation for **several saved models** — the
    "select the best performers and compare their within-session
    ceilings" flow.

    Returns ``{model_name: pooled EvaluationResult}``. Models that fail
    entirely are recorded as missing (skipped) rather than aborting the
    suite, so one broken artefact doesn't lose the others.
    """
    base = base_settings or IntraSessionSettings()
    out: Dict[str, EvaluationResult] = {}
    n = max(len(model_names), 1)
    for i, name in enumerate(model_names):
        if progress:
            progress(i / n, f"Model {name}…")
        # Copy base settings but pin this model name (and clear any
        # explicit type so the recipe is recovered from the artefact).
        s = IntraSessionSettings(**{**base.__dict__, "model_name": name,
                                    "model_type": None})
        try:
            out[name] = evaluate_intra_session(data_dir, recordings, s)
        except Exception as exc:                              # noqa: BLE001
            log.warning("Intra-session suite skipped model %s: %s", name, exc)
            continue
    if progress:
        progress(1.0, "Done.")
    return out


# ---------------------------------------------------------------------------
# Recipe resolution
# ---------------------------------------------------------------------------

def _resolve_recipe(mm, settings: IntraSessionSettings) -> Dict[str, Any]:
    """
    Build the model recipe dict used for (re)fitting:

        {
          "model_type":     str,
          "model_params":   dict,
          "feature_config": Optional[dict],
          "window_size_ms": int,
          "sampling_rate":  Optional[int],
          "label_names":    {int: str},
          "saved_model":    Optional[<loaded model>],  # only when refit=False
        }

    When ``settings.model_name`` is given the recipe is recovered from
    the saved model's metadata; explicit settings always override the
    recovered values where provided.
    """
    recipe: Dict[str, Any] = {
        "model_type":     settings.model_type,
        "model_params":   dict(settings.model_params),
        "feature_config": settings.feature_config,
        "window_size_ms": settings.window_size_ms,
        "sampling_rate":  None,
        "label_names":    {},
        "saved_model":    None,
    }

    if settings.model_name:
        model = mm.load_model(settings.model_name)
        md = getattr(model, "metadata", None)
        recipe["saved_model"] = model
        # Model type: classifiers store it under a few possible names.
        recovered_type = (
            getattr(md, "model_type", None)
            or getattr(md, "architecture", None)
            or getattr(md, "type", None)
        )
        recipe["model_type"] = settings.model_type or recovered_type
        # Hyper-params: best-effort recovery; explicit params win.
        recovered_params = (
            getattr(md, "params", None)
            or getattr(md, "hyperparameters", None)
            or {}
        )
        if isinstance(recovered_params, dict):
            merged = dict(recovered_params)
            merged.update(settings.model_params)   # explicit overrides
            recipe["model_params"] = merged
        # Feature config / window / sampling rate / class names.
        recipe["feature_config"] = (
            settings.feature_config
            if settings.feature_config is not None
            else getattr(md, "feature_config", None)
        )
        recipe["window_size_ms"] = int(
            settings.window_size_ms or getattr(md, "window_size_ms", 0) or 200
        )
        recipe["sampling_rate"] = int(getattr(md, "sampling_rate", 0) or 0) or None
        recipe["label_names"] = _label_names_from_metadata(md)

    if not recipe["model_type"] and settings.refit:
        raise ValueError(
            "Could not determine model_type. The saved model's metadata "
            "didn't expose it under model_type / architecture / type — "
            "pass model_type explicitly in IntraSessionSettings."
        )
    if recipe["window_size_ms"] in (None, 0):
        recipe["window_size_ms"] = int(settings.window_size_ms or 200)
    return recipe


def _make_extractor(recipe: Dict[str, Any], EMGFeatureExtractor):
    """Construct the feature extractor for this recipe, or None for raw."""
    fc = recipe.get("feature_config")
    model_type = str(recipe.get("model_type") or "").lower()
    if model_type in _RAW_WINDOW_MODELS:
        return None                              # deep models want raw windows
    if fc and fc.get("mode") not in (None, "raw"):
        return EMGFeatureExtractor(fc)
    return None


# ---------------------------------------------------------------------------
# Split container + builder
# ---------------------------------------------------------------------------

@dataclass
class _IntraSplit:
    """Materialised intra-session split for one session."""
    X_train: np.ndarray
    y_train: np.ndarray
    X_test:  np.ndarray
    y_test:  np.ndarray
    label_names: Dict[int, str]
    notes: List[str] = field(default_factory=list)
    # Raw (pre-feature) test windows kept only when refit is False and we
    # need the saved model's own feature path — populated lazily.


def _build_intra_session_split(
    rec: RecordingDescriptor,
    recipe: Dict[str, Any],
    settings: IntraSessionSettings,
    *,
    bad_channel_apply_fn,
) -> _IntraSplit:
    """
    Window one session trial-by-trial, then split per class temporally.

    Feature extraction (if any) is applied *after* the split, separately
    to each side, so the two halves never share an extraction call. EMG
    features are per-window so this changes nothing numerically, but it
    keeps the leakage story airtight and matches how the runner builds
    its train/test views.
    """
    win_ms = int(recipe["window_size_ms"])
    stride_ms = int(settings.window_stride_ms or 50)

    windows = _windows_by_trial(
        rec, win_ms, stride_ms,
        sampling_rate=recipe.get("sampling_rate"),
        bad_channel_mode=settings.bad_channel_mode,
        apply_rotation=settings.apply_rotation,
        include_invalid=settings.include_invalid,
        bad_channel_apply_fn=bad_channel_apply_fn,
    )
    label_names = windows.label_names
    if windows.X.shape[0] == 0:
        return _IntraSplit(
            _empty3(), _empty1(), _empty3(), _empty1(),
            label_names, notes=["0 windows after slicing"],
        )

    if settings.split_unit == "window":
        train_mask, test_mask, notes = _per_class_window_mask(
            windows, test_fraction=settings.test_fraction,
        )
    else:  # "trial" (default, recommended)
        train_mask, test_mask, notes = _per_class_trial_mask(
            windows,
            test_fraction=settings.test_fraction,
            min_test_trials_per_class=settings.min_test_trials_per_class,
            singleton_policy=settings.singleton_policy,
        )

    X_tr_raw, y_tr = windows.X[train_mask], windows.y[train_mask]
    X_te_raw, y_te = windows.X[test_mask],  windows.y[test_mask]

    # Feature extraction per side (skipped for raw-window deep models).
    extractor = recipe.get("_extractor_obj")
    if extractor is not None:
        X_tr = extractor.extract_features(X_tr_raw) if X_tr_raw.shape[0] else X_tr_raw
        X_te = extractor.extract_features(X_te_raw) if X_te_raw.shape[0] else X_te_raw
    else:
        X_tr, X_te = X_tr_raw, X_te_raw

    return _IntraSplit(X_tr, y_tr, X_te, y_te, label_names, notes=notes)


# ---------------------------------------------------------------------------
# Pure split logic (no ML deps — unit-testable in isolation)
# ---------------------------------------------------------------------------

def _temporal_trial_split(
    class_to_trialids: Dict[int, List[int]],
    *,
    test_fraction: float,
    min_test_trials_per_class: int,
) -> Tuple[Set[int], Set[int], List[int]]:
    """
    Decide, per class, which *trial ids* are train vs test.

    ``class_to_trialids`` maps each class label to its trial ids in
    **temporal order** (earliest first). For each class with ≥2 trials,
    the temporally-last ``n_test`` trials become test; the rest train.
    ``n_test`` is ``max(min_test_trials_per_class, round(n*frac))``
    clamped to ``[1, n-1]`` so at least one trial stays on each side.

    Returns ``(train_ids, test_ids, singleton_classes)`` where
    ``singleton_classes`` lists labels that had a single trial and so
    were left for the caller's singleton policy to handle.
    """
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be in (0, 1)")
    train_ids: Set[int] = set()
    test_ids:  Set[int] = set()
    singletons: List[int] = []
    for label, tids in class_to_trialids.items():
        n = len(tids)
        if n <= 1:
            singletons.append(label)
            continue
        n_test = max(int(min_test_trials_per_class),
                     int(round(n * test_fraction)))
        n_test = max(1, min(n_test, n - 1))
        test_ids.update(tids[-n_test:])
        train_ids.update(tids[:-n_test])
    return train_ids, test_ids, singletons


# ---------------------------------------------------------------------------
# Windowing — trial-by-trial, tagging each window with its trial
# ---------------------------------------------------------------------------

@dataclass
class _WindowedSession:
    X: np.ndarray               # (n_windows, win, n_ch) raw windows
    y: np.ndarray               # (n_windows,) class label per window
    trial_id: np.ndarray        # (n_windows,) trial index per window
    win_start: np.ndarray       # (n_windows,) absolute start sample (temporal key)
    trial_order: Dict[int, int] # trial_id -> temporal rank (by start_sample)
    label_names: Dict[int, str]


def _windows_by_trial(
    rec: RecordingDescriptor,
    window_size_ms: int,
    window_stride_ms: int,
    sampling_rate: Optional[int],
    *,
    bad_channel_mode: str,
    apply_rotation: bool,
    include_invalid: bool,
    bad_channel_apply_fn,
) -> _WindowedSession:
    """
    Build raw windows for one session, tagging every window with the
    trial it came from and its absolute start sample.

    Mirrors ``session_eval._windows_for_session`` preprocessing (bad
    channels, per-session rotation, trial filtering) so the numbers line
    up with training-time windows — but keeps windows *raw* and *tagged*
    so the caller can split by trial before extracting features.
    """
    data, meta, trials = load_session_data(rec)
    fs = int(sampling_rate or meta.get("sampling_rate") or 2000)

    win = max(1, int(round(window_size_ms * fs / 1000.0)))
    stride = max(1, int(round(window_stride_ms * fs / 1000.0)))

    bad_channels = list(meta.get("bad_channels") or [])
    if bad_channels:
        data = bad_channel_apply_fn(data, bad_channels, mode=bad_channel_mode)

    if apply_rotation:
        mapping = meta.get("channel_mapping")
        rot = int(meta.get("rotation_offset") or 0)
        if mapping and len(mapping) == data.shape[1]:
            data = data[:, list(mapping)]
        elif rot != 0:
            n_ch = int(meta.get("num_channels") or data.shape[1])
            data = data[:, [(i + rot) % n_ch for i in range(n_ch)]]

    label_names = load_session_gesture_set(rec)

    Xs: List[np.ndarray] = []
    ys: List[int] = []
    tids: List[int] = []
    starts: List[int] = []
    trial_start_sample: Dict[int, int] = {}

    tindex = 0
    for t in trials:
        if not include_invalid and not t.get("is_valid", True):
            continue
        if t.get("trial_type", "gesture") != "gesture":
            continue
        s, e = int(t.get("start_sample", 0)), int(t.get("end_sample", 0))
        if e <= s + win:
            continue
        lbl = int(t.get("gesture_label", -1))
        if lbl < 0:
            continue
        this_tid = tindex
        trial_start_sample[this_tid] = s
        for start in range(s, e - win + 1, stride):
            Xs.append(data[start:start + win])
            ys.append(lbl)
            tids.append(this_tid)
            starts.append(start)
        tindex += 1

    if not Xs:
        return _WindowedSession(
            _empty3(), _empty1(), _empty1(), _empty1(), {}, label_names,
        )

    X = np.stack(Xs).astype(np.float32, copy=False)
    y = np.asarray(ys, dtype=np.int64)
    trial_id = np.asarray(tids, dtype=np.int64)
    win_start = np.asarray(starts, dtype=np.int64)

    # Temporal rank of each trial by its start sample (stable tie-break
    # on trial id) so "later" trials are well-defined.
    ranked = sorted(trial_start_sample, key=lambda k: (trial_start_sample[k], k))
    trial_order = {tid: rank for rank, tid in enumerate(ranked)}

    return _WindowedSession(X, y, trial_id, win_start, trial_order, label_names)


def _per_class_trial_mask(
    w: _WindowedSession,
    *,
    test_fraction: float,
    min_test_trials_per_class: int,
    singleton_policy: str,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Boolean train/test masks over windows from a per-class trial split."""
    notes: List[str] = []

    # class -> trial ids in temporal order
    class_to_trialids: Dict[int, List[int]] = {}
    for lbl in np.unique(w.y):
        tids_for_class = np.unique(w.trial_id[w.y == lbl]).tolist()
        tids_for_class.sort(key=lambda tid: w.trial_order.get(tid, tid))
        class_to_trialids[int(lbl)] = tids_for_class

    train_tids, test_tids, singletons = _temporal_trial_split(
        class_to_trialids,
        test_fraction=test_fraction,
        min_test_trials_per_class=min_test_trials_per_class,
    )

    train_mask = np.isin(w.trial_id, list(train_tids))
    test_mask = np.isin(w.trial_id, list(test_tids))

    # Singleton classes (one trial) — can't trial-split.
    for label in singletons:
        only_tid = class_to_trialids[label][0]
        sel = w.trial_id == only_tid
        if singleton_policy == "within_trial":
            # Temporal window split inside the single trial.
            idx = np.where(sel)[0]
            order = idx[np.argsort(w.win_start[idx])]
            cut = int(len(order) * (1.0 - test_fraction))
            cut = max(1, min(cut, len(order) - 1)) if len(order) >= 2 else len(order)
            train_mask[order[:cut]] = True
            test_mask[order[cut:]] = True
            notes.append(
                f"class {label}: single trial — temporal window split "
                f"inside it (mild within-trial correlation)."
            )
        else:  # "train_only"
            train_mask[sel] = True
            notes.append(f"class {label}: single trial — train-only, left unscored.")

    n_tr = int(train_mask.sum())
    n_te = int(test_mask.sum())
    notes.append(
        f"per-class trial split: {len(class_to_trialids)} class(es), "
        f"{n_tr} train / {n_te} test windows."
    )
    return train_mask, test_mask, notes


def _per_class_window_mask(
    w: _WindowedSession,
    *,
    test_fraction: float,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Alternative split: per class, hold out the temporally-last
    ``test_fraction`` of *windows* (not trials).

    Keeps every class in both halves (unlike the runner's global cut)
    but, unlike the trial split, windows from one trial can land on both
    sides — so this is the looser, more optimistic option. Offered for
    parity / quick looks; ``split_unit="trial"`` is recommended.
    """
    notes: List[str] = []
    train_mask = np.zeros(w.y.shape[0], dtype=bool)
    test_mask = np.zeros(w.y.shape[0], dtype=bool)
    for lbl in np.unique(w.y):
        idx = np.where(w.y == lbl)[0]
        order = idx[np.argsort(w.win_start[idx])]
        n = len(order)
        if n < 2:
            train_mask[order] = True
            continue
        cut = int(round(n * (1.0 - test_fraction)))
        cut = max(1, min(cut, n - 1))
        train_mask[order[:cut]] = True
        test_mask[order[cut:]] = True
    notes.append(
        f"per-class window split: {int(train_mask.sum())} train / "
        f"{int(test_mask.sum())} test windows "
        f"(windows from one trial may straddle the cut)."
    )
    return train_mask, test_mask, notes


# ---------------------------------------------------------------------------
# Fit + predict for one session
# ---------------------------------------------------------------------------

def _fit_and_predict(
    mm,
    recipe: Dict[str, Any],
    settings: IntraSessionSettings,
    split: _IntraSplit,
    data_dir: Path,
    *,
    seed: int,
) -> Tuple[Any, np.ndarray, Optional[np.ndarray], Optional[float]]:
    """
    Fit a fresh model (refit) or reuse the saved model (refit=False),
    then predict the held-out section. Returns
    ``(model, y_pred, y_proba_or_None, inference_ms_per_window_or_None)``.
    """
    if not settings.refit:
        model = recipe.get("saved_model")
        if model is None:
            raise RuntimeError("refit=False needs a saved model_name.")
        return (model, *_predict(model, split.X_test))

    model_type = str(recipe["model_type"])
    params = dict(recipe.get("model_params") or {})
    # Validation-time SVM defaults, copied from runner._fit_model: the
    # deployed defaults (probability=True, no sample cap) turn a fit into
    # a multi-hour blocking op for no reportable benefit here.
    if model_type.lower() == "svm":
        params.setdefault("probability", True)   # we DO want proba for calibration
        params.setdefault("cache_size", 1024.0)
        params.setdefault("max_train_samples", 20000)

    model = mm.create_model(model_type, name=f"_intra_{model_type}", **params)

    fs = int(recipe.get("sampling_rate") or 2000)
    n_ch = int(split.X_train.shape[-1]) if split.X_train.ndim >= 2 else 0
    payload = {
        "X": split.X_train,
        "y": split.y_train,
        "metadata": {
            "name":           f"_intra_{model_type}",
            "window_size_ms": int(recipe["window_size_ms"]),
            "sampling_rate":  fs,
            "num_channels":   n_ch,
        },
    }
    # Pass seed through when the installed ModelManager supports it
    # (older signatures don't) — mirrors runner._fit_model.
    try:
        mm.train_model(model, payload, random_state=seed, save=False)
    except TypeError:
        mm.train_model(model, payload, save=False)

    return (model, *_predict(model, split.X_test))


def _predict(
    model, X_test: np.ndarray,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[float]]:
    t0 = time.time()
    y_pred = np.asarray(model.predict(X_test))
    inf_ms = (time.time() - t0) * 1000.0 / max(len(X_test), 1)
    try:
        y_proba = np.asarray(model.predict_proba(X_test))
    except Exception:                                         # noqa: BLE001
        y_proba = None
    return y_pred, y_proba, inf_ms


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _empty3() -> np.ndarray:
    return np.empty((0, 0, 0), dtype=np.float32)


def _empty1() -> np.ndarray:
    return np.empty((0,), dtype=np.int64)


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
    split: _IntraSplit,
    y_pred: np.ndarray,
) -> Dict[str, Any]:
    from sklearn.metrics import f1_score
    y_true = split.y_test
    n = int(y_true.size)
    correct = int(np.sum(y_pred == y_true))
    pcf1: Dict[int, float] = {}
    if n > 0:
        labels = sorted(np.unique(np.concatenate([y_true, y_pred])).tolist())
        if labels:
            f1s = f1_score(y_true, y_pred, labels=labels,
                           average=None, zero_division=0)
            pcf1 = {int(l): float(f) for l, f in zip(labels, f1s)}
    return {
        "subject_id":     rec.subject_id,
        "session_id":     rec.session_id,
        "n_train":        int(split.X_train.shape[0]),
        "n_test":         n,
        "correct":        correct,
        "accuracy":       correct / n if n else 0.0,
        "per_class_f1":   pcf1,
    }


def _session_seed(base_seed: int, rec: RecordingDescriptor) -> int:
    """Stable per-session seed so repeated runs are reproducible."""
    h = hash((int(base_seed), str(rec.subject_id), str(rec.session_id)))
    return int(h & 0x7FFF_FFFF)


def _seed_everything(seed: int) -> None:
    _random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    try:
        import torch  # noqa: WPS433
        torch.manual_seed(seed)
    except Exception:                                         # noqa: BLE001
        pass


def _make_title(
    recs: List[RecordingDescriptor], model_label: str, refit: bool,
) -> str:
    tag = "intra-session" if refit else "intra-session (no refit)"
    subjects = sorted({r.subject_id for r in recs})
    if len(recs) == 1:
        return f"{model_label} · {tag} · {recs[0].label}"
    return (f"{model_label} · {tag} · {len(recs)} sessions "
            f"({', '.join(subjects)})")
