"""
runner.py  (v2)
────────────────
Execute one ExperimentConfig end-to-end with live progress reporting,
per-fold determinism, and confusion-matrix capture.

What changed vs v1 (and why)
────────────────────────────
• Per-fold seeding        — hash(base_seed, fold_idx, model_type) seeds
                           numpy / random / torch before each
                           (model × fold). v1 seeded numpy once for the
                           whole run, so classical model results were
                           non-deterministic across repeats.
• Live progress           — an optional ProgressReporter is invoked at
                           run_start / fold_start / fold_done / run_done
                           and a free-form log() channel. Used by the
                           GUI Validation tab to render fold-by-fold
                           updates with ETA.
• Cancellation            — progress.should_cancel() is checked between
                           folds. A cancelled run still persists the
                           folds it has completed.
• Confusion matrices      — FoldResult.confusion carries the per-fold
                           matrix; RunResult.aggregate_confusion() sums
                           across folds per model. Plot generators no
                           longer need to re-run inference.
• Slim train_meta         — epoch-by-epoch training curves are
                           summarised (num_epochs, final losses/accs)
                           before being placed in FoldResult.extra.
                           v1 dumped the full curves into results.json
                           which grew to tens of MB for deep-model
                           runs.
• Batched val eval        — deep models evaluate the val set in
                           mini-batches. v1 pushed the whole val tensor
                           through the model in one forward pass, which
                           OOMed on MPS for realistic val sizes.
• Correct sampling rate   — pulled from session metadata instead of
                           hardcoded 2000 Hz.
• Pass seed into
  ModelManager.train_model  — the classical path honours cfg.seed
                           instead of the hardcoded random_state=42.

The module stays import-light at the top and defers heavy ML imports
into the methods that actually need them, so headless environments
without torch/catboost can still import runner.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import platform
import random as _random
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

import numpy as np

from .config import ExperimentConfig, dump_experiment
from .corpus import SessionCorpus, SessionRecord
from .cv_strategies import get_strategy

log = logging.getLogger(__name__)


# Models that require 3-D raw windows (N, C, T) and reject 2-D feature
# tensors. Their predict() path unconditionally unpacks `N, C, T =
# X.shape` so passing (N, F) crashes with a "not enough values to
# unpack" ValueError. MLP / LDA / RF / SVM / CatBoost all accept both
# shapes via the `X if X.ndim == 2 else self.extract_features(X)`
# pattern, so only these three care.
_RAW_WINDOW_MODELS = frozenset({"cnn", "attention_net", "mstnet"})


def _model_wants_raw_windows(model_type: str) -> bool:
    return str(model_type).lower() in _RAW_WINDOW_MODELS


# ---------------------------------------------------------------------------
# Progress reporting — pure Python, thread-safe to call from a worker.
# The GUI wraps this in QMetaObject.invokeMethod to bounce to the UI thread.
# ---------------------------------------------------------------------------

class ProgressReporter(Protocol):
    """
    Callable hooks invoked by ValidationRunner at key milestones.

    All methods are optional — the default NoopProgress base class
    provides do-nothing implementations so implementations only need to
    override what they care about.
    """

    def on_run_start(self, total_folds: int, total_models: int,
                     records: List[SessionRecord]) -> None: ...

    def on_fold_start(self, fold_idx: int, total_evals: int,
                      fold_id: str, model_type: str) -> None: ...

    def on_fold_done(self, fold_idx: int, total_evals: int,
                     fold_result: "FoldResult") -> None: ...

    def on_run_done(self, result: "RunResult") -> None: ...

    def log(self, message: str) -> None: ...

    def should_cancel(self) -> bool: ...


class NoopProgress:
    """Default reporter — silent, never cancels."""

    def on_run_start(self, *_a, **_k): pass
    def on_fold_start(self, *_a, **_k): pass
    def on_fold_done(self, *_a, **_k): pass
    def on_run_done(self, *_a, **_k): pass
    def log(self, message: str) -> None:
        log.info(message)
    def should_cancel(self) -> bool:
        return False


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class FoldResult:
    fold_id:        str
    model_type:     str
    n_train_windows: int
    n_test_windows:  int
    accuracy:       float
    macro_f1:       float
    per_class_f1:   Dict[str, float] = field(default_factory=dict)
    train_seconds:  float = 0.0
    inference_ms:   float = 0.0

    # Validation-set metrics — populated only when the CV strategy
    # provides an explicit val split (currently: holdout_split).
    n_val_windows:  int = 0
    val_accuracy:   Optional[float] = None
    val_macro_f1:   Optional[float] = None

    # Confusion matrix rows/cols are ordered by `confusion_labels`
    # (integer label IDs). Use `label_names` to resolve to names.
    confusion:        Optional[List[List[int]]] = None
    confusion_labels: Optional[List[int]] = None
    label_names:      Dict[int, str] = field(default_factory=dict)

    # seed actually used for this (fold, model), for reproduction
    fold_seed:    int = 0

    extra:        Dict[str, Any] = field(default_factory=dict)


@dataclass
class RunResult:
    experiment:    ExperimentConfig
    folds:         List[FoldResult] = field(default_factory=list)
    started_at:    str = ""
    finished_at:   str = ""
    output_dir:    Optional[Path] = None
    cancelled:     bool = False

    def to_dict(self) -> dict:
        return {
            "experiment":  self.experiment.to_dict(),
            "started_at":  self.started_at,
            "finished_at": self.finished_at,
            "cancelled":   self.cancelled,
            "folds":       [asdict(f) for f in self.folds],
            "aggregate":   self.aggregate(),
            "aggregate_confusion": {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in self.aggregate_confusion().items()
            },
        }

    def aggregate(self) -> Dict[str, Dict[str, float]]:
        """Mean ± std per model across folds."""
        out: Dict[str, Dict[str, float]] = {}
        if not self.folds:
            return out
        by_model: Dict[str, List[FoldResult]] = {}
        for f in self.folds:
            by_model.setdefault(f.model_type, []).append(f)
        for model, fs in by_model.items():
            accs = np.array([f.accuracy for f in fs], dtype=float)
            f1s  = np.array([f.macro_f1 for f in fs], dtype=float)
            train_secs = np.array([f.train_seconds for f in fs], dtype=float)

            # Per-class F1 averaged across folds, keyed by class name.
            # Union of class names seen across folds; missing values are
            # excluded from the mean rather than counted as zero (that
            # would punish models evaluated on folds where the class
            # simply wasn't present).
            class_names: set = set()
            for f in fs:
                class_names.update(f.per_class_f1.keys())
            per_class_mean: Dict[str, float] = {}
            for cn in sorted(class_names):
                vals = [f.per_class_f1[cn] for f in fs if cn in f.per_class_f1]
                if vals:
                    per_class_mean[cn] = float(np.mean(vals))

            out[model] = {
                "n_folds":        len(fs),
                "accuracy_mean":  float(accs.mean()),
                "accuracy_std":   float(accs.std(ddof=0)),
                "macro_f1_mean":  float(f1s.mean()),
                "macro_f1_std":   float(f1s.std(ddof=0)),
                "train_seconds_mean": float(train_secs.mean()),
                "per_class_f1":   per_class_mean,
            }
        return out

    def aggregate_confusion(self) -> Dict[str, np.ndarray]:
        """
        Sum of per-fold confusion matrices, per model.

        Aligns matrices across folds by class-label union, padding with
        zero rows/columns when a fold didn't see every class. Returns
        `{model_type: (matrix, labels)}` collapsed to just the matrix
        for JSON friendliness — call `aggregate_confusion_labels()` for
        the matching label list.
        """
        out: Dict[str, np.ndarray] = {}
        by_model: Dict[str, List[FoldResult]] = {}
        for f in self.folds:
            if f.confusion is not None and f.confusion_labels is not None:
                by_model.setdefault(f.model_type, []).append(f)
        for model, fs in by_model.items():
            all_labels = sorted({l for f in fs for l in f.confusion_labels})
            if not all_labels:
                continue
            idx = {l: i for i, l in enumerate(all_labels)}
            acc = np.zeros((len(all_labels), len(all_labels)), dtype=np.int64)
            for fr in fs:
                cm = np.asarray(fr.confusion, dtype=np.int64)
                for i, li in enumerate(fr.confusion_labels):
                    for j, lj in enumerate(fr.confusion_labels):
                        acc[idx[li], idx[lj]] += cm[i, j]
            out[model] = acc
        return out

    def aggregate_confusion_labels(self) -> Dict[str, List[int]]:
        """Label list matching the matrices from aggregate_confusion()."""
        out: Dict[str, List[int]] = {}
        by_model: Dict[str, List[FoldResult]] = {}
        for f in self.folds:
            if f.confusion_labels is not None:
                by_model.setdefault(f.model_type, []).append(f)
        for model, fs in by_model.items():
            out[model] = sorted({l for f in fs for l in f.confusion_labels})
        return out


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class ValidationRunner:
    """
    Execute one :class:`ExperimentConfig` end-to-end.

    Parameters
    ----------
    data_dir : Path
        Root pipeline data dir (the one that contains ``sessions/``).
    output_root : Path
        Where to write the timestamped result folder. Defaults to
        ``<data_dir>/validation_runs``.
    """

    def __init__(self, data_dir: Path, output_root: Optional[Path] = None):
        self.data_dir = Path(data_dir)
        self.output_root = Path(output_root) if output_root else (self.data_dir / "validation_runs")
        self.corpus = SessionCorpus(self.data_dir)

    # ------------------------------------------------------------------

    def run(
        self,
        cfg: ExperimentConfig,
        progress: Optional[ProgressReporter] = None,
    ) -> RunResult:
        prog = progress or NoopProgress()
        result = RunResult(experiment=cfg, started_at=_now_iso())
        result.output_dir = self._make_output_dir(cfg)

        # Persist config + environment up front so an interrupted run
        # still leaves enough information to know what was attempted.
        dump_experiment(cfg, result.output_dir / "experiment.json")
        _write_json(result.output_dir / "environment.json", _capture_environment())

        records = self._select_sessions(cfg)
        _write_json(
            result.output_dir / "session_index.json",
            {"records": [r.to_dict() for r in records]},
        )

        if not records:
            prog.log("No sessions matched the data filter — nothing to do.")
            result.finished_at = _now_iso()
            self._persist_results(result)
            prog.on_run_done(result)
            return result

        strategy = get_strategy(cfg.cv.strategy)
        folds = list(strategy(records, **cfg.cv.kwargs))
        n_folds = len(folds)
        n_models = max(1, len(cfg.models))
        total_evals = n_folds * n_models

        prog.log(
            f"{n_folds} fold(s) × {n_models} model(s) = {total_evals} evaluation(s)"
        )
        prog.on_run_start(n_folds, n_models, records)

        eval_idx = 0
        for model_cfg in cfg.models:
            for fold in folds:
                if prog.should_cancel():
                    prog.log("Cancellation requested — stopping.")
                    result.cancelled = True
                    break

                eval_idx += 1
                fold_id = str(fold.get("id", f"fold_{eval_idx}"))
                prog.on_fold_start(eval_idx, total_evals, fold_id, model_cfg.type)

                fr = self._run_one_fold(cfg, model_cfg, fold, eval_idx)
                if fr is not None:
                    result.folds.append(fr)
                    prog.on_fold_done(eval_idx, total_evals, fr)
                    prog.log(
                        f"  [{eval_idx}/{total_evals}] {model_cfg.type} · {fold_id} · "
                        f"acc {fr.accuracy:.3f}  F1 {fr.macro_f1:.3f}  "
                        f"({fr.train_seconds:.1f}s)"
                    )
                else:
                    prog.log(f"  [{eval_idx}/{total_evals}] {model_cfg.type} · {fold_id} · SKIPPED")
            if result.cancelled:
                break

        result.finished_at = _now_iso()
        self._persist_results(result)
        prog.on_run_done(result)
        return result

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _select_sessions(self, cfg: ExperimentConfig) -> List[SessionRecord]:
        ds = cfg.data
        if ds.explicit:
            wanted = set(ds.explicit)
            return [
                r for r in self.corpus.all()
                if f"{r.subject_id}/{r.session_id}" in wanted
            ]
        return self.corpus.filter(
            subjects=ds.subjects,
            domains=ds.domains,
            min_channels=ds.min_channels,
            sampling_rate=ds.sampling_rate,
        )

    def _run_one_fold(
        self,
        cfg: ExperimentConfig,
        model_cfg,
        fold: Dict[str, Any],
        eval_idx: int,
    ) -> Optional[FoldResult]:
        # Deterministic per-fold seed.
        fold_idx = int(fold.get("idx", eval_idx))
        seed = _fold_seed(cfg.seed, fold_idx, model_cfg.type)
        _seed_everything(seed)

        try:
            (X_train, y_train,
             X_val,   y_val,
             X_test,  y_test,
             label_names) = self._materialise_fold(
                cfg, fold,
                raw_windows=_model_wants_raw_windows(model_cfg.type),
            )
        except Exception as e:  # noqa: BLE001
            log.exception("Failed to materialise fold %s: %s", fold.get("id"), e)
            return None

        if len(X_train) == 0 or len(X_test) == 0:
            log.warning("Fold %s has empty train/test set; skipping.", fold.get("id"))
            return None

        t0 = time.time()
        try:
            model, train_meta = self._fit_model(
                cfg, model_cfg, seed,
                X_train, y_train,
                X_val if len(X_val) > 0 else None,
                y_val if len(X_val) > 0 else None,
            )
        except Exception as e:  # noqa: BLE001
            log.exception("Model fit failed for fold %s: %s", fold.get("id"), e)
            return None
        train_secs = time.time() - t0

        try:
            metrics = self._evaluate(model, X_test, y_test, label_names)
        except Exception as e:  # noqa: BLE001
            log.exception("Evaluation failed for fold %s: %s", fold.get("id"), e)
            return None

        # Optional val metrics (cheaper than test: we just reuse _evaluate).
        val_acc = val_f1 = None
        if len(X_val) > 0:
            try:
                val_metrics = self._evaluate(model, X_val, y_val, label_names)
                val_acc = val_metrics["accuracy"]
                val_f1  = val_metrics["macro_f1"]
            except Exception as e:  # noqa: BLE001
                log.warning("Val evaluation failed for fold %s: %s", fold.get("id"), e)

        return FoldResult(
            fold_id=str(fold["id"]),
            model_type=model_cfg.type,
            n_train_windows=int(len(X_train)),
            n_test_windows=int(len(X_test)),
            n_val_windows=int(len(X_val)),
            accuracy=metrics["accuracy"],
            macro_f1=metrics["macro_f1"],
            per_class_f1=metrics["per_class_f1"],
            val_accuracy=val_acc,
            val_macro_f1=val_f1,
            train_seconds=train_secs,
            inference_ms=metrics["inference_ms"],
            confusion=metrics["confusion"],
            confusion_labels=metrics["confusion_labels"],
            label_names=label_names,
            fold_seed=seed,
            extra={"train_meta": _slim_train_meta(train_meta)},
        )

    # ------------------------------------------------------------------
    # Glue: materialise a fold via DataManager.create_dataset
    # ------------------------------------------------------------------

    def _materialise_fold(
        self,
        cfg: ExperimentConfig,
        fold: Dict[str, Any],
        raw_windows: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray,
               np.ndarray, np.ndarray,
               np.ndarray, np.ndarray,
               Dict[int, str]]:
        """
        Build (X_train, y_train, X_val, y_val, X_test, y_test, label_names).

        Parameters
        ----------
        raw_windows : bool
            When True, skip feature extraction and return the raw
            windowed tensor. CNN-family deep models crash on 2-D
            feature input, so the runner sets this per-model. Classical
            models and MLP leave it False and get the feature-extracted
            view.

        For folds that don't define a separate validation set the val
        arrays are returned empty (shape ``(0,)``). Caller checks
        ``len()`` before forwarding them to the model.

        Reuses ``DataManager.create_dataset`` so the windowing /
        per-session-rotation / bad-channel / feature-extraction code
        path is identical to the one the GUI Train tab uses.

        Mixed-channel-count folds
        ─────────────────────────
        ``DataManager.create_dataset`` rejects sessions with different
        channel counts.  When a LOSO fold contains sessions recorded
        with different hardware (16-ch vs 32-ch), ``_filter_by_channels``
        keeps only the sessions whose channel count matches the test
        split (the split being evaluated).  Dropped sessions are
        logged at WARNING level so the decision is visible in the run
        log.
        """
        from playagain_pipeline.core.data_manager import DataManager  # lazy
        dm = DataManager(self.data_dir)

        feature_config = None if raw_windows else self._build_feature_config(cfg)

        def _empty():
            return np.empty((0,)), np.empty((0,), dtype=np.int64), {}

        def _load_sessions(records):
            sess = []
            for rec in records:
                try:
                    sess.append(dm.load_session(rec.subject_id, rec.session_id))
                except Exception as e:  # noqa: BLE001
                    log.warning("Could not load %s/%s: %s",
                                rec.subject_id, rec.session_id, e)
            return sess

        def _dominant_channel_count(records: list) -> Optional[int]:
            from collections import Counter
            counts = Counter(r.num_channels for r in records if r.num_channels > 0)
            if not counts:
                return None
            return counts.most_common(1)[0][0]

        all_fold_records = (
            list(fold.get("train") or [])
            + list(fold.get("val") or [])
            + list(fold.get("test") or [])
        )
        ref_ch = _dominant_channel_count(list(fold.get("test") or [])) \
            or _dominant_channel_count(all_fold_records)

        def _filter_by_channels(records: list) -> list:
            if ref_ch is None:
                return records
            kept, dropped = [], []
            for r in records:
                if r.num_channels == 0 or r.num_channels == ref_ch:
                    kept.append(r)
                else:
                    dropped.append(r)
            if dropped:
                log.warning(
                    "Fold %s: dropping %d session(s) with %s channels "
                    "(reference channel count is %d)",
                    fold.get("id"), len(dropped),
                    sorted({r.num_channels for r in dropped}), ref_ch,
                )
            return kept

        # Per-split temporary dataset names need to be unique across
        # parallel calls — include the fold id so retries don't collide
        # with an in-flight build of the same fold. Tag raw-window
        # materialisations separately so a fold's feature-view and
        # raw-view don't stomp on each other when the runner iterates
        # both classical and deep models over the same fold.
        view_tag = "raw" if raw_windows else "feat"
        safe_fold_id = "".join(
            c if (c.isalnum() or c in "._-") else "_"
            for c in str(fold.get("id", "f"))
        )

        def _make_xy(records, name_suffix: str):
            sessions = _load_sessions(_filter_by_channels(records))
            if not sessions:
                return _empty()
            ds = dm.create_dataset(
                name=f"_validation_tmp_{safe_fold_id}_{view_tag}_{name_suffix}",
                sessions=sessions,
                window_size_ms=cfg.windowing.window_ms,
                window_stride_ms=cfg.windowing.stride_ms,
                feature_config=feature_config,
                use_per_session_rotation=False,
            )
            return ds["X"], ds["y"], ds["metadata"].get("label_names", {})

        # ── Within-session (temporal tail) path ────────────────────────
        # This is deliberately a TEMPORAL split — the last `test_fraction`
        # of each session's windows is held out. That's the intended
        # semantics of "within-session" evaluation (drift / fatigue
        # robustness).
        #
        # Caveat: because the pipeline stores windows trial-by-trial,
        # the temporal tail often contains only the last few gesture
        # classes. We log a warning if the class distribution between
        # head and tail is badly skewed so users know to treat the
        # number with care.
        is_within = fold.get("split_kind") == "temporal_tail"
        if is_within:
            X_all, y_all, label_names = _make_xy(fold["train"], "within")
            if len(X_all) == 0:
                X_empty, y_empty, _ = _empty()
                return (X_empty, y_empty,
                        X_empty, y_empty,
                        X_empty, y_empty,
                        label_names)
            test_frac = float(fold.get("test_fraction", 0.2))
            cut = int(len(X_all) * (1.0 - test_frac))
            X_tr, y_tr = X_all[:cut], y_all[:cut]
            X_te, y_te = X_all[cut:], y_all[cut:]
            _warn_if_class_skewed(y_tr, y_te, fold.get("id"), label_names)
            X_va, y_va, _ = _empty()
            return X_tr, y_tr, X_va, y_va, X_te, y_te, label_names

        X_tr, y_tr, names_tr = _make_xy(fold["train"], "train")
        X_te, y_te, names_te = _make_xy(fold["test"],  "test")

        val_records = fold.get("val") or []
        if val_records:
            X_va, y_va, names_va = _make_xy(val_records, "val")
        else:
            X_va, y_va, names_va = _empty()

        label_names = {**names_tr, **names_va, **names_te}
        return X_tr, y_tr, X_va, y_va, X_te, y_te, label_names

    @staticmethod
    def _build_feature_config(cfg: ExperimentConfig) -> Optional[Dict[str, Any]]:
        """Translate ExperimentConfig.features into the dict that
        DataManager.create_dataset expects (or None for raw windows)."""
        if not cfg.features:
            return None
        return {
            "mode": "features",
            "features": [
                {"name": f.name, "params": f.params}
                for f in cfg.features
            ],
        }

    # ------------------------------------------------------------------
    # Model fit / eval
    # ------------------------------------------------------------------

    def _fit_model(self, cfg: ExperimentConfig, model_cfg, seed: int,
                   X_train, y_train, X_val=None, y_val=None):
        """
        Fit one model.

        Path A — explicit val provided (e.g. holdout_split)
            Call model.train(X_train, y_train, X_val, y_val, ...)
            directly so deep models can use the real held-out set for
            early stopping and learning-rate scheduling.

        Path B — no val provided
            Reuse ModelManager.train_model which does an internal
            80/20 stratified split of the training data. We pass the
            fold seed through so that split is deterministic.
        """
        from playagain_pipeline.models.classifier import ModelManager  # lazy

        mm = ModelManager(self.data_dir / "models")
        model = mm.create_model(
            model_cfg.type,
            name=f"_validation_{model_cfg.type}",
            **(model_cfg.params or {}),
        )

        # Sampling rate lives on the session metadata; we assume every
        # session in a fold shares it (the corpus filter guarantees this
        # when `sampling_rate` is pinned on DataSelection).
        sr = self._sampling_rate_for_fold(cfg)
        n_ch = int(X_train.shape[-1]) if X_train.ndim >= 2 else 0

        if X_val is not None and len(X_val) > 0:
            try:
                train_meta = model.train(
                    X_train, y_train,
                    X_val, y_val,
                    window_size_ms=cfg.windowing.window_ms,
                    sampling_rate=sr,
                    num_channels=n_ch,
                    random_state=seed,
                )
            except TypeError:
                # Older classifier.train signatures don't accept
                # random_state; fall back gracefully.
                train_meta = model.train(
                    X_train, y_train, X_val, y_val,
                    window_size_ms=cfg.windowing.window_ms,
                    sampling_rate=sr,
                    num_channels=n_ch,
                )
            return model, train_meta

        # No val supplied → reuse ModelManager.train_model so behaviour
        # is byte-identical to the Train tab. Pass seed through if the
        # installed ModelManager supports it (v1 did not; the v2 patch
        # adds it).
        train_kwargs = {
            "save": False,
        }
        try:
            train_meta = mm.train_model(
                model,
                {
                    "X": X_train,
                    "y": y_train,
                    "metadata": {
                        "name": f"_validation_{model_cfg.type}",
                        "window_size_ms": cfg.windowing.window_ms,
                        "sampling_rate": sr,
                        "num_channels": n_ch,
                    },
                },
                random_state=seed,
                **train_kwargs,
            )
        except TypeError:
            train_meta = mm.train_model(
                model,
                {
                    "X": X_train,
                    "y": y_train,
                    "metadata": {
                        "name": f"_validation_{model_cfg.type}",
                        "window_size_ms": cfg.windowing.window_ms,
                        "sampling_rate": sr,
                        "num_channels": n_ch,
                    },
                },
                **train_kwargs,
            )
        return model, train_meta

    def _sampling_rate_for_fold(self, cfg: ExperimentConfig) -> int:
        """Best-effort sampling rate: selection pin → corpus median → 2000."""
        if cfg.data.sampling_rate:
            return int(cfg.data.sampling_rate)
        try:
            rates = [int(r.sampling_rate) for r in self.corpus.all()
                     if r.sampling_rate]
            if rates:
                rates.sort()
                return rates[len(rates) // 2]
        except Exception:  # noqa: BLE001
            pass
        return 2000

    def _evaluate(self, model, X_test, y_test, label_names):
        from sklearn.metrics import (
            f1_score, accuracy_score, confusion_matrix,
        )
        t0 = time.time()
        y_pred = model.predict(X_test)
        inf_ms = (time.time() - t0) * 1000.0 / max(len(X_test), 1)

        acc = float(accuracy_score(y_test, y_pred))
        f1m = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
        labels = sorted(set(int(v) for v in y_test) | set(int(v) for v in y_pred))
        per_class = f1_score(y_test, y_pred, labels=labels,
                             average=None, zero_division=0)
        per_class_named = {
            label_names.get(int(lbl), str(lbl)): float(score)
            for lbl, score in zip(labels, per_class)
        }
        cm = confusion_matrix(y_test, y_pred, labels=labels).tolist()
        return {
            "accuracy":         acc,
            "macro_f1":         f1m,
            "per_class_f1":     per_class_named,
            "inference_ms":     inf_ms,
            "confusion":        cm,
            "confusion_labels": [int(l) for l in labels],
        }

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def _make_output_dir(self, cfg: ExperimentConfig) -> Path:
        ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        safe_name = "".join(
            c if (c.isalnum() or c in "._-") else "_" for c in cfg.name
        )
        out = self.output_root / f"{ts}__{safe_name}"
        out.mkdir(parents=True, exist_ok=True)
        return out

    def _persist_results(self, result: RunResult) -> None:
        if result.output_dir is None:
            return
        _write_json(result.output_dir / "results.json", result.to_dict())
        _write_csv(result.output_dir / "results.csv", result.folds)
        # Dedicated per-class file, easier to pivot in pandas / Excel.
        _write_per_class_csv(
            result.output_dir / "per_class_f1.csv",
            result.folds,
        )
        log.info("Wrote validation run to %s", result.output_dir)


# ---------------------------------------------------------------------------
# Free helpers
# ---------------------------------------------------------------------------

def _fold_seed(base_seed: int, fold_idx: int, model_type: str) -> int:
    """Deterministic seed for one (fold, model) unit of work."""
    raw = f"{int(base_seed)}|{int(fold_idx)}|{str(model_type)}".encode()
    h = hashlib.sha1(raw).digest()
    # Keep it in the safe int32 range — some libraries reject larger.
    return int.from_bytes(h[:4], "big") & 0x7FFFFFFF


def _seed_everything(seed: int) -> None:
    """Seed all of numpy / stdlib random / torch deterministically."""
    _random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Note: we deliberately do NOT set torch.backends.cudnn.deterministic
        # = True here — it slows things down significantly and we don't
        # promise bit-exact reproducibility on CUDA.
    except ImportError:
        pass


def _slim_train_meta(meta: Optional[dict]) -> dict:
    """
    Compact representation of training metadata suitable for JSON dump.

    Full epoch curves (`training_history`, `history`) are reduced to a
    summary with num_epochs + final metrics; everything else is passed
    through.
    """
    if not meta:
        return {}
    out: Dict[str, Any] = {}
    for k, v in meta.items():
        if k in ("training_history", "history") and isinstance(v, dict):
            def _last(seq):
                return seq[-1] if seq else None
            out[f"{k}_summary"] = {
                "num_epochs":        len(v.get("train_loss", []) or []),
                "final_train_loss":  _last(v.get("train_loss") or []),
                "final_val_loss":    _last(v.get("val_loss") or []),
                "final_train_acc":   _last(v.get("train_acc") or []),
                "final_val_acc":     _last(v.get("val_acc") or []),
            }
        else:
            out[k] = v
    return out


def _warn_if_class_skewed(
    y_tr: np.ndarray,
    y_te: np.ndarray,
    fold_id: Any,
    label_names: Dict[int, str],
) -> None:
    """Log a warning when within-session tail misses classes the head had."""
    if len(y_tr) == 0 or len(y_te) == 0:
        return
    tr_classes = set(int(v) for v in np.unique(y_tr))
    te_classes = set(int(v) for v in np.unique(y_te))
    missing = tr_classes - te_classes
    extra = te_classes - tr_classes
    if missing or extra:
        log.warning(
            "Within-session fold %s has skewed class distribution "
            "(train-only classes: %s; test-only classes: %s). "
            "Temporal tail splits don't rebalance — treat this fold's "
            "F1 with care.",
            fold_id,
            [label_names.get(c, str(c)) for c in sorted(missing)] or "none",
            [label_names.get(c, str(c)) for c in sorted(extra)] or "none",
        )


# ---------------------------------------------------------------------------
# Reproducibility helpers
# ---------------------------------------------------------------------------

def _capture_environment() -> dict:
    env: Dict[str, Any] = {
        "captured_at":   _now_iso(),
        "python":        sys.version,
        "platform":      platform.platform(),
        "executable":    sys.executable,
    }

    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            cwd=Path(__file__).parent,
        ).decode().strip()
        env["git_sha"] = sha
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
            cwd=Path(__file__).parent,
        ).decode().strip()
        env["git_dirty"] = bool(dirty)
    except Exception:  # noqa: BLE001
        env["git_sha"] = None
        env["git_dirty"] = None

    versions: Dict[str, str] = {}
    for pkg in ("numpy", "scipy", "scikit-learn", "catboost",
                "torch", "pandas", "PySide6"):
        try:
            mod = __import__(pkg.replace("-", "_"))
            versions[pkg] = getattr(mod, "__version__", "unknown")
        except Exception:  # noqa: BLE001
            versions[pkg] = "not-installed"
    env["package_versions"] = versions
    return env


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)


def _write_csv(path: Path, folds: List[FoldResult]) -> None:
    """Flat per-fold CSV — handy for pandas / Excel inspection."""
    if not folds:
        return
    fieldnames = [
        "fold_id", "model_type",
        "n_train_windows", "n_val_windows", "n_test_windows",
        "accuracy", "macro_f1",
        "val_accuracy", "val_macro_f1",
        "train_seconds", "inference_ms",
        "fold_seed",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for fr in folds:
            w.writerow({k: getattr(fr, k) for k in fieldnames})


def _write_per_class_csv(path: Path, folds: List[FoldResult]) -> None:
    """Long-format per-class F1 — one row per (fold, model, class)."""
    if not folds:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["fold_id", "model_type", "class", "f1"])
        for fr in folds:
            for cls, score in fr.per_class_f1.items():
                w.writerow([fr.fold_id, fr.model_type, cls, f"{score:.6f}"])