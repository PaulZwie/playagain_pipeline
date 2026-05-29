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
import os
import platform
import random as _random
import subprocess
import sys
import time
import threading
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


# Per-model wall-clock budgets for a single fold's fit phase. When the
# heartbeat thread crosses one of these it escalates from a plain "still
# fitting" message to a "this looks slow, consider cancelling" hint.
# Tuned so the budget is comfortably above a healthy run on the same
# data — anything past it is genuinely abnormal and worth flagging.
_FIT_BUDGET_SECONDS: Dict[str, float] = {
    "lda":            60.0,
    "random_forest":  180.0,
    "catboost":       300.0,
    "mlp":            600.0,
    "svm":            600.0,        # 10 min — RBF on 20k+ samples is fine; >10 min is not
    "attention_net":  900.0,
    "cnn":            900.0,
    "mstnet":         900.0,
}


def _fit_budget_seconds(model_type: str) -> float:
    return _FIT_BUDGET_SECONDS.get(str(model_type).lower(), 600.0)


class _FitHeartbeat(threading.Thread):
    """
    Daemon thread that periodically logs "still fitting, elapsed X" so
    the user has live feedback during long sklearn fits that hold the
    GIL (libsvm RBF in particular). Self-terminates when ``stop()`` is
    called.

    Two reasons this isn't just a Qt timer:
      • the runner runs in a worker thread, but it's not strictly a
        Qt worker — non-GUI callers (CLI, tests) need feedback too;
      • the work being measured is blocking C code holding the GIL,
        so the only thread that can post status updates is one we
        start *before* the blocking call.
    """

    # Heartbeat cadence. Short enough that the user sees movement
    # within a minute; long enough that the log doesn't get spammed
    # for a model that finishes fine in 90 seconds.
    INTERVAL_SECONDS = 30.0

    def __init__(
        self,
        prog: "ProgressReporter",
        eval_idx: int,
        total_evals: int,
        fold_id: str,
        model_type: str,
        budget_seconds: float,
    ):
        super().__init__(daemon=True, name=f"fit-heartbeat-{model_type}")
        self._prog = prog
        self._eval_idx = eval_idx
        self._total_evals = total_evals
        self._fold_id = fold_id
        self._model_type = model_type
        self._budget = float(budget_seconds)
        self._t0 = time.time()
        self._stop = threading.Event()
        self._budget_warned = False

    def stop(self) -> None:
        self._stop.set()
        # Don't join() — the worker is a daemon and we just want to
        # stop the next tick. Joining would block the fit-complete
        # path waiting up to INTERVAL_SECONDS.

    def run(self) -> None:
        # Skip the first tick so quick fits never produce noise.
        if self._stop.wait(self.INTERVAL_SECONDS):
            return
        while not self._stop.is_set():
            elapsed = time.time() - self._t0
            self._emit(elapsed)
            # Re-arm; bail out immediately if stop() is called between
            # ticks. wait() returns True when the event is set.
            if self._stop.wait(self.INTERVAL_SECONDS):
                return

    def _emit(self, elapsed: float) -> None:
        try:
            self._prog.on_fold_phase(
                self._eval_idx, self._total_evals,
                self._fold_id, self._model_type,
                "fit",
                f"still fitting · {_fmt_dur(elapsed)}",
            )
            self._prog.log(
                f"  · still fitting {self._model_type} on {self._fold_id} "
                f"· {_fmt_dur(elapsed)}"
            )
            if elapsed > self._budget and not self._budget_warned:
                self._budget_warned = True
                # One-shot escalation — a clear, actionable suggestion
                # the user can act on without having to know what
                # libsvm's complexity class is. Subsequent ticks fall
                # back to the regular elapsed-time line.
                self._prog.log(self._budget_hint(elapsed))
        except Exception:  # noqa: BLE001
            # Never let a logging failure crash the heartbeat. We
            # really do want to ignore everything here.
            pass

    def _budget_hint(self, elapsed: float) -> str:
        mt = self._model_type.lower()
        if mt == "svm":
            return (
                f"  ⚠ {self._model_type} on {self._fold_id} has been "
                f"fitting for {_fmt_dur(elapsed)}, well past the "
                f"{_fmt_dur(self._budget)} budget. RBF SVMs are O(N²); "
                f"if you didn't set max_train_samples, the runner now "
                f"defaults it to 20 000 — but a previously-started "
                f"run won't pick that up. Consider cancelling and "
                f"restarting, or skipping SVM for this suite."
            )
        if mt in {"cnn", "attention_net", "mstnet"}:
            return (
                f"  ⚠ {self._model_type} on {self._fold_id} has been "
                f"fitting for {_fmt_dur(elapsed)}, past the "
                f"{_fmt_dur(self._budget)} budget. Deep models can "
                f"genuinely take this long on CPU; check whether the "
                f"PyTorch backend picked up a GPU as expected."
            )
        return (
            f"  ⚠ {self._model_type} on {self._fold_id} has been "
            f"fitting for {_fmt_dur(elapsed)}, past the "
            f"{_fmt_dur(self._budget)} budget. If this seems wrong "
            f"you can cancel and re-queue with fewer sessions."
        )


# Heuristics for "hey the next fit may take a while" messages. Tuned
# from real LOSO runs: SVM(RBF) is dominated by the kernel matrix
# build, CatBoost rebuilds its internal data caches, and the
# deep-net models have to lazily allocate GPU/CPU tensors and
# re-window. None of these are bugs; they're just unhelpfully quiet.
_SLOW_FIRST_FOLD_MODELS = {
    "svm":           "RBF SVM is O(N²); validation runs default to "
                     "max_train_samples=20 000 + probability=False to "
                     "keep folds tractable. Even so, expect a few "
                     "minutes per fold on large LOSO splits.",
    "catboost":      "CatBoost rebuilds its internal data layout on "
                     "every fit; first fold is the slowest.",
    "cnn":           "Raw windows aren't cached for deep models — the "
                     "first fold re-extracts every session.",
    "attention_net": "Raw windows aren't cached for deep models — the "
                     "first fold re-extracts every session.",
    "mstnet":        "Raw windows aren't cached for deep models — the "
                     "first fold re-extracts every session.",
    "mlp":           "Feature mode is cached, but the first PyTorch "
                     "import + tensor allocation has some warm-up cost.",
}


def _first_fold_hint(model_type: str) -> Optional[str]:
    """Return a 'first fold may be slow' message, or None."""
    return _SLOW_FIRST_FOLD_MODELS.get(str(model_type).lower())


def _fmt_dur(seconds: float) -> str:
    """Pretty duration: 13.4s, 2m 41s, 1h 04m."""
    s = max(0.0, float(seconds))
    if s < 60:
        return f"{s:.1f}s"
    m, rs = divmod(int(round(s)), 60)
    if m < 60:
        return f"{m}m {rs:02d}s"
    h, rm = divmod(m, 60)
    return f"{h}h {rm:02d}m"


def _shape_brief(arr) -> str:
    """One-line shape descriptor used in phase log lines."""
    try:
        if arr.ndim == 2:
            return f"{arr.shape[1]} features"
        if arr.ndim == 3:
            return f"{arr.shape[1]}×{arr.shape[2]} window"
        return f"ndim={arr.ndim}"
    except Exception:  # noqa: BLE001
        return "shape=?"


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

    def on_model_start(self, model_idx: int, total_models: int,
                       model_type: str, total_folds: int) -> None: ...

    def on_fold_start(self, fold_idx: int, total_evals: int,
                      fold_id: str, model_type: str) -> None: ...

    def on_fold_phase(self, fold_idx: int, total_evals: int,
                      fold_id: str, model_type: str,
                      phase: str, detail: str = "") -> None:
        """
        Fired between fold milestones (materialise → fit → evaluate)
        so the UI can show movement during long-running phases such as
        SVM kernel fitting on 100 k+ samples. Optional; default no-op.
        """
        ...

    def on_fold_done(self, fold_idx: int, total_evals: int,
                     fold_result: "FoldResult") -> None: ...

    def on_model_done(self, model_idx: int, total_models: int,
                      model_type: str, n_folds_completed: int,
                      total_seconds: float) -> None: ...

    def on_run_done(self, result: "RunResult") -> None: ...

    def log(self, message: str) -> None: ...

    def should_cancel(self) -> bool: ...


class NoopProgress:
    """Default reporter — silent, never cancels."""

    def on_run_start(self, *_a, **_k): pass
    def on_model_start(self, *_a, **_k): pass
    def on_fold_start(self, *_a, **_k): pass
    def on_fold_phase(self, *_a, **_k): pass
    def on_fold_done(self, *_a, **_k): pass
    def on_model_done(self, *_a, **_k): pass
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

    test_subjects: List[str] = field(default_factory=list)
    test_sessions: List[Tuple[str, str]] = field(default_factory=list)  # (subject, session_id)
    split_kind: str = ""

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
        # Lazily-constructed per-runner feature cache. Lives under
        # <data_dir>/.feature_cache/ and survives across runs in the
        # same process; suite-style use (thesis report) gets the
        # biggest payoff because every fold of every preset shares
        # the same per-session feature blobs.
        self._feature_cache = None
        # Lazily-discovered game-recording corpus used by the
        # session_to_game CV strategy. Populated on first call to
        # _materialise_game_test_split.
        self._game_test_recordings: Optional[List[Any]] = None

    def _get_or_make_cache(self):
        """Return the lazily-constructed FeatureCache for this runner."""
        if self._feature_cache is None:
            # Local import keeps the package import-light when callers
            # never touch _materialise_fold (e.g. report aggregation only).
            from .feature_cache import FeatureCache
            enabled = os.environ.get(
                "PLAYAGAIN_FEATURE_CACHE", "1",
            ).strip().lower() not in {"0", "off", "false", "no"}
            self._feature_cache = FeatureCache(self.data_dir, enabled=enabled)
            if enabled:
                log.info(
                    "Feature cache enabled at %s "
                    "(set PLAYAGAIN_FEATURE_CACHE=0 to disable).",
                    self._feature_cache.root,
                )
            else:
                log.info("Feature cache disabled via env.")
        return self._feature_cache

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
        # Track per-model timing so we can report a model-level summary
        # at on_model_done — useful when one model (e.g. SVM with RBF
        # on 100k+ samples) is many minutes per fold and the user
        # otherwise sees no movement between fold-complete messages.
        for model_idx, model_cfg in enumerate(cfg.models, start=1):
            model_t0 = time.time()
            model_folds_done = 0
            model_type = str(model_cfg.type)
            prog.on_model_start(model_idx, n_models, model_type, n_folds)
            prog.log(
                f"━━ model {model_idx}/{n_models}: {model_type} "
                f"({n_folds} fold{'s' if n_folds != 1 else ''}) ━━"
            )
            # First-fold warning for models that are known to be slow on
            # the first fit — gives the user something to read while the
            # process appears frozen.
            _hint = _first_fold_hint(model_type)
            if _hint:
                prog.log(f"  ℹ  {_hint}")

            for fold in folds:
                if prog.should_cancel():
                    prog.log("Cancellation requested — stopping.")
                    result.cancelled = True
                    break

                eval_idx += 1
                fold_id = str(fold.get("id", f"fold_{eval_idx}"))
                prog.on_fold_start(eval_idx, total_evals, fold_id, model_type)

                fr = self._run_one_fold(cfg, model_cfg, fold, eval_idx,
                                        progress=prog,
                                        total_evals=total_evals)
                if fr is not None:
                    result.folds.append(fr)
                    model_folds_done += 1
                    prog.on_fold_done(eval_idx, total_evals, fr)
                    prog.log(
                        f"  [{eval_idx}/{total_evals}] {model_type} · {fold_id} · "
                        f"acc {fr.accuracy:.3f}  F1 {fr.macro_f1:.3f}  "
                        f"({fr.train_seconds:.1f}s)"
                    )
                else:
                    prog.log(f"  [{eval_idx}/{total_evals}] {model_type} · {fold_id} · SKIPPED")

            model_secs = time.time() - model_t0
            prog.on_model_done(model_idx, n_models, model_type,
                               model_folds_done, model_secs)
            prog.log(
                f"━━ model {model_idx}/{n_models} {model_type} done · "
                f"{model_folds_done}/{n_folds} folds · "
                f"{_fmt_dur(model_secs)} total ━━"
            )
            if result.cancelled:
                break

        result.finished_at = _now_iso()
        # Cache stats — quietly informative; useful when one user complains
        # "the second run is suddenly fast" and another asks why their disk
        # filled up. Always printed at INFO so it shows in suite logs.
        if self._feature_cache is not None:
            s = self._feature_cache.stats()
            prog.log(
                f"Feature cache: {s['hits']} hits / {s['misses']} misses, "
                f"{s['writes']} new entries "
                f"({s['bytes_written'] / 1_048_576:.1f} MiB written)."
            )
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
        progress: Optional[ProgressReporter] = None,
        total_evals: int = 0,
    ) -> Optional[FoldResult]:
        # Deterministic per-fold seed.
        fold_idx = int(fold.get("idx", eval_idx))
        seed = _fold_seed(cfg.seed, fold_idx, model_cfg.type)
        _seed_everything(seed)

        fold_id = str(fold.get("id", f"fold_{eval_idx}"))
        model_type = str(model_cfg.type)
        prog = progress or NoopProgress()

        # Phase 1 — materialise data for this fold (cached for feature
        # mode, full re-window for raw-window deep models). Long-ish
        # for the very first fold of a new feature config; cheap after
        # that. Reporting the start so a user staring at the screen
        # during a 5-minute SVM fit sees that materialisation finished.
        t_mat = time.time()
        n_train_recs = len(fold.get("train") or [])
        n_test_recs  = len(fold.get("test") or [])
        prog.on_fold_phase(
            eval_idx, total_evals, fold_id, model_type,
            "materialise",
            f"loading + windowing {n_train_recs} train + {n_test_recs} test session(s)",
        )
        is_session_to_game = str(fold.get("split_kind", "")) == "session_to_game"
        try:
            (X_train, y_train,
             X_val,   y_val,
             X_test,  y_test,
             label_names) = self._materialise_fold(
                cfg, fold,
                raw_windows=_model_wants_raw_windows(model_type),
            )
        except Exception as e:  # noqa: BLE001
            log.exception("Failed to materialise fold %s: %s", fold_id, e)
            return None
        mat_secs = time.time() - t_mat

        if len(X_train) == 0:
            log.warning("Fold %s has empty train set; skipping.", fold_id)
            return None
        if len(X_test) == 0 and not is_session_to_game:
            log.warning("Fold %s has empty test set; skipping.", fold_id)
            return None

        # Phase 2 — fit. We announce sample/feature shape so a user
        # facing a long wait can sanity-check that it isn't surprising.
        # 100 k samples × 224 features with an RBF SVM is genuinely
        # slow and that's the most common "why is it stuck" scenario.
        prog.on_fold_phase(
            eval_idx, total_evals, fold_id, model_type,
            "fit",
            f"{len(X_train):,} train · {len(X_test):,} test · "
            f"{_shape_brief(X_train)}",
        )
        t0 = time.time()
        # Heartbeat thread — sklearn's RBF SVM fit holds the GIL inside
        # libsvm so we can't get fit-internal progress. What we *can*
        # do is post an elapsed-time line every minute so the user
        # sees the process is alive, and escalate to a "consider
        # cancelling" hint once we cross a model-specific budget.
        heartbeat = _FitHeartbeat(
            prog, eval_idx, total_evals, fold_id, model_type,
            budget_seconds=_fit_budget_seconds(model_type),
        )
        heartbeat.start()
        try:
            try:
                model, train_meta = self._fit_model(
                    cfg, model_cfg, seed,
                    X_train, y_train,
                    X_val if len(X_val) > 0 else None,
                    y_val if len(X_val) > 0 else None,
                )
            except Exception as e:  # noqa: BLE001
                log.exception("Model fit failed for fold %s: %s", fold_id, e)
                return None
        finally:
            heartbeat.stop()
        train_secs = time.time() - t0

        # session_to_game: test data is in data/game_recordings/, not
        # SessionCorpus. Window them now using the trained model's
        # feature path, deriving y from the CSV's ground-truth column.
        if is_session_to_game:
            try:
                X_test, y_test, label_names = self._materialise_game_test_split(
                    cfg, label_names,
                    raw_windows=_model_wants_raw_windows(model_type),
                )
            except Exception as e:  # noqa: BLE001
                log.exception("Failed to materialise game test split for "
                              "fold %s: %s", fold_id, e)
                return None
            if len(X_test) == 0:
                log.warning("Fold %s: no usable windows in game_recordings.",
                            fold_id)
                return None

        # Phase 3 — eval. Usually cheap, but worth a marker so the
        # GUI's stage display doesn't sit on "fitting" after fit ends.
        prog.on_fold_phase(
            eval_idx, total_evals, fold_id, model_type,
            "evaluate", f"scoring {len(X_test):,} test windows",
        )
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

        test_records = fold.get("test", []) or []
        if is_session_to_game:
            game_test = list(self._game_test_recordings or [])
            test_subjects = sorted({str(r.subject_id) for r in game_test})
            test_sessions = [(str(r.subject_id), str(r.session_id))
                             for r in game_test]
        else:
            test_subjects = sorted({r.subject_id for r in test_records})
            test_sessions = [(r.subject_id, r.session_id) for r in test_records]
        split_kind = str(fold.get("split_kind", ""))

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
            test_subjects=test_subjects,
            test_sessions=test_sessions,
            split_kind=split_kind,
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

        Feature cache
        ─────────────
        For non-raw (i.e. feature-extracted) folds, this method goes
        through :class:`FeatureCache` instead of rebuilding the
        feature view from scratch every fold. The cache keys on the
        full set of inputs that affect the output (windowing, feature
        config, bad-channel mode, per-session rotation flag, per-
        session bad channels/rotation/mapping, plus the data.npy
        mtime+size), so it self-invalidates when any of those change.
        Raw-window folds (deep models) bypass the cache and use the
        original code path — the same cache key would balloon to
        gigabytes per session and those models aren't the bottleneck.

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
        from .feature_cache import (                                  # lazy
            FeatureCache, materialise_split_with_cache,
        )
        dm = DataManager(self.data_dir)
        cache = self._get_or_make_cache()

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
            kept = _filter_by_channels(records)
            if not kept:
                return _empty()
            # Feature-mode folds go through the cache; raw-window folds
            # (CNN/TCN) fall back to the original session-batched path
            # via materialise_split_with_cache's internal fallback.
            X, y, names = materialise_split_with_cache(
                kept,
                cache=cache,
                data_manager=dm,
                window_ms=cfg.windowing.window_ms,
                stride_ms=cfg.windowing.stride_ms,
                feature_config=feature_config,
                use_per_session_rotation=False,
                bad_channel_mode="interpolate",
                name_suffix=f"{safe_fold_id}_{view_tag}_{name_suffix}",
            )
            if X.size == 0:
                return _empty()
            return X, y, names

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

    # ------------------------------------------------------------------
    # Game-recordings test split (session_to_game strategy)
    # ------------------------------------------------------------------

    def _materialise_game_test_split(
        self,
        cfg: ExperimentConfig,
        train_label_names: Dict[int, str],
        raw_windows: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[int, str]]:
        """
        Build (X_test, y_test, label_names) from ``data/game_recordings/``.

        Game CSVs already carry per-sample ground truth (``RawGroundTruth``
        is the numeric class id of the requested gesture, or -1 when no
        gesture is requested). We window each CSV with the same
        window/stride as training, align labels at window centres, and
        keep only frames where a gesture is actively requested.

        The trained model's class ids define the canonical label space —
        we re-map game-side class ids to that space by gesture name.
        """
        from playagain_pipeline.evaluation.loaders import (
            discover_game_recordings, load_game_csv,
        )
        from playagain_pipeline.models.classifier import EMGFeatureExtractor

        if self._game_test_recordings is None:
            self._game_test_recordings = discover_game_recordings(self.data_dir)
        recs = self._game_test_recordings or []
        if not recs:
            log.warning("No game recordings found under %s/game_recordings",
                        self.data_dir)
            return np.empty((0,)), np.empty((0,), dtype=np.int64), train_label_names

        feature_config = None if raw_windows else self._build_feature_config(cfg)
        extractor = EMGFeatureExtractor(feature_config) if (
            feature_config and feature_config.get("mode") not in (None, "raw")
        ) else None

        # Canonical name → training class id. The model was trained with
        # train_label_names so this is the only label space the model
        # predicts in.
        name_to_train_id = {
            str(name).strip().lower(): int(idx)
            for idx, name in train_label_names.items()
        }

        win_ms    = int(cfg.windowing.window_ms)
        stride_ms = int(cfg.windowing.stride_ms)

        pooled_X: List[np.ndarray] = []
        pooled_y: List[np.ndarray] = []
        n_kept_recs = 0

        for rec in recs:
            try:
                game = load_game_csv(rec)
            except Exception as exc:  # noqa: BLE001
                log.warning("Skipped game recording %s: %s", rec.label, exc)
                continue
            emg = game.emg_matrix()
            if emg.size == 0:
                continue
            fs = int(game.sampling_rate or self._sampling_rate_for_fold(cfg))
            win    = max(1, int(round(win_ms    * fs / 1000.0)))
            stride = max(1, int(round(stride_ms * fs / 1000.0)))
            if emg.shape[0] < win:
                continue

            starts = np.arange(0, emg.shape[0] - win + 1, stride, dtype=np.int64)
            if starts.size == 0:
                continue
            centres = starts + win // 2

            # Build ground truth aligned to window centres. Prefer
            # RawGroundTruth (multi-class numeric class id, -1 when no
            # gesture is requested) and re-map to the training label
            # space by gesture name via game.class_names.
            df = game.df
            keep = np.ones(centres.size, dtype=bool)
            if "RawGroundTruth" in df.columns:
                raw = df["RawGroundTruth"].to_numpy(dtype=np.int64)[centres]
                # Map game-side class id → gesture name → training id.
                game_cn = list(game.class_names or [])
                mapped = np.full(raw.size, -1, dtype=np.int64)
                for i, cid in enumerate(raw):
                    if cid < 0 or cid >= len(game_cn):
                        continue
                    name = str(game_cn[cid]).strip().lower()
                    tid = name_to_train_id.get(name, -1)
                    if tid >= 0:
                        mapped[i] = tid
                keep &= (mapped >= 0)
                y = mapped
            elif "RequestedGesture" in df.columns:
                req = (df["RequestedGesture"].astype(str).str.strip()
                       .str.lower().to_numpy())[centres]
                y = np.array(
                    [name_to_train_id.get(str(n), -1) for n in req],
                    dtype=np.int64,
                )
                keep &= (y >= 0)
            else:
                continue

            if keep.sum() == 0:
                continue

            X = np.stack([emg[s:s + win] for s in starts]).astype(np.float32)
            X = X[keep]
            y = y[keep]

            if extractor is not None:
                X = extractor.extract_features(X)

            pooled_X.append(X)
            pooled_y.append(y)
            n_kept_recs += 1

        if not pooled_X:
            return np.empty((0,)), np.empty((0,), dtype=np.int64), train_label_names

        log.info("game_recordings: %d/%d recording(s) yielded %d window(s)",
                 n_kept_recs, len(recs),
                 sum(len(x) for x in pooled_X))
        X_out = np.concatenate(pooled_X, axis=0)
        y_out = np.concatenate(pooled_y, axis=0)
        return X_out, y_out, train_label_names

    @staticmethod
    def _build_feature_config(cfg: ExperimentConfig) -> Optional[Dict[str, Any]]:
        """Translate ExperimentConfig.features into the dict that
        DataManager.create_dataset expects (or None for raw windows).

        Emits the bare-string ``custom`` form the extractor's custom
        branch reads natively. The older dict form
        (``{"name": ..., "params": ...}``) is still accepted post-fix
        for any caller that depends on it, but strings are the
        unambiguous shape — and the only one that correctly distinguishes
        a single-feature ablation row from a multi-feature run, since
        the previous output (``mode="features"``, dict entries) silently
        fell through to the extractor's default branch and returned the
        full six-feature stack regardless of ``cfg.features``.

        Per-feature params (e.g. ZC threshold) are forwarded through a
        parallel dict-form entry so the extractor's custom branch can
        pick them up; absent params let the extractor use its built-in
        defaults.
        """
        if not cfg.features:
            return None
        # Only emit a per-feature dict when at least one feature carries
        # non-empty params — keeps the common case (all defaults) as a
        # bare-string list that's trivially diff-friendly in config logs.
        any_params = any(
            bool(getattr(f, "params", None)) for f in cfg.features
        )
        if any_params:
            entries: List[Any] = [
                {"name": f.name, "params": dict(f.params or {})}
                for f in cfg.features
            ]
        else:
            entries = [f.name for f in cfg.features]
        return {
            "mode": "custom",
            "features": entries,
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

        # Validation-only sensible defaults the user can override via
        # ``model_cfg.params``. Right now this only matters for SVM:
        # the deployed-model defaults (probability=True, no sample cap)
        # turn a single LOSO fold into a multi-hour blocking operation
        # for no reportable benefit — the runner never asks the SVM
        # for predict_proba, and a stratified subsample is within
        # ~1 percentage point of the full-fit macro-F1.
        params = dict(model_cfg.params or {})
        if str(model_cfg.type).lower() == "svm":
            params.setdefault("probability", False)
            params.setdefault("cache_size", 1024.0)
            params.setdefault("max_train_samples", 20000)

        model = mm.create_model(
            model_cfg.type,
            name=f"_validation_{model_cfg.type}",
            **params,
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