"""
runner.py
─────────
The ValidationRunner ties everything together:

    1. Load an ExperimentConfig (YAML / JSON / dataclass).
    2. Discover sessions via SessionCorpus and apply the data filter.
    3. For every (model × CV-fold) combination:
         a. Extract windows from the train / test sessions.
         b. Compute the configured feature pipeline.
         c. Fit the model on train, evaluate on test.
         d. Record per-fold metrics.
    4. Aggregate across folds (mean ± std) and write everything to a
       timestamped result directory:

           validation_runs/
             2026-04-14_113022__loso_subject_baseline/
               experiment.json     ← exact config used
               environment.json    ← git SHA, python, package versions
               results.json        ← per-fold + aggregate metrics
               results.csv         ← flat per-fold table
               session_index.json  ← which sessions were used (paths!)

This is the file you commit alongside a paper to make the experiment
reproducible.

Note on hooks
─────────────
This module purposely does NOT import the pipeline's feature / model /
windowing code at the top level. Those imports happen inside
`_extract_features` and `_fit_eval_model` so the validation package
stays usable in a headless environment that does not have all of the
heavyweight ML deps installed.
"""

from __future__ import annotations

import json
import logging
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .config import ExperimentConfig, dump_experiment
from .corpus import SessionCorpus, SessionRecord
from .cv_strategies import get_strategy

log = logging.getLogger(__name__)


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
    extra:          Dict[str, Any] = field(default_factory=dict)


@dataclass
class RunResult:
    experiment:    ExperimentConfig
    folds:         List[FoldResult] = field(default_factory=list)
    started_at:    str = ""
    finished_at:   str = ""
    output_dir:    Optional[Path] = None

    def to_dict(self) -> dict:
        return {
            "experiment":  self.experiment.to_dict(),
            "started_at":  self.started_at,
            "finished_at": self.finished_at,
            "folds":       [asdict(f) for f in self.folds],
            "aggregate":   self.aggregate(),
        }

    def aggregate(self) -> Dict[str, Dict[str, float]]:
        """Mean ± std per (model, metric) across folds."""
        out: Dict[str, Dict[str, float]] = {}
        if not self.folds:
            return out
        by_model: Dict[str, List[FoldResult]] = {}
        for f in self.folds:
            by_model.setdefault(f.model_type, []).append(f)
        for model, fs in by_model.items():
            accs = np.array([f.accuracy for f in fs], dtype=float)
            f1s  = np.array([f.macro_f1 for f in fs], dtype=float)
            out[model] = {
                "n_folds":        len(fs),
                "accuracy_mean":  float(accs.mean()),
                "accuracy_std":   float(accs.std(ddof=0)),
                "macro_f1_mean":  float(f1s.mean()),
                "macro_f1_std":   float(f1s.std(ddof=0)),
            }
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

    def run(self, cfg: ExperimentConfig) -> RunResult:
        result = RunResult(experiment=cfg, started_at=_now_iso())
        result.output_dir = self._make_output_dir(cfg)

        # Persist the config + environment up front so an interrupted
        # run still leaves enough information to know what was being
        # attempted.
        dump_experiment(cfg, result.output_dir / "experiment.json")
        _write_json(result.output_dir / "environment.json", _capture_environment())

        records = self._select_sessions(cfg)
        _write_json(
            result.output_dir / "session_index.json",
            {"records": [r.to_dict() for r in records]},
        )

        if not records:
            log.warning("No sessions matched the data filter — nothing to do.")
            result.finished_at = _now_iso()
            self._persist_results(result)
            return result

        strategy = get_strategy(cfg.cv.strategy)
        folds = list(strategy(records, **cfg.cv.kwargs))
        log.info("Generated %d CV folds with strategy '%s'", len(folds), cfg.cv.strategy)

        np.random.seed(cfg.seed)

        for model_cfg in cfg.models:
            for fold in folds:
                fr = self._run_one_fold(cfg, model_cfg, fold)
                if fr is not None:
                    result.folds.append(fr)
                    log.info("[%s | %s]  acc=%.3f  f1=%.3f",
                             model_cfg.type, fr.fold_id, fr.accuracy, fr.macro_f1)

        result.finished_at = _now_iso()
        self._persist_results(result)
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
    ) -> Optional[FoldResult]:
        try:
            (X_train, y_train,
             X_val,   y_val,
             X_test,  y_test,
             label_names) = self._materialise_fold(cfg, fold)
        except Exception as e:  # noqa: BLE001
            log.exception("Failed to materialise fold %s: %s", fold.get("id"), e)
            return None

        if len(X_train) == 0 or len(X_test) == 0:
            log.warning("Fold %s has empty train/test set; skipping.", fold.get("id"))
            return None

        t0 = time.time()
        try:
            model, train_meta = self._fit_model(
                cfg, model_cfg,
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

        # Optional val metrics
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
            per_class_f1=metrics.get("per_class_f1", {}),
            val_accuracy=val_acc,
            val_macro_f1=val_f1,
            train_seconds=train_secs,
            inference_ms=metrics.get("inference_ms", 0.0),
            extra={"train_meta": train_meta},
        )

    # ------------------------------------------------------------------
    # The three glue points to the existing pipeline.
    # Kept thin and isolated so they can be retargeted without touching
    # the rest of the runner.
    # ------------------------------------------------------------------

    def _materialise_fold(
        self,
        cfg: ExperimentConfig,
        fold: Dict[str, Any],
    ) -> Tuple[np.ndarray, np.ndarray,
               np.ndarray, np.ndarray,
               np.ndarray, np.ndarray,
               Dict[int, str]]:
        """
        Turn a fold of SessionRecords into

            (X_train, y_train, X_val, y_val, X_test, y_test, label_names)

        For folds that don't define a separate validation set
        (everything except ``holdout_split``) the val arrays are
        returned empty (shape ``(0,)``) — the caller checks ``len()``
        before passing them to the model.

        Reuses ``DataManager.create_dataset`` so the windowing /
        per-session-rotation / bad-channel / feature-extraction code
        path is identical to the one the GUI Train tab uses.
        """
        from playagain_pipeline.core.data_manager import DataManager  # lazy
        dm = DataManager(self.data_dir)

        feature_config = self._build_feature_config(cfg)

        def _empty():
            return np.empty((0,)), np.empty((0,), dtype=int), {}

        def _load_sessions(records):
            sess = []
            for rec in records:
                try:
                    sess.append(dm.load_session(rec.subject_id, rec.session_id))
                except Exception as e:  # noqa: BLE001
                    log.warning("Could not load %s/%s: %s",
                                rec.subject_id, rec.session_id, e)
            return sess

        def _make_xy(records, name_suffix: str):
            sessions = _load_sessions(records)
            if not sessions:
                return _empty()
            ds = dm.create_dataset(
                name=f"_validation_tmp_{name_suffix}",
                sessions=sessions,
                window_size_ms=cfg.windowing.window_ms,
                window_stride_ms=cfg.windowing.stride_ms,
                feature_config=feature_config,
                use_per_session_rotation=False,
            )
            return ds["X"], ds["y"], ds["metadata"].get("label_names", {})

        is_within = fold.get("split_kind") == "temporal_tail"
        if is_within:
            X_all, y_all, label_names = _make_xy(fold["train"], "within")
            if len(X_all) == 0:
                return X_all, y_all, *_empty()[:2], X_all, y_all, label_names
            test_frac = float(fold.get("test_fraction", 0.2))
            cut = int(len(X_all) * (1.0 - test_frac))
            return (X_all[:cut], y_all[:cut],
                    *_empty()[:2],
                    X_all[cut:], y_all[cut:],
                    label_names)

        X_tr, y_tr, names_tr = _make_xy(fold["train"], "train")
        X_te, y_te, names_te = _make_xy(fold["test"],  "test")

        # holdout_split (and any future strategy) may attach a separate
        # val record list. If absent, return empty val arrays.
        val_records = fold.get("val") or []
        if val_records:
            X_va, y_va, names_va = _make_xy(val_records, "val")
        else:
            X_va, y_va, names_va = _empty()

        label_names = {**names_tr, **names_va, **names_te}
        return X_tr, y_tr, X_va, y_va, X_te, y_te, label_names

    @staticmethod
    def _build_feature_config(cfg: ExperimentConfig) -> Optional[Dict[str, Any]]:
        """Translate ExperimentConfig.features into the dict shape
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


    def _fit_model(self, cfg: ExperimentConfig, model_cfg,
                   X_train, y_train, X_val=None, y_val=None):
        """
        Fit a model. If a held-out val set is provided, pass it
        directly to the model's ``train()`` method so deep-learning
        models can use it for early stopping / learning-rate scheduling
        / live curves. When no val set is provided we fall back to
        ``ModelManager.train_model`` which performs its own internal
        80/20 split — keeping behaviour identical to the GUI Train tab.
        """
        from playagain_pipeline.models.classifier import ModelManager  # lazy

        mm = ModelManager(self.data_dir / "models")
        model = mm.create_model(
            model_cfg.type,
            name=f"_validation_{model_cfg.type}",
            **(model_cfg.params or {}),
        )

        if X_val is not None and len(X_val) > 0:
            # Use the supplied val split — gives the model a real
            # held-out evaluation set rather than an internal re-split
            # of the training data.
            train_meta = model.train(
                X_train, y_train,
                X_val, y_val,
                window_size_ms=cfg.windowing.window_ms,
                sampling_rate=getattr(cfg.windowing, "sampling_rate", 2000),
                num_channels=int(X_train.shape[-1]) if X_train.ndim >= 2 else 0,
            )
            # Don't persist these throwaway models to disk — the
            # validation run produces its own results dir.
            return model, train_meta

        # No val supplied → reuse ModelManager.train_model so the
        # behaviour is byte-identical to the Train tab.
        train_meta = mm.train_model(
            model,
            {
                "X": X_train,
                "y": y_train,
                "metadata": {
                    "name": f"_validation_{model_cfg.type}",
                    "window_size_ms": cfg.windowing.window_ms,
                    "sampling_rate": 2000,
                    "num_channels": int(X_train.shape[-1]) if X_train.ndim >= 2 else 0,
                },
            },
            save=False,
        )
        return model, train_meta

    def _evaluate(self, model, X_test, y_test, label_names):
        from sklearn.metrics import f1_score, accuracy_score  # lazy
        t0 = time.time()
        y_pred = model.predict(X_test)
        inf_ms = (time.time() - t0) * 1000.0 / max(len(X_test), 1)

        acc = float(accuracy_score(y_test, y_pred))
        f1m = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
        per_class = f1_score(y_test, y_pred, average=None, zero_division=0)
        labels = sorted(set(int(v) for v in y_test) | set(int(v) for v in y_pred))
        per_class_named = {
            label_names.get(int(lbl), str(lbl)): float(score)
            for lbl, score in zip(labels, per_class)
        }
        return {
            "accuracy":     acc,
            "macro_f1":     f1m,
            "per_class_f1": per_class_named,
            "inference_ms": inf_ms,
        }

    # ------------------------------------------------------------------

    def _make_output_dir(self, cfg: ExperimentConfig) -> Path:
        ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        safe_name = "".join(c if (c.isalnum() or c in "._-") else "_" for c in cfg.name)
        out = self.output_root / f"{ts}__{safe_name}"
        out.mkdir(parents=True, exist_ok=True)
        return out

    def _persist_results(self, result: RunResult) -> None:
        if result.output_dir is None:
            return
        _write_json(result.output_dir / "results.json", result.to_dict())
        _write_csv(result.output_dir / "results.csv", result.folds)
        log.info("Wrote validation run to %s", result.output_dir)


# ---------------------------------------------------------------------------
# Reproducibility helpers
# ---------------------------------------------------------------------------

def _capture_environment() -> dict:
    """Grab the bits of context needed to reproduce a run later."""
    env: Dict[str, Any] = {
        "captured_at":   _now_iso(),
        "python":        sys.version,
        "platform":      platform.platform(),
        "executable":    sys.executable,
    }

    # Git SHA, if we are inside a repo. Best-effort — never fails the run.
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

    # Versions of the libraries we actually rely on for the result.
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
    import csv
    fieldnames = [
        "fold_id", "model_type",
        "n_train_windows", "n_val_windows", "n_test_windows",
        "accuracy", "macro_f1",
        "val_accuracy", "val_macro_f1",
        "train_seconds", "inference_ms",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for fr in folds:
            w.writerow({k: getattr(fr, k) for k in fieldnames})
