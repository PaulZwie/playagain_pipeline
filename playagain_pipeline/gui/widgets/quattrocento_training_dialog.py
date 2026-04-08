"""
QuattrocentoTrainingDialog  (v3)
=================================
Improvements over v2:
  • Cleaner left-panel layout: logical top-to-bottom flow with collapsible
    sections and inline tooltips.
  • Subject-aware card-based split builder replacing raw tag text-fields:
    drag-and-drop feel with Train / Val / Test badges per (subject, trial).
  • Real-time data preview with window counts, class balance bar, and a
    class-imbalance warning.
  • Class-weight compensation option (weighted loss / oversampling).
  • Early-stopping patience control exposed in the UI.
  • F1-macro score added alongside accuracy everywhere.
  • Macro-averaged per-class metrics table with sortable columns.
  • Normalisation now fitted on train windows only and applied correctly to
    val AND test sets separately (was a subtle bug in v2).
  • Learning-curve panel no longer blocks the UI (runs in a QThread).
  • Confusion-matrix canvas resizes correctly when the dialog is resized.
  • Logging uses HTML colour coding and auto-scrolls.
  • Export CSV now includes F1, precision, recall per class.
  • All QThread workers properly joined before destruction.
"""

from __future__ import annotations

import csv
import threading
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

from PySide6.QtCore import Qt, Signal, Slot, QThread, QTimer
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QLabel, QPushButton, QLineEdit, QFileDialog, QComboBox, QCheckBox,
    QListWidget, QListWidgetItem, QProgressBar, QTextEdit, QSpinBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QSplitter, QTabWidget,
    QWidget, QMessageBox, QAbstractItemView, QRadioButton, QButtonGroup,
    QScrollArea, QDoubleSpinBox, QFrame, QSizePolicy,
)
from PySide6.QtGui import QFont, QColor, QBrush

from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

# ─── constants ────────────────────────────────────────────────────────────────

_DEFAULT_PRIMARY_CONFIG: Tuple[int, int] = (408, 64)
_DEFAULT_INTRA_SPLIT_RATIOS: Tuple[float, float, float] = (0.70, 0.15, 0.15)

_MODEL_TYPES = [
    ("SVM",           "svm"),
    ("LDA",           "lda"),
    ("Random Forest", "random_forest"),
    ("CatBoost",      "catboost"),
    ("MLP",           "mlp"),
    ("CNN",           "cnn"),
    ("AttentionNet",  "attention_net"),
    ("MSTNet",        "mstnet"),
]

_DEFAULT_LABEL_ORDER = [
    "rest", "power_grasp", "pinch", "tripod_pinch",
    "thumb", "index", "middle", "ring", "pinky",
    "open_hand", "wrist_flex", "wrist_extend", "pronation", "supination",
]

# ─── colour palette ───────────────────────────────────────────────────────────

_C = {
    "bg":      "#1e1e2e",
    "panel":   "#2a2a3e",
    "card":    "#313145",
    "accent":  "#7c3aed",
    "accent2": "#06b6d4",
    "success": "#22c55e",
    "warning": "#f59e0b",
    "danger":  "#ef4444",
    "text":    "#e2e8f0",
    "muted":   "#94a3b8",
    "border":  "#3f3f5c",
    "train":   "#22c55e",
    "val":     "#f59e0b",
    "test":    "#ef4444",
    "excl":    "#6b7280",
}

_PANEL_STYLE = f"""
    QGroupBox {{
        background: {_C['panel']};
        border: 1px solid {_C['border']};
        border-radius: 8px;
        font-weight: 600;
        color: {_C['text']};
        padding-top: 16px;
        margin-top: 6px;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        left: 10px;
        padding: 0 6px;
        color: {_C['accent2']};
    }}
"""
_BTN_PRIMARY = f"""
    QPushButton {{
        background: {_C['accent']};
        color: white;
        border: none;
        border-radius: 6px;
        padding: 6px 18px;
        font-weight: 700;
        font-size: 13px;
    }}
    QPushButton:hover {{ background: #6d28d9; }}
    QPushButton:disabled {{ background: #3f3f5c; color: #6b7280; }}
"""
_BTN_SECONDARY = f"""
    QPushButton {{
        background: {_C['panel']};
        color: {_C['text']};
        border: 1px solid {_C['border']};
        border-radius: 6px;
        padding: 5px 14px;
    }}
    QPushButton:hover {{ border-color: {_C['accent2']}; color: {_C['accent2']}; }}
"""


# ─── data helpers ─────────────────────────────────────────────────────────────

def _resize_window(w: np.ndarray, target_ws: int, target_nc: int) -> np.ndarray:
    ws, nc = w.shape
    if nc > target_nc:
        w = w[:, :target_nc]
    elif nc < target_nc:
        w = np.concatenate([w, np.zeros((ws, target_nc - nc), dtype=w.dtype)], 1)
    if ws > target_ws:
        return w[:target_ws]
    elif ws < target_ws:
        pad = np.zeros((target_ws - ws, w.shape[1]), dtype=w.dtype)
        return np.concatenate([w, pad], 0)
    return w


def _augment_windows(X: np.ndarray, noise_std: float = 0.005,
                     time_shift: int = 0) -> np.ndarray:
    aug = X.copy()
    if noise_std > 0:
        aug += np.random.randn(*aug.shape).astype(np.float32) * noise_std
    if time_shift > 0:
        shift = np.random.randint(-time_shift, time_shift + 1, size=len(aug))
        for i, s in enumerate(shift):
            if s != 0:
                aug[i] = np.roll(aug[i], s, axis=0)
    return aug


def _split_bounds(total_windows: int,
                  ratios: Tuple[float, float, float] = _DEFAULT_INTRA_SPLIT_RATIOS) -> Tuple[int, int]:
    """Return (train_end, val_end) for deterministic intra-trial slicing."""
    if total_windows <= 1:
        return total_windows, total_windows
    train_ratio, val_ratio, _ = ratios
    train_end = int(total_windows * train_ratio)
    val_end = int(total_windows * (train_ratio + val_ratio))
    train_end = max(1, min(train_end, total_windows - 1))
    val_end = max(train_end, min(val_end, total_windows))
    return train_end, val_end


class _Normaliser:
    """Fit on train, apply to any split. Fixes the v2 bug where val/test
    normalisation could use stale statistics."""

    def __init__(self, mode: str):
        self.mode = mode
        self.mu: Any = None
        self.sd: Any = None

    def fit(self, X: np.ndarray) -> "_Normaliser":
        if self.mode == "z_channel":
            self.mu = X.mean(axis=(0, 1), keepdims=True)
            self.sd = X.std(axis=(0, 1), keepdims=True).clip(min=1e-8)
        elif self.mode == "z_global":
            self.mu = X.mean()
            self.sd = max(X.std(), 1e-8)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mode == "none" or self.mu is None:
            return X
        return (X - self.mu) / self.sd


# ─── background workers ───────────────────────────────────────────────────────

class _TrainWorker(QThread):
    """Full training pipeline: load → augment → normalise → CV → train → evaluate."""

    progress     = Signal(int, str)
    epoch_update = Signal(int, float, float, float, float)
    fold_result  = Signal(int, float, float)  # fold_idx, acc, f1
    finished     = Signal(object, object)      # results_dict, model
    error        = Signal(str)

    def __init__(
        self,
        loader,
        train_recs: list,
        test_recs: list,
        val_recs: list,
        model,
        feature_config: dict,
        sampling_rate: float,
        window_size: int,
        n_channels: int,
        cv_mode: str,
        cv_folds: int,
        norm_mode: str,
        augment: bool,
        noise_std: float,
        time_shift: int,
        channel_reduce: str,
        channel_reduce_n: int,
        class_weight: str,
        early_stopping_patience: int,
        use_trigger_segments: bool = True,
        onset_delay_ms: float = 150.0,
        intra_trial_split: bool = False,
        split_ratios: Tuple[float, float, float] = _DEFAULT_INTRA_SPLIT_RATIOS,
    ):
        super().__init__()
        self._loader    = loader
        self._train_recs = train_recs
        self._test_recs  = test_recs
        self._val_recs   = val_recs
        self._model      = model
        self._feat_cfg   = feature_config
        self._sr         = sampling_rate
        self._ws         = window_size
        self._nc         = n_channels
        self._cv_mode    = cv_mode
        self._cv_folds   = cv_folds
        self._norm       = norm_mode
        self._augment    = augment
        self._noise_std  = noise_std
        self._time_shift = time_shift
        self._ch_reduce  = channel_reduce
        self._ch_n       = channel_reduce_n
        self._class_weight = class_weight
        self._es_patience  = early_stopping_patience
        self._use_trigger  = use_trigger_segments
        self._onset_delay  = onset_delay_ms
        self._intra_trial_split = intra_trial_split
        self._split_ratios = split_ratios
        self._test_split_counts: Optional[List[int]] = None

    def run(self):
        try:
            from sklearn.metrics import f1_score, classification_report, precision_score, recall_score
            self._test_split_counts = None

            # ── Load ──────────────────────────────────────────────────────
            # Build ONE shared label map from the union of all splits so that
            # train / val / test use identical integer class indices even when
            # a subject in one split is missing a gesture that another has.
            self.progress.emit(5, "Loading training data…")
            all_recs_for_map = (self._train_recs +
                                (self._val_recs  or []) +
                                (self._test_recs or []))
            shared_g2l, label_names = self._build_g2l(all_recs_for_map)

            X_train, y_train, _ = self._load(self._train_recs, shared_g2l, split_role="train")
            self.progress.emit(18, f"Loaded {len(X_train):,} training windows")

            X_test, y_test, _ = (self._load(self._test_recs, shared_g2l, split_role="test") if self._test_recs else
                (np.empty((0, self._ws, self._nc), np.float32),
                 np.empty(0, np.int64), label_names))

            X_val, y_val, _ = (self._load(self._val_recs, shared_g2l, split_role="val") if self._val_recs else
                (np.empty((0, self._ws, self._nc), np.float32),
                 np.empty(0, np.int64), label_names))

            self.progress.emit(28, f"Train:{len(X_train):,}  Val:{len(X_val):,}  Test:{len(X_test):,}")

            # ── Augment ───────────────────────────────────────────────────
            if self._augment and len(X_train) > 0:
                self.progress.emit(30, "Augmenting training windows…")
                aug = _augment_windows(X_train, self._noise_std, self._time_shift)
                X_train = np.concatenate([X_train, aug], 0)
                y_train = np.concatenate([y_train, y_train], 0)
                self.progress.emit(33, f"After augmentation: {len(X_train):,} windows")

            # ── Channel reduction ─────────────────────────────────────────
            if self._ch_reduce != "none" and len(X_train) > 0:
                X_train, X_val, X_test, _ = self._apply_channel_reduce(X_train, X_val, X_test)
                self.progress.emit(36, f"Channel reduction → {X_train.shape[2]} channels")

            # ── Normalise (fit on train only, transform val+test separately) ──
            norm = _Normaliser(self._norm).fit(X_train)
            X_train = norm.transform(X_train)
            if len(X_val)  > 0: X_val  = norm.transform(X_val)
            if len(X_test) > 0: X_test = norm.transform(X_test)

            # ── Cross-validation ──────────────────────────────────────────
            cv_results: List[Tuple[float, float]] = []  # (acc, f1) per fold
            if self._cv_mode == "kfold" and len(X_train) > 0:
                cv_results = self._run_kfold(X_train, y_train, label_names, norm)
            elif self._cv_mode == "loso":
                cv_results = self._run_loso(label_names)

            # ── Class weights ─────────────────────────────────────────────
            class_weight_dict = None
            if self._class_weight == "balanced" and len(X_train) > 0:
                classes, counts = np.unique(y_train, return_counts=True)
                total = len(y_train)
                n_cls = len(classes)
                class_weight_dict = {int(c): total / (n_cls * cnt)
                                     for c, cnt in zip(classes, counts)}

            # ── Train final model ─────────────────────────────────────────
            self.progress.emit(55, "Training final model…")
            t0 = time.time()

            _stop = threading.Event()
            def _sim():
                p = 57
                while not _stop.is_set() and p < 88:
                    self.progress.emit(p, "Training final model…")
                    _stop.wait(0.3)
                    p = min(p + 2, 88)
            sim_t = threading.Thread(target=_sim, daemon=True)
            sim_t.start()

            n_ch_final = X_train.shape[2]

            def _epoch_cb(epoch, tl, vl, ta=0.0, va=0.0):
                self.epoch_update.emit(epoch, float(tl), float(vl), float(ta), float(va))

            extra_kw: Dict[str, Any] = {"callback": _epoch_cb}
            if class_weight_dict is not None:
                extra_kw["class_weight"] = class_weight_dict
            if self._es_patience > 0:
                extra_kw["early_stopping_patience"] = self._es_patience

            window_size_ms = round(self._ws / self._sr * 1000, 1)

            if len(X_val) > 0:
                train_results = self._model.train(
                    X_train, y_train, X_val, y_val,
                    window_size_ms=window_size_ms,
                    sampling_rate=self._sr, num_channels=n_ch_final,
                    feature_config=self._feat_cfg, **extra_kw)
            else:
                train_results = self._model.train(
                    X_train, y_train,
                    window_size_ms=window_size_ms,
                    sampling_rate=self._sr, num_channels=n_ch_final,
                    feature_config=self._feat_cfg, **extra_kw)

            _stop.set(); sim_t.join(1)
            train_time = time.time() - t0
            self.progress.emit(90, "Evaluating on test set…")

            # ── Evaluate on held-out test set ─────────────────────────────
            if len(X_test) > 0:
                y_pred   = self._model.predict(X_test)
                test_acc = float(np.mean(y_pred == y_test))
                test_f1  = float(f1_score(y_test, y_pred, average="macro", zero_division=0))

                classes  = sorted(set(y_test))
                n_cls    = len(classes)
                conf_mat = np.zeros((n_cls, n_cls), dtype=int)
                for tr, pr in zip(y_test, y_pred):
                    ti = classes.index(tr)
                    pi = classes.index(pr) if pr in classes else 0
                    conf_mat[ti, pi] += 1

                per_class_acc = {}
                per_class_f1  = {}
                per_class_prec= {}
                per_class_rec = {}
                ci_low, ci_high = {}, {}
                for i, c in enumerate(classes):
                    row_sum = int(conf_mat[i].sum())
                    p = conf_mat[i, i] / row_sum if row_sum else 0.0
                    per_class_acc[c] = p
                    # Wilson CI
                    if row_sum > 0:
                        z = 1.96
                        denom = 1 + z**2 / row_sum
                        centre = (p + z**2 / (2 * row_sum)) / denom
                        margin = z * np.sqrt(p*(1-p)/row_sum + z**2/(4*row_sum**2)) / denom
                        ci_low[c]  = max(0.0, centre - margin)
                        ci_high[c] = min(1.0, centre + margin)
                    else:
                        ci_low[c] = ci_high[c] = 0.0

                # Per-class F1/precision/recall from sklearn
                for c in classes:
                    mask_true = (y_test == c)
                    mask_pred = (y_pred == c)
                    tp = int(np.sum(mask_true & mask_pred))
                    fp = int(np.sum(~mask_true & mask_pred))
                    fn = int(np.sum(mask_true & ~mask_pred))
                    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                    f1c  = 2*prec*rec / (prec + rec) if (prec + rec) > 0 else 0.0
                    per_class_f1[c]   = f1c
                    per_class_prec[c] = prec
                    per_class_rec[c]  = rec

                per_subject: Dict[str, Dict[str, float]] = {}
                if self._test_recs and hasattr(self._test_recs[0], "subject_id"):
                    per_subject = self._per_subject_metrics(X_test, y_test)
            else:
                test_acc = 0.0
                test_f1  = 0.0
                classes  = sorted(set(y_train))
                n_cls    = len(classes)
                conf_mat = np.zeros((n_cls, n_cls), dtype=int)
                per_class_acc  = {c: 0.0 for c in classes}
                per_class_f1   = {c: 0.0 for c in classes}
                per_class_prec = {c: 0.0 for c in classes}
                per_class_rec  = {c: 0.0 for c in classes}
                ci_low = ci_high = {c: 0.0 for c in classes}
                per_subject = {}

            # Retrieve val metrics from training results dict
            val_acc = float(train_results.get("validation_accuracy", 0.0))
            val_f1  = float(train_results.get("validation_f1", 0.0))

            # Update model metadata
            if hasattr(self._model, "metadata") and self._model.metadata is not None:
                self._model.metadata.class_names    = label_names
                self._model.metadata.num_channels   = n_ch_final
                self._model.metadata.sampling_rate  = int(self._sr)
                self._model.metadata.window_samples = self._ws

            cv_accs = [r[0] for r in cv_results]
            cv_f1s  = [r[1] for r in cv_results]

            results_out = {
                "train_acc":      float(train_results.get("training_accuracy", 0.0)),
                "train_f1":       float(train_results.get("training_f1", 0.0)),
                "val_acc":        val_acc,
                "val_f1":         val_f1,
                "test_acc":       test_acc,
                "test_f1":        test_f1,
                "train_time":     train_time,
                "n_train":        len(X_train),
                "n_val":          len(X_val),
                "n_test":         len(X_test),
                "confusion_matrix": conf_mat,
                "classes":        classes,
                "per_class_acc":  per_class_acc,
                "per_class_f1":   per_class_f1,
                "per_class_prec": per_class_prec,
                "per_class_rec":  per_class_rec,
                "ci_low":         ci_low,
                "ci_high":        ci_high,
                "per_subject":    per_subject,
                "label_names":    label_names,
                "cv_results_acc": cv_accs,
                "cv_results_f1":  cv_f1s,
                "cv_mean_acc":    float(np.mean(cv_accs)) if cv_accs else None,
                "cv_std_acc":     float(np.std(cv_accs))  if cv_accs else None,
                "cv_mean_f1":     float(np.mean(cv_f1s))  if cv_f1s  else None,
                "augmented":      self._augment,
                "norm_mode":      self._norm,
                "ch_reduce":      self._ch_reduce,
                "n_ch_final":     n_ch_final,
                "class_weight":   self._class_weight,
            }

            self.progress.emit(100, "Done")
            self.finished.emit(results_out, self._model)

        except Exception as exc:
            import traceback
            self.error.emit(f"{exc}\n\n{traceback.format_exc()}")

    # ── loaders ──────────────────────────────────────────────────────────────

    @staticmethod
    def _build_g2l(recs: list) -> Tuple[Dict[str, int], Dict[int, str]]:
        """Build a gesture→label mapping from a list of records.

        Always called on the UNION of all splits so that train, val and test
        share identical integer class indices.
        """
        gestures = sorted({r.gesture for r in recs},
                          key=lambda g: (_DEFAULT_LABEL_ORDER.index(g)
                                         if g in _DEFAULT_LABEL_ORDER else 999))
        g2l = {g: i for i, g in enumerate(gestures)}
        label_names = {int(v): k for k, v in g2l.items()}
        return g2l, label_names

    def _load(self, recs: list,
              g2l: Optional[Dict[str, int]] = None,
              split_role: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, Dict[int, str]]:
        """Load windows for *recs* using the provided *g2l* mapping.

        If *g2l* is None a local mapping is built from *recs* only (legacy
        behaviour, kept for internal callers that already own a consistent map).
        Pass a shared mapping built from all splits to avoid label-index
        mismatches between train / val / test.
        """
        from playagain_pipeline.gui.widgets.quattrocento_loader import QuattrocentoFileLoader
        if g2l is None:
            g2l, _ = self._build_g2l(recs)
        label_names = {int(v): k for k, v in g2l.items()}
        total = sum(r.n_windows for r in recs)
        X = np.empty((total, self._ws, self._nc), dtype=np.float32)
        y = np.empty(total, dtype=np.int64)
        wp = 0
        test_counts: List[int] = []
        for rec in recs:
            wins = QuattrocentoFileLoader.load_windows(
                rec,
                use_trigger_segments=getattr(self, '_use_trigger', False),
                onset_delay_ms=getattr(self, '_onset_delay', 150.0),
            )
            if self._intra_trial_split and split_role in {"train", "val", "test"}:
                tr_end, va_end = _split_bounds(len(wins), self._split_ratios)
                if split_role == "train":
                    wins = wins[:tr_end]
                elif split_role == "val":
                    wins = wins[tr_end:va_end]
                else:
                    wins = wins[va_end:]
            if split_role == "test":
                test_counts.append(len(wins))
            lbl  = g2l[rec.gesture]
            for w in wins:
                X[wp] = _resize_window(w, self._ws, self._nc)
                y[wp] = lbl
                wp += 1
        if split_role == "test":
            self._test_split_counts = test_counts
        return X[:wp], y[:wp], label_names

    def _apply_channel_reduce(self, X_tr, X_v, X_te):
        n_keep = min(self._ch_n, X_tr.shape[2])
        if self._ch_reduce == "variance_top_n":
            var = X_tr.var(axis=(0, 1))
            idx = np.sort(np.argsort(var)[::-1][:n_keep])
            return (X_tr[:, :, idx],
                    X_v[:, :, idx]  if len(X_v)  else X_v,
                    X_te[:, :, idx] if len(X_te) else X_te,
                    idx)
        elif self._ch_reduce == "pca":
            from sklearn.decomposition import PCA
            flat_tr = X_tr.reshape(len(X_tr), -1)
            pca = PCA(n_components=n_keep, svd_solver="randomized", random_state=42)
            flat_tr2 = pca.fit_transform(flat_tr).astype(np.float32)
            flat_v   = pca.transform(X_v.reshape(len(X_v), -1)).astype(np.float32) if len(X_v)  else np.empty((0, n_keep), np.float32)
            flat_te  = pca.transform(X_te.reshape(len(X_te), -1)).astype(np.float32) if len(X_te) else np.empty((0, n_keep), np.float32)
            def _rs(a, n): return a.reshape(len(a), 1, n)
            return _rs(flat_tr2, n_keep), _rs(flat_v, n_keep), _rs(flat_te, n_keep), pca
        return X_tr, X_v, X_te, None

    def _run_kfold(self, X: np.ndarray, y: np.ndarray,
                   label_names: dict, norm: _Normaliser) -> List[Tuple[float, float]]:
        from sklearn.model_selection import StratifiedKFold
        from sklearn.metrics import f1_score
        # ⚠ K-Fold CV operates only on the training-subject windows that were
        # passed in.  All folds therefore see data from the same subjects,
        # which gives a within-subject accuracy estimate.  This will look much
        # higher than cross-subject test performance and is NOT a reliable
        # predictor of how the model generalises to new subjects.  Use LOSO
        # whenever train / test are split by subject.
        skf = StratifiedKFold(n_splits=self._cv_folds, shuffle=True, random_state=42)
        results = []
        for fold_i, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
            self.progress.emit(38 + fold_i * 3, f"CV fold {fold_i+1}/{self._cv_folds}…")
            Xf_tr, yf_tr = X[tr_idx], y[tr_idx]
            Xf_va, yf_va = X[va_idx], y[va_idx]
            fold_norm = _Normaliser(self._norm).fit(Xf_tr)
            Xf_tr = fold_norm.transform(Xf_tr)
            Xf_va = fold_norm.transform(Xf_va)
            try:
                m = self._model.__class__.__new__(self._model.__class__)
                # Minimal clone: deep copy then retrain
                import copy
                m = copy.deepcopy(self._model)
                m.train(Xf_tr, yf_tr, Xf_va, yf_va,
                        window_size_ms=round(self._ws / self._sr * 1000, 1),
                        sampling_rate=self._sr, num_channels=Xf_tr.shape[2],
                        feature_config=self._feat_cfg)
                preds = m.predict(Xf_va)
                acc = float(np.mean(preds == yf_va))
                f1  = float(f1_score(yf_va, preds, average="macro", zero_division=0))
            except Exception:
                acc = f1 = 0.0
            results.append((acc, f1))
            self.fold_result.emit(fold_i, acc, f1)
        return results

    def _run_loso(self, label_names: dict) -> List[Tuple[float, float]]:
        from sklearn.metrics import f1_score
        all_recs = []
        seen = set()
        for rec in (self._train_recs + self._test_recs):
            key = str(getattr(rec, "path", id(rec)))
            if key in seen:
                continue
            seen.add(key)
            all_recs.append(rec)
        # Build the shared label map once so every fold uses the same indices.
        shared_g2l, _ = self._build_g2l(all_recs)
        subjects = sorted({r.subject_id for r in all_recs})
        results = []
        for si, subj in enumerate(subjects):
            self.progress.emit(38 + si * 3, f"LOSO: leaving out {subj} ({si+1}/{len(subjects)})…")
            train_r = [r for r in all_recs if r.subject_id != subj]
            test_r  = [r for r in all_recs if r.subject_id == subj]
            if not train_r or not test_r:
                continue
            Xtr, ytr, _ = self._load(train_r, shared_g2l)
            Xte, yte, _ = self._load(test_r,  shared_g2l)
            norm = _Normaliser(self._norm).fit(Xtr)
            Xtr = norm.transform(Xtr)
            Xte = norm.transform(Xte)
            try:
                import copy
                m = copy.deepcopy(self._model)
                m.train(Xtr, ytr, window_size_ms=round(self._ws / self._sr * 1000, 1),
                        sampling_rate=self._sr, num_channels=Xtr.shape[2],
                        feature_config=self._feat_cfg)
                preds = m.predict(Xte)
                acc = float(np.mean(preds == yte))
                f1  = float(f1_score(yte, preds, average="macro", zero_division=0))
            except Exception:
                acc = f1 = 0.0
            results.append((acc, f1))
            self.fold_result.emit(si, acc, f1)
        return results

    def _per_subject_metrics(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Dict[str, float]]:
        from sklearn.metrics import f1_score
        result: Dict[str, Dict[str, float]] = {}
        wp = 0
        test_counts = self._test_split_counts if self._test_split_counts is not None else []
        for i, rec in enumerate(self._test_recs):
            n = test_counts[i] if i < len(test_counts) else rec.n_windows
            if wp + n > len(X_test):
                break
            X_s = X_test[wp:wp+n]
            y_s = y_test[wp:wp+n]
            wp += n
            try:
                preds = self._model.predict(X_s)
                result[rec.subject_id] = {
                    "acc": float(np.mean(preds == y_s)),
                    "f1":  float(f1_score(y_s, preds, average="macro", zero_division=0)),
                }
            except Exception:
                pass
        return result


class _LearningCurveWorker(QThread):
    """Computes learning curve without blocking the UI."""
    progress = Signal(int, str)
    finished = Signal(list, list, list)  # sizes, train_accs, val_accs
    error    = Signal(str)

    def __init__(self, loader, train_recs, test_recs, model_manager, model_type,
                 ws, nc, sr, norm_mode, feat_cfg, label_order):
        super().__init__()
        self._loader = loader
        self._train_recs = train_recs
        self._test_recs  = test_recs
        self._model_manager = model_manager
        self._model_type = model_type
        self._ws = ws
        self._nc = nc
        self._sr = sr
        self._norm_mode = norm_mode
        self._feat_cfg  = feat_cfg
        self._label_order = label_order

    def run(self):
        try:
            from playagain_pipeline.gui.widgets.quattrocento_loader import QuattrocentoFileLoader
            from sklearn.metrics import f1_score

            def load_all(r_list, g2l):
                total = sum(r.n_windows for r in r_list)
                X = np.empty((total, self._ws, self._nc), np.float32)
                y = np.empty(total, np.int64)
                wp = 0
                for rec in r_list:
                    wins = QuattrocentoFileLoader.load_windows(rec)
                    lbl = g2l[rec.gesture]
                    for w in wins:
                        X[wp] = _resize_window(w, self._ws, self._nc)
                        y[wp] = lbl
                        wp += 1
                return X[:wp], y[:wp]

            # Build one shared map from all records so train and test indices align.
            all_recs = self._train_recs + self._test_recs
            all_gestures = sorted({r.gesture for r in all_recs},
                                  key=lambda g: (self._label_order.index(g)
                                                 if g in self._label_order else 999))
            shared_g2l = {g: i for i, g in enumerate(all_gestures)}

            self.progress.emit(5, "Loading test data…")
            X_te, y_te = load_all(self._test_recs, shared_g2l)

            steps = [int(len(self._train_recs) * f)
                     for f in [0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0]]
            steps = sorted(set(max(1, s) for s in steps))

            tr_accs, te_accs, win_counts = [], [], []
            for si, step in enumerate(steps):
                self.progress.emit(10 + si * 12, f"Step {si+1}/{len(steps)}: {step} files…")
                sub_recs = self._train_recs[:step]
                X_tr, y_tr = load_all(sub_recs, shared_g2l)
                norm = _Normaliser(self._norm_mode).fit(X_tr)
                X_tr2 = norm.transform(X_tr)
                X_te2 = norm.transform(X_te)
                try:
                    m = self._model_manager.create_model(self._model_type, name=f"__lc_{step}__")
                    m.train(X_tr2, y_tr, window_size_ms=round(self._ws/self._sr*1000,1),
                            sampling_rate=self._sr, num_channels=self._nc,
                            feature_config=self._feat_cfg)
                    tr_accs.append(float(np.mean(m.predict(X_tr2) == y_tr)))
                    te_accs.append(float(np.mean(m.predict(X_te2) == y_te)))
                    win_counts.append(len(X_tr2))
                except Exception:
                    tr_accs.append(0.0); te_accs.append(0.0); win_counts.append(len(X_tr))

            self.progress.emit(100, "Learning curve done.")
            self.finished.emit(win_counts, tr_accs, te_accs)
        except Exception as exc:
            import traceback
            self.error.emit(traceback.format_exc())


# ─── matplotlib canvases ──────────────────────────────────────────────────────

class _ConfusionMatrixCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None):
        self._fig = Figure(facecolor="#1a1a2e")
        super().__init__(self._fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def plot(self, conf_mat: np.ndarray, class_names: List[str], title: str = ""):
        self._fig.clear()
        ax = self._fig.add_subplot(111, facecolor="#1a1a2e")
        row_sums = conf_mat.sum(1, keepdims=True).clip(min=1)
        cm_pct   = conf_mat / row_sums * 100
        im = ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=100, aspect="auto")
        cbar = self._fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        cbar.set_label("%", color="#e2e8f0", fontsize=8)
        cbar.ax.yaxis.set_tick_params(color="#e2e8f0", labelcolor="#e2e8f0")
        n = len(class_names)
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels(class_names, rotation=40, ha="right", fontsize=max(5, 9 - n//4), color="#e2e8f0")
        ax.set_yticklabels(class_names, fontsize=max(5, 9 - n//4), color="#e2e8f0")
        ax.set_xlabel("Predicted", fontsize=9, color="#94a3b8")
        ax.set_ylabel("True", fontsize=9, color="#94a3b8")
        ax.set_title(title, fontsize=9, color="#e2e8f0", pad=8)
        ax.tick_params(colors="#94a3b8")
        for spine in ax.spines.values():
            spine.set_edgecolor("#3f3f5c")
        thresh = 50
        for i in range(n):
            for j in range(n):
                col = "white" if cm_pct[i, j] > thresh else "#94a3b8"
                ax.text(j, i, f"{cm_pct[i,j]:.0f}%\n({conf_mat[i,j]})",
                        ha="center", va="center",
                        fontsize=max(4, 7 - n//5), color=col)
        self._fig.tight_layout(pad=1.2)
        self.draw()


class _LearningCurveCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None):
        self._fig = Figure(facecolor="#1a1a2e")
        super().__init__(self._fig)
        self.setParent(parent)

    def plot(self, sizes, train_accs, val_accs, title="Learning Curve"):
        self._fig.clear()
        ax = self._fig.add_subplot(111, facecolor="#1a1a2e")
        ax.plot(sizes, train_accs, "o-", color="#22c55e", lw=2, label="Train", ms=5)
        ax.plot(sizes, val_accs,   "s-", color="#f59e0b", lw=2, label="Test",  ms=5)
        ax.fill_between(sizes, train_accs, val_accs,
                        alpha=0.08, color="#94a3b8")
        ax.set_xlabel("Training windows", fontsize=9, color="#94a3b8")
        ax.set_ylabel("Accuracy", fontsize=9, color="#94a3b8")
        ax.set_title(title, fontsize=10, color="#e2e8f0")
        ax.set_ylim(0, 1.05)
        ax.tick_params(colors="#94a3b8")
        ax.legend(framealpha=0.3, fontsize=8, labelcolor="#e2e8f0")
        for spine in ax.spines.values():
            spine.set_edgecolor("#3f3f5c")
        self._fig.tight_layout(pad=1.2)
        self.draw()


class _LiveCurveCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None):
        self._fig = Figure(facecolor="#1a1a2e")
        super().__init__(self._fig)
        self.setParent(parent)
        self._epochs: List[int] = []
        self._tl: List[float]   = []
        self._vl: List[float]   = []
        self._ta: List[float]   = []
        self._va: List[float]   = []

    def reset(self):
        self._epochs = []; self._tl = []; self._vl = []
        self._ta = []; self._va = []
        self._fig.clear(); self.draw()

    def add_epoch(self, epoch, tl, vl, ta, va):
        self._epochs.append(epoch)
        self._tl.append(tl); self._vl.append(vl)
        self._ta.append(ta); self._va.append(va)
        self._redraw()

    def _redraw(self):
        self._fig.clear()
        ax1 = self._fig.add_subplot(121, facecolor="#1a1a2e")
        ax1.plot(self._epochs, self._tl, color="#22c55e", lw=1.5, label="Train")
        ax1.plot(self._epochs, self._vl, color="#ef4444", lw=1.5, label="Val",
                 linestyle="--")
        ax1.set_title("Loss", fontsize=9, color="#e2e8f0")
        ax1.tick_params(colors="#94a3b8", labelsize=7)
        ax1.legend(fontsize=7, framealpha=0.3, labelcolor="#e2e8f0")

        ax2 = self._fig.add_subplot(122, facecolor="#1a1a2e")
        ax2.plot(self._epochs, self._ta, color="#22c55e", lw=1.5, label="Train")
        ax2.plot(self._epochs, self._va, color="#f59e0b", lw=1.5, label="Val",
                 linestyle="--")
        ax2.set_title("Accuracy", fontsize=9, color="#e2e8f0")
        ax2.set_ylim(0, 1.05)
        ax2.tick_params(colors="#94a3b8", labelsize=7)
        ax2.legend(fontsize=7, framealpha=0.3, labelcolor="#e2e8f0")

        for ax in (ax1, ax2):
            for sp in ax.spines.values():
                sp.set_edgecolor("#3f3f5c")
        self._fig.tight_layout(pad=0.8)
        self.draw()


# ─── split card widget ────────────────────────────────────────────────────────

class _SplitCardWidget(QWidget):
    """
    Visual card-based (subject × trial) split builder.

    Each row is one (subject, trial) pair. Clicking a card cycles it through
    Train → Val → Test → Skip. Quick-assign buttons on each subject row.
    """

    changed = Signal()

    _CYCLE = {"skip": "train", "train": "val", "val": "test", "test": "skip"}
    _BADGE = {
        "train": ("TRAIN", _C["train"]),
        "val":   ("VAL",   _C["val"]),
        "test":  ("TEST",  _C["test"]),
        "skip":  ("—",     _C["excl"]),
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self._assignments: Dict[Tuple[str, str], str] = {}  # (subject, trial) → role
        self._subject_trials: Dict[str, List[str]] = {}
        self._cards: Dict[Tuple[str, str], QFrame] = {}

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._container = QWidget()
        self._lay = QVBoxLayout(self._container)
        self._lay.setSpacing(5)
        self._lay.setContentsMargins(0, 0, 0, 0)
        scroll.setWidget(self._container)
        outer.addWidget(scroll)

        self._stats_lbl = QLabel("")
        self._stats_lbl.setStyleSheet(f"color:{_C['muted']};font-size:10px;padding:2px;")
        self._stats_lbl.setWordWrap(True)
        outer.addWidget(self._stats_lbl)

    def load(self, subject_trials: Dict[str, List[str]],
             default_fn=None):
        """
        subject_trials: {subject_id: [trial_label, ...]}
        default_fn: callable(subject, trial) -> role str, or None for auto-assign
        """
        self._subject_trials = subject_trials
        self._assignments.clear()
        all_subjects = sorted(subject_trials.keys())
        all_trials_per_subj = {s: sorted(trials) for s, trials in subject_trials.items()}

        for subj in all_subjects:
            trials = all_trials_per_subj[subj]
            for trial in trials:
                if default_fn:
                    role = default_fn(subj, trial)
                else:
                    role = self._auto_assign(subj, trial, trials)
                self._assignments[(subj, trial)] = role

        self._rebuild()

    def _auto_assign(self, subj: str, trial: str, all_trials: List[str]) -> str:
        """Heuristic: last trial → test, second-to-last → val, rest → train."""
        n = len(all_trials)
        idx = all_trials.index(trial)
        if n == 1:
            return "train"
        if n == 2:
            return "test" if idx == n - 1 else "train"
        if idx == n - 1:
            return "test"
        if idx == n - 2:
            return "val"
        return "train"

    def get_roles(self) -> Dict[Tuple[str, str], str]:
        return dict(self._assignments)

    def set_all(self, role: str):
        for k in self._assignments:
            self._assignments[k] = role
        self._rebuild()
        self.changed.emit()

    def _rebuild(self):
        while self._lay.count():
            item = self._lay.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._cards.clear()

        for subj in sorted(self._subject_trials.keys()):
            trials = sorted(self._subject_trials[subj])
            block = QFrame()
            block.setStyleSheet(
                f"QFrame{{background:{_C['panel']};border:1px solid {_C['border']};"
                "border-radius:6px;}}"
            )
            bl = QVBoxLayout(block)
            bl.setContentsMargins(6, 4, 6, 6)
            bl.setSpacing(4)

            # Header
            hdr = QHBoxLayout()
            lbl = QLabel(subj)
            lbl.setStyleSheet(f"font-weight:bold;font-size:11px;color:{_C['text']};border:none;")
            hdr.addWidget(lbl)
            hdr.addStretch()
            for role, color, text in [("train", _C["train"], "▶ Train"),
                                       ("val",   _C["val"],   "▶ Val"),
                                       ("test",  _C["test"],  "▶ Test"),
                                       ("skip",  _C["excl"],  "✕ Skip")]:
                b = QPushButton(text)
                b.setFixedHeight(18)
                b.setStyleSheet(
                    f"QPushButton{{background:{color}22;color:{color};"
                    f"border:1px solid {color}44;border-radius:3px;"
                    f"font-size:9px;padding:0 5px;}}"
                    f"QPushButton:hover{{background:{color}44;}}"
                )
                b.clicked.connect(lambda _, s=subj, r=role: self._set_subject(s, r))
                hdr.addWidget(b)
            bl.addLayout(hdr)

            # Trial cards
            cards_row = QHBoxLayout()
            cards_row.setSpacing(4)
            for trial in trials:
                card = self._make_card(subj, trial)
                cards_row.addWidget(card)
                self._cards[(subj, trial)] = card
            cards_row.addStretch()
            bl.addLayout(cards_row)
            self._lay.addWidget(block)

        self._lay.addStretch()
        self._update_stats()

    def _make_card(self, subj: str, trial: str) -> QFrame:
        role = self._assignments.get((subj, trial), "skip")
        badge_txt, badge_clr = self._BADGE.get(role, ("?", "#888"))

        card = QFrame()
        card.setFixedSize(68, 52)
        card.setCursor(Qt.CursorShape.PointingHandCursor)
        self._apply_card_style(card, role)

        cl = QVBoxLayout(card)
        cl.setContentsMargins(4, 3, 4, 3)
        cl.setSpacing(2)

        badge = QLabel(badge_txt)
        badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        badge.setFixedHeight(15)
        badge.setStyleSheet(
            f"background:{badge_clr};color:#0f0f1a;font-size:8px;"
            "font-weight:bold;border-radius:2px;border:none;"
        )
        cl.addWidget(badge)

        tl = QLabel(trial)
        tl.setToolTip(f"{subj} — {trial}")
        tl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        tl.setStyleSheet(f"color:{_C['text']};font-size:8px;border:none;")
        cl.addWidget(tl)

        for w in (card, badge, tl):
            w.mousePressEvent = lambda e, s=subj, t=trial: self._on_click(s, t)
        return card

    @staticmethod
    def _apply_card_style(card: QFrame, role: str):
        bg = {"train": "#1a3a2a", "val": "#3a2a10", "test": "#3a1a1a",
              "skip": _C["panel"]}.get(role, _C["panel"])
        bc = {"train": _C["train"], "val": _C["val"], "test": _C["test"],
              "skip": _C["border"]}.get(role, _C["border"])
        card.setStyleSheet(f"QFrame{{background:{bg};border:1px solid {bc};border-radius:4px;}}")

    def _on_click(self, subj: str, trial: str):
        cur = self._assignments.get((subj, trial), "skip")
        self._assignments[(subj, trial)] = self._CYCLE.get(cur, "train")
        self._rebuild()
        self.changed.emit()

    def _set_subject(self, subj: str, role: str):
        for trial in self._subject_trials.get(subj, []):
            self._assignments[(subj, trial)] = role
        self._rebuild()
        self.changed.emit()

    def _update_stats(self):
        roles = list(self._assignments.values())
        nt = roles.count("train")
        nv = roles.count("val")
        nte = roles.count("test")
        ns = roles.count("skip")
        self._stats_lbl.setText(
            f"🟢 Train: {nt}  🟡 Val: {nv}  🔴 Test: {nte}  ⬜ Skip: {ns} trials"
        )


# ─── model comparison run ─────────────────────────────────────────────────────

class _ModelRun:
    def __init__(self, name, model_type, results, model, timestamp):
        self.name       = name
        self.model_type = model_type
        self.results    = results
        self.model      = model
        self.timestamp  = timestamp


# ─── main dialog ──────────────────────────────────────────────────────────────

class QuattrocentoTrainingDialog(QDialog):
    """
    Professional ML training dialog for Quattrocento HD-EMG data (v3).

    Key improvements over v2:
    • Card-based (subject × trial) split builder — no more raw tag strings.
    • Normalisation correctly fitted on train only, applied to val+test independently.
    • F1-macro reported alongside accuracy throughout.
    • Class-weight compensation for imbalanced data.
    • Early-stopping patience control.
    • Learning curve runs in a non-blocking thread.
    • Cleaner, more logical left-panel layout.
    """

    def __init__(self, model_manager, data_dir: Path, parent=None):
        super().__init__(parent)
        self._model_manager = model_manager
        self._data_dir      = Path(data_dir)
        self._loader        = None
        self._worker: Optional[_TrainWorker]         = None
        self._lc_worker: Optional[_LearningCurveWorker] = None
        self._trained_model = None
        self._results       = None
        self._all_runs: List[_ModelRun] = []

        self.setWindowTitle("Quattrocento Training & Evaluation")
        self.setMinimumSize(1280, 820)
        from playagain_pipeline.gui.gui_style import apply_app_style
        apply_app_style(self, theme="dark")
        # Overlay a few extra rules that the training dialog needs on top of
        # the shared stylesheet (splitter handle, scrollbar, font-size tweak).
        self.setStyleSheet(self.styleSheet() + f"""
            QSplitter::handle {{ background: {_C['border']}; width: 1px; }}
            QScrollBar:vertical {{
                background: {_C['bg']}; width: 8px; margin: 0;
            }}
            QScrollBar::handle:vertical {{
                background: {_C['border']}; border-radius: 4px; min-height: 20px;
            }}
            QDialog, QWidget {{ font-size: 12px; }}
        """)
        self._setup_ui()
        self._try_auto_detect()

    # ─── UI ────────────────────────────────────────────────────────────────────

    def _setup_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        self._main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self._main_splitter.setChildrenCollapsible(True)
        root.addWidget(self._main_splitter)

        # Left panel
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setMinimumWidth(0)
        left_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._left_scroll = left_scroll
        left_content = QWidget()
        left_content.setMinimumWidth(460)
        left_lay = QVBoxLayout(left_content)
        left_lay.setContentsMargins(4, 4, 4, 4)
        left_lay.setSpacing(8)
        left_scroll.setWidget(left_content)
        self._main_splitter.addWidget(left_scroll)

        # ── 1. Data Root ─────────────────────────────────────────────────────
        root_grp = QGroupBox("1 · Data Source")
        root_grp.setStyleSheet(_PANEL_STYLE)
        rgl = QVBoxLayout(root_grp)

        dir_row = QHBoxLayout()
        self._root_edit = QLineEdit()
        self._root_edit.setPlaceholderText("Path to quattrocento folder…")
        self._root_edit.setReadOnly(True)
        dir_row.addWidget(self._root_edit)
        browse_btn = QPushButton("…")
        browse_btn.setStyleSheet(_BTN_SECONDARY)
        browse_btn.setFixedWidth(34)
        browse_btn.clicked.connect(self._on_browse)
        dir_row.addWidget(browse_btn)
        rgl.addLayout(dir_row)

        scan_row = QHBoxLayout()
        self._scan_btn = QPushButton("Scan Directory")
        self._scan_btn.setStyleSheet(_BTN_SECONDARY)
        self._scan_btn.setFixedHeight(28)
        self._scan_btn.setEnabled(False)
        self._scan_btn.clicked.connect(self._on_scan)
        scan_row.addWidget(self._scan_btn)

        for short, tip, attr, rng, default, suffix in [
            ("Win ms:", "Window length in milliseconds", "_ws_spin", (10, 2000), 200, " ms"),
            ("Ch:",     "Target channel count (0 = all)", "_nc_spin", (0, 512),  0,   ""),
        ]:
            hw_lbl = QLabel(short)
            hw_lbl.setStyleSheet(f"color:{_C['muted']};font-size:10px;")
            scan_row.addWidget(hw_lbl)
            spin = QSpinBox()
            spin.setRange(*rng)
            spin.setValue(default)
            if suffix:
                spin.setSuffix(suffix)
            if rng[0] == 0 and attr == "_nc_spin":
                spin.setSpecialValueText("all")
            spin.setMaximumWidth(72)
            spin.setToolTip(tip)
            scan_row.addWidget(spin)
            setattr(self, attr, spin)

        side_lbl = QLabel("Side:")
        side_lbl.setStyleSheet(f"color:{_C['muted']};font-size:10px;")
        scan_row.addWidget(side_lbl)
        self._side_combo = QComboBox()
        self._side_combo.addItems(["left", "right", "both"])
        self._side_combo.setMaximumWidth(60)
        self._side_combo.setToolTip("Which electrode side to load")
        scan_row.addWidget(self._side_combo)
        rgl.addLayout(scan_row)

        # Trigger / RMS segmentation controls
        seg_row = QHBoxLayout()
        self._use_trigger_cb = QCheckBox("Trigger segmentation")
        self._use_trigger_cb.setChecked(True)
        self._use_trigger_cb.setToolTip(
            "Extract individual gesture repetitions using the trigger column\n"
            "(or RMS fallback). Recommended for Quattrocento data with children.\n"
            "Each file typically contains ~5 repetitions of the gesture."
        )
        seg_row.addWidget(self._use_trigger_cb)

        delay_lbl = QLabel("Onset delay:")
        delay_lbl.setStyleSheet(f"color:{_C['muted']};font-size:10px;")
        seg_row.addWidget(delay_lbl)
        self._onset_delay_spin = QSpinBox()
        self._onset_delay_spin.setRange(0, 1000)
        self._onset_delay_spin.setValue(150)
        self._onset_delay_spin.setSuffix(" ms")
        self._onset_delay_spin.setMaximumWidth(85)
        self._onset_delay_spin.setToolTip(
            "Shift RMS-detected onset forward to compensate for\n"
            "children's reaction-time lag (default 150 ms)."
        )
        seg_row.addWidget(self._onset_delay_spin)
        seg_row.addStretch()
        rgl.addLayout(seg_row)

        self._scan_info = QLabel("No directory selected.")
        self._scan_info.setStyleSheet(f"color:{_C['muted']};font-size:10px;")
        self._scan_info.setWordWrap(True)
        rgl.addWidget(self._scan_info)
        left_lay.addWidget(root_grp)

        self._ws_spin.valueChanged.connect(self._on_config_changed)
        self._nc_spin.valueChanged.connect(self._on_config_changed)

        # ── 2. Gesture & Subject Filters ─────────────────────────────────────
        filter_grp = QGroupBox("2 · Gesture, Subject & Condition Filters")
        filter_grp.setStyleSheet(_PANEL_STYLE)
        fgl = QVBoxLayout(filter_grp)

        gest_lbl = QLabel("Gestures:")
        gest_lbl.setStyleSheet(f"color:{_C['muted']};font-size:10px;")
        fgl.addWidget(gest_lbl)
        self._gest_list = QListWidget()
        self._gest_list.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        self._gest_list.setMaximumHeight(110)
        self._gest_list.itemSelectionChanged.connect(self._update_preview)
        fgl.addWidget(self._gest_list)

        gb_row = QHBoxLayout()
        for lbl, fn in [("All", self._gest_list.selectAll),
                        ("None", self._gest_list.clearSelection)]:
            b = QPushButton(lbl); b.setStyleSheet(_BTN_SECONDARY)
            b.setFixedHeight(20); b.clicked.connect(fn); gb_row.addWidget(b)
        gb_row.addStretch()
        fgl.addLayout(gb_row)

        subj_lbl = QLabel("Subjects:")
        subj_lbl.setStyleSheet(f"color:{_C['muted']};font-size:10px;")
        fgl.addWidget(subj_lbl)
        self._subj_list = QListWidget()
        self._subj_list.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        self._subj_list.setMaximumHeight(80)
        self._subj_list.itemSelectionChanged.connect(self._update_preview)
        fgl.addWidget(self._subj_list)

        sb_row = QHBoxLayout()
        for lbl, fn in [("All", self._subj_list.selectAll),
                        ("None", self._subj_list.clearSelection)]:
            b = QPushButton(lbl); b.setStyleSheet(_BTN_SECONDARY)
            b.setFixedHeight(20); b.clicked.connect(fn); sb_row.addWidget(b)
        sb_row.addStretch()
        fgl.addLayout(sb_row)

        # Condition filter (healthy / paralysed / …)
        # Hidden by default; appears automatically after scan if conditions are found.
        self._cond_container = QWidget()
        cond_inner = QVBoxLayout(self._cond_container)
        cond_inner.setContentsMargins(0, 0, 0, 0)
        cond_inner.setSpacing(2)

        cond_lbl = QLabel("Condition:")
        cond_lbl.setStyleSheet(f"color:{_C['muted']};font-size:10px;")
        cond_inner.addWidget(cond_lbl)

        self._cond_list = QListWidget()
        self._cond_list.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        self._cond_list.setMaximumHeight(58)
        self._cond_list.itemSelectionChanged.connect(self._update_preview)
        cond_inner.addWidget(self._cond_list)

        cond_btn_row = QHBoxLayout()
        for lbl, fn in [("All", self._cond_list.selectAll),
                        ("None", self._cond_list.clearSelection)]:
            b = QPushButton(lbl); b.setStyleSheet(_BTN_SECONDARY)
            b.setFixedHeight(20); b.clicked.connect(fn); cond_btn_row.addWidget(b)
        cond_btn_row.addStretch()
        cond_inner.addLayout(cond_btn_row)

        fgl.addWidget(self._cond_container)
        self._cond_container.setVisible(False)   # shown only when conditions exist
        left_lay.addWidget(filter_grp)

        # ── 3. Split Assignment ───────────────────────────────────────────────
        split_grp = QGroupBox("3 · Train / Val / Test Assignment")
        split_grp.setStyleSheet(_PANEL_STYLE)
        split_lay = QVBoxLayout(split_grp)

        split_info = QLabel(
            "Click a card to cycle it: TRAIN → VAL → TEST → skip.\n"
            "Use the quick-assign buttons per subject row."
        )
        split_info.setStyleSheet(f"color:{_C['muted']};font-size:10px;")
        split_info.setWordWrap(True)
        split_lay.addWidget(split_info)

        intra_row = QHBoxLayout()
        intra_row.addWidget(QLabel("Intra-trial fallback:"))
        self._intra_mode_combo = QComboBox()
        self._intra_mode_combo.addItem("Auto (if split missing)", "auto")
        self._intra_mode_combo.addItem("Force ON", "force_on")
        self._intra_mode_combo.addItem("Force OFF", "force_off")
        self._intra_mode_combo.setMaximumWidth(180)
        self._intra_mode_combo.currentIndexChanged.connect(self._update_preview)
        intra_row.addWidget(self._intra_mode_combo)

        intra_row.addSpacing(8)
        intra_row.addWidget(QLabel("Ratios (%):"))
        self._split_train_pct = QDoubleSpinBox()
        self._split_train_pct.setRange(0.0, 100.0)
        self._split_train_pct.setValue(70.0)
        self._split_train_pct.setDecimals(1)
        self._split_train_pct.setSingleStep(1.0)
        self._split_train_pct.setSuffix(" T")
        self._split_train_pct.setMaximumWidth(74)
        self._split_train_pct.valueChanged.connect(self._update_preview)
        intra_row.addWidget(self._split_train_pct)

        self._split_val_pct = QDoubleSpinBox()
        self._split_val_pct.setRange(0.0, 100.0)
        self._split_val_pct.setValue(15.0)
        self._split_val_pct.setDecimals(1)
        self._split_val_pct.setSingleStep(1.0)
        self._split_val_pct.setSuffix(" V")
        self._split_val_pct.setMaximumWidth(74)
        self._split_val_pct.valueChanged.connect(self._update_preview)
        intra_row.addWidget(self._split_val_pct)

        self._split_test_pct = QDoubleSpinBox()
        self._split_test_pct.setRange(0.0, 100.0)
        self._split_test_pct.setValue(15.0)
        self._split_test_pct.setDecimals(1)
        self._split_test_pct.setSingleStep(1.0)
        self._split_test_pct.setSuffix(" Te")
        self._split_test_pct.setMaximumWidth(74)
        self._split_test_pct.valueChanged.connect(self._update_preview)
        intra_row.addWidget(self._split_test_pct)
        intra_row.addStretch()
        split_lay.addLayout(intra_row)

        # Global quick buttons
        gb2 = QHBoxLayout()
        gb2.setSpacing(4)
        for role, color, txt in [
            ("train", _C["train"], "All → Train"),
            ("val",   _C["val"],   "All → Val"),
            ("test",  _C["test"],  "All → Test"),
        ]:
            b = QPushButton(txt)
            b.setFixedHeight(22)
            b.setStyleSheet(
                f"QPushButton{{background:{color}22;color:{color};"
                f"border:1px solid {color}55;border-radius:3px;font-size:9px;}}"
                f"QPushButton:hover{{background:{color}44;}}"
            )
            b.clicked.connect(lambda _, r=role: self._split_cards.set_all(r))
            gb2.addWidget(b)
        gb2.addStretch()
        split_lay.addLayout(gb2)

        self._split_cards = _SplitCardWidget()
        self._split_cards.setMinimumHeight(160)
        self._split_cards.changed.connect(self._update_preview)
        split_lay.addWidget(self._split_cards)

        # Preview banner
        self._preview_lbl = QLabel("—")
        self._preview_lbl.setStyleSheet(
            f"color:{_C['muted']};font-size:10px;"
            f"background:{_C['card']};border:1px solid {_C['border']};"
            "border-radius:4px;padding:6px;")
        self._preview_lbl.setWordWrap(True)
        split_lay.addWidget(self._preview_lbl)
        left_lay.addWidget(split_grp)

        # ── 4. Cross-validation ───────────────────────────────────────────────
        cv_grp = QGroupBox("4 · Cross-Validation  (optional)")
        cv_grp.setStyleSheet(_PANEL_STYLE)
        cv_lay = QVBoxLayout(cv_grp)
        self._cv_btn_group = QButtonGroup(self)

        self._cv_none  = QRadioButton("None  (use split assignments above)")
        self._cv_kfold = QRadioButton("K-Fold stratified")
        self._cv_loso  = QRadioButton("Leave-One-Subject-Out")
        self._cv_none.setChecked(True)
        for btn in (self._cv_none, self._cv_kfold, self._cv_loso):
            self._cv_btn_group.addButton(btn)

        kf_row = QHBoxLayout()
        kf_row.addWidget(self._cv_kfold)
        kf_row.addWidget(QLabel("K:"))
        self._cv_k_spin = QSpinBox()
        self._cv_k_spin.setRange(2, 10); self._cv_k_spin.setValue(5)
        self._cv_k_spin.setMaximumWidth(55)
        kf_row.addWidget(self._cv_k_spin)
        kf_row.addStretch()
        cv_lay.addWidget(self._cv_none)
        cv_lay.addLayout(kf_row)
        cv_lay.addWidget(self._cv_loso)
        left_lay.addWidget(cv_grp)

        # ── 5. Preprocessing ──────────────────────────────────────────────────
        pre_grp = QGroupBox("5 · Preprocessing")
        pre_grp.setStyleSheet(_PANEL_STYLE)
        pre_form = QFormLayout(pre_grp)
        pre_form.setSpacing(5)

        self._norm_combo = QComboBox()
        self._norm_combo.addItem("Z-score per channel (recommended)", "z_channel")
        self._norm_combo.addItem("Z-score global", "z_global")
        self._norm_combo.addItem("None", "none")
        self._norm_combo.setToolTip(
            "Normalisation is always fitted on training data only,\n"
            "then applied to val and test sets to prevent data leakage.")
        pre_form.addRow("Normalise:", self._norm_combo)

        self._ch_reduce_combo = QComboBox()
        self._ch_reduce_combo.addItem("None",              "none")
        self._ch_reduce_combo.addItem("Top-N by variance", "variance_top_n")
        self._ch_reduce_combo.addItem("PCA (flatten)",     "pca")
        self._ch_reduce_combo.currentIndexChanged.connect(
            lambda: self._ch_n_spin.setEnabled(
                self._ch_reduce_combo.currentData() != "none"))
        pre_form.addRow("Channel reduce:", self._ch_reduce_combo)

        self._ch_n_spin = QSpinBox()
        self._ch_n_spin.setRange(1, 512); self._ch_n_spin.setValue(32)
        self._ch_n_spin.setEnabled(False)
        pre_form.addRow("# channels:", self._ch_n_spin)

        self._augment_cb = QCheckBox("Augment training windows")
        self._augment_cb.setToolTip("Adds a copy of the training data with Gaussian noise and/or time-shift.")
        pre_form.addRow(self._augment_cb)

        self._noise_spin = QDoubleSpinBox()
        self._noise_spin.setRange(0.0, 0.5); self._noise_spin.setSingleStep(0.005)
        self._noise_spin.setValue(0.005); self._noise_spin.setEnabled(False)
        self._noise_spin.setToolTip("Standard deviation of Gaussian noise added to each channel.")
        pre_form.addRow("Noise σ:", self._noise_spin)

        self._shift_spin = QSpinBox()
        self._shift_spin.setRange(0, 50); self._shift_spin.setValue(5)
        self._shift_spin.setSuffix(" samples"); self._shift_spin.setEnabled(False)
        self._shift_spin.setToolTip("Maximum circular time-shift applied randomly to each window.")
        pre_form.addRow("Time shift ±:", self._shift_spin)

        self._augment_cb.toggled.connect(
            lambda c: (self._noise_spin.setEnabled(c), self._shift_spin.setEnabled(c)))

        self._class_weight_combo = QComboBox()
        self._class_weight_combo.addItem("None (equal weights)",     "none")
        self._class_weight_combo.addItem("Balanced (inverse freq.)", "balanced")
        self._class_weight_combo.setToolTip(
            "Balanced weighting compensates for unequal class sizes\n"
            "by up-weighting rare classes during training.")
        pre_form.addRow("Class weights:", self._class_weight_combo)

        self._es_patience_spin = QSpinBox()
        self._es_patience_spin.setRange(0, 50); self._es_patience_spin.setValue(10)
        self._es_patience_spin.setSpecialValueText("disabled")
        self._es_patience_spin.setToolTip(
            "For deep models: stop training if val loss doesn't improve\n"
            "for this many epochs (0 = disabled).")
        pre_form.addRow("Early-stop patience:", self._es_patience_spin)

        left_lay.addWidget(pre_grp)

        # ── 6. Model ──────────────────────────────────────────────────────────
        model_grp = QGroupBox("6 · Model")
        model_grp.setStyleSheet(_PANEL_STYLE)
        model_form = QFormLayout(model_grp)
        model_form.setSpacing(5)

        self._model_combo = QComboBox()
        for display, _ in _MODEL_TYPES:
            self._model_combo.addItem(display)
        model_form.addRow("Model type:", self._model_combo)

        self._feat_combo = QComboBox()
        self._feat_combo.addItem("Default features (MAV, RMS, WL, ZC, SSC, …)")
        self._feat_combo.addItem("Raw EMG windows")
        self._feat_combo.setToolTip(
            "Feature-based models (SVM, LDA, RF, CatBoost) use hand-crafted features.\n"
            "Deep models (MLP, CNN, …) can operate on either features or raw windows.")
        model_form.addRow("Features:", self._feat_combo)

        self._model_name_edit = QLineEdit()
        self._model_name_edit.setPlaceholderText("auto-generated if empty")
        model_form.addRow("Model name:", self._model_name_edit)
        left_lay.addWidget(model_grp)

        # ── Train / Stop / Close ──────────────────────────────────────────────
        btn_row = QHBoxLayout()
        self._train_btn = QPushButton("▶  Train Model")
        self._train_btn.setStyleSheet(_BTN_PRIMARY)
        self._train_btn.setFixedHeight(38)
        self._train_btn.setEnabled(False)
        self._train_btn.clicked.connect(self._on_train)
        btn_row.addWidget(self._train_btn)

        self._stop_btn = QPushButton("■  Stop")
        self._stop_btn.setStyleSheet(_BTN_SECONDARY)
        self._stop_btn.setFixedHeight(38)
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self._on_stop_training)
        btn_row.addWidget(self._stop_btn)

        close_btn = QPushButton("Close")
        close_btn.setStyleSheet(_BTN_SECONDARY)
        close_btn.setFixedHeight(38)
        close_btn.clicked.connect(self.reject)
        btn_row.addWidget(close_btn)
        left_lay.addLayout(btn_row)

        self._progress = QProgressBar()
        self._progress.setRange(0, 100); self._progress.setValue(0)
        left_lay.addWidget(self._progress)

        self._status_lbl = QLabel("")
        self._status_lbl.setStyleSheet(f"color:{_C['muted']};font-size:10px;")
        self._status_lbl.setWordWrap(True)
        left_lay.addWidget(self._status_lbl)
        left_lay.addStretch()

        # ── Right panel: results tabs ─────────────────────────────────────────
        right_widget = QWidget()
        right_lay = QVBoxLayout(right_widget)
        right_lay.setContentsMargins(0, 0, 0, 0)
        self._main_splitter.addWidget(right_widget)

        top_right_row = QHBoxLayout()
        top_right_row.addStretch()
        self._toggle_settings_btn = QPushButton("Hide Settings")
        self._toggle_settings_btn.setStyleSheet(_BTN_SECONDARY)
        self._toggle_settings_btn.setFixedHeight(26)
        self._toggle_settings_btn.setToolTip("Collapse/expand the left settings panel")
        self._toggle_settings_btn.clicked.connect(self._toggle_settings_panel)
        top_right_row.addWidget(self._toggle_settings_btn)
        right_lay.addLayout(top_right_row)

        self._left_restore_width = 520
        self._main_splitter.setStretchFactor(0, 0)
        self._main_splitter.setStretchFactor(1, 1)
        self._main_splitter.setSizes([self._left_restore_width, 1000])

        self._results_tabs = QTabWidget()
        right_lay.addWidget(self._results_tabs)

        # Tab 0: Summary
        summary_tab = QWidget()
        sum_lay = QVBoxLayout(summary_tab)

        self._banner = QLabel("Train a model to see results")
        self._banner.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._banner.setStyleSheet(
            f"font-size:15px;font-weight:700;color:{_C['accent2']};"
            f"background:{_C['panel']};border-radius:8px;padding:12px;")
        sum_lay.addWidget(self._banner)

        metrics_row = QHBoxLayout()
        self._metric_cards: Dict[str, QLabel] = {}
        for key, label, color in [
            ("train_acc", "Train Acc",  _C["success"]),
            ("train_f1",  "Train F1",   _C["success"]),
            ("val_acc",   "Val Acc",    _C["warning"]),
            ("test_acc",  "Test Acc",   _C["accent2"]),
            ("test_f1",   "Test F1",    _C["accent2"]),
            ("cv_mean",   "CV Mean Acc",_C["accent"]),
        ]:
            card = QFrame()
            card.setStyleSheet(
                f"background:{_C['panel']};border:1px solid {_C['border']};"
                "border-radius:8px;padding:4px;")
            card_lay = QVBoxLayout(card)
            tl = QLabel(label)
            tl.setStyleSheet(f"color:{_C['muted']};font-size:9px;font-weight:600;")
            tl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            card_lay.addWidget(tl)
            vl = QLabel("—")
            vl.setStyleSheet(f"color:{color};font-size:18px;font-weight:700;")
            vl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            card_lay.addWidget(vl)
            metrics_row.addWidget(card)
            self._metric_cards[key] = vl
        sum_lay.addLayout(metrics_row)

        self._summary_lbl = QLabel("")
        self._summary_lbl.setStyleSheet(f"font-size:10px;color:{_C['muted']};padding:2px 4px;")
        sum_lay.addWidget(self._summary_lbl)

        self._imbalance_warn = QLabel("")
        self._imbalance_warn.setStyleSheet(
            f"color:{_C['warning']};font-size:10px;background:{_C['card']};"
            "border-radius:4px;padding:4px;")
        self._imbalance_warn.setWordWrap(True)
        self._imbalance_warn.setVisible(False)
        sum_lay.addWidget(self._imbalance_warn)

        self._results_table = QTableWidget()
        self._results_table.setColumnCount(7)
        self._results_table.setHorizontalHeaderLabels(
            ["Gesture", "Acc", "F1", "Prec", "Recall", "95% CI", "N windows"])
        self._results_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._results_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._results_table.setAlternatingRowColors(True)
        self._results_table.setSortingEnabled(True)
        sum_lay.addWidget(self._results_table)

        self._subj_result_lbl = QLabel("Per-subject accuracy:")
        self._subj_result_lbl.setStyleSheet(f"color:{_C['muted']};font-size:10px;")
        sum_lay.addWidget(self._subj_result_lbl)
        self._subj_table = QTableWidget()
        self._subj_table.setColumnCount(3)
        self._subj_table.setHorizontalHeaderLabels(["Subject", "Acc", "F1"])
        self._subj_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._subj_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._subj_table.setMaximumHeight(110)
        sum_lay.addWidget(self._subj_table)

        save_export_row = QHBoxLayout()
        self._save_btn = QPushButton("Save Model")
        self._save_btn.setStyleSheet(_BTN_PRIMARY)
        self._save_btn.setFixedHeight(32)
        self._save_btn.setEnabled(False)
        self._save_btn.clicked.connect(self._on_save_model)
        save_export_row.addWidget(self._save_btn)
        self._export_btn = QPushButton("📊  Export CSV")
        self._export_btn.setStyleSheet(_BTN_SECONDARY)
        self._export_btn.setFixedHeight(32)
        self._export_btn.setEnabled(False)
        self._export_btn.clicked.connect(self._on_export_csv)
        save_export_row.addWidget(self._export_btn)
        sum_lay.addLayout(save_export_row)
        self._results_tabs.addTab(summary_tab, "Summary")

        # Tab 1: Confusion matrix
        cm_tab = QWidget()
        cm_lay = QVBoxLayout(cm_tab)
        self._cm_canvas = _ConfusionMatrixCanvas()
        cm_lay.addWidget(self._cm_canvas)
        self._results_tabs.addTab(cm_tab, "Confusion Matrix")

        # Tab 2: Live training curves
        live_tab = QWidget()
        live_lay = QVBoxLayout(live_tab)
        live_info = QLabel("Loss/accuracy curves update each epoch (deep models only).")
        live_info.setStyleSheet(f"color:{_C['muted']};font-size:10px;")
        live_lay.addWidget(live_info)
        self._live_canvas = _LiveCurveCanvas()
        live_lay.addWidget(self._live_canvas)
        self._results_tabs.addTab(live_tab, "Training Curves")

        # Tab 3: Learning curve
        lc_tab = QWidget()
        lc_lay = QVBoxLayout(lc_tab)
        lc_hdr = QHBoxLayout()
        lc_hdr.addWidget(QLabel("Accuracy vs. training data size (runs in background):"))
        self._lc_btn = QPushButton("▶ Compute")
        self._lc_btn.setStyleSheet(_BTN_SECONDARY)
        self._lc_btn.setFixedHeight(28)
        self._lc_btn.setEnabled(False)
        self._lc_btn.clicked.connect(self._on_learning_curve)
        lc_hdr.addWidget(self._lc_btn)
        lc_hdr.addStretch()
        lc_lay.addLayout(lc_hdr)
        self._lc_progress = QProgressBar()
        self._lc_progress.setRange(0, 100)
        self._lc_progress.setVisible(False)
        self._lc_progress.setFixedHeight(6)
        lc_lay.addWidget(self._lc_progress)
        self._lc_canvas = _LearningCurveCanvas()
        lc_lay.addWidget(self._lc_canvas)
        self._results_tabs.addTab(lc_tab, "Learning Curve")

        # Tab 4: CV results
        cv_tab = QWidget()
        cv_lay = QVBoxLayout(cv_tab)
        self._cv_table = QTableWidget()
        self._cv_table.setColumnCount(3)
        self._cv_table.setHorizontalHeaderLabels(["Fold / Subject", "Accuracy", "F1 (macro)"])
        self._cv_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._cv_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        cv_lay.addWidget(self._cv_table)
        self._cv_summary_lbl = QLabel("")
        self._cv_summary_lbl.setStyleSheet(f"color:{_C['accent2']};font-weight:600;")
        cv_lay.addWidget(self._cv_summary_lbl)
        self._results_tabs.addTab(cv_tab, "CV Folds")

        # Tab 5: Model comparison
        cmp_tab = QWidget()
        cmp_lay = QVBoxLayout(cmp_tab)
        cmp_hdr = QHBoxLayout()
        cmp_hdr.addWidget(QLabel("All trained models this session (sorted by test F1):"))
        clear_cmp_btn = QPushButton("Clear")
        clear_cmp_btn.setStyleSheet(_BTN_SECONDARY)
        clear_cmp_btn.setFixedHeight(24)
        clear_cmp_btn.clicked.connect(self._on_clear_comparison)
        cmp_hdr.addWidget(clear_cmp_btn)
        cmp_hdr.addStretch()
        cmp_lay.addLayout(cmp_hdr)
        self._cmp_table = QTableWidget()
        self._cmp_table.setColumnCount(8)
        self._cmp_table.setHorizontalHeaderLabels(
            ["Name", "Type", "Train Acc", "Val Acc", "Test Acc", "Test F1", "CV Acc", "Time"])
        self._cmp_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._cmp_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._cmp_table.setAlternatingRowColors(True)
        cmp_lay.addWidget(self._cmp_table)
        save_best_btn = QPushButton("Save Selected Model")
        save_best_btn.setStyleSheet(_BTN_PRIMARY)
        save_best_btn.setFixedHeight(30)
        save_best_btn.clicked.connect(self._on_save_selected_from_comparison)
        cmp_lay.addWidget(save_best_btn)
        self._results_tabs.addTab(cmp_tab, "Compare Models")

        # Tab 6: Log
        log_tab = QWidget()
        log_lay = QVBoxLayout(log_tab)
        self._log_text = QTextEdit()
        self._log_text.setReadOnly(True)
        self._log_text.setStyleSheet(
            f"QTextEdit{{font-family:monospace;font-size:11px;"
            f"background:{_C['bg']};color:{_C['text']};"
            f"border:1px solid {_C['border']};border-radius:3px;padding:4px;}}")
        log_lay.addWidget(self._log_text)
        clr = QPushButton("Clear log")
        clr.setFixedHeight(22); clr.setFixedWidth(72)
        clr.setStyleSheet("font-size:10px;")
        clr.clicked.connect(self._log_text.clear)
        log_lay.addWidget(clr, alignment=Qt.AlignmentFlag.AlignRight)
        self._results_tabs.addTab(log_tab, "Log")

    # ─── helpers ──────────────────────────────────────────────────────────────

    def _try_auto_detect(self):
        try:
            from playagain_pipeline.gui.widgets.quattrocento_loader import discover_quattrocento_root
            p = discover_quattrocento_root(self._data_dir)
            if p:
                self._set_root(p)
        except Exception:
            pass

    def _primary_config(self) -> Tuple[int, int]:
        return self._ws_spin.value(), self._nc_spin.value()

    def _primary_recs(self):
        if self._loader is None:
            return []
        nc_target = self._nc_spin.value()
        if nc_target > 0:
            return [r for r in self._loader.recordings if r.n_channels == nc_target]
        return list(self._loader.recordings)

    def _selected_gestures(self) -> List[str]:
        return [item.data(Qt.ItemDataRole.UserRole)
                for item in self._gest_list.selectedItems()]

    def _selected_subjects(self) -> Optional[List[str]]:
        sel = [item.data(Qt.ItemDataRole.UserRole)
               for item in self._subj_list.selectedItems()]
        return sel if sel else None

    def _selected_conditions(self) -> Optional[List[str]]:
        sel = [item.data(Qt.ItemDataRole.UserRole)
               for item in self._cond_list.selectedItems()]
        return sel if sel else None

    def _filtered_recs(self):
        gestures   = self._selected_gestures()
        subjects   = self._selected_subjects()
        conditions = self._selected_conditions()
        recs = [r for r in self._primary_recs() if r.gesture in gestures]
        if subjects:   recs = [r for r in recs if r.subject_id in subjects]
        if conditions: recs = [r for r in recs if r.condition in conditions]
        return recs

    def _toggle_settings_panel(self):
        sizes = self._main_splitter.sizes() if hasattr(self, "_main_splitter") else []
        if len(sizes) < 2:
            return
        left_size = sizes[0]
        total = max(sum(sizes), 1)
        if left_size <= 24:
            restore = max(360, int(getattr(self, "_left_restore_width", 520)))
            self._main_splitter.setSizes([restore, max(300, total - restore)])
            self._toggle_settings_btn.setText("Hide Settings")
        else:
            self._left_restore_width = left_size
            self._main_splitter.setSizes([0, total])
            self._toggle_settings_btn.setText("Show Settings")

    def _effective_split(self, recs: list, roles: Dict[Tuple[str, str], str]):
        train_recs = [r for r in recs if roles.get((r.subject_id, r.trial_label)) == "train"]
        val_recs   = [r for r in recs if roles.get((r.subject_id, r.trial_label)) == "val"]
        test_recs  = [r for r in recs if roles.get((r.subject_id, r.trial_label)) == "test"]

        included = [r for r in recs
                    if roles.get((r.subject_id, r.trial_label)) in {"train", "val", "test"}]
        if not included:
            return train_recs, val_recs, test_recs, False

        mode = self._intra_mode_combo.currentData() if hasattr(self, "_intra_mode_combo") else "auto"
        if mode == "force_off":
            return train_recs, val_recs, test_recs, False
        if mode == "force_on":
            return included, included, included, True

        subj_trials: Dict[str, set] = {}
        for r in included:
            subj_trials.setdefault(r.subject_id, set()).add(r.trial_label)
        low_trial_subjects = bool(subj_trials) and all(len(trials) <= 2 for trials in subj_trials.values())
        missing_split = not train_recs or not val_recs or not test_recs
        use_intra_split = low_trial_subjects and missing_split
        if use_intra_split:
            return included, included, included, True
        return train_recs, val_recs, test_recs, False

    def _get_intra_split_ratios(self) -> Tuple[float, float, float]:
        train = float(getattr(self, "_split_train_pct", None).value()) if hasattr(self, "_split_train_pct") else 70.0
        val = float(getattr(self, "_split_val_pct", None).value()) if hasattr(self, "_split_val_pct") else 15.0
        test = float(getattr(self, "_split_test_pct", None).value()) if hasattr(self, "_split_test_pct") else 15.0
        total = train + val + test
        if total <= 0:
            return _DEFAULT_INTRA_SPLIT_RATIOS
        return train / total, val / total, test / total

    @staticmethod
    def _count_split_windows(total_windows: int, split_role: str,
                             ratios: Tuple[float, float, float]) -> int:
        tr_end, va_end = _split_bounds(total_windows, ratios)
        if split_role == "train":
            return tr_end
        if split_role == "val":
            return max(0, va_end - tr_end)
        if split_role == "test":
            return max(0, total_windows - va_end)
        return 0

    # ─── slots ────────────────────────────────────────────────────────────────

    @Slot()
    def _on_config_changed(self):
        if self._loader is not None:
            self._loader = None
            self._gest_list.clear()
            self._subj_list.clear()
            self._cond_list.clear()
            self._cond_container.setVisible(False)
            self._train_btn.setEnabled(False); self._lc_btn.setEnabled(False)
            if self._root_edit.text():
                self._scan_info.setText("⚠ Config changed — click 'Scan Directory' to reload.")

    @Slot()
    def _on_browse(self):
        p = QFileDialog.getExistingDirectory(
            self, "Select Quattrocento Data Root", str(self._data_dir))
        if p:
            self._set_root(Path(p))

    def _set_root(self, path: Path):
        self._root_edit.setText(str(path))
        self._scan_btn.setEnabled(True)
        self._scan_info.setText(f"Click 'Scan' to discover recordings in:\n{path}")

    @Slot()
    def _on_scan(self):
        from playagain_pipeline.gui.widgets.quattrocento_loader import QuattrocentoSubjectLoader
        from playagain_pipeline.gui.widgets.busy_overlay import run_blocking
        root = Path(self._root_edit.text())
        window_ms, nc_target = self._primary_config()
        side = self._side_combo.currentText() if hasattr(self, "_side_combo") else "left"
        self._scan_btn.setEnabled(False)
        self._scan_info.setText("Scanning…")

        def _do_scan():
            loader = QuattrocentoSubjectLoader(root, side=side, window_ms=window_ms)
            all_recs = loader.scan(verbose=False)
            return loader, all_recs

        def _scan_done(result):
            loader, all_recs = result
            self._scan_btn.setEnabled(True)
            self._finish_scan(loader, all_recs, nc_target, window_ms)

        def _scan_error(tb):
            self._scan_btn.setEnabled(True)
            self._scan_info.setText(f"Scan error — see log")
            self._log(f"Scan error:\n{tb}")

        run_blocking(self, _do_scan, _scan_done, _scan_error, label="Scanning directory…")

    def _finish_scan(self, loader, all_recs, nc_target, window_ms):
        try:
            primary = [r for r in all_recs if r.n_channels == nc_target] if nc_target > 0 else all_recs
            skipped = len(all_recs) - len(primary)
            self._loader = loader

            gestures = sorted({r.gesture for r in primary},
                              key=lambda g: (_DEFAULT_LABEL_ORDER.index(g)
                                              if g in _DEFAULT_LABEL_ORDER else 999))
            self._gest_list.clear()
            for g in gestures:
                recs_g = [r for r in primary if r.gesture == g]
                n = len(recs_g)
                total_reps = sum(r.n_repetitions for r in recs_g)
                has_trig = sum(1 for r in recs_g if r.has_trigger)
                trig_str = f"trig={has_trig}/{n}" if has_trig > 0 else f"rms only"
                item = QListWidgetItem(f"{g}  ({n} files, ~{total_reps} reps, {trig_str})")
                item.setData(Qt.ItemDataRole.UserRole, g)
                self._gest_list.addItem(item)
            self._gest_list.selectAll()

            subjects = sorted({r.subject_id for r in primary})
            self._subj_list.clear()
            for s in subjects:
                item = QListWidgetItem(s)
                item.setData(Qt.ItemDataRole.UserRole, s)
                self._subj_list.addItem(item)
            self._subj_list.selectAll()

            # Condition filter (only shown when the new folder layout is used)
            conditions = sorted({r.condition for r in primary if r.condition})
            self._cond_list.clear()
            for cond in conditions:
                n_cond = sum(1 for r in primary if r.condition == cond)
                item = QListWidgetItem(f"{cond}  ({n_cond} files)")
                item.setData(Qt.ItemDataRole.UserRole, cond)
                self._cond_list.addItem(item)
            self._cond_list.selectAll()
            self._cond_container.setVisible(bool(conditions))

            subj_trials: Dict[str, List[str]] = {}
            for r in primary:
                subj_trials.setdefault(r.subject_id, [])
                if r.trial_label not in subj_trials[r.subject_id]:
                    subj_trials[r.subject_id].append(r.trial_label)
            self._split_cards.load(subj_trials)

            trial_labels = sorted({r.trial_label for r in primary})
            n_ch_found = sorted({r.n_channels for r in primary})
            n_trigger = sum(1 for r in primary if r.has_trigger)
            total_reps_all = sum(r.n_repetitions for r in primary)
            seg_method = "trigger" if n_trigger > len(primary) * 0.5 else "RMS (no trigger)"
            side = getattr(self._side_combo, 'currentText', lambda: 'left')()

            info = (f"Found {len(primary)} CSV files  ({window_ms} ms windows)\n"
                    f"Channels: {n_ch_found}  |  Side: {side}\n"
                    f"Subjects: {', '.join(subjects)}\n"
                    f"Trial labels: {', '.join(trial_labels)}\n"
                    f"Repetitions: ~{total_reps_all} total  |  Seg: {seg_method}\n"
                    f"Trigger files: {n_trigger}/{len(primary)}")
            if skipped:
                info += f"\n(skipped {skipped} files with different channel config)"
            self._scan_info.setText(info)
            self._train_btn.setEnabled(True)
            self._lc_btn.setEnabled(False)
            self._update_preview()
            self._log(f"Scan complete: {len(primary)} files, {len(gestures)} gestures, "
                      f"~{total_reps_all} reps, trigger={n_trigger}/{len(primary)}")
        except Exception as e:
            self._scan_info.setText(f"Finish-scan error: {e}")
            self._log(f"Finish-scan error: {e}")

    def _update_preview(self):
        if self._loader is None:
            return
        recs = self._filtered_recs()
        if not recs:
            self._preview_lbl.setText("No data matched current filters.")
            return

        roles = self._split_cards.get_roles()
        train_recs, val_recs, test_recs, use_intra_split = self._effective_split(recs, roles)
        ratios = self._get_intra_split_ratios()
        if use_intra_split:
            train_wins = sum(self._count_split_windows(r.n_windows, "train", ratios) for r in train_recs)
            val_wins   = sum(self._count_split_windows(r.n_windows, "val", ratios)   for r in val_recs)
            test_wins  = sum(self._count_split_windows(r.n_windows, "test", ratios)  for r in test_recs)
        else:
            train_wins = sum(r.n_windows for r in train_recs)
            val_wins   = sum(r.n_windows for r in val_recs)
            test_wins  = sum(r.n_windows for r in test_recs)
        total = train_wins + val_wins + test_wins

        gestures = self._selected_gestures()

        # Class balance check
        imbalance_txt = ""
        if train_wins > 0:
            counts_per_class = {}
            for r in train_recs:
                n_train = (self._count_split_windows(r.n_windows, "train", ratios)
                           if use_intra_split else r.n_windows)
                counts_per_class[r.gesture] = counts_per_class.get(r.gesture, 0) + n_train
            if counts_per_class:
                max_c = max(counts_per_class.values())
                min_c = min(counts_per_class.values())
                ratio = max_c / max(min_c, 1)
                if ratio > 2.5:
                    imbalance_txt = (
                        f"⚠ Class imbalance detected (ratio {ratio:.1f}×). "
                        "Consider enabling 'Balanced' class weights."
                    )

        self._imbalance_warn.setVisible(bool(imbalance_txt))
        self._imbalance_warn.setText(imbalance_txt)

        if use_intra_split:
            p_train, p_val, p_test = [100.0 * r for r in ratios]
            split_note = f"\nIntra-trial split active: {p_train:.1f}% / {p_val:.1f}% / {p_test:.1f}%."
        else:
            split_note = ""
        txt = (f"Train: {train_wins:,}  |  Val: {val_wins:,}  |  Test: {test_wins:,}  "
               f"(total {total:,})\n{len(gestures)} classes, "
               f"{len({r.subject_id for r in recs})} subjects"
               f"{split_note}")
        self._preview_lbl.setText(txt)

    @Slot()
    def _on_train(self):
        if self._loader is None:
            return
        gestures = self._selected_gestures()
        if not gestures:
            QMessageBox.warning(self, "No gestures", "Select at least one gesture.")
            return

        recs = self._filtered_recs()
        if not recs:
            QMessageBox.warning(self, "No data", "No recordings matched current filters.")
            return

        roles = self._split_cards.get_roles()
        train_recs, val_recs, test_recs, use_intra_split = self._effective_split(recs, roles)
        split_ratios = self._get_intra_split_ratios()

        if not train_recs:
            QMessageBox.warning(self, "No training data",
                                "No trials are assigned to TRAIN.\n"
                                "Click cards in the split panel to assign roles.")
            return

        cv_map = {self._cv_none: "none", self._cv_kfold: "kfold", self._cv_loso: "loso"}
        cv_mode = next(k for btn, k in cv_map.items() if btn.isChecked())
        norm_mode    = self._norm_combo.currentData()
        ch_reduce    = self._ch_reduce_combo.currentData()
        augment      = self._augment_cb.isChecked()
        class_weight = self._class_weight_combo.currentData()
        mi           = self._model_combo.currentIndex()
        model_type   = _MODEL_TYPES[mi][1]
        ts_str       = datetime.now().strftime("%H%M%S")
        model_name   = (self._model_name_edit.text().strip() or
                        f"q4_{model_type}_{ts_str}")

        try:
            model = self._model_manager.create_model(model_type, name=model_name)
        except Exception as e:
            QMessageBox.critical(self, "Model creation failed", str(e))
            return

        feat_cfg = ({"mode": "raw", "features": []}
                    if self._feat_combo.currentIndex() == 1
                    else {"mode": "default", "features": []})

        sr         = float(np.median([r.sampling_rate for r in recs]))
        window_ms, nc_target = self._primary_config()
        ws         = max(1, int(round(sr * window_ms / 1000.0)))
        nc         = nc_target if nc_target > 0 else recs[0].n_channels

        self._log("=" * 58)
        self._log(f"Model: {model_name}  |  Type: {model_type}")
        self._log(f"SR: {sr:.0f} Hz  |  Window: {ws} smp ({window_ms} ms)  |  Channels: {nc}")
        self._log(f"Train files: {len(train_recs)}  |  Val: {len(val_recs)}  |  Test: {len(test_recs)}")
        self._log(f"Norm: {norm_mode}  |  Augment: {augment}  |  CV: {cv_mode}  |  CW: {class_weight}")
        if use_intra_split:
            self._log(
                "Using intra-trial split: "
                f"{split_ratios[0]*100:.1f}% train / {split_ratios[1]*100:.1f}% val / "
                f"{split_ratios[2]*100:.1f}% test per recording."
            )

        self._train_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._save_btn.setEnabled(False)
        self._export_btn.setEnabled(False)
        self._lc_btn.setEnabled(False)
        self._progress.setValue(0)
        self._live_canvas.reset()
        self._cv_table.setRowCount(0)

        use_trigger = self._use_trigger_cb.isChecked() if hasattr(self, '_use_trigger_cb') else False
        onset_delay = float(self._onset_delay_spin.value()) if hasattr(self, '_onset_delay_spin') else 150.0
        self._worker = _TrainWorker(
            loader=self._loader,
            train_recs=train_recs, test_recs=test_recs, val_recs=val_recs,
            model=model, feature_config=feat_cfg,
            sampling_rate=sr, window_size=ws, n_channels=nc,
            cv_mode=cv_mode, cv_folds=self._cv_k_spin.value(),
            norm_mode=norm_mode, augment=augment,
            noise_std=self._noise_spin.value(),
            time_shift=self._shift_spin.value(),
            channel_reduce=ch_reduce,
            channel_reduce_n=self._ch_n_spin.value(),
            class_weight=class_weight,
            early_stopping_patience=self._es_patience_spin.value(),
            use_trigger_segments=use_trigger,
            onset_delay_ms=onset_delay,
            intra_trial_split=use_intra_split,
            split_ratios=split_ratios,
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.epoch_update.connect(self._on_epoch_update)
        self._worker.fold_result.connect(self._on_fold_result)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    @Slot()
    def _on_stop_training(self):
        if self._worker and self._worker.isRunning():
            self._worker.terminate()
            self._worker.wait(3000)
            self._worker = None
        self._status_lbl.setText("Training stopped.")
        self._train_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._lc_btn.setEnabled(self._trained_model is not None)
        self._log("Training stopped by user.")

    @Slot(int, str)
    def _on_progress(self, pct: int, msg: str):
        self._progress.setValue(pct)
        self._status_lbl.setText(msg)
        if pct % 10 == 0 or pct >= 100:
            self._log(msg)

    @Slot(int, float, float, float, float)
    def _on_epoch_update(self, epoch, tl, vl, ta, va):
        self._live_canvas.add_epoch(epoch, tl, vl, ta, va)

    @Slot(int, float, float)
    def _on_fold_result(self, fold_i: int, acc: float, f1: float):
        row = self._cv_table.rowCount()
        self._cv_table.insertRow(row)
        cv_mode = "kfold" if self._cv_kfold.isChecked() else "loso"
        label = f"Fold {fold_i+1}" if cv_mode == "kfold" else f"Subject {fold_i+1}"
        self._cv_table.setItem(row, 0, QTableWidgetItem(label))
        for col, val, name in [(1, acc, "Acc"), (2, f1, "F1")]:
            it = QTableWidgetItem(f"{val:.2%}")
            it.setForeground(QColor(
                _C["success"] if val >= 0.80 else
                _C["warning"] if val >= 0.60 else _C["danger"]))
            self._cv_table.setItem(row, col, it)

    @Slot(object, object)
    def _on_finished(self, results: dict, model):
        self._results       = results
        self._trained_model = model
        self._train_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._save_btn.setEnabled(True)
        self._export_btn.setEnabled(True)
        self._lc_btn.setEnabled(True)

        tr  = results["train_acc"]
        tf1 = results["train_f1"]
        val = results["val_acc"]
        tst = results["test_acc"]
        tsf = results["test_f1"]
        t   = results["train_time"]
        n_tr = results["n_train"]
        n_va = results["n_val"]
        n_te = results["n_test"]
        cv_m = results.get("cv_mean_acc")
        cv_s = results.get("cv_std_acc")

        self._metric_cards["train_acc"].setText(f"{tr:.1%}")
        self._metric_cards["train_f1"].setText(f"{tf1:.1%}")
        self._metric_cards["val_acc"].setText(f"{val:.1%}" if n_va > 0 else "N/A")
        self._metric_cards["test_acc"].setText(f"{tst:.1%}" if n_te > 0 else "N/A")
        self._metric_cards["test_f1"].setText(f"{tsf:.1%}" if n_te > 0 else "N/A")
        self._metric_cards["cv_mean"].setText(
            f"{cv_m:.1%} ±{cv_s:.1%}" if cv_m is not None else "—")

        banner = (f"Test Acc {tst:.2%}  |  Test F1 {tsf:.2%}"
                  if n_te > 0 else f"Train Acc {tr:.2%}  |  Train F1 {tf1:.2%}")
        self._banner.setText(banner)

        self._summary_lbl.setText(
            f"Train:{n_tr:,}  Val:{n_va:,}  Test:{n_te:,}  |  "
            f"Time: {t:.1f}s  |  "
            f"Norm: {results['norm_mode']}  |  "
            f"CW: {results['class_weight']}  |  "
            f"Channels: {results['n_ch_final']}")

        self._log("─" * 58)
        self._log("RESULTS")
        self._log(f"  Train acc/F1  : {tr:.2%} / {tf1:.2%}  ({n_tr:,} windows)")
        self._log(f"  Val   acc     : {val:.2%}  ({n_va:,} windows)")
        self._log(f"  Test  acc/F1  : {tst:.2%} / {tsf:.2%}  ({n_te:,} windows)")
        if cv_m is not None:
            self._log(f"  CV    mean±std: {cv_m:.2%} ±{cv_s:.2%}")
        self._log(f"  Train time    : {t:.2f}s")

        # Per-class table
        self._results_table.setSortingEnabled(False)
        self._results_table.setRowCount(0)
        label_names = results.get("label_names", {})
        ci_low  = results.get("ci_low", {})
        ci_high = results.get("ci_high", {})
        for cls_id in results["classes"]:
            acc  = results["per_class_acc"].get(cls_id, 0.0)
            f1c  = results["per_class_f1"].get(cls_id, 0.0)
            prec = results["per_class_prec"].get(cls_id, 0.0)
            rec  = results["per_class_rec"].get(cls_id, 0.0)
            name = label_names.get(cls_id, str(cls_id))
            row  = self._results_table.rowCount()
            self._results_table.insertRow(row)
            self._results_table.setItem(row, 0, QTableWidgetItem(name))
            color = (_C["success"] if acc >= 0.80 else
                     _C["warning"] if acc >= 0.60 else _C["danger"])
            for col, val_f in [(1, acc), (2, f1c), (3, prec), (4, rec)]:
                it = QTableWidgetItem(f"{val_f:.1%}")
                it.setForeground(QColor(color))
                self._results_table.setItem(row, col, it)
            lo = ci_low.get(cls_id, acc); hi = ci_high.get(cls_id, acc)
            self._results_table.setItem(row, 5, QTableWidgetItem(f"[{lo:.1%}, {hi:.1%}]"))
            cm = results["confusion_matrix"]
            ci = results["classes"].index(cls_id)
            self._results_table.setItem(row, 6, QTableWidgetItem(str(cm[ci].sum())))
            self._log(f"  {name:18s}: acc={acc:.1%}  f1={f1c:.1%}  CI=[{lo:.1%},{hi:.1%}]  n={cm[ci].sum()}")
        self._results_table.setSortingEnabled(True)

        # Per-subject table
        self._subj_table.setRowCount(0)
        for subj, metrics in sorted(results.get("per_subject", {}).items()):
            r = self._subj_table.rowCount()
            self._subj_table.insertRow(r)
            self._subj_table.setItem(r, 0, QTableWidgetItem(subj))
            for c, k in [(1, "acc"), (2, "f1")]:
                v = metrics.get(k, 0.0)
                it = QTableWidgetItem(f"{v:.1%}")
                it.setForeground(QColor(
                    _C["success"] if v >= 0.80 else
                    _C["warning"] if v >= 0.60 else _C["danger"]))
                self._subj_table.setItem(r, c, it)

        # CV summary
        cv_accs = results.get("cv_results_acc", [])
        cv_f1s  = results.get("cv_results_f1",  [])
        if cv_accs:
            self._cv_summary_lbl.setText(
                f"CV acc: mean={cv_m:.2%}  std={cv_s:.2%}  "
                f"min={min(cv_accs):.2%}  max={max(cv_accs):.2%}  |  "
                f"CV F1: mean={np.mean(cv_f1s):.2%}")

        # Confusion matrix
        classes    = results["classes"]
        names_ord  = [label_names.get(c, str(c)) for c in classes]
        self._cm_canvas.plot(results["confusion_matrix"], names_ord,
                             title=f"Confusion Matrix  |  Test acc: {tst:.1%}  F1: {tsf:.1%}")

        # Comparison table
        run = _ModelRun(
            name=model.name if model else "?",
            model_type=_MODEL_TYPES[self._model_combo.currentIndex()][0],
            results=results, model=model,
            timestamp=datetime.now().strftime("%H:%M:%S"))
        self._all_runs.append(run)
        self._refresh_comparison()
        self._results_tabs.setCurrentIndex(0)

    @Slot(str)
    def _on_error(self, msg: str):
        self._train_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._lc_btn.setEnabled(self._trained_model is not None)
        self._log(f"ERROR: {msg}")
        QMessageBox.critical(self, "Training failed", msg[:800])

    @Slot()
    def _on_save_model(self):
        if self._trained_model is None:
            return
        try:
            save_path = self._model_manager.models_dir / self._trained_model.name
            self._trained_model.save(save_path)
            self._log(f"Model saved → {save_path}")
            QMessageBox.information(self, "Saved",
                f"Model '{self._trained_model.name}' saved.\n"
                "It will appear in the Training tab after Refresh.")
            if hasattr(self.parent(), "_refresh_models"):
                self.parent()._refresh_models()
        except Exception as e:
            self._log(f"Save error: {e}")
            QMessageBox.critical(self, "Save failed", str(e))

    @Slot()
    def _on_export_csv(self):
        if self._results is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Results CSV", str(self._data_dir / "results.csv"),
            "CSV Files (*.csv)")
        if not path:
            return
        try:
            with open(path, "w", newline="") as fh:
                wr = csv.writer(fh)
                r = self._results
                ln = r.get("label_names", {})
                wr.writerow(["Metric", "Value"])
                for k, v in [("train_acc", r["train_acc"]), ("train_f1", r["train_f1"]),
                              ("val_acc",   r["val_acc"]),
                              ("test_acc",  r["test_acc"]), ("test_f1", r["test_f1"]),
                              ("train_time", f"{r['train_time']:.2f}s"),
                              ("n_train",    r["n_train"]),
                              ("n_val",      r["n_val"]),
                              ("n_test",     r["n_test"])]:
                    wr.writerow([k, f"{v:.4f}" if isinstance(v, float) else v])
                if r.get("cv_mean_acc") is not None:
                    wr.writerow(["cv_mean_acc", f"{r['cv_mean_acc']:.4f}"])
                    wr.writerow(["cv_std_acc",  f"{r['cv_std_acc']:.4f}"])
                    wr.writerow(["cv_mean_f1",  f"{r.get('cv_mean_f1', 0):.4f}"])
                wr.writerow([])
                wr.writerow(["Class", "Accuracy", "F1", "Precision", "Recall",
                             "CI_low", "CI_high", "N_windows"])
                for cls_id in r["classes"]:
                    name = ln.get(cls_id, str(cls_id))
                    cm   = r["confusion_matrix"]
                    ci   = r["classes"].index(cls_id)
                    wr.writerow([
                        name,
                        f"{r['per_class_acc'].get(cls_id, 0):.4f}",
                        f"{r['per_class_f1'].get(cls_id, 0):.4f}",
                        f"{r['per_class_prec'].get(cls_id, 0):.4f}",
                        f"{r['per_class_rec'].get(cls_id, 0):.4f}",
                        f"{r.get('ci_low', {}).get(cls_id, 0):.4f}",
                        f"{r.get('ci_high', {}).get(cls_id, 0):.4f}",
                        cm[ci].sum(),
                    ])
                wr.writerow([])
                wr.writerow(["cv_fold", "accuracy", "f1"])
                for i, (a, f) in enumerate(zip(r.get("cv_results_acc", []),
                                               r.get("cv_results_f1", []))):
                    wr.writerow([i+1, f"{a:.4f}", f"{f:.4f}"])
            self._log(f"Results exported → {path}")
            QMessageBox.information(self, "Exported", f"Saved to:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Export failed", str(e))

    @Slot()
    def _on_learning_curve(self):
        if self._loader is None or self._trained_model is None:
            QMessageBox.information(self, "Learning Curve",
                "Train a model first, then compute the learning curve.")
            return
        if self._lc_worker and self._lc_worker.isRunning():
            return

        recs = self._filtered_recs()
        roles = self._split_cards.get_roles()
        train_recs, _, test_recs, _ = self._effective_split(recs, roles)
        if not train_recs or not test_recs:
            QMessageBox.warning(self, "No data", "Need both TRAIN and TEST trials assigned.")
            return

        window_ms, nc_target = self._primary_config()
        sr = float(np.median([r.sampling_rate for r in recs]))
        ws = max(1, int(round(sr * window_ms / 1000.0)))
        nc = nc_target if nc_target > 0 else (recs[0].n_channels if recs else 64)
        mi = self._model_combo.currentIndex()
        model_type = _MODEL_TYPES[mi][1]
        feat_cfg = ({"mode": "raw"} if self._feat_combo.currentIndex() == 1
                    else {"mode": "default"})

        self._lc_progress.setVisible(True)
        self._lc_btn.setEnabled(False)
        self._results_tabs.setCurrentIndex(3)

        self._lc_worker = _LearningCurveWorker(
            loader=self._loader,
            train_recs=train_recs, test_recs=test_recs,
            model_manager=self._model_manager, model_type=model_type,
            ws=ws, nc=nc, sr=sr, norm_mode=self._norm_combo.currentData(),
            feat_cfg=feat_cfg, label_order=_DEFAULT_LABEL_ORDER,
        )
        self._lc_worker.progress.connect(
            lambda p, m: (self._lc_progress.setValue(p), self._status_lbl.setText(m)))
        self._lc_worker.finished.connect(self._on_lc_finished)
        self._lc_worker.error.connect(lambda e: (
            self._log(f"LC error: {e}"),
            self._lc_btn.setEnabled(True),
            self._lc_progress.setVisible(False),
        ))
        self._lc_worker.start()

    @Slot(list, list, list)
    def _on_lc_finished(self, sizes, tr_accs, te_accs):
        model_type = _MODEL_TYPES[self._model_combo.currentIndex()][0]
        self._lc_canvas.plot(sizes, tr_accs, te_accs,
                             title=f"Learning Curve — {model_type}")
        self._lc_progress.setVisible(False)
        self._lc_btn.setEnabled(True)
        self._status_lbl.setText("Learning curve ready.")

    def _refresh_comparison(self):
        runs_sorted = sorted(self._all_runs,
                             key=lambda r: r.results["test_f1"], reverse=True)
        self._cmp_table.setRowCount(0)
        for i, run in enumerate(runs_sorted):
            r = run.results
            row = self._cmp_table.rowCount()
            self._cmp_table.insertRow(row)
            self._cmp_table.setItem(row, 0, QTableWidgetItem(run.name))
            self._cmp_table.setItem(row, 1, QTableWidgetItem(run.model_type))
            for col, val_f in enumerate([r["train_acc"], r["val_acc"],
                                          r["test_acc"], r["test_f1"]], start=2):
                it = QTableWidgetItem(f"{val_f:.1%}")
                it.setForeground(QColor(
                    _C["success"] if val_f >= 0.80 else
                    _C["warning"] if val_f >= 0.60 else _C["danger"]))
                self._cmp_table.setItem(row, col, it)
            cv_m = r.get("cv_mean_acc")
            self._cmp_table.setItem(row, 6, QTableWidgetItem(
                f"{cv_m:.1%}" if cv_m is not None else "—"))
            self._cmp_table.setItem(row, 7, QTableWidgetItem(f"{r['train_time']:.1f}s"))
            if i == 0:
                for c in range(self._cmp_table.columnCount()):
                    it = self._cmp_table.item(row, c)
                    if it:
                        it.setBackground(QBrush(QColor("#1a2e1a")))

    @Slot()
    def _on_clear_comparison(self):
        self._all_runs.clear()
        self._cmp_table.setRowCount(0)

    @Slot()
    def _on_save_selected_from_comparison(self):
        row = self._cmp_table.currentRow()
        if row < 0:
            QMessageBox.information(self, "No selection",
                                    "Select a row in the comparison table first.")
            return
        runs_sorted = sorted(self._all_runs,
                             key=lambda r: r.results["test_f1"], reverse=True)
        if row >= len(runs_sorted):
            return
        run = runs_sorted[row]
        try:
            save_path = self._model_manager.models_dir / run.model.name
            run.model.save(save_path)
            self._log(f"Saved '{run.model.name}' from comparison → {save_path}")
            if hasattr(self.parent(), "_refresh_models"):
                self.parent()._refresh_models()
            QMessageBox.information(self, "Saved", f"Model '{run.model.name}' saved.")
        except Exception as e:
            QMessageBox.critical(self, "Save failed", str(e))

    def _log(self, msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        msg_lower = msg.lower()
        if "error" in msg_lower:
            color = _C["danger"]
        elif any(k in msg_lower for k in ("complete", "saved", "done", "result", "acc", "f1")):
            color = _C["success"]
        elif any(k in msg_lower for k in ("warning", "warn", "skip", "⚠")):
            color = _C["warning"]
        else:
            color = _C["text"]
        ts_html  = f'<span style="color:{_C["muted"]};">[{ts}]</span>'
        msg_html = f'<span style="color:{color};">{msg}</span>'
        self._log_text.append(f"{ts_html} {msg_html}")
        self._log_text.verticalScrollBar().setValue(
            self._log_text.verticalScrollBar().maximum())

    def closeEvent(self, event):
        for w in (self._worker, self._lc_worker):
            if w and w.isRunning():
                w.terminate()
                w.wait(3000)
        super().closeEvent(event)