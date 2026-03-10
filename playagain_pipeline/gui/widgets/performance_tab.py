"""
Performance Review Tab  (v2)
────────────────────────────
Improvements over v1:
  • Card-based session picker: each subject gets a collapsible block with
    per-subject "All TRAIN / All VAL / Exclude" quick-assign buttons and
    individual clickable session cards that show a coloured role badge.
  • Live training plot for deep-learning models (MLP / CNN / AttentionNet /
    MSTNet): an inline pyqtgraph panel with loss + accuracy curves that
    update epoch-by-epoch while the model trains.
  • Global "All → TRAIN" / "All → VAL" buttons for quick splits.
"""

from __future__ import annotations

import re
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pyqtgraph as pg
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QColor, QFont, QPixmap
from PySide6.QtWidgets import (
    QButtonGroup, QCheckBox, QFormLayout, QFrame, QGridLayout,
    QGroupBox, QHBoxLayout, QLabel, QMessageBox, QProgressBar,
    QPushButton, QRadioButton, QScrollArea, QSizePolicy,
    QSpinBox, QSplitter, QTabWidget, QTextEdit, QVBoxLayout, QWidget,
)

# ── project root on path ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from playagain_pipeline.core.data_manager import DataManager
from playagain_pipeline.models.classifier import ModelManager

# ── constants ─────────────────────────────────────────────────────────────
_DEEP_MODELS = {"mlp", "cnn", "attention_net", "mstnet"}

_C = {
    "bg":     "#1e1e2e", "panel":  "#2a2a3e", "card":   "#313145",
    "train":  "#22c55e", "val":    "#f59e0b", "use":    "#38bdf8",
    "excl":   "#6b7280", "accent": "#7c3aed", "text":   "#e2e8f0",
    "sub":    "#94a3b8", "border": "#3f3f5a",
}


# ═══════════════════════════════════════════════════════════════════════════
# Background worker
# ═══════════════════════════════════════════════════════════════════════════

class ComparisonWorker(QThread):
    """Runs model_comparison.run_comparison in a background thread."""

    log_line   = Signal(str)
    epoch_data = Signal(str, int, float, float, float, float)
    # (model_name, epoch, train_loss, val_loss, train_acc, val_acc)
    finished   = Signal(dict)
    error      = Signal(str)

    def __init__(
        self,
        data_dir: Path,
        model_types: List[str],
        mode: str,
        train_sessions: Optional[List[Any]],
        val_sessions:   Optional[List[Any]],
        cv_sessions:    Optional[List[Any]],
        cv_folds: int,
        window_size_ms:  int,
        window_stride_ms: int,
        parent=None,
    ):
        super().__init__(parent)
        self._data_dir         = data_dir
        self._model_types      = model_types
        self._mode             = mode
        self._train_sessions   = train_sessions
        self._val_sessions     = val_sessions
        self._cv_sessions      = cv_sessions
        self._cv_folds         = cv_folds
        self._window_size_ms   = window_size_ms
        self._window_stride_ms = window_stride_ms

    def run(self):
        try:
            import builtins, time, warnings
            import numpy as np
            from sklearn.metrics import (
                accuracy_score, f1_score, precision_score,
                recall_score, confusion_matrix, classification_report,
            )

            original_print = builtins.print
            worker_ref = self

            def _qt_print(*args, **kwargs):
                worker_ref.log_line.emit(" ".join(str(a) for a in args))
                original_print(*args, **kwargs)

            builtins.print = _qt_print

            try:
                import performance_assessment.model_comparison as _mc
                _orig = _mc._train_and_evaluate_fold

                def _patched(model_type, X_train, y_train, X_val, y_val,
                             metadata, fold_idx):
                    from playagain_pipeline.models.classifier import ModelManager as MM
                    mm    = MM(self._data_dir / "models" / "_comparison_tmp")
                    model = mm.create_model(
                        model_type, name=f"_cmp_{model_type}_f{fold_idx}"
                    )
                    cb_kwargs: Dict[str, Any] = {}
                    if model_type in _DEEP_MODELS:
                        _mt = model_type
                        def _cb(epoch, tl, vl, ta=0.0, va=0.0):
                            worker_ref.epoch_data.emit(
                                _mt, int(epoch),
                                float(tl), float(vl),
                                float(ta), float(va),
                            )
                        cb_kwargs["callback"] = _cb

                    t0 = time.time()
                    try:
                        model.train(
                            X_train, y_train, X_val, y_val,
                            window_size_ms=metadata.get("window_size_ms", 200),
                            sampling_rate=metadata.get("sampling_rate", 2000),
                            num_channels=metadata.get("num_channels", 0),
                            **cb_kwargs,
                        )
                    except Exception as e:
                        print(f"    [!] {model_type} fold {fold_idx} failed: {e}")
                        return {"error": str(e)}
                    elapsed = time.time() - t0

                    y_pred = model.predict(X_val)
                    y_proba = None
                    try:
                        y_proba = model.predict_proba(X_val)
                    except Exception:
                        pass

                    lnames = metadata.get("label_names", {})
                    ulabels = sorted(np.unique(np.concatenate([y_train, y_val])))
                    tnames  = [lnames.get(str(l), lnames.get(l, str(l))) for l in ulabels]

                    result: Dict[str, Any] = {
                        "model_type":          model_type,
                        "fold":                fold_idx,
                        "accuracy":            float(accuracy_score(y_val, y_pred)),
                        "f1_weighted":         float(f1_score(y_val, y_pred, average="weighted", zero_division=0)),
                        "precision_weighted":  float(precision_score(y_val, y_pred, average="weighted", zero_division=0)),
                        "recall_weighted":     float(recall_score(y_val, y_pred, average="weighted", zero_division=0)),
                        "train_time_s":        round(elapsed, 2),
                        "confusion_matrix":    confusion_matrix(y_val, y_pred, labels=ulabels).tolist(),
                        "labels":              [int(l) for l in ulabels],
                        "label_names":         tnames,
                        "classification_report": classification_report(
                            y_val, y_pred, labels=ulabels, target_names=tnames,
                            output_dict=True, zero_division=0,
                        ),
                    }
                    if y_proba is not None:
                        cm = y_pred == y_val
                        result["mean_confidence_correct"]   = float(np.mean(np.max(y_proba[cm],  axis=1))) if np.any(cm)  else 0.0
                        result["mean_confidence_incorrect"] = float(np.mean(np.max(y_proba[~cm], axis=1))) if np.any(~cm) else 0.0
                    mm.delete_model(f"_cmp_{model_type}_f{fold_idx}")

                    # Explicitly release GPU memory to prevent SIGBUS
                    # on macOS when MPS tensors are GC'd later.
                    del model
                    import gc, torch as _torch
                    gc.collect()
                    if _torch.backends.mps.is_available():
                        _torch.mps.empty_cache()

                    return result

                _mc._train_and_evaluate_fold = _patched
                from performance_assessment.model_comparison import run_comparison

                if self._mode == "holdout":
                    res = run_comparison(
                        model_types=self._model_types,
                        window_size_ms=self._window_size_ms,
                        window_stride_ms=self._window_stride_ms,
                        _holdout_sessions=(self._train_sessions, self._val_sessions),
                    )
                else:
                    res = run_comparison(
                        model_types=self._model_types,
                        window_size_ms=self._window_size_ms,
                        window_stride_ms=self._window_stride_ms,
                        _cv_sessions_and_folds=(self._cv_sessions, self._cv_folds),
                    )

                self.finished.emit(res or {})
            finally:
                _mc._train_and_evaluate_fold = _orig
                builtins.print = original_print

        except Exception:
            self.error.emit(traceback.format_exc())


# ═══════════════════════════════════════════════════════════════════════════
# Card-based session picker
# ═══════════════════════════════════════════════════════════════════════════

class SessionPickerWidget(QWidget):
    """
    Improved card-based session picker.

    Each subject gets a header block with quick-assign buttons.
    Each session is a small clickable card showing the current role badge.
    """

    roles_changed = Signal()

    _HOLDOUT_CYCLE: Dict[str, str] = {"train": "val", "val": "excluded", "excluded": "train"}
    _CV_CYCLE:      Dict[str, str] = {"use": "excluded", "excluded": "use"}

    _ROLE_STYLE: Dict[str, Tuple[str, str]] = {
        "train":    ("TRAIN", _C["train"]),
        "val":      ("VAL",   _C["val"]),
        "use":      ("USE",   _C["use"]),
        "excluded": ("—",     _C["excl"]),
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self._subject_sessions: Dict[str, List[Any]] = {}
        self._session_role:     Dict[str, str]       = {}
        self._mode      = "holdout"
        self._val_ratio = 0.20

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(4)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)

        self._container = QWidget()
        self._container_layout = QVBoxLayout(self._container)
        self._container_layout.setSpacing(6)
        self._container_layout.setContentsMargins(2, 2, 2, 4)
        self._scroll.setWidget(self._container)
        outer.addWidget(self._scroll)

        self._stats_lbl = QLabel("")
        self._stats_lbl.setStyleSheet(
            f"color:{_C['sub']}; font-size:10px; padding:2px 4px;"
        )
        self._stats_lbl.setWordWrap(True)
        outer.addWidget(self._stats_lbl)

    # ── Public API ────────────────────────────────────────────────────

    def set_mode(self, mode: str):
        self._mode = mode
        self._recompute_defaults()
        self._rebuild()

    def set_val_ratio(self, ratio: float):
        self._val_ratio = ratio
        if self._mode == "holdout":
            self._recompute_defaults()
            self._rebuild()

    def load_sessions(self, subject_sessions: Dict[str, List[Any]]):
        self._subject_sessions = subject_sessions
        self._recompute_defaults()
        self._rebuild()

    def get_roles(self) -> Dict[str, str]:
        return dict(self._session_role)

    # ── Split logic ───────────────────────────────────────────────────

    def _recompute_defaults(self):
        for subj, sessions in self._subject_sessions.items():
            if self._mode == "cv":
                for s in sessions:
                    self._session_role[s.metadata.session_id] = "use"
            else:
                n     = len(sessions)
                n_val = max(1, round(n * self._val_ratio)) if n else 0
                for i, s in enumerate(sessions):
                    self._session_role[s.metadata.session_id] = (
                        "train" if i < n - n_val else "val"
                    )

    # ── Build ─────────────────────────────────────────────────────────

    def _rebuild(self):
        while self._container_layout.count():
            item = self._container_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        for subj, sessions in self._subject_sessions.items():
            self._container_layout.addWidget(
                self._make_subject_block(subj, sessions)
            )

        self._container_layout.addStretch()
        self._update_stats()

    def _make_subject_block(self, subj: str, sessions: List[Any]) -> QFrame:
        block = QFrame()
        block.setStyleSheet(
            f"QFrame{{background:{_C['panel']};border:1px solid {_C['border']};"
            f"border-radius:6px;}}"
        )
        bl = QVBoxLayout(block)
        bl.setContentsMargins(8, 6, 8, 8)
        bl.setSpacing(4)

        # ── header row ────────────────────────────────────────────
        hdr = QHBoxLayout()
        hdr.setSpacing(4)
        lbl = QLabel(subj)
        lbl.setStyleSheet(
            f"font-weight:bold;font-size:12px;color:{_C['text']};border:none;"
        )
        hdr.addWidget(lbl)
        hdr.addStretch()

        if self._mode == "holdout":
            for role, color, text in [
                ("train",    _C["train"], "▶ TRAIN"),
                ("val",      _C["val"],   "▶ VAL"),
                ("excluded", _C["excl"],  "✕ Excl"),
            ]:
                b = self._mini_btn(text, color)
                b.clicked.connect(
                    lambda _, s=subj, r=role: self._set_subject_role(s, r)
                )
                hdr.addWidget(b)
        else:
            for role, color, text in [
                ("use",      _C["use"],  "✔ All"),
                ("excluded", _C["excl"], "✕ None"),
            ]:
                b = self._mini_btn(text, color)
                b.clicked.connect(
                    lambda _, s=subj, r=role: self._set_subject_role(s, r)
                )
                hdr.addWidget(b)

        bl.addLayout(hdr)

        # ── session cards (3-column grid) ─────────────────────────
        wrap = QWidget()
        wrap.setStyleSheet("background:transparent;border:none;")
        grid = QGridLayout(wrap)
        grid.setContentsMargins(0, 2, 0, 0)
        grid.setSpacing(4)

        max_cols = 3
        for idx, s in enumerate(sessions):
            row, col = divmod(idx, max_cols)
            grid.addWidget(
                self._make_session_card(s.metadata.session_id), row, col
            )

        bl.addWidget(wrap)
        return block

    def _make_session_card(self, sid: str) -> QFrame:
        role = self._session_role.get(sid, "excluded")
        badge_txt, badge_clr = self._ROLE_STYLE.get(role, ("?", "#888"))

        card = QFrame()
        card.setFixedHeight(54)
        card.setCursor(Qt.CursorShape.PointingHandCursor)
        self._apply_card_style(card, role)

        cl = QVBoxLayout(card)
        cl.setContentsMargins(5, 3, 5, 3)
        cl.setSpacing(2)

        badge = QLabel(badge_txt)
        badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        badge.setFixedHeight(16)
        badge.setStyleSheet(
            f"background:{badge_clr};color:#0f0f1a;font-size:9px;"
            f"font-weight:bold;border-radius:3px;padding:0 4px;border:none;"
        )
        cl.addWidget(badge)

        short = self._shorten(sid)
        name_lbl = QLabel(short)
        name_lbl.setToolTip(sid)
        name_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        name_lbl.setStyleSheet(
            f"color:{'#e2e8f0' if role != 'excluded' else _C['sub']};"
            f"font-size:9px;border:none;"
        )
        cl.addWidget(name_lbl)

        for w in (card, badge, name_lbl):
            w.mousePressEvent = lambda e, s=sid: self._on_card_click(s)

        return card

    @staticmethod
    def _shorten(sid: str) -> str:
        sid = re.sub(r"^\d{4}-\d{2}-\d{2}_", "", sid)
        sid = re.sub(r"^\d{8}_\d{6}_", "", sid)
        return (sid[:16] + "…") if len(sid) > 16 else sid

    @staticmethod
    def _apply_card_style(card: QFrame, role: str):
        bg = {"train": "#1a3a2a", "val": "#3a2a10",
              "use": "#0e2a3a", "excluded": _C["panel"]}.get(role, _C["panel"])
        bc = {"train": _C["train"], "val": _C["val"],
              "use": _C["use"], "excluded": _C["border"]}.get(role, _C["border"])
        card.setStyleSheet(
            f"QFrame{{background:{bg};border:1px solid {bc};border-radius:5px;}}"
        )

    @staticmethod
    def _mini_btn(text: str, color: str) -> QPushButton:
        b = QPushButton(text)
        b.setFixedHeight(20)
        b.setStyleSheet(
            f"QPushButton{{background:{color}22;color:{color};"
            f"border:1px solid {color}55;border-radius:3px;"
            f"font-size:9px;font-weight:bold;padding:0 5px;}}"
            f"QPushButton:hover{{background:{color}44;}}"
        )
        return b

    # ── Interaction ───────────────────────────────────────────────────

    def _on_card_click(self, sid: str):
        cur = self._session_role.get(sid, "excluded")
        new = (self._CV_CYCLE if self._mode == "cv" else self._HOLDOUT_CYCLE).get(
            cur, "use" if self._mode == "cv" else "train"
        )
        self._session_role[sid] = new
        self._rebuild()
        self.roles_changed.emit()

    def _set_subject_role(self, subj: str, role: str):
        for s in self._subject_sessions.get(subj, []):
            self._session_role[s.metadata.session_id] = role
        self._rebuild()
        self.roles_changed.emit()

    def _update_stats(self):
        roles = list(self._session_role.values())
        if self._mode == "cv":
            n = roles.count("use")
            self._stats_lbl.setText(f"✔  {n} session{'s' if n!=1 else ''} selected")
        else:
            nt, nv = roles.count("train"), roles.count("val")
            tot = nt + nv
            pct = f"  ({nv/tot:.0%} val)" if tot else ""
            self._stats_lbl.setText(f"🟢 Train: {nt}   🟡 Val: {nv}{pct}")


# ═══════════════════════════════════════════════════════════════════════════
# Live training plot panel
# ═══════════════════════════════════════════════════════════════════════════

class LiveTrainingPlot(QWidget):
    """
    Inline pyqtgraph widget showing loss + accuracy curves for deep models.
    One tab per model, auto-switched when a new model starts training.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._data:       Dict[str, Dict[str, List]] = {}
        self._model_plots: Dict[str, Dict]           = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        hdr = QLabel("Deep Model Training Progress")
        hdr.setStyleSheet(
            f"font-weight:bold;font-size:11px;color:{_C['text']};padding:2px 0;"
        )
        layout.addWidget(hdr)

        self._tabs = QTabWidget()
        self._tabs.setStyleSheet(
            "QTabWidget::pane{border:1px solid #3f3f5a;background:#1e1e2e;}"
            "QTabBar::tab{background:#2a2a3e;color:#94a3b8;padding:4px 10px;}"
            "QTabBar::tab:selected{background:#313145;color:#e2e8f0;}"
        )
        layout.addWidget(self._tabs)

    def reset(self):
        self._data.clear()
        self._model_plots.clear()
        while self._tabs.count():
            self._tabs.removeTab(0)

    def ensure_model(self, model_name: str):
        if model_name in self._model_plots:
            return

        self._data[model_name] = {
            "epochs": [], "train_loss": [], "val_loss": [],
            "train_acc": [], "val_acc": [],
        }

        tab = QWidget()
        tl  = QVBoxLayout(tab)
        tl.setContentsMargins(4, 4, 4, 4)
        tl.setSpacing(4)

        # Loss plot
        loss_pw = pg.PlotWidget(background="#1e1e2e")
        loss_pw.setLabel("left", "Loss",  color="#94a3b8")
        loss_pw.setLabel("bottom", "Epoch", color="#94a3b8")
        loss_pw.addLegend(offset=(10, 10))
        loss_pw.setMinimumHeight(150)
        tr_loss = loss_pw.plot(pen=pg.mkPen("#38bdf8", width=2), name="Train Loss")
        va_loss = loss_pw.plot(
            pen=pg.mkPen("#f59e0b", width=2, style=Qt.PenStyle.DashLine),
            name="Val Loss",
        )
        tl.addWidget(loss_pw, stretch=1)

        # Accuracy plot
        acc_pw = pg.PlotWidget(background="#1e1e2e")
        acc_pw.setLabel("left", "Accuracy", color="#94a3b8")
        acc_pw.setLabel("bottom", "Epoch",  color="#94a3b8")
        acc_pw.setYRange(0, 1)
        acc_pw.addLegend(offset=(10, 10))
        acc_pw.setMinimumHeight(150)
        tr_acc = acc_pw.plot(pen=pg.mkPen("#22c55e", width=2), name="Train Acc")
        va_acc = acc_pw.plot(
            pen=pg.mkPen("#7c3aed", width=2, style=Qt.PenStyle.DashLine),
            name="Val Acc",
        )
        tl.addWidget(acc_pw, stretch=1)

        # Stats line
        stats = QLabel("Epoch —")
        stats.setStyleSheet(
            f"color:{_C['sub']};font-size:10px;font-family:monospace;"
        )
        tl.addWidget(stats)

        self._tabs.addTab(tab, model_name)
        self._model_plots[model_name] = {
            "tr_loss": tr_loss, "va_loss": va_loss,
            "tr_acc":  tr_acc,  "va_acc":  va_acc,
            "stats":   stats,
        }

    def update_epoch(self, model_name: str, epoch: int,
                     tl: float, vl: float, ta: float, va: float):
        self.ensure_model(model_name)
        d = self._data[model_name]
        d["epochs"].append(epoch)
        d["train_loss"].append(tl)
        d["val_loss"].append(vl)
        d["train_acc"].append(ta)
        d["val_acc"].append(va)

        p = self._model_plots[model_name]
        ep = d["epochs"]
        p["tr_loss"].setData(ep, d["train_loss"])
        p["va_loss"].setData(ep, d["val_loss"])
        p["tr_acc"].setData(ep, d["train_acc"])
        p["va_acc"].setData(ep, d["val_acc"])
        p["stats"].setText(
            f"Epoch {epoch:>4}   "
            f"Train loss {tl:.4f}   Val loss {vl:.4f}   "
            f"Val acc {va:.1%}"
        )

        # Switch to the active model's tab
        for i in range(self._tabs.count()):
            if self._tabs.tabText(i) == model_name:
                self._tabs.setCurrentIndex(i)
                break


# ═══════════════════════════════════════════════════════════════════════════
# Main widget
# ═══════════════════════════════════════════════════════════════════════════

class PerformanceReviewTab(QWidget):
    """
    'Performance Review' tab embedded in the main window.

    Left  : mode + card-based session picker + model list + windowing
    Right : live deep-model training plot (collapsible) + log + gallery
    """

    def __init__(self, data_manager: DataManager, parent=None):
        super().__init__(parent)
        self._dm = data_manager
        self._worker: Optional[ComparisonWorker] = None
        self._last_output_dir: Optional[str] = None
        self._build_ui()
        self.refresh_sessions()

    # ──────────────────────────────────────────────────────────────────
    # UI construction
    # ──────────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter)
        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setSizes([400, 760])

    # ── Left ──────────────────────────────────────────────────────────

    def _build_left_panel(self) -> QWidget:
        panel = QWidget()
        pl = QVBoxLayout(panel)
        pl.setContentsMargins(0, 0, 0, 0)
        pl.setSpacing(6)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        content = QWidget()
        cl = QVBoxLayout(content)
        cl.setSpacing(8)
        cl.setContentsMargins(0, 0, 4, 0)

        # ── Evaluation mode ────────────────────────────────────────
        mode_box = QGroupBox("Evaluation Mode")
        ml = QVBoxLayout(mode_box)
        self._mode_grp     = QButtonGroup(self)
        self._holdout_radio = QRadioButton("Hold-out  (train / val split)")
        self._cv_radio      = QRadioButton("Cross-Validation  (k-fold)")
        self._holdout_radio.setChecked(True)
        self._mode_grp.addButton(self._holdout_radio, 0)
        self._mode_grp.addButton(self._cv_radio,      1)
        ml.addWidget(self._holdout_radio)
        ml.addWidget(self._cv_radio)

        self._ratio_row = QWidget()
        rf = QFormLayout(self._ratio_row)
        rf.setContentsMargins(16, 0, 0, 0)
        self._val_ratio_spin = QSpinBox()
        self._val_ratio_spin.setRange(10, 50)
        self._val_ratio_spin.setValue(20)
        self._val_ratio_spin.setSuffix(" %")
        self._val_ratio_spin.valueChanged.connect(
            lambda v: self._session_picker.set_val_ratio(v / 100.0)
        )
        rf.addRow("Val ratio:", self._val_ratio_spin)
        ml.addWidget(self._ratio_row)

        self._folds_row = QWidget()
        ff = QFormLayout(self._folds_row)
        ff.setContentsMargins(16, 0, 0, 0)
        self._cv_folds_spin = QSpinBox()
        self._cv_folds_spin.setRange(2, 20)
        self._cv_folds_spin.setValue(5)
        ff.addRow("Folds:", self._cv_folds_spin)
        self._folds_row.setVisible(False)
        ml.addWidget(self._folds_row)

        self._holdout_radio.toggled.connect(self._on_mode_changed)
        cl.addWidget(mode_box)

        # ── Session picker ─────────────────────────────────────────
        sess_box = QGroupBox("Sessions")
        sbl = QVBoxLayout(sess_box)

        top_row = QHBoxLayout()
        top_row.setSpacing(4)
        ref_btn = QPushButton("↺  Refresh")
        ref_btn.setFixedHeight(26)
        ref_btn.clicked.connect(self.refresh_sessions)
        top_row.addWidget(ref_btn)
        top_row.addStretch()

        # Global quick-assign (hold-out only)
        self._global_train_btn = QPushButton("All → TRAIN")
        self._global_train_btn.setFixedHeight(24)
        self._global_train_btn.setStyleSheet(
            f"font-size:9px;background:{_C['train']}22;color:{_C['train']};"
            f"border:1px solid {_C['train']}55;border-radius:3px;"
        )
        self._global_train_btn.clicked.connect(lambda: self._set_global_role("train"))

        self._global_val_btn = QPushButton("All → VAL")
        self._global_val_btn.setFixedHeight(24)
        self._global_val_btn.setStyleSheet(
            f"font-size:9px;background:{_C['val']}22;color:{_C['val']};"
            f"border:1px solid {_C['val']}55;border-radius:3px;"
        )
        self._global_val_btn.clicked.connect(lambda: self._set_global_role("val"))

        top_row.addWidget(self._global_train_btn)
        top_row.addWidget(self._global_val_btn)
        sbl.addLayout(top_row)

        self._session_picker = SessionPickerWidget()
        self._session_picker.setMinimumHeight(260)
        self._session_picker.roles_changed.connect(self._update_run_btn)
        sbl.addWidget(self._session_picker)
        cl.addWidget(sess_box)

        # ── Model selection ────────────────────────────────────────
        model_box = QGroupBox("Models")
        mbl = QVBoxLayout(model_box)
        self._model_cbs: Dict[str, QCheckBox] = {}
        for m in list(ModelManager.AVAILABLE_MODELS.keys()):
            cb = QCheckBox(m)
            cb.setChecked(True)
            mbl.addWidget(cb)
            self._model_cbs[m] = cb

        btn_r = QHBoxLayout()
        for lbl, chk in [("All", True), ("None", False)]:
            b = QPushButton(lbl)
            b.setFixedHeight(24)
            b.clicked.connect(lambda _, c=chk: [cb.setChecked(c) for cb in self._model_cbs.values()])
            btn_r.addWidget(b)
        btn_r.addStretch()
        mbl.addLayout(btn_r)
        cl.addWidget(model_box)

        # ── Windowing ──────────────────────────────────────────────
        win_box  = QGroupBox("Windowing")
        win_form = QFormLayout(win_box)
        self._win_size_spin = QSpinBox()
        self._win_size_spin.setRange(50, 2000)
        self._win_size_spin.setValue(200)
        self._win_size_spin.setSuffix(" ms")
        win_form.addRow("Window size:", self._win_size_spin)
        self._win_stride_spin = QSpinBox()
        self._win_stride_spin.setRange(10, 1000)
        self._win_stride_spin.setValue(50)
        self._win_stride_spin.setSuffix(" ms")
        win_form.addRow("Stride:", self._win_stride_spin)
        cl.addWidget(win_box)

        cl.addStretch()
        scroll.setWidget(content)
        pl.addWidget(scroll)

        # ── Run / Stop ─────────────────────────────────────────────
        self._run_btn = QPushButton("▶  Run Comparison")
        self._run_btn.setFixedHeight(40)
        self._run_btn.setStyleSheet(
            f"QPushButton{{background:{_C['accent']};color:white;font-size:13px;"
            f"font-weight:bold;border-radius:4px;}}"
            f"QPushButton:hover{{background:#6d28d9;}}"
            f"QPushButton:disabled{{background:#555;color:#888;}}"
        )
        self._run_btn.clicked.connect(self._on_run)
        pl.addWidget(self._run_btn)

        self._stop_btn = QPushButton("✕  Stop")
        self._stop_btn.setFixedHeight(32)
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self._on_stop)
        pl.addWidget(self._stop_btn)

        return panel

    # ── Right ─────────────────────────────────────────────────────────

    def _build_right_panel(self) -> QWidget:
        panel = QWidget()
        pl = QVBoxLayout(panel)
        pl.setContentsMargins(0, 0, 0, 0)

        vsplit = QSplitter(Qt.Orientation.Vertical)
        pl.addWidget(vsplit)

        # TOP: live plot (hidden until a deep model runs)
        self._live_plot = LiveTrainingPlot()
        self._live_plot_frame = QFrame()
        self._live_plot_frame.setStyleSheet(
            f"QFrame{{background:{_C['panel']};border:1px solid {_C['border']};"
            f"border-radius:6px;}}"
        )
        fpl = QVBoxLayout(self._live_plot_frame)
        fpl.setContentsMargins(6, 6, 6, 6)
        fpl.addWidget(self._live_plot)
        self._live_plot_frame.setVisible(False)
        vsplit.addWidget(self._live_plot_frame)

        # MID: progress + log
        log_w = QWidget()
        ll = QVBoxLayout(log_w)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.setSpacing(3)

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 0)
        self._progress_bar.setVisible(False)
        self._progress_bar.setFixedHeight(6)
        self._progress_bar.setStyleSheet(
            f"QProgressBar{{border:none;background:{_C['panel']};border-radius:3px;}}"
            f"QProgressBar::chunk{{background:{_C['accent']};border-radius:3px;}}"
        )
        ll.addWidget(self._progress_bar)

        self._status_lbl = QLabel(
            "Ready — configure sessions and click ▶ Run Comparison"
        )
        self._status_lbl.setStyleSheet(
            "color:#888;font-size:10px;padding:1px 0;"
        )
        ll.addWidget(self._status_lbl)

        self._log_text = QTextEdit()
        self._log_text.setReadOnly(True)
        self._log_text.setStyleSheet(
            "QTextEdit{font-family:monospace;font-size:10px;"
            "background:#1e1e1e;color:#d4d4d4;"
            "border:1px solid #444;border-radius:3px;padding:4px;}"
        )
        ll.addWidget(self._log_text)

        clr = QPushButton("Clear log")
        clr.setFixedHeight(22)
        clr.setFixedWidth(72)
        clr.setStyleSheet("font-size:10px;")
        clr.clicked.connect(self._log_text.clear)
        ll.addWidget(clr, alignment=Qt.AlignmentFlag.AlignRight)
        vsplit.addWidget(log_w)

        # BOT: results gallery
        gal_w = QWidget()
        gl = QVBoxLayout(gal_w)
        gl.setContentsMargins(0, 0, 0, 0)

        ghdr = QHBoxLayout()
        gt = QLabel("Results")
        gt.setStyleSheet("font-weight:bold;font-size:12px;")
        ghdr.addWidget(gt)
        ghdr.addStretch()
        self._open_folder_btn = QPushButton("📂  Open results folder")
        self._open_folder_btn.setFixedHeight(24)
        self._open_folder_btn.setEnabled(False)
        self._open_folder_btn.clicked.connect(self._on_open_folder)
        ghdr.addWidget(self._open_folder_btn)
        gl.addLayout(ghdr)

        gal_scroll = QScrollArea()
        gal_scroll.setWidgetResizable(True)
        gal_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._gallery_widget = QWidget()
        self._gallery_grid   = QGridLayout(self._gallery_widget)
        self._gallery_grid.setSpacing(10)
        gal_scroll.setWidget(self._gallery_widget)
        gl.addWidget(gal_scroll)

        ph = QLabel("No results yet.\nRun a comparison to see plots here.")
        ph.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ph.setStyleSheet("color:#666;font-size:12px;")
        self._gallery_grid.addWidget(ph, 0, 0, 1, 2)
        vsplit.addWidget(gal_w)

        vsplit.setSizes([220, 280, 500])
        return panel

    # ──────────────────────────────────────────────────────────────────
    # Session loading
    # ──────────────────────────────────────────────────────────────────

    def refresh_sessions(self):
        subj_sessions: Dict[str, List[Any]] = {}
        for subj in self._dm.list_subjects():
            sessions = []
            for sid in self._dm.list_sessions(subj):
                try:
                    s = self._dm.load_session(subj, sid)
                    if s.get_data().shape[1] == s.metadata.num_channels:
                        sessions.append(s)
                except Exception:
                    pass
            if sessions:
                subj_sessions[subj] = sessions
        self._session_picker.load_sessions(subj_sessions)
        self._update_run_btn()

    # ──────────────────────────────────────────────────────────────────
    # Mode / UI helpers
    # ──────────────────────────────────────────────────────────────────

    def _on_mode_changed(self):
        is_ho = self._holdout_radio.isChecked()
        self._ratio_row.setVisible(is_ho)
        self._folds_row.setVisible(not is_ho)
        self._global_train_btn.setVisible(is_ho)
        self._global_val_btn.setVisible(is_ho)
        self._session_picker.set_mode("holdout" if is_ho else "cv")
        self._update_run_btn()

    def _set_global_role(self, role: str):
        sp = self._session_picker
        for sid in sp._session_role:
            sp._session_role[sid] = role
        sp._rebuild()

    def _update_run_btn(self):
        self._run_btn.setEnabled(self._worker is None)

    # ──────────────────────────────────────────────────────────────────
    # Run / Stop
    # ──────────────────────────────────────────────────────────────────

    def _on_run(self):
        mode   = "cv" if self._cv_radio.isChecked() else "holdout"
        roles  = self._session_picker.get_roles()
        sp     = self._session_picker
        all_s  = {s.metadata.session_id: s
                  for sessions in sp._subject_sessions.values()
                  for s in sessions}

        sel_models = [m for m, cb in self._model_cbs.items() if cb.isChecked()]
        if not sel_models:
            QMessageBox.warning(self, "No models", "Select at least one model.")
            return

        if mode == "holdout":
            train = [all_s[sid] for sid, r in roles.items() if r == "train" and sid in all_s]
            val   = [all_s[sid] for sid, r in roles.items() if r == "val"   and sid in all_s]
            if not train:
                QMessageBox.warning(self, "No TRAIN sessions",
                    "Assign at least one session as TRAIN.\nClick a card or use 'All → TRAIN'.")
                return
            if not val:
                QMessageBox.warning(self, "No VAL sessions",
                    "Assign at least one session as VAL.\nClick a card or use 'All → VAL'.")
                return
            cv_sessions = None
        else:
            train = val = None
            cv_sessions = [all_s[sid] for sid, r in roles.items() if r == "use" and sid in all_s]
            if not cv_sessions:
                QMessageBox.warning(self, "No sessions", "Mark at least one session as USE.")
                return

        has_deep = any(m in _DEEP_MODELS for m in sel_models)
        self._live_plot_frame.setVisible(has_deep)
        if has_deep:
            self._live_plot.reset()

        self._log_text.clear()
        self._log("Starting comparison …")
        self._status_lbl.setText("Running …")
        self._progress_bar.setVisible(True)
        self._run_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._last_output_dir = None

        self._worker = ComparisonWorker(
            data_dir         = self._dm.data_dir,
            model_types      = sel_models,
            mode             = mode,
            train_sessions   = train,
            val_sessions     = val,
            cv_sessions      = cv_sessions,
            cv_folds         = self._cv_folds_spin.value(),
            window_size_ms   = self._win_size_spin.value(),
            window_stride_ms = self._win_stride_spin.value(),
        )
        self._worker.log_line.connect(self._log)
        self._worker.epoch_data.connect(self._on_epoch_data)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_stop(self):
        if self._worker and self._worker.isRunning():
            self._worker.terminate()
            self._worker.wait(3000)
        self._on_run_done()
        self._log("⚠  Run stopped by user.")

    # ── Worker signals ────────────────────────────────────────────────

    def _on_epoch_data(self, model: str, epoch: int,
                       tl: float, vl: float, ta: float, va: float):
        self._live_plot.update_epoch(model, epoch, tl, vl, ta, va)

    def _on_finished(self, result: dict):
        self._on_run_done()
        if not result:
            self._status_lbl.setText("Finished — no results")
            return
        out = result.get("output_dir", "")
        self._last_output_dir = out
        self._status_lbl.setText(f"Done  →  {out}")
        self._open_folder_btn.setEnabled(bool(out))
        self._log(f"✓ Complete. Results: {out}")
        summary = result.get("summary")
        if summary is not None:
            self._log("\n── Summary ──\n" + summary.to_string(index=False))
        if out:
            self._load_plots(Path(out))

    def _on_error(self, tb: str):
        self._on_run_done()
        self._log(f"ERROR:\n{tb}")
        self._status_lbl.setText("Error — see log")
        QMessageBox.critical(self, "Comparison Error",
                             "Pipeline error — see log for details.")

    def _on_run_done(self):
        self._progress_bar.setVisible(False)
        self._run_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        if self._worker:
            for sig in (self._worker.log_line, self._worker.epoch_data,
                        self._worker.finished, self._worker.error):
                try:
                    sig.disconnect()
                except RuntimeError:
                    pass
            self._worker = None

    def _on_open_folder(self):
        if self._last_output_dir:
            import subprocess
            subprocess.Popen(["open", self._last_output_dir])

    # ── Log helper ────────────────────────────────────────────────────

    def _log(self, text: str):
        self._log_text.append(text)
        self._log_text.verticalScrollBar().setValue(
            self._log_text.verticalScrollBar().maximum()
        )

    # ── Plot gallery ──────────────────────────────────────────────────

    _PLOT_FILES = [
        ("metric_comparison.png",   "Metric Comparison"),
        ("confusion_matrices.png",  "Confusion Matrices"),
        ("per_class_f1.png",        "Per-Class F1"),
        ("training_time.png",       "Training Time"),
        ("confidence_analysis.png", "Confidence Analysis"),
    ]

    def _load_plots(self, out_dir: Path):
        for i in reversed(range(self._gallery_grid.count())):
            w = self._gallery_grid.itemAt(i).widget()
            if w:
                w.deleteLater()

        col = row = found = 0
        for filename, title in self._PLOT_FILES:
            fp = out_dir / filename
            if not fp.exists():
                continue
            found += 1
            frame = QFrame()
            frame.setStyleSheet(
                f"QFrame{{background:{_C['panel']};border:1px solid {_C['border']};"
                f"border-radius:6px;}}"
            )
            fl = QVBoxLayout(frame)
            fl.setContentsMargins(6, 6, 6, 6)
            tl = QLabel(title)
            tl.setStyleSheet(
                f"font-weight:bold;font-size:11px;color:{_C['text']};border:none;"
            )
            fl.addWidget(tl)
            pix = QPixmap(str(fp))
            img = QLabel()
            img.setPixmap(
                pix.scaledToWidth(500, Qt.TransformationMode.SmoothTransformation)
            )
            img.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
            )
            img.setAlignment(Qt.AlignmentFlag.AlignCenter)
            fl.addWidget(img)
            self._gallery_grid.addWidget(frame, row, col)
            col += 1
            if col >= 2:
                col = 0
                row += 1

        if found == 0:
            ph = QLabel("Plots not found in output directory.")
            ph.setAlignment(Qt.AlignmentFlag.AlignCenter)
            ph.setStyleSheet("color:#666;font-size:12px;")
            self._gallery_grid.addWidget(ph, 0, 0, 1, 2)
