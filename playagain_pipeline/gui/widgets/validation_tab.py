"""
validation_tab.py  (v2)
────────────────────────
GUI front-end for the playagain_pipeline.validation package — reworked
for clarity, live progress, and comparison across runs.

What's new vs v1
─────────────────
1. **Live fold-by-fold progress.** A QThread-based worker drives the
   runner with a ProgressReporter hook. Users see the current fold
   index, model, train time, per-fold accuracy, and an ETA computed
   from the running mean fold time — instead of a featureless spinner.

2. **Cancellation.** A Cancel button sets a flag the runner checks
   between folds. A cancelled run still keeps the folds it has done
   so the user can inspect partial results.

3. **Pre-flight preview.** Before running, a small panel shows
   "N folds × M models = T evaluations · ~E subjects selected" so
   mistakes are caught before a 20-minute run.

4. **Per-class F1 table.** The runner now exposes per-class F1 and a
   confusion matrix per fold. A dedicated tab on the right aggregates
   them into a class × model table so you can see which gestures each
   model struggles with.

5. **Previous runs browser.** A tab lists every run in
   `data/validation_runs/` with strategy, best model, best accuracy,
   and an "Open folder" shortcut. Makes it easy to compare today's
   run with yesterday's without remembering timestamps.

6. **Holdout UI reachable again.** v1 had the full holdout-ratio
   controls wired up but never added "holdout_split" to the strategy
   combo. Fixed.

7. **Guardrail warnings** before running:
   - no sessions matched
   - only one subject with LOSO (degenerate fold)
   - very short recordings given the window size
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from PySide6.QtCore import (
    Qt, Signal, Slot, QTimer, QThread, QObject,
)
from PySide6.QtGui import QFont, QColor, QBrush
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QGroupBox, QPushButton, QListWidget, QListWidgetItem,
    QCheckBox, QComboBox, QSpinBox, QLineEdit,
    QTableWidget, QTableWidgetItem, QHeaderView, QSplitter,
    QScrollArea, QFileDialog, QMessageBox, QTabWidget, QFrame,
    QTextEdit, QProgressBar,
    QTreeWidget, QTreeWidgetItem, QSlider, QSizePolicy,
)

from playagain_pipeline.validation import (
    SessionCorpus,
    ExperimentConfig,
    ValidationRunner,
    RunResult,
)
from playagain_pipeline.validation.runner import (
    FoldResult,
    NoopProgress,
)
from playagain_pipeline.validation.config import (
    DataSelection, WindowingConfig, FeatureConfig, ModelConfig, CVConfig,
    load_experiment, dump_experiment,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULT_FEATURES = [
    ("mav",  "Mean Absolute Value"),
    ("rms",  "Root Mean Square"),
    ("wl",   "Waveform Length"),
    ("zc",   "Zero Crossings"),
    ("ssc",  "Slope Sign Changes"),
    ("var",  "Variance"),
    ("iemg", "Integrated EMG"),
]

_DEFAULT_MODELS = [
    ("lda",           "Very fast linear discriminant — strong EMG baseline."),
    ("random_forest", "Robust, handles noise well, no scaling needed."),
    ("catboost",      "Gradient boosting — often best on tabular features."),
    ("svm",           "Linear / RBF support vector machine."),
    ("mlp",           "Small neural net — needs ≳10 k windows."),
    ("attention_net", "Transformer attention — strongest temporal model."),
]

_MODEL_KEY_NORMALISE: dict = {
    "lda": "lda", "LDA": "lda",
    "randomforest": "random_forest", "RandomForest": "random_forest",
    "random_forest": "random_forest",
    "catboost": "catboost", "CatBoost": "catboost",
    "svm": "svm", "SVM": "svm",
    "mlp": "mlp", "MLP": "mlp",
    "attentionnet": "attention_net", "AttentionNet": "attention_net",
    "attention_net": "attention_net",
    "cnn": "cnn", "CNN": "cnn",
    "mstnet": "mstnet", "MSTNet": "mstnet",
}

_CV_DESCRIPTIONS = {
    "loso_subject":     "Leave-One-Subject-Out  (the honest single number)",
    "loso_session":     "Leave-One-Session-Out  (per-session generalisation)",
    "within_session":   "Within-Session  (temporal tail — optimistic baseline)",
    "k_fold_subjects":  "k-Fold over subjects",
    "cross_domain":     "Cross-Domain  (pipeline ↔ unity)",
    "holdout_split":    "Holdout  (explicit Train / Val / Test — for tuning)",
}

# Short help string surfaced in the hover tooltip for each CV strategy.
# More guidance than the combo label, less than the README.
_CV_TOOLTIPS = {
    "loso_subject":
        "Train on every subject but one, test on the held-out subject. "
        "Repeat for every subject. Reports how well the pipeline "
        "generalises to an unseen person — the most honest single "
        "number for a paper.",
    "loso_session":
        "Like LOSO but at session granularity. Good when you have many "
        "sessions per subject and want to see session-to-session "
        "drift within the same person.",
    "within_session":
        "Train on the first 80% of each session's windows, test on the "
        "last 20%. Optimistic — windows from the same session share "
        "hardware placement and user state — but useful as a ceiling.",
    "k_fold_subjects":
        "Shuffle subjects, split into k roughly-equal groups, train on "
        "k-1 and test on the remaining one. Useful when LOSO has too "
        "many subjects to run quickly.",
    "cross_domain":
        "Train only on one domain (pipeline or unity) and test on the "
        "other. Measures transfer from one recorder to the other.",
    "holdout_split":
        "One fold with explicit Train / Val / Test ratios. Best for "
        "tuning a single model with early stopping — deep models use "
        "the val split for schedulers; the test split is untouched "
        "until the final number.",
}


# ---------------------------------------------------------------------------
# A helper for grouping checkboxes
# ---------------------------------------------------------------------------

class _CheckboxGroup(QGroupBox):
    """A group-box of checkboxes with select-all / select-none buttons."""

    selection_changed = Signal()

    def __init__(self, title: str, items: List[tuple], parent=None):
        super().__init__(title, parent)
        self._checkboxes: Dict[str, QCheckBox] = {}

        outer = QVBoxLayout(self)

        btn_row = QHBoxLayout()
        all_btn = QPushButton("All")
        all_btn.setFixedHeight(22)
        all_btn.clicked.connect(self.select_all)
        btn_row.addWidget(all_btn)
        none_btn = QPushButton("None")
        none_btn.setFixedHeight(22)
        none_btn.clicked.connect(self.select_none)
        btn_row.addWidget(none_btn)
        btn_row.addStretch()
        outer.addLayout(btn_row)

        for key, description in items:
            cb = QCheckBox(f"{key}")
            cb.setToolTip(description)
            cb.toggled.connect(lambda _checked: self.selection_changed.emit())
            outer.addWidget(cb)
            self._checkboxes[key] = cb

    def selected(self) -> List[str]:
        return [k for k, cb in self._checkboxes.items() if cb.isChecked()]

    def set_selected(self, keys: List[str]):
        wanted = set(keys)
        for k, cb in self._checkboxes.items():
            cb.blockSignals(True)
            cb.setChecked(k in wanted)
            cb.blockSignals(False)
        self.selection_changed.emit()

    @Slot()
    def select_all(self):
        for cb in self._checkboxes.values():
            cb.setChecked(True)

    @Slot()
    def select_none(self):
        for cb in self._checkboxes.values():
            cb.setChecked(False)


# ---------------------------------------------------------------------------
# Qt-signal bridge to the runner's ProgressReporter
# ---------------------------------------------------------------------------

class _ProgressBridge(QObject):
    """
    Emits Qt signals from a runner ProgressReporter. The signals are
    connected via Qt.QueuedConnection so the worker thread can call
    ProgressReporter methods and the GUI updates land on the UI thread.
    """

    run_started   = Signal(int, int)            # n_folds, n_models
    fold_started  = Signal(int, int, str, str)  # idx, total, fold_id, model
    fold_finished = Signal(int, int, object)    # idx, total, FoldResult
    run_finished  = Signal(object)              # RunResult
    log_line      = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cancel = False

    def request_cancel(self) -> None:
        self._cancel = True

    # ProgressReporter API (called from worker thread) ────────────────

    def on_run_start(self, total_folds, total_models, _records):
        self.run_started.emit(total_folds, total_models)

    def on_fold_start(self, idx, total, fold_id, model_type):
        self.fold_started.emit(idx, total, fold_id, model_type)

    def on_fold_done(self, idx, total, fold_result):
        self.fold_finished.emit(idx, total, fold_result)

    def on_run_done(self, result):
        self.run_finished.emit(result)

    def log(self, message: str):
        self.log_line.emit(message)

    def should_cancel(self) -> bool:
        return self._cancel


class _ValidationWorker(QThread):
    """Runs ValidationRunner.run(cfg, progress) on a worker thread."""

    failed = Signal(str)
    # run_finished fires on success via the progress bridge; failed()
    # fires separately if an exception bubbles up.

    def __init__(
        self,
        data_dir: Path,
        cfg: ExperimentConfig,
        bridge: _ProgressBridge,
        parent=None,
    ):
        super().__init__(parent)
        self._data_dir = data_dir
        self._cfg = cfg
        self._bridge = bridge

    def run(self) -> None:
        try:
            runner = ValidationRunner(self._data_dir)
            runner.run(self._cfg, progress=self._bridge)
        except Exception:  # noqa: BLE001
            import traceback
            self.failed.emit(traceback.format_exc())


# ---------------------------------------------------------------------------
# Previous-runs browser
# ---------------------------------------------------------------------------

@dataclass
class _PastRunSummary:
    path:       Path
    name:       str
    timestamp:  str
    strategy:   str
    n_folds:    int
    best_model: str
    best_acc:   float
    cancelled:  bool


class _RunsBrowser(QWidget):
    """Lists every run under validation_runs/ with quick 'open' / 'reload'."""

    reload_requested = Signal(object)  # ExperimentConfig

    def __init__(self, runs_root: Path, parent=None):
        super().__init__(parent)
        self._runs_root = runs_root

        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)

        hdr = QHBoxLayout()
        hdr.addWidget(QLabel("Previous runs"))
        hdr.addStretch()
        refresh = QPushButton("↻ Refresh")
        refresh.setFixedHeight(24)
        refresh.clicked.connect(self.refresh)
        hdr.addWidget(refresh)
        lay.addLayout(hdr)

        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(
            ["When", "Name", "Strategy", "Folds", "Best model", "Best acc"]
        )
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        lay.addWidget(self.table, 1)

        row = QHBoxLayout()
        open_btn = QPushButton("Open folder")
        open_btn.clicked.connect(self._on_open_selected)
        row.addWidget(open_btn)
        reload_btn = QPushButton("Reload config ↺")
        reload_btn.setToolTip("Copy this run's experiment config into the "
                              "left panel so you can re-run / tweak it.")
        reload_btn.clicked.connect(self._on_reload_selected)
        row.addWidget(reload_btn)
        row.addStretch()
        lay.addLayout(row)

        self._rows: List[_PastRunSummary] = []
        self.refresh()

    def refresh(self) -> None:
        self._rows = self._scan()
        self.table.setRowCount(len(self._rows))
        for i, s in enumerate(self._rows):
            self.table.setItem(i, 0, QTableWidgetItem(s.timestamp))
            self.table.setItem(i, 1, QTableWidgetItem(s.name))
            self.table.setItem(i, 2, QTableWidgetItem(s.strategy))
            self.table.setItem(i, 3, QTableWidgetItem(str(s.n_folds)))
            self.table.setItem(i, 4, QTableWidgetItem(s.best_model))
            acc_item = QTableWidgetItem(f"{s.best_acc:.3f}"
                                        if s.best_acc > 0 else "—")
            if s.cancelled:
                for col in range(6):
                    itm = self.table.item(i, col)
                    if itm:
                        itm.setForeground(QBrush(QColor("#9ca3af")))
            self.table.setItem(i, 5, acc_item)

    def _scan(self) -> List[_PastRunSummary]:
        if not self._runs_root.exists():
            return []
        out: List[_PastRunSummary] = []
        for entry in sorted(self._runs_root.iterdir(), reverse=True):
            if not entry.is_dir():
                continue
            exp_path = entry / "experiment.json"
            res_path = entry / "results.json"
            if not exp_path.exists():
                continue
            try:
                exp = json.loads(exp_path.read_text())
            except Exception:  # noqa: BLE001
                continue

            ts = entry.name.split("__", 1)[0]
            name = exp.get("name", entry.name.split("__", 1)[-1])
            strategy = (exp.get("cv") or {}).get("strategy", "?")

            n_folds, best_model, best_acc, cancelled = 0, "—", 0.0, False
            if res_path.exists():
                try:
                    res = json.loads(res_path.read_text())
                    n_folds = len(res.get("folds") or [])
                    cancelled = bool(res.get("cancelled"))
                    agg = res.get("aggregate") or {}
                    if agg:
                        best_model, best_meta = max(
                            agg.items(),
                            key=lambda kv: kv[1].get("accuracy_mean", 0.0),
                        )
                        best_acc = float(best_meta.get("accuracy_mean", 0.0))
                except Exception:  # noqa: BLE001
                    pass

            out.append(_PastRunSummary(
                path=entry, name=name, timestamp=ts, strategy=strategy,
                n_folds=n_folds, best_model=best_model, best_acc=best_acc,
                cancelled=cancelled,
            ))
        return out

    def _selected(self) -> Optional[_PastRunSummary]:
        row = self.table.currentRow()
        if row < 0 or row >= len(self._rows):
            return None
        return self._rows[row]

    @Slot()
    def _on_open_selected(self):
        sel = self._selected()
        if sel is None:
            return
        _open_in_file_manager(sel.path)

    @Slot()
    def _on_reload_selected(self):
        sel = self._selected()
        if sel is None:
            return
        try:
            cfg = load_experiment(sel.path / "experiment.json")
            self.reload_requested.emit(cfg)
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "Reload failed", str(e))


# ---------------------------------------------------------------------------
# The main tab
# ---------------------------------------------------------------------------

class ValidationTab(QWidget):
    """
    Modular, reproducible validation harness embedded in the main window.

    Constructed with the application's ``DataManager`` so it can find the
    same data the rest of the GUI uses.
    """

    def __init__(self, data_manager, parent=None):
        super().__init__(parent)
        self.data_manager = data_manager
        self.data_dir: Path = Path(data_manager.data_dir)

        self._features = self._discover_features()
        self._models = self._discover_models()
        self._corpus = SessionCorpus(self.data_dir)

        self._last_result: Optional[RunResult] = None

        # Worker state
        self._worker: Optional[_ValidationWorker] = None
        self._bridge: Optional[_ProgressBridge] = None

        # Progress tracking
        self._fold_times: List[float] = []
        self._fold_start_time: float = 0.0

        self._build_ui()
        self._refresh_corpus()

    # ------------------------------------------------------------------
    # Registry discovery
    # ------------------------------------------------------------------

    def _discover_features(self) -> List[tuple]:
        try:
            from playagain_pipeline.models.feature_pipeline import get_registered_features
            names = list(get_registered_features())
            if names:
                lookup = dict(_DEFAULT_FEATURES)
                return [(n, lookup.get(n, n)) for n in names]
        except Exception:  # noqa: BLE001
            pass
        return list(_DEFAULT_FEATURES)

    def _discover_models(self) -> List[tuple]:
        return list(_DEFAULT_MODELS)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)

        outer.addWidget(self._build_header())

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setSizes([480, 760])
        outer.addWidget(splitter, 1)

    # -- Header --------------------------------------------------------

    def _build_header(self) -> QWidget:
        wrap = QFrame()
        wrap.setStyleSheet(
            "QFrame { background: #f1f5f9; border-left: 4px solid #0284c7; "
            "border-radius: 4px; }"
        )
        lay = QHBoxLayout(wrap)
        lay.setContentsMargins(10, 6, 10, 6)

        text_col = QVBoxLayout()
        title = QLabel("Validation")
        tf = QFont()
        tf.setBold(True)
        tf.setPointSize(12)
        title.setFont(tf)
        title.setStyleSheet("color: #0284c7; background: transparent; border: none;")
        text_col.addWidget(title)

        sub = QLabel(
            "Reproducible cross-validation across feature sets, models, and CV strategies. "
            "Every run is saved to data/validation_runs/ with the exact config and environment."
        )
        sub.setWordWrap(True)
        sub.setStyleSheet("color: #475569; font-size: 11px; background: transparent; border: none;")
        text_col.addWidget(sub)
        lay.addLayout(text_col, 1)

        load_btn = QPushButton("Load YAML…")
        load_btn.clicked.connect(self._on_load_yaml)
        lay.addWidget(load_btn)

        save_btn = QPushButton("Save as YAML…")
        save_btn.clicked.connect(self._on_save_yaml)
        lay.addWidget(save_btn)

        return wrap

    # -- Left panel: all the knobs -------------------------------------

    def _build_left_panel(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        layout = QVBoxLayout(content)

        # ── Experiment metadata
        meta_box = QGroupBox("Experiment")
        meta_form = QFormLayout(meta_box)
        self.name_edit = QLineEdit("loso_baseline")
        meta_form.addRow("Name:", self.name_edit)

        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 2**31 - 1)
        self.seed_spin.setValue(42)
        meta_form.addRow("Random seed:", self.seed_spin)
        layout.addWidget(meta_box)

        # ── Data selection
        data_box = QGroupBox("Data")
        data_lay = QVBoxLayout(data_box)

        domain_row = QHBoxLayout()
        domain_row.addWidget(QLabel("Domains:"))
        self.cb_pipeline = QCheckBox("pipeline")
        self.cb_pipeline.setChecked(True)
        self.cb_pipeline.toggled.connect(self._refresh_pickers)
        self.cb_pipeline.toggled.connect(self._refresh_preview)
        domain_row.addWidget(self.cb_pipeline)
        self.cb_unity = QCheckBox("unity")
        self.cb_unity.toggled.connect(self._refresh_pickers)
        self.cb_unity.toggled.connect(self._refresh_preview)
        domain_row.addWidget(self.cb_unity)
        domain_row.addStretch()
        data_lay.addLayout(domain_row)

        self._data_tabs = QTabWidget()
        self._data_tabs.currentChanged.connect(lambda *_: self._refresh_preview())

        # Tab 1 — pick subjects
        subj_tab = QWidget()
        subj_lay = QVBoxLayout(subj_tab)
        subj_lay.setContentsMargins(4, 4, 4, 4)
        subj_lay.addWidget(QLabel(
            "Multi-select subjects (empty = all matching the domain filter)."
        ))
        self.subject_list = QListWidget()
        self.subject_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.subject_list.setMinimumHeight(110)
        self.subject_list.itemSelectionChanged.connect(self._refresh_preview)
        subj_lay.addWidget(self.subject_list)
        self._data_tabs.addTab(subj_tab, "By subject")

        # Tab 2 — pick individual sessions
        sess_tab = QWidget()
        sess_lay = QVBoxLayout(sess_tab)
        sess_lay.setContentsMargins(4, 4, 4, 4)
        sess_help = QLabel(
            "Tick the exact sessions to include. Tick a subject node "
            "to toggle all of its sessions at once."
        )
        sess_help.setWordWrap(True)
        sess_help.setStyleSheet("color: #64748b; font-size: 10px;")
        sess_lay.addWidget(sess_help)

        self.session_tree = QTreeWidget()
        self.session_tree.setHeaderLabels(["Subject / Session", "Channels", "SR"])
        self.session_tree.setMinimumHeight(160)
        self.session_tree.itemChanged.connect(self._on_session_tree_changed)
        sess_lay.addWidget(self.session_tree)

        sess_btn_row = QHBoxLayout()
        sess_all = QPushButton("Tick all")
        sess_all.setFixedHeight(22)
        sess_all.clicked.connect(lambda: self._set_all_session_ticks(True))
        sess_btn_row.addWidget(sess_all)
        sess_none = QPushButton("Untick all")
        sess_none.setFixedHeight(22)
        sess_none.clicked.connect(lambda: self._set_all_session_ticks(False))
        sess_btn_row.addWidget(sess_none)
        self.sess_count_lbl = QLabel("0 selected")
        self.sess_count_lbl.setStyleSheet("color: #64748b; font-size: 10px;")
        sess_btn_row.addStretch()
        sess_btn_row.addWidget(self.sess_count_lbl)
        sess_lay.addLayout(sess_btn_row)

        self._data_tabs.addTab(sess_tab, "By session")
        data_lay.addWidget(self._data_tabs)

        refresh_btn = QPushButton("↻ Re-scan corpus")
        refresh_btn.clicked.connect(self._refresh_corpus)
        data_lay.addWidget(refresh_btn)

        self.corpus_summary_lbl = QLabel("")
        self.corpus_summary_lbl.setStyleSheet("color: #64748b; font-size: 10px;")
        data_lay.addWidget(self.corpus_summary_lbl)

        layout.addWidget(data_box)

        # ── Windowing
        win_box = QGroupBox("Windowing")
        win_form = QFormLayout(win_box)
        self.window_ms = QSpinBox()
        self.window_ms.setRange(20, 2000)
        self.window_ms.setValue(200)
        self.window_ms.setSuffix(" ms")
        win_form.addRow("Window size:", self.window_ms)

        self.stride_ms = QSpinBox()
        self.stride_ms.setRange(5, 1000)
        self.stride_ms.setValue(50)
        self.stride_ms.setSuffix(" ms")
        win_form.addRow("Stride:", self.stride_ms)
        layout.addWidget(win_box)

        # ── Modular features
        self.features_box = _CheckboxGroup("Features", self._features)
        self.features_box.set_selected(
            [k for k, _ in self._features if k in {"mav", "rms", "wl", "zc"}]
        )
        self.features_box.selection_changed.connect(self._refresh_preview)
        layout.addWidget(self.features_box)

        # ── Modular models
        self.models_box = _CheckboxGroup("Models", self._models)
        self.models_box.set_selected(["lda", "random_forest", "catboost"])
        self.models_box.selection_changed.connect(self._refresh_preview)
        layout.addWidget(self.models_box)

        # ── CV strategy
        cv_box = QGroupBox("Cross-Validation")
        cv_lay = QVBoxLayout(cv_box)
        self.cv_combo = QComboBox()
        # NOTE: v1 forgot to add holdout_split here even though the
        # rest of the code supported it. Now included.
        for key in ("loso_subject", "loso_session", "within_session",
                    "k_fold_subjects", "cross_domain", "holdout_split"):
            self.cv_combo.addItem(_CV_DESCRIPTIONS.get(key, key), userData=key)
        self.cv_combo.currentIndexChanged.connect(self._on_cv_changed)
        self.cv_combo.currentIndexChanged.connect(self._refresh_preview)
        cv_lay.addWidget(self.cv_combo)

        # Tooltip tracker — hover text matches current selection
        self.cv_hint = QLabel("")
        self.cv_hint.setWordWrap(True)
        self.cv_hint.setStyleSheet(
            "color: #475569; font-size: 10px; padding: 4px 2px;"
        )
        cv_lay.addWidget(self.cv_hint)

        # k-fold knob
        self._kfold_container = QWidget()
        kfold_row = QFormLayout(self._kfold_container)
        kfold_row.setContentsMargins(0, 0, 0, 0)
        self.kfold_spin = QSpinBox()
        self.kfold_spin.setRange(2, 20)
        self.kfold_spin.setValue(5)
        self.kfold_spin.valueChanged.connect(self._refresh_preview)
        kfold_row.addRow("k (k-fold only):", self.kfold_spin)
        cv_lay.addWidget(self._kfold_container)

        # cross-domain direction
        self._xd_container = QWidget()
        xd_row = QFormLayout(self._xd_container)
        xd_row.setContentsMargins(0, 0, 0, 0)
        self.xd_train_combo = QComboBox()
        self.xd_train_combo.addItems(["pipeline", "unity"])
        xd_row.addRow("Train on (cross-domain):", self.xd_train_combo)
        self.xd_test_combo = QComboBox()
        self.xd_test_combo.addItems(["unity", "pipeline"])
        xd_row.addRow("Test on (cross-domain):", self.xd_test_combo)
        cv_lay.addWidget(self._xd_container)

        # holdout ratios
        self._holdout_container = self._build_holdout_controls()
        cv_lay.addWidget(self._holdout_container)

        self._on_cv_changed()
        layout.addWidget(cv_box)

        # ── Preview panel ("about to run…")
        self.preview_box = QFrame()
        self.preview_box.setStyleSheet(
            "QFrame { background: #ecfeff; border: 1px solid #a5f3fc; "
            "border-radius: 4px; padding: 2px; }"
        )
        prev_lay = QVBoxLayout(self.preview_box)
        prev_lay.setContentsMargins(8, 6, 8, 6)
        pt = QLabel("About to run")
        ptf = QFont(); ptf.setBold(True); pt.setFont(ptf)
        pt.setStyleSheet("color: #0e7490; background: transparent; border: none;")
        prev_lay.addWidget(pt)
        self.preview_label = QLabel("")
        self.preview_label.setWordWrap(True)
        self.preview_label.setStyleSheet(
            "color: #155e75; font-size: 11px; background: transparent; border: none;"
        )
        prev_lay.addWidget(self.preview_label)
        layout.addWidget(self.preview_box)

        # ── Run controls
        run_row = QHBoxLayout()
        self.run_btn = QPushButton("▶ Run validation")
        self.run_btn.setFixedHeight(36)
        self.run_btn.setStyleSheet(
            "QPushButton{background:#1a2e4a;color:#06b6d4;"
            "border:1px solid #06b6d4;border-radius:6px;"
            "font-weight:700;font-size:12px;padding:6px 18px;}"
            "QPushButton:hover{background:#06b6d4;color:#fff;}"
            "QPushButton:disabled{background:#e5e7eb;color:#9ca3af;"
            "border:1px solid #d1d5db;}"
        )
        self.run_btn.clicked.connect(self._on_run)
        run_row.addWidget(self.run_btn, 1)

        self.cancel_btn = QPushButton("■ Cancel")
        self.cancel_btn.setFixedHeight(36)
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.setStyleSheet(
            "QPushButton{background:#7f1d1d;color:#fecaca;"
            "border:1px solid #ef4444;border-radius:6px;"
            "font-weight:700;padding:6px 18px;}"
            "QPushButton:hover:enabled{background:#b91c1c;color:#fff;}"
            "QPushButton:disabled{background:#f3f4f6;color:#d1d5db;"
            "border:1px solid #e5e7eb;}"
        )
        self.cancel_btn.clicked.connect(self._on_cancel)
        run_row.addWidget(self.cancel_btn)

        layout.addLayout(run_row)

        layout.addStretch()
        scroll.setWidget(content)
        self._on_cv_changed()  # set initial hint
        self._refresh_preview()
        return scroll

    # -- Right panel: progress + results -------------------------------

    def _build_right_panel(self) -> QWidget:
        wrap = QWidget()
        layout = QVBoxLayout(wrap)
        layout.setContentsMargins(0, 0, 0, 0)

        # ── Progress panel (visible while running OR shows last-run status)
        self._progress_panel = QFrame()
        self._progress_panel.setStyleSheet(
            "QFrame { background: #0f172a; border-radius: 6px; border: 1px solid #1e3a5f; }"
        )
        prog_lay = QVBoxLayout(self._progress_panel)
        prog_lay.setContentsMargins(10, 8, 10, 8)
        prog_lay.setSpacing(4)

        # Title row
        prog_title_row = QHBoxLayout()
        self._prog_title_lbl = QLabel("Ready")
        _ptf = QFont(); _ptf.setBold(True); _ptf.setPointSize(11)
        self._prog_title_lbl.setFont(_ptf)
        self._prog_title_lbl.setStyleSheet("color: #06b6d4; background: transparent; border: none;")
        prog_title_row.addWidget(self._prog_title_lbl)
        prog_title_row.addStretch()
        self._prog_eta_lbl = QLabel("")
        self._prog_eta_lbl.setStyleSheet("color: #94a3b8; font-size: 10px; background: transparent; border: none;")
        prog_title_row.addWidget(self._prog_eta_lbl)
        prog_lay.addLayout(prog_title_row)

        # Determinate progress bar with N/M label
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 1)
        self._progress_bar.setValue(0)
        self._progress_bar.setFixedHeight(8)
        self._progress_bar.setTextVisible(False)
        self._progress_bar.setStyleSheet(
            "QProgressBar { background: #1e293b; border-radius: 4px; border: none; }"
            "QProgressBar::chunk { background: #06b6d4; border-radius: 4px; }"
        )
        prog_lay.addWidget(self._progress_bar)

        # Live log
        self._prog_log = QTextEdit()
        self._prog_log.setReadOnly(True)
        self._prog_log.setFixedHeight(130)
        self._prog_log.setStyleSheet(
            "QTextEdit { background: transparent; color: #cbd5e1; "
            "font-family: 'Menlo', 'Consolas', monospace; font-size: 10px; border: none; }"
        )
        prog_lay.addWidget(self._prog_log)

        layout.addWidget(self._progress_panel)

        # ── Results tabs: Overview · Per-class · Previous runs
        self._results_tabs = QTabWidget()

        # Tab 1: Overview
        overview_tab = QWidget()
        ov_lay = QVBoxLayout(overview_tab)

        agg_box = QGroupBox("Aggregate — mean ± std across folds")
        agg_lay = QVBoxLayout(agg_box)
        self.agg_table = QTableWidget(0, 4)
        self.agg_table.setHorizontalHeaderLabels(
            ["Model", "Folds", "Accuracy", "Macro-F1"]
        )
        self.agg_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.agg_table.verticalHeader().setVisible(False)
        agg_lay.addWidget(self.agg_table)
        ov_lay.addWidget(agg_box, 1)

        fold_box = QGroupBox("Per-fold results")
        fold_lay = QVBoxLayout(fold_box)
        self.fold_table = QTableWidget(0, 6)
        self.fold_table.setHorizontalHeaderLabels(
            ["Fold", "Model", "n_train", "n_test", "Accuracy", "Macro-F1"]
        )
        self.fold_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.fold_table.verticalHeader().setVisible(False)
        fold_lay.addWidget(self.fold_table)
        ov_lay.addWidget(fold_box, 2)

        footer = QHBoxLayout()
        self.output_path_lbl = QLabel("No run yet.")
        self.output_path_lbl.setStyleSheet("color: #64748b; font-size: 10px;")
        footer.addWidget(self.output_path_lbl, 1)
        self.open_folder_btn = QPushButton("Open results folder")
        self.open_folder_btn.setEnabled(False)
        self.open_folder_btn.clicked.connect(self._on_open_folder)
        footer.addWidget(self.open_folder_btn)
        ov_lay.addLayout(footer)

        self._results_tabs.addTab(overview_tab, "Overview")

        # Tab 2: Per-class breakdown
        perclass_tab = QWidget()
        pc_lay = QVBoxLayout(perclass_tab)
        pc_help = QLabel(
            "Mean F1 per class across folds. Lower scores mark "
            "gestures the model struggles with."
        )
        pc_help.setWordWrap(True)
        pc_help.setStyleSheet("color: #64748b; font-size: 10px; padding: 4px;")
        pc_lay.addWidget(pc_help)
        self.perclass_table = QTableWidget(0, 0)
        self.perclass_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.perclass_table.verticalHeader().setVisible(True)
        pc_lay.addWidget(self.perclass_table, 1)
        self._results_tabs.addTab(perclass_tab, "Per-class F1")

        # Tab 3: Previous runs
        self.runs_browser = _RunsBrowser(
            self.data_dir / "validation_runs", parent=self,
        )
        self.runs_browser.reload_requested.connect(self._apply_config)
        self._results_tabs.addTab(self.runs_browser, "Previous runs")

        layout.addWidget(self._results_tabs, 1)

        return wrap

    # ------------------------------------------------------------------
    # Preview ("about to run…") panel
    # ------------------------------------------------------------------

    def _refresh_preview(self, *_):
        """Compute and display what clicking Run *would* do, right now.

        Signals like ``_data_tabs.currentChanged`` can fire during
        construction of the left panel — before ``models_box`` /
        ``features_box`` / ``preview_label`` have been instantiated —
        so we guard against every attribute we need and silently
        no-op if the UI isn't fully built yet.
        """
        required = ("models_box", "features_box", "preview_label",
                    "cv_combo", "subject_list", "session_tree")
        if not all(hasattr(self, name) for name in required):
            return
        try:
            recs = self._selected_records()
            n_subjects = len({r.subject_id for r in recs})
            n_sessions = len(recs)
            n_models = len(self.models_box.selected())
            n_features = len(self.features_box.selected())

            strategy_key = self.cv_combo.currentData()
            strategy = _CV_DESCRIPTIONS.get(strategy_key, strategy_key)

            # Rough fold-count estimate (without instantiating the strategy)
            est_folds = self._estimate_fold_count(strategy_key, n_subjects, n_sessions)
            est_evals = est_folds * max(1, n_models)

            msg = (
                f"<b>{n_sessions}</b> session(s) · "
                f"<b>{n_subjects}</b> subject(s) · "
                f"<b>{n_features}</b> feature(s) · "
                f"<b>{n_models}</b> model(s)<br>"
                f"Strategy: {strategy}<br>"
                f"Estimated <b>{est_folds}</b> fold(s) "
                f"→ <b>{est_evals}</b> total evaluation(s)."
            )

            # Guardrails
            warnings = []
            if n_sessions == 0:
                warnings.append(
                    "⚠ No sessions selected — the run would do nothing."
                )
            if n_models == 0:
                warnings.append("⚠ No models selected.")
            if n_features == 0:
                warnings.append(
                    "⚠ No features selected — the dataset will be raw windows."
                )
            if strategy_key == "loso_subject" and n_subjects < 2:
                warnings.append(
                    "⚠ LOSO-Subject needs ≥2 subjects — adjust the filter."
                )
            if strategy_key == "loso_session" and n_sessions < 2:
                warnings.append(
                    "⚠ LOSO-Session needs ≥2 sessions."
                )
            if strategy_key == "cross_domain":
                selected_domains = set(self._selected_domains())
                if len(selected_domains) < 2 and (selected_domains != set()):
                    warnings.append(
                        "⚠ Cross-domain needs both 'pipeline' and 'unity' "
                        "ticked (or neither, to use all)."
                    )

            # Deep CNN-family models expect raw windows; features break
            # their predict path. The runner now auto-detects this and
            # materialises raw windows for those models, but surface a
            # hint so users aren't surprised that their feature selection
            # is silently skipped for CNN folds.
            raw_models = {"cnn", "attention_net", "mstnet"}
            selected_raw = [m for m in self.models_box.selected()
                            if _MODEL_KEY_NORMALISE.get(m, m.lower()) in raw_models]
            if selected_raw and n_features > 0:
                warnings.append(
                    f"ℹ {', '.join(selected_raw)} need raw windows — "
                    "the feature selection will be ignored for those "
                    "models (other models still use it)."
                )

            if warnings:
                msg += "<br><br>" + "<br>".join(warnings)

            self.preview_label.setText(msg)
        except Exception as e:  # noqa: BLE001
            self.preview_label.setText(f"(preview failed: {e})")

    @staticmethod
    def _estimate_fold_count(strategy: str, n_subjects: int, n_sessions: int) -> int:
        # Best-effort; exact counts require instantiating the strategy.
        if strategy == "loso_subject":
            return max(0, n_subjects)
        if strategy == "loso_session":
            return max(0, n_sessions)
        if strategy == "within_session":
            return max(0, n_sessions)
        if strategy == "k_fold_subjects":
            return 0  # shown separately; users see k in the spin box
        if strategy == "cross_domain":
            return 1
        if strategy == "holdout_split":
            return 1
        return 0

    # ------------------------------------------------------------------
    # Corpus + UI refresh
    # ------------------------------------------------------------------

    def _refresh_corpus(self):
        self._corpus = SessionCorpus(self.data_dir)
        self._corpus.discover()
        self.corpus_summary_lbl.setText(
            self._corpus.summary().replace("\n", "  ·  ")
        )
        self._refresh_pickers()
        self._refresh_preview()

    def _refresh_pickers(self):
        domains = self._selected_domains()
        recs = self._corpus.filter(domains=domains) if domains else self._corpus.all()

        previously_selected = set(self._selected_subjects() or [])
        self.subject_list.clear()
        for s in sorted({r.subject_id for r in recs}):
            n = sum(1 for r in recs if r.subject_id == s)
            item = QListWidgetItem(f"{s}  ({n} sessions)")
            item.setData(Qt.ItemDataRole.UserRole, s)
            self.subject_list.addItem(item)
            if s in previously_selected:
                item.setSelected(True)

        previously_ticked = set(self._selected_sessions() or [])
        self.session_tree.blockSignals(True)
        try:
            self.session_tree.clear()
            recs_by_subject: Dict[str, list] = {}
            for r in recs:
                recs_by_subject.setdefault(r.subject_id, []).append(r)

            for subj in sorted(recs_by_subject):
                subj_recs = sorted(recs_by_subject[subj],
                                   key=lambda r: r.session_id)
                subj_item = QTreeWidgetItem([subj, "", ""])
                subj_item.setFlags(
                    subj_item.flags()
                    | Qt.ItemFlag.ItemIsUserCheckable
                    | Qt.ItemFlag.ItemIsAutoTristate
                )
                subj_item.setCheckState(0, Qt.CheckState.Unchecked)
                f = subj_item.font(0); f.setBold(True); subj_item.setFont(0, f)
                subj_item.setData(0, Qt.ItemDataRole.UserRole, ("subject", subj))
                self.session_tree.addTopLevelItem(subj_item)

                for rec in subj_recs:
                    sess_item = QTreeWidgetItem([
                        rec.session_id,
                        str(rec.num_channels) if rec.num_channels else "?",
                        f"{int(rec.sampling_rate)} Hz" if rec.sampling_rate else "?",
                    ])
                    sess_item.setFlags(
                        sess_item.flags() | Qt.ItemFlag.ItemIsUserCheckable
                    )
                    key = f"{rec.subject_id}/{rec.session_id}"
                    sess_item.setData(0, Qt.ItemDataRole.UserRole,
                                      ("session", key))
                    sess_item.setCheckState(
                        0,
                        Qt.CheckState.Checked if key in previously_ticked
                        else Qt.CheckState.Unchecked,
                    )
                    subj_item.addChild(sess_item)

                subj_item.setExpanded(True)
            self.session_tree.resizeColumnToContents(0)
        finally:
            self.session_tree.blockSignals(False)
        self._update_session_count()

    @Slot()
    def _on_session_tree_changed(self, *_):
        self._update_session_count()
        self._refresh_preview()

    def _update_session_count(self):
        n = len(self._selected_sessions() or [])
        total = sum(
            self.session_tree.topLevelItem(i).childCount()
            for i in range(self.session_tree.topLevelItemCount())
        )
        self.sess_count_lbl.setText(f"{n} of {total} selected")

    def _set_all_session_ticks(self, on: bool):
        state = Qt.CheckState.Checked if on else Qt.CheckState.Unchecked
        self.session_tree.blockSignals(True)
        try:
            for i in range(self.session_tree.topLevelItemCount()):
                top = self.session_tree.topLevelItem(i)
                top.setCheckState(0, state)
                for j in range(top.childCount()):
                    top.child(j).setCheckState(0, state)
        finally:
            self.session_tree.blockSignals(False)
        self._update_session_count()
        self._refresh_preview()

    def _selected_domains(self) -> List[str]:
        out = []
        if self.cb_pipeline.isChecked():
            out.append("pipeline")
        if self.cb_unity.isChecked():
            out.append("unity")
        return out

    def _selected_subjects(self) -> Optional[List[str]]:
        items = self.subject_list.selectedItems()
        if not items:
            return None
        return [it.data(Qt.ItemDataRole.UserRole) for it in items]

    def _selected_sessions(self) -> Optional[List[str]]:
        keys: List[str] = []
        for i in range(self.session_tree.topLevelItemCount()):
            top = self.session_tree.topLevelItem(i)
            for j in range(top.childCount()):
                child = top.child(j)
                if child.checkState(0) == Qt.CheckState.Checked:
                    kind, key = child.data(0, Qt.ItemDataRole.UserRole)
                    if kind == "session":
                        keys.append(key)
        return keys or None

    def _data_picker_mode(self) -> str:
        return "sessions" if self._data_tabs.currentIndex() == 1 else "subjects"

    def _selected_records(self) -> list:
        """Resolve the currently-selected (subjects|sessions|domains) to records."""
        domains = self._selected_domains() or None
        if self._data_picker_mode() == "sessions":
            keys = set(self._selected_sessions() or [])
            if not keys:
                return []
            return [
                r for r in self._corpus.all()
                if f"{r.subject_id}/{r.session_id}" in keys
                and (domains is None or r.source_domain in domains)
            ]
        subjects = self._selected_subjects()
        return self._corpus.filter(subjects=subjects, domains=domains)

    # ------------------------------------------------------------------
    # Holdout ratio controls
    # ------------------------------------------------------------------

    def _build_holdout_controls(self) -> QWidget:
        wrap = QWidget()
        lay = QVBoxLayout(wrap)
        lay.setContentsMargins(0, 4, 0, 0)
        lay.setSpacing(2)

        info = QLabel(
            "A single fold with explicit train / val / test ratios. "
            "Sessions are split — never windows from the same session."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #64748b; font-size: 10px;")
        lay.addWidget(info)

        # Val ratio
        val_row = QHBoxLayout()
        val_row.addWidget(QLabel("Val %:"))
        self.holdout_val_slider = QSlider(Qt.Orientation.Horizontal)
        self.holdout_val_slider.setRange(0, 50)
        self.holdout_val_slider.setValue(15)
        self.holdout_val_slider.valueChanged.connect(self._update_holdout_readout)
        val_row.addWidget(self.holdout_val_slider, 1)
        self.holdout_val_lbl = QLabel("15%")
        self.holdout_val_lbl.setMinimumWidth(34)
        val_row.addWidget(self.holdout_val_lbl)
        lay.addLayout(val_row)

        # Test ratio
        test_row = QHBoxLayout()
        test_row.addWidget(QLabel("Test %:"))
        self.holdout_test_slider = QSlider(Qt.Orientation.Horizontal)
        self.holdout_test_slider.setRange(5, 50)
        self.holdout_test_slider.setValue(15)
        self.holdout_test_slider.valueChanged.connect(self._update_holdout_readout)
        test_row.addWidget(self.holdout_test_slider, 1)
        self.holdout_test_lbl = QLabel("15%")
        self.holdout_test_lbl.setMinimumWidth(34)
        test_row.addWidget(self.holdout_test_lbl)
        lay.addLayout(test_row)

        # Stratification
        strat_row = QHBoxLayout()
        strat_row.addWidget(QLabel("Stratify by:"))
        self.holdout_strat_combo = QComboBox()
        self.holdout_strat_combo.addItem("subject (recommended)", userData="subject")
        self.holdout_strat_combo.addItem("none (global shuffle)", userData="none")
        strat_row.addWidget(self.holdout_strat_combo, 1)
        lay.addLayout(strat_row)

        self.holdout_summary_lbl = QLabel("")
        self.holdout_summary_lbl.setStyleSheet(
            "color: #06b6d4; font-weight: 600; font-size: 11px;"
        )
        lay.addWidget(self.holdout_summary_lbl)

        self._update_holdout_readout()
        return wrap

    def _update_holdout_readout(self, *_):
        v = self.holdout_val_slider.value()
        t = self.holdout_test_slider.value()
        if v + t > 95:
            t = 95 - v
            self.holdout_test_slider.blockSignals(True)
            self.holdout_test_slider.setValue(t)
            self.holdout_test_slider.blockSignals(False)
        train = 100 - v - t
        self.holdout_val_lbl.setText(f"{v}%")
        self.holdout_test_lbl.setText(f"{t}%")
        self.holdout_summary_lbl.setText(
            f"Train / Val / Test  =  {train}% / {v}% / {t}%"
        )

    @Slot()
    def _on_cv_changed(self):
        key = self.cv_combo.currentData()
        self._kfold_container.setVisible(key == "k_fold_subjects")
        self._xd_container.setVisible(key == "cross_domain")
        self._holdout_container.setVisible(key == "holdout_split")
        self.cv_hint.setText(_CV_TOOLTIPS.get(key, ""))

    # ------------------------------------------------------------------
    # Build an ExperimentConfig from the live UI state
    # ------------------------------------------------------------------

    def _build_config(self) -> ExperimentConfig:
        feats = [FeatureConfig(name=n) for n in self.features_box.selected()]
        models = [
            ModelConfig(type=_MODEL_KEY_NORMALISE.get(m, m.lower()))
            for m in self.models_box.selected()
        ]

        cv_key = self.cv_combo.currentData()
        kwargs: Dict = {}
        if cv_key == "k_fold_subjects":
            kwargs = {"k": int(self.kfold_spin.value()),
                      "seed": int(self.seed_spin.value())}
        elif cv_key == "cross_domain":
            kwargs = {"train_domain": self.xd_train_combo.currentText(),
                      "test_domain":  self.xd_test_combo.currentText()}
        elif cv_key == "holdout_split":
            kwargs = {
                "val_ratio":   self.holdout_val_slider.value() / 100.0,
                "test_ratio":  self.holdout_test_slider.value() / 100.0,
                "seed":        int(self.seed_spin.value()),
                "stratify_by": self.holdout_strat_combo.currentData(),
            }

        if self._data_picker_mode() == "sessions":
            data_sel = DataSelection(
                domains=self._selected_domains() or None,
                explicit=self._selected_sessions(),
            )
        else:
            data_sel = DataSelection(
                subjects=self._selected_subjects(),
                domains=self._selected_domains() or None,
            )

        return ExperimentConfig(
            name=self.name_edit.text().strip() or "unnamed",
            description="Built from the GUI Validation tab.",
            seed=int(self.seed_spin.value()),
            data=data_sel,
            windowing=WindowingConfig(
                window_ms=int(self.window_ms.value()),
                stride_ms=int(self.stride_ms.value()),
                drop_rest=False,
            ),
            features=feats,
            models=models,
            cv=CVConfig(strategy=cv_key, kwargs=kwargs),
        )

    # ------------------------------------------------------------------
    # Run / Cancel
    # ------------------------------------------------------------------

    @Slot()
    def _on_run(self):
        try:
            cfg = self._build_config()
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, "Invalid configuration", str(e))
            return

        if not cfg.features:
            if QMessageBox.question(
                self, "No features",
                "No features are selected — models will be trained on raw "
                "windows (only supported by deep models). Continue?",
            ) != QMessageBox.StandardButton.Yes:
                return
        if not cfg.models:
            QMessageBox.warning(self, "No models",
                                "Select at least one model.")
            return

        # Hard guardrails — we refuse to start runs that can't produce
        # meaningful output. Warnings only (preview catches soft issues).
        n_sessions = len(self._selected_records())
        if n_sessions == 0:
            QMessageBox.warning(
                self, "No sessions",
                "No sessions match the current filter. Check the Data "
                "panel on the left.",
            )
            return

        # Reset progress view
        self._fold_times = []
        self._prog_log.clear()
        self._progress_bar.setRange(0, 1)
        self._progress_bar.setValue(0)
        self._progress_bar.setStyleSheet(
            "QProgressBar { background: #1e293b; border-radius: 4px; border: none; }"
            "QProgressBar::chunk { background: #06b6d4; border-radius: 4px; }"
        )
        self._prog_title_lbl.setText("⚙  Starting…")
        self._prog_eta_lbl.setText("")

        self.run_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)

        # Log the plan
        cv_label = _CV_DESCRIPTIONS.get(cfg.cv.strategy, cfg.cv.strategy)
        self._log_line(f"Strategy : {cv_label}")
        self._log_line(f"Features : {', '.join(f.name for f in cfg.features) or '(raw)'}")
        self._log_line(f"Models   : {', '.join(m.type for m in cfg.models)}")
        if cfg.data.explicit:
            self._log_line(f"Sessions : {len(cfg.data.explicit)} explicit session(s)")
        else:
            subjects = cfg.data.subjects or ["all"]
            self._log_line(f"Subjects : {', '.join(subjects)}")
        if cfg.cv.strategy == "holdout_split":
            v = float(cfg.cv.kwargs.get("val_ratio", 0.0)) * 100
            t = float(cfg.cv.kwargs.get("test_ratio", 0.0)) * 100
            self._log_line(f"Ratios   : train {100 - v - t:.0f}% / val {v:.0f}% / test {t:.0f}%")
        self._log_line("─" * 42)

        # Start worker
        self._bridge = _ProgressBridge(parent=self)
        self._bridge.run_started.connect(self._on_run_started, Qt.ConnectionType.QueuedConnection)
        self._bridge.fold_started.connect(self._on_fold_started, Qt.ConnectionType.QueuedConnection)
        self._bridge.fold_finished.connect(self._on_fold_finished, Qt.ConnectionType.QueuedConnection)
        self._bridge.run_finished.connect(self._on_run_finished, Qt.ConnectionType.QueuedConnection)
        self._bridge.log_line.connect(self._log_line, Qt.ConnectionType.QueuedConnection)

        self._worker = _ValidationWorker(self.data_dir, cfg, self._bridge, parent=self)
        self._worker.failed.connect(self._on_run_failed, Qt.ConnectionType.QueuedConnection)
        self._worker.finished.connect(self._on_worker_finished)  # cleanup only
        self._worker.start()

    @Slot()
    def _on_cancel(self):
        if self._bridge is None:
            return
        self.cancel_btn.setEnabled(False)
        self._bridge.request_cancel()
        self._log_line("✖ Cancellation requested — will stop after current fold.")
        self._prog_title_lbl.setText("⚙  Cancelling…")

    # -- Worker signal handlers ---------------------------------------

    @Slot(int, int)
    def _on_run_started(self, n_folds: int, n_models: int):
        total = n_folds * n_models
        self._progress_bar.setRange(0, max(1, total))
        self._progress_bar.setValue(0)
        self._prog_title_lbl.setText(
            f"⚙  Running  ·  0 / {total} evaluation(s)"
        )
        self._prog_eta_lbl.setText("")

    @Slot(int, int, str, str)
    def _on_fold_started(self, idx: int, total: int, fold_id: str, model_type: str):
        self._fold_start_time = time.time()
        self._prog_title_lbl.setText(
            f"⚙  {idx} / {total}  ·  {model_type}  ·  {fold_id}"
        )

    @Slot(int, int, object)
    def _on_fold_finished(self, idx: int, total: int, fold_result: FoldResult):
        dt = time.time() - self._fold_start_time
        self._fold_times.append(dt)
        self._progress_bar.setValue(idx)

        # ETA from running mean
        if self._fold_times:
            mean_dt = sum(self._fold_times) / len(self._fold_times)
            remaining = total - idx
            eta_sec = mean_dt * remaining
            self._prog_eta_lbl.setText(
                f"~{_fmt_dur(eta_sec)} remaining  ·  {_fmt_dur(mean_dt)}/fold avg"
            )
        self._prog_title_lbl.setText(
            f"⚙  {idx} / {total}  ·  last: {fold_result.model_type} "
            f"acc {fold_result.accuracy:.3f}"
        )

    @Slot(object)
    def _on_run_finished(self, result: RunResult):
        self._last_result = result
        self.run_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)

        n_folds = len(result.folds)
        if result.cancelled:
            self._prog_title_lbl.setText(f"Cancelled  ·  {n_folds} fold(s) completed")
            self._progress_bar.setStyleSheet(
                "QProgressBar { background: #1e293b; border-radius: 4px; border: none; }"
                "QProgressBar::chunk { background: #f59e0b; border-radius: 4px; }"
            )
        else:
            self._prog_title_lbl.setText(f"✓ Done  ·  {n_folds} fold(s)")
            self._progress_bar.setRange(0, 1)
            self._progress_bar.setValue(1)
            self._progress_bar.setStyleSheet(
                "QProgressBar { background: #1e293b; border-radius: 4px; border: none; }"
                "QProgressBar::chunk { background: #16a34a; border-radius: 4px; }"
            )
        self._prog_eta_lbl.setText("")

        agg = result.aggregate()
        for model, m in sorted(agg.items()):
            self._log_line(
                f"  {model:<18}  "
                f"acc {m['accuracy_mean']:.3f} ± {m['accuracy_std']:.3f}  "
                f"F1  {m['macro_f1_mean']:.3f} ± {m['macro_f1_std']:.3f}"
            )

        self._populate_results(result)
        self.runs_browser.refresh()

    @Slot(str)
    def _on_run_failed(self, tb: str):
        self.run_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self._prog_title_lbl.setText("✗ Failed")
        self._progress_bar.setStyleSheet(
            "QProgressBar { background: #1e293b; border-radius: 4px; border: none; }"
            "QProgressBar::chunk { background: #dc2626; border-radius: 4px; }"
        )
        self._log_line("✗ Run failed.")
        self._log_line(tb[-800:])
        log.error("Validation run failed:\n%s", tb)
        QMessageBox.critical(self, "Validation failed",
                             f"The run raised an exception:\n\n{tb[-1200:]}")

    @Slot()
    def _on_worker_finished(self):
        # Thread-cleanup only — results are delivered via run_finished.
        self._worker = None
        self._bridge = None

    @Slot(str)
    def _log_line(self, msg: str):
        self._prog_log.append(msg)
        sb = self._prog_log.verticalScrollBar()
        sb.setValue(sb.maximum())

    # ------------------------------------------------------------------
    # Results rendering
    # ------------------------------------------------------------------

    def _populate_results(self, result: RunResult):
        any_val = any(fr.val_accuracy is not None for fr in result.folds)

        # ── Aggregate table ────────────────────────────────────────────
        agg_headers = ["Model", "Folds", "Accuracy", "Macro-F1", "Mean train time"]
        if any_val:
            agg_headers += ["Val Accuracy", "Val Macro-F1"]
        self.agg_table.setColumnCount(len(agg_headers))
        self.agg_table.setHorizontalHeaderLabels(agg_headers)

        agg = result.aggregate()
        val_by_model: Dict[str, list] = {}
        for fr in result.folds:
            if fr.val_accuracy is not None:
                val_by_model.setdefault(fr.model_type, []).append(
                    (fr.val_accuracy, fr.val_macro_f1 or 0.0)
                )

        self.agg_table.setRowCount(len(agg))
        for row, (model, m) in enumerate(sorted(agg.items())):
            self.agg_table.setItem(row, 0, QTableWidgetItem(model))
            self.agg_table.setItem(row, 1, QTableWidgetItem(str(m["n_folds"])))
            self.agg_table.setItem(
                row, 2,
                QTableWidgetItem(f"{m['accuracy_mean']:.3f} ± {m['accuracy_std']:.3f}")
            )
            self.agg_table.setItem(
                row, 3,
                QTableWidgetItem(f"{m['macro_f1_mean']:.3f} ± {m['macro_f1_std']:.3f}")
            )
            self.agg_table.setItem(
                row, 4,
                QTableWidgetItem(_fmt_dur(m["train_seconds_mean"]))
            )
            if any_val:
                vals = val_by_model.get(model, [])
                if vals:
                    import numpy as _np
                    accs = _np.array([v[0] for v in vals], dtype=float)
                    f1s  = _np.array([v[1] for v in vals], dtype=float)
                    self.agg_table.setItem(
                        row, 5,
                        QTableWidgetItem(f"{accs.mean():.3f} ± {accs.std():.3f}")
                    )
                    self.agg_table.setItem(
                        row, 6,
                        QTableWidgetItem(f"{f1s.mean():.3f} ± {f1s.std():.3f}")
                    )
                else:
                    self.agg_table.setItem(row, 5, QTableWidgetItem("—"))
                    self.agg_table.setItem(row, 6, QTableWidgetItem("—"))

        # Highlight best accuracy row
        if agg:
            best_model = max(agg.items(),
                             key=lambda kv: kv[1]["accuracy_mean"])[0]
            for row, (model, _m) in enumerate(sorted(agg.items())):
                if model == best_model:
                    for col in range(self.agg_table.columnCount()):
                        itm = self.agg_table.item(row, col)
                        if itm:
                            itm.setBackground(QBrush(QColor("#ecfdf5")))

        # ── Per-fold table ─────────────────────────────────────────────
        fold_headers = ["Fold", "Model", "n_train", "n_test",
                        "Accuracy", "Macro-F1"]
        if any_val:
            fold_headers = ["Fold", "Model", "n_train", "n_val", "n_test",
                            "Accuracy", "Val Accuracy", "Macro-F1"]
        self.fold_table.setColumnCount(len(fold_headers))
        self.fold_table.setHorizontalHeaderLabels(fold_headers)

        self.fold_table.setRowCount(len(result.folds))
        for row, fr in enumerate(result.folds):
            if any_val:
                self.fold_table.setItem(row, 0, QTableWidgetItem(fr.fold_id))
                self.fold_table.setItem(row, 1, QTableWidgetItem(fr.model_type))
                self.fold_table.setItem(row, 2, QTableWidgetItem(str(fr.n_train_windows)))
                self.fold_table.setItem(row, 3, QTableWidgetItem(str(fr.n_val_windows)))
                self.fold_table.setItem(row, 4, QTableWidgetItem(str(fr.n_test_windows)))
                self.fold_table.setItem(row, 5, QTableWidgetItem(f"{fr.accuracy:.3f}"))
                self.fold_table.setItem(
                    row, 6,
                    QTableWidgetItem(
                        f"{fr.val_accuracy:.3f}" if fr.val_accuracy is not None else "—"
                    ),
                )
                self.fold_table.setItem(row, 7, QTableWidgetItem(f"{fr.macro_f1:.3f}"))
            else:
                self.fold_table.setItem(row, 0, QTableWidgetItem(fr.fold_id))
                self.fold_table.setItem(row, 1, QTableWidgetItem(fr.model_type))
                self.fold_table.setItem(row, 2, QTableWidgetItem(str(fr.n_train_windows)))
                self.fold_table.setItem(row, 3, QTableWidgetItem(str(fr.n_test_windows)))
                self.fold_table.setItem(row, 4, QTableWidgetItem(f"{fr.accuracy:.3f}"))
                self.fold_table.setItem(row, 5, QTableWidgetItem(f"{fr.macro_f1:.3f}"))

        # ── Per-class F1 table (class × model) ─────────────────────────
        all_classes = sorted({
            cls for fr in result.folds for cls in fr.per_class_f1.keys()
        })
        all_models = sorted({fr.model_type for fr in result.folds})
        self.perclass_table.setRowCount(len(all_classes))
        self.perclass_table.setColumnCount(len(all_models))
        self.perclass_table.setHorizontalHeaderLabels(all_models)
        self.perclass_table.setVerticalHeaderLabels(all_classes)

        import numpy as _np
        for ri, cls in enumerate(all_classes):
            for ci, model in enumerate(all_models):
                vals = [
                    fr.per_class_f1.get(cls)
                    for fr in result.folds
                    if fr.model_type == model and cls in fr.per_class_f1
                ]
                if vals:
                    mean = float(_np.mean(vals))
                    item = QTableWidgetItem(f"{mean:.3f}")
                    # Colour-scale: red < 0.5 < orange < 0.75 < green
                    if mean >= 0.75:
                        item.setBackground(QBrush(QColor("#dcfce7")))
                    elif mean >= 0.5:
                        item.setBackground(QBrush(QColor("#fef3c7")))
                    else:
                        item.setBackground(QBrush(QColor("#fee2e2")))
                else:
                    item = QTableWidgetItem("—")
                    item.setForeground(QBrush(QColor("#9ca3af")))
                self.perclass_table.setItem(ri, ci, item)

        if result.output_dir is not None:
            self.output_path_lbl.setText(f"Wrote: {result.output_dir}")
            self.open_folder_btn.setEnabled(True)
        else:
            self.output_path_lbl.setText("Run finished but no output directory was created.")
            self.open_folder_btn.setEnabled(False)

    # ------------------------------------------------------------------
    # YAML save/load
    # ------------------------------------------------------------------

    @Slot()
    def _on_save_yaml(self):
        cfg = self._build_config()
        path, _ = QFileDialog.getSaveFileName(
            self, "Save experiment as YAML",
            str(self.data_dir / f"{cfg.name}.yaml"),
            "YAML / JSON (*.yaml *.yml *.json)",
        )
        if not path:
            return
        p = Path(path)
        try:
            if p.suffix.lower() in (".yaml", ".yml"):
                try:
                    import yaml
                    p.write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False),
                                 encoding="utf-8")
                except ImportError:
                    p = p.with_suffix(".json")
                    dump_experiment(cfg, p)
            else:
                dump_experiment(cfg, p)
            QMessageBox.information(self, "Saved", f"Wrote: {p}")
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "Save failed", str(e))

    @Slot()
    def _on_load_yaml(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load experiment YAML",
            str(self.data_dir),
            "YAML / JSON (*.yaml *.yml *.json)",
        )
        if not path:
            return
        try:
            cfg = load_experiment(Path(path))
            self._apply_config(cfg)
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "Load failed", str(e))

    def _apply_config(self, cfg: ExperimentConfig):
        self.name_edit.setText(cfg.name)
        self.seed_spin.setValue(cfg.seed)
        self.window_ms.setValue(cfg.windowing.window_ms)
        self.stride_ms.setValue(cfg.windowing.stride_ms)

        domains = set(cfg.data.domains or [])
        self.cb_pipeline.setChecked("pipeline" in domains or not domains)
        self.cb_unity.setChecked("unity" in domains)
        self._refresh_pickers()

        if cfg.data.explicit:
            self._data_tabs.setCurrentIndex(1)
            wanted = set(cfg.data.explicit)
            self.session_tree.blockSignals(True)
            try:
                for i in range(self.session_tree.topLevelItemCount()):
                    top = self.session_tree.topLevelItem(i)
                    for j in range(top.childCount()):
                        child = top.child(j)
                        kind, key = child.data(0, Qt.ItemDataRole.UserRole)
                        if kind == "session":
                            child.setCheckState(
                                0,
                                Qt.CheckState.Checked if key in wanted
                                else Qt.CheckState.Unchecked,
                            )
            finally:
                self.session_tree.blockSignals(False)
            self._update_session_count()
        elif cfg.data.subjects:
            self._data_tabs.setCurrentIndex(0)
            wanted = set(cfg.data.subjects)
            for i in range(self.subject_list.count()):
                it = self.subject_list.item(i)
                it.setSelected(it.data(Qt.ItemDataRole.UserRole) in wanted)

        self.features_box.set_selected([f.name for f in cfg.features])
        self.models_box.set_selected(
            [_MODEL_KEY_NORMALISE.get(m.type, m.type.lower()) for m in cfg.models]
        )

        for i in range(self.cv_combo.count()):
            if self.cv_combo.itemData(i) == cfg.cv.strategy:
                self.cv_combo.setCurrentIndex(i)
                break
        if cfg.cv.strategy == "k_fold_subjects":
            self.kfold_spin.setValue(int(cfg.cv.kwargs.get("k", 5)))
        elif cfg.cv.strategy == "cross_domain":
            td = cfg.cv.kwargs.get("train_domain", "pipeline")
            ed = cfg.cv.kwargs.get("test_domain", "unity")
            self.xd_train_combo.setCurrentText(td)
            self.xd_test_combo.setCurrentText(ed)
        elif cfg.cv.strategy == "holdout_split":
            self.holdout_val_slider.setValue(
                int(round(float(cfg.cv.kwargs.get("val_ratio", 0.15)) * 100))
            )
            self.holdout_test_slider.setValue(
                int(round(float(cfg.cv.kwargs.get("test_ratio", 0.15)) * 100))
            )
            strat = cfg.cv.kwargs.get("stratify_by", "subject")
            for i in range(self.holdout_strat_combo.count()):
                if self.holdout_strat_combo.itemData(i) == strat:
                    self.holdout_strat_combo.setCurrentIndex(i)
                    break

        self._refresh_preview()

    # ------------------------------------------------------------------
    # Open output folder
    # ------------------------------------------------------------------

    @Slot()
    def _on_open_folder(self):
        if self._last_result is None or self._last_result.output_dir is None:
            return
        _open_in_file_manager(self._last_result.output_dir)


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------

def _fmt_dur(seconds: float) -> str:
    """Human-readable short duration: 0.8s, 12s, 3m 20s, 1h 15m."""
    if seconds < 0:
        seconds = 0
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        m = int(seconds // 60)
        s = int(seconds % 60)
        return f"{m}m {s}s"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    return f"{h}h {m}m"


def _open_in_file_manager(path: Path) -> None:
    try:
        if sys.platform.startswith("darwin"):
            subprocess.Popen(["open", str(path)])
        elif sys.platform.startswith("win"):
            subprocess.Popen(["explorer", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)])
    except Exception as e:  # noqa: BLE001
        log.warning("Could not open %s: %s", path, e)