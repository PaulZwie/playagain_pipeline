"""
validation_tab.py
─────────────────
GUI front-end for the playagain_pipeline.validation package.

Replaces the legacy PerformanceReviewTab. The legacy tab grew over time
into an ad-hoc validation harness (`SessionPickerWidget` +
`ComparisonWorker` + a feature picker). This widget delivers the same
goal — comparing models / features across recording sessions — but
through the new, reproducible, config-driven harness.

Design goals
────────────
1. **Modular.** Every "knob" is a checkbox or a dropdown — the user
   picks which features, which models, which CV strategy, and which
   subjects in a few clicks.
2. **Reproducible.** Every run is just a temporary
   :class:`ExperimentConfig` that gets serialised to disk alongside its
   results. There is a "Save as YAML" button so an exploratory GUI run
   can be re-run from the CLI later, byte-for-byte.
3. **Honest.** All splitters operate at session granularity — no
   window leakage. The default is leave-one-subject-out, which is the
   single number most appropriate for a paper or thesis.
4. **Non-blocking.** Heavy work runs through the existing
   :func:`run_blocking` helper so the GUI stays responsive and the
   user gets a busy overlay.

The widget purposely depends only on the public API of the validation
package (corpus, config, runner, cv_strategies) and on the existing
busy-overlay helper. It makes no assumptions about which features /
models the project happens to ship today — it queries the live
registries on construction and renders whatever is there.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

from PySide6.QtCore import Qt, Signal, Slot, QTimer
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QGroupBox, QPushButton, QListWidget, QListWidgetItem,
    QCheckBox, QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit,
    QTableWidget, QTableWidgetItem, QHeaderView, QSplitter,
    QScrollArea, QFileDialog, QMessageBox, QTabWidget, QFrame,
    QTextEdit, QProgressBar,
)

from playagain_pipeline.gui.widgets.busy_overlay import run_blocking
from playagain_pipeline.validation import (
    SessionCorpus,
    ExperimentConfig,
    ValidationRunner,
    RunResult,
    cv_strategies,
)
from playagain_pipeline.validation.config import (
    DataSelection, WindowingConfig, FeatureConfig, ModelConfig, CVConfig,
    load_experiment, dump_experiment,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default knobs — sensible offerings if the live registry queries fail.
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

# Normalise any PascalCase / legacy spelling a saved YAML might contain
# to the snake_case key the model registry expects.
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
    "within_session":   "Within-Session  (temporal tail split — optimistic baseline)",
    "k_fold_subjects":  "k-Fold over subjects",
    "cross_domain":     "Cross-Domain  (pipeline ↔ unity)",
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
            # QCheckBox.toggled emits a bool; adapt to this group's no-arg signal.
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
# The main widget
# ---------------------------------------------------------------------------

class ValidationTab(QWidget):
    """
    Modular, reproducible validation harness embedded in the main window.

    Constructed with the application's ``DataManager`` so it can find the
    same data the rest of the GUI uses. All other dependencies are
    discovered lazily.
    """

    def __init__(self, data_manager, parent=None):
        super().__init__(parent)
        self.data_manager = data_manager
        self.data_dir: Path = Path(data_manager.data_dir)

        self._features = self._discover_features()
        self._models = self._discover_models()
        self._corpus = SessionCorpus(self.data_dir)

        self._last_result: Optional[RunResult] = None

        self._build_ui()
        self._refresh_corpus()

    # ------------------------------------------------------------------
    # Registry discovery (with graceful fallback)
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
        # We can't reliably introspect ModelManager (it varies by model),
        # so we offer the same canonical list the Train tab uses.
        return list(_DEFAULT_MODELS)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)

        # Header with one-line explanation + buttons
        header = self._build_header()
        outer.addWidget(header)

        # Main horizontal splitter: knobs on the left, results on the right
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setSizes([460, 700])
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
        self.cb_pipeline.toggled.connect(self._refresh_subject_list)
        domain_row.addWidget(self.cb_pipeline)
        self.cb_unity = QCheckBox("unity")
        self.cb_unity.toggled.connect(self._refresh_subject_list)
        domain_row.addWidget(self.cb_unity)
        domain_row.addStretch()
        data_lay.addLayout(domain_row)

        data_lay.addWidget(QLabel("Subjects (multi-select; empty = all):"))
        self.subject_list = QListWidget()
        self.subject_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.subject_list.setMaximumHeight(110)
        data_lay.addWidget(self.subject_list)

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
        # Sensible defaults: the classic Hudgins set
        self.features_box.set_selected(
            [k for k, _ in self._features if k in {"mav", "rms", "wl", "zc"}]
        )
        layout.addWidget(self.features_box)

        # ── Modular models
        self.models_box = _CheckboxGroup("Models", self._models)
        # Default: enable the three tabular baselines
        self.models_box.set_selected(["lda", "random_forest", "catboost"])
        layout.addWidget(self.models_box)

        # ── CV strategy
        cv_box = QGroupBox("Cross-Validation")
        cv_lay = QVBoxLayout(cv_box)
        self.cv_combo = QComboBox()
        for key in ("loso_subject", "loso_session", "within_session",
                    "k_fold_subjects", "cross_domain"):
            self.cv_combo.addItem(_CV_DESCRIPTIONS.get(key, key), userData=key)
        self.cv_combo.currentIndexChanged.connect(self._on_cv_changed)
        cv_lay.addWidget(self.cv_combo)

        # k-fold knob (only relevant for k_fold_subjects)
        self._kfold_container = QWidget()
        kfold_row = QFormLayout(self._kfold_container)
        kfold_row.setContentsMargins(0, 0, 0, 0)
        self.kfold_spin = QSpinBox()
        self.kfold_spin.setRange(2, 20)
        self.kfold_spin.setValue(5)
        kfold_row.addRow("k (k-fold only):", self.kfold_spin)
        cv_lay.addWidget(self._kfold_container)

        # cross-domain direction (only relevant for cross_domain)
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

        # Initialise visibility
        self._on_cv_changed()

        layout.addWidget(cv_box)

        # ── Run controls
        run_row = QHBoxLayout()
        self.run_btn = QPushButton("▶ Run validation")
        self.run_btn.setFixedHeight(36)
        self.run_btn.setStyleSheet(
            "QPushButton{background:#1a2e4a;color:#06b6d4;"
            "border:1px solid #06b6d4;border-radius:6px;"
            "font-weight:700;font-size:12px;padding:6px 18px;}"
            "QPushButton:hover{background:#06b6d4;color:#fff;}")
        self.run_btn.clicked.connect(self._on_run)
        run_row.addWidget(self.run_btn, 1)
        layout.addLayout(run_row)

        layout.addStretch()
        scroll.setWidget(content)
        return scroll

    # -- Right panel: results table + per-fold breakdown ---------------

    def _build_right_panel(self) -> QWidget:
        wrap = QWidget()
        layout = QVBoxLayout(wrap)

        # ── Live progress panel (shown while a run is in flight) ─────────
        self._progress_panel = QFrame()
        self._progress_panel.setStyleSheet(
            "QFrame { background: #0f172a; border-radius: 6px; border: 1px solid #1e3a5f; }"
        )
        prog_lay = QVBoxLayout(self._progress_panel)
        prog_lay.setContentsMargins(10, 8, 10, 8)
        prog_lay.setSpacing(4)

        prog_title_row = QHBoxLayout()
        _prog_title = QLabel("⚙  Validation in progress…")
        _prog_title_font = QFont()
        _prog_title_font.setBold(True)
        _prog_title_font.setPointSize(11)
        _prog_title.setFont(_prog_title_font)
        _prog_title.setStyleSheet("color: #06b6d4; background: transparent; border: none;")
        prog_title_row.addWidget(_prog_title)
        prog_title_row.addStretch()
        self._prog_step_lbl = QLabel("")
        self._prog_step_lbl.setStyleSheet("color: #94a3b8; font-size: 10px; background: transparent; border: none;")
        prog_title_row.addWidget(self._prog_step_lbl)
        prog_lay.addLayout(prog_title_row)

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 0)          # indeterminate (spinning)
        self._progress_bar.setFixedHeight(6)
        self._progress_bar.setTextVisible(False)
        self._progress_bar.setStyleSheet(
            "QProgressBar { background: #1e293b; border-radius: 3px; border: none; }"
            "QProgressBar::chunk { background: #06b6d4; border-radius: 3px; }"
        )
        prog_lay.addWidget(self._progress_bar)

        self._prog_log = QTextEdit()
        self._prog_log.setReadOnly(True)
        self._prog_log.setFixedHeight(90)
        self._prog_log.setStyleSheet(
            "QTextEdit { background: transparent; color: #94a3b8; "
            "font-family: monospace; font-size: 10px; border: none; }"
        )
        prog_lay.addWidget(self._prog_log)

        self._progress_panel.setVisible(False)
        layout.addWidget(self._progress_panel)

        # Aggregate (per-model) summary
        agg_box = QGroupBox("Aggregate results  (mean ± std across folds)")
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
        layout.addWidget(agg_box, 1)

        # Per-fold breakdown
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
        layout.addWidget(fold_box, 2)

        # Footer with output-folder access
        footer = QHBoxLayout()
        self.output_path_lbl = QLabel("No run yet.")
        self.output_path_lbl.setStyleSheet("color: #64748b; font-size: 10px;")
        footer.addWidget(self.output_path_lbl, 1)

        self.open_folder_btn = QPushButton("Open results folder")
        self.open_folder_btn.setEnabled(False)
        self.open_folder_btn.clicked.connect(self._on_open_folder)
        footer.addWidget(self.open_folder_btn)
        layout.addLayout(footer)

        return wrap

    # ------------------------------------------------------------------
    # Corpus + UI refresh
    # ------------------------------------------------------------------

    def _refresh_corpus(self):
        # Force a re-scan by rebuilding the corpus.
        self._corpus = SessionCorpus(self.data_dir)
        self._corpus.discover()
        self.corpus_summary_lbl.setText(self._corpus.summary().replace("\n", "  ·  "))
        self._refresh_subject_list()

    def _refresh_subject_list(self):
        domains = self._selected_domains()
        # When no domain is checked, show all subjects rather than nothing —
        # an empty domain selection is more likely "don't filter" than "hide all".
        recs = self._corpus.filter(domains=domains) if domains else self._corpus.all()
        subjects = sorted({r.subject_id for r in recs})

        self.subject_list.clear()
        for s in subjects:
            n = sum(1 for r in recs if r.subject_id == s)
            item = QListWidgetItem(f"{s}  ({n} sessions)")
            item.setData(Qt.ItemDataRole.UserRole, s)
            self.subject_list.addItem(item)

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
            return None  # → "all"
        return [it.data(Qt.ItemDataRole.UserRole) for it in items]

    @Slot()
    def _on_cv_changed(self):
        key = self.cv_combo.currentData()
        self._kfold_container.setVisible(key == "k_fold_subjects")
        self._xd_container.setVisible(key == "cross_domain")

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

        return ExperimentConfig(
            name=self.name_edit.text().strip() or "unnamed",
            description="Built from the GUI Validation tab.",
            seed=int(self.seed_spin.value()),
            data=DataSelection(
                subjects=self._selected_subjects(),
                domains=self._selected_domains() or None,
            ),
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
    # Run
    # ------------------------------------------------------------------

    @Slot()
    def _on_run(self):
        try:
            cfg = self._build_config()
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, "Invalid configuration", str(e))
            return

        if not cfg.features:
            QMessageBox.warning(self, "No features",
                                "Select at least one feature.")
            return
        if not cfg.models:
            QMessageBox.warning(self, "No models",
                                "Select at least one model.")
            return

        runner = ValidationRunner(self.data_dir)
        self.run_btn.setEnabled(False)

        # ── Show progress panel ──────────────────────────────────────────
        n_models = len(cfg.models)
        n_features = len(cfg.features)
        cv_label = _CV_DESCRIPTIONS.get(cfg.cv.strategy, cfg.cv.strategy)
        self._prog_log.clear()
        self._progress_panel.setVisible(True)
        self._prog_step_lbl.setText(f"{n_models} model(s) · {n_features} feature(s)")

        def _log(msg: str):
            self._prog_log.append(msg)
            # Scroll to bottom
            sb = self._prog_log.verticalScrollBar()
            sb.setValue(sb.maximum())

        _log(f"Strategy : {cv_label}")
        _log(f"Features : {', '.join(f.name for f in cfg.features)}")
        _log(f"Models   : {', '.join(m.type for m in cfg.models)}")
        subjects = cfg.data.subjects or ["all"]
        _log(f"Subjects : {', '.join(subjects)}")
        _log("─" * 42)

        # Animate step label while running
        _dots = ["   ", ".  ", ".. ", "..."]
        _dot_idx = [0]

        def _tick():
            _dot_idx[0] = (_dot_idx[0] + 1) % len(_dots)
            self._prog_step_lbl.setText(
                f"Running{_dots[_dot_idx[0]]}  "
                f"{n_models} model(s) · {n_features} feature(s)"
            )

        self._run_timer = QTimer(self)
        self._run_timer.setInterval(400)
        self._run_timer.timeout.connect(_tick)
        self._run_timer.start()

        def _do():
            return runner.run(cfg)

        def _done(result: RunResult):
            self._run_timer.stop()
            self.run_btn.setEnabled(True)
            self._last_result = result
            n_folds = len(result.folds)
            _log(f"✓ Finished — {n_folds} fold(s) completed.")
            agg = result.aggregate()
            for model, m in sorted(agg.items()):
                _log(
                    f"  {model:<18}  "
                    f"acc {m['accuracy_mean']:.3f} ± {m['accuracy_std']:.3f}  "
                    f"F1 {m['macro_f1_mean']:.3f}"
                )
            self._prog_step_lbl.setText("Done ✓")
            self._progress_bar.setRange(0, 1)
            self._progress_bar.setValue(1)
            self._progress_bar.setStyleSheet(
                "QProgressBar { background: #1e293b; border-radius: 3px; border: none; }"
                "QProgressBar::chunk { background: #16a34a; border-radius: 3px; }"
            )
            self._populate_results(result)

        def _err(tb: str):
            self._run_timer.stop()
            self.run_btn.setEnabled(True)
            _log(f"✗ Run failed.")
            _log(tb[-600:])
            self._prog_step_lbl.setText("Failed ✗")
            self._progress_bar.setRange(0, 1)
            self._progress_bar.setValue(1)
            self._progress_bar.setStyleSheet(
                "QProgressBar { background: #1e293b; border-radius: 3px; border: none; }"
                "QProgressBar::chunk { background: #dc2626; border-radius: 3px; }"
            )
            log.error("Validation run failed:\n%s", tb)
            QMessageBox.critical(self, "Validation failed",
                                 f"The run raised an exception:\n\n{tb[-1000:]}")

        run_blocking(self, _do, _done, _err,
                     label=f"Running validation ({cfg.cv.strategy})…")

    # ------------------------------------------------------------------
    # Results rendering
    # ------------------------------------------------------------------

    def _populate_results(self, result: RunResult):
        # Aggregate table
        agg = result.aggregate()
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

        # Per-fold table
        self.fold_table.setRowCount(len(result.folds))
        for row, fr in enumerate(result.folds):
            self.fold_table.setItem(row, 0, QTableWidgetItem(fr.fold_id))
            self.fold_table.setItem(row, 1, QTableWidgetItem(fr.model_type))
            self.fold_table.setItem(row, 2, QTableWidgetItem(str(fr.n_train_windows)))
            self.fold_table.setItem(row, 3, QTableWidgetItem(str(fr.n_test_windows)))
            self.fold_table.setItem(row, 4, QTableWidgetItem(f"{fr.accuracy:.3f}"))
            self.fold_table.setItem(row, 5, QTableWidgetItem(f"{fr.macro_f1:.3f}"))

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
                # Best-effort YAML; fall back to JSON if PyYAML missing.
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
        self._refresh_subject_list()

        if cfg.data.subjects:
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

    # ------------------------------------------------------------------
    # Open output folder in the OS file manager
    # ------------------------------------------------------------------

    @Slot()
    def _on_open_folder(self):
        if self._last_result is None or self._last_result.output_dir is None:
            return
        path = str(self._last_result.output_dir)
        try:
            if sys.platform.startswith("darwin"):
                subprocess.Popen(["open", path])
            elif sys.platform.startswith("win"):
                subprocess.Popen(["explorer", path])
            else:
                subprocess.Popen(["xdg-open", path])
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, "Open failed", str(e))