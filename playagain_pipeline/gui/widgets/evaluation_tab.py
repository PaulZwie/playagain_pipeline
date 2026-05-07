"""
gui/widgets/evaluation_tab.py  (v3 — merged)
═══════════════════════════════════════════
The unified evaluation tab.

This file folds the previous ``CrossValidationTab`` (cross_validation_tab.py)
into ``EvaluationTab`` and exposes a single class. Instead of two top-level
sibling tabs, the user picks an *evaluation type* from a dropdown at the
top, and the relevant settings groups, picker, and results panel
appear/hide.

The evaluation types are:

  • Sessions          — run a saved model over training-session recordings
  • Game recordings   — evaluate logged game predictions or replay a model
  • Unity recordings  — RMS-threshold binary evaluation
  • Cross-validation  — multi-run sweep (over models, features or named
                        data subsets), driven by the existing
                        ``playagain_pipeline.validation`` runner

What changed in v3 (vs the previous evaluation_tab.py + cross_validation_tab.py)
─────────────────────────────────────────────────────────────────────────────
* Two files merged into one. Single class, ``EvaluationTab``. The CV
  tab's ``CrossValidationTab`` symbol is kept as a back-compat alias
  pointing at the same class.
* No more inner mode-tab strip — the type dropdown drives a
  :class:`QStackedWidget` for the picker, a :class:`QStackedWidget` for
  the settings, and a :class:`QStackedWidget` for the results.
* Run button label, primary action, and which signal is emitted change
  with the type. The constructor signature is unchanged so wiring in
  ``main_window_v2.py`` only needs to import this class — drop-in.
* All shared primitives (palette, ``_styled_group``, ``_ghost_button``,
  ``_primary_button``, ``_Pill``, ``_form``) live exactly once at the
  top of the file. Each tab used to ship its own copy.
* The cross-validation worker, planner, axis editors, and comparison
  table are unchanged behaviourally — they just live in this file now.

Wiring
──────
``main_window_v2.py`` imports the same name as before::

    from playagain_pipeline.gui.widgets.evaluation_tab import EvaluationTab

The constructor takes a ``DataManager`` (or anything with a ``data_dir``
attribute, or a Path / string pointing at the data directory). The
default mode on open is *Sessions*, matching the old behaviour.
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from PySide6.QtCore import (
    Qt, Signal, Slot, QObject, QThread,
)
from PySide6.QtGui import QBrush, QColor, QFont
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
    QFileDialog, QFormLayout, QFrame, QGroupBox, QHBoxLayout, QHeaderView,
    QLabel, QLineEdit, QListWidget, QListWidgetItem,
    QMenu, QMessageBox, QProgressBar, QPushButton, QScrollArea,
    QSizePolicy, QSpinBox, QSplitter, QStackedWidget, QTabWidget,
    QTableWidget, QTableWidgetItem, QTextEdit, QToolButton,
    QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget,
)

try:
    import pyqtgraph as pg
    _HAS_PYQTGRAPH = True
except Exception:                                            # pragma: no cover
    pg = None
    _HAS_PYQTGRAPH = False

# matplotlib is used only when the user clicks "Save plots…". Imported
# lazily inside _ResultsView._on_export so the GUI starts fast even on
# systems without it.

# ── Evaluation backend (Qt-free) — drives modes 1 to 3 ────────────────────
from playagain_pipeline.evaluation import (
    EvaluationResult, RecordingDescriptor, RecordingKind,
    SessionEvalSettings, GameEvalSettings, UnityEvalSettings,
    discover_sessions, discover_game_recordings, discover_unity_recordings,
    evaluate_sessions, evaluate_games, evaluate_unity,
    evaluate_features_lda,
    TRUTH_RAW, TRUTH_REQUESTED, TRUTH_ACTIVE,
)

# ── Validation runner (CLI-shared) — drives the CV mode ───────────────────
from playagain_pipeline.validation import (
    SessionCorpus, ExperimentConfig, ValidationRunner, RunResult,
)
from playagain_pipeline.validation.runner import FoldResult
from playagain_pipeline.validation.config import (
    DataSelection, WindowingConfig, FeatureConfig, ModelConfig, CVConfig,
)

# ── App style + busy overlay ──────────────────────────────────────────────
from playagain_pipeline.gui.gui_style import apply_app_style
from playagain_pipeline.gui.widgets.busy_overlay import run_blocking

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Bright palette  (kept in sync with gui_style.py "bright")
# ═══════════════════════════════════════════════════════════════════════════

C_BG       = "#f6f8fc"
C_PANEL    = "#ffffff"
C_CARD     = "#eef2ff"
C_CARD2    = "#f1f5f9"      # softer card for nested groups
C_ACCENT   = "#6d28d9"
C_ACCENT2  = "#0284c7"
C_TEXT     = "#111827"
C_MUTED    = "#4b5563"
C_BORDER   = "#c7d2fe"
C_BORDER2  = "#e2e8f0"      # softer border for nested groups
C_GOOD     = "#16a34a"
C_WARN     = "#d97706"
C_BAD      = "#dc2626"


# ═══════════════════════════════════════════════════════════════════════════
# Evaluation-type identifiers
# ═══════════════════════════════════════════════════════════════════════════
# A small, hand-rolled enum-ish set of strings is plenty here — the set is
# closed, consumed only by this file, and using strings keeps the QComboBox
# itemData() round-trip frictionless.

EVAL_TYPE_SESSIONS = "sessions"
EVAL_TYPE_GAMES    = "games"
EVAL_TYPE_UNITY    = "unity"
EVAL_TYPE_CV       = "cross_validation"

_EVAL_TYPE_LABELS: List[Tuple[str, str]] = [
    (EVAL_TYPE_SESSIONS, "Sessions  (run a saved model over training recordings)"),
    (EVAL_TYPE_GAMES,    "Game recordings  (logged predictions or replay)"),
    (EVAL_TYPE_UNITY,    "Unity recordings  (RMS threshold)"),
    (EVAL_TYPE_CV,       "Cross-validation  (multi-run sweep)"),
]


# ═══════════════════════════════════════════════════════════════════════════
# CV-mode catalog  (models / features / strategies)
# ═══════════════════════════════════════════════════════════════════════════
# These lists are the SOURCE OF TRUTH for what shows up in the CV combo
# boxes. They are intentionally separate from any defaults the runner
# itself ships with: those defaults are a runner concern, not a UI one,
# and the UI's job is to be honest about *exactly* what the user picked.

_AVAILABLE_FEATURES: List[Tuple[str, str]] = [
    ("mav",  "Mean Absolute Value"),
    ("rms",  "Root Mean Square"),
    ("wl",   "Waveform Length"),
    ("zc",   "Zero Crossings"),
    ("ssc",  "Slope Sign Changes"),
    ("var",  "Variance"),
    ("iemg", "Integrated EMG"),
]

_AVAILABLE_MODELS: List[Tuple[str, str]] = [
    ("lda",           "Very fast linear discriminant — strong EMG baseline"),
    ("random_forest", "Robust, handles noise well, no scaling needed"),
    ("catboost",      "Gradient boosting — often best on tabular features"),
    ("svm",           "Linear / RBF support vector machine"),
    ("mlp",           "Small neural net — needs ≳10 k windows"),
    ("attention_net", "Transformer attention — strongest temporal model"),
]

_CV_STRATEGIES: List[Tuple[str, str, str]] = [
    # (key, short label, longer tooltip)
    ("loso_subject",    "Leave-One-Subject-Out",
        "Train on every subject but one, test on the held-out subject. "
        "Repeats once per subject. The honest single number for a paper."),
    ("loso_session",    "Leave-One-Session-Out",
        "Train on every session but one, test on the held-out session. "
        "Useful for measuring session-to-session drift within the same person."),
    ("k_fold_subjects", "k-Fold over subjects",
        "Shuffle subjects, split into k roughly-equal groups, train on k-1 "
        "and test on the remaining one. Useful when LOSO has too many "
        "subjects to run quickly."),
    ("within_session",  "Within-Session",
        "Train on the first 80% of each session's windows, test on the last "
        "20%. Optimistic — windows from the same session share hardware "
        "placement and user state — but useful as a ceiling."),
    ("cross_domain",    "Cross-Domain",
        "Train only on one domain (pipeline / unity / game) and test on "
        "another. Measures transfer between recorder types."),
    ("holdout_split",   "Holdout (train / val / test)",
        "One fold with explicit Train / Val / Test ratios. Best for tuning "
        "a single model with early stopping; the test split stays untouched "
        "until the final number."),
]

_DEFAULT_FEATURE_SET: List[str] = ["mav", "rms", "wl"]
_DEFAULT_MODEL: str             = "random_forest"
_DEFAULT_FEATURE_PRESETS: List[Tuple[str, List[str]]] = [
    ("Time-domain trio  (mav, rms, wl)",       ["mav", "rms", "wl"]),
    ("Hudgins five  (mav, wl, zc, ssc, var)",  ["mav", "wl", "zc", "ssc", "var"]),
    ("Full battery  (all 7)",
        ["mav", "rms", "wl", "zc", "ssc", "var", "iemg"]),
    ("Energy only  (rms, iemg, var)",          ["rms", "iemg", "var"]),
]


# ═══════════════════════════════════════════════════════════════════════════
# Shared UI primitives (single copy; reused across all four modes)
# ═══════════════════════════════════════════════════════════════════════════

def _styled_group(title: str) -> QGroupBox:
    """A group box with the same look across all settings panels."""
    g = QGroupBox(title)
    g.setStyleSheet(
        f"QGroupBox {{"
        f"  background:{C_PANEL}; border:1px solid {C_BORDER2};"
        f"  border-radius:8px; margin-top:14px; padding-top:14px;"
        f"  font-weight:600; color:{C_TEXT};"
        f"}}"
        f"QGroupBox::title {{"
        f"  subcontrol-origin: margin; left:12px; padding:0 6px;"
        f"  color:{C_ACCENT2}; font-size:10px; letter-spacing:0.05em;"
        f"  text-transform: uppercase;"
        f"}}"
    )
    return g


def _ghost_button(text: str, *, accent: bool = False) -> QToolButton:
    """A small flat button matching the bright-theme pillbar look."""
    b = QToolButton(); b.setText(text)
    color, border = ((C_ACCENT, C_ACCENT) if accent else (C_TEXT, C_BORDER))
    b.setStyleSheet(
        f"QToolButton {{ background:{C_PANEL}; color:{color};"
        f"  border:1px solid {border}; border-radius:6px; padding:5px 11px;"
        f"  font-size:11px; }}"
        f"QToolButton:hover {{ color:{C_ACCENT2}; border-color:{C_ACCENT2}; }}"
        f"QToolButton::menu-indicator {{ image: none; }}"
    )
    return b


def _primary_button(text: str) -> QPushButton:
    """The big, hard-to-miss Run button at the bottom of each panel."""
    b = QPushButton(text)
    b.setStyleSheet(
        f"QPushButton {{ background:{C_ACCENT}; color:white; border:none;"
        f"  border-radius:8px; padding:11px 18px; font-size:13px; font-weight:600;"
        f"  letter-spacing:0.02em; }}"
        f"QPushButton:hover {{ background:{C_ACCENT2}; }}"
        f"QPushButton:disabled {{ background:#cbd5e1; color:#94a3b8; }}"
    )
    b.setMinimumHeight(40)
    return b


def _form() -> QFormLayout:
    """A consistent form layout used inside every settings group-box."""
    f = QFormLayout()
    f.setHorizontalSpacing(12); f.setVerticalSpacing(8)
    f.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    f.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
    return f


class _Pill(QLabel):
    """Compact status chip used in headers."""

    def __init__(self, text: str = "", parent: Optional[QWidget] = None):
        super().__init__(text, parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet(
            f"QLabel {{ background:{C_CARD}; color:{C_ACCENT2};"
            f"  border:1px solid {C_BORDER}; border-radius:9px;"
            f"  padding:2px 9px; font-size:10px; font-weight:600; }}"
        )


class _MetricCard(QFrame):
    """Labelled tile with a big number and a small caption."""

    def __init__(self, title: str, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setStyleSheet(
            f"_MetricCard {{ background:{C_CARD}; border:1px solid {C_BORDER};"
            f"  border-radius:10px; }}"
        )
        self.setMinimumHeight(82); self.setMinimumWidth(130)
        col = QVBoxLayout(self); col.setContentsMargins(15, 9, 15, 11); col.setSpacing(2)

        self._title = QLabel(title)
        f = QFont(); f.setPointSize(9); f.setBold(True); self._title.setFont(f)
        self._title.setStyleSheet(f"color:{C_MUTED}; letter-spacing:0.05em;")
        col.addWidget(self._title)

        self._value = QLabel("—")
        f2 = QFont(); f2.setPointSize(22); f2.setBold(True); self._value.setFont(f2)
        self._value.setStyleSheet(f"color:{C_TEXT};")
        col.addWidget(self._value)

        self._caption = QLabel("")
        f3 = QFont(); f3.setPointSize(8); self._caption.setFont(f3)
        self._caption.setStyleSheet(f"color:{C_MUTED};")
        col.addWidget(self._caption)

    def set_title(self, title: str) -> None:
        self._title.setText(title)

    def set_value(self, value, *, fmt: str = "{:.3f}", caption: str = "") -> None:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            self._value.setText("—")
        elif isinstance(value, float) and 0.0 <= value <= 1.0 and "%" not in fmt:
            self._value.setText(f"{value * 100:.1f}%")
        else:
            self._value.setText(fmt.format(value))
        self._caption.setText(caption)


def _model_dirs(data_dir: Path) -> List[str]:
    """Return saved model names from <data_dir>/models, newest first."""
    md = Path(data_dir) / "models"
    if not md.exists():
        return []
    children = [c for c in md.iterdir() if c.is_dir() and not c.name.startswith("_")]
    children.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return [c.name for c in children]


def _settings_scroll_host(group_widgets: List[QWidget],
                           run_row_widget: QWidget) -> QWidget:
    """
    Build a settings column: a scroll area holding a stack of
    group-boxes, plus a button row pinned to the bottom outside the
    scroll area so it stays visible even when the settings overflow.
    """
    column = QWidget()
    col_lay = QVBoxLayout(column)
    col_lay.setContentsMargins(0, 0, 0, 0)
    col_lay.setSpacing(10)

    scroll = QScrollArea()
    scroll.setFrameShape(QFrame.Shape.NoFrame)
    scroll.setWidgetResizable(True)
    inner = QWidget()
    inner_lay = QVBoxLayout(inner)
    inner_lay.setContentsMargins(2, 2, 2, 2)
    inner_lay.setSpacing(12)
    for g in group_widgets:
        inner_lay.addWidget(g)
    inner_lay.addStretch(1)
    scroll.setWidget(inner)
    col_lay.addWidget(scroll, 1)

    col_lay.addWidget(run_row_widget)
    return column


def _score_colour(score: float) -> str:
    """Map a 0-1 score to a green/amber/red foreground colour."""
    if score >= 0.8:  return C_GOOD
    if score >= 0.6:  return C_WARN
    return C_BAD


def _fmt_dur(seconds: float) -> str:
    """1.2 s · 4.7 m · 1 h 02 m — the same format the legacy validation tab used."""
    if seconds is None or seconds != seconds:  # NaN check
        return "—"
    if seconds < 60:
        return f"{seconds:.1f} s"
    minutes = seconds / 60.0
    if minutes < 60:
        return f"{minutes:.1f} m"
    hours = int(minutes // 60)
    mins  = int(minutes % 60)
    return f"{hours} h {mins:02d} m"


# ═══════════════════════════════════════════════════════════════════════════
# Subject-grouped tree picker  (used by the three evaluation modes)
# ═══════════════════════════════════════════════════════════════════════════

# Marker stored in the tree-item user-role data so we can tell
# parent (subject) items apart from leaf (recording) items.
_PARENT_ROLE = Qt.ItemDataRole.UserRole + 1
_INDEX_ROLE  = Qt.ItemDataRole.UserRole + 2


class _TreePicker(QWidget):
    """
    Subject-grouped recording picker.

    Each subject is a parent row with a tristate checkbox. Children are the
    individual recordings. Toggling a parent toggles all its children;
    individual children can also be toggled, which puts the parent in the
    PartiallyChecked state.
    """

    selection_changed = Signal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._descriptors: List[RecordingDescriptor] = []
        # subject -> list of descriptor indices, preserving insertion order
        self._by_subject: Dict[str, List[int]] = {}

        outer = QVBoxLayout(self); outer.setContentsMargins(0, 0, 0, 0); outer.setSpacing(8)

        # Toolbar
        toolbar = QHBoxLayout(); toolbar.setSpacing(6)
        self._btn_all     = _ghost_button("Select all")
        self._btn_none    = _ghost_button("Clear")
        self._btn_expand  = _ghost_button("Expand all")
        self._btn_collapse = _ghost_button("Collapse all")
        self._btn_refresh = _ghost_button("⟳  Refresh")
        for b in (self._btn_all, self._btn_none, self._btn_expand,
                  self._btn_collapse, self._btn_refresh):
            toolbar.addWidget(b)
        toolbar.addStretch(1)
        outer.addLayout(toolbar)

        self._btn_all     .clicked.connect(self._select_all)
        self._btn_none    .clicked.connect(self._select_none)
        self._btn_expand  .clicked.connect(lambda: self._tree.expandAll())
        self._btn_collapse.clicked.connect(lambda: self._tree.collapseAll())
        self._btn_refresh .clicked.connect(self.refresh)

        # The tree itself
        self._tree = QTreeWidget()
        self._tree.setHeaderHidden(True)
        self._tree.setUniformRowHeights(False)
        self._tree.setIndentation(18)
        self._tree.setStyleSheet(
            f"QTreeWidget {{ background:{C_PANEL}; border:1px solid {C_BORDER};"
            f"  border-radius:7px; padding:4px;"
            f"  outline: none; }}"
            f"QTreeWidget::item {{ padding:4px 4px; border-radius:4px;}}"
            f"QTreeWidget::item:selected {{ background:{C_CARD}; color:{C_TEXT}; }}"
            f"QTreeWidget::item:hover {{ background:{C_CARD2}; }}"
            f"QTreeWidget::branch {{ background: transparent; }}"
        )
        self._tree.itemChanged.connect(self._on_item_changed)
        outer.addWidget(self._tree, 1)

        # Footer
        self._count_lbl = QLabel("")
        self._count_lbl.setStyleSheet(f"color:{C_MUTED}; font-size:10px;")
        outer.addWidget(self._count_lbl)

        # Block signal storms while we re-set check states programmatically
        self._suppress_changes = False

    # ── public API ────────────────────────────────────────────────────

    def refresh(self) -> None:                  # subclasses override
        ...

    def selected(self) -> List[RecordingDescriptor]:
        """All checked leaves, in their original discovery order."""
        out: List[RecordingDescriptor] = []
        for i in range(self._tree.topLevelItemCount()):
            parent = self._tree.topLevelItem(i)
            for j in range(parent.childCount()):
                child = parent.child(j)
                if child.checkState(0) == Qt.CheckState.Checked:
                    idx = child.data(0, _INDEX_ROLE)
                    if isinstance(idx, int) and 0 <= idx < len(self._descriptors):
                        out.append(self._descriptors[idx])
        return out

    # ── helpers used by subclasses ───────────────────────────────────

    def _populate(
        self,
        descriptors: List[RecordingDescriptor],
        sublines_by_index: Optional[Dict[int, str]] = None,
    ) -> None:
        """Rebuild the tree from a flat descriptor list."""
        self._descriptors = list(descriptors)
        self._by_subject = {}
        for i, d in enumerate(descriptors):
            self._by_subject.setdefault(d.subject_id, []).append(i)

        self._suppress_changes = True
        self._tree.clear()
        for subject in sorted(self._by_subject.keys(),
                              key=lambda s: (not s.upper().startswith("VP_"), s)):
            indices = self._by_subject[subject]
            parent = QTreeWidgetItem(self._tree)
            parent.setData(0, _PARENT_ROLE, True)
            parent.setText(0, f"{subject}    ({len(indices)})")
            f = QFont(); f.setBold(True); f.setPointSize(11); parent.setFont(0, f)
            parent.setForeground(0, QBrush(QColor(C_TEXT)))
            parent.setFlags(
                parent.flags()
                | Qt.ItemFlag.ItemIsUserCheckable
                | Qt.ItemFlag.ItemIsAutoTristate
            )
            parent.setCheckState(0, Qt.CheckState.Unchecked)

            for idx in indices:
                d = descriptors[idx]
                # The leaf label is the session/recording id; the subject
                # is already in the parent row, so we drop "VP_xx · " here.
                label = d.session_id if d.session_id else d.label
                child = QTreeWidgetItem(parent)
                child.setData(0, _PARENT_ROLE, False)
                child.setData(0, _INDEX_ROLE, idx)
                sub = (sublines_by_index or {}).get(idx, "")
                child.setText(0, f"{label}\n  {sub}" if sub else label)
                child.setForeground(0, QBrush(QColor(C_TEXT)))
                child.setFlags(child.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                child.setCheckState(0, Qt.CheckState.Unchecked)
        # Start collapsed when there are many subjects, expanded when few.
        if self._tree.topLevelItemCount() <= 4:
            self._tree.expandAll()
        self._suppress_changes = False
        self._update_count()

    # ── internal slots ────────────────────────────────────────────────

    def _on_item_changed(self, item: QTreeWidgetItem, _col: int) -> None:
        if self._suppress_changes:
            return
        is_parent = bool(item.data(0, _PARENT_ROLE))
        if is_parent:
            # Propagate parent → children, but only when fully checked or
            # fully unchecked (PartiallyChecked is set by Qt itself when
            # children disagree, and we should NOT cascade in that case).
            state = item.checkState(0)
            if state in (Qt.CheckState.Checked, Qt.CheckState.Unchecked):
                self._suppress_changes = True
                for j in range(item.childCount()):
                    item.child(j).setCheckState(0, state)
                self._suppress_changes = False
        self._update_count()
        self.selection_changed.emit()

    def _select_all(self) -> None:
        self._suppress_changes = True
        for i in range(self._tree.topLevelItemCount()):
            self._tree.topLevelItem(i).setCheckState(0, Qt.CheckState.Checked)
            for j in range(self._tree.topLevelItem(i).childCount()):
                self._tree.topLevelItem(i).child(j).setCheckState(0, Qt.CheckState.Checked)
        self._suppress_changes = False
        self._update_count()
        self.selection_changed.emit()

    def _select_none(self) -> None:
        self._suppress_changes = True
        for i in range(self._tree.topLevelItemCount()):
            self._tree.topLevelItem(i).setCheckState(0, Qt.CheckState.Unchecked)
            for j in range(self._tree.topLevelItem(i).childCount()):
                self._tree.topLevelItem(i).child(j).setCheckState(0, Qt.CheckState.Unchecked)
        self._suppress_changes = False
        self._update_count()
        self.selection_changed.emit()

    def _update_count(self) -> None:
        n_total = len(self._descriptors)
        n_sel = len(self.selected())
        n_subj = len(self._by_subject)
        self._count_lbl.setText(
            f"{n_sel} of {n_total} recordings selected   ·   {n_subj} subjects"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Concrete pickers (one per evaluation mode)
# ═══════════════════════════════════════════════════════════════════════════

# Subdirectories of data/sessions/ that aren't real subjects but contain
# loose recordings. We still want to show them so the user can act on
# them, but they're listed below the real VP_xx subjects.
_PSEUDO_SUBJECTS = {"emg", "spacebar", "Calibration", "KinderUni"}


class _SessionPicker(_TreePicker):
    def __init__(self, data_dir: Path, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._data_dir = Path(data_dir)
        self.refresh()

    def refresh(self) -> None:
        descs = discover_sessions(self._data_dir, include_unity=False)
        sublines = {
            i: f"{d.meta.get('device_name','?')} · "
               f"{d.meta.get('num_channels','?')} ch · "
               f"{d.meta.get('n_trials','?')} trials"
            for i, d in enumerate(descs)
        }
        self._populate(descs, sublines)


class _GamePicker(_TreePicker):
    def __init__(self, data_dir: Path, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._data_dir = Path(data_dir)
        self.refresh()

    def refresh(self) -> None:
        descs = discover_game_recordings(self._data_dir)
        sublines: Dict[int, str] = {}
        for i, d in enumerate(descs):
            classes = d.meta.get("class_names") or []
            secs = d.meta.get("duration_seconds")
            dur = f"{secs:.0f} s" if isinstance(secs, (int, float)) else "? s"
            sublines[i] = (
                f"{dur}  ·  {len(classes)} classes  ·  "
                f"model: {d.meta.get('model_name', '?')}"
            )
        self._populate(descs, sublines)


class _UnityPicker(_TreePicker):
    """Unity picker also has a folder selector — user picks where to scan."""

    def __init__(self, data_dir: Path, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._data_dir = Path(data_dir)
        self._source: Path = self._data_dir / "sessions" / "unity_sessions"

        # Insert a path-selector row above the toolbar
        path_row = QHBoxLayout(); path_row.setSpacing(6)
        path_row.addWidget(QLabel("Source:"))
        self._path_edit = QLineEdit(str(self._source)); self._path_edit.setReadOnly(True)
        self._path_edit.setStyleSheet(
            f"QLineEdit {{ background:{C_BG}; color:{C_MUTED};"
            f"  border:1px solid {C_BORDER2}; border-radius:5px; padding:4px 8px;"
            f"  font-family: 'Menlo','DejaVu Sans Mono', monospace; font-size:10px; }}"
        )
        browse_btn = _ghost_button("Browse…")
        browse_btn.clicked.connect(self._on_browse)
        path_row.addWidget(self._path_edit, 1)
        path_row.addWidget(browse_btn)
        self.layout().insertLayout(0, path_row)

        self.refresh()

    def _on_browse(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Select Unity recordings folder",
                                              str(self._source))
        if d:
            self._source = Path(d); self._path_edit.setText(d); self.refresh()

    def refresh(self) -> None:
        descs = discover_unity_recordings(self._source, recurse=True)
        sublines: Dict[int, str] = {}
        for i, d in enumerate(descs):
            fmt = d.meta.get("format", "?")
            ch  = d.meta.get("num_channels", "?")
            trials = d.meta.get("n_trials")
            sublines[i] = (
                f"{fmt}" + (f" · {ch} ch · {trials} trials"
                            if fmt == "session" else "")
            )
        self._populate(descs, sublines)


# ═══════════════════════════════════════════════════════════════════════════
# Per-result detail widgets  (used by the eval-result panel)
# ═══════════════════════════════════════════════════════════════════════════

class _ConfusionView(QWidget):
    """Heat-mapped confusion matrix as a styled QTableWidget."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        outer = QVBoxLayout(self); outer.setContentsMargins(0, 0, 0, 0)

        cap = QLabel("Predicted →    True ↓"); cap.setStyleSheet(f"color:{C_MUTED}; font-size:10px;")
        outer.addWidget(cap)

        self._table = QTableWidget()
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self._table.setShowGrid(False)
        self._table.setStyleSheet(
            f"QTableWidget {{ background:{C_PANEL}; border:1px solid {C_BORDER2};"
            f"  border-radius:7px; gridline-color:{C_BORDER2}; }}"
            f"QTableWidget::item {{ padding:8px; }}"
            f"QHeaderView::section {{ background:{C_BG}; border:none;"
            f"  padding:6px 8px; font-weight:600; color:{C_MUTED}; }}"
        )
        outer.addWidget(self._table, 1)

    def set_matrix(self, cm) -> None:
        if cm is None:
            self._table.clear(); self._table.setRowCount(0); self._table.setColumnCount(0); return
        labels = list(cm.label_names); n = len(labels)
        m = np.asarray(cm.matrix, dtype=np.int64)
        self._table.setRowCount(n); self._table.setColumnCount(n)
        self._table.setHorizontalHeaderLabels(labels)
        self._table.setVerticalHeaderLabels(labels)
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

        row_sums = m.sum(axis=1, keepdims=True); row_sums[row_sums == 0] = 1
        for i in range(n):
            for j in range(n):
                val = int(m[i, j])
                ratio = float(m[i, j]) / float(row_sums[i, 0])
                item = QTableWidgetItem(f"{val:,}\n{ratio * 100:.1f}%")
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if i == j:
                    color = QColor(34, 197, 94, int(50 + ratio * 175))
                else:
                    color = QColor(220, 38, 38, int(35 + ratio * 175))
                item.setBackground(QBrush(color))
                item.setForeground(QBrush(QColor("white") if ratio > 0.55 else QColor(C_TEXT)))
                self._table.setItem(i, j, item)


class _PerClassTable(QWidget):
    """Sortable table of per-class precision / recall / F1 / support."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        outer = QVBoxLayout(self); outer.setContentsMargins(0, 0, 0, 0)

        self._table = QTableWidget()
        self._table.setColumnCount(5)
        self._table.setHorizontalHeaderLabels(["class", "precision", "recall", "F1", "support"])
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setAlternatingRowColors(True)
        self._table.setStyleSheet(
            f"QTableWidget {{ background:{C_PANEL}; border:1px solid {C_BORDER2};"
            f"  border-radius:7px; alternate-background-color:{C_BG};"
            f"  gridline-color:{C_BORDER2}; }}"
            f"QTableWidget::item {{ padding:6px 9px; }}"
            f"QHeaderView::section {{ background:{C_BG}; border:none;"
            f"  padding:6px 8px; font-weight:600; color:{C_MUTED}; }}"
        )
        self._table.setSortingEnabled(True)
        outer.addWidget(self._table)

    def set_rows(self, rows: List[Any]) -> None:
        self._table.setSortingEnabled(False)
        self._table.setRowCount(len(rows))
        for i, r in enumerate(rows):
            name_item = QTableWidgetItem(str(r.name))
            name_item.setData(Qt.ItemDataRole.UserRole, int(r.label))
            self._table.setItem(i, 0, name_item)
            for j, v in enumerate((r.precision, r.recall, r.f1)):
                cell = QTableWidgetItem(f"{v * 100:.1f}%")
                cell.setData(Qt.ItemDataRole.UserRole, float(v))
                cell.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if v >= 0.8:    cell.setForeground(QBrush(QColor(C_GOOD)))
                elif v >= 0.6:  cell.setForeground(QBrush(QColor(C_WARN)))
                else:           cell.setForeground(QBrush(QColor(C_BAD)))
                self._table.setItem(i, 1 + j, cell)
            sup = QTableWidgetItem(f"{r.support:,}")
            sup.setData(Qt.ItemDataRole.UserRole, int(r.support))
            sup.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self._table.setItem(i, 4, sup)
        self._table.setSortingEnabled(True)
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)


class _ThresholdSweepView(QWidget):
    """pyqtgraph plot of threshold-sweep metrics + a marker at the chosen threshold."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        outer = QVBoxLayout(self); outer.setContentsMargins(0, 0, 0, 0)
        if not _HAS_PYQTGRAPH:
            outer.addWidget(QLabel("pyqtgraph not installed — threshold sweep unavailable.")); return
        pg.setConfigOption("background", "w"); pg.setConfigOption("foreground", C_TEXT)
        self._plot = pg.PlotWidget()
        self._plot.setLabel("left",   "metric value")
        self._plot.setLabel("bottom", "RMS threshold")
        self._plot.showGrid(x=True, y=True, alpha=0.3)
        self._plot.addLegend(offset=(10, 5))
        outer.addWidget(self._plot)

    def set_data(self, sweep: List[Any], *, chosen: Optional[float] = None) -> None:
        if not _HAS_PYQTGRAPH:
            return
        self._plot.clear()
        if not sweep:
            return
        x  = np.asarray([p.threshold    for p in sweep])
        f1 = np.asarray([p.f1           for p in sweep])
        pr = np.asarray([p.precision    for p in sweep])
        rc = np.asarray([p.recall       for p in sweep])
        sp = np.asarray([p.specificity  for p in sweep])
        self._plot.plot(x, f1, name="F1",          pen=pg.mkPen(C_ACCENT,  width=3))
        self._plot.plot(x, pr, name="precision",   pen=pg.mkPen(C_ACCENT2, width=2))
        self._plot.plot(x, rc, name="recall",      pen=pg.mkPen(C_GOOD,    width=2))
        self._plot.plot(x, sp, name="specificity", pen=pg.mkPen(C_WARN,    width=2))
        if chosen is not None and not np.isnan(chosen):
            line = pg.InfiniteLine(pos=float(chosen), angle=90,
                                    pen=pg.mkPen(C_BAD, width=2, style=Qt.PenStyle.DashLine),
                                    label=f"chosen = {chosen:.4g}",
                                    labelOpts={"color": C_BAD, "position": 0.85})
            self._plot.addItem(line)


class _FeatureBarView(QWidget):
    """Horizontal bar chart of per-feature accuracy."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        outer = QVBoxLayout(self); outer.setContentsMargins(0, 0, 0, 0)
        if not _HAS_PYQTGRAPH:
            outer.addWidget(QLabel("pyqtgraph not installed — feature chart unavailable.")); return
        self._plot = pg.PlotWidget()
        self._plot.setLabel("bottom", "validation accuracy")
        self._plot.showGrid(x=True, y=False, alpha=0.3)
        outer.addWidget(self._plot)

    def set_data(self, scores: Dict[str, float]) -> None:
        if not _HAS_PYQTGRAPH:
            return
        self._plot.clear()
        if not scores:
            return
        items = sorted(scores.items(), key=lambda kv: kv[1])
        names = [k for k, _ in items]
        vals  = [float(v) if v == v else 0.0 for _, v in items]
        bg = pg.BarGraphItem(x0=0, y=range(len(vals)), height=0.7, width=vals,
                             brush=C_ACCENT2, pen=pg.mkPen(C_ACCENT, width=1))
        self._plot.addItem(bg)
        ax = self._plot.getAxis("left"); ax.setTicks([list(enumerate(names))])
        self._plot.setXRange(0, 1); self._plot.setYRange(-0.5, len(vals) - 0.5)


class _BoxPlotView(QWidget):
    """Per-class F1 box-and-whisker plot across recordings (pyqtgraph)."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        outer = QVBoxLayout(self); outer.setContentsMargins(0, 0, 0, 0)
        if not _HAS_PYQTGRAPH:
            outer.addWidget(QLabel("pyqtgraph not installed — box plot unavailable.")); return
        self._plot = pg.PlotWidget()
        self._plot.setLabel("left",   "F1")
        self._plot.setLabel("bottom", "class")
        self._plot.showGrid(x=False, y=True, alpha=0.3)
        self._plot.setYRange(0, 1)
        outer.addWidget(self._plot)

    def set_data(
        self,
        per_recording: List[Dict[str, Any]],
        label_names: Dict[int, str],
    ) -> None:
        if not _HAS_PYQTGRAPH:
            return
        self._plot.clear()
        if not per_recording:
            return
        # Gather {label_id: [f1 across recordings]}
        buckets: Dict[int, List[float]] = {}
        for row in per_recording:
            pcf1 = row.get("per_class_f1") or {}
            for k, v in pcf1.items():
                try:
                    lbl = int(k)
                except (TypeError, ValueError):
                    continue
                buckets.setdefault(lbl, []).append(float(v))
        if not buckets:
            return
        labels = sorted(buckets.keys())
        names  = [label_names.get(int(l), f"class_{int(l)}") for l in labels]
        # Draw one box per class
        for x, lbl in enumerate(labels):
            vals = np.asarray(buckets[lbl], dtype=np.float64)
            if vals.size == 0:
                continue
            q1, med, q3 = np.percentile(vals, [25, 50, 75])
            lo, hi = float(np.min(vals)), float(np.max(vals))
            # Whiskers
            self._plot.plot([x, x], [lo, hi],
                            pen=pg.mkPen(C_ACCENT, width=2))
            # Box body (filled rectangle as a BarGraphItem)
            bg = pg.BarGraphItem(x=[x], height=[q3 - q1], y0=q1, width=0.5,
                                  brush=C_CARD, pen=pg.mkPen(C_ACCENT, width=2))
            self._plot.addItem(bg)
            # Median line
            self._plot.plot([x - 0.25, x + 0.25], [med, med],
                            pen=pg.mkPen(C_ACCENT, width=3))
            # Individual sample dots (jittered)
            jitter = (np.random.RandomState(lbl).rand(vals.size) - 0.5) * 0.18
            sx = np.full(vals.size, x) + jitter
            self._plot.plot(sx, vals, pen=None,
                            symbol="o", symbolSize=5,
                            symbolBrush=pg.mkBrush(C_ACCENT2),
                            symbolPen=None)
        # X-axis tick labels
        ax = self._plot.getAxis("bottom")
        ax.setTicks([list(zip(range(len(names)), names))])
        self._plot.setXRange(-0.6, len(labels) - 0.4)


# ═══════════════════════════════════════════════════════════════════════════
# Per-result results view  (used for evaluation modes 1-3)
# ═══════════════════════════════════════════════════════════════════════════

class _EvaluationResultView(QWidget):
    """Renders any :class:`EvaluationResult` regardless of source kind."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._current: Optional[EvaluationResult] = None
        outer = QVBoxLayout(self); outer.setContentsMargins(0, 0, 0, 0); outer.setSpacing(10)

        # ── Header  (title + provenance + export menu)
        header = QFrame()
        hlay = QHBoxLayout(header); hlay.setContentsMargins(2, 2, 2, 2); hlay.setSpacing(8)

        text_col = QVBoxLayout(); text_col.setContentsMargins(0, 0, 0, 0); text_col.setSpacing(2)
        self._title_lbl = QLabel("No evaluation run yet")
        f = QFont(); f.setPointSize(15); f.setBold(True); self._title_lbl.setFont(f)
        self._title_lbl.setStyleSheet(f"color:{C_TEXT};")
        text_col.addWidget(self._title_lbl)
        self._sub_lbl = QLabel(
            "Pick a source on the left, set options on the right, then press Run.")
        self._sub_lbl.setStyleSheet(f"color:{C_MUTED}; font-size:11px;")
        self._sub_lbl.setWordWrap(True)
        text_col.addWidget(self._sub_lbl)
        self._pills = QHBoxLayout()
        self._pills.setSpacing(6); self._pills.setContentsMargins(0, 4, 0, 0)
        ph = QWidget(); ph.setLayout(self._pills); text_col.addWidget(ph)
        text_holder = QWidget(); text_holder.setLayout(text_col)
        hlay.addWidget(text_holder, 1)

        # Export button (with dropdown)
        self._btn_export = _ghost_button("⤓  Export…", accent=True)
        self._btn_export.setEnabled(False)
        self._export_menu = QMenu(self._btn_export)
        a_pkg   = self._export_menu.addAction("Save full bundle (folder)…")
        self._export_menu.addSeparator()
        a_cm    = self._export_menu.addAction("Confusion matrix PNG…")
        a_box   = self._export_menu.addAction("Per-class F1 box plot PNG…")
        a_sweep = self._export_menu.addAction("Threshold sweep PNG…")
        a_feat  = self._export_menu.addAction("Per-feature accuracy PNG…")
        a_roc   = self._export_menu.addAction("ROC curve PNG…")
        a_calib = self._export_menu.addAction("Calibration plot PNG…")
        a_scatter = self._export_menu.addAction("Per-recording scatter PNG…")
        self._export_menu.addSeparator()
        a_csv   = self._export_menu.addAction("Per-class metrics CSV…")
        a_json  = self._export_menu.addAction("Result JSON…")
        a_pkg   .triggered.connect(self._on_export_bundle)
        a_cm    .triggered.connect(lambda: self._on_export_one("confusion_png"))
        a_box   .triggered.connect(lambda: self._on_export_one("boxplot_png"))
        a_sweep .triggered.connect(lambda: self._on_export_one("threshold_png"))
        a_feat  .triggered.connect(lambda: self._on_export_one("feature_bar_png"))
        a_roc   .triggered.connect(lambda: self._on_export_one("roc_png"))
        a_calib .triggered.connect(lambda: self._on_export_one("calibration_png"))
        a_scatter.triggered.connect(lambda: self._on_export_one("scatter_png"))
        a_csv   .triggered.connect(lambda: self._on_export_one("metrics_csv"))
        a_json  .triggered.connect(lambda: self._on_export_one("result_json"))
        self._btn_export.setMenu(self._export_menu)
        self._btn_export.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        hlay.addWidget(self._btn_export, 0, Qt.AlignmentFlag.AlignTop)

        outer.addWidget(header)

        # ── Cards row
        ch = QWidget()
        clay = QHBoxLayout(ch); clay.setContentsMargins(0, 0, 0, 0); clay.setSpacing(10)
        self._card_acc   = _MetricCard("ACCURACY")
        self._card_f1    = _MetricCard("F1  (MACRO)")
        self._card_prec  = _MetricCard("PRECISION")
        self._card_rec   = _MetricCard("RECALL")
        self._card_spec  = _MetricCard("SPECIFICITY")
        self._card_extra = _MetricCard("AUROC")
        for c in (self._card_acc, self._card_f1, self._card_prec,
                  self._card_rec, self._card_spec, self._card_extra):
            clay.addWidget(c)
        clay.addStretch(1)
        outer.addWidget(ch)

        # ── Detail tabs
        self._tabs = QTabWidget(); self._tabs.setDocumentMode(True)
        self._tabs.setStyleSheet(f"QTabWidget::pane {{ border-top: 1px solid {C_BORDER2}; }}")
        outer.addWidget(self._tabs, 1)

        self._confusion = _ConfusionView();   self._tabs.addTab(self._confusion, "Confusion matrix")
        self._per_class = _PerClassTable();   self._tabs.addTab(self._per_class, "Per-class")
        self._boxplot   = _BoxPlotView()
        self._box_idx   = self._tabs.addTab(self._boxplot, "Per-class box plot")
        self._tabs.setTabVisible(self._box_idx, False)
        self._sweep     = _ThresholdSweepView()
        self._sweep_idx = self._tabs.addTab(self._sweep, "Threshold sweep")
        self._tabs.setTabVisible(self._sweep_idx, False)
        self._features  = _FeatureBarView()
        self._feat_idx  = self._tabs.addTab(self._features, "Per-feature")
        self._tabs.setTabVisible(self._feat_idx, False)

        self._notes = QTextEdit(); self._notes.setReadOnly(True)
        self._notes.setStyleSheet(
            f"QTextEdit {{ background:{C_PANEL}; color:{C_TEXT};"
            f"  border:1px solid {C_BORDER2}; border-radius:7px;"
            f"  font-family:'Menlo','DejaVu Sans Mono',monospace;"
            f"  font-size:11px; padding:8px; }}"
        )
        self._tabs.addTab(self._notes, "Notes")

    # ──────────────────────────────────────────────────────────────────

    def clear(self) -> None:
        self._current = None
        self._title_lbl.setText("No evaluation run yet")
        self._sub_lbl.setText("Pick a source on the left, set options on the right, then press Run.")
        for c in (self._card_acc, self._card_f1, self._card_prec,
                  self._card_rec, self._card_spec, self._card_extra):
            c.set_value(float("nan"))
        self._confusion.set_matrix(None)
        self._per_class.set_rows([])
        self._boxplot.set_data([], {})
        self._sweep.set_data([])
        self._features.set_data({})
        self._notes.clear()
        self._clear_pills()
        self._tabs.setTabVisible(self._sweep_idx, False)
        self._tabs.setTabVisible(self._feat_idx, False)
        self._tabs.setTabVisible(self._box_idx, False)
        self._btn_export.setEnabled(False)

    def show_result(self, result: EvaluationResult) -> None:
        self._current = result
        self._btn_export.setEnabled(True)

        self._title_lbl.setText(result.title)
        self._sub_lbl.setText(self._format_subtitle(result))

        self._clear_pills()
        self._pills.addWidget(_Pill(result.kind.name.title()))
        self._pills.addWidget(_Pill(result.mode.name.replace("_", " ").title()))
        self._pills.addWidget(_Pill(f"n = {result.n_samples:,}"))
        if result.model_name:
            self._pills.addWidget(_Pill(f"model: {result.model_name}"))
        self._pills.addStretch(1)

        self._card_acc .set_value(result.accuracy)
        self._card_f1  .set_value(result.f1_macro,
                                   caption=f"weighted: {result.f1_weighted * 100:.1f}%"
                                           if not np.isnan(result.f1_weighted) else "")
        self._card_prec.set_value(result.precision_macro)
        self._card_rec .set_value(result.recall_macro)
        self._card_spec.set_value(result.specificity)

        if result.kind == RecordingKind.UNITY:
            self._card_extra.set_title("AUROC")
            self._card_extra.set_value(
                result.auroc if result.auroc is not None else float("nan"),
                caption=f"thresh: {result.chosen_threshold:.4g}"
                        if result.chosen_threshold is not None else "",
            )
        elif result.mean_confidence_correct is not None:
            self._card_extra.set_title("CONFIDENCE  (✓)")
            self._card_extra.set_value(
                result.mean_confidence_correct,
                caption=f"on miss: {result.mean_confidence_incorrect * 100:.1f}%"
                        if result.mean_confidence_incorrect is not None else "",
            )
        elif result.expected_calibration_error is not None:
            self._card_extra.set_title("ECE")
            self._card_extra.set_value(result.expected_calibration_error)
        else:
            self._card_extra.set_title("CLASSES")
            self._card_extra.set_value(
                len(result.per_class) if result.per_class else float("nan"),
                fmt="{:.0f}",
            )

        self._confusion.set_matrix(result.confusion)
        self._per_class.set_rows(result.per_class)

        # Box plot (per-class F1 across recordings) — only when ≥2 recordings
        # have per_class_f1 data to plot.
        per_rec = self._per_recording_rows(result)
        rec_with_f1 = [r for r in per_rec if r.get("per_class_f1")]
        if len(rec_with_f1) >= 2:
            label_names = self._label_name_lookup(result)
            self._boxplot.set_data(rec_with_f1, label_names)
            self._tabs.setTabVisible(self._box_idx, True)
        else:
            self._tabs.setTabVisible(self._box_idx, False)

        if result.threshold_sweep:
            self._sweep.set_data(result.threshold_sweep, chosen=result.chosen_threshold)
            self._tabs.setTabVisible(self._sweep_idx, True)
        else:
            self._tabs.setTabVisible(self._sweep_idx, False)

        if result.per_feature:
            self._features.set_data(result.per_feature)
            self._tabs.setTabVisible(self._feat_idx, True)
        else:
            self._tabs.setTabVisible(self._feat_idx, False)

        # Notes
        lines: List[str] = list(result.notes)
        for key in ("per_recording", "per_session"):
            rows = result.settings.get(key) or []
            if rows:
                lines.append(""); lines.append(f"── {key} ──")
                for r in rows:
                    n = int(r.get("n", 0))
                    acc = float(r.get("accuracy", 0.0)) * 100
                    sid = r.get("session_id", "?")
                    sub = r.get("subject_id", "")
                    lines.append(f"  {sub:>10}  {sid:<40}  n={n:>6,}  acc={acc:5.1f}%")
        self._notes.setPlainText("\n".join(lines) if lines else "(no notes)")

    # ── helpers used by show_result + export ───────────────────────────

    @staticmethod
    def _per_recording_rows(result: EvaluationResult) -> List[Dict[str, Any]]:
        for key in ("per_recording", "per_session"):
            rows = result.settings.get(key) or []
            if rows:
                return list(rows)
        return []

    @staticmethod
    def _label_name_lookup(result: EvaluationResult) -> Dict[int, str]:
        """Build a {label_id: display_name} map from the result's per-class rows."""
        return {int(c.label): str(c.name) for c in result.per_class}

    def _clear_pills(self) -> None:
        while self._pills.count():
            item = self._pills.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

    @staticmethod
    def _format_subtitle(result: EvaluationResult) -> str:
        s = result.settings; bits = []
        if "window_size_ms" in s:
            bits.append(f"window {s['window_size_ms']} ms / stride {s.get('window_stride_ms', '?')} ms")
        if s.get("truth_source"):
            bits.append(f"truth: {s['truth_source']}")
        if s.get("objective"):
            bits.append(f"objective: {s['objective']}")
        return "  ·  ".join(bits) if bits else f"Run at {result.created_at:%H:%M:%S}"

    # ──────────────────────────────────────────────────────────────────
    # Export
    # ──────────────────────────────────────────────────────────────────

    def _on_export_bundle(self) -> None:
        if self._current is None:
            return
        d = QFileDialog.getExistingDirectory(self, "Save evaluation bundle to folder",
                                              str(Path.home()))
        if not d:
            return
        target = Path(d) / self._suggested_dirname()
        try:
            target.mkdir(parents=True, exist_ok=True)
            self._export_all(target)
        except Exception as exc:
            log.exception("Bundle export failed")
            QMessageBox.critical(self, "Export failed", str(exc))
            return
        QMessageBox.information(self, "Export complete",
                                f"Saved evaluation bundle to:\n{target}")

    def _on_export_one(self, kind: str) -> None:
        if self._current is None:
            return
        if kind == "confusion_png":
            path, _ = QFileDialog.getSaveFileName(
                self, "Save confusion matrix PNG", "confusion_matrix.png", "PNG (*.png)")
            if path:
                self._save_confusion_png(Path(path))
        elif kind == "boxplot_png":
            path, _ = QFileDialog.getSaveFileName(
                self, "Save per-class F1 box plot PNG", "per_class_f1_box.png", "PNG (*.png)")
            if path:
                self._save_boxplot_png(Path(path))
        elif kind == "threshold_png":
            path, _ = QFileDialog.getSaveFileName(
                self, "Save threshold sweep PNG", "threshold_sweep.png", "PNG (*.png)")
            if path:
                self._save_threshold_sweep_png(Path(path))
        elif kind == "feature_bar_png":
            path, _ = QFileDialog.getSaveFileName(
                self, "Save per-feature accuracy PNG", "per_feature_accuracy.png", "PNG (*.png)")
            if path:
                self._save_feature_bar_png(Path(path))
        elif kind == "roc_png":
            path, _ = QFileDialog.getSaveFileName(
                self, "Save ROC curve PNG", "roc_curve.png", "PNG (*.png)")
            if path:
                self._save_roc_png(Path(path))
        elif kind == "calibration_png":
            path, _ = QFileDialog.getSaveFileName(
                self, "Save calibration plot PNG", "calibration.png", "PNG (*.png)")
            if path:
                self._save_calibration_png(Path(path))
        elif kind == "scatter_png":
            path, _ = QFileDialog.getSaveFileName(
                self, "Save per-recording scatter PNG", "per_recording_scatter.png", "PNG (*.png)")
            if path:
                self._save_per_recording_scatter_png(Path(path))
        elif kind == "metrics_csv":
            path, _ = QFileDialog.getSaveFileName(
                self, "Save per-class metrics CSV", "per_class_metrics.csv", "CSV (*.csv)")
            if path:
                self._save_per_class_csv(Path(path))
        elif kind == "result_json":
            path, _ = QFileDialog.getSaveFileName(
                self, "Save result JSON", "result.json", "JSON (*.json)")
            if path:
                self._save_result_json(Path(path))

    def _suggested_dirname(self) -> str:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        kind = self._current.kind.value if self._current else "eval"
        return f"eval_{ts}_{kind}"

    def _export_all(self, target: Path) -> None:
        """Write every artifact to ``target`` (must already exist)."""
        self._save_result_json(target / "result.json")
        self._save_per_class_csv(target / "per_class_metrics.csv")
        self._save_per_recording_csv(target / "per_recording.csv")
        self._save_confusion_png(target / "confusion_matrix.png")
        self._save_boxplot_png(target / "per_class_f1_box.png")
        self._save_threshold_sweep_csv(target / "threshold_sweep.csv")
        self._save_threshold_sweep_png(target / "threshold_sweep.png")
        self._save_feature_csv(target / "per_feature_accuracy.csv")
        self._save_feature_bar_png(target / "per_feature_accuracy.png")
        self._save_roc_png(target / "roc_curve.png")
        self._save_calibration_png(target / "calibration.png")
        self._save_per_recording_scatter_png(target / "per_recording_scatter.png")

    # ── individual export operations (each is a no-op if the result
    # doesn't have data of that kind) ─────────────────────────────────

    def _save_result_json(self, path: Path) -> None:
        if self._current is None:
            return
        with open(path, "w") as f:
            json.dump(self._current.to_dict(), f, indent=2, default=str)

    def _save_per_class_csv(self, path: Path) -> None:
        r = self._current
        if r is None or not r.per_class:
            return
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["label_id", "class", "precision", "recall", "f1", "support"])
            for c in r.per_class:
                w.writerow([c.label, c.name,
                            f"{c.precision:.6f}", f"{c.recall:.6f}",
                            f"{c.f1:.6f}", c.support])

    def _save_per_recording_csv(self, path: Path) -> None:
        r = self._current
        if r is None:
            return
        rows = self._per_recording_rows(r)
        if not rows:
            return
        # Build a stable column set: all label ids that appear anywhere
        all_labels = sorted({int(k) for row in rows for k in (row.get("per_class_f1") or {})})
        names = self._label_name_lookup(r)
        header = ["subject_id", "session_id", "n", "accuracy"]
        header += [f"f1_{names.get(l, f'class_{l}')}" for l in all_labels]
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            for row in rows:
                pcf1 = row.get("per_class_f1") or {}
                # support both int and str keys (json round-trip)
                pcf1_int = {int(k): float(v) for k, v in pcf1.items()}
                base = [row.get("subject_id", ""), row.get("session_id", ""),
                        row.get("n", 0), f"{float(row.get('accuracy', 0.0)):.6f}"]
                base += [f"{pcf1_int.get(l, float('nan')):.6f}" for l in all_labels]
                w.writerow(base)

    def _save_confusion_png(self, path: Path) -> None:
        r = self._current
        if r is None or r.confusion is None:
            return
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as exc:
            log.warning("matplotlib unavailable — confusion PNG skipped: %s", exc)
            return
        labels = list(r.confusion.label_names)
        m = np.asarray(r.confusion.matrix, dtype=np.int64)
        # Row-normalise for shading; print raw counts as the cell text.
        row_sums = m.sum(axis=1, keepdims=True); row_sums[row_sums == 0] = 1
        norm = m.astype(np.float64) / row_sums

        fig, ax = plt.subplots(figsize=(max(5, 0.9 * len(labels) + 2),
                                         max(4, 0.7 * len(labels) + 2)),
                               dpi=140)
        im = ax.imshow(norm, vmin=0, vmax=1, cmap="Purples")
        ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(r.title, fontsize=11, pad=12)
        for i in range(len(labels)):
            for j in range(len(labels)):
                txt_color = "white" if norm[i, j] > 0.55 else "#1f2937"
                ax.text(j, i, f"{int(m[i, j])}\n{norm[i, j]*100:.1f}%",
                        ha="center", va="center", fontsize=8, color=txt_color)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("share of true class", fontsize=9)
        fig.tight_layout()
        fig.savefig(path, bbox_inches="tight"); plt.close(fig)

    def _save_boxplot_png(self, path: Path) -> None:
        r = self._current
        if r is None:
            return
        rows = [row for row in self._per_recording_rows(r) if row.get("per_class_f1")]
        if len(rows) < 2:
            return
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as exc:
            log.warning("matplotlib unavailable — boxplot PNG skipped: %s", exc)
            return
        names = self._label_name_lookup(r)
        buckets: Dict[int, List[float]] = {}
        for row in rows:
            pcf1 = row.get("per_class_f1") or {}
            for k, v in pcf1.items():
                buckets.setdefault(int(k), []).append(float(v))
        labels = sorted(buckets.keys())
        data   = [buckets[l] for l in labels]
        ticks  = [names.get(int(l), f"class_{int(l)}") for l in labels]

        fig, ax = plt.subplots(figsize=(max(5, 1.2 * len(labels) + 2), 4.5), dpi=140)
        bp = ax.boxplot(data, patch_artist=True, showmeans=False, widths=0.55)
        for patch in bp["boxes"]:
            patch.set_facecolor("#eef2ff"); patch.set_edgecolor("#6d28d9"); patch.set_linewidth(2)
        for whisker in bp["whiskers"]:
            whisker.set_color("#6d28d9"); whisker.set_linewidth(1.5)
        for cap in bp["caps"]:
            cap.set_color("#6d28d9")
        for median in bp["medians"]:
            median.set_color("#0284c7"); median.set_linewidth(2.5)
        # Jittered scatter on top
        rng = np.random.RandomState(0)
        for i, vals in enumerate(data):
            x = np.full(len(vals), i + 1) + (rng.rand(len(vals)) - 0.5) * 0.18
            ax.scatter(x, vals, color="#0284c7", alpha=0.7, s=22, zorder=3,
                       edgecolor="white", linewidth=0.5)
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(ticks, rotation=20, ha="right")
        ax.set_ylim(0, 1)
        ax.set_ylabel("F1")
        ax.set_title(f"Per-class F1 across {len(rows)} recordings", fontsize=11, pad=12)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(path, bbox_inches="tight"); plt.close(fig)

    def _save_threshold_sweep_csv(self, path: Path) -> None:
        r = self._current
        if r is None or not r.threshold_sweep:
            return
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["threshold", "accuracy", "precision", "recall",
                        "specificity", "f1", "tp", "fp", "tn", "fn"])
            for p in r.threshold_sweep:
                w.writerow([f"{p.threshold:.8g}", f"{p.accuracy:.6f}",
                            f"{p.precision:.6f}", f"{p.recall:.6f}",
                            f"{p.specificity:.6f}", f"{p.f1:.6f}",
                            p.tp, p.fp, p.tn, p.fn])

    def _save_feature_csv(self, path: Path) -> None:
        r = self._current
        if r is None or not r.per_feature:
            return
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["feature", "validation_accuracy"])
            for k in sorted(r.per_feature, key=lambda k: -r.per_feature[k]):
                w.writerow([k, f"{r.per_feature[k]:.6f}"])

    def _save_threshold_sweep_png(self, path: Path) -> None:
        """Line chart of F1 / precision / recall / specificity vs threshold."""
        r = self._current
        if r is None or not r.threshold_sweep:
            return
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as exc:
            log.warning("matplotlib unavailable — threshold sweep PNG skipped: %s", exc)
            return
        sweep = r.threshold_sweep
        x  = np.asarray([p.threshold   for p in sweep])
        f1 = np.asarray([p.f1          for p in sweep])
        pr = np.asarray([p.precision   for p in sweep])
        rc = np.asarray([p.recall      for p in sweep])
        sp = np.asarray([p.specificity for p in sweep])

        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=140)
        ax.plot(x, f1, label="F1",          color="#6d28d9", linewidth=2.5)
        ax.plot(x, pr, label="Precision",   color="#0284c7", linewidth=1.8)
        ax.plot(x, rc, label="Recall",      color="#16a34a", linewidth=1.8)
        ax.plot(x, sp, label="Specificity", color="#d97706", linewidth=1.8)

        chosen = r.chosen_threshold
        if chosen is not None and not np.isnan(chosen):
            ax.axvline(chosen, color="#dc2626", linewidth=1.5, linestyle="--",
                       label=f"chosen = {chosen:.4g}")

        # Mark the best-F1 threshold
        best_idx = int(np.argmax(f1))
        ax.scatter([x[best_idx]], [f1[best_idx]], color="#6d28d9", s=60, zorder=5)

        ax.set_xlim(x[0], x[-1])
        ax.set_ylim(0, 1)
        ax.set_xlabel("RMS threshold")
        ax.set_ylabel("Metric value")
        ax.set_title(f"Threshold sweep — {r.title}", fontsize=11, pad=12)
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)

    def _save_feature_bar_png(self, path: Path) -> None:
        """Horizontal bar chart: per-feature validation accuracy, sorted best-first."""
        r = self._current
        if r is None or not r.per_feature:
            return
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as exc:
            log.warning("matplotlib unavailable — feature bar PNG skipped: %s", exc)
            return

        items = sorted(r.per_feature.items(), key=lambda kv: kv[1])
        names = [k for k, _ in items]
        vals  = [float(v) if v == v else 0.0 for _, v in items]

        fig, ax = plt.subplots(figsize=(7, max(3, 0.55 * len(names) + 1.5)), dpi=140)
        colors = ["#16a34a" if v >= 0.8 else "#d97706" if v >= 0.6 else "#dc2626"
                  for v in vals]
        bars = ax.barh(names, vals, color=colors, edgecolor="#6d28d9", linewidth=0.8, height=0.65)
        for bar, val in zip(bars, vals):
            ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{val * 100:.1f}%", va="center", fontsize=9, color="#1f2937")

        ax.set_xlim(0, 1.12)
        ax.axvline(0.8, color="#16a34a", linewidth=1, linestyle=":", alpha=0.6)
        ax.axvline(0.6, color="#d97706", linewidth=1, linestyle=":", alpha=0.6)
        ax.set_xlabel("Validation accuracy (LDA, single feature)")
        ax.set_title(f"Per-feature accuracy — {r.title}", fontsize=11, pad=12)
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)

    def _save_roc_png(self, path: Path) -> None:
        """ROC curve for binary / threshold-sweep results (Unity mode).

        For multi-class results we draw one micro-averaged curve using the
        per-class confusion counts reconstructed from the threshold sweep or,
        if no sweep is available, we fall back to a note that the data is
        insufficient.
        """
        r = self._current
        if r is None:
            return
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as exc:
            log.warning("matplotlib unavailable — ROC PNG skipped: %s", exc)
            return

        if r.threshold_sweep:
            # Unity / binary: derive FPR & TPR from the sweep
            sweep = r.threshold_sweep
            fpr = np.asarray([1.0 - p.specificity for p in sweep])
            tpr = np.asarray([p.recall           for p in sweep])
            # Sort by ascending FPR so the curve draws left-to-right
            order = np.argsort(fpr)
            fpr, tpr = fpr[order], tpr[order]

            # AUC via trapezoidal rule
            auc = float(np.trapz(tpr, fpr))

            fig, ax = plt.subplots(figsize=(5.5, 5), dpi=140)
            ax.plot(fpr, tpr, color="#6d28d9", linewidth=2.5,
                    label=f"ROC (AUC = {auc:.3f})")
            # Mark the chosen threshold
            chosen = r.chosen_threshold
            if chosen is not None and not np.isnan(chosen):
                thresholds = np.asarray([p.threshold for p in sweep])
                idx = int(np.argmin(np.abs(thresholds - chosen)))
                ci = order.tolist().index(idx) if idx in order else -1
                if 0 <= ci < len(fpr):
                    ax.scatter([fpr[ci]], [tpr[ci]], color="#dc2626", s=80, zorder=5,
                               label=f"chosen thr = {chosen:.4g}")
        elif r.auroc is not None and not np.isnan(r.auroc):
            # We have a pre-computed AUROC but no sweep — draw a placeholder
            auc = float(r.auroc)
            fig, ax = plt.subplots(figsize=(5.5, 5), dpi=140)
            ax.text(0.5, 0.5, f"AUROC = {auc:.3f}\n(no per-point sweep data)",
                    ha="center", va="center", fontsize=13, color="#6d28d9",
                    transform=ax.transAxes)
        else:
            return  # nothing to draw

        ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.4, label="random")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("False positive rate  (1 − specificity)")
        ax.set_ylabel("True positive rate  (recall / sensitivity)")
        ax.set_title(f"ROC curve — {r.title}", fontsize=11, pad=12)
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)

    def _save_calibration_png(self, path: Path) -> None:
        """Reliability / calibration diagram.

        We use ``mean_confidence_correct`` and ``mean_confidence_incorrect``
        if available, together with overall accuracy, to draw a two-point
        calibration sketch.  Full calibration curves (binned confidence
        histograms) require raw prediction probabilities that aren't stored
        in ``EvaluationResult``; this lightweight version is still informative.
        """
        r = self._current
        if r is None:
            return
        mcc = r.mean_confidence_correct
        mci = r.mean_confidence_incorrect
        if mcc is None or mci is None:
            return  # no confidence data in this result
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as exc:
            log.warning("matplotlib unavailable — calibration PNG skipped: %s", exc)
            return

        fig, axes = plt.subplots(1, 2, figsize=(9, 4), dpi=140)

        # Left: bar chart of mean confidence on correct vs incorrect predictions
        ax = axes[0]
        bars = ax.bar(["Correct", "Incorrect"], [mcc, mci],
                      color=["#16a34a", "#dc2626"], edgecolor="#1f2937",
                      linewidth=0.8, width=0.5)
        for bar, val in zip(bars, [mcc, mci]):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                    f"{val * 100:.1f}%", ha="center", va="bottom",
                    fontsize=10, fontweight="bold")
        ax.set_ylim(0, 1.12)
        ax.axhline(1.0, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.set_ylabel("Mean predicted confidence")
        ax.set_title("Confidence: correct vs incorrect", fontsize=10)
        ax.grid(axis="y", alpha=0.3)

        # Right: expected calibration sketch
        # Ideal calibration: confidence ≈ accuracy.
        ax2 = axes[1]
        acc = r.accuracy if not np.isnan(r.accuracy) else None

        points_x = [mci, mcc]
        points_y = [1.0 - (acc or 0.5), acc or 0.5]
        ax2.scatter(points_x, points_y, s=100, zorder=5,
                    color=["#dc2626", "#16a34a"],
                    label=["incorrect", "correct"])
        ax2.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.4, label="perfect calibration")
        ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)
        ax2.set_xlabel("Mean predicted confidence")
        ax2.set_ylabel("Observed fraction correct")
        ax2.set_title("Calibration sketch", fontsize=10)
        ax2.legend(fontsize=8, loc="upper left")
        ax2.grid(alpha=0.3)
        # Annotate ECE if available
        ece = r.expected_calibration_error
        if ece is not None and not np.isnan(ece):
            ax2.text(0.98, 0.04, f"ECE = {ece:.4f}",
                     ha="right", va="bottom", fontsize=9, color="#6d28d9",
                     transform=ax2.transAxes)

        fig.suptitle(f"Calibration — {r.title}", fontsize=11, y=1.02)
        fig.tight_layout()
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)

    def _save_per_recording_scatter_png(self, path: Path) -> None:
        """Scatter plot: accuracy per recording, one dot per session/subject.

        X-axis  = recording index (sorted by subject then session).
        Y-axis  = accuracy on that recording.
        Coloured by subject.  Useful to spot subject-level clusters,
        drift over the course of a session series, or outlier recordings.
        """
        r = self._current
        if r is None:
            return
        rows = self._per_recording_rows(r)
        if len(rows) < 2:
            return
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib.cm import get_cmap
        except Exception as exc:
            log.warning("matplotlib unavailable — scatter PNG skipped: %s", exc)
            return

        subjects = sorted({row.get("subject_id", "?") for row in rows})
        cmap     = get_cmap("tab10")
        subj_color = {s: cmap(i % 10) for i, s in enumerate(subjects)}

        accs   = [float(row.get("accuracy", 0.0)) for row in rows]
        colors = [subj_color[row.get("subject_id", "?")] for row in rows]
        labels = [
            f"{row.get('subject_id', '')} · {row.get('session_id', '')}"
            for row in rows
        ]

        fig, ax = plt.subplots(figsize=(max(7, 0.35 * len(rows) + 3), 4.5), dpi=140)
        xs = list(range(len(rows)))
        ax.scatter(xs, accs, c=colors, s=55, zorder=4, edgecolors="white", linewidth=0.5)
        ax.plot(xs, accs, color="#cbd5e1", linewidth=1, zorder=3)  # thin connecting line

        # Mean line
        mean_acc = float(np.mean(accs))
        ax.axhline(mean_acc, color="#6d28d9", linewidth=1.5, linestyle="--",
                   label=f"mean = {mean_acc * 100:.1f}%")

        # Legend: one entry per subject
        from matplotlib.lines import Line2D
        handles = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor=subj_color[s],
                   markersize=8, label=s)
            for s in subjects
        ]
        handles.append(
            Line2D([0], [0], color="#6d28d9", linewidth=1.5, linestyle="--",
                   label=f"mean = {mean_acc * 100:.1f}%")
        )
        ax.legend(handles=handles, fontsize=8, loc="lower right",
                  ncol=max(1, len(subjects) // 6))

        ax.set_xticks(xs)
        ax.set_xticklabels(
            [row.get("session_id", str(i)) for i, row in enumerate(rows)],
            rotation=55, ha="right", fontsize=7
        )
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Accuracy")
        ax.set_title(f"Per-recording accuracy — {r.title}", fontsize=11, pad=12)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# CV mode — data subsets  (named groups the user can compare across)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class DataSubset:
    """
    A named subset of the available training data.

    The validation runner consumes this as a :class:`DataSelection` —
    see :meth:`to_data_selection`. We keep our own dataclass instead of
    using :class:`DataSelection` directly because we want richer metadata
    (a human label, optional notes) without polluting the runner's
    config schema.
    """
    name: str
    subjects: List[str] = field(default_factory=list)   # explicit list; empty ⇒ "all"
    domains:  List[str] = field(default_factory=list)   # ["pipeline"] / ["unity"] / [] for both
    notes:    str       = ""

    def to_data_selection(self) -> DataSelection:
        return DataSelection(
            subjects=list(self.subjects) if self.subjects else None,
            domains=list(self.domains)   if self.domains  else None,
        )

    def describe(self) -> str:
        bits: List[str] = []
        bits.append(f"{len(self.subjects)} subjects" if self.subjects else "all subjects")
        bits.append(" + ".join(self.domains) if self.domains else "all domains")
        return "  ·  ".join(bits)


class _SubsetEditorDialog(QDialog):
    """Modal dialog editing one :class:`DataSubset`. Returns it via accept()."""

    def __init__(self, available_subjects: List[str],
                 subset: Optional[DataSubset] = None,
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Edit data subset")
        self.setMinimumWidth(420)
        self._subset = subset or DataSubset(name="new subset")

        outer = QVBoxLayout(self); outer.setSpacing(10)

        # Name + notes
        form = _form()
        self._name_edit = QLineEdit(self._subset.name)
        form.addRow("Name:", self._name_edit)
        self._notes_edit = QLineEdit(self._subset.notes)
        self._notes_edit.setPlaceholderText("Optional one-line description")
        form.addRow("Notes:", self._notes_edit)
        outer.addLayout(form)

        # Domains
        dom_group = _styled_group("Domains")
        dl = QVBoxLayout()
        self._dom_pipeline = QCheckBox("Training sessions  (pipeline)")
        self._dom_unity    = QCheckBox("Unity sessions  (unity)")
        self._dom_pipeline.setChecked("pipeline" in self._subset.domains
                                       or not self._subset.domains)
        self._dom_unity   .setChecked("unity" in self._subset.domains
                                       or not self._subset.domains)
        dl.addWidget(self._dom_pipeline); dl.addWidget(self._dom_unity)
        hint = QLabel("Both checked  ⇒  use everything available.")
        hint.setStyleSheet(f"color:{C_MUTED}; font-size:10px;")
        dl.addWidget(hint)
        dom_group.setLayout(dl)
        outer.addWidget(dom_group)

        # Subjects (multi-select list)
        sg = _styled_group("Subjects  (none checked  ⇒  use all)")
        sl = QVBoxLayout()
        self._subj_list = QListWidget()
        self._subj_list.setSelectionMode(QListWidget.SelectionMode.NoSelection)
        for s in available_subjects:
            it = QListWidgetItem(s)
            it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            it.setCheckState(Qt.CheckState.Checked
                             if s in self._subset.subjects
                             else Qt.CheckState.Unchecked)
            self._subj_list.addItem(it)
        sl.addWidget(self._subj_list)
        # Quick toggles
        toggles = QHBoxLayout()
        b_all  = _ghost_button("Check all")
        b_none = _ghost_button("Uncheck all")
        b_all .clicked.connect(lambda: self._toggle_all(True))
        b_none.clicked.connect(lambda: self._toggle_all(False))
        toggles.addWidget(b_all); toggles.addWidget(b_none); toggles.addStretch(1)
        sl.addLayout(toggles)
        sg.setLayout(sl)
        outer.addWidget(sg, 1)

        # OK / Cancel
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                              QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self._accept_if_valid)
        bb.rejected.connect(self.reject)
        outer.addWidget(bb)

    def _toggle_all(self, on: bool) -> None:
        state = Qt.CheckState.Checked if on else Qt.CheckState.Unchecked
        for i in range(self._subj_list.count()):
            self._subj_list.item(i).setCheckState(state)

    def _accept_if_valid(self) -> None:
        if not self._name_edit.text().strip():
            QMessageBox.warning(self, "Name required",
                                "Each subset needs a name so it can be told apart "
                                "in the result table.")
            return
        self.accept()

    def result_subset(self) -> DataSubset:
        # Read back into a fresh DataSubset so the caller doesn't share
        # references with the dialog's internal state.
        subjects = [
            self._subj_list.item(i).text()
            for i in range(self._subj_list.count())
            if self._subj_list.item(i).checkState() == Qt.CheckState.Checked
        ]
        domains: List[str] = []
        if self._dom_pipeline.isChecked(): domains.append("pipeline")
        if self._dom_unity   .isChecked(): domains.append("unity")
        # A subset with both domains and no subject filter is just "all data" —
        # storing an empty domains list in that case keeps to_data_selection()
        # happy (it emits None, which the runner treats as "no filter").
        if len(domains) == 2:
            domains = []
        return DataSubset(
            name=self._name_edit.text().strip(),
            subjects=subjects,
            domains=domains,
            notes=self._notes_edit.text().strip(),
        )


class _DataSubsetListWidget(QWidget):
    """
    Editable list of named :class:`DataSubset` rows.

    Used in two places:
      - as the "default subset" when the comparison axis is NOT data,
      - as the list of values when the comparison axis IS data.
    """

    changed = Signal()

    def __init__(self, available_subjects: List[str],
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._available_subjects = list(available_subjects)
        self._subsets: List[DataSubset] = []

        outer = QVBoxLayout(self); outer.setContentsMargins(0, 0, 0, 0); outer.setSpacing(6)

        self._list = QListWidget()
        self._list.setStyleSheet(
            f"QListWidget {{ background:{C_PANEL}; border:1px solid {C_BORDER2};"
            f"  border-radius:6px; padding:4px; }}"
            f"QListWidget::item {{ padding:6px 8px; border-radius:4px; }}"
            f"QListWidget::item:selected {{ background:{C_CARD}; color:{C_TEXT}; }}"
        )
        outer.addWidget(self._list, 1)

        bar = QHBoxLayout(); bar.setSpacing(6)
        self._btn_add  = _ghost_button("+ Add…")
        self._btn_edit = _ghost_button("Edit…")
        self._btn_dup  = _ghost_button("Duplicate")
        self._btn_del  = _ghost_button("Delete")
        for b in (self._btn_add, self._btn_edit, self._btn_dup, self._btn_del):
            bar.addWidget(b)
        bar.addStretch(1)
        outer.addLayout(bar)

        self._btn_add .clicked.connect(self._on_add)
        self._btn_edit.clicked.connect(self._on_edit)
        self._btn_dup .clicked.connect(self._on_duplicate)
        self._btn_del .clicked.connect(self._on_delete)
        self._list.itemDoubleClicked.connect(lambda _it: self._on_edit())

    # ── public API ────────────────────────────────────────────────────

    def set_available_subjects(self, subjects: List[str]) -> None:
        self._available_subjects = list(subjects)

    def subsets(self) -> List[DataSubset]:
        return list(self._subsets)

    def set_subsets(self, subsets: List[DataSubset]) -> None:
        self._subsets = [
            DataSubset(s.name, list(s.subjects), list(s.domains), s.notes)
            for s in subsets
        ]
        self._refresh()

    def add_subset(self, subset: DataSubset) -> None:
        self._subsets.append(subset)
        self._refresh()
        self.changed.emit()

    # ── internal slots ────────────────────────────────────────────────

    def _on_add(self) -> None:
        dlg = _SubsetEditorDialog(self._available_subjects, parent=self)
        if dlg.exec():
            self.add_subset(dlg.result_subset())

    def _on_edit(self) -> None:
        idx = self._list.currentRow()
        if idx < 0 or idx >= len(self._subsets):
            return
        dlg = _SubsetEditorDialog(self._available_subjects,
                                   subset=self._subsets[idx], parent=self)
        if dlg.exec():
            self._subsets[idx] = dlg.result_subset()
            self._refresh()
            self.changed.emit()

    def _on_duplicate(self) -> None:
        idx = self._list.currentRow()
        if idx < 0 or idx >= len(self._subsets):
            return
        s = self._subsets[idx]
        clone = DataSubset(name=f"{s.name} (copy)",
                           subjects=list(s.subjects),
                           domains=list(s.domains), notes=s.notes)
        self._subsets.insert(idx + 1, clone)
        self._refresh(); self.changed.emit()

    def _on_delete(self) -> None:
        idx = self._list.currentRow()
        if idx < 0 or idx >= len(self._subsets):
            return
        del self._subsets[idx]
        self._refresh(); self.changed.emit()

    def _refresh(self) -> None:
        self._list.clear()
        for s in self._subsets:
            text = f"{s.name}\n  {s.describe()}"
            if s.notes:
                text += f"  ·  {s.notes}"
            it = QListWidgetItem(text)
            self._list.addItem(it)


class _CheckList(QWidget):
    """A vertical list of checkboxes plus Select-all / Clear quick toggles."""

    selection_changed = Signal()

    def __init__(self,
                 entries: List[Tuple[str, str]],
                 *,
                 initial_checked: Optional[List[str]] = None,
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._boxes: Dict[str, QCheckBox] = {}
        initial_checked = initial_checked or []

        outer = QVBoxLayout(self); outer.setContentsMargins(0, 0, 0, 0); outer.setSpacing(6)

        # Quick toggles
        bar = QHBoxLayout(); bar.setSpacing(6)
        b_all  = _ghost_button("Select all")
        b_none = _ghost_button("Clear")
        b_all .clicked.connect(self.select_all)
        b_none.clicked.connect(self.select_none)
        bar.addWidget(b_all); bar.addWidget(b_none); bar.addStretch(1)
        outer.addLayout(bar)

        # The boxes themselves
        for key, label in entries:
            row = QHBoxLayout(); row.setContentsMargins(0, 0, 0, 0); row.setSpacing(6)
            cb = QCheckBox(key); cb.setChecked(key in initial_checked)
            cb.toggled.connect(self._on_toggled)
            self._boxes[key] = cb
            row.addWidget(cb)
            desc = QLabel(label)
            desc.setStyleSheet(f"color:{C_MUTED}; font-size:10px;")
            row.addWidget(desc, 1)
            outer.addLayout(row)
        outer.addStretch(1)

    def selected(self) -> List[str]:
        return [k for k, cb in self._boxes.items() if cb.isChecked()]

    def set_selected(self, keys: List[str]) -> None:
        for k, cb in self._boxes.items():
            cb.setChecked(k in keys)

    def select_all(self) -> None:
        for cb in self._boxes.values():
            cb.setChecked(True)

    def select_none(self) -> None:
        for cb in self._boxes.values():
            cb.setChecked(False)

    def _on_toggled(self, _checked: bool) -> None:
        self.selection_changed.emit()


# ═══════════════════════════════════════════════════════════════════════════
# CV mode — sweep planner  (translates UI state into a list of ExperimentConfigs)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SweepPlan:
    """One planned validation run, plus a human label for the result table."""
    label: str               # e.g. "model=catboost", "subset=early VPs"
    cfg:   ExperimentConfig

    def short_axis_value(self) -> str:
        """The varied-axis value alone, e.g. 'catboost' (without 'model=')."""
        if "=" in self.label:
            return self.label.split("=", 1)[1]
        return self.label


@dataclass
class SweepDefaults:
    """Per-axis defaults used to fill in the values that aren't being varied."""
    name:             str
    seed:             int
    window_ms:        int
    stride_ms:        int
    drop_rest:        bool
    cv_strategy:      str
    cv_kwargs:        Dict[str, Any]
    default_subset:   DataSubset
    default_models:   List[str]
    default_features: List[str]


def plan_sweep(
    *,
    axis: str,                         # "none" | "model" | "features" | "data_subset"
    defaults: SweepDefaults,
    # Per-axis values (only the matching axis is consulted)
    axis_models:   Optional[List[List[str]]]   = None,   # one list of models per run
    axis_features: Optional[List[List[str]]]   = None,   # one feature set per run
    axis_subsets:  Optional[List[DataSubset]]  = None,
    axis_labels:   Optional[List[str]]         = None,
) -> List[SweepPlan]:
    """
    Build the list of :class:`SweepPlan` to execute.

    The CV runner takes a list of models and a list of features per
    config, but we run **one varied value at a time** (separate runs
    per dimension). So when ``axis == "model"``, every entry in
    ``axis_models`` becomes its own ExperimentConfig that uses the
    default features and the default subset.
    """
    plans: List[SweepPlan] = []

    def _make(label: str, models: List[str], features: List[str],
              subset: DataSubset) -> SweepPlan:
        cfg = ExperimentConfig(
            name=f"{defaults.name}__{label.replace(' ', '_').replace('=', '_')}",
            description=f"CV sweep: {label}",
            seed=defaults.seed,
            data=subset.to_data_selection(),
            windowing=WindowingConfig(
                window_ms=defaults.window_ms,
                stride_ms=defaults.stride_ms,
                drop_rest=defaults.drop_rest,
            ),
            features=[FeatureConfig(name=n) for n in features],
            models=[ModelConfig(type=m) for m in models],
            cv=CVConfig(strategy=defaults.cv_strategy,
                        kwargs=dict(defaults.cv_kwargs)),
        )
        return SweepPlan(label=label, cfg=cfg)

    if axis == "none":
        plans.append(_make(
            label="single run",
            models=defaults.default_models,
            features=defaults.default_features,
            subset=defaults.default_subset,
        ))
        return plans

    if axis == "model":
        if not axis_models:
            return []
        labels = axis_labels or [
            ", ".join(m) if len(m) > 1 else (m[0] if m else "(empty)")
            for m in axis_models
        ]
        for label, models in zip(labels, axis_models):
            plans.append(_make(
                label=f"model={label}",
                models=models,
                features=defaults.default_features,
                subset=defaults.default_subset,
            ))
        return plans

    if axis == "features":
        if not axis_features:
            return []
        labels = axis_labels or [
            "+".join(fs) if fs else "(empty)" for fs in axis_features
        ]
        for label, fs in zip(labels, axis_features):
            plans.append(_make(
                label=f"features={label}",
                models=defaults.default_models,
                features=fs,
                subset=defaults.default_subset,
            ))
        return plans

    if axis == "data_subset":
        if not axis_subsets:
            return []
        for sub in axis_subsets:
            plans.append(_make(
                label=f"subset={sub.name}",
                models=defaults.default_models,
                features=defaults.default_features,
                subset=sub,
            ))
        return plans

    raise ValueError(f"Unknown sweep axis: {axis!r}")


# ═══════════════════════════════════════════════════════════════════════════
# CV mode — threaded sweep runner
# ═══════════════════════════════════════════════════════════════════════════

class _SweepProgressBridge(QObject):
    """
    Adapts the runner's ProgressReporter API onto Qt signals.

    The runner calls these methods from its worker thread; the signals
    are connected with QueuedConnection so the GUI updates land on the
    UI thread. The "sweep" semantics layer (run_index / total_runs) is
    set by the worker before each individual ValidationRunner.run call.
    """

    sweep_started   = Signal(int)                 # total_runs
    sweep_finished  = Signal(object)              # List[(label, RunResult)]
    sweep_failed    = Signal(str)                 # traceback string

    run_started     = Signal(int, int, str)       # run_idx, total_runs, label
    run_finished    = Signal(int, int, str, object)  # run_idx, total_runs, label, RunResult

    fold_started    = Signal(int, int, str, str)  # fold idx, total_folds, fold_id, model
    fold_finished   = Signal(int, int, object)    # fold idx, total_folds, FoldResult
    log_line        = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cancel = False
        self._current_run_idx = 0
        self._current_total   = 0
        self._current_label   = ""

    # ── used by the worker between runs ──────────────────────────────

    def begin_run(self, idx: int, total: int, label: str) -> None:
        self._current_run_idx = idx
        self._current_total   = total
        self._current_label   = label
        self.run_started.emit(idx, total, label)

    def request_cancel(self) -> None:
        self._cancel = True

    # ── ProgressReporter API (called from worker thread) ─────────────

    def on_run_start(self, total_folds, total_models, _records):
        self.log_line.emit(
            f"[{self._current_run_idx}/{self._current_total}] "
            f"{self._current_label}: {total_folds} folds × {total_models} models"
        )

    def on_fold_start(self, idx, total, fold_id, model_type):
        self.fold_started.emit(idx, total, fold_id, model_type)

    def on_fold_done(self, idx, total, fold_result):
        self.fold_finished.emit(idx, total, fold_result)

    def on_run_done(self, result):
        # The worker emits run_finished itself so it can attach the label,
        # but the runner does still call on_run_done — we just log it.
        self.log_line.emit(
            f"[{self._current_run_idx}/{self._current_total}] "
            f"{self._current_label}: done"
        )

    def log(self, message: str):
        self.log_line.emit(message)

    def should_cancel(self) -> bool:
        return self._cancel


class _SweepWorker(QThread):
    """
    Runs N ValidationRunner.run(cfg) calls sequentially on a worker
    thread. Cancellation between runs is honoured (we check before
    starting each new ExperimentConfig); cancellation mid-run is
    handled by the runner itself via ``progress.should_cancel()``.
    """

    def __init__(self, data_dir: Path, plans: List[SweepPlan],
                 bridge: _SweepProgressBridge, parent=None):
        super().__init__(parent)
        self._data_dir = Path(data_dir)
        self._plans    = list(plans)
        self._bridge   = bridge

    def run(self) -> None:
        results: List[Tuple[str, Any]] = []
        try:
            self._bridge.sweep_started.emit(len(self._plans))
            runner = ValidationRunner(self._data_dir)
            for i, plan in enumerate(self._plans, start=1):
                if self._bridge.should_cancel():
                    self._bridge.log_line.emit(
                        f"Sweep cancelled before run {i}/{len(self._plans)}.")
                    break
                self._bridge.begin_run(i, len(self._plans), plan.label)
                try:
                    rr = runner.run(plan.cfg, progress=self._bridge)
                except Exception as exc:
                    # Don't kill the entire sweep when one run blows up — log
                    # and continue. This matters when the user is comparing 6
                    # models and one of them fails to import (e.g. xgboost).
                    self._bridge.log_line.emit(
                        f"⚠ run {i}/{len(self._plans)} ({plan.label}) FAILED: {exc}"
                    )
                    log.exception("sweep run failed: %s", plan.label)
                    rr = None
                results.append((plan.label, rr))
                self._bridge.run_finished.emit(i, len(self._plans), plan.label, rr)
            self._bridge.sweep_finished.emit(results)
        except Exception:
            import traceback
            self._bridge.sweep_failed.emit(traceback.format_exc())


# ═══════════════════════════════════════════════════════════════════════════
# CV mode — axis-specific value pickers
# ═══════════════════════════════════════════════════════════════════════════

class _ModelAxisPanel(QWidget):
    """
    Editor for the "vary by model" axis.

    Lets the user pick which models to compare. Each checked model
    becomes one ExperimentConfig with the default features and subset.
    """

    changed = Signal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        outer = QVBoxLayout(self); outer.setContentsMargins(0, 0, 0, 0); outer.setSpacing(8)

        info = QLabel("Each checked model becomes one validation run, using the "
                      "default features and data subset configured on the left.")
        info.setStyleSheet(f"color:{C_MUTED}; font-size:11px;")
        info.setWordWrap(True)
        outer.addWidget(info)

        self._list = _CheckList(_AVAILABLE_MODELS,
                                 initial_checked=[m for m, _ in _AVAILABLE_MODELS[:3]])
        self._list.selection_changed.connect(self.changed.emit)
        outer.addWidget(self._list, 1)

    def axis_values(self) -> Tuple[List[List[str]], List[str]]:
        """Returns (models_per_run, labels) — one entry per run."""
        keys = self._list.selected()
        return [[k] for k in keys], list(keys)


class _FeaturesAxisPanel(QWidget):
    """Editor for the "vary by features" axis — manages a list of feature presets."""

    changed = Signal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        # Each row: (label, [feature keys])
        self._presets: List[Tuple[str, List[str]]] = list(_DEFAULT_FEATURE_PRESETS)

        outer = QVBoxLayout(self); outer.setContentsMargins(0, 0, 0, 0); outer.setSpacing(8)

        info = QLabel("Each row below becomes one validation run with that feature "
                      "set, using the default model and data subset configured "
                      "on the left.")
        info.setStyleSheet(f"color:{C_MUTED}; font-size:11px;")
        info.setWordWrap(True)
        outer.addWidget(info)

        self._list = QListWidget()
        self._list.setStyleSheet(
            f"QListWidget {{ background:{C_PANEL}; border:1px solid {C_BORDER2};"
            f"  border-radius:6px; padding:4px; }}"
            f"QListWidget::item {{ padding:6px 8px; border-radius:4px; }}"
            f"QListWidget::item:selected {{ background:{C_CARD}; color:{C_TEXT}; }}"
        )
        outer.addWidget(self._list, 1)

        bar = QHBoxLayout(); bar.setSpacing(6)
        b_add  = _ghost_button("+ Add…")
        b_edit = _ghost_button("Edit…")
        b_del  = _ghost_button("Delete")
        b_reset = _ghost_button("Reset presets")
        for b in (b_add, b_edit, b_del, b_reset):
            bar.addWidget(b)
        bar.addStretch(1)
        outer.addLayout(bar)

        b_add  .clicked.connect(self._on_add)
        b_edit .clicked.connect(self._on_edit)
        b_del  .clicked.connect(self._on_delete)
        b_reset.clicked.connect(self._on_reset)
        self._list.itemDoubleClicked.connect(lambda _it: self._on_edit())

        self._refresh()

    # ── public API ───────────────────────────────────────────────────

    def axis_values(self) -> Tuple[List[List[str]], List[str]]:
        feats = [list(p[1]) for p in self._presets]
        labels = [p[0] for p in self._presets]
        return feats, labels

    # ── internal slots ───────────────────────────────────────────────

    def _on_add(self) -> None:
        preset = self._prompt_for_preset()
        if preset is not None:
            self._presets.append(preset)
            self._refresh(); self.changed.emit()

    def _on_edit(self) -> None:
        idx = self._list.currentRow()
        if idx < 0 or idx >= len(self._presets):
            return
        preset = self._prompt_for_preset(initial=self._presets[idx])
        if preset is not None:
            self._presets[idx] = preset
            self._refresh(); self.changed.emit()

    def _on_delete(self) -> None:
        idx = self._list.currentRow()
        if idx < 0 or idx >= len(self._presets):
            return
        del self._presets[idx]
        self._refresh(); self.changed.emit()

    def _on_reset(self) -> None:
        if QMessageBox.question(
            self, "Reset feature presets",
            "Replace the current preset list with the default four?",
        ) == QMessageBox.StandardButton.Yes:
            self._presets = list(_DEFAULT_FEATURE_PRESETS)
            self._refresh(); self.changed.emit()

    def _refresh(self) -> None:
        self._list.clear()
        for label, feats in self._presets:
            text = f"{label}\n  {'+'.join(feats)}"
            self._list.addItem(QListWidgetItem(text))

    def _prompt_for_preset(
        self,
        initial: Optional[Tuple[str, List[str]]] = None,
    ) -> Optional[Tuple[str, List[str]]]:
        dlg = QDialog(self); dlg.setWindowTitle("Feature preset")
        dlg.setMinimumWidth(380)
        dl = QVBoxLayout(dlg)
        form = _form()
        name_edit = QLineEdit(initial[0] if initial else "")
        name_edit.setPlaceholderText("e.g. 'time-domain trio'")
        form.addRow("Preset name:", name_edit)
        dl.addLayout(form)

        feat_group = _styled_group("Features in this preset")
        fl = QVBoxLayout()
        check = _CheckList(_AVAILABLE_FEATURES,
                           initial_checked=initial[1] if initial else _DEFAULT_FEATURE_SET)
        fl.addWidget(check)
        feat_group.setLayout(fl)
        dl.addWidget(feat_group, 1)

        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                              QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        dl.addWidget(bb)

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return None
        name = name_edit.text().strip() or "(unnamed)"
        feats = check.selected()
        if not feats:
            QMessageBox.warning(self, "No features",
                                "Select at least one feature for this preset.")
            return None
        return (name, feats)


class _DataSubsetAxisPanel(QWidget):
    """Editor for the "vary by data subset" axis — wraps a _DataSubsetListWidget."""

    changed = Signal()

    def __init__(self, available_subjects: List[str],
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        outer = QVBoxLayout(self); outer.setContentsMargins(0, 0, 0, 0); outer.setSpacing(8)

        info = QLabel("Each named subset below becomes one validation run, "
                      "comparing how the same model + features perform on "
                      "different slices of your data.")
        info.setStyleSheet(f"color:{C_MUTED}; font-size:11px;")
        info.setWordWrap(True)
        outer.addWidget(info)

        self._list = _DataSubsetListWidget(available_subjects)
        self._list.changed.connect(self.changed.emit)
        outer.addWidget(self._list, 1)

    def set_available_subjects(self, subjects: List[str]) -> None:
        self._list.set_available_subjects(subjects)

    def axis_values(self) -> List[DataSubset]:
        return self._list.subsets()


# ═══════════════════════════════════════════════════════════════════════════
# CV mode — comparison + per-model tables
# ═══════════════════════════════════════════════════════════════════════════

class _ComparisonTable(QWidget):
    """
    One row per sweep run, columns: label, n folds, accuracy, F1, train time.

    When there are multiple models per run (the user picked a list under
    "vary by model") each row is collapsed to the best model. The full
    per-model breakdown lives in :class:`_PerModelTable`.
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        outer = QVBoxLayout(self); outer.setContentsMargins(0, 0, 0, 0)
        self._table = QTableWidget(0, 0)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setAlternatingRowColors(True)
        self._table.setStyleSheet(
            f"QTableWidget {{ background:{C_PANEL}; border:1px solid {C_BORDER2};"
            f"  border-radius:6px; alternate-background-color:{C_BG};"
            f"  gridline-color:{C_BORDER2}; }}"
            f"QTableWidget::item {{ padding:6px 9px; }}"
            f"QHeaderView::section {{ background:{C_BG}; border:none;"
            f"  padding:6px 8px; font-weight:600; color:{C_MUTED}; }}"
        )
        self._table.verticalHeader().setVisible(False)
        outer.addWidget(self._table)

    def set_rows(self, rows: List[Tuple[str, Dict[str, Any]]]) -> None:
        """
        Each ``rows`` entry is ``(axis_value, aggregate_dict)``, where
        ``aggregate_dict`` is the per-model summary returned by
        ``RunResult.aggregate()``. We pick the best model per row.
        """
        headers = ["Run", "Best model", "Folds",
                    "Accuracy", "Macro-F1", "Mean train time"]
        self._table.setColumnCount(len(headers))
        self._table.setHorizontalHeaderLabels(headers)
        self._table.setRowCount(len(rows))

        for i, (label, agg) in enumerate(rows):
            if not agg:
                self._table.setItem(i, 0, QTableWidgetItem(label))
                self._table.setItem(i, 1, QTableWidgetItem("(no result)"))
                for c in range(2, len(headers)):
                    self._table.setItem(i, c, QTableWidgetItem("—"))
                continue
            best_model, m = max(agg.items(),
                                 key=lambda kv: kv[1].get("accuracy_mean", 0.0))
            self._table.setItem(i, 0, QTableWidgetItem(label))
            self._table.setItem(i, 1, QTableWidgetItem(best_model))
            self._table.setItem(i, 2, QTableWidgetItem(str(int(m.get("n_folds", 0)))))
            acc_item = QTableWidgetItem(
                f"{m.get('accuracy_mean', 0.0):.3f} ± {m.get('accuracy_std', 0.0):.3f}"
            )
            acc_item.setForeground(QBrush(QColor(_score_colour(m.get("accuracy_mean", 0.0)))))
            self._table.setItem(i, 3, acc_item)
            self._table.setItem(i, 4, QTableWidgetItem(
                f"{m.get('macro_f1_mean', 0.0):.3f} ± {m.get('macro_f1_std', 0.0):.3f}"
            ))
            self._table.setItem(i, 5, QTableWidgetItem(_fmt_dur(m.get("train_seconds_mean", 0.0))))

        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)


class _PerModelTable(QWidget):
    """Per-model breakdown of every sweep run, so multi-model runs are visible."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        outer = QVBoxLayout(self); outer.setContentsMargins(0, 0, 0, 0)
        self._table = QTableWidget(0, 0)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setAlternatingRowColors(True)
        self._table.setStyleSheet(
            f"QTableWidget {{ background:{C_PANEL}; border:1px solid {C_BORDER2};"
            f"  border-radius:6px; alternate-background-color:{C_BG};"
            f"  gridline-color:{C_BORDER2}; }}"
            f"QTableWidget::item {{ padding:6px 9px; }}"
            f"QHeaderView::section {{ background:{C_BG}; border:none;"
            f"  padding:6px 8px; font-weight:600; color:{C_MUTED}; }}"
        )
        self._table.verticalHeader().setVisible(False)
        outer.addWidget(self._table)

    def set_rows(self, rows: List[Tuple[str, Dict[str, Any]]]) -> None:
        headers = ["Run", "Model", "Folds", "Accuracy", "Macro-F1", "Train time"]
        self._table.setColumnCount(len(headers))
        self._table.setHorizontalHeaderLabels(headers)

        flat: List[Tuple[str, str, Dict[str, float]]] = []
        for label, agg in rows:
            if not agg:
                flat.append((label, "(no result)", {}))
                continue
            for model, m in sorted(agg.items()):
                flat.append((label, model, m))

        self._table.setRowCount(len(flat))
        for i, (label, model, m) in enumerate(flat):
            self._table.setItem(i, 0, QTableWidgetItem(label))
            self._table.setItem(i, 1, QTableWidgetItem(model))
            if not m:
                for c in range(2, len(headers)):
                    self._table.setItem(i, c, QTableWidgetItem("—"))
                continue
            self._table.setItem(i, 2, QTableWidgetItem(str(int(m.get("n_folds", 0)))))
            acc_item = QTableWidgetItem(
                f"{m.get('accuracy_mean', 0.0):.3f} ± {m.get('accuracy_std', 0.0):.3f}"
            )
            acc_item.setForeground(QBrush(QColor(_score_colour(m.get("accuracy_mean", 0.0)))))
            self._table.setItem(i, 3, acc_item)
            self._table.setItem(i, 4, QTableWidgetItem(
                f"{m.get('macro_f1_mean', 0.0):.3f} ± {m.get('macro_f1_std', 0.0):.3f}"
            ))
            self._table.setItem(i, 5, QTableWidgetItem(_fmt_dur(m.get("train_seconds_mean", 0.0))))
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)


class _CrossValidationResultsView(QWidget):
    """
    Bottom panel for CV mode: live progress, summary table, per-model
    breakdown, and an export menu. Owns its own export button so the
    CV-specific export targets (comparison CSV / per-model CSV / sweep
    JSON / bundle) are clearly separated from the eval-result PNG/CSV
    targets used by :class:`_EvaluationResultView`.
    """

    export_requested = Signal(str)   # kind: "comparison_csv" / "per_model_csv" / "summary_json" / "bundle"

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        outer = QVBoxLayout(self); outer.setContentsMargins(0, 0, 0, 0); outer.setSpacing(10)

        # ── Header (title + export menu) ────────────────────────────
        header = QHBoxLayout(); header.setSpacing(10)
        title_col = QVBoxLayout(); title_col.setSpacing(2); title_col.setContentsMargins(0,0,0,0)
        self._title = QLabel("No sweep run yet")
        f = QFont(); f.setPointSize(15); f.setBold(True); self._title.setFont(f)
        self._title.setStyleSheet(f"color:{C_TEXT};")
        title_col.addWidget(self._title)
        self._sub = QLabel("Configure an axis on the left, then press Run sweep.")
        self._sub.setStyleSheet(f"color:{C_MUTED}; font-size:11px;")
        self._sub.setWordWrap(True)
        title_col.addWidget(self._sub)
        self._pills = QHBoxLayout()
        self._pills.setSpacing(6); self._pills.setContentsMargins(0, 4, 0, 0)
        ph = QWidget(); ph.setLayout(self._pills); title_col.addWidget(ph)
        tw = QWidget(); tw.setLayout(title_col)
        header.addWidget(tw, 1)

        self._btn_export = _ghost_button("⤓ Export…", accent=True)
        self._btn_export.setEnabled(False)
        menu = QMenu(self._btn_export)
        a_csv  = menu.addAction("Comparison CSV…")
        a_pmcsv = menu.addAction("Per-model CSV…")
        a_json = menu.addAction("Sweep JSON…")
        a_pkg  = menu.addAction("Save full bundle (folder)…")
        a_csv .triggered.connect(lambda: self.export_requested.emit("comparison_csv"))
        a_pmcsv.triggered.connect(lambda: self.export_requested.emit("per_model_csv"))
        a_json.triggered.connect(lambda: self.export_requested.emit("summary_json"))
        a_pkg .triggered.connect(lambda: self.export_requested.emit("bundle"))
        self._btn_export.setMenu(menu)
        self._btn_export.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        header.addWidget(self._btn_export, 0, Qt.AlignmentFlag.AlignTop)
        outer.addLayout(header)

        # ── Progress strip ───────────────────────────────────────────
        self._progress = QProgressBar()
        self._progress.setRange(0, 1)
        self._progress.setValue(0)
        self._progress.setTextVisible(True)
        self._progress.setStyleSheet(
            f"QProgressBar {{ border:1px solid {C_BORDER2}; border-radius:5px;"
            f"  background:{C_PANEL}; color:{C_TEXT}; padding:1px;"
            f"  text-align:center; font-size:11px; }}"
            f"QProgressBar::chunk {{ background:{C_ACCENT}; border-radius:4px; }}"
        )
        outer.addWidget(self._progress)

        # ── Tabs: Comparison / Per-model / Log ──────────────────────
        self._tabs = QTabWidget(); self._tabs.setDocumentMode(True)
        outer.addWidget(self._tabs, 1)

        self._comparison = _ComparisonTable()
        self._per_model  = _PerModelTable()
        self._tabs.addTab(self._comparison, "Comparison")
        self._tabs.addTab(self._per_model,  "Per-model")

        self._log = QTextEdit(); self._log.setReadOnly(True)
        self._log.setStyleSheet(
            f"QTextEdit {{ background:{C_PANEL}; color:{C_TEXT};"
            f"  border:1px solid {C_BORDER2}; border-radius:7px;"
            f"  font-family:'Menlo','DejaVu Sans Mono',monospace;"
            f"  font-size:11px; padding:8px; }}"
        )
        self._tabs.addTab(self._log, "Live log")

    # ── public API ──────────────────────────────────────────────────

    def reset_for_sweep(self, total_runs: int, axis_label: str) -> None:
        self._title.setText(f"Sweep over {total_runs} run(s)")
        self._sub.setText(f"Comparison axis: {axis_label}.")
        self._clear_pills()
        self._pills.addWidget(_Pill(f"axis: {axis_label}"))
        self._pills.addWidget(_Pill(f"{total_runs} run{'s' if total_runs != 1 else ''}"))
        self._pills.addStretch(1)
        self._progress.setRange(0, max(total_runs, 1))
        self._progress.setValue(0)
        self._progress.setFormat("preparing…")
        self._comparison.set_rows([])
        self._per_model.set_rows([])
        self._log.clear()
        self._btn_export.setEnabled(False)

    def on_run_started(self, idx: int, total: int, label: str) -> None:
        self._progress.setFormat(f"run {idx} / {total}  ·  {label}")

    def on_run_finished(self, idx: int, total: int, label: str, _result: Any) -> None:
        self._progress.setValue(idx)

    def append_log(self, line: str) -> None:
        self._log.append(line)
        bar = self._log.verticalScrollBar()
        bar.setValue(bar.maximum())

    def show_completed_sweep(self, rows: List[Tuple[str, Dict[str, Any]]]) -> None:
        self._comparison.set_rows(rows)
        self._per_model.set_rows(rows)
        self._progress.setFormat(f"done — {len(rows)} run(s)")
        self._btn_export.setEnabled(True)

    # ── helpers ─────────────────────────────────────────────────────

    def _clear_pills(self) -> None:
        while self._pills.count():
            item = self._pills.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()


# ═══════════════════════════════════════════════════════════════════════════
# Top-level merged tab
# ═══════════════════════════════════════════════════════════════════════════

class EvaluationTab(QWidget):
    """
    Single class that hosts all four evaluation modes.

    The user picks an *evaluation type* from the dropdown at the top
    and the relevant settings groups, picker, and results panel become
    visible. Constructor signature is unchanged from the v2
    ``EvaluationTab`` for drop-in wiring.

    Public surface:
      • ``EvaluationTab(data_manager)`` — same as before.
      • ``result_ready``    Signal(:class:`EvaluationResult`) — fired
                            after a Sessions / Games / Unity run.
      • ``sweep_finished``  Signal(``list``) — fired after the CV
                            sweep finishes; payload is
                            ``[(label, RunResult), ...]``.
      • ``refresh()`` — re-scan models, recordings, and CV subjects
                       for all four modes.
    """

    result_ready    = Signal(object)   # EvaluationResult     (modes 1-3)
    sweep_finished  = Signal(object)   # List[(label, RunResult)]  (mode 4)

    # Selection-stack and results-stack page indices
    _PICKER_PAGE_SESSIONS = 0
    _PICKER_PAGE_GAMES    = 1
    _PICKER_PAGE_UNITY    = 2
    _PICKER_PAGE_CV       = 3
    _RESULTS_PAGE_EVAL    = 0
    _RESULTS_PAGE_CV      = 1

    # CV "vary by" sub-stack page indices
    _CV_AXIS_PAGE_NONE    = 0
    _CV_AXIS_PAGE_MODEL   = 1
    _CV_AXIS_PAGE_FEATURES = 2
    _CV_AXIS_PAGE_SUBSET  = 3

    def __init__(self, data_manager: Any, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._data_dir: Path = Path(getattr(data_manager, "data_dir", data_manager))
        if not self._data_dir.exists():
            log.warning("Data directory does not exist: %s", self._data_dir)

        apply_app_style(self, theme="bright")

        # Internal state for modes 1-3 (eval) and 4 (CV)
        self._worker: Optional[Any] = None  # busy_overlay run_blocking handle
        self._cv_bridge: Optional[_SweepProgressBridge] = None
        self._cv_worker: Optional[_SweepWorker] = None
        self._cv_completed_rows: List[Tuple[str, Dict[str, Any]]] = []
        self._cv_completed_runs: List[Tuple[str, Any]] = []
        self._cv_axis_label: str = "(none)"

        # Default subset for CV mode (used when axis ≠ data_subset)
        self._cv_default_subset: DataSubset = DataSubset(
            name="default", subjects=[], domains=[],
        )

        # Build UI top-down
        outer = QVBoxLayout(self); outer.setContentsMargins(12, 12, 12, 12); outer.setSpacing(12)

        outer.addWidget(self._build_header())
        outer.addWidget(self._build_type_picker_bar())

        body = QSplitter(Qt.Orientation.Vertical)
        body.addWidget(self._build_top_pane())
        body.addWidget(self._build_results_pane())
        body.setStretchFactor(0, 1); body.setStretchFactor(1, 1)
        body.setSizes([460, 600])
        body.setHandleWidth(6)
        outer.addWidget(body, 1)

        # Default mode
        self._on_eval_type_changed()
        self._refresh_cv_subjects()

    # ──────────────────────────────────────────────────────────────────
    # Header + type-picker bar
    # ──────────────────────────────────────────────────────────────────

    def _build_header(self) -> QWidget:
        header = QFrame()
        header.setStyleSheet(
            f"QFrame {{ background: qlineargradient(x1:0,y1:0,x2:1,y2:0,"
            f"  stop:0 {C_PANEL}, stop:1 {C_CARD});"
            f"  border:1px solid {C_BORDER}; border-radius:10px; }}"
        )
        h_lay = QHBoxLayout(header); h_lay.setContentsMargins(18, 12, 18, 12); h_lay.setSpacing(12)
        title = QLabel("Evaluation")
        ft = QFont(); ft.setPointSize(16); ft.setBold(True); title.setFont(ft)
        title.setStyleSheet(f"color:{C_ACCENT};")
        h_lay.addWidget(title)
        sub = QLabel("Sessions  ·  game recordings  ·  Unity threshold  ·  cross-validation")
        sub.setStyleSheet(f"color:{C_MUTED}; font-size:11px;")
        h_lay.addWidget(sub)
        h_lay.addStretch(1)
        self._data_dir_lbl = QLabel(str(self._data_dir))
        self._data_dir_lbl.setStyleSheet(
            f"color:{C_MUTED}; font-size:10px; "
            f"font-family:'Menlo','DejaVu Sans Mono',monospace;")
        self._data_dir_lbl.setToolTip("Active data directory")
        h_lay.addWidget(self._data_dir_lbl)
        return header

    def _build_type_picker_bar(self) -> QWidget:
        """
        Single-row picker that drives the whole tab.

        This is the *only* mode-switching control. Picking a different
        type rotates the selection panel, the visible group boxes, the
        Run-button label, and the results view in lock-step.
        """
        bar = QFrame()
        bar.setStyleSheet(
            f"QFrame {{ background:{C_PANEL}; border:1px solid {C_BORDER2};"
            f"  border-radius:8px; }}"
        )
        lay = QHBoxLayout(bar); lay.setContentsMargins(14, 10, 14, 10); lay.setSpacing(10)

        lbl = QLabel("Evaluation type:")
        f = QFont(); f.setBold(True); lbl.setFont(f)
        lbl.setStyleSheet(f"color:{C_ACCENT2}; letter-spacing:0.04em;"
                          f"font-size:11px;")
        lay.addWidget(lbl)

        self._eval_type_combo = QComboBox()
        for key, label in _EVAL_TYPE_LABELS:
            self._eval_type_combo.addItem(label, key)
        self._eval_type_combo.setCurrentIndex(0)
        self._eval_type_combo.setMinimumWidth(360)
        self._eval_type_combo.currentIndexChanged.connect(self._on_eval_type_changed)
        lay.addWidget(self._eval_type_combo)

        lay.addStretch(1)

        # A small hint that updates with the chosen mode
        self._type_hint = QLabel("")
        self._type_hint.setStyleSheet(f"color:{C_MUTED}; font-size:10px;")
        lay.addWidget(self._type_hint)

        return bar

    # ──────────────────────────────────────────────────────────────────
    # Top pane: selection (left) | settings (right)
    # ──────────────────────────────────────────────────────────────────

    def _build_top_pane(self) -> QWidget:
        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(self._build_selection_stack())
        split.addWidget(self._build_settings_column())
        split.setStretchFactor(0, 5); split.setStretchFactor(1, 6)
        split.setSizes([520, 660])
        return split

    def _build_selection_stack(self) -> QWidget:
        """
        Left-side stacked picker. One page per evaluation type.

        Sessions / Games / Unity get a dedicated tree picker; the CV
        page contains an inner stack that mirrors the CV "vary by"
        combo (None / Model / Features / DataSubset).
        """
        self._selection_stack = QStackedWidget()

        # Pages for evaluation modes
        self._sessions_picker = _SessionPicker(self._data_dir)
        self._games_picker    = _GamePicker(self._data_dir)
        self._unity_picker    = _UnityPicker(self._data_dir)

        # Wrap each picker in a thin frame so they all align
        self._selection_stack.addWidget(self._wrap_picker(
            self._sessions_picker,
            "Pick training sessions to evaluate."))
        self._selection_stack.addWidget(self._wrap_picker(
            self._games_picker,
            "Pick game recordings to evaluate."))
        self._selection_stack.addWidget(self._wrap_picker(
            self._unity_picker,
            "Pick Unity recordings to evaluate."))

        # CV page — inner stack swapped by the "vary by" combo (which
        # lives in the right-side settings column).
        self._cv_axis_stack = QStackedWidget()

        # 0: none — single run, no axis values to pick
        none_w = QWidget(); none_l = QVBoxLayout(none_w)
        none_l.setContentsMargins(8, 8, 8, 8)
        none_msg = QLabel(
            "No comparison axis selected.\n\n"
            "The single-run mode runs one validation experiment using the "
            "default model, features, and data subset configured on the "
            "right. Pick a comparison axis from the «Vary by» combo to "
            "sweep over multiple values."
        )
        none_msg.setWordWrap(True)
        none_msg.setStyleSheet(f"color:{C_MUTED}; font-size:11px; padding:20px;")
        none_l.addWidget(none_msg); none_l.addStretch(1)
        self._cv_axis_stack.addWidget(none_w)

        # 1, 2, 3: model / features / subsets
        self._cv_model_axis    = _ModelAxisPanel()
        self._cv_features_axis = _FeaturesAxisPanel()
        self._cv_subset_axis   = _DataSubsetAxisPanel(available_subjects=[])
        self._cv_axis_stack.addWidget(self._cv_model_axis)
        self._cv_axis_stack.addWidget(self._cv_features_axis)
        self._cv_axis_stack.addWidget(self._cv_subset_axis)

        cv_wrap = QWidget()
        cv_lay  = QVBoxLayout(cv_wrap); cv_lay.setContentsMargins(0, 0, 0, 0); cv_lay.setSpacing(8)
        cv_title = _styled_group("Comparison values")
        cv_title_lay = QVBoxLayout(); cv_title_lay.addWidget(self._cv_axis_stack)
        cv_title.setLayout(cv_title_lay)
        cv_lay.addWidget(cv_title, 1)

        self._selection_stack.addWidget(cv_wrap)
        return self._selection_stack

    @staticmethod
    def _wrap_picker(picker: QWidget, hint: str) -> QWidget:
        """Wrap a tree picker in a labelled card so all modes look uniform."""
        wrap = QWidget()
        lay = QVBoxLayout(wrap); lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(6)
        hint_lbl = QLabel(hint)
        hint_lbl.setStyleSheet(f"color:{C_MUTED}; font-size:11px;")
        hint_lbl.setWordWrap(True)
        lay.addWidget(hint_lbl)
        lay.addWidget(picker, 1)
        return wrap

    def _build_settings_column(self) -> QWidget:
        """
        Single flat scroll column holding ALL group-boxes for all four
        modes. Each group-box is shown/hidden as the type combo changes.

        Group-boxes are built in helper methods that store their own
        state as instance attributes so the run handlers can read them
        directly without delegate signals.
        """
        # ── Sessions-mode groups ─────────────────────────────────────
        self._g_sessions_model     = self._build_g_sessions_model()
        self._g_sessions_windowing = self._build_g_sessions_windowing()
        self._g_sessions_preprocessing = self._build_g_sessions_preprocessing()
        self._g_sessions_features  = self._build_g_sessions_features()

        # ── Games-mode groups ────────────────────────────────────────
        self._g_games_source   = self._build_g_games_source()
        self._g_games_filter   = self._build_g_games_filter()
        self._g_games_replay   = self._build_g_games_replay()

        # ── Unity-mode groups ────────────────────────────────────────
        self._g_unity_about    = self._build_g_unity_about()
        self._g_unity_objective = self._build_g_unity_objective()
        self._g_unity_rms      = self._build_g_unity_rms()

        # ── CV-mode groups ───────────────────────────────────────────
        self._g_cv_axis        = self._build_g_cv_axis()
        self._g_cv_strategy    = self._build_g_cv_strategy()
        self._g_cv_windowing   = self._build_g_cv_windowing()
        self._g_cv_defaults    = self._build_g_cv_defaults()

        # ── Run row (bottom of the scroll) ───────────────────────────
        # The Run button label changes per mode. Cancel is only useful
        # for CV (a sweep is the only thing that runs long enough to
        # need cancellation; the eval modes block via run_blocking and
        # finish quickly).
        self._btn_features = _ghost_button("Rank features (LDA)", accent=True)
        self._btn_features.clicked.connect(self._on_run_features_clicked)
        self._btn_run    = _primary_button("▶  Run evaluation")
        self._btn_run.clicked.connect(self._on_run_clicked)
        self._btn_cancel = _ghost_button("Cancel")
        self._btn_cancel.setEnabled(False)
        self._btn_cancel.clicked.connect(self._on_cancel_clicked)

        run_row = QHBoxLayout(); run_row.setSpacing(8)
        run_row.addWidget(self._btn_features)
        run_row.addStretch(1)
        run_row.addWidget(self._btn_cancel)
        run_row.addWidget(self._btn_run)
        run_holder = QWidget(); run_holder.setLayout(run_row)

        # Stack ALL groups in one scroll, hide based on mode
        all_groups = [
            self._g_sessions_model, self._g_sessions_windowing,
            self._g_sessions_preprocessing, self._g_sessions_features,
            self._g_games_source, self._g_games_filter, self._g_games_replay,
            self._g_unity_about, self._g_unity_objective, self._g_unity_rms,
            self._g_cv_axis, self._g_cv_strategy, self._g_cv_windowing,
            self._g_cv_defaults,
        ]
        return _settings_scroll_host(all_groups, run_holder)

    # ──────────────────────────────────────────────────────────────────
    # Settings group builders — Sessions mode
    # ──────────────────────────────────────────────────────────────────

    def _build_g_sessions_model(self) -> QGroupBox:
        g = _styled_group("Model  (sessions)")
        gm_lay = _form()
        self._sessions_model_combo = QComboBox(); self._sessions_model_combo.setMinimumWidth(280)
        self._refresh_sessions_models()
        ref_btn = _ghost_button("⟳"); ref_btn.setToolTip("Re-scan data/models")
        ref_btn.clicked.connect(self._refresh_sessions_models)
        model_row = QHBoxLayout(); model_row.setContentsMargins(0,0,0,0); model_row.setSpacing(6)
        model_row.addWidget(self._sessions_model_combo, 1); model_row.addWidget(ref_btn)
        mw = QWidget(); mw.setLayout(model_row)
        gm_lay.addRow("Saved model:", mw)
        g.setLayout(gm_lay)
        return g

    def _build_g_sessions_windowing(self) -> QGroupBox:
        g = _styled_group("Windowing  (sessions)")
        gw_lay = _form()
        self._sessions_win_spin = QSpinBox(); self._sessions_win_spin.setRange(0, 5000); self._sessions_win_spin.setSingleStep(50)
        self._sessions_win_spin.setSpecialValueText("(use model)"); self._sessions_win_spin.setSuffix(" ms")
        gw_lay.addRow("Window length:", self._sessions_win_spin)
        self._sessions_stride_spin = QSpinBox(); self._sessions_stride_spin.setRange(1, 5000)
        self._sessions_stride_spin.setValue(50); self._sessions_stride_spin.setSuffix(" ms")
        gw_lay.addRow("Window stride:", self._sessions_stride_spin)
        g.setLayout(gw_lay)
        return g

    def _build_g_sessions_preprocessing(self) -> QGroupBox:
        g = _styled_group("Preprocessing  (sessions)")
        gp_lay = _form()
        self._sessions_bad_ch_combo = QComboBox()
        self._sessions_bad_ch_combo.addItem("Interpolate", "interpolate")
        self._sessions_bad_ch_combo.addItem("Zero",        "zero")
        gp_lay.addRow("Bad channels:", self._sessions_bad_ch_combo)
        self._sessions_rotation_chk = QCheckBox("Apply per-session rotation")
        self._sessions_rotation_chk.setChecked(True)
        gp_lay.addRow("", self._sessions_rotation_chk)
        self._sessions_invalid_chk = QCheckBox("Include trials marked invalid")
        gp_lay.addRow("", self._sessions_invalid_chk)
        self._sessions_per_session_chk = QCheckBox("Show per-session breakdown in notes")
        self._sessions_per_session_chk.setChecked(True)
        gp_lay.addRow("", self._sessions_per_session_chk)
        g.setLayout(gp_lay)
        return g

    def _build_g_sessions_features(self) -> QGroupBox:
        g = _styled_group("Optional · feature ranking  (sessions)")
        gf = QVBoxLayout(); gf.setSpacing(8)
        info = QLabel("Score each feature individually with an LDA, in addition to "
                      "running the saved model. Useful as a quick 'which feature carries "
                      "the gesture signal best on this dataset' check.")
        info.setStyleSheet(f"color:{C_MUTED}; font-size:10px;")
        info.setWordWrap(True); gf.addWidget(info)
        feat_grid = QHBoxLayout(); feat_grid.setSpacing(10)
        self._sessions_feat_choices: List[QCheckBox] = []
        for f in ("rms", "mav", "wl", "zc", "ssc", "var", "iemg"):
            cb = QCheckBox(f); cb.setChecked(f in {"rms", "mav", "wl"})
            self._sessions_feat_choices.append(cb); feat_grid.addWidget(cb)
        feat_grid.addStretch(1)
        gf.addLayout(feat_grid)
        # Note: the "Rank features (LDA)" button lives in the run row at
        # the bottom, not inside this group, so it stays visible even
        # when this group is scrolled past.
        g.setLayout(gf)
        return g

    # ──────────────────────────────────────────────────────────────────
    # Settings group builders — Games mode
    # ──────────────────────────────────────────────────────────────────

    def _build_g_games_source(self) -> QGroupBox:
        g = _styled_group("Prediction source  (games)")
        gs = _form()
        self._games_mode_combo = QComboBox()
        self._games_mode_combo.addItem("Logged predictions  (use Pred* columns)", "logged")
        self._games_mode_combo.addItem("Replay model on EMG   (re-run a saved model)", "replay")
        self._games_mode_combo.currentIndexChanged.connect(self._on_games_source_changed)
        gs.addRow("Source:", self._games_mode_combo)

        self._games_truth_combo = QComboBox()
        # Order reflects post-migration semantics: RequestedGesture is the
        # cleanest multi-class ground truth (with "rest" as a real class
        # corresponding to walking-between-animals phases).
        self._games_truth_combo.addItem("RequestedGesture  (multi-class — recommended)", TRUTH_REQUESTED)
        self._games_truth_combo.addItem("GroundTruthActive  (binary, with camera-blocking grace)", TRUTH_ACTIVE)
        self._games_truth_combo.addItem("RawGroundTruth  (binary, raw Unity flag)", TRUTH_RAW)
        gs.addRow("Ground truth:", self._games_truth_combo)
        g.setLayout(gs)
        return g

    def _build_g_games_filter(self) -> QGroupBox:
        g = _styled_group("Filtering  (games)")
        gf = _form()
        self._games_drop_chk = QCheckBox(
            "Restrict to active-gesture frames  (drop walking-phase rest periods)"
        )
        self._games_drop_chk.setToolTip(
            "When OFF (default): the rest periods between animals are evaluated "
            "as legitimate 'rest' (label 0) ground truth — false fists during "
            "walking count as errors.\n"
            "When ON: only frames where the game was actively asking for a "
            "specific gesture are evaluated. Useful for measuring per-gesture "
            "accuracy in isolation, ignoring rest-period behaviour."
        )
        self._games_drop_chk.setChecked(False)
        gf.addRow("", self._games_drop_chk)
        self._games_min_conf_chk = QCheckBox("Filter by minimum confidence")
        self._games_min_conf_spin = QDoubleSpinBox()
        self._games_min_conf_spin.setRange(0.0, 1.0); self._games_min_conf_spin.setSingleStep(0.05)
        self._games_min_conf_spin.setValue(0.5); self._games_min_conf_spin.setEnabled(False)
        self._games_min_conf_chk.toggled.connect(self._games_min_conf_spin.setEnabled)
        conf_row = QHBoxLayout(); conf_row.setContentsMargins(0,0,0,0); conf_row.setSpacing(8)
        conf_row.addWidget(self._games_min_conf_chk); conf_row.addWidget(self._games_min_conf_spin); conf_row.addStretch(1)
        cw = QWidget(); cw.setLayout(conf_row)
        gf.addRow("", cw)
        self._games_per_chk = QCheckBox("Show per-recording breakdown in notes")
        self._games_per_chk.setChecked(True)
        gf.addRow("", self._games_per_chk)
        g.setLayout(gf)
        return g

    def _build_g_games_replay(self) -> QGroupBox:
        g = _styled_group("Replay options  (games)")
        gr = _form()
        self._games_model_combo = QComboBox(); self._refresh_games_models()
        ref_btn = _ghost_button("⟳"); ref_btn.clicked.connect(self._refresh_games_models)
        mr = QHBoxLayout(); mr.setContentsMargins(0,0,0,0); mr.setSpacing(6)
        mr.addWidget(self._games_model_combo, 1); mr.addWidget(ref_btn)
        mw = QWidget(); mw.setLayout(mr)
        gr.addRow("Saved model:", mw)
        self._games_win_spin = QSpinBox(); self._games_win_spin.setRange(0, 5000); self._games_win_spin.setSingleStep(50)
        self._games_win_spin.setSpecialValueText("(use model)"); self._games_win_spin.setSuffix(" ms")
        gr.addRow("Window length:", self._games_win_spin)
        self._games_stride_spin = QSpinBox(); self._games_stride_spin.setRange(1, 5000); self._games_stride_spin.setValue(50)
        self._games_stride_spin.setSuffix(" ms")
        gr.addRow("Window stride:", self._games_stride_spin)
        g.setLayout(gr)
        return g

    # ──────────────────────────────────────────────────────────────────
    # Settings group builders — Unity mode
    # ──────────────────────────────────────────────────────────────────

    def _build_g_unity_about(self) -> QGroupBox:
        g = _styled_group("Threshold model  (unity)")
        ga = QVBoxLayout(); ga.setSpacing(6)
        info = QLabel(
            "Unity recordings use a binary RMS-threshold model: "
            "<i>RMS &gt; threshold ⇒ active gesture</i>. We sweep thresholds, "
            "pick the optimum by your chosen objective, and report binary "
            "metrics (accuracy, precision, recall/sensitivity, specificity, F1) "
            "plus AUROC across the sweep."
        )
        info.setWordWrap(True); info.setStyleSheet(f"color:{C_MUTED}; font-size:11px; padding:2px;")
        ga.addWidget(info)
        g.setLayout(ga)
        return g

    def _build_g_unity_objective(self) -> QGroupBox:
        g = _styled_group("Threshold objective  (unity)")
        go = _form()
        self._unity_objective_combo = QComboBox()
        self._unity_objective_combo.addItem("Maximise F1",         "f1")
        self._unity_objective_combo.addItem("Maximise Youden's J", "youden")
        self._unity_objective_combo.addItem("Maximise accuracy",   "accuracy")
        go.addRow("Objective:", self._unity_objective_combo)
        self._unity_fixed_chk = QCheckBox("Use a fixed threshold instead")
        self._unity_fixed_spin = QDoubleSpinBox()
        self._unity_fixed_spin.setRange(0.0, 1e6); self._unity_fixed_spin.setDecimals(8)
        self._unity_fixed_spin.setSingleStep(1e-5); self._unity_fixed_spin.setValue(1e-4)
        self._unity_fixed_spin.setEnabled(False)
        self._unity_fixed_chk.toggled.connect(self._unity_fixed_spin.setEnabled)
        fixed_row = QHBoxLayout(); fixed_row.setContentsMargins(0,0,0,0); fixed_row.setSpacing(8)
        fixed_row.addWidget(self._unity_fixed_chk); fixed_row.addWidget(self._unity_fixed_spin); fixed_row.addStretch(1)
        fw = QWidget(); fw.setLayout(fixed_row)
        go.addRow("Fixed:", fw)
        self._unity_n_thresh_spin = QSpinBox(); self._unity_n_thresh_spin.setRange(20, 1000); self._unity_n_thresh_spin.setValue(200)
        go.addRow("Sweep points:", self._unity_n_thresh_spin)
        g.setLayout(go)
        return g

    def _build_g_unity_rms(self) -> QGroupBox:
        g = _styled_group("Computed-RMS  (unity, converted sessions only)")
        gr = _form()
        rms_info = QLabel("These settings only apply when the source is a "
                           "converted Unity session — for raw Unity CSVs the "
                           "logged RMS column is used directly.")
        rms_info.setWordWrap(True); rms_info.setStyleSheet(f"color:{C_MUTED}; font-size:10px;")
        gr.addRow(rms_info)
        self._unity_rms_win_spin = QSpinBox(); self._unity_rms_win_spin.setRange(1, 5000); self._unity_rms_win_spin.setValue(100)
        self._unity_rms_win_spin.setSuffix(" ms")
        gr.addRow("RMS window:", self._unity_rms_win_spin)
        self._unity_rms_stride_spin = QSpinBox(); self._unity_rms_stride_spin.setRange(1, 5000); self._unity_rms_stride_spin.setValue(25)
        self._unity_rms_stride_spin.setSuffix(" ms")
        gr.addRow("RMS stride:", self._unity_rms_stride_spin)
        self._unity_active_label_spin = QSpinBox(); self._unity_active_label_spin.setRange(0, 100); self._unity_active_label_spin.setValue(1)
        gr.addRow("Active label id:", self._unity_active_label_spin)
        g.setLayout(gr)
        return g

    # ──────────────────────────────────────────────────────────────────
    # Settings group builders — Cross-validation mode
    # ──────────────────────────────────────────────────────────────────

    def _build_g_cv_axis(self) -> QGroupBox:
        g = _styled_group("Comparison axis  (cv)")
        ga = _form()
        self._cv_axis_combo = QComboBox()
        self._cv_axis_combo.addItem("Single run  (no comparison)",   "none")
        self._cv_axis_combo.addItem("Vary by model",                 "model")
        self._cv_axis_combo.addItem("Vary by feature set",           "features")
        self._cv_axis_combo.addItem("Vary by data subset",           "data_subset")
        self._cv_axis_combo.currentIndexChanged.connect(self._on_cv_axis_changed)
        ga.addRow("Vary by:", self._cv_axis_combo)

        self._cv_name_edit = QLineEdit("cv_sweep")
        ga.addRow("Run name:", self._cv_name_edit)
        self._cv_seed_spin = QSpinBox(); self._cv_seed_spin.setRange(0, 99999); self._cv_seed_spin.setValue(42)
        ga.addRow("Seed:", self._cv_seed_spin)
        g.setLayout(ga)
        return g

    def _build_g_cv_strategy(self) -> QGroupBox:
        g = _styled_group("CV strategy  (cv)")
        gv = QVBoxLayout(); gv.setSpacing(8)
        cv_form = _form()
        self._cv_strategy_combo = QComboBox()
        for key, label, tooltip in _CV_STRATEGIES:
            self._cv_strategy_combo.addItem(label, key)
            self._cv_strategy_combo.setItemData(
                self._cv_strategy_combo.count() - 1, tooltip,
                Qt.ItemDataRole.ToolTipRole,
            )
        self._cv_strategy_combo.currentIndexChanged.connect(self._on_cv_strategy_changed)
        cv_form.addRow("Strategy:", self._cv_strategy_combo)
        gv.addLayout(cv_form)

        # Strategy-specific parameter stack
        self._cv_params_stack = QStackedWidget()
        # 0: empty (loso_subject / loso_session / within_session)
        self._cv_params_stack.addWidget(QWidget())
        # 1: k-fold
        kf_w = QWidget(); kf_l = _form()
        self._cv_kfold_spin = QSpinBox(); self._cv_kfold_spin.setRange(2, 20); self._cv_kfold_spin.setValue(5)
        kf_l.addRow("k:", self._cv_kfold_spin)
        kf_w.setLayout(kf_l); self._cv_params_stack.addWidget(kf_w)
        # 2: cross-domain
        xd_w = QWidget(); xd_l = _form()
        self._cv_xd_train = QComboBox(); self._cv_xd_train.addItems(["pipeline", "unity"])
        self._cv_xd_test  = QComboBox(); self._cv_xd_test .addItems(["unity", "pipeline"])
        xd_l.addRow("Train domain:", self._cv_xd_train)
        xd_l.addRow("Test domain:",  self._cv_xd_test)
        xd_w.setLayout(xd_l); self._cv_params_stack.addWidget(xd_w)
        # 3: holdout
        ho_w = QWidget(); ho_l = _form()
        self._cv_ho_val  = QDoubleSpinBox(); self._cv_ho_val .setRange(0.0, 0.5); self._cv_ho_val .setSingleStep(0.05); self._cv_ho_val .setValue(0.15)
        self._cv_ho_test = QDoubleSpinBox(); self._cv_ho_test.setRange(0.0, 0.5); self._cv_ho_test.setSingleStep(0.05); self._cv_ho_test.setValue(0.20)
        self._cv_ho_strat = QComboBox()
        self._cv_ho_strat.addItem("None",     "none")
        self._cv_ho_strat.addItem("By class", "class")
        self._cv_ho_strat.addItem("By subject", "subject")
        ho_l.addRow("Val ratio:",  self._cv_ho_val)
        ho_l.addRow("Test ratio:", self._cv_ho_test)
        ho_l.addRow("Stratify:",   self._cv_ho_strat)
        ho_w.setLayout(ho_l); self._cv_params_stack.addWidget(ho_w)
        gv.addWidget(self._cv_params_stack)
        g.setLayout(gv)
        return g

    def _build_g_cv_windowing(self) -> QGroupBox:
        g = _styled_group("Windowing  (cv)")
        gw = _form()
        self._cv_win_spin = QSpinBox(); self._cv_win_spin.setRange(50, 5000); self._cv_win_spin.setSingleStep(50); self._cv_win_spin.setValue(200); self._cv_win_spin.setSuffix(" ms")
        gw.addRow("Window length:", self._cv_win_spin)
        self._cv_stride_spin = QSpinBox(); self._cv_stride_spin.setRange(1, 5000); self._cv_stride_spin.setValue(50); self._cv_stride_spin.setSuffix(" ms")
        gw.addRow("Window stride:", self._cv_stride_spin)
        self._cv_drop_rest_chk = QCheckBox("Drop rest-class windows during training")
        self._cv_drop_rest_chk.setChecked(False)
        self._cv_drop_rest_chk.setToolTip(
            "When checked, the runner removes label-0 (rest) windows before "
            "training. Useful for measuring per-active-gesture accuracy in "
            "isolation. Leave OFF for the honest 'how does the model handle "
            "rest periods' picture."
        )
        gw.addRow("", self._cv_drop_rest_chk)
        g.setLayout(gw)
        return g

    def _build_g_cv_defaults(self) -> QGroupBox:
        g = _styled_group("Defaults for non-varied axes  (cv)")
        gd = QVBoxLayout(); gd.setSpacing(10)

        # Default model — combo (used when axis ≠ "model")
        def_form = _form()
        self._cv_default_model = QComboBox()
        for key, _ in _AVAILABLE_MODELS:
            self._cv_default_model.addItem(key, key)
        idx = self._cv_default_model.findData(_DEFAULT_MODEL)
        if idx >= 0:
            self._cv_default_model.setCurrentIndex(idx)
        def_form.addRow("Default model:", self._cv_default_model)
        gd.addLayout(def_form)

        # Default features — checklist (used when axis ≠ "features")
        feat_lab = QLabel("Default features  (used when axis ≠ features):")
        feat_lab.setStyleSheet(f"color:{C_MUTED}; font-size:10px; letter-spacing:0.04em;")
        gd.addWidget(feat_lab)
        self._cv_default_features = _CheckList(_AVAILABLE_FEATURES,
                                                initial_checked=_DEFAULT_FEATURE_SET)
        gd.addWidget(self._cv_default_features)

        # Default subset — single editable subset
        sub_lab = QLabel("Default data subset  (used when axis ≠ data subset):")
        sub_lab.setStyleSheet(f"color:{C_MUTED}; font-size:10px; letter-spacing:0.04em;")
        gd.addWidget(sub_lab)
        sub_row = QHBoxLayout(); sub_row.setContentsMargins(0, 0, 0, 0); sub_row.setSpacing(6)
        self._cv_default_subset_lbl = QLabel(self._cv_default_subset.describe())
        self._cv_default_subset_lbl.setStyleSheet(
            f"color:{C_TEXT}; padding:6px 10px; background:{C_BG};"
            f"  border:1px solid {C_BORDER2}; border-radius:5px;"
        )
        edit_btn = _ghost_button("Edit…")
        edit_btn.clicked.connect(self._on_cv_edit_default_subset)
        sub_row.addWidget(self._cv_default_subset_lbl, 1)
        sub_row.addWidget(edit_btn)
        gd.addLayout(sub_row)
        g.setLayout(gd)
        return g

    # ──────────────────────────────────────────────────────────────────
    # Results pane (stacked: eval result | CV comparison)
    # ──────────────────────────────────────────────────────────────────

    def _build_results_pane(self) -> QWidget:
        self._results_stack = QStackedWidget()

        self._eval_results = _EvaluationResultView()
        self._cv_results   = _CrossValidationResultsView()
        self._cv_results.export_requested.connect(self._on_cv_export)

        self._results_stack.addWidget(self._eval_results)   # page 0
        self._results_stack.addWidget(self._cv_results)     # page 1
        return self._results_stack

    # ──────────────────────────────────────────────────────────────────
    # Mode-switching slots
    # ──────────────────────────────────────────────────────────────────

    def _on_eval_type_changed(self, *_args) -> None:
        """
        Driver for the entire tab. Toggles the visibility of every
        settings group, swaps the left-hand picker, swaps the result
        panel, and updates the Run button label / Sessions-only feature
        button visibility.

        Each group is bound to exactly one mode; we just iterate the
        per-mode lists and call setVisible(True/False).
        """
        mode = self._eval_type_combo.currentData()

        # All the sets are computed up-front to make the visibility
        # toggle a single linear scan. Each group only ever lives in
        # one of these sets, so flipping every group is correct.
        sessions_groups = (self._g_sessions_model, self._g_sessions_windowing,
                           self._g_sessions_preprocessing, self._g_sessions_features)
        games_groups    = (self._g_games_source, self._g_games_filter,
                           self._g_games_replay)
        unity_groups    = (self._g_unity_about, self._g_unity_objective,
                           self._g_unity_rms)
        cv_groups       = (self._g_cv_axis, self._g_cv_strategy,
                           self._g_cv_windowing, self._g_cv_defaults)

        all_off = (
            *sessions_groups, *games_groups, *unity_groups, *cv_groups,
        )
        for g in all_off:
            g.setVisible(False)

        if mode == EVAL_TYPE_SESSIONS:
            for g in sessions_groups: g.setVisible(True)
            self._selection_stack.setCurrentIndex(self._PICKER_PAGE_SESSIONS)
            self._results_stack.setCurrentIndex(self._RESULTS_PAGE_EVAL)
            self._btn_features.setVisible(True)
            self._btn_run.setText("▶  Run evaluation")
            self._type_hint.setText("Run a saved model over training-session recordings.")
        elif mode == EVAL_TYPE_GAMES:
            for g in games_groups: g.setVisible(True)
            self._selection_stack.setCurrentIndex(self._PICKER_PAGE_GAMES)
            self._results_stack.setCurrentIndex(self._RESULTS_PAGE_EVAL)
            self._btn_features.setVisible(False)
            self._btn_run.setText("▶  Run evaluation")
            self._type_hint.setText("Evaluate logged predictions or replay a model on EMG.")
            # Keep Games' replay-options visibility honest with the source combo
            self._on_games_source_changed()
        elif mode == EVAL_TYPE_UNITY:
            for g in unity_groups: g.setVisible(True)
            self._selection_stack.setCurrentIndex(self._PICKER_PAGE_UNITY)
            self._results_stack.setCurrentIndex(self._RESULTS_PAGE_EVAL)
            self._btn_features.setVisible(False)
            self._btn_run.setText("▶  Run evaluation")
            self._type_hint.setText("Sweep RMS thresholds to find an optimal binary cutoff.")
        elif mode == EVAL_TYPE_CV:
            for g in cv_groups: g.setVisible(True)
            self._selection_stack.setCurrentIndex(self._PICKER_PAGE_CV)
            self._results_stack.setCurrentIndex(self._RESULTS_PAGE_CV)
            self._btn_features.setVisible(False)
            self._btn_run.setText("▶  Run sweep")
            self._type_hint.setText("Sweep multiple validation runs to compare models / features / data.")
            # Sync the CV inner stacks with the combos
            self._on_cv_axis_changed()
            self._on_cv_strategy_changed()
        else:
            log.warning("Unknown evaluation type: %r", mode)

        # Cancel button is only useful in CV mode AND only enabled while
        # a sweep is in flight; here we just make sure it's disabled
        # whenever the mode changes (a fresh switch never has a worker
        # running).
        self._btn_cancel.setEnabled(False)

    def _on_cv_axis_changed(self, *_args) -> None:
        """Swap the left-side CV axis-values stack to match the combo."""
        axis = self._cv_axis_combo.currentData()
        idx_map = {
            "none":         self._CV_AXIS_PAGE_NONE,
            "model":        self._CV_AXIS_PAGE_MODEL,
            "features":     self._CV_AXIS_PAGE_FEATURES,
            "data_subset":  self._CV_AXIS_PAGE_SUBSET,
        }
        self._cv_axis_stack.setCurrentIndex(idx_map.get(axis, self._CV_AXIS_PAGE_NONE))

    def _on_cv_strategy_changed(self, *_args) -> None:
        """Swap the strategy-specific parameters stack to match the combo."""
        key = self._cv_strategy_combo.currentData()
        idx_map = {
            "loso_subject":    0,
            "loso_session":    0,
            "within_session":  0,
            "k_fold_subjects": 1,
            "cross_domain":    2,
            "holdout_split":   3,
        }
        self._cv_params_stack.setCurrentIndex(idx_map.get(key, 0))

    def _on_games_source_changed(self, *_args) -> None:
        """
        Keep the Games-mode replay options visible only when source = replay.

        Note this applies *within* games mode; in other modes the entire
        replay group is hidden by ``_on_eval_type_changed`` regardless.
        """
        if self._eval_type_combo.currentData() != EVAL_TYPE_GAMES:
            return
        is_replay = (self._games_mode_combo.currentData() == "replay")
        self._g_games_replay.setVisible(is_replay)

    # ──────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────

    def refresh(self) -> None:
        """Re-scan recordings, models, and CV subjects across all modes."""
        self._sessions_picker.refresh()
        self._games_picker.refresh()
        self._unity_picker.refresh()
        self._refresh_sessions_models()
        self._refresh_games_models()
        self._refresh_cv_subjects()

    # ──────────────────────────────────────────────────────────────────
    # Refresh helpers
    # ──────────────────────────────────────────────────────────────────

    def _refresh_sessions_models(self) -> None:
        prev = self._sessions_model_combo.currentText() if hasattr(self, "_sessions_model_combo") else ""
        self._sessions_model_combo.clear()
        names = _model_dirs(self._data_dir)
        if not names:
            self._sessions_model_combo.addItem("(no models found in data/models)", None)
            self._sessions_model_combo.setEnabled(False)
        else:
            self._sessions_model_combo.setEnabled(True)
            for n in names:
                self._sessions_model_combo.addItem(n, n)
            if prev and prev in names:
                self._sessions_model_combo.setCurrentText(prev)

    def _refresh_games_models(self) -> None:
        prev = self._games_model_combo.currentText() if hasattr(self, "_games_model_combo") else ""
        self._games_model_combo.clear()
        names = _model_dirs(self._data_dir)
        if not names:
            self._games_model_combo.addItem("(no models found)", None)
            self._games_model_combo.setEnabled(False)
        else:
            self._games_model_combo.setEnabled(True)
            for n in names:
                self._games_model_combo.addItem(n, n)
            if prev and prev in names:
                self._games_model_combo.setCurrentText(prev)

    def _refresh_cv_subjects(self) -> None:
        subjects = self._discover_subjects()
        self._cv_subset_axis.set_available_subjects(subjects)

    def _discover_subjects(self) -> List[str]:
        """Pull subject IDs from ``<data_dir>/sessions/`` and ``unity_sessions``."""
        out: set = set()
        sess_root = self._data_dir / "sessions"
        if sess_root.exists():
            for d in sess_root.iterdir():
                if d.is_dir() and d.name not in {"unity_sessions"}:
                    out.add(d.name)
            unity_root = sess_root / "unity_sessions"
            if unity_root.exists():
                for d in unity_root.iterdir():
                    if d.is_dir():
                        out.add(d.name)
        return sorted(out, key=lambda s: (not s.upper().startswith("VP_"), s))

    def _on_cv_edit_default_subset(self) -> None:
        dlg = _SubsetEditorDialog(self._discover_subjects(),
                                   subset=self._cv_default_subset, parent=self)
        if dlg.exec():
            self._cv_default_subset = dlg.result_subset()
            self._cv_default_subset_lbl.setText(self._cv_default_subset.describe())

    # ──────────────────────────────────────────────────────────────────
    # Run dispatch  (one button, four code paths)
    # ──────────────────────────────────────────────────────────────────

    def _on_run_clicked(self) -> None:
        """
        The single Run button at the bottom of the settings column
        dispatches to the appropriate handler based on the active mode.
        """
        mode = self._eval_type_combo.currentData()
        if mode == EVAL_TYPE_SESSIONS:
            self._run_sessions()
        elif mode == EVAL_TYPE_GAMES:
            self._run_games()
        elif mode == EVAL_TYPE_UNITY:
            self._run_unity()
        elif mode == EVAL_TYPE_CV:
            self._on_run_cv_clicked()
        else:
            log.warning("Unknown evaluation type at run time: %r", mode)

    def _on_cancel_clicked(self) -> None:
        """Only meaningful in CV mode while a sweep is running."""
        if self._cv_bridge is not None:
            self._cv_bridge.request_cancel()
            self._cv_results.append_log(
                "Cancellation requested — finishing the current run, then stopping."
            )
            self._btn_cancel.setEnabled(False)

    def _on_run_features_clicked(self) -> None:
        """Sessions-only: rank features individually with an LDA."""
        recs = self._sessions_picker.selected()
        if not recs:
            QMessageBox.information(self, "No sessions",
                                    "Select at least one session on the left.")
            return
        feats = [c.text() for c in self._sessions_feat_choices if c.isChecked()]
        if not feats:
            QMessageBox.information(self, "No features",
                                    "Select at least one feature to rank.")
            return
        data_dir = self._data_dir
        def task() -> EvaluationResult:
            return evaluate_features_lda(data_dir, recs, feats)
        self._launch_eval(task, "Ranking features…")

    # ──────────────────────────────────────────────────────────────────
    # Settings collection helpers
    # ──────────────────────────────────────────────────────────────────

    def _collect_sessions_settings(self) -> Optional[SessionEvalSettings]:
        model_name = self._sessions_model_combo.currentData()
        if not model_name:
            QMessageBox.warning(self, "No model",
                                "No saved model is available — train a model first.")
            return None
        return SessionEvalSettings(
            model_name=str(model_name),
            window_size_ms=self._sessions_win_spin.value() or None,
            window_stride_ms=int(self._sessions_stride_spin.value()),
            apply_rotation=self._sessions_rotation_chk.isChecked(),
            bad_channel_mode=self._sessions_bad_ch_combo.currentData(),
            include_invalid=self._sessions_invalid_chk.isChecked(),
            per_session_breakdown=self._sessions_per_session_chk.isChecked(),
        )

    def _collect_games_settings(self) -> Optional[GameEvalSettings]:
        mode = self._games_mode_combo.currentData()
        s = GameEvalSettings(
            mode=mode,
            truth_source=self._games_truth_combo.currentData(),
            drop_inactive_truth_frames=self._games_drop_chk.isChecked(),
            min_confidence=(self._games_min_conf_spin.value()
                            if self._games_min_conf_chk.isChecked() else None),
            per_recording_breakdown=self._games_per_chk.isChecked(),
        )
        if mode == "replay":
            model_name = self._games_model_combo.currentData()
            if not model_name:
                QMessageBox.warning(self, "No model",
                                    "Replay mode requires a saved model.")
                return None
            s.model_name = str(model_name)
            s.window_size_ms = self._games_win_spin.value() or None
            s.window_stride_ms = int(self._games_stride_spin.value())
        return s

    def _collect_unity_settings(self) -> UnityEvalSettings:
        return UnityEvalSettings(
            active_label=int(self._unity_active_label_spin.value()),
            rms_window_ms=int(self._unity_rms_win_spin.value()),
            rms_stride_ms=int(self._unity_rms_stride_spin.value()),
            n_thresholds=int(self._unity_n_thresh_spin.value()),
            objective=str(self._unity_objective_combo.currentData()),
            fixed_threshold=(self._unity_fixed_spin.value()
                             if self._unity_fixed_chk.isChecked() else None),
        )

    def _collect_cv_defaults(self) -> SweepDefaults:
        # CV kwargs depend on which strategy is active
        cv_key = self._cv_strategy_combo.currentData()
        cv_kwargs: Dict[str, Any] = {}
        if cv_key == "k_fold_subjects":
            cv_kwargs = {"k": int(self._cv_kfold_spin.value()),
                         "seed": int(self._cv_seed_spin.value())}
        elif cv_key == "cross_domain":
            cv_kwargs = {"train_domain": self._cv_xd_train.currentText(),
                         "test_domain":  self._cv_xd_test.currentText()}
        elif cv_key == "holdout_split":
            cv_kwargs = {"val_ratio":  float(self._cv_ho_val .value()),
                         "test_ratio": float(self._cv_ho_test.value()),
                         "seed":       int(self._cv_seed_spin.value()),
                         "stratify_by": self._cv_ho_strat.currentData()}

        return SweepDefaults(
            name=self._cv_name_edit.text().strip() or "cv_sweep",
            seed=int(self._cv_seed_spin.value()),
            window_ms=int(self._cv_win_spin.value()),
            stride_ms=int(self._cv_stride_spin.value()),
            drop_rest=self._cv_drop_rest_chk.isChecked(),
            cv_strategy=cv_key,
            cv_kwargs=cv_kwargs,
            default_subset=self._cv_default_subset,
            default_models=[self._cv_default_model.currentData()],
            default_features=self._cv_default_features.selected(),
        )

    # ──────────────────────────────────────────────────────────────────
    # Run handlers — Sessions / Games / Unity (modes 1-3)
    # ──────────────────────────────────────────────────────────────────

    def _run_sessions(self) -> None:
        recs = self._sessions_picker.selected()
        if not recs:
            QMessageBox.information(self, "No sessions",
                                    "Select at least one session on the left.")
            return
        settings = self._collect_sessions_settings()
        if settings is None:
            return
        data_dir = self._data_dir
        def task() -> EvaluationResult:
            return evaluate_sessions(data_dir, recs, settings)
        self._launch_eval(task, "Running session evaluation…")

    def _run_games(self) -> None:
        recs = self._games_picker.selected()
        if not recs:
            QMessageBox.information(self, "No recordings",
                                    "Select at least one game recording on the left.")
            return
        settings = self._collect_games_settings()
        if settings is None:
            return
        data_dir = self._data_dir
        def task() -> EvaluationResult:
            return evaluate_games(data_dir, recs, settings)
        self._launch_eval(task, "Evaluating game recordings…")

    def _run_unity(self) -> None:
        recs = self._unity_picker.selected()
        if not recs:
            QMessageBox.information(self, "No recordings",
                                    "Select at least one Unity recording on the left.")
            return
        settings = self._collect_unity_settings()
        def task() -> EvaluationResult:
            return evaluate_unity(recs, settings)
        self._launch_eval(task, "Sweeping RMS thresholds…")

    def _launch_eval(self, task: Callable[[], EvaluationResult], label: str) -> None:
        """Run a single eval-mode task on a worker thread via run_blocking."""
        self._worker = run_blocking(
            parent_widget=self,
            fn=task,
            on_done=self._on_eval_result_ready,
            on_error=self._on_eval_error,
            label=label,
        )

    def _on_eval_result_ready(self, result: EvaluationResult) -> None:
        self._eval_results.show_result(result)
        self.result_ready.emit(result)

    def _on_eval_error(self, tb: str) -> None:
        log.error("Evaluation failed:\n%s", tb)
        last_line = next((ln for ln in reversed(tb.splitlines()) if ln.strip()), "Unknown error")
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Icon.Critical)
        msg.setWindowTitle("Evaluation failed")
        msg.setText(last_line)
        msg.setDetailedText(tb)
        msg.setStandardButtons(QMessageBox.StandardButton.Ok)
        msg.exec()

    # ──────────────────────────────────────────────────────────────────
    # Run handler — Cross-validation  (mode 4)
    # ──────────────────────────────────────────────────────────────────

    def _on_run_cv_clicked(self) -> None:
        axis = self._cv_axis_combo.currentData()
        defaults = self._collect_cv_defaults()

        # Pre-flight: defaults sanity
        if not defaults.default_features:
            QMessageBox.warning(self, "No default features",
                                "Select at least one default feature in the "
                                "«Defaults for non-varied axes» group.")
            return

        # Build plans per axis. Each branch validates its own per-axis
        # values before calling plan_sweep so the user gets an explicit
        # 'nothing checked' message instead of a silent zero-run sweep.
        if axis == "none":
            plans = plan_sweep(axis="none", defaults=defaults)
        elif axis == "model":
            models, labels = self._cv_model_axis.axis_values()
            if not models:
                QMessageBox.warning(self, "No models selected",
                                    "Tick at least one model in the comparison list.")
                return
            plans = plan_sweep(axis="model", defaults=defaults,
                                axis_models=models, axis_labels=labels)
        elif axis == "features":
            feats, labels = self._cv_features_axis.axis_values()
            if not feats:
                QMessageBox.warning(self, "No feature presets",
                                    "Add at least one feature preset to compare.")
                return
            plans = plan_sweep(axis="features", defaults=defaults,
                                axis_features=feats, axis_labels=labels)
        elif axis == "data_subset":
            subsets = self._cv_subset_axis.axis_values()
            if not subsets:
                QMessageBox.warning(self, "No data subsets",
                                    "Define at least one named subset to compare.")
                return
            plans = plan_sweep(axis="data_subset", defaults=defaults,
                                axis_subsets=subsets)
        else:
            QMessageBox.warning(self, "Unknown axis", f"Unknown comparison axis: {axis}")
            return

        if not plans:
            QMessageBox.warning(self, "Nothing to run",
                                "The current configuration produced zero runs.")
            return

        # Confirm large sweeps
        if len(plans) > 8:
            if QMessageBox.question(
                self, "Large sweep",
                f"This will run {len(plans)} validation experiments back-to-back. "
                f"Continue?",
            ) != QMessageBox.StandardButton.Yes:
                return

        self._launch_sweep(plans, axis_label=axis)

    def _launch_sweep(self, plans: List[SweepPlan], *, axis_label: str) -> None:
        self._cv_axis_label = axis_label
        self._cv_completed_rows = []
        self._cv_completed_runs = []
        # Make sure the results panel is the CV one — defensive in case
        # the user pressed Run while in CV mode but the stack got out of
        # sync somehow.
        self._results_stack.setCurrentIndex(self._RESULTS_PAGE_CV)
        self._cv_results.reset_for_sweep(len(plans), axis_label)

        bridge = _SweepProgressBridge()
        worker = _SweepWorker(self._data_dir, plans, bridge)

        # Wire signals — QueuedConnection is implicit because the worker
        # lives on a different thread. The bridge proxies the runner's
        # ProgressReporter callbacks onto these signals.
        bridge.run_started   .connect(self._on_cv_run_started)
        bridge.run_finished  .connect(self._on_cv_run_finished)
        bridge.fold_finished .connect(self._on_cv_fold_finished)
        bridge.log_line      .connect(self._cv_results.append_log)
        bridge.sweep_finished.connect(self._on_cv_sweep_finished)
        bridge.sweep_failed  .connect(self._on_cv_sweep_failed)
        worker.finished      .connect(self._cleanup_cv_worker)

        self._cv_bridge = bridge
        self._cv_worker = worker

        self._btn_run   .setEnabled(False)
        self._btn_cancel.setEnabled(True)
        worker.start()

    @Slot(int, int, str)
    def _on_cv_run_started(self, idx: int, total: int, label: str) -> None:
        self._cv_results.on_run_started(idx, total, label)

    @Slot(int, int, str, object)
    def _on_cv_run_finished(self, idx: int, total: int, label: str, result: Any) -> None:
        agg = result.aggregate() if result is not None else {}
        self._cv_completed_rows.append((label, agg))
        self._cv_completed_runs.append((label, result))
        self._cv_results.show_completed_sweep(self._cv_completed_rows)
        self._cv_results.on_run_finished(idx, total, label, result)

    @Slot(int, int, object)
    def _on_cv_fold_finished(self, idx: int, total: int, fr: FoldResult) -> None:
        self._cv_results.append_log(
            f"  fold {idx}/{total}  ·  {fr.model_type}  ·  "
            f"acc={fr.accuracy:.3f}  f1={fr.macro_f1:.3f}"
        )

    @Slot(object)
    def _on_cv_sweep_finished(self, runs: List[Tuple[str, Any]]) -> None:
        self._cv_results.append_log(
            f"Sweep finished — {len(self._cv_completed_rows)} run(s) total."
        )
        self.sweep_finished.emit(runs)

    @Slot(str)
    def _on_cv_sweep_failed(self, tb: str) -> None:
        log.error("Sweep failed:\n%s", tb)
        last = tb.splitlines()[-1] if tb.splitlines() else tb
        QMessageBox.critical(self, "Sweep failed", last)
        self._cv_results.append_log(tb)

    def _cleanup_cv_worker(self) -> None:
        # Re-enable Run only if we're still in CV mode — switching to a
        # different mode while a sweep was running is unusual but not
        # impossible.
        if self._eval_type_combo.currentData() == EVAL_TYPE_CV:
            self._btn_run.setEnabled(True)
        self._btn_cancel.setEnabled(False)
        self._cv_bridge = None
        # Worker is auto-deleted via QThread.finished handling

    # ──────────────────────────────────────────────────────────────────
    # CV exports
    # ──────────────────────────────────────────────────────────────────

    @Slot(str)
    def _on_cv_export(self, kind: str) -> None:
        if not self._cv_completed_rows:
            return
        if kind == "comparison_csv":
            path, _ = QFileDialog.getSaveFileName(
                self, "Save comparison CSV", "cv_comparison.csv", "CSV (*.csv)")
            if path: self._write_cv_comparison_csv(Path(path))
        elif kind == "per_model_csv":
            path, _ = QFileDialog.getSaveFileName(
                self, "Save per-model CSV", "cv_per_model.csv", "CSV (*.csv)")
            if path: self._write_cv_per_model_csv(Path(path))
        elif kind == "summary_json":
            path, _ = QFileDialog.getSaveFileName(
                self, "Save sweep JSON", "cv_sweep.json", "JSON (*.json)")
            if path: self._write_cv_summary_json(Path(path))
        elif kind == "bundle":
            d = QFileDialog.getExistingDirectory(self, "Save full bundle to folder",
                                                  str(Path.home()))
            if d:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                target = Path(d) / f"cv_sweep_{ts}"
                target.mkdir(parents=True, exist_ok=True)
                self._write_cv_comparison_csv(target / "comparison.csv")
                self._write_cv_per_model_csv(target / "per_model.csv")
                self._write_cv_summary_json(target / "sweep.json")
                QMessageBox.information(self, "Export complete",
                                        f"Saved bundle to:\n{target}")

    def _write_cv_comparison_csv(self, path: Path) -> None:
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["run", "best_model", "n_folds",
                        "accuracy_mean", "accuracy_std",
                        "macro_f1_mean", "macro_f1_std",
                        "train_seconds_mean"])
            for label, agg in self._cv_completed_rows:
                if not agg:
                    w.writerow([label, "", 0, "", "", "", "", ""])
                    continue
                best, m = max(agg.items(), key=lambda kv: kv[1].get("accuracy_mean", 0.0))
                w.writerow([label, best, int(m.get("n_folds", 0)),
                             f"{m.get('accuracy_mean', 0.0):.6f}",
                             f"{m.get('accuracy_std', 0.0):.6f}",
                             f"{m.get('macro_f1_mean', 0.0):.6f}",
                             f"{m.get('macro_f1_std', 0.0):.6f}",
                             f"{m.get('train_seconds_mean', 0.0):.3f}"])

    def _write_cv_per_model_csv(self, path: Path) -> None:
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["run", "model", "n_folds",
                        "accuracy_mean", "accuracy_std",
                        "macro_f1_mean", "macro_f1_std",
                        "train_seconds_mean"])
            for label, agg in self._cv_completed_rows:
                for model, m in sorted((agg or {}).items()):
                    w.writerow([label, model, int(m.get("n_folds", 0)),
                                 f"{m.get('accuracy_mean', 0.0):.6f}",
                                 f"{m.get('accuracy_std', 0.0):.6f}",
                                 f"{m.get('macro_f1_mean', 0.0):.6f}",
                                 f"{m.get('macro_f1_std', 0.0):.6f}",
                                 f"{m.get('train_seconds_mean', 0.0):.3f}"])

    def _write_cv_summary_json(self, path: Path) -> None:
        payload = {
            "axis":       self._cv_axis_label,
            "created_at": datetime.now().isoformat(),
            "data_dir":   str(self._data_dir),
            "runs": [
                {"label": label, "aggregate": agg}
                for label, agg in self._cv_completed_rows
            ],
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2, default=str)


# ═══════════════════════════════════════════════════════════════════════════
# Back-compat alias
# ═══════════════════════════════════════════════════════════════════════════
# The old cross-validation tab module exposed a class called
# ``CrossValidationTab``. Anything still importing that name will get
# the merged tab back instead — opening on the CV mode is still a
# matter of one line:
#
#     tab = CrossValidationTab(self.data_manager)
#     tab._eval_type_combo.setCurrentIndex(3)
#
# but new code should import ``EvaluationTab`` directly.

CrossValidationTab = EvaluationTab