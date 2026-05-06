"""
gui/widgets/evaluation_tab.py  (v2)
───────────────────────────────────
Unified evaluation tab. Replaces both Performance Review and Validation.

What changed in v2
──────────────────
* Subject-grouped **tree picker** with expandable VP_xx headers, tristate
  parent checkboxes, and live "selected" counts. Replaces the v1 flat
  93-row scroll list.
* Settings panels now use **separate group-boxes per concern** (Model /
  Windowing / Preprocessing / Feature ranking) so nothing feels cramped.
  Real spacing between rows; real margins between groups.
* **Plot & data export** from any result: confusion matrix PNG, per-class
  F1 box plot across recordings, raw CSV/JSON dumps. One "Save…" button
  picks a directory and writes everything atomically.
* Game-recording discovery now walks ``<root>/<subject>/<dir>/recording.csv``
  (the actual GameRecorder layout) — v1 only walked one level so the
  Games tab came up empty.
* Confusion-matrix labels are de-duplicated with a `(id N)` suffix so a
  bad gesture_set with two ``Tripod`` entries no longer renders as
  "Tripod, Tripod" with one column silently dropped.
"""

from __future__ import annotations

import csv
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QIcon
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
    QFormLayout, QFrame, QGroupBox, QHBoxLayout, QHeaderView,
    QLabel, QLineEdit, QMenu, QMessageBox, QPushButton, QScrollArea,
    QSizePolicy, QSpinBox, QSplitter, QStackedWidget, QTabWidget,
    QTableWidget, QTableWidgetItem, QTextEdit, QToolButton, QTreeWidget,
    QTreeWidgetItem, QVBoxLayout, QWidget,
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

from playagain_pipeline.evaluation import (
    EvaluationResult, RecordingDescriptor, RecordingKind,
    SessionEvalSettings, GameEvalSettings, UnityEvalSettings,
    discover_sessions, discover_game_recordings, discover_unity_recordings,
    evaluate_sessions, evaluate_games, evaluate_unity,
    evaluate_features_lda,
    TRUTH_RAW, TRUTH_REQUESTED, TRUTH_ACTIVE,
)
from playagain_pipeline.gui.gui_style import apply_app_style
from playagain_pipeline.gui.widgets.busy_overlay import run_blocking

log = logging.getLogger(__name__)


# ── Bright palette (kept in sync with gui_style.py "bright") ───────────────
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
# Small primitives
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
    color, border = ((C_ACCENT, C_ACCENT)  if accent else (C_TEXT, C_BORDER))
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


# ═══════════════════════════════════════════════════════════════════════════
# Subject-grouped tree picker
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
# Concrete pickers (one per mode)
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
# Result detail widgets
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
# Shared results panel
# ═══════════════════════════════════════════════════════════════════════════

class _ResultsView(QWidget):
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
        a_pkg  = self._export_menu.addAction("Save full bundle (folder)…")
        a_cm   = self._export_menu.addAction("Confusion matrix PNG…")
        a_box  = self._export_menu.addAction("Per-class F1 box plot PNG…")
        a_csv  = self._export_menu.addAction("Per-class metrics CSV…")
        a_json = self._export_menu.addAction("Result JSON…")
        a_pkg .triggered.connect(self._on_export_bundle)
        a_cm  .triggered.connect(lambda: self._on_export_one("confusion_png"))
        a_box .triggered.connect(lambda: self._on_export_one("boxplot_png"))
        a_csv .triggered.connect(lambda: self._on_export_one("metrics_csv"))
        a_json.triggered.connect(lambda: self._on_export_one("result_json"))
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
        self._save_feature_csv(target / "per_feature_accuracy.csv")

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


# ═══════════════════════════════════════════════════════════════════════════
# Mode panels — picker on the left, grouped settings on the right
# ═══════════════════════════════════════════════════════════════════════════

def _model_dirs(data_dir: Path) -> List[str]:
    """Return saved model names from <data_dir>/models, newest first."""
    md = Path(data_dir) / "models"
    if not md.exists():
        return []
    children = [c for c in md.iterdir() if c.is_dir() and not c.name.startswith("_")]
    children.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return [c.name for c in children]


def _form_layout() -> QFormLayout:
    """A consistent form layout used inside every settings group-box."""
    f = QFormLayout()
    f.setHorizontalSpacing(12)
    f.setVerticalSpacing(8)
    f.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    f.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
    return f


def _settings_scroll_host(group_widgets: List[QWidget],
                           run_button: QWidget) -> QWidget:
    """
    Build the right-hand settings column: a scroll area holding a stack
    of group-boxes, plus the Run button pinned to the bottom outside
    the scroll area.

    This decoupling matters: when the window is short, the scroll keeps
    the Run button visible at the bottom even when the settings overflow.
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

    col_lay.addWidget(run_button)
    return column


# ── Sessions panel ─────────────────────────────────────────────────────────

class _SessionsPanel(QWidget):
    """Source picker + grouped settings + Run button for the Sessions mode."""

    run_clicked      = Signal(SessionEvalSettings, list)
    features_clicked = Signal(list, list)

    def __init__(self, data_dir: Path, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._data_dir = Path(data_dir)

        outer = QHBoxLayout(self); outer.setContentsMargins(0, 0, 0, 0); outer.setSpacing(14)

        # Left: picker
        self._picker = _SessionPicker(self._data_dir)
        outer.addWidget(self._picker, 5)

        # ── Right: groups
        # Group: Model
        g_model = _styled_group("Model")
        gm_lay = _form_layout()
        self._model_combo = QComboBox(); self._model_combo.setMinimumWidth(280)
        self._refresh_models()
        ref_btn = _ghost_button("⟳"); ref_btn.setToolTip("Re-scan data/models")
        ref_btn.clicked.connect(self._refresh_models)
        model_row = QHBoxLayout(); model_row.setContentsMargins(0,0,0,0); model_row.setSpacing(6)
        model_row.addWidget(self._model_combo, 1); model_row.addWidget(ref_btn)
        mw = QWidget(); mw.setLayout(model_row)
        gm_lay.addRow("Saved model:", mw)
        g_model.setLayout(gm_lay)

        # Group: Windowing
        g_win = _styled_group("Windowing")
        gw_lay = _form_layout()
        self._win_spin = QSpinBox(); self._win_spin.setRange(0, 5000); self._win_spin.setSingleStep(50)
        self._win_spin.setSpecialValueText("(use model)"); self._win_spin.setSuffix(" ms")
        gw_lay.addRow("Window length:", self._win_spin)
        self._stride_spin = QSpinBox(); self._stride_spin.setRange(1, 5000)
        self._stride_spin.setValue(50); self._stride_spin.setSuffix(" ms")
        gw_lay.addRow("Window stride:", self._stride_spin)
        g_win.setLayout(gw_lay)

        # Group: Preprocessing
        g_pre = _styled_group("Preprocessing")
        gp_lay = _form_layout()
        self._bad_ch_combo = QComboBox()
        self._bad_ch_combo.addItem("Interpolate", "interpolate")
        self._bad_ch_combo.addItem("Zero",        "zero")
        gp_lay.addRow("Bad channels:", self._bad_ch_combo)
        self._rotation_chk = QCheckBox("Apply per-session rotation")
        self._rotation_chk.setChecked(True)
        gp_lay.addRow("", self._rotation_chk)
        self._invalid_chk = QCheckBox("Include trials marked invalid")
        gp_lay.addRow("", self._invalid_chk)
        self._per_session_chk = QCheckBox("Show per-session breakdown in notes")
        self._per_session_chk.setChecked(True)
        gp_lay.addRow("", self._per_session_chk)
        g_pre.setLayout(gp_lay)

        # Group: Feature ranking
        g_feat = _styled_group("Optional · feature ranking")
        gf = QVBoxLayout(); gf.setSpacing(8)
        info = QLabel("Score each feature individually with an LDA, in addition to "
                      "running the saved model. Useful as a quick 'which feature carries "
                      "the gesture signal best on this dataset' check.")
        info.setStyleSheet(f"color:{C_MUTED}; font-size:10px;")
        info.setWordWrap(True); gf.addWidget(info)
        feat_grid = QHBoxLayout(); feat_grid.setSpacing(10)
        self._feat_choices: List[QCheckBox] = []
        for f in ("rms", "mav", "wl", "zc", "ssc", "var", "iemg"):
            cb = QCheckBox(f); cb.setChecked(f in {"rms", "mav", "wl"})
            self._feat_choices.append(cb); feat_grid.addWidget(cb)
        feat_grid.addStretch(1)
        gf.addLayout(feat_grid)
        feat_btn_row = QHBoxLayout(); feat_btn_row.addStretch(1)
        self._btn_features = _ghost_button("Rank features (LDA)", accent=True)
        self._btn_features.clicked.connect(self._on_features_clicked)
        feat_btn_row.addWidget(self._btn_features)
        gf.addLayout(feat_btn_row)
        g_feat.setLayout(gf)

        self._btn_run = _primary_button("▶  Run evaluation")
        self._btn_run.clicked.connect(self._on_run_clicked)

        right = _settings_scroll_host([g_model, g_win, g_pre, g_feat], self._btn_run)
        outer.addWidget(right, 6)

    # ──────────────────────────────────────────────────────────────────

    def _refresh_models(self) -> None:
        prev = self._model_combo.currentText()
        self._model_combo.clear()
        names = _model_dirs(self._data_dir)
        if not names:
            self._model_combo.addItem("(no models found in data/models)", None)
            self._model_combo.setEnabled(False)
        else:
            self._model_combo.setEnabled(True)
            for n in names:
                self._model_combo.addItem(n, n)
            if prev and prev in names:
                self._model_combo.setCurrentText(prev)

    def refresh(self) -> None:
        self._picker.refresh(); self._refresh_models()

    def _settings_from_ui(self) -> Optional[SessionEvalSettings]:
        model_name = self._model_combo.currentData()
        if not model_name:
            QMessageBox.warning(self, "No model",
                                "No saved model is available — train a model first.")
            return None
        return SessionEvalSettings(
            model_name=str(model_name),
            window_size_ms=self._win_spin.value() or None,
            window_stride_ms=int(self._stride_spin.value()),
            apply_rotation=self._rotation_chk.isChecked(),
            bad_channel_mode=self._bad_ch_combo.currentData(),
            include_invalid=self._invalid_chk.isChecked(),
            per_session_breakdown=self._per_session_chk.isChecked(),
        )

    def _on_run_clicked(self) -> None:
        recs = self._picker.selected()
        if not recs:
            QMessageBox.information(self, "No sessions",
                                    "Select at least one session on the left.")
            return
        s = self._settings_from_ui()
        if s is None:
            return
        self.run_clicked.emit(s, recs)

    def _on_features_clicked(self) -> None:
        recs = self._picker.selected()
        if not recs:
            QMessageBox.information(self, "No sessions",
                                    "Select at least one session on the left.")
            return
        feats = [c.text() for c in self._feat_choices if c.isChecked()]
        if not feats:
            QMessageBox.information(self, "No features",
                                    "Select at least one feature to rank.")
            return
        self.features_clicked.emit(feats, recs)


# ── Game recordings panel ──────────────────────────────────────────────────

class _GamesPanel(QWidget):

    run_clicked = Signal(GameEvalSettings, list)

    def __init__(self, data_dir: Path, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._data_dir = Path(data_dir)

        outer = QHBoxLayout(self); outer.setContentsMargins(0, 0, 0, 0); outer.setSpacing(14)

        self._picker = _GamePicker(self._data_dir)
        outer.addWidget(self._picker, 5)

        # Group: source mode
        g_src = _styled_group("Prediction source")
        gs = _form_layout()
        self._mode_combo = QComboBox()
        self._mode_combo.addItem("Logged predictions  (use Pred* columns)", "logged")
        self._mode_combo.addItem("Replay model on EMG   (re-run a saved model)", "replay")
        self._mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        gs.addRow("Source:", self._mode_combo)

        self._truth_combo = QComboBox()
        self._truth_combo.addItem("RawGroundTruth  (multi-class — preferred)", TRUTH_RAW)
        self._truth_combo.addItem("GroundTruthActive  (binary)",                TRUTH_ACTIVE)
        self._truth_combo.addItem("RequestedGesture  (multi-class fallback)",   TRUTH_REQUESTED)
        gs.addRow("Ground truth:", self._truth_combo)
        g_src.setLayout(gs)

        # Group: Filtering
        g_filter = _styled_group("Filtering")
        gf = _form_layout()
        self._drop_chk = QCheckBox("Drop frames where the game wasn't asking for a gesture")
        self._drop_chk.setChecked(True)
        gf.addRow("", self._drop_chk)
        self._min_conf_chk = QCheckBox("Filter by minimum confidence")
        self._min_conf_spin = QDoubleSpinBox()
        self._min_conf_spin.setRange(0.0, 1.0); self._min_conf_spin.setSingleStep(0.05)
        self._min_conf_spin.setValue(0.5); self._min_conf_spin.setEnabled(False)
        self._min_conf_chk.toggled.connect(self._min_conf_spin.setEnabled)
        conf_row = QHBoxLayout(); conf_row.setContentsMargins(0,0,0,0); conf_row.setSpacing(8)
        conf_row.addWidget(self._min_conf_chk); conf_row.addWidget(self._min_conf_spin); conf_row.addStretch(1)
        cw = QWidget(); cw.setLayout(conf_row)
        gf.addRow("", cw)
        self._per_chk = QCheckBox("Show per-recording breakdown in notes")
        self._per_chk.setChecked(True)
        gf.addRow("", self._per_chk)
        g_filter.setLayout(gf)

        # Group: replay options (model + window)
        self._g_replay = _styled_group("Replay options")
        gr = _form_layout()
        self._model_combo = QComboBox(); self._refresh_models()
        ref_btn = _ghost_button("⟳"); ref_btn.clicked.connect(self._refresh_models)
        mr = QHBoxLayout(); mr.setContentsMargins(0,0,0,0); mr.setSpacing(6)
        mr.addWidget(self._model_combo, 1); mr.addWidget(ref_btn)
        mw = QWidget(); mw.setLayout(mr)
        gr.addRow("Saved model:", mw)
        self._win_spin = QSpinBox(); self._win_spin.setRange(0, 5000); self._win_spin.setSingleStep(50)
        self._win_spin.setSpecialValueText("(use model)"); self._win_spin.setSuffix(" ms")
        gr.addRow("Window length:", self._win_spin)
        self._stride_spin = QSpinBox(); self._stride_spin.setRange(1, 5000); self._stride_spin.setValue(50)
        self._stride_spin.setSuffix(" ms")
        gr.addRow("Window stride:", self._stride_spin)
        self._g_replay.setLayout(gr)

        self._btn_run = _primary_button("▶  Run evaluation")
        self._btn_run.clicked.connect(self._on_run_clicked)

        right = _settings_scroll_host([g_src, g_filter, self._g_replay], self._btn_run)
        outer.addWidget(right, 6)

        self._on_mode_changed()

    # ──────────────────────────────────────────────────────────────────

    def _refresh_models(self) -> None:
        prev = self._model_combo.currentText()
        self._model_combo.clear()
        names = _model_dirs(self._data_dir)
        if not names:
            self._model_combo.addItem("(no models found)", None)
            self._model_combo.setEnabled(False)
        else:
            self._model_combo.setEnabled(True)
            for n in names:
                self._model_combo.addItem(n, n)
            if prev and prev in names:
                self._model_combo.setCurrentText(prev)

    def refresh(self) -> None:
        self._picker.refresh(); self._refresh_models()

    def _on_mode_changed(self) -> None:
        is_replay = (self._mode_combo.currentData() == "replay")
        self._g_replay.setVisible(is_replay)

    def _settings_from_ui(self) -> Optional[GameEvalSettings]:
        mode = self._mode_combo.currentData()
        s = GameEvalSettings(
            mode=mode,
            truth_source=self._truth_combo.currentData(),
            drop_inactive_truth_frames=self._drop_chk.isChecked(),
            min_confidence=(self._min_conf_spin.value() if self._min_conf_chk.isChecked() else None),
            per_recording_breakdown=self._per_chk.isChecked(),
        )
        if mode == "replay":
            model_name = self._model_combo.currentData()
            if not model_name:
                QMessageBox.warning(self, "No model",
                                    "Replay mode requires a saved model.")
                return None
            s.model_name = str(model_name)
            s.window_size_ms = self._win_spin.value() or None
            s.window_stride_ms = int(self._stride_spin.value())
        return s

    def _on_run_clicked(self) -> None:
        recs = self._picker.selected()
        if not recs:
            QMessageBox.information(self, "No recordings",
                                    "Select at least one game recording on the left.")
            return
        s = self._settings_from_ui()
        if s is None:
            return
        self.run_clicked.emit(s, recs)


# ── Unity panel ────────────────────────────────────────────────────────────

class _UnityPanel(QWidget):

    run_clicked = Signal(UnityEvalSettings, list)

    def __init__(self, data_dir: Path, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._data_dir = Path(data_dir)

        outer = QHBoxLayout(self); outer.setContentsMargins(0, 0, 0, 0); outer.setSpacing(14)

        self._picker = _UnityPicker(self._data_dir)
        outer.addWidget(self._picker, 5)

        # Group: about
        g_about = _styled_group("Threshold model")
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
        g_about.setLayout(ga)

        # Group: objective
        g_obj = _styled_group("Threshold objective")
        go = _form_layout()
        self._objective_combo = QComboBox()
        self._objective_combo.addItem("Maximise F1",         "f1")
        self._objective_combo.addItem("Maximise Youden's J", "youden")
        self._objective_combo.addItem("Maximise accuracy",   "accuracy")
        go.addRow("Objective:", self._objective_combo)
        self._fixed_chk = QCheckBox("Use a fixed threshold instead")
        self._fixed_spin = QDoubleSpinBox()
        self._fixed_spin.setRange(0.0, 1e6); self._fixed_spin.setDecimals(8)
        self._fixed_spin.setSingleStep(1e-5); self._fixed_spin.setValue(1e-4)
        self._fixed_spin.setEnabled(False)
        self._fixed_chk.toggled.connect(self._fixed_spin.setEnabled)
        fixed_row = QHBoxLayout(); fixed_row.setContentsMargins(0,0,0,0); fixed_row.setSpacing(8)
        fixed_row.addWidget(self._fixed_chk); fixed_row.addWidget(self._fixed_spin); fixed_row.addStretch(1)
        fw = QWidget(); fw.setLayout(fixed_row)
        go.addRow("Fixed:", fw)
        self._n_thresh_spin = QSpinBox(); self._n_thresh_spin.setRange(20, 1000); self._n_thresh_spin.setValue(200)
        go.addRow("Sweep points:", self._n_thresh_spin)
        g_obj.setLayout(go)

        # Group: computed RMS settings
        g_rms = _styled_group("Computed-RMS  (converted sessions only)")
        gr = _form_layout()
        rms_info = QLabel("These settings only apply when the source is a "
                           "converted Unity session — for raw Unity CSVs the "
                           "logged RMS column is used directly.")
        rms_info.setWordWrap(True); rms_info.setStyleSheet(f"color:{C_MUTED}; font-size:10px;")
        gr.addRow(rms_info)
        self._rms_win_spin = QSpinBox(); self._rms_win_spin.setRange(1, 5000); self._rms_win_spin.setValue(100)
        self._rms_win_spin.setSuffix(" ms")
        gr.addRow("RMS window:", self._rms_win_spin)
        self._rms_stride_spin = QSpinBox(); self._rms_stride_spin.setRange(1, 5000); self._rms_stride_spin.setValue(25)
        self._rms_stride_spin.setSuffix(" ms")
        gr.addRow("RMS stride:", self._rms_stride_spin)
        self._active_label_spin = QSpinBox(); self._active_label_spin.setRange(0, 100); self._active_label_spin.setValue(1)
        gr.addRow("Active label id:", self._active_label_spin)
        g_rms.setLayout(gr)

        self._btn_run = _primary_button("▶  Run evaluation")
        self._btn_run.clicked.connect(self._on_run_clicked)

        right = _settings_scroll_host([g_about, g_obj, g_rms], self._btn_run)
        outer.addWidget(right, 6)

    # ──────────────────────────────────────────────────────────────────

    def refresh(self) -> None:
        self._picker.refresh()

    def _settings_from_ui(self) -> UnityEvalSettings:
        return UnityEvalSettings(
            active_label=int(self._active_label_spin.value()),
            rms_window_ms=int(self._rms_win_spin.value()),
            rms_stride_ms=int(self._rms_stride_spin.value()),
            n_thresholds=int(self._n_thresh_spin.value()),
            objective=str(self._objective_combo.currentData()),
            fixed_threshold=(self._fixed_spin.value() if self._fixed_chk.isChecked() else None),
        )

    def _on_run_clicked(self) -> None:
        recs = self._picker.selected()
        if not recs:
            QMessageBox.information(self, "No recordings",
                                    "Select at least one Unity recording on the left.")
            return
        self.run_clicked.emit(self._settings_from_ui(), recs)


# ═══════════════════════════════════════════════════════════════════════════
# Top-level tab
# ═══════════════════════════════════════════════════════════════════════════

class EvaluationTab(QWidget):
    """
    Public widget for the redesigned Evaluation tab.

    Construct with the project's ``DataManager`` (or any object that exposes
    ``data_dir``), then add it to the main window like any other tab.
    """

    result_ready = Signal(object)   # EvaluationResult

    def __init__(self, data_manager: Any, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._data_dir: Path = Path(getattr(data_manager, "data_dir", data_manager))
        if not self._data_dir.exists():
            log.warning("Data directory does not exist: %s", self._data_dir)

        apply_app_style(self, theme="bright")

        outer = QVBoxLayout(self); outer.setContentsMargins(12, 12, 12, 12); outer.setSpacing(12)

        # ── Header banner
        header = QFrame()
        header.setStyleSheet(
            f"QFrame {{ background: qlineargradient(x1:0,y1:0,x2:1,y2:0,"
            f"  stop:0 {C_PANEL}, stop:1 {C_CARD});"
            f"  border:1px solid {C_BORDER}; border-radius:10px; }}"
        )
        h_lay = QHBoxLayout(header); h_lay.setContentsMargins(18, 12, 18, 12); h_lay.setSpacing(12)
        title = QLabel("Evaluation"); f = QFont(); f.setPointSize(16); f.setBold(True); title.setFont(f)
        title.setStyleSheet(f"color:{C_ACCENT};")
        h_lay.addWidget(title)
        sub = QLabel("Sessions  ·  game recordings  ·  Unity threshold")
        sub.setStyleSheet(f"color:{C_MUTED}; font-size:11px;")
        h_lay.addWidget(sub)
        h_lay.addStretch(1)
        self._data_dir_lbl = QLabel(str(self._data_dir))
        self._data_dir_lbl.setStyleSheet(f"color:{C_MUTED}; font-size:10px; font-family:'Menlo','DejaVu Sans Mono',monospace;")
        self._data_dir_lbl.setToolTip("Active data directory")
        h_lay.addWidget(self._data_dir_lbl)
        outer.addWidget(header)

        # ── Mode tabs
        self._mode_tabs = QTabWidget(); self._mode_tabs.setDocumentMode(True)
        self._sessions = _SessionsPanel(self._data_dir)
        self._games    = _GamesPanel(self._data_dir)
        self._unity    = _UnityPanel(self._data_dir)
        self._mode_tabs.addTab(self._sessions, "  Sessions  ")
        self._mode_tabs.addTab(self._games,    "  Game recordings  ")
        self._mode_tabs.addTab(self._unity,    "  Unity recordings  ")

        # ── Results panel
        self._results = _ResultsView()

        split = QSplitter(Qt.Orientation.Vertical)
        split.addWidget(self._mode_tabs)
        split.addWidget(self._results)
        split.setStretchFactor(0, 1); split.setStretchFactor(1, 1)
        split.setSizes([460, 600])
        split.setHandleWidth(6)
        outer.addWidget(split, 1)

        # ── Wiring
        self._sessions.run_clicked     .connect(self._run_sessions)
        self._sessions.features_clicked.connect(self._run_features)
        self._games   .run_clicked     .connect(self._run_games)
        self._unity   .run_clicked     .connect(self._run_unity)

        self._worker = None    # held to keep run_blocking's QThread alive

    # ── public API ────────────────────────────────────────────────────

    def refresh(self) -> None:
        self._sessions.refresh(); self._games.refresh(); self._unity.refresh()

    # ── run handlers ──────────────────────────────────────────────────

    def _run_sessions(self, settings, recordings):
        data_dir = self._data_dir
        def task() -> EvaluationResult:
            return evaluate_sessions(data_dir, recordings, settings)
        self._launch(task, "Running session evaluation…")

    def _run_features(self, features, recordings):
        data_dir = self._data_dir
        def task() -> EvaluationResult:
            return evaluate_features_lda(data_dir, recordings, features)
        self._launch(task, "Ranking features…")

    def _run_games(self, settings, recordings):
        data_dir = self._data_dir
        def task() -> EvaluationResult:
            return evaluate_games(data_dir, recordings, settings)
        self._launch(task, "Evaluating game recordings…")

    def _run_unity(self, settings, recordings):
        def task() -> EvaluationResult:
            return evaluate_unity(recordings, settings)
        self._launch(task, "Sweeping RMS thresholds…")

    def _launch(self, task, label: str) -> None:
        self._worker = run_blocking(
            parent_widget=self,
            fn=task,
            on_done=self._on_result_ready,
            on_error=self._on_error,
            label=label,
        )

    # ── result + error handlers ───────────────────────────────────────

    def _on_result_ready(self, result: EvaluationResult) -> None:
        self._results.show_result(result)
        self.result_ready.emit(result)

    def _on_error(self, tb: str) -> None:
        log.error("Evaluation failed:\n%s", tb)
        last_line = next((ln for ln in reversed(tb.splitlines()) if ln.strip()), "Unknown error")
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Icon.Critical)
        msg.setWindowTitle("Evaluation failed")
        msg.setText(last_line)
        msg.setDetailedText(tb)
        msg.setStandardButtons(QMessageBox.StandardButton.Ok)
        msg.exec()