"""
rotation_validation_dialog.py
─────────────────────────────
Batch rotation-detection validator with embedded matplotlib plots.

Why this dialog exists
──────────────────────
``CalibrationDialog`` validates *one* calibration in isolation. That answers
the question "is this single calibration usable?" but not "how does the
rotation-detection pipeline perform across my recorded sessions?".

This dialog answers the second question. You pick subjects and sessions,
optionally restrict to a chosen subset of gestures, and the batch validator
runs all three checks across every session. The results are presented as:

  • A summary card                — counts and pass-rates at a glance.
  • A per-session table           — sortable, every check spelled out.
  • A confusion-matrix heatmap    — where the held-out 1-NN classifier
                                    fails. Aggregated across all sessions.
  • Per-gesture box plot          — accuracy distribution per gesture
                                    with one point per session.
  • Distribution plots            — offset histogram, confidence histogram,
                                    self-consistency drift, symmetry ratio.
  • A raw log                     — for debugging.

Why gesture filtering matters
─────────────────────────────
Sessions in the wild often disagree on their gesture set — one participant
recorded "tripod" in five sessions and forgot it in the sixth. Naïvely
aggregating across that session pulls down the confusion matrix and the
box plot for "tripod" because the sixth session contributes zero correct
predictions for a gesture it never recorded. The dialog detects the gesture
intersection, lets you pick which gestures count, and (in strict mode) drops
sessions that don't have all of the chosen gestures so every cell of the
aggregate is built from comparable data.

Plot style
──────────
Matplotlib is loaded lazily so a missing matplotlib doesn't block the table
and log views. The plots use a Qt-native canvas (``backend_qtagg``) so they
behave correctly inside a modal dialog.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PySide6.QtCore import Qt, QObject, QThread, Signal, Slot
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Matplotlib loaded lazily so a missing install doesn't kill the dialog
# ---------------------------------------------------------------------------

def _load_matplotlib():
    """Return ``(FigureCanvas, Figure)`` or ``(None, None)`` if unavailable."""
    try:
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure
        return FigureCanvasQTAgg, Figure
    except Exception as e:
        log.warning("matplotlib unavailable for batch validation plots: %s", e)
        return None, None


# ---------------------------------------------------------------------------
# Worker thread
# ---------------------------------------------------------------------------

class _BatchWorker(QObject):
    """
    Runs ``BatchValidator.run`` off the GUI thread.

    Loading sessions is I/O-bound and validating is CPU-bound; either one
    is happy to block for several seconds, which would freeze the dialog
    if we did it on the GUI thread.
    """

    progress = Signal(int, int, str)        # i, n, label
    finished = Signal(object)               # BatchReport
    failed = Signal(str)

    def __init__(
        self,
        calibrator,
        data_manager,
        selections: List[Tuple[str, str]],
        gesture_filter: Optional[List[str]],
        strict: bool,
        validator_kwargs: dict,
    ):
        super().__init__()
        self._calibrator = calibrator
        self._data_manager = data_manager
        self._selections = selections
        self._gesture_filter = gesture_filter
        self._strict = strict
        self._validator_kwargs = validator_kwargs

    @Slot()
    def run(self) -> None:
        try:
            from playagain_pipeline.calibration.calibration_validation import (
                BatchValidator, gestures_in_session,
            )
        except Exception as e:
            self.failed.emit(f"Could not import validator: {e}")
            return

        # Load each session and (in strict mode) drop sessions that don't
        # have every gesture in the filter.
        loaded: List[Tuple[str, str, Any]] = []
        skipped_strict: Dict[str, str] = {}
        for subject, session_id in self._selections:
            try:
                session = self._data_manager.load_session(subject, session_id)
            except Exception as e:
                skipped_strict[f"{subject}/{session_id}"] = f"load failed: {e}"
                continue
            if self._strict and self._gesture_filter:
                have = gestures_in_session(session)
                missing = [g for g in self._gesture_filter if g not in have]
                if missing:
                    skipped_strict[f"{subject}/{session_id}"] = (
                        f"missing gestures: {', '.join(missing)}"
                    )
                    continue
            loaded.append((subject, session_id, session))

        try:
            batch = BatchValidator(
                self._calibrator, validator_kwargs=self._validator_kwargs,
            )
            report = batch.run(
                loaded,
                gesture_filter=self._gesture_filter,
                progress_cb=lambda i, n, label: self.progress.emit(i, n, label),
            )
        except Exception as e:
            log.exception("Batch worker crashed")
            self.failed.emit(str(e))
            return

        # Merge in the strict-mode skip reasons so the user sees them.
        for k, v in skipped_strict.items():
            report.skipped.setdefault(k, v)
        self.finished.emit(report)


# ---------------------------------------------------------------------------
# Plot canvases
# ---------------------------------------------------------------------------

class _ConfusionCanvas(QWidget):
    """Heatmap of held-out true-vs-predicted gesture counts."""

    def __init__(self, parent=None):
        super().__init__(parent)
        Canvas, Figure = _load_matplotlib()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._available = Canvas is not None
        if self._available:
            self._fig = Figure(figsize=(5.5, 5), tight_layout=True)
            self._canvas = Canvas(self._fig)
            layout.addWidget(self._canvas)
        else:
            layout.addWidget(QLabel("matplotlib not available — install it to see plots."))

    def update_from(self, details, normalize: bool = True):
        if not self._available:
            return
        self._fig.clear()
        labels, m = details.confusion()
        ax = self._fig.add_subplot(111)
        if not labels or m.size == 0:
            ax.text(0.5, 0.5, "No held-out predictions yet.",
                    ha="center", va="center", transform=ax.transAxes)
            self._canvas.draw_idle()
            return
        display = m.astype(float)
        if normalize:
            row_sums = display.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            display = display / row_sums
        im = ax.imshow(display, cmap="Blues", vmin=0,
                       vmax=1.0 if normalize else max(1, display.max()))
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Held-out confusion (row-normalized)" if normalize
                     else "Held-out confusion (counts)")
        # Annotate cells. Pick text colour for contrast.
        thresh = (display.max() if not normalize else 1.0) * 0.5
        for i in range(len(labels)):
            for j in range(len(labels)):
                val = display[i, j]
                if normalize:
                    txt = f"{val:.0%}"
                else:
                    txt = str(int(val))
                ax.text(j, i, txt, ha="center", va="center",
                        color="white" if val > thresh else "#1f2937",
                        fontsize=9)
        self._fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        self._canvas.draw_idle()

    def figure(self):
        return getattr(self, "_fig", None)


class _BoxPlotCanvas(QWidget):
    """Per-gesture accuracy distribution across sessions."""

    def __init__(self, parent=None):
        super().__init__(parent)
        Canvas, Figure = _load_matplotlib()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._available = Canvas is not None
        if self._available:
            self._fig = Figure(figsize=(6, 4), tight_layout=True)
            self._canvas = Canvas(self._fig)
            layout.addWidget(self._canvas)
        else:
            layout.addWidget(QLabel("matplotlib not available — install it to see plots."))

    def update_from(self, per_gesture: Dict[str, List[float]]):
        if not self._available:
            return
        self._fig.clear()
        ax = self._fig.add_subplot(111)
        if not per_gesture:
            ax.text(0.5, 0.5, "No per-gesture data — run validation first.",
                    ha="center", va="center", transform=ax.transAxes)
            self._canvas.draw_idle()
            return
        labels = sorted(per_gesture.keys())
        data = [per_gesture[g] for g in labels]
        bp = ax.boxplot(
            data, labels=labels, showmeans=True, meanline=False,
            patch_artist=True, widths=0.55,
        )
        for patch in bp["boxes"]:
            patch.set_facecolor("#bfdbfe")
            patch.set_edgecolor("#1d4ed8")
        for line in bp["medians"]:
            line.set_color("#1e3a8a")
        # Overlay points so users can see the actual sessions.
        import numpy as np
        for i, vals in enumerate(data, start=1):
            if not vals:
                continue
            jitter = np.random.uniform(-0.07, 0.07, size=len(vals))
            ax.scatter(np.full(len(vals), i) + jitter, vals,
                       s=18, color="#1d4ed8", alpha=0.55, edgecolor="white",
                       linewidths=0.5, zorder=3)
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel("Held-out accuracy")
        ax.set_title("Per-gesture accuracy distribution (one point = one session)")
        ax.axhline(0.7, color="#dc2626", linestyle="--", linewidth=1, alpha=0.7,
                   label="70% threshold")
        ax.legend(loc="lower right", fontsize=8)
        for label in ax.get_xticklabels():
            label.set_rotation(30)
            label.set_horizontalalignment("right")
        self._canvas.draw_idle()

    def figure(self):
        return getattr(self, "_fig", None)


class _DistributionsCanvas(QWidget):
    """Four small panels: offsets, confidence, drift, symmetry ratio."""

    def __init__(self, parent=None):
        super().__init__(parent)
        Canvas, Figure = _load_matplotlib()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._available = Canvas is not None
        if self._available:
            self._fig = Figure(figsize=(7.5, 5.5), tight_layout=True)
            self._canvas = Canvas(self._fig)
            layout.addWidget(self._canvas)
        else:
            layout.addWidget(QLabel("matplotlib not available — install it to see plots."))

    def update_from(self, batch):
        if not self._available:
            return
        import numpy as np
        self._fig.clear()
        gs = self._fig.add_gridspec(2, 2)

        # Offset distribution
        ax1 = self._fig.add_subplot(gs[0, 0])
        offsets = batch.offset_distribution()
        if offsets:
            counts = Counter(offsets)
            xs = sorted(counts)
            ax1.bar(xs, [counts[x] for x in xs], color="#0ea5e9",
                    edgecolor="#0369a1")
            ax1.set_xlabel("Rotation offset (sectors)")
            ax1.set_ylabel("# sessions")
        ax1.set_title("Offset across sessions")

        # Confidence distribution
        ax2 = self._fig.add_subplot(gs[0, 1])
        confs = batch.confidence_distribution()
        if confs:
            ax2.hist(confs, bins=min(20, max(5, len(confs))),
                     color="#10b981", edgecolor="#065f46")
            ax2.set_xlabel("Calibrator confidence")
            ax2.set_ylabel("# sessions")
        ax2.set_title("Confidence (xcorr peak z-score)")

        # Self-consistency drift across sessions
        ax3 = self._fig.add_subplot(gs[1, 0])
        drifts = []
        for r in batch.reports:
            sc = r.get_check("self_consistency")
            if sc is not None:
                drifts.append(sc.score)
        if drifts:
            ax3.hist(drifts, bins=range(0, max(2, int(max(drifts)) + 2)),
                     color="#a78bfa", edgecolor="#5b21b6")
            ax3.set_xlabel("Self-consistency drift (sectors)")
            ax3.set_ylabel("# sessions")
        ax3.set_title("Self-consistency drift")

        # Symmetry ratio
        ax4 = self._fig.add_subplot(gs[1, 1])
        ratios = []
        for r in batch.reports:
            sym = r.get_check("symmetry")
            if sym is not None and sym.score != float("inf"):
                ratios.append(min(sym.score, 5.0))
        if ratios:
            ax4.hist(ratios, bins=min(20, max(5, len(ratios))),
                     color="#f59e0b", edgecolor="#92400e")
            ax4.axvline(1.20, color="#dc2626", linestyle="--", linewidth=1,
                        label="threshold 1.20")
            ax4.set_xlabel("Symmetry ratio (peak / antipode)")
            ax4.set_ylabel("# sessions")
            ax4.legend(fontsize=8)
        ax4.set_title("Symmetry ratio")

        self._canvas.draw_idle()

    def figure(self):
        return getattr(self, "_fig", None)


# ---------------------------------------------------------------------------
# The dialog itself
# ---------------------------------------------------------------------------

class RotationValidationDialog(QDialog):
    """
    Modeless dialog for batch rotation validation across recorded sessions.

    Open it from the main window or from the Validate tab in
    ``CalibrationDialog``. Requires a ``DataManager`` so it can list and
    load sessions.
    """

    def __init__(self, calibrator, data_manager, parent=None):
        super().__init__(parent)
        self.calibrator = calibrator
        self.data_manager = data_manager
        self._batch_report = None
        self._worker_thread: Optional[QThread] = None
        self._worker: Optional[_BatchWorker] = None

        self.setWindowTitle("Rotation Detection — Batch Validation")
        self.setMinimumSize(960, 660)
        self.resize(1100, 720)
        self.setModal(False)

        self._setup_ui()
        self._refresh_subjects()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # Header
        title = QLabel("Rotation detection — batch validation")
        f = title.font()
        f.setPointSize(13); f.setBold(True)
        title.setFont(f)
        root.addWidget(title)

        info = QLabel(
            "Validate the rotation-detection pipeline across many sessions at "
            "once. Pick sessions, choose which gestures count, and look at the "
            "aggregated results."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color:#475569;")
        root.addWidget(info)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_selection_panel())
        splitter.addWidget(self._build_results_panel())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([340, 760])
        root.addWidget(splitter, 1)

        # Bottom row
        bottom = QHBoxLayout()
        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.progress.setFormat("%v / %m sessions")
        bottom.addWidget(self.progress, 1)

        self.export_btn = QPushButton("Export results…")
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self._on_export)
        bottom.addWidget(self.export_btn)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        bottom.addWidget(close_btn)
        root.addLayout(bottom)

    def _build_selection_panel(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)

        # Subject list
        subj_box = QGroupBox("Subjects")
        sb = QVBoxLayout(subj_box)
        self.subject_list = QListWidget()
        self.subject_list.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.subject_list.itemChanged.connect(self._on_subject_toggled)
        sb.addWidget(self.subject_list)
        srow = QHBoxLayout()
        all_subj = QPushButton("All")
        all_subj.clicked.connect(lambda: self._set_all_checked(self.subject_list, True))
        none_subj = QPushButton("None")
        none_subj.clicked.connect(lambda: self._set_all_checked(self.subject_list, False))
        srow.addWidget(all_subj); srow.addWidget(none_subj); srow.addStretch()
        sb.addLayout(srow)
        lay.addWidget(subj_box, 1)

        # Session list
        sess_box = QGroupBox("Sessions")
        seb = QVBoxLayout(sess_box)
        self.session_list = QListWidget()
        self.session_list.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.session_list.itemChanged.connect(self._on_session_toggled)
        seb.addWidget(self.session_list)
        sesrow = QHBoxLayout()
        all_sess = QPushButton("All")
        all_sess.clicked.connect(lambda: self._set_all_checked(self.session_list, True))
        none_sess = QPushButton("None")
        none_sess.clicked.connect(lambda: self._set_all_checked(self.session_list, False))
        refresh_sess = QPushButton("⟳ Refresh")
        refresh_sess.clicked.connect(self._refresh_subjects)
        sesrow.addWidget(all_sess); sesrow.addWidget(none_sess)
        sesrow.addWidget(refresh_sess); sesrow.addStretch()
        seb.addLayout(sesrow)
        lay.addWidget(sess_box, 1)

        # Gesture filter
        gest_box = QGroupBox("Gestures")
        gb = QVBoxLayout(gest_box)
        self.gesture_info = QLabel("Pick sessions to populate.")
        self.gesture_info.setStyleSheet("color:#64748b; font-size:11px;")
        self.gesture_info.setWordWrap(True)
        gb.addWidget(self.gesture_info)
        self.gesture_list = QListWidget()
        self.gesture_list.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.gesture_list.setMaximumHeight(120)
        gb.addWidget(self.gesture_list)
        self.strict_chk = QCheckBox(
            "Strict — only validate sessions that have every selected gesture"
        )
        self.strict_chk.setChecked(True)
        self.strict_chk.setToolTip(
            "Sessions missing any selected gesture are skipped instead of \n"
            "polluting the aggregate with absent data."
        )
        gb.addWidget(self.strict_chk)
        lay.addWidget(gest_box)

        # Run button
        self.run_btn = QPushButton("▶ Run batch validation")
        self.run_btn.setFixedHeight(36)
        self.run_btn.setStyleSheet(
            "QPushButton { background:#0284c7; color:white; border-radius:4px; "
            "font-weight:600; padding:6px 12px; } "
            "QPushButton:hover { background:#0369a1; } "
            "QPushButton:disabled { background:#cbd5e1; color:#64748b; }"
        )
        self.run_btn.clicked.connect(self._on_run)
        lay.addWidget(self.run_btn)

        return w

    def _build_results_panel(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)

        self.tabs = QTabWidget()
        lay.addWidget(self.tabs, 1)

        # Summary
        self.summary_lbl = QLabel(
            "No results yet — pick sessions and click Run."
        )
        self.summary_lbl.setWordWrap(True)
        self.summary_lbl.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.summary_lbl.setStyleSheet(
            "background:#f8fafc; border:1px solid #e2e8f0; padding:10px; "
            "font-family: monospace; font-size: 11px;"
        )
        sw = QWidget(); sl = QVBoxLayout(sw); sl.addWidget(self.summary_lbl); sl.addStretch()
        self.tabs.addTab(sw, "Summary")

        # Per-session table
        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels([
            "Subject", "Session", "Offset", "Conf",
            "Self-cons.", "Symmetry", "Held-out acc.",
        ])
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.tabs.addTab(self.table, "Per-session")

        # Confusion matrix
        self.confusion_canvas = _ConfusionCanvas()
        self.tabs.addTab(self.confusion_canvas, "Confusion matrix")

        # Box plot
        self.box_canvas = _BoxPlotCanvas()
        self.tabs.addTab(self.box_canvas, "Per-gesture box plot")

        # Distributions
        self.dist_canvas = _DistributionsCanvas()
        self.tabs.addTab(self.dist_canvas, "Distributions")

        # Log
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Menlo, Consolas, monospace", 9))
        self.tabs.addTab(self.log_text, "Log")

        return w

    # ------------------------------------------------------------------
    # Selection plumbing
    # ------------------------------------------------------------------

    def _refresh_subjects(self) -> None:
        self.subject_list.blockSignals(True)
        self.subject_list.clear()
        try:
            subjects = self.data_manager.list_subjects() or []
        except Exception as e:
            self._log(f"Could not list subjects: {e}")
            subjects = []
        for s in subjects:
            it = QListWidgetItem(s)
            it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            it.setCheckState(Qt.CheckState.Unchecked)
            self.subject_list.addItem(it)
        self.subject_list.blockSignals(False)
        self._refresh_sessions()

    def _refresh_sessions(self) -> None:
        self.session_list.blockSignals(True)
        self.session_list.clear()
        for subject in self._checked(self.subject_list):
            try:
                sessions = self.data_manager.list_sessions(subject) or []
            except Exception as e:
                self._log(f"Could not list sessions for {subject}: {e}")
                continue
            for sid in sessions:
                it = QListWidgetItem(f"{subject} / {sid}")
                it.setData(Qt.ItemDataRole.UserRole, (subject, sid))
                it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                it.setCheckState(Qt.CheckState.Unchecked)
                self.session_list.addItem(it)
        self.session_list.blockSignals(False)
        self._refresh_gestures()

    def _refresh_gestures(self) -> None:
        # Collect gestures from currently-checked sessions. We use the
        # union for the picker (so the user can opt in to gestures that
        # aren't in every session) but flag each row with the session
        # count so they can pick consciously.
        try:
            from playagain_pipeline.calibration.calibration_validation import (
                gestures_in_session,
            )
        except Exception:
            self.gesture_info.setText("Validator module unavailable.")
            return

        selections = self._selected_session_pairs()
        if not selections:
            self.gesture_info.setText("Select sessions to populate gestures.")
            self.gesture_list.clear()
            return

        # Counts
        gest_counts: Counter = Counter()
        loaded: List[Tuple[str, str, Any]] = []
        for subject, sid in selections:
            try:
                session = self.data_manager.load_session(subject, sid)
            except Exception as e:
                self._log(f"Could not load {subject}/{sid}: {e}")
                continue
            for g in gestures_in_session(session):
                gest_counts[g] += 1
            loaded.append((subject, sid, session))
        n = len(loaded)

        # Show gestures, with counts. Default to checking gestures present
        # in every selected session (the intersection); the user can adjust.
        self.gesture_list.blockSignals(True)
        self.gesture_list.clear()
        for g in sorted(gest_counts):
            count = gest_counts[g]
            it = QListWidgetItem(f"{g}   ({count}/{n} sessions)")
            it.setData(Qt.ItemDataRole.UserRole, g)
            it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            # Skip "rest" — held-out check excludes it anyway.
            if g.lower() == "rest":
                it.setCheckState(Qt.CheckState.Unchecked)
                it.setToolTip("Rest is excluded from accuracy by design.")
            else:
                it.setCheckState(
                    Qt.CheckState.Checked if count == n else Qt.CheckState.Unchecked
                )
            self.gesture_list.addItem(it)
        self.gesture_list.blockSignals(False)
        self.gesture_info.setText(
            f"{n} session(s) selected. "
            f"Gestures in all sessions: "
            f"{', '.join(g for g, c in gest_counts.items() if c == n) or '(none)'}."
        )

    def _on_subject_toggled(self, _item: QListWidgetItem) -> None:
        self._refresh_sessions()

    def _on_session_toggled(self, _item: QListWidgetItem) -> None:
        self._refresh_gestures()

    @staticmethod
    def _checked(lst: QListWidget) -> List[str]:
        out = []
        for i in range(lst.count()):
            it = lst.item(i)
            if it.checkState() == Qt.CheckState.Checked:
                out.append(it.text())
        return out

    def _selected_session_pairs(self) -> List[Tuple[str, str]]:
        out: List[Tuple[str, str]] = []
        for i in range(self.session_list.count()):
            it = self.session_list.item(i)
            if it.checkState() == Qt.CheckState.Checked:
                pair = it.data(Qt.ItemDataRole.UserRole)
                if pair is not None:
                    out.append(pair)
        return out

    def _selected_gestures(self) -> List[str]:
        out: List[str] = []
        for i in range(self.gesture_list.count()):
            it = self.gesture_list.item(i)
            if it.checkState() == Qt.CheckState.Checked:
                g = it.data(Qt.ItemDataRole.UserRole)
                if g:
                    out.append(g)
        return out

    @staticmethod
    def _set_all_checked(lst: QListWidget, checked: bool) -> None:
        state = Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
        # Block signals to avoid cascading refreshes during bulk toggle.
        lst.blockSignals(True)
        for i in range(lst.count()):
            lst.item(i).setCheckState(state)
        lst.blockSignals(False)
        # Manually emit one refresh so dependent panels update.
        if lst.count() > 0:
            lst.itemChanged.emit(lst.item(0))

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def _on_run(self) -> None:
        sessions = self._selected_session_pairs()
        if not sessions:
            QMessageBox.warning(
                self, "Pick at least one session",
                "Tick the sessions you want to validate."
            )
            return
        gestures = self._selected_gestures()
        strict = self.strict_chk.isChecked()
        if strict and not gestures:
            QMessageBox.information(
                self, "No gestures selected",
                "Strict mode is on but no gestures are checked. Either pick \n"
                "the gestures you care about or uncheck strict mode."
            )
            return

        if self._worker_thread is not None and self._worker_thread.isRunning():
            QMessageBox.information(
                self, "Already running",
                "A batch is in flight — wait for it to finish."
            )
            return

        self.log_text.clear()
        self.summary_lbl.setText("Running…")
        self._log(
            f"Starting batch validation: {len(sessions)} session(s), "
            f"{len(gestures)} gesture(s) selected, strict={strict}"
        )
        self.progress.setRange(0, len(sessions))
        self.progress.setValue(0)
        self.run_btn.setEnabled(False)
        self.export_btn.setEnabled(False)

        self._worker_thread = QThread(self)
        self._worker = _BatchWorker(
            calibrator=self.calibrator,
            data_manager=self.data_manager,
            selections=sessions,
            gesture_filter=gestures or None,
            strict=strict,
            validator_kwargs={},
        )
        self._worker.moveToThread(self._worker_thread)
        self._worker_thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.finished.connect(self._worker_thread.quit)
        self._worker.failed.connect(self._worker_thread.quit)
        self._worker_thread.finished.connect(self._cleanup_worker)
        self._worker_thread.start()

    @Slot(int, int, str)
    def _on_progress(self, i: int, n: int, label: str) -> None:
        self.progress.setRange(0, max(1, n))
        self.progress.setValue(i)
        self._log(f"  [{i}/{n}] {label}")

    @Slot(object)
    def _on_finished(self, batch) -> None:
        self._batch_report = batch
        self.run_btn.setEnabled(True)
        self.export_btn.setEnabled(True)
        self._log(f"Done. {batch.n_sessions} session(s) validated, "
                  f"{batch.n_passing} passed all checks. "
                  f"{len(batch.skipped)} skipped.")
        for label, reason in batch.skipped.items():
            self._log(f"  skip {label}: {reason}")
        self._render_summary(batch)
        self._render_table(batch)
        self._render_plots(batch)

    @Slot(str)
    def _on_failed(self, msg: str) -> None:
        self.run_btn.setEnabled(True)
        self.export_btn.setEnabled(False)
        self.summary_lbl.setText(f"Batch validation failed: {msg}")
        self._log(f"ERROR: {msg}")
        QMessageBox.critical(self, "Batch failed", msg)

    def _cleanup_worker(self) -> None:
        if self._worker_thread is not None:
            self._worker_thread.deleteLater()
        self._worker_thread = None
        self._worker = None

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render_summary(self, batch) -> None:
        if batch.n_sessions == 0:
            self.summary_lbl.setText("No sessions were validated.")
            return
        agg = batch.aggregate_held_out()
        pass_rates = batch.check_pass_rates()
        offsets = batch.offset_distribution()
        confs = batch.confidence_distribution()
        offset_counts = Counter(offsets)
        most_common_offset = offset_counts.most_common(1)[0] if offsets else (None, 0)

        lines = [
            f"Sessions validated:    {batch.n_sessions}",
            f"Sessions passing all:  {batch.n_passing}  "
            f"({batch.n_passing / max(batch.n_sessions, 1):.0%})",
            f"Sessions skipped:      {len(batch.skipped)}",
            "",
            f"Gesture filter:        {', '.join(batch.gesture_filter) if batch.gesture_filter else '(all)'}",
            "",
            "Pass rates per check:",
        ]
        for name, rate in pass_rates.items():
            lines.append(f"  {name:24s} {rate:.0%}")
        lines.append("")
        if offsets:
            lines.append(
                f"Offset:  most common = {most_common_offset[0]} "
                f"({most_common_offset[1]}/{len(offsets)} sessions); "
                f"unique = {len(offset_counts)}"
            )
        if confs:
            import statistics as _stats
            lines.append(
                f"Confidence: median {_stats.median(confs):.0%}, "
                f"min {min(confs):.0%}, max {max(confs):.0%}"
            )
        if agg.n_trials:
            lines.append("")
            lines.append(
                f"Held-out (1-NN, cosine):  "
                f"{agg.overall_accuracy:.1%} top-1 over {agg.n_trials} trial(s)"
            )
            for g, acc in sorted(agg.per_gesture_accuracy().items()):
                lines.append(f"  {g:20s} {acc:.0%}")
        else:
            lines.append("")
            lines.append("Held-out: no trials classified (check filter and "
                         "session contents).")
        self.summary_lbl.setText("\n".join(lines))

    def _render_table(self, batch) -> None:
        self.table.setRowCount(0)
        for r in batch.reports:
            row = self.table.rowCount()
            self.table.insertRow(row)

            def cell(text, ok=None):
                it = QTableWidgetItem(text)
                if ok is True:
                    it.setForeground(Qt.GlobalColor.darkGreen)
                elif ok is False:
                    it.setForeground(Qt.GlobalColor.red)
                return it

            self.table.setItem(row, 0, cell(r.subject_id or "?"))
            self.table.setItem(row, 1, cell(r.session_id or "?"))
            self.table.setItem(row, 2, cell(str(r.rotation_offset)))
            self.table.setItem(row, 3, cell(f"{r.confidence:.0%}"))

            sc = r.get_check("self_consistency")
            if sc is not None:
                self.table.setItem(
                    row, 4,
                    cell(f"{'pass' if sc.passed else 'FAIL'} (Δ {sc.score:.0f})", sc.passed),
                )
            else:
                self.table.setItem(row, 4, cell("—"))

            sym = r.get_check("symmetry")
            if sym is not None:
                self.table.setItem(
                    row, 5,
                    cell(f"{'pass' if sym.passed else 'FAIL'} ({sym.score:.2f})", sym.passed),
                )
            else:
                self.table.setItem(row, 5, cell("—"))

            ho = r.get_check("held_out_accuracy")
            if ho is not None:
                self.table.setItem(
                    row, 6,
                    cell(f"{'pass' if ho.passed else 'FAIL'} ({ho.score:.0f}%)", ho.passed),
                )
            else:
                self.table.setItem(row, 6, cell("—"))

    def _render_plots(self, batch) -> None:
        self.confusion_canvas.update_from(batch.aggregate_held_out())
        self.box_canvas.update_from(batch.per_gesture_session_accuracy())
        self.dist_canvas.update_from(batch)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def _on_export(self) -> None:
        if self._batch_report is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export results — pick a directory or file prefix",
            "rotation_validation",
        )
        if not path:
            return
        prefix = Path(path)
        prefix.parent.mkdir(parents=True, exist_ok=True)

        # JSON of every report (no plots)
        try:
            import json
            payload = {
                "n_sessions":      self._batch_report.n_sessions,
                "n_passing":       self._batch_report.n_passing,
                "gesture_filter":  self._batch_report.gesture_filter,
                "skipped":         self._batch_report.skipped,
                "reports": [
                    {
                        "subject_id":      r.subject_id,
                        "session_id":      r.session_id,
                        "rotation_offset": r.rotation_offset,
                        "confidence":      r.confidence,
                        "is_acceptable":   r.is_acceptable,
                        "checks": [
                            {
                                "name":   c.name,
                                "passed": c.passed,
                                "score":  c.score,
                                "unit":   c.unit,
                                "detail": c.detail,
                            } for c in r.checks
                        ],
                        "per_gesture_accuracy": r.per_gesture_accuracy,
                    } for r in self._batch_report.reports
                ],
            }
            with open(prefix.with_suffix(".json"), "w") as f:
                json.dump(payload, f, indent=2)
        except Exception as e:
            self._log(f"JSON export failed: {e}")

        # PNG of each plot canvas
        for tag, canvas in [
            ("confusion",   self.confusion_canvas),
            ("box",         self.box_canvas),
            ("distributions", self.dist_canvas),
        ]:
            fig = canvas.figure() if hasattr(canvas, "figure") else None
            if fig is None:
                continue
            try:
                fig.savefig(str(prefix.with_name(f"{prefix.name}_{tag}.png")), dpi=130)
            except Exception as e:
                self._log(f"PNG export failed for {tag}: {e}")

        self._log(f"Exported to {prefix}.json + plot PNGs.")
        QMessageBox.information(
            self, "Exported",
            f"Wrote results to:\n{prefix.with_suffix('.json')}\n"
            f"plus PNGs alongside it."
        )

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log(self, msg: str) -> None:
        self.log_text.append(msg)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def closeEvent(self, event) -> None:
        if self._worker_thread is not None and self._worker_thread.isRunning():
            self._worker_thread.quit()
            self._worker_thread.wait(2000)
        super().closeEvent(event)
