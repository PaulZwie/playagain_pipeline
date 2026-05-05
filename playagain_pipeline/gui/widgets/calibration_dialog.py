"""
calibration_dialog.py
─────────────────────
Standalone calibration dialog — four tabs in one window.

Tab 1: Live Calibration
    Record gestures on the connected device right now.

Tab 2: From Session
    Pick any previously-recorded session and derive the rotation offset
    from its gesture trials.  Works without a device.

Tab 3: Validate
    Run independent checks on the active calibration.  Self-consistency
    and symmetry are free; held-out classification needs a labelled
    session.

Tab 4: Rotation Stats   (new)
    Run the rotation detector across many sessions and see how stable
    the offset is across the corpus.  Useful when you suspect the
    bracelet is shifting between sessions, or when you want a single-
    glance view of "is rotation detection working consistently?"

What changed (v2)
─────────────────
- The held-out check no longer crashes with
  ``'RecordingSession' object has no attribute 'load_signal'`` — the
  validator now uses the real session API.
- The Validate tab no longer logs the same free-check block multiple
  times when the user runs both free and held-out checks; held-out
  output reuses the report header.
- The calibration "Confidence" line is shown alongside the verdict as
  *informational*, not as a primary acceptance criterion. Low confidence
  no longer blocks acceptance by default — it surfaces as a warning.
- New "Rotation Stats" tab driven by ``RotationDetectionStudy``.
"""

from __future__ import annotations

import io
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from PySide6.QtCore import Qt, Signal, Slot, QTimer
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
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
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_GESTURE_EMOJIS: Dict[str, str] = {
    "rest": "🖐",      "open_hand": "🖐",   "fist": "✊",
    "pinch": "👌",     "tripod": "🤌",      "index_point": "☝",
    "thumb_out": "👍", "waveout": "🤚",
    "index_flex": "☝", "middle_flex": "🖕", "ring_flex": "💍",
    "pinky_flex": "🤙", "thumb_flex": "👍",
}


def _info(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setWordWrap(True)
    lbl.setStyleSheet("color: #555; font-size: 10px;")
    return lbl


# ---------------------------------------------------------------------------
# Off-screen matplotlib rendering — *deliberately* avoids the Qt backend.
# ---------------------------------------------------------------------------
#
# Mixing matplotlib's Qt backend (FigureCanvasQTAgg) with PySide6 has a
# tendency to SIGBUS on macOS Apple Silicon (and sometimes Linux): both
# load Qt at the C level, and conflicting symbols/event-loop ownership
# cause the process to die mid-event-dispatch. The bullet-proof fix is
# to render with the headless ``Agg`` backend, get a PNG byte-string
# back, and stuff it into a ``QPixmap``. Matplotlib never sees Qt at
# all; PySide6 just paints a bitmap.

_MPL_LOADED = False
_Figure = None  # type: ignore[assignment]


def _ensure_matplotlib() -> bool:
    """Lazily import matplotlib in headless mode.

    Returns True on success. Safe to call multiple times.
    """
    global _MPL_LOADED, _Figure
    if _MPL_LOADED:
        return _Figure is not None
    _MPL_LOADED = True
    try:
        import matplotlib
        # MUST be called before any pyplot import / backend selection.
        # 'Agg' is a pure-Python/C raster backend with no GUI dependency.
        matplotlib.use("Agg", force=True)
        from matplotlib.figure import Figure
        _Figure = Figure
        return True
    except Exception:  # noqa: BLE001
        _Figure = None
        return False


class _AggPlot(QLabel):
    """A QLabel that renders a matplotlib Figure into a QPixmap.

    The figure is built off-screen via the Agg backend, serialised to a
    PNG buffer, then loaded as a pixmap. No Qt-aware matplotlib backend
    is ever imported, so this is immune to the FigureCanvasQTAgg-vs-
    PySide6 SIGBUS class of problem.
    """

    def __init__(self, *, min_height: int = 280, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumHeight(min_height)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setStyleSheet(
            "background:#fafafa; border:1px solid #e2e8f0; border-radius:4px;"
        )
        self._placeholder("(no plot yet — run the analysis first)")
        self._last_png: Optional[bytes] = None

    def _placeholder(self, text: str) -> None:
        self.setText(text)
        self.setStyleSheet(
            "background:#fafafa; border:1px solid #e2e8f0; border-radius:4px; "
            "color:#94a3b8; padding:24px;"
        )

    def show_message(self, text: str) -> None:
        self.setPixmap(QPixmap())
        self._placeholder(text)
        self._last_png = None

    def render_with(self, draw_fn, *, figsize=(7.0, 4.5), dpi=110) -> None:
        """Call ``draw_fn(fig)`` to populate the figure, then display it.

        ``draw_fn`` may raise; exceptions are caught and shown as the
        plot's placeholder so one bad plot never takes down the dialog.
        """
        if not _ensure_matplotlib():
            self.show_message(
                "matplotlib is not available — install it to enable plots.\n"
                "(pip install matplotlib)"
            )
            return
        try:
            fig = _Figure(figsize=figsize, dpi=dpi, tight_layout=True)
            draw_fn(fig)
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                        facecolor="white")
            buf.seek(0)
            png_bytes = buf.read()
        except Exception as e:  # noqa: BLE001
            self.show_message(f"(plot error: {type(e).__name__}: {e})")
            return

        pix = QPixmap()
        pix.loadFromData(png_bytes, "PNG")
        if pix.isNull():
            self.show_message("(failed to decode rendered PNG)")
            return
        self._last_png = png_bytes
        self.setText("")
        self.setStyleSheet(
            "background:#fafafa; border:1px solid #e2e8f0; border-radius:4px;"
        )
        self.setPixmap(pix)

    @property
    def last_png(self) -> Optional[bytes]:
        """The PNG bytes of the most recently rendered figure (or None)."""
        return self._last_png


# ---------------------------------------------------------------------------
# Plot drawing functions — pure functions of (Figure, data) → None.
#
# Kept at module level so they can be tested without instantiating the
# dialog. Each receives a matplotlib Figure already created by _AggPlot.
# ---------------------------------------------------------------------------

# Shared palette
_C_PRIMARY   = "#0284c7"
_C_PRIMARY_L = "#bae6fd"
_C_GOOD      = "#16a34a"
_C_WARN      = "#f59e0b"
_C_BAD       = "#dc2626"
_C_GRID      = "#e2e8f0"
_C_TEXT      = "#334155"


def _style_axes(ax) -> None:
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(colors=_C_TEXT, labelsize=9)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(_C_GRID)
    ax.title.set_color(_C_TEXT)
    ax.xaxis.label.set_color(_C_TEXT)
    ax.yaxis.label.set_color(_C_TEXT)


def _draw_offset_bars(fig, rot_report) -> None:
    """Bar chart: rotation offset detected for each session."""
    ax = fig.add_subplot(111)
    succ = rot_report.successful
    if not succ:
        ax.text(0.5, 0.5, "(no successful sessions)",
                ha="center", va="center", transform=ax.transAxes,
                color="#94a3b8")
        ax.set_axis_off()
        return

    labels = [f"{r.subject_id}/{r.session_id}" for r in succ]
    offsets = [r.rotation_offset for r in succ]
    drifts = [r.drift_from_reference for r in succ]
    # Color by drift: 0 = good, 1 = warn, ≥2 = bad
    colors = [
        _C_GOOD if d == 0 else (_C_WARN if d <= 1 else _C_BAD)
        for d in drifts
    ]
    x = np.arange(len(labels))
    ax.bar(x, offsets, color=colors, edgecolor="#0c4a6e", linewidth=0.5)
    ax.axhline(rot_report.reference_offset,
               linestyle="--", color=_C_TEXT, linewidth=1, alpha=0.6,
               label=f"reference offset = {rot_report.reference_offset}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Detected offset (channels)")
    ax.set_title(f"Rotation offset per session  (ring size {rot_report.ring_size})")
    ax.legend(loc="best", fontsize=8, frameon=False)
    ax.grid(axis="y", linestyle=":", alpha=0.5, color=_C_GRID)
    _style_axes(ax)

    # Legend for color coding
    from matplotlib.patches import Patch
    legend_elems = [
        Patch(facecolor=_C_GOOD, label="drift 0 ch"),
        Patch(facecolor=_C_WARN, label="drift 1 ch"),
        Patch(facecolor=_C_BAD,  label="drift ≥ 2 ch"),
    ]
    ax2 = ax.twinx()
    ax2.axis("off")
    ax2.legend(handles=legend_elems, loc="upper right", fontsize=8,
               frameon=True, framealpha=0.9, title="color = drift")


def _draw_drift_hist(fig, rot_report) -> None:
    ax = fig.add_subplot(111)
    drifts = rot_report.drifts()
    if drifts.size == 0:
        ax.text(0.5, 0.5, "(no successful sessions)",
                ha="center", va="center", transform=ax.transAxes,
                color="#94a3b8")
        ax.set_axis_off()
        return
    # Integer bins from 0 .. max
    max_d = max(int(drifts.max()), 1)
    bins = np.arange(-0.5, max_d + 1.5, 1.0)
    ax.hist(drifts, bins=bins, color=_C_PRIMARY_L,
            edgecolor=_C_PRIMARY, linewidth=1.0)
    ax.set_xlabel("Drift from reference (ring distance, channels)")
    ax.set_ylabel("Number of sessions")
    ax.set_title("Drift distribution")
    ax.set_xticks(np.arange(0, max_d + 1, 1))
    ax.grid(axis="y", linestyle=":", alpha=0.5, color=_C_GRID)
    _style_axes(ax)


def _draw_conf_hist(fig, rot_report) -> None:
    ax = fig.add_subplot(111)
    confs = rot_report.confidences()
    if confs.size == 0:
        ax.text(0.5, 0.5, "(no successful sessions)",
                ha="center", va="center", transform=ax.transAxes,
                color="#94a3b8")
        ax.set_axis_off()
        return
    bins = np.linspace(0.0, 1.0, 21)
    ax.hist(confs, bins=bins, color="#fde68a", edgecolor="#92400e",
            linewidth=1.0)
    ax.set_xlabel("Confidence (xcorr peak z-score, normalised)")
    ax.set_ylabel("Number of sessions")
    ax.set_title("Detector confidence")
    ax.set_xlim(0.0, 1.0)
    ax.grid(axis="y", linestyle=":", alpha=0.5, color=_C_GRID)
    _style_axes(ax)


def _draw_conf_vs_drift(fig, rot_report) -> None:
    ax = fig.add_subplot(111)
    succ = rot_report.successful
    if not succ:
        ax.text(0.5, 0.5, "(no successful sessions)",
                ha="center", va="center", transform=ax.transAxes,
                color="#94a3b8")
        ax.set_axis_off()
        return
    rng = np.random.default_rng(0)
    drifts = np.array([r.drift_from_reference for r in succ], dtype=float)
    confs = np.array([r.confidence for r in succ], dtype=float)
    # Jitter drift slightly so identical points don't overlap
    jitter = rng.normal(0.0, 0.06, size=drifts.size)
    ax.scatter(drifts + jitter, confs, color=_C_PRIMARY,
               alpha=0.7, s=42, edgecolor="#0c4a6e", linewidth=0.5)
    for r, jx in zip(succ, jitter):
        ax.annotate(
            f"{r.subject_id}/{r.session_id}",
            (r.drift_from_reference + jx, r.confidence),
            fontsize=7, color=_C_TEXT, alpha=0.6,
            xytext=(4, 2), textcoords="offset points",
        )
    ax.set_xlabel("Drift from reference (channels)")
    ax.set_ylabel("Confidence")
    ax.set_title("Are low-confidence sessions also off-axis?")
    ax.set_ylim(0.0, max(1.0, float(confs.max()) + 0.1))
    ax.grid(linestyle=":", alpha=0.5, color=_C_GRID)
    _style_axes(ax)


def _draw_gesture_coverage(fig, loaded: List[Tuple[str, str, Any]]) -> None:
    """Heat-map: rows = sessions, cols = gestures, cell = present?"""
    ax = fig.add_subplot(111)
    if not loaded:
        ax.text(0.5, 0.5, "(no sessions loaded yet)",
                ha="center", va="center", transform=ax.transAxes,
                color="#94a3b8")
        ax.set_axis_off()
        return
    try:
        from playagain_pipeline.calibration.calibration_validation import (
            gestures_in_session,
        )
    except ImportError:
        ax.text(0.5, 0.5, "(calibration_validation.py not on path)",
                ha="center", va="center", transform=ax.transAxes,
                color="#94a3b8")
        ax.set_axis_off()
        return

    per_session: Dict[str, Set[str]] = {}
    for subj, sid, ses in loaded:
        per_session[f"{subj}/{sid}"] = gestures_in_session(ses)
    all_gestures = sorted({g for s in per_session.values() for g in s})
    if not all_gestures:
        ax.text(0.5, 0.5, "(no gestures recorded)",
                ha="center", va="center", transform=ax.transAxes,
                color="#94a3b8")
        ax.set_axis_off()
        return

    sess_labels = list(per_session.keys())
    mat = np.zeros((len(sess_labels), len(all_gestures)), dtype=int)
    for i, sl in enumerate(sess_labels):
        for j, g in enumerate(all_gestures):
            mat[i, j] = 1 if g in per_session[sl] else 0

    # Custom 2-color colormap: white (missing) → primary (present)
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(["#f1f5f9", _C_PRIMARY])
    im = ax.imshow(mat, cmap=cmap, aspect="auto", vmin=0, vmax=1,
                   interpolation="nearest")
    ax.set_xticks(np.arange(len(all_gestures)))
    ax.set_xticklabels(all_gestures, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(np.arange(len(sess_labels)))
    ax.set_yticklabels(sess_labels, fontsize=8)

    # Coverage row — fraction of sessions with each gesture
    coverage = mat.sum(axis=0) / max(len(sess_labels), 1)
    for j, c in enumerate(coverage):
        ax.text(j, len(sess_labels) - 0.4, f"{int(c*100)}%",
                ha="center", va="bottom", fontsize=7, color=_C_TEXT)

    ax.set_title("Gesture coverage  (■ present  ·  □ missing  ·  % across sessions)")
    _style_axes(ax)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_visible(False)


def _draw_aggregate_confusion(fig, batch_report) -> None:
    ax = fig.add_subplot(111)
    agg = batch_report.aggregate_held_out_details()
    if agg.n_trials == 0:
        ax.text(0.5, 0.5, "(no held-out trials)",
                ha="center", va="center", transform=ax.transAxes,
                color="#94a3b8")
        ax.set_axis_off()
        return
    labels, mat = agg.confusion()
    if mat.size == 0 or len(labels) == 0:
        ax.text(0.5, 0.5, "(empty confusion matrix)",
                ha="center", va="center", transform=ax.transAxes,
                color="#94a3b8")
        ax.set_axis_off()
        return
    # Row-normalise (recall view)
    row_sums = mat.sum(axis=1, keepdims=True).astype(float)
    row_sums[row_sums == 0] = 1.0
    norm = mat.astype(float) / row_sums

    im = ax.imshow(norm, cmap="Blues", vmin=0.0, vmax=1.0,
                   aspect="auto", interpolation="nearest")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    overall = float(np.trace(mat)) / max(int(mat.sum()), 1)
    ax.set_title(
        f"Aggregate confusion  ·  {agg.n_trials} trials  ·  "
        f"overall accuracy {overall:.1%}"
    )
    if len(labels) <= 12:
        for r in range(norm.shape[0]):
            for c in range(norm.shape[1]):
                v = norm[r, c]
                if v >= 0.005:
                    ax.text(
                        c, r, f"{v:.2f}",
                        ha="center", va="center", fontsize=8,
                        color=("white" if v > 0.55 else "#1e3a8a"),
                    )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _style_axes(ax)


def _draw_per_gesture_box(fig, batch_report) -> None:
    ax = fig.add_subplot(111)
    pgs = batch_report.per_gesture_session_accuracy()
    if not pgs:
        ax.text(0.5, 0.5, "(no per-gesture accuracy data)",
                ha="center", va="center", transform=ax.transAxes,
                color="#94a3b8")
        ax.set_axis_off()
        return
    # Sort by median ascending — easier to spot weak gestures
    gestures = sorted(pgs.keys(),
                      key=lambda k: float(np.median(pgs[k])) if pgs[k] else 0.0)
    data = [pgs[g] for g in gestures]
    labels = [f"{g}\n(n={len(pgs[g])})" for g in gestures]

    bp = ax.boxplot(
        data, labels=labels, vert=True, patch_artist=True,
        showmeans=True,
        meanprops={"marker": "D", "markerfacecolor": _C_PRIMARY,
                   "markeredgecolor": _C_PRIMARY, "markersize": 5},
    )
    for patch in bp["boxes"]:
        patch.set_facecolor(_C_PRIMARY_L)
        patch.set_edgecolor("#0c4a6e")
    for el in bp["whiskers"] + bp["caps"]:
        el.set_color("#0c4a6e")
    for m in bp["medians"]:
        m.set_color("#0c4a6e")
        m.set_linewidth(2)

    # Overlay scatter — one point per session
    rng = np.random.default_rng(0)
    for i, vs in enumerate(data, start=1):
        xs = rng.normal(loc=i, scale=0.05, size=len(vs))
        ax.scatter(xs, vs, color="#0c4a6e", alpha=0.6, s=18, zorder=3)

    ax.axhline(0.70, color=_C_WARN, linestyle="--", linewidth=1, alpha=0.6,
               label="70% threshold")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Per-session accuracy")
    ax.set_title("Per-gesture accuracy spread across sessions")
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    ax.grid(axis="y", linestyle=":", alpha=0.5, color=_C_GRID)
    if len(gestures) > 4:
        for tick in ax.get_xticklabels():
            tick.set_rotation(20)
            tick.set_ha("right")
    _style_axes(ax)


def _draw_pass_rates(fig, batch_report) -> None:
    ax = fig.add_subplot(111)
    rates = batch_report.check_pass_rates()
    if not rates:
        ax.text(0.5, 0.5, "(no checks recorded)",
                ha="center", va="center", transform=ax.transAxes,
                color="#94a3b8")
        ax.set_axis_off()
        return
    names = list(rates.keys())
    vals = [rates[n] for n in names]
    colors = [
        _C_GOOD if v >= 0.9 else (_C_WARN if v >= 0.6 else _C_BAD)
        for v in vals
    ]
    y = np.arange(len(names))
    ax.barh(y, vals, color=colors, edgecolor="#0c4a6e", linewidth=0.5)
    for i, v in enumerate(vals):
        ax.text(v + 0.01, i, f"{v:.0%}", va="center", fontsize=9, color=_C_TEXT)
    ax.set_yticks(y)
    ax.set_yticklabels([n.replace("_", " ") for n in names], fontsize=10)
    ax.set_xlim(0.0, 1.05)
    ax.set_xlabel("Fraction of sessions passing")
    ax.set_title("Validation check pass rates")
    ax.grid(axis="x", linestyle=":", alpha=0.5, color=_C_GRID)
    _style_axes(ax)


# ---------------------------------------------------------------------------
# Note: the previous QThread-based ``_RotationStudyWorker`` was removed.
# Loading sessions via DataManager → h5py from a worker thread on macOS
# (and sometimes Linux) caused ``SIGBUS`` (signal 10) crashes because
# h5py / HDF5 / file-mapped numpy arrays are not safely re-entrant
# across threads. The rotation study now runs synchronously on the
# main thread with ``QApplication.processEvents()`` between sessions.
# That is what ``CalibrationDialog._on_run_rotation_study`` does below.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# CalibrationDialog
# ---------------------------------------------------------------------------

class CalibrationDialog(QDialog):
    """Tabbed calibration dialog. Live, From Session, Validate, and Rotation Stats."""

    calibration_complete = Signal(object)  # CalibrationResult

    # Standard 8-gesture live-calibration protocol
    CALIBRATION_GESTURES = [
        ("waveout",      "Extend your wrist outward (bend the back of your hand upward)"),
        ("rest",         "Keep your hand completely relaxed, palm up"),
        ("index_flex",   "Bend your INDEX finger down toward your palm"),
        ("middle_flex",  "Bend your MIDDLE finger down toward your palm"),
        ("ring_flex",    "Bend your RING finger down toward your palm"),
        ("pinky_flex",   "Bend your PINKY finger down toward your palm"),
        ("thumb_flex",   "Bend your THUMB inward toward your palm"),
        ("fist",         "Close all fingers into a firm FIST"),
    ]

    def __init__(
        self,
        calibrator,
        device=None,
        data_manager=None,
        parent=None,
    ):
        super().__init__(parent)
        self.calibrator   = calibrator
        self.device       = device
        self.data_manager = data_manager

        self._calibration_result = None
        # Gesture arrays captured during live or extracted from a session,
        # kept for the validator.
        self._gesture_data: Dict[str, np.ndarray] = {}

        # Live-recording state machine
        self._recording_timer    = QTimer(self)
        self._recording_duration = 3.0
        self._live_idx           = 0
        self._live_countdown     = 0
        self._live_remaining     = 0.0
        self._is_recording       = False
        self._live_buffer: list  = []

        # Rotation-stats / batch validation state (synchronous main thread)
        self._rot_cancel_requested: bool = False
        self._rot_session_cache: Dict[Tuple[str, str], Any] = {}
        self._rot_last_report = None
        self._rot_last_batch  = None

        self.setWindowTitle("EMG Calibration")
        # Larger default size — Tab 4 has a left/right splitter with plots.
        self.setMinimumSize(820, 640)
        self.resize(1080, 760)
        self._setup_ui()
        QTimer.singleShot(0, self._refresh_sessions)
        QTimer.singleShot(0, self._refresh_rot_subjects)

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)

        self._tabs = QTabWidget()
        root.addWidget(self._tabs, 1)

        self._tabs.addTab(self._build_live_tab(),     "① Live")
        self._tabs.addTab(self._build_session_tab(),  "② From Session")
        self._tabs.addTab(self._build_validate_tab(), "③ Validate")
        self._tabs.addTab(self._build_rotation_stats_tab(), "④ Rotation Stats")

        log_box = QGroupBox("Log")
        ll = QVBoxLayout(log_box)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFixedHeight(90)
        ll.addWidget(self.log_text)
        root.addWidget(log_box)

        brow = QHBoxLayout()
        brow.addStretch()
        self.finish_btn = QPushButton("Apply && Close")
        self.finish_btn.setToolTip("Apply the last successful calibration and close.")
        self.finish_btn.setEnabled(False)
        self.finish_btn.setStyleSheet(
            "QPushButton { background:#16a34a; color:white; border-radius:4px; "
            "font-weight:600; padding:4px 16px; } "
            "QPushButton:hover { background:#15803d; } "
            "QPushButton:disabled { background:#cbd5e1; color:#64748b; }"
        )
        self.finish_btn.clicked.connect(self.accept)
        brow.addWidget(self.finish_btn)

        cancel_btn = QPushButton("Close without applying")
        cancel_btn.clicked.connect(self.reject)
        brow.addWidget(cancel_btn)
        root.addLayout(brow)

        # Disable session/stats tabs when no data_manager is available
        if self.data_manager is None:
            for idx in (1, 3):
                self._tabs.setTabEnabled(idx, False)
                self._tabs.setTabToolTip(
                    idx, "Requires a DataManager. Open the dialog from the main window."
                )
            self._tabs.setCurrentIndex(0)

    # ── Tab 1: Live ──────────────────────────────────────────────────

    def _build_live_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)

        lay.addWidget(_info(
            "Record gestures live with the connected device.\n"
            "waveOut (first gesture) is the primary sync signal "
            "(Barona López et al., 2020). Device must be streaming."
        ))

        self.live_gesture_lbl = QLabel("—")
        self.live_gesture_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        f = self.live_gesture_lbl.font()
        f.setPointSize(28); f.setBold(True)
        self.live_gesture_lbl.setFont(f)
        lay.addWidget(self.live_gesture_lbl)

        self.live_instruction_lbl = QLabel("")
        self.live_instruction_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.live_instruction_lbl.setWordWrap(True)
        lay.addWidget(self.live_instruction_lbl)

        self.live_progress = QProgressBar()
        self.live_progress.setMaximum(int(self._recording_duration * 10))
        self.live_progress.setValue(0)
        lay.addWidget(self.live_progress)

        self.live_status_lbl = QLabel("Ready — click Start when device is streaming.")
        self.live_status_lbl.setStyleSheet("font-weight:bold;")
        lay.addWidget(self.live_status_lbl)

        brow = QHBoxLayout()
        self.live_start_btn = QPushButton("▶  Start Live Calibration")
        self.live_start_btn.setFixedHeight(34)
        self.live_start_btn.clicked.connect(self._on_live_start)
        brow.addWidget(self.live_start_btn)

        self.live_cancel_btn = QPushButton("✕  Cancel")
        self.live_cancel_btn.setFixedHeight(34)
        self.live_cancel_btn.setEnabled(False)
        self.live_cancel_btn.clicked.connect(self._on_live_cancel)
        brow.addWidget(self.live_cancel_btn)
        lay.addLayout(brow)

        lay.addStretch()
        return w

    # ── Tab 2: From Session ──────────────────────────────────────────

    def _build_session_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)

        lay.addWidget(_info(
            "Derive the rotation offset from a previously-recorded session.\n"
            "No device needed — works entirely from saved EMG data.\n\n"
            "If the session has dedicated calibration_sync trials (waveOut), "
            "those are used first. Otherwise all gesture trials are used."
        ))

        form = QFormLayout()
        self.ses_subject_combo = QComboBox()
        self.ses_subject_combo.currentTextChanged.connect(self._on_subject_changed)
        form.addRow("Subject:", self.ses_subject_combo)

        self.ses_session_combo = QComboBox()
        form.addRow("Session:", self.ses_session_combo)
        lay.addLayout(form)

        refresh_btn = QPushButton("⟳  Refresh list")
        refresh_btn.clicked.connect(self._refresh_sessions)
        lay.addWidget(refresh_btn)

        self.ses_cal_btn = QPushButton("Calibrate from this session")
        self.ses_cal_btn.setFixedHeight(36)
        self.ses_cal_btn.setStyleSheet(
            "QPushButton { background:#0284c7; color:white; border-radius:4px; "
            "font-weight:600; } QPushButton:hover { background:#0369a1; } "
            "QPushButton:disabled { background:#cbd5e1; color:#64748b; }"
        )
        self.ses_cal_btn.clicked.connect(self._on_calibrate_from_session)
        lay.addWidget(self.ses_cal_btn)

        self.ses_status_lbl = QLabel("")
        lay.addWidget(self.ses_status_lbl)
        lay.addStretch()
        return w

    # ── Tab 3: Validate ──────────────────────────────────────────────

    def _build_validate_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)

        lay.addWidget(_info(
            "Three independent checks verify the calibration quality.\n\n"
            "Free (no extra data):\n"
            "  • Self-consistency — re-run xcorr; drift > 1 ch = unreliable.\n"
            "  • Symmetry — primary peak vs the half-ring twin (catches "
            "bracelet-on-backwards specifically).\n"
            "  • Confidence — informational only by default; low confidence "
            "with a self-consistent offset is OK.\n\n"
            "Held-out (needs a labelled session): per-trial 1-NN gesture "
            "discrimination accuracy ≥ 70%."
        ))

        self.val_free_btn = QPushButton("Run free checks (self-consistency + symmetry + confidence)")
        self.val_free_btn.setFixedHeight(34)
        self.val_free_btn.clicked.connect(self._on_run_free_checks)
        lay.addWidget(self.val_free_btn)

        self.val_ho_btn = QPushButton("Run held-out check (pick a session)…")
        self.val_ho_btn.setFixedHeight(34)
        self.val_ho_btn.clicked.connect(self._on_run_held_out_check)
        lay.addWidget(self.val_ho_btn)

        self.val_result_lbl = QLabel("No results yet.")
        self.val_result_lbl.setWordWrap(True)
        self.val_result_lbl.setStyleSheet(
            "background:#f8fafc; border:1px solid #e2e8f0; "
            "padding:6px; border-radius:4px; font-family:monospace;"
        )
        self.val_result_lbl.setMinimumHeight(140)
        lay.addWidget(self.val_result_lbl)

        lay.addStretch()
        return w

    # ── Tab 4: Rotation Stats ────────────────────────────────────────

    def _build_rotation_stats_tab(self) -> QWidget:
        """Validation studio — rotation stats + optional held-out classification.

        Layout: horizontal splitter with selection controls on the left,
        results (table + plots) on the right. Runs synchronously on the
        main thread with ``QApplication.processEvents`` between sessions
        — no QThread, no h5py-from-worker SIGBUS hazard.
        """
        w = QWidget()
        outer = QVBoxLayout(w)
        outer.setContentsMargins(4, 4, 4, 4)

        outer.addWidget(_info(
            "Run rotation detection across many sessions to see how stable "
            "the offset is over the corpus. Optionally also run held-out "
            "gesture classification to measure how well the rotation-corrected "
            "patterns generalise. Sessions, gestures, and metrics are all "
            "selectable on the left."
        ))

        splitter = QSplitter(Qt.Orientation.Horizontal)
        outer.addWidget(splitter, 1)

        # ───── LEFT PANEL: selection ─────────────────────────────────────
        left = QWidget()
        left_lay = QVBoxLayout(left)
        left_lay.setContentsMargins(0, 0, 0, 0)

        # Subject combo
        form = QFormLayout()
        self.rot_subject_combo = QComboBox()
        self.rot_subject_combo.addItem("(all subjects)", userData="__all__")
        self.rot_subject_combo.currentIndexChanged.connect(self._refresh_rot_sessions)
        form.addRow("Subject:", self.rot_subject_combo)
        left_lay.addLayout(form)

        # Session list
        sess_grp = QGroupBox("Sessions")
        sess_lay = QVBoxLayout(sess_grp)
        self.rot_session_list = QListWidget()
        self.rot_session_list.setSelectionMode(QListWidget.SelectionMode.NoSelection)
        self.rot_session_list.setMinimumHeight(140)
        # When the user toggles which sessions are included, refresh the
        # gesture coverage table on the fly.
        self.rot_session_list.itemChanged.connect(self._on_rot_session_toggled)
        sess_lay.addWidget(self.rot_session_list)

        sel_row = QHBoxLayout()
        sel_all_btn = QPushButton("Tick all")
        sel_all_btn.setFixedHeight(22)
        sel_all_btn.clicked.connect(lambda: self._set_all_rot_session_ticks(True))
        sel_row.addWidget(sel_all_btn)
        sel_none_btn = QPushButton("Untick all")
        sel_none_btn.setFixedHeight(22)
        sel_none_btn.clicked.connect(lambda: self._set_all_rot_session_ticks(False))
        sel_row.addWidget(sel_none_btn)
        sel_row.addStretch()
        rot_refresh_btn = QPushButton("⟳ Refresh")
        rot_refresh_btn.setFixedHeight(22)
        rot_refresh_btn.clicked.connect(self._refresh_rot_subjects)
        sel_row.addWidget(rot_refresh_btn)
        sess_lay.addLayout(sel_row)
        left_lay.addWidget(sess_grp)

        # Gesture filter
        gest_grp = QGroupBox("Gestures (held-out classification)")
        gest_lay = QVBoxLayout(gest_grp)
        gest_lay.addWidget(_info(
            "Coverage is shown as 'gesture (X/N sessions)'. The held-out "
            "classifier will only score the gestures you tick. The mode "
            "below decides what to do with sessions that are missing one "
            "of the selected gestures."
        ))
        self.rot_gesture_list = QListWidget()
        self.rot_gesture_list.setSelectionMode(QListWidget.SelectionMode.NoSelection)
        self.rot_gesture_list.setMinimumHeight(110)
        gest_lay.addWidget(self.rot_gesture_list)

        gbtn_row = QHBoxLayout()
        g_all_btn = QPushButton("All")
        g_all_btn.setFixedHeight(22)
        g_all_btn.clicked.connect(lambda: self._set_all_rot_gesture_ticks(True))
        gbtn_row.addWidget(g_all_btn)
        g_none_btn = QPushButton("None")
        g_none_btn.setFixedHeight(22)
        g_none_btn.clicked.connect(lambda: self._set_all_rot_gesture_ticks(False))
        gbtn_row.addWidget(g_none_btn)
        g_common_btn = QPushButton("Common only")
        g_common_btn.setFixedHeight(22)
        g_common_btn.setToolTip("Tick only the gestures that appear in EVERY ticked session.")
        g_common_btn.clicked.connect(self._tick_common_gestures)
        gbtn_row.addWidget(g_common_btn)
        gbtn_row.addStretch()
        gest_lay.addLayout(gbtn_row)

        # Mode radios
        self.rot_mode_lenient = QRadioButton(
            "Use whatever ticked gestures each session has (skip nothing)"
        )
        self.rot_mode_lenient.setChecked(True)
        self.rot_mode_lenient.setToolTip(
            "A session that's missing one selected gesture is still kept; "
            "it just contributes only to the gestures it has."
        )
        gest_lay.addWidget(self.rot_mode_lenient)
        self.rot_mode_strict = QRadioButton(
            "Skip sessions missing any selected gesture (strict)"
        )
        self.rot_mode_strict.setToolTip(
            "Drops sessions where any ticked gesture is absent. Use this for "
            "perfectly-comparable per-gesture metrics across all sessions."
        )
        gest_lay.addWidget(self.rot_mode_strict)
        left_lay.addWidget(gest_grp)

        # Metrics group
        met_grp = QGroupBox("Metrics")
        met_lay = QVBoxLayout(met_grp)
        self.rot_chk_held_out = QCheckBox("Run held-out gesture classification")
        self.rot_chk_held_out.setChecked(True)
        self.rot_chk_held_out.setToolTip(
            "When ticked, each session is also evaluated by the gesture "
            "classifier. Adds the confusion-matrix and per-gesture box-plot "
            "tabs to the results."
        )
        met_lay.addWidget(self.rot_chk_held_out)
        left_lay.addWidget(met_grp)

        # Run / cancel / progress
        run_row = QHBoxLayout()
        self.rot_run_btn = QPushButton("▶ Analyze")
        self.rot_run_btn.setFixedHeight(34)
        self.rot_run_btn.setStyleSheet(
            "QPushButton { background:#0284c7; color:white; border-radius:4px; "
            "font-weight:600; } QPushButton:hover { background:#0369a1; } "
            "QPushButton:disabled { background:#cbd5e1; color:#64748b; }"
        )
        self.rot_run_btn.clicked.connect(self._on_run_rotation_study)
        run_row.addWidget(self.rot_run_btn, 1)
        self.rot_cancel_btn = QPushButton("■ Cancel")
        self.rot_cancel_btn.setFixedHeight(34)
        self.rot_cancel_btn.setEnabled(False)
        self.rot_cancel_btn.clicked.connect(self._on_cancel_rotation_study)
        run_row.addWidget(self.rot_cancel_btn)
        left_lay.addLayout(run_row)

        self.rot_progress = QProgressBar()
        self.rot_progress.setRange(0, 1)
        self.rot_progress.setValue(0)
        self.rot_progress.setTextVisible(True)
        left_lay.addWidget(self.rot_progress)

        self.rot_summary_lbl = QLabel("")
        self.rot_summary_lbl.setWordWrap(True)
        self.rot_summary_lbl.setStyleSheet(
            "color:#0e7490; font-weight:600; padding:4px 0;"
        )
        left_lay.addWidget(self.rot_summary_lbl)
        left_lay.addStretch()

        # ───── RIGHT PANEL: results ──────────────────────────────────────
        right = QWidget()
        right_lay = QVBoxLayout(right)
        right_lay.setContentsMargins(0, 0, 0, 0)
        self.rot_result_tabs = QTabWidget()
        right_lay.addWidget(self.rot_result_tabs)

        # Tab: Per-session table
        tbl_page = QWidget()
        tbl_lay = QVBoxLayout(tbl_page)
        tbl_lay.setContentsMargins(2, 2, 2, 2)
        self.rot_table = QTableWidget(0, 7)
        self.rot_table.setHorizontalHeaderLabels(
            ["Subject", "Session", "Offset", "Drift", "Confidence",
             "Held-out acc", "Sync gesture / error"]
        )
        self.rot_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        self.rot_table.horizontalHeader().setStretchLastSection(True)
        self.rot_table.verticalHeader().setVisible(False)
        self.rot_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        tbl_lay.addWidget(self.rot_table)
        self.rot_result_tabs.addTab(tbl_page, "Per-session")

        # Tab: Rotation plots — three figures stacked in a scroll area
        rot_page = QScrollArea()
        rot_page.setWidgetResizable(True)
        rot_inner = QWidget()
        rot_inner_lay = QVBoxLayout(rot_inner)
        rot_inner_lay.setContentsMargins(2, 2, 2, 2)
        self.rot_plot_offsets    = _AggPlot(min_height=260)
        self.rot_plot_drift_hist = _AggPlot(min_height=240)
        self.rot_plot_conf_hist  = _AggPlot(min_height=240)
        self.rot_plot_conf_drift = _AggPlot(min_height=260)
        rot_inner_lay.addWidget(self._titled("Offset per session", self.rot_plot_offsets))
        rot_inner_lay.addWidget(self._titled("Drift from reference (ring distance)",
                                             self.rot_plot_drift_hist))
        rot_inner_lay.addWidget(self._titled("Detector confidence",
                                             self.rot_plot_conf_hist))
        rot_inner_lay.addWidget(self._titled("Confidence vs drift (per session)",
                                             self.rot_plot_conf_drift))
        rot_inner_lay.addStretch()
        rot_page.setWidget(rot_inner)
        self.rot_result_tabs.addTab(rot_page, "Rotation plots")

        # Tab: Gesture coverage heatmap
        cov_page = QWidget()
        cov_lay = QVBoxLayout(cov_page)
        cov_lay.setContentsMargins(2, 2, 2, 2)
        self.rot_plot_coverage = _AggPlot(min_height=320)
        cov_lay.addWidget(self._titled(
            "Gesture coverage (which gestures each session has)",
            self.rot_plot_coverage,
        ))
        cov_lay.addStretch()
        self.rot_result_tabs.addTab(cov_page, "Gesture coverage")

        # Tab: Confusion matrix (held-out)
        conf_page = QScrollArea()
        conf_page.setWidgetResizable(True)
        conf_inner = QWidget()
        conf_inner_lay = QVBoxLayout(conf_inner)
        conf_inner_lay.setContentsMargins(2, 2, 2, 2)
        self.rot_plot_conf_matrix = _AggPlot(min_height=380)
        conf_inner_lay.addWidget(self._titled(
            "Confusion matrix (aggregated, row-normalised)",
            self.rot_plot_conf_matrix,
        ))
        conf_inner_lay.addStretch()
        conf_page.setWidget(conf_inner)
        self.rot_result_tabs.addTab(conf_page, "Confusion matrix")

        # Tab: per-gesture box plot
        box_page = QWidget()
        box_lay = QVBoxLayout(box_page)
        box_lay.setContentsMargins(2, 2, 2, 2)
        self.rot_plot_box = _AggPlot(min_height=320)
        box_lay.addWidget(self._titled(
            "Per-gesture accuracy across sessions",
            self.rot_plot_box,
        ))
        box_lay.addStretch()
        self.rot_result_tabs.addTab(box_page, "Per-gesture spread")

        # Tab: Pass-rate bars (rotation checks)
        pass_page = QWidget()
        pass_lay = QVBoxLayout(pass_page)
        pass_lay.setContentsMargins(2, 2, 2, 2)
        self.rot_plot_pass = _AggPlot(min_height=280)
        pass_lay.addWidget(self._titled(
            "Validation check pass rates",
            self.rot_plot_pass,
        ))
        pass_lay.addStretch()
        self.rot_result_tabs.addTab(pass_page, "Check pass rates")

        # ───── assemble splitter ─────────────────────────────────────────
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([320, 720])

        # State holders for the synchronous run
        self._rot_cancel_requested: bool = False
        self._rot_session_cache: Dict[Tuple[str, str], Any] = {}
        self._rot_last_report = None
        self._rot_last_batch  = None

        return w

    # ─── helpers used by the new layout ────────────────────────────────

    def _titled(self, title: str, widget: QWidget) -> QWidget:
        """Wrap a plot widget with a small title above it."""
        box = QWidget()
        lay = QVBoxLayout(box)
        lay.setContentsMargins(0, 0, 0, 0)
        lbl = QLabel(title)
        lbl.setStyleSheet("font-weight:600; color:#334155; padding:2px 4px;")
        lay.addWidget(lbl)
        lay.addWidget(widget, 1)
        return box

    # ------------------------------------------------------------------
    # Live calibration
    # ------------------------------------------------------------------

    @Slot()
    def _on_live_start(self) -> None:
        if self.device is None or not getattr(self.device, "is_streaming", False):
            QMessageBox.warning(
                self, "Device not streaming",
                "Please connect a device and start streaming before live calibration."
            )
            return

        self._live_idx      = 0
        self._live_countdown = 3
        self._gesture_data  = {}
        self._live_buffer   = []
        self._is_recording  = False

        self.live_start_btn.setEnabled(False)
        self.live_cancel_btn.setEnabled(True)
        self.live_progress.setValue(0)
        self._live_set("GET READY",
                       f"First: {self.CALIBRATION_GESTURES[0][0]}",
                       f"Starting in {self._live_countdown}s")

        # Use 1-second ticks for the countdown phase
        try:
            self._recording_timer.timeout.disconnect()
        except Exception:
            pass
        self._recording_timer.timeout.connect(self._live_countdown_tick)
        self._recording_timer.start(1000)

    @Slot()
    def _on_live_cancel(self) -> None:
        self._recording_timer.stop()
        try:
            self._recording_timer.timeout.disconnect()
        except Exception:
            pass
        self._is_recording = False
        self._live_buffer  = []
        self._live_disconnect()
        self.live_start_btn.setEnabled(True)
        self.live_cancel_btn.setEnabled(False)
        self._live_set("Cancelled", "", "")

    def _live_countdown_tick(self) -> None:
        self._live_countdown -= 1
        if self._live_countdown > 0:
            g = self.CALIBRATION_GESTURES[self._live_idx][0]
            self._live_set("GET READY", f"Next: {g}", f"Starting in {self._live_countdown}s")
        else:
            # Switch to 100 ms recording ticks
            self._recording_timer.stop()
            try:
                self._recording_timer.timeout.disconnect()
            except Exception:
                pass
            self._is_recording   = True
            self._live_buffer    = []
            self._live_remaining = self._recording_duration
            g, instr = self.CALIBRATION_GESTURES[self._live_idx]
            emoji = _GESTURE_EMOJIS.get(g, "")
            self._live_set(f"{emoji} {g.upper()}" if emoji else g.upper(),
                           instr, f"Hold… {self._live_remaining:.0f}s")
            self.live_progress.setValue(int(self._recording_duration * 10))
            self._live_connect()
            self._recording_timer.timeout.connect(self._live_record_tick)
            self._recording_timer.start(100)

    def _live_record_tick(self) -> None:
        self._live_remaining -= 0.1
        self.live_progress.setValue(max(0, int(self._live_remaining * 10)))
        if self._live_remaining > 0:
            return

        # Done recording this gesture
        self._recording_timer.stop()
        try:
            self._recording_timer.timeout.disconnect()
        except Exception:
            pass
        self._is_recording = False
        self._live_disconnect()

        g, _ = self.CALIBRATION_GESTURES[self._live_idx]
        if self._live_buffer:
            arr = np.vstack(self._live_buffer)
            self._gesture_data[g] = arr
            self._log(f"✓ '{g}': {arr.shape[0]} samples")
        else:
            self._log(f"⚠ No data for '{g}' — skipped")

        self._live_idx += 1
        if self._live_idx >= len(self.CALIBRATION_GESTURES):
            self._live_finish()
            return

        # Pause then countdown for next gesture
        self._live_countdown = 2
        nxt = self.CALIBRATION_GESTURES[self._live_idx][0]
        self._live_set("DONE", f"Next: {nxt}", f"Pause {self._live_countdown}s")
        self.live_progress.setValue(0)
        self._recording_timer.timeout.connect(self._live_countdown_tick)
        self._recording_timer.start(1000)

    def _live_finish(self) -> None:
        self._recording_timer.stop()
        try:
            self._recording_timer.timeout.disconnect()
        except Exception:
            pass
        self.live_start_btn.setEnabled(True)
        self.live_cancel_btn.setEnabled(False)

        if not self._gesture_data:
            self._live_set("Failed", "No data collected.", "")
            return

        self._live_set("COMPUTING", "Analysing patterns…", "")
        try:
            device_name = getattr(self.device, "name", "unknown")
            result = self.calibrator.calibrate(
                calibration_data=self._gesture_data,
                device_name=device_name,
            )
            self._calibration_result = result
            self.finish_btn.setEnabled(True)
            self.calibration_complete.emit(result)
            self._log("Live calibration complete!")
            self._log(f"  Offset: {result.rotation_offset} ch  "
                      f"Confidence: {result.confidence:.2%}")
            self._live_set(
                "COMPLETE ✓",
                f"Offset {result.rotation_offset} ch  |  "
                f"Confidence {result.confidence:.0%}",
                "Click 'Apply & Close' or check the Validate tab.",
            )
            self._run_free_checks_internal(silent=False)
        except Exception as e:  # noqa: BLE001
            self._log(f"Live calibration error: {e}")
            self._live_set("ERROR", str(e), "")

    def _live_set(self, title: str, instruction: str, detail: str) -> None:
        self.live_gesture_lbl.setText(title)
        self.live_instruction_lbl.setText(instruction)
        self.live_status_lbl.setText(detail)

    def _live_connect(self) -> None:
        if self.device and hasattr(self.device, "data_ready"):
            try:
                self.device.data_ready.connect(self._on_emg_data)
            except Exception:
                pass

    def _live_disconnect(self) -> None:
        if self.device and hasattr(self.device, "data_ready"):
            try:
                self.device.data_ready.disconnect(self._on_emg_data)
            except Exception:
                pass

    @Slot(object)
    def _on_emg_data(self, data) -> None:
        if self._is_recording and data is not None:
            self._live_buffer.append(np.asarray(data).copy())

    # ------------------------------------------------------------------
    # Session-based calibration
    # ------------------------------------------------------------------

    def _refresh_sessions(self) -> None:
        if self.data_manager is None:
            return
        try:
            self.ses_subject_combo.blockSignals(True)
            self.ses_subject_combo.clear()
            subjects = self.data_manager.list_subjects()
            self.ses_subject_combo.addItems(subjects)
            self.ses_subject_combo.blockSignals(False)
            if subjects:
                self._on_subject_changed(self.ses_subject_combo.currentText())
        except Exception as e:  # noqa: BLE001
            self._log(f"Could not list subjects: {e}")

    @Slot(str)
    def _on_subject_changed(self, subject: str) -> None:
        self.ses_session_combo.clear()
        if not subject or self.data_manager is None:
            return
        try:
            sessions = self.data_manager.list_sessions(subject)
            self.ses_session_combo.addItems(sessions)
            if sessions:
                self.ses_session_combo.setCurrentText(sessions[-1])
        except Exception as e:  # noqa: BLE001
            self._log(f"Could not list sessions: {e}")

    @Slot()
    def _on_calibrate_from_session(self) -> None:
        subject    = self.ses_subject_combo.currentText()
        session_id = self.ses_session_combo.currentText()
        if not subject or not session_id:
            QMessageBox.warning(
                self, "Nothing selected",
                "Select a subject and session.\n"
                "If the list is empty, record a session first."
            )
            return

        self.ses_cal_btn.setEnabled(False)
        self.ses_status_lbl.setText("Loading session…")
        self._log(f"Loading {subject}/{session_id}…")

        try:
            session = self.data_manager.load_session(subject, session_id)
        except Exception as e:  # noqa: BLE001
            self._log(f"Load error: {e}")
            self.ses_cal_btn.setEnabled(True)
            self.ses_status_lbl.setText("Load failed.")
            QMessageBox.critical(self, "Load error", str(e))
            return

        if not getattr(session, "trials", None):
            self.ses_cal_btn.setEnabled(True)
            self.ses_status_lbl.setText("No trials in session.")
            QMessageBox.warning(
                self, "No trials",
                f"Session '{session_id}' has no recorded trials."
            )
            return

        self.ses_status_lbl.setText("Calibrating…")
        try:
            num_ch = getattr(getattr(session, "metadata", None),
                             "num_channels", None)
            if num_ch:
                self.calibrator.processor.num_channels = num_ch
            result = self.calibrator.calibrate_from_session(session)
        except Exception as e:  # noqa: BLE001
            self._log(f"Calibration error: {e}")
            self.ses_cal_btn.setEnabled(True)
            self.ses_status_lbl.setText("Calibration failed.")
            QMessageBox.critical(self, "Calibration failed", str(e))
            return

        self._calibration_result = result
        self.ses_cal_btn.setEnabled(True)
        self.finish_btn.setEnabled(True)
        self.calibration_complete.emit(result)
        self.ses_status_lbl.setText(
            f"✓ Offset: {result.rotation_offset} ch  "
            f"Confidence: {result.confidence:.2%}"
        )

        # Log details
        valid_trials = (session.get_valid_trials()
                        if hasattr(session, "get_valid_trials") else [])
        if valid_trials:
            gestures = {t.gesture_name for t in valid_trials}
            self._log(f"  Gestures: {', '.join(sorted(gestures))}")
            self._log(f"  Valid trials: {len(valid_trials)}")
        self._log(f"  Offset: {result.rotation_offset} ch  "
                  f"Confidence: {result.confidence:.2%}")

        # Auto-save as reference when none exists
        saved_new_ref = False
        if not self.calibrator.has_reference:
            self.calibrator.save_as_reference(result)
            self._log("  No reference existed — saved as reference.")
            saved_new_ref = True

        QMessageBox.information(
            self, "Calibration complete",
            f"{subject} / {session_id}\n"
            f"Offset: {result.rotation_offset} ch\n"
            f"Confidence: {result.confidence:.2%}"
            + ("\n\nSaved as new reference calibration." if saved_new_ref else "")
            + "\n\nClick 'Apply & Close' to use this calibration."
        )

        # Extract gesture arrays for the validator
        self._gesture_data = {}
        try:
            all_data = session.get_data()
            for trial in valid_trials:
                chunk = all_data[trial.start_sample:trial.end_sample]
                if chunk.shape[0] >= 10:
                    self._gesture_data.setdefault(trial.gesture_name, [])
                    self._gesture_data[trial.gesture_name].append(chunk)
            for g, chunks in list(self._gesture_data.items()):
                self._gesture_data[g] = np.vstack(chunks)
        except Exception:  # noqa: BLE001
            pass

        # Run the free checks once. Held-out is opt-in via the Validate tab.
        self._run_free_checks_internal(silent=False)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _get_validator_class(self):
        try:
            from playagain_pipeline.calibration.calibration_validation import (
                CalibrationValidator,
            )
            return CalibrationValidator
        except ImportError:
            return None

    @Slot()
    def _on_run_free_checks(self) -> None:
        if self._calibration_result is None:
            QMessageBox.information(
                self, "No calibration yet",
                "Run a calibration first (Live or From Session tab)."
            )
            return
        if not self._gesture_data:
            QMessageBox.information(
                self, "No gesture data",
                "The free checks need the gesture data captured during calibration. "
                "Re-run calibration, then validate."
            )
            return
        self._run_free_checks_internal(silent=False)

    def _run_free_checks_internal(self, *, silent: bool = False) -> None:
        ValidatorClass = self._get_validator_class()
        if ValidatorClass is None or not self._gesture_data:
            return
        try:
            report = ValidatorClass(self.calibrator).run_all(
                gesture_data=self._gesture_data
            )
        except Exception as e:  # noqa: BLE001
            self._log(f"Validation error: {e}")
            return

        self._render_report(report, header=None, silent=silent)

    @Slot()
    def _on_run_held_out_check(self) -> None:
        if self._calibration_result is None:
            QMessageBox.information(
                self, "No calibration yet",
                "Run a calibration first, then validate against a session."
            )
            return
        ValidatorClass = self._get_validator_class()
        if ValidatorClass is None:
            QMessageBox.warning(
                self, "Module missing",
                "calibration_validation.py not found. "
                "Add it to playagain_pipeline/calibration/."
            )
            return
        if self.data_manager is None:
            QMessageBox.warning(
                self, "No data manager",
                "Held-out validation needs a DataManager. "
                "Open the dialog from the main window."
            )
            return

        # Mini session picker
        picker = QDialog(self)
        picker.setWindowTitle("Pick held-out session")
        picker.resize(320, 140)
        pl = QVBoxLayout(picker)
        pf = QFormLayout()
        subj_combo = QComboBox()
        sess_combo = QComboBox()
        try:
            subj_combo.addItems(self.data_manager.list_subjects())
        except Exception:  # noqa: BLE001
            pass

        def _upd(s: str) -> None:
            sess_combo.clear()
            if not s:
                return
            try:
                sessions = self.data_manager.list_sessions(s)
                sess_combo.addItems(sessions)
                if sessions:
                    sess_combo.setCurrentText(sessions[-1])
            except Exception:  # noqa: BLE001
                pass

        subj_combo.currentTextChanged.connect(_upd)
        if subj_combo.count():
            _upd(subj_combo.currentText())
        pf.addRow("Subject:", subj_combo)
        pf.addRow("Session:", sess_combo)
        pl.addLayout(pf)
        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(picker.accept)
        btns.rejected.connect(picker.reject)
        pl.addWidget(btns)

        if not picker.exec():
            return

        subject    = subj_combo.currentText()
        session_id = sess_combo.currentText()
        if not subject or not session_id:
            return

        try:
            session = self.data_manager.load_session(subject, session_id)
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "Load error", str(e))
            return

        try:
            report = ValidatorClass(self.calibrator).run_all(
                gesture_data=self._gesture_data,
                held_out_session=session,
            )
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, "Validation failed", str(e))
            return

        self._render_report(report, header=f"Held-out vs {session_id}", silent=False)

    def _render_report(self, report, *, header: Optional[str] = None,
                       silent: bool = False) -> None:
        """Render a CalibrationReport to the validate panel + log.

        ``silent=True`` skips the log spam (used to avoid duplicating
        output when the same report is rendered to the panel only).
        """
        if not silent:
            if header:
                self._log(f"{header}:")
            else:
                self._log("Validation checks:")
            for check in report.checks:
                self._log(check.line())
            verdict = self._verdict_text(report)
            self._log(f"  → {verdict}")
            for g, acc in sorted(report.per_gesture_accuracy.items()):
                self._log(f"  per-gesture {g}: {acc:.2%}")

        # Panel display
        lines: List[str] = [
            f"Offset: {report.rotation_offset}  "
            f"Confidence: {report.confidence:.0%}",
            self._verdict_text(report),
        ]
        if header:
            lines.append(header)
        lines.append("")
        lines += [c.line() for c in report.checks]
        if report.per_gesture_accuracy:
            lines += ["", "Per-gesture accuracy:"]
            lines += [
                f"  {g}: {acc:.2%}"
                for g, acc in sorted(report.per_gesture_accuracy.items())
            ]
        # Confusion matrix (compact)
        if report.confusion:
            lines += ["", "Confusion (true → predicted):"]
            for true_g, row in sorted(report.confusion.items()):
                preds = ", ".join(
                    f"{p}×{n}" for p, n in sorted(row.items(), key=lambda kv: -kv[1])
                )
                lines.append(f"  {true_g} → {preds}")

        self.val_result_lbl.setText("\n".join(lines))

    @staticmethod
    def _verdict_text(report) -> str:
        if not report.is_acceptable:
            return "✗ Failed: at least one error-level check did not pass."
        if report.has_warnings:
            return "✓ Acceptable (with warnings — see ! marked checks)."
        return "✓ Looks good."

    # ------------------------------------------------------------------
    # Rotation Stats tab
    # ------------------------------------------------------------------

    def _refresh_rot_subjects(self) -> None:
        if self.data_manager is None:
            return
        try:
            self.rot_subject_combo.blockSignals(True)
            self.rot_subject_combo.clear()
            self.rot_subject_combo.addItem("(all subjects)", userData="__all__")
            for s in self.data_manager.list_subjects():
                self.rot_subject_combo.addItem(s, userData=s)
        except Exception as e:  # noqa: BLE001
            self._log(f"Could not list subjects: {e}")
        finally:
            self.rot_subject_combo.blockSignals(False)
        self._refresh_rot_sessions()

    def _refresh_rot_sessions(self) -> None:
        # Block itemChanged while we're rebuilding so the gesture-list
        # refresh doesn't fire once per added item.
        self.rot_session_list.blockSignals(True)
        try:
            self.rot_session_list.clear()
            if self.data_manager is None:
                return
            chosen = self.rot_subject_combo.currentData()
            try:
                if chosen == "__all__":
                    subjects = self.data_manager.list_subjects()
                else:
                    subjects = [chosen] if chosen else []
                for subj in subjects:
                    try:
                        sessions = self.data_manager.list_sessions(subj)
                    except Exception:  # noqa: BLE001
                        sessions = []
                    for sid in sessions:
                        item = QListWidgetItem(f"{subj} / {sid}")
                        item.setData(Qt.ItemDataRole.UserRole, (subj, sid))
                        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                        item.setCheckState(Qt.CheckState.Checked)  # default: include
                        self.rot_session_list.addItem(item)
            except Exception as e:  # noqa: BLE001
                self._log(f"Could not list sessions: {e}")
        finally:
            self.rot_session_list.blockSignals(False)
        # After the session list changes, refresh the gesture-coverage view
        self._refresh_rot_gesture_list()

    def _on_rot_session_toggled(self, _item) -> None:
        # Light-weight refresh — just re-counts coverage from the cache.
        self._refresh_rot_gesture_list()

    def _set_all_rot_session_ticks(self, on: bool) -> None:
        state = Qt.CheckState.Checked if on else Qt.CheckState.Unchecked
        self.rot_session_list.blockSignals(True)
        try:
            for i in range(self.rot_session_list.count()):
                self.rot_session_list.item(i).setCheckState(state)
        finally:
            self.rot_session_list.blockSignals(False)
        self._refresh_rot_gesture_list()

    def _selected_rot_sessions(self) -> List:
        out = []
        for i in range(self.rot_session_list.count()):
            it = self.rot_session_list.item(i)
            if it.checkState() == Qt.CheckState.Checked:
                out.append(it.data(Qt.ItemDataRole.UserRole))
        return out

    # ─── gesture filter ────────────────────────────────────────────────

    def _refresh_rot_gesture_list(self) -> None:
        """Re-scan ticked sessions for their gestures and populate the list.

        Only sessions whose data we've already loaded (cached) contribute
        accurate counts. Un-cached sessions are listed as "?" so the user
        knows the count will be exact only after the first run.
        """
        self.rot_gesture_list.blockSignals(True)
        try:
            # Preserve current ticked gestures
            previously_ticked: Set[str] = set()
            for i in range(self.rot_gesture_list.count()):
                it = self.rot_gesture_list.item(i)
                if it.checkState() == Qt.CheckState.Checked:
                    previously_ticked.add(
                        str(it.data(Qt.ItemDataRole.UserRole) or "")
                    )

            self.rot_gesture_list.clear()
            ticked_sessions = self._selected_rot_sessions()  # [(subj, sid), ...]

            # Use the cache (filled during the latest analysis run).
            try:
                from playagain_pipeline.calibration.calibration_validation import (
                    gestures_in_session,
                )
            except ImportError:
                self.rot_gesture_list.addItem(
                    "(install calibration_validation.py to enable gesture filter)"
                )
                return

            counts: Dict[str, int] = {}
            scanned = 0
            for key in ticked_sessions:
                ses = self._rot_session_cache.get(tuple(key))
                if ses is None:
                    continue
                scanned += 1
                for g in gestures_in_session(ses):
                    counts[g] = counts.get(g, 0) + 1

            if not counts:
                self.rot_gesture_list.addItem(
                    "(gesture coverage shown after first analysis run)"
                )
                return

            # Sort: most frequent first, alphabetic tiebreak
            n_sessions = max(scanned, 1)
            for g in sorted(counts.keys(),
                            key=lambda k: (-counts[k], k)):
                pct = counts[g] / n_sessions
                emoji = _GESTURE_EMOJIS.get(g, "•")
                item = QListWidgetItem(
                    f"{emoji}  {g}  ({counts[g]}/{n_sessions})"
                )
                item.setData(Qt.ItemDataRole.UserRole, g)
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                # Default: tick gestures present in ALL ticked sessions
                # (most likely what the user wants for a fair comparison).
                if previously_ticked:
                    state = (Qt.CheckState.Checked
                             if g in previously_ticked
                             else Qt.CheckState.Unchecked)
                else:
                    state = (Qt.CheckState.Checked
                             if pct >= 1.0
                             else Qt.CheckState.PartiallyChecked
                             if pct >= 0.5
                             else Qt.CheckState.Unchecked)
                item.setCheckState(state)
                self.rot_gesture_list.addItem(item)
        finally:
            self.rot_gesture_list.blockSignals(False)

    def _set_all_rot_gesture_ticks(self, on: bool) -> None:
        state = Qt.CheckState.Checked if on else Qt.CheckState.Unchecked
        for i in range(self.rot_gesture_list.count()):
            it = self.rot_gesture_list.item(i)
            if it.flags() & Qt.ItemFlag.ItemIsUserCheckable:
                it.setCheckState(state)

    def _tick_common_gestures(self) -> None:
        """Tick only gestures that are in *every* selected session."""
        for i in range(self.rot_gesture_list.count()):
            it = self.rot_gesture_list.item(i)
            text = it.text()
            # Detect "(N/M)" suffix
            try:
                inside = text.rsplit("(", 1)[1].rstrip(")")
                n, m = inside.split("/")
                state = (Qt.CheckState.Checked
                         if int(n) == int(m)
                         else Qt.CheckState.Unchecked)
            except (ValueError, IndexError):
                state = Qt.CheckState.Unchecked
            if it.flags() & Qt.ItemFlag.ItemIsUserCheckable:
                it.setCheckState(state)

    def _selected_gestures(self) -> Set[str]:
        out: Set[str] = set()
        for i in range(self.rot_gesture_list.count()):
            it = self.rot_gesture_list.item(i)
            if not (it.flags() & Qt.ItemFlag.ItemIsUserCheckable):
                continue
            if it.checkState() == Qt.CheckState.Checked:
                g = it.data(Qt.ItemDataRole.UserRole)
                if g:
                    out.add(str(g))
        return out

    @Slot()
    def _on_run_rotation_study(self) -> None:
        """Run the rotation study synchronously on the main thread.

        Why synchronous: ``data_manager.load_session`` typically reaches
        h5py / HDF5 / file-mapped numpy arrays which are *not*
        thread-safe at the C level. Calling them from a QThread on
        macOS Apple Silicon (and sometimes Linux) can SIGBUS — that is
        what crashed the old worker-thread version. We instead loop on
        the main thread and call ``QApplication.processEvents()``
        between sessions so the UI stays responsive without sharing any
        state across threads.
        """
        try:
            from playagain_pipeline.calibration.calibration_validation import (
                BatchValidator,
                RotationDetectionStudy,
            )
        except ImportError:
            QMessageBox.warning(
                self, "Module missing",
                "calibration_validation.py is missing — drop the new "
                "version into playagain_pipeline/calibration/."
            )
            return

        if self.data_manager is None:
            QMessageBox.warning(
                self, "No data manager",
                "Rotation study needs a DataManager. Open the dialog "
                "from the main window."
            )
            return

        keys = self._selected_rot_sessions()
        if not keys:
            QMessageBox.information(
                self, "Nothing selected",
                "Tick at least one session to analyze."
            )
            return

        run_held_out = bool(self.rot_chk_held_out.isChecked())
        gesture_filter = self._selected_gestures() if run_held_out else set()
        strict = bool(self.rot_mode_strict.isChecked())

        # Reset UI state
        self.rot_table.setRowCount(0)
        self.rot_summary_lbl.setText("")
        self.rot_progress.setRange(0, len(keys))
        self.rot_progress.setValue(0)
        self.rot_progress.setFormat("Loading sessions…")
        self.rot_run_btn.setEnabled(False)
        self.rot_cancel_btn.setEnabled(True)
        self._rot_cancel_requested = False
        QApplication.processEvents()

        self._log(f"Rotation study: {len(keys)} sessions queued"
                  + (f" · gesture filter {sorted(gesture_filter)}"
                     if gesture_filter else "")
                  + (" · strict" if strict else "")
                  + (" · held-out classification" if run_held_out else ""))

        # ── Phase 1: load sessions on the MAIN thread ────────────────
        loaded: List[Tuple[str, str, Any]] = []
        for i, key in enumerate(keys, start=1):
            if self._rot_cancel_requested:
                break
            subj, sid = key
            self.rot_progress.setFormat(f"Loading {subj}/{sid}  ({i}/{len(keys)})")
            self.rot_progress.setValue(i - 1)
            QApplication.processEvents()

            cache_key = (subj, sid)
            session = self._rot_session_cache.get(cache_key)
            if session is None:
                try:
                    session = self.data_manager.load_session(subj, sid)
                    self._rot_session_cache[cache_key] = session
                except Exception as e:  # noqa: BLE001
                    self._log(f"  ⚠ {subj}/{sid}: load failed — {e}")
                    continue
            loaded.append((subj, sid, session))

        if self._rot_cancel_requested:
            self._finish_rot_run("✗ Cancelled.")
            return
        if not loaded:
            self._finish_rot_run("✗ No sessions could be loaded.")
            return

        # ── Phase 2: optional strict-mode pre-filter ─────────────────
        skipped_strict: List[Tuple[str, str, str]] = []
        if strict and gesture_filter:
            from playagain_pipeline.calibration.calibration_validation import (
                gestures_in_session,
            )
            kept: List[Tuple[str, str, Any]] = []
            for subj, sid, ses in loaded:
                have = gestures_in_session(ses)
                missing = gesture_filter - have
                if missing:
                    skipped_strict.append(
                        (subj, sid, f"missing {sorted(missing)}")
                    )
                else:
                    kept.append((subj, sid, ses))
            self._log(
                f"Strict mode: kept {len(kept)}/{len(loaded)} sessions "
                f"(dropped {len(skipped_strict)} for missing gestures)"
            )
            loaded = kept

        # ── Phase 3: rotation study (always, fast) ───────────────────
        self.rot_progress.setRange(0, len(loaded) * (2 if run_held_out else 1))
        self.rot_progress.setValue(0)
        self.rot_progress.setFormat("Analyzing rotation…")
        QApplication.processEvents()

        rot_study = RotationDetectionStudy(self.calibrator)
        for subj, sid, ses in loaded:
            rot_study.add_session(subj, sid, ses)

        def _rot_progress(idx, total, label):
            if self._rot_cancel_requested:
                raise RuntimeError("cancelled")
            self.rot_progress.setRange(0, total)
            self.rot_progress.setValue(idx)
            self.rot_progress.setFormat(label)
            QApplication.processEvents()

        try:
            rot_report = rot_study.analyze(progress=_rot_progress)
        except RuntimeError as e:
            if "cancelled" in str(e):
                self._finish_rot_run("✗ Cancelled.")
                return
            self._log(f"Rotation study failed: {e}")
            self._finish_rot_run(f"✗ Failed: {e}")
            return
        except Exception as e:  # noqa: BLE001
            self._log(f"Rotation study failed: {e}")
            self._finish_rot_run(f"✗ Failed: {e}")
            return

        # ── Phase 4: optional held-out validation ────────────────────
        batch_report = None
        if run_held_out and not self._rot_cancel_requested:
            self.rot_progress.setFormat("Held-out validation…")
            QApplication.processEvents()

            bv = BatchValidator(self.calibrator)
            try:
                batch_report = bv.run(
                    loaded,
                    gesture_filter=(gesture_filter or None),
                    progress_cb=lambda i, n, lab: (
                        self.rot_progress.setRange(0, n),
                        self.rot_progress.setValue(i),
                        self.rot_progress.setFormat(f"Held-out: {lab}"),
                    ) if not self._rot_cancel_requested else None,
                    event_pump=QApplication.processEvents,
                )
            except Exception as e:  # noqa: BLE001
                self._log(f"Held-out validation failed: {e}")
                # Continue — we still have rotation results to show.

        # Save the user's prior calibration (BatchValidator restores per-session
        # but the last call to calibrate_from_session in RotationDetectionStudy
        # may have left a stale current_calibration — refresh it from the
        # saved reference so the rest of the dialog reads consistent state).
        self._rot_last_report = rot_report
        self._rot_last_batch  = batch_report

        # Populate UI
        self._populate_rot_table(rot_report, batch_report, skipped_strict)
        self.rot_summary_lbl.setText(self._compose_summary(
            rot_report, batch_report, skipped_strict
        ))
        self._render_rotation_plots(rot_report, batch_report, loaded)
        self._refresh_rot_gesture_list()  # update counts now that cache is filled

        self._finish_rot_run("✓ Done.")
        self._log("Rotation study finished. " + rot_report.summary())
        if batch_report is not None and batch_report.reports:
            agg = batch_report.aggregate_held_out_details()
            self._log(
                f"Held-out: {agg.n_trials} trials across "
                f"{len(batch_report.reports)} sessions, "
                f"aggregate accuracy {agg.overall_accuracy:.1%}"
            )

    @Slot()
    def _on_cancel_rotation_study(self) -> None:
        self._rot_cancel_requested = True
        self.rot_cancel_btn.setEnabled(False)
        self.rot_progress.setFormat("Cancelling…")

    def _finish_rot_run(self, status: str) -> None:
        self.rot_run_btn.setEnabled(True)
        self.rot_cancel_btn.setEnabled(False)
        self.rot_progress.setFormat(status)

    # ─── results population & plot rendering ───────────────────────────

    def _compose_summary(self, rot_report, batch_report, skipped_strict) -> str:
        bits = [rot_report.summary()]
        if skipped_strict:
            bits.append(f"Strict mode dropped {len(skipped_strict)} session(s).")
        if batch_report is not None and batch_report.reports:
            agg = batch_report.aggregate_held_out_details()
            bits.append(
                f"Held-out: {agg.overall_accuracy:.1%} on {agg.n_trials} trials, "
                f"{len(batch_report.reports)} sessions."
            )
        return "  •  ".join(bits)

    def _populate_rot_table(self, rot_report, batch_report,
                            skipped_strict) -> None:
        # Build accuracy lookup keyed by (subject, session_id)
        ho_lookup: Dict[Tuple[str, str], float] = {}
        if batch_report is not None:
            for r in batch_report.reports:
                d = r.held_out_details
                if d is None or d.n_trials == 0:
                    continue
                ho_lookup[(r.subject_id or "?", r.session_id or "?")] = d.overall_accuracy

        rows = list(rot_report.results)
        # Append strict-mode skipped rows so the user sees them
        for subj, sid, reason in skipped_strict:
            from playagain_pipeline.calibration.calibration_validation import (
                SessionRotationResult,
            )
            rows.append(SessionRotationResult(
                subject_id=subj, session_id=sid,
                rotation_offset=0, confidence=0.0,
                sync_gesture="", n_trials=0,
                drift_from_reference=0,
                error=f"strict-mode skip: {reason}",
            ))

        self.rot_table.setRowCount(len(rows))
        for row, r in enumerate(rows):
            self.rot_table.setItem(row, 0, QTableWidgetItem(r.subject_id))
            self.rot_table.setItem(row, 1, QTableWidgetItem(r.session_id))
            if r.ok:
                self.rot_table.setItem(row, 2, QTableWidgetItem(f"{r.rotation_offset}"))
                self.rot_table.setItem(row, 3, QTableWidgetItem(f"{r.drift_from_reference}"))
                self.rot_table.setItem(row, 4, QTableWidgetItem(f"{r.confidence:.0%}"))
                ho = ho_lookup.get((r.subject_id, r.session_id))
                self.rot_table.setItem(
                    row, 5,
                    QTableWidgetItem(f"{ho:.0%}" if ho is not None else "—"),
                )
                self.rot_table.setItem(row, 6, QTableWidgetItem(r.sync_gesture))
            else:
                for c in range(2, 6):
                    self.rot_table.setItem(row, c, QTableWidgetItem("—"))
                err = QTableWidgetItem(f"⚠ {r.error}")
                self.rot_table.setItem(row, 6, err)

    def _render_rotation_plots(
        self, rot_report, batch_report, loaded: List[Tuple[str, str, Any]]
    ) -> None:
        """Render every plot from the most recent run."""
        # 1) Offset bar chart
        self.rot_plot_offsets.render_with(
            lambda fig: _draw_offset_bars(fig, rot_report)
        )
        # 2) Drift histogram
        self.rot_plot_drift_hist.render_with(
            lambda fig: _draw_drift_hist(fig, rot_report)
        )
        # 3) Confidence histogram
        self.rot_plot_conf_hist.render_with(
            lambda fig: _draw_conf_hist(fig, rot_report)
        )
        # 4) Confidence vs drift scatter
        self.rot_plot_conf_drift.render_with(
            lambda fig: _draw_conf_vs_drift(fig, rot_report)
        )
        # 5) Gesture coverage heatmap (independent of held-out)
        self.rot_plot_coverage.render_with(
            lambda fig: _draw_gesture_coverage(fig, loaded)
        )
        # 6) Confusion matrix (only if we have held-out data)
        if batch_report is not None and batch_report.reports:
            self.rot_plot_conf_matrix.render_with(
                lambda fig: _draw_aggregate_confusion(fig, batch_report)
            )
            # 7) Per-gesture box plot
            self.rot_plot_box.render_with(
                lambda fig: _draw_per_gesture_box(fig, batch_report)
            )
            # 8) Pass-rate bars
            self.rot_plot_pass.render_with(
                lambda fig: _draw_pass_rates(fig, batch_report)
            )
        else:
            self.rot_plot_conf_matrix.show_message(
                "Tick 'Run held-out gesture classification' to compute the "
                "confusion matrix."
            )
            self.rot_plot_box.show_message(
                "Tick 'Run held-out gesture classification' to compute "
                "per-gesture spread."
            )
            self.rot_plot_pass.show_message(
                "Tick 'Run held-out gesture classification' to compute "
                "check pass rates."
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _log(self, msg: str) -> None:
        self.log_text.append(msg)

    def get_calibration_result(self):
        return self._calibration_result

    def closeEvent(self, event) -> None:
        self._recording_timer.stop()
        self._live_disconnect()
        self._is_recording = False
        # Signal a graceful stop in case a synchronous run is in progress
        # (the loop checks this flag between sessions).
        self._rot_cancel_requested = True
        super().closeEvent(event)