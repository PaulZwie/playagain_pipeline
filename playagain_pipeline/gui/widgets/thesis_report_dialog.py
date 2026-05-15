"""
gui/widgets/thesis_report_dialog.py
═══════════════════════════════════
A self-contained dialog that:

    1.  Builds the standard set of validation_runs/ that Chapter 6
        depends on (LOSO-session, LOSO-subject, the 8-feature ablation
        suite, the 4 cross-domain runs), driving the existing
        :class:`ValidationRunner` in a worker thread.

    2.  Lets the user map already-completed runs on disk to their
        thesis-section roles (§6.3 primary, §6.4 LOSO-subject, §6.5
        ablation, §6.6 cross-domain).

    3.  Triggers the chapter-6/7/8 report generator
        (:func:`playagain_pipeline.validation.generate_thesis_outputs.run`)
        which writes every table and figure referenced from
        ``06_Results.tex``, ``07_Discussion.tex`` and
        ``08_Conclusion_and_Outlook.tex``.

Why a dialog and not a tab
──────────────────────────
The thesis output generation is a one-shot operation performed at
write-up time, not a recurring activity that needs permanent screen
real estate. Putting it behind ``Tools ▸ Build thesis report…`` keeps
the main workflow uncluttered and makes the action discoverable to
the writer without surprising day-to-day users of the recording GUI.

Threading model
───────────────
* Validation runs go through :class:`_SuiteRunWorker` — a QThread that
  loops over a list of :class:`ExperimentConfig` objects, calling
  :func:`ValidationRunner.run` once per config.  Mirrors the existing
  :class:`_SweepWorker` in ``evaluation_tab.py``.
* Output generation goes through :class:`_ReportBuildWorker` — a QThread
  that invokes the orchestrator and proxies log-records back to the UI
  through a :class:`QObject` bridge.

Neither path touches the GUI thread for long-running work, so the
dialog stays responsive (and cancellable) throughout.
"""

from __future__ import annotations

import logging
import platform
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from PySide6.QtCore import Qt, Signal, Slot, QObject, QThread, QTimer
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QFileDialog, QFormLayout, QFrame,
    QGridLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit,
    QMessageBox, QProgressBar, QPushButton, QScrollArea, QSizePolicy,
    QSplitter, QTableWidget, QTableWidgetItem, QTextEdit, QToolButton,
    QVBoxLayout, QWidget,
)

# ── Validation backend ────────────────────────────────────────────────────
# These mirror the imports already used in evaluation_tab.py, so the
# dialog plays well inside the same project layout.
from playagain_pipeline.validation import (
    SessionCorpus, ExperimentConfig, ValidationRunner, RunResult,
)
from playagain_pipeline.validation.config import (
    DataSelection, WindowingConfig, FeatureConfig, ModelConfig, CVConfig,
)
from playagain_pipeline.validation.runner import FoldResult, ProgressReporter

# ── Participant-group registry (healthy / impaired cohorts) ───────────────
# Qt-free; safe to import at module load. The editor dialog is imported
# lazily inside the click handler so this module still opens if the
# editor file is missing from an older checkout.
from playagain_pipeline.validation.participant_groups import (
    GROUP_HEALTHY, GROUP_IMPAIRED, GROUP_UNKNOWN,
    ParticipantGroups, default_groups_path,
)

# ── Thesis-report generator (added in the chapter-6 bundle) ───────────────
# Imported lazily inside the build worker so the dialog opens fast even
# if matplotlib needs to initialise — matplotlib's first import is by far
# the slowest thing in the whole module graph.
# (See _ReportBuildWorker.run for the actual import.)

# ── Optional app style (same convention as evaluation_tab.py) ─────────────
try:
    from playagain_pipeline.gui.gui_style import apply_app_style  # type: ignore
except Exception:  # noqa: BLE001
    apply_app_style = None  # type: ignore[assignment]


log = logging.getLogger(__name__)


# ───────────────────────────────────────────────────────────────────────────
# Defaults
# ───────────────────────────────────────────────────────────────────────────

# Single source of truth for which models the "create suites" presets
# include. Kept in sync with _AVAILABLE_MODELS in evaluation_tab.py — if
# you add a new model there, add it here too so the thesis Table 6.3
# automatically picks it up.
_THESIS_MODELS: List[Tuple[str, str]] = [
    ("lda",           "LDA"),
    ("svm",           "SVM"),
    ("random_forest", "Random Forest"),
    ("catboost",      "CatBoost"),
    ("mlp",           "MLP"),
    ("attention_net", "Attention-CNN"),
    # The thesis text also references "CNN" and "MSTNet". They are
    # listed below conditionally — see _filter_available_models. Models
    # not present in the runner's registry are filtered out at launch
    # time so the GUI doesn't dispatch configs that would fail.
    ("cnn",           "CNN"),
    ("mstnet",        "MSTNet"),
]

# The thesis ablation table lists 8 features (rms, mav, var, wl, zc,
# ssc, iemg, ssi). The Python pipeline currently exposes the first 7 —
# ssi is computed identically to iemg² so it's usually added as part of
# the iemg feature module. We iterate whatever the user has registered;
# if "ssi" isn't a known feature the preset just produces 7 ablation
# runs and the chapter table renders 7 rows instead of 8.
_THESIS_FEATURES: List[str] = ["rms", "mav", "var", "wl", "zc", "ssc", "iemg", "ssi"]

# Default 200 ms / 50 ms windowing — what the thesis quotes in §6.1 and
# what the rest of the GUI already defaults to.
_DEFAULT_WINDOW_MS: int = 200
_DEFAULT_STRIDE_MS: int = 50
_DEFAULT_SEED:      int = 42

# Map each thesis-section role to (UI label, file dialog caption).
_ROLE_PRIMARY    = "primary"        # §6.3 LOSO-session
_ROLE_LOSO_SUBJ  = "loso_subj"      # §6.4 LOSO-subject
# The ablation and cross-domain roles use one row per feature/direction.

# Filename prefix the runner uses for output dirs:
#   <YYYY-MM-DD__HH-MM-SS>__<safe_name>/
# The dialog scans the validation_runs root for any directory containing
# a results.json — that's the reliable signal it was a runner output.


# ═══════════════════════════════════════════════════════════════════════════
# Helpers — experiment construction
# ═══════════════════════════════════════════════════════════════════════════

def _model_cfgs(models: Sequence[str]) -> List[ModelConfig]:
    """Map ['lda','catboost',...] to bare-default ModelConfig objects."""
    return [ModelConfig(type=m, params={}) for m in models]


def _feature_cfgs(features: Sequence[str]) -> List[FeatureConfig]:
    return [FeatureConfig(name=f, params={}) for f in features]


def _ts() -> str:
    """Short timestamp suffix for run names."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _filter_available_models(models: Sequence[str]) -> List[str]:
    """
    Drop model keys that aren't actually wired into the runner's
    ModelManager. The runner will raise a KeyError later if we leave
    them in. The check is best-effort — we don't have a public registry
    API to query, so we attempt an import and fall back to allowing the
    name through if the introspection fails.
    """
    try:
        from playagain_pipeline.models import ModelManager  # type: ignore
        mm = ModelManager()
        known = set(getattr(mm, "available_models", lambda: [])() or [])
        if not known:
            return list(models)  # fail open
        return [m for m in models if m in known]
    except Exception:  # noqa: BLE001
        return list(models)


def _make_primary_cfg(
    models: Sequence[str],
    *,
    window_ms: int = _DEFAULT_WINDOW_MS,
    stride_ms: int = _DEFAULT_STRIDE_MS,
    seed:      int = _DEFAULT_SEED,
    features:  Sequence[str] = ("mav", "rms", "wl", "zc", "ssc", "var", "iemg"),
) -> ExperimentConfig:
    """§6.3 — Leave-One-Session-Out across all models with combined features."""
    return ExperimentConfig(
        name=f"thesis_loso_session_{_ts()}",
        description="Chapter 6 §6.3 primary evaluation.",
        seed=seed,
        windowing=WindowingConfig(window_ms=window_ms, stride_ms=stride_ms),
        features=_feature_cfgs(features),
        models=_model_cfgs(models),
        cv=CVConfig(strategy="loso_session", kwargs={}),
    )


def _make_loso_subj_cfg(
    models: Sequence[str],
    *,
    window_ms: int = _DEFAULT_WINDOW_MS,
    stride_ms: int = _DEFAULT_STRIDE_MS,
    seed:      int = _DEFAULT_SEED,
    features:  Sequence[str] = ("mav", "rms", "wl", "zc", "ssc", "var", "iemg"),
) -> ExperimentConfig:
    """§6.4 — Leave-One-Subject-Out (classical models only per thesis text)."""
    return ExperimentConfig(
        name=f"thesis_loso_subject_{_ts()}",
        description="Chapter 6 §6.4 secondary evaluation.",
        seed=seed,
        windowing=WindowingConfig(window_ms=window_ms, stride_ms=stride_ms),
        features=_feature_cfgs(features),
        models=_model_cfgs(models),
        cv=CVConfig(strategy="loso_subject", kwargs={}),
    )


def _make_ablation_cfgs(
    primary_model: str,
    features: Sequence[str],
    *,
    window_ms: int = _DEFAULT_WINDOW_MS,
    stride_ms: int = _DEFAULT_STRIDE_MS,
    seed:      int = _DEFAULT_SEED,
) -> List[Tuple[str, ExperimentConfig]]:
    """
    §6.5 — One run per single feature + one "combined" run, all under
    LOSO-session, all with the user's chosen primary model. Returns a
    list of ``(condition_name, cfg)`` pairs — the same shape the
    chapter-6 ``feature_ablation`` aggregator consumes.
    """
    out: List[Tuple[str, ExperimentConfig]] = []
    for feat in features:
        out.append((feat, ExperimentConfig(
            name=f"thesis_ablation_{feat}_{_ts()}",
            description=f"Chapter 6 §6.5 ablation — feature: {feat}.",
            seed=seed,
            windowing=WindowingConfig(window_ms=window_ms, stride_ms=stride_ms),
            features=_feature_cfgs([feat]),
            models=_model_cfgs([primary_model]),
            cv=CVConfig(strategy="loso_session", kwargs={}),
        )))
    out.append(("combined", ExperimentConfig(
        name=f"thesis_ablation_combined_{_ts()}",
        description="Chapter 6 §6.5 ablation — all features combined.",
        seed=seed,
        windowing=WindowingConfig(window_ms=window_ms, stride_ms=stride_ms),
        features=_feature_cfgs(features),
        models=_model_cfgs([primary_model]),
        cv=CVConfig(strategy="loso_session", kwargs={}),
    )))
    return out


def _make_xdomain_cfgs(
    primary_model: str,
    *,
    window_ms: int = _DEFAULT_WINDOW_MS,
    stride_ms: int = _DEFAULT_STRIDE_MS,
    seed:      int = _DEFAULT_SEED,
    features:  Sequence[str] = ("mav", "rms", "wl", "zc", "ssc", "var", "iemg"),
) -> List[Tuple[str, ExperimentConfig]]:
    """
    §6.6 — Four runs: within-pipeline, within-unity, pipeline→unity,
    unity→pipeline. ``within_*`` uses LOSO-session restricted to the
    corresponding domain; cross-direction uses the ``cross_domain``
    strategy.
    """
    def _cfg(name: str, kwargs: Dict[str, Any], data: DataSelection,
            strategy: str) -> ExperimentConfig:
        return ExperimentConfig(
            name=f"thesis_xdomain_{name}_{_ts()}",
            description=f"Chapter 6 §6.6 cross-domain — {name}.",
            seed=seed,
            data=data,
            windowing=WindowingConfig(window_ms=window_ms, stride_ms=stride_ms),
            features=_feature_cfgs(features),
            models=_model_cfgs([primary_model]),
            cv=CVConfig(strategy=strategy, kwargs=kwargs),
        )

    return [
        ("within_pipe",   _cfg("within_pipeline",
                               {}, DataSelection(domains=["pipeline"]),
                               "loso_session")),
        ("within_unity",  _cfg("within_unity",
                               {}, DataSelection(domains=["unity"]),
                               "loso_session")),
        ("p2u",           _cfg("pipeline_to_unity",
                               {"train_domain": "pipeline", "test_domain": "unity"},
                               DataSelection(),
                               "cross_domain")),
        ("u2p",           _cfg("unity_to_pipeline",
                               {"train_domain": "unity", "test_domain": "pipeline"},
                               DataSelection(),
                               "cross_domain")),
    ]


# ═══════════════════════════════════════════════════════════════════════════
# Run discovery
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class _RunInfo:
    """One row in the run picker — a completed (or in-progress) validation run."""
    path:        Path
    name:        str
    strategy:    str
    n_folds:     int
    timestamp:   str
    models:      List[str]
    finished:    bool

    @property
    def display(self) -> str:
        status = "" if self.finished else "  (incomplete)"
        return f"{self.timestamp}  ·  {self.name}  ·  {self.strategy}{status}"


def _discover_runs(validation_runs_root: Path) -> List[_RunInfo]:
    """
    Walk ``validation_runs/`` and return one :class:`_RunInfo` per
    directory containing a ``results.json``. Newer runs first.
    """
    out: List[_RunInfo] = []
    if not validation_runs_root.exists():
        return out
    for sub in sorted(validation_runs_root.iterdir(), reverse=True):
        if not sub.is_dir():
            continue
        res = sub / "results.json"
        exp = sub / "experiment.json"
        # Allow incomplete runs too — they show up greyed out in the UI.
        # We only require *something* recognisable on disk so the user
        # can spot runs that crashed mid-way and re-launch them.
        if not (res.exists() or exp.exists()):
            continue
        try:
            import json
            cfg = json.loads(exp.read_text()) if exp.exists() else {}
            results = json.loads(res.read_text()) if res.exists() else {}
        except Exception:  # noqa: BLE001
            cfg, results = {}, {}
        name      = str(cfg.get("name") or sub.name)
        strategy  = str(((cfg.get("cv") or {}).get("strategy")) or "?")
        models    = [m.get("type", "?") for m in (cfg.get("models") or [])]
        n_folds   = len(results.get("folds") or [])
        finished  = bool(results.get("finished_at"))
        ts        = sub.name.split("__")[0] if "__" in sub.name else sub.name
        out.append(_RunInfo(
            path=sub, name=name, strategy=strategy,
            n_folds=n_folds, timestamp=ts, models=models,
            finished=finished,
        ))
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Suite-run worker  (queues multiple ExperimentConfig calls)
# ═══════════════════════════════════════════════════════════════════════════

class _RunBridge(QObject):
    """Qt-side bridge so the worker can emit signals into the GUI thread."""
    run_started   = Signal(int, int, str)        # (idx, total, label)
    fold_done     = Signal(int, int, object)     # (idx, total, FoldResult)
    run_finished  = Signal(int, int, str, object)  # (idx, total, label, RunResult|None)
    suite_started = Signal(int)                  # (total)
    suite_finished = Signal(object)              # List[(label, RunResult)]
    suite_failed  = Signal(str)
    log_line      = Signal(str)
    _cancel       = False

    def request_cancel(self) -> None:
        self._cancel = True

    def should_cancel(self) -> bool:
        return self._cancel

    # ProgressReporter protocol stubs the runner can call from the worker.
    # We forward selected events into Qt signals; everything else is a
    # no-op. The runner's on_log_line is the high-volume one, so we
    # debounce it lightly to avoid spamming the GUI text area.
    def on_run_start(self, total_folds, total_models, records): pass
    def on_run_done(self, result):                              pass
    def on_fold_start(self, *_a, **_kw):                        pass

    def on_fold_done(self, eval_idx, total_evals, fold_result):
        # eval_idx/total_evals are within a single run; we let the
        # _SuiteRunWorker fill in the suite-level position.
        self.fold_done.emit(int(eval_idx), int(total_evals), fold_result)

    def log(self, msg: str) -> None:
        self.log_line.emit(str(msg))


class _SuiteRunWorker(QThread):
    """
    Run a list of ``(label, ExperimentConfig)`` pairs sequentially. One
    failing config does not abort the suite — we log the error and move
    on, just like ``_SweepWorker`` in evaluation_tab.py.
    """

    def __init__(self,
                 data_dir: Path,
                 plans: Sequence[Tuple[str, ExperimentConfig]],
                 bridge: _RunBridge,
                 parent: Optional[QObject] = None):
        super().__init__(parent)
        self._data_dir = Path(data_dir)
        self._plans    = list(plans)
        self._bridge   = bridge

    def run(self) -> None:
        results: List[Tuple[str, Optional[RunResult]]] = []
        try:
            self._bridge.suite_started.emit(len(self._plans))
            runner = ValidationRunner(self._data_dir)
            for i, (label, cfg) in enumerate(self._plans, start=1):
                if self._bridge.should_cancel():
                    self._bridge.log_line.emit(
                        f"Cancelled before run {i}/{len(self._plans)}."
                    )
                    break
                self._bridge.run_started.emit(i, len(self._plans), label)
                self._bridge.log_line.emit(
                    f"▶ [{i}/{len(self._plans)}] {label}  ({cfg.cv.strategy})"
                )
                rr: Optional[RunResult] = None
                try:
                    rr = runner.run(cfg, progress=self._bridge)
                except Exception as exc:  # noqa: BLE001
                    self._bridge.log_line.emit(
                        f"⚠ run {i}/{len(self._plans)} '{label}' FAILED: {exc}"
                    )
                    log.exception("suite run failed: %s", label)
                results.append((label, rr))
                self._bridge.run_finished.emit(i, len(self._plans), label, rr)
            self._bridge.suite_finished.emit(results)
        except Exception:  # noqa: BLE001
            import traceback
            self._bridge.suite_failed.emit(traceback.format_exc())


# ═══════════════════════════════════════════════════════════════════════════
# Report-build worker  (runs the generate_thesis_outputs orchestrator)
# ═══════════════════════════════════════════════════════════════════════════

class _ReportBridge(QObject):
    started        = Signal()
    log_line       = Signal(str)
    finished       = Signal(object)   # Dict[str, List[Path]] from orchestrator
    failed         = Signal(str)


class _QtLogHandler(logging.Handler):
    """logging.Handler that emits each record as a Qt signal."""
    def __init__(self, bridge: _ReportBridge):
        super().__init__(level=logging.INFO)
        self._bridge = bridge
        self.setFormatter(logging.Formatter("%(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._bridge.log_line.emit(self.format(record))
        except Exception:  # noqa: BLE001
            # Never let a logging failure kill the worker.
            pass


@dataclass
class ReportBuildArgs:
    """All the inputs :func:`generate_thesis_outputs.run` needs."""
    data_dir:        Path
    out:             Path
    primary:         Optional[Path]
    loso_subj:       Optional[Path]
    ablation:        Optional[str]
    xdomain:         Optional[str]
    primary_model:   str = "catboost"
    window_ms:       int = _DEFAULT_WINDOW_MS
    stride_ms:       int = _DEFAULT_STRIDE_MS
    drop_rest:       bool = False
    flag_threshold:  float = 0.5
    gate_ms:         float = 150.0
    verbose:         bool = True
    # Healthy / impaired cohort split. ``groups`` is the registry file
    # passed through as ``--groups``; when None the generator falls back
    # to its own default-location lookup + metadata inference.
    groups:          Optional[Path] = None
    # When False, ``--skip-game`` is added so the game-recording report
    # (§6.8–6.9) is not generated.
    include_game:    bool = True


class _ReportBuildWorker(QThread):
    """
    Calls :func:`playagain_pipeline.validation.generate_thesis_outputs.run`
    in a background thread, mirroring the orchestrator's logger into a
    Qt signal so the dialog can stream progress lines.
    """

    def __init__(self, args: ReportBuildArgs, bridge: _ReportBridge,
                 parent: Optional[QObject] = None):
        super().__init__(parent)
        self._args   = args
        self._bridge = bridge

    def run(self) -> None:
        # Run the orchestrator in a completely separate Python process so
        # that matplotlib never initialises inside the Qt process.  On
        # macOS ARM the Agg renderer crashes with SIGBUS when a second
        # rendering context competes with Qt's.
        import sys
        import os

        self._bridge.started.emit()

        cmd = [
            sys.executable, "-m",
            "playagain_pipeline.validation.generate_thesis_outputs",
        ] + self._to_argv()

        # The child process needs the same sys.path as the GUI so it can
        # import playagain_pipeline. Pass via PYTHONPATH and set cwd to
        # the project root (the directory that contains the
        # playagain_pipeline package folder).
        env = os.environ.copy()
        existing = env.get("PYTHONPATH", "")
        extra = os.pathsep.join(p for p in sys.path if p)
        env["PYTHONPATH"] = (extra + os.pathsep + existing) if existing else extra

        # Walk up from this file to find the directory that contains
        # playagain_pipeline/__init__.py.
        project_root = Path(__file__).resolve().parent
        for _ in range(8):
            if (project_root / "playagain_pipeline" / "__init__.py").exists():
                break
            project_root = project_root.parent

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,   # merge so we see everything
                text=True,
                bufsize=1,                  # line-buffered
                cwd=str(project_root),
                env=env,
            )
        except Exception as exc:  # noqa: BLE001
            self._bridge.failed.emit(
                f"Could not launch the report generator subprocess:\n{exc}\n\n"
                "Make sure thesis_reports.py, plots_thesis.py and "
                "generate_thesis_outputs.py have been added to "
                "playagain_pipeline/validation/."
            )
            return

        # Stream every output line into the UI log while the child runs.
        assert proc.stdout is not None
        for line in proc.stdout:
            self._bridge.log_line.emit(line.rstrip())

        proc.wait()

        if proc.returncode == 0:
            # The subprocess wrote all files to disk; we don't have the
            # produced-dict in-process, so emit an empty dict — the
            # caller only uses it to display a file count.
            self._bridge.finished.emit({})
        else:
            self._bridge.failed.emit(
                f"Report generator exited with code {proc.returncode}.\n"
                "See the log above for details."
            )

    # ------------------------------------------------------------------

    def _to_argv(self) -> List[str]:
        a = self._args
        argv: List[str] = [
            "--data-dir", str(a.data_dir),
            "--out",      str(a.out),
            "--primary-model", a.primary_model,
            "--window-ms", str(a.window_ms),
            "--stride-ms", str(a.stride_ms),
            "--flag-threshold", str(a.flag_threshold),
            "--gate-ms", str(a.gate_ms),
        ]
        if a.primary:    argv += ["--primary",   str(a.primary)]
        if a.loso_subj:  argv += ["--loso-subj", str(a.loso_subj)]
        if a.ablation:   argv += ["--ablation",  a.ablation]
        if a.xdomain:    argv += ["--xdomain",   a.xdomain]
        if a.drop_rest:  argv += ["--drop-rest"]
        if a.groups:     argv += ["--groups", str(a.groups)]
        if not a.include_game:
            argv += ["--skip-game"]
        if a.verbose:    argv += ["--verbose"]
        return argv


# ═══════════════════════════════════════════════════════════════════════════
# Helper widget: dropdown of discovered runs
# ═══════════════════════════════════════════════════════════════════════════

class _RunPicker(QWidget):
    """
    A combo-box + browse button pair. The combo is populated from the
    project's ``validation_runs/`` directory; the browse button lets
    the user point at a run outside that directory (e.g. on a network
    share).
    """

    run_changed = Signal(object)   # emits Path | None

    def __init__(self, validation_runs_root: Path,
                 *, strategy_hint: Optional[str] = None,
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._root = Path(validation_runs_root)
        self._strategy_hint = strategy_hint
        self._current: Optional[Path] = None

        lay = QHBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(6)
        self._combo = QComboBox()
        self._combo.setSizePolicy(QSizePolicy.Policy.Expanding,
                                  QSizePolicy.Policy.Preferred)
        self._combo.currentIndexChanged.connect(self._on_choice)
        lay.addWidget(self._combo, 1)

        self._browse = QPushButton("Browse…")
        self._browse.clicked.connect(self._on_browse)
        lay.addWidget(self._browse, 0)

        self.refresh()

    def refresh(self) -> None:
        cur_path = self._current
        self._combo.blockSignals(True)
        self._combo.clear()
        self._combo.addItem("(not selected)", None)
        for info in _discover_runs(self._root):
            # Hint by strategy — but keep all runs visible. We just
            # mark the recommended ones with an arrow so the user can
            # tell which run goes where without micro-managing them.
            prefix = "→ " if (self._strategy_hint and
                              info.strategy == self._strategy_hint) else "   "
            self._combo.addItem(prefix + info.display, str(info.path))
        # Restore previous selection if it's still there.
        if cur_path is not None:
            for i in range(self._combo.count()):
                if self._combo.itemData(i) == str(cur_path):
                    self._combo.setCurrentIndex(i)
                    break
        self._combo.blockSignals(False)

    def selected_path(self) -> Optional[Path]:
        return self._current

    def set_selected_path(self, path: Optional[Path]) -> None:
        self._current = path
        self.refresh()

    @Slot()
    def _on_choice(self) -> None:
        data = self._combo.currentData()
        self._current = Path(data) if data else None
        self.run_changed.emit(self._current)

    @Slot()
    def _on_browse(self) -> None:
        chosen = QFileDialog.getExistingDirectory(
            self, "Select a validation run directory", str(self._root)
        )
        if not chosen:
            return
        p = Path(chosen)
        # Sanity-check: must contain results.json or experiment.json.
        if not ((p / "results.json").exists() or (p / "experiment.json").exists()):
            QMessageBox.warning(
                self, "Not a run directory",
                f"{p}\n\nThis folder doesn't look like a validation_runs/ output "
                "(no results.json or experiment.json found).",
            )
            return
        # Make sure the item exists in the combo, then select it.
        for i in range(self._combo.count()):
            if self._combo.itemData(i) == str(p):
                self._combo.setCurrentIndex(i)
                return
        self._combo.addItem("   " + str(p), str(p))
        self._combo.setCurrentIndex(self._combo.count() - 1)


# ═══════════════════════════════════════════════════════════════════════════
# Helper widget: ablation / cross-domain mapping table
# ═══════════════════════════════════════════════════════════════════════════

class _RoleMappingTable(QWidget):
    """
    A small table with one row per slot (feature name or cross-domain
    direction). Each row has a label and a _RunPicker. The whole
    mapping serialises to the ``name=path,name=path`` string the
    orchestrator expects.
    """

    def __init__(self, slots: Sequence[str], validation_runs_root: Path,
                 *, strategy_hint: Optional[str] = None,
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._pickers: Dict[str, _RunPicker] = {}

        outer = QVBoxLayout(self); outer.setContentsMargins(0, 0, 0, 0); outer.setSpacing(4)
        for slot in slots:
            row = QWidget()
            row_lay = QHBoxLayout(row); row_lay.setContentsMargins(0, 0, 0, 0)
            row_lay.setSpacing(8)
            lbl = QLabel(slot)
            lbl.setMinimumWidth(110)
            lbl.setStyleSheet("color:#475569;")
            row_lay.addWidget(lbl)
            picker = _RunPicker(validation_runs_root, strategy_hint=strategy_hint)
            row_lay.addWidget(picker, 1)
            outer.addWidget(row)
            self._pickers[slot] = picker

    def refresh(self) -> None:
        for p in self._pickers.values():
            p.refresh()

    def spec_string(self) -> str:
        """Build the ``slot=path,slot=path`` string for the orchestrator."""
        parts: List[str] = []
        for slot, picker in self._pickers.items():
            p = picker.selected_path()
            if p is not None:
                parts.append(f"{slot}={p}")
        return ",".join(parts)

    def n_selected(self) -> int:
        return sum(1 for p in self._pickers.values() if p.selected_path())


# ═══════════════════════════════════════════════════════════════════════════
# Main dialog
# ═══════════════════════════════════════════════════════════════════════════

class ThesisReportDialog(QDialog):
    """
    Top-level dialog. See module docstring for the structure.

    Parameters
    ----------
    data_dir : Path
        Project data directory (contains ``sessions/`` and
        ``validation_runs/``).
    parent : QWidget, optional
        Parent window. Typically the v2 main window.
    """

    DEFAULT_TITLE = "Build thesis report"

    def __init__(self, data_dir: Path, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle(self.DEFAULT_TITLE)
        self.setModal(False)

        # A plain QDialog only gets a close button. Add the maximize and
        # minimize hints so this behaves like the Evaluation tab's window
        # — the user can go full-screen when generating a big report.
        self.setWindowFlags(
            self.windowFlags()
            | Qt.WindowType.WindowMaximizeButtonHint
            | Qt.WindowType.WindowMinimizeButtonHint
        )
        # Initial size is comfortable on a laptop; the minimum is small
        # enough to fit a 720p panel, and the scrollable body (see
        # _build_ui) keeps every section reachable below that.
        self.resize(1000, 800)
        self.setMinimumSize(560, 420)

        if apply_app_style is not None:
            try:
                apply_app_style(self, theme="bright")
            except Exception:  # noqa: BLE001
                pass

        self._data_dir: Path = Path(data_dir)
        self._validation_runs_root: Path = self._data_dir / "validation_runs"
        self._validation_runs_root.mkdir(parents=True, exist_ok=True)

        # Default output directory: <data_dir>/thesis_outputs
        self._default_out: Path = self._data_dir / "thesis_outputs"

        # Healthy / impaired registry. Defaults to the canonical
        # <data_dir>/participant_groups.json; the user can repoint or
        # edit it from the "Participant groups" sub-section below.
        self._groups_path: Path = default_groups_path(self._data_dir)

        # Workers (kept as members so we can cancel cleanly)
        self._suite_bridge: Optional[_RunBridge] = None
        self._suite_worker: Optional[_SuiteRunWorker] = None
        self._report_bridge: Optional[_ReportBridge] = None
        self._report_worker: Optional[_ReportBuildWorker] = None

        self._build_ui()

    # ──────────────────────────────────────────────────────────────────
    # UI construction
    # ──────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(14, 14, 14, 14)
        outer.setSpacing(12)

        outer.addWidget(self._build_header())

        # The three sections are stacked vertically inside a scroll area.
        # Previously a QSplitter shared a fixed window height between
        # them, which crushed every section on smaller displays. With a
        # scroll area each section keeps its natural height and the user
        # scrolls if the window can't show them all at once.
        body_scroll = QScrollArea()
        body_scroll.setWidgetResizable(True)
        body_scroll.setFrameShape(QFrame.Shape.NoFrame)
        body_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )

        body_host = QWidget()
        body_lay = QVBoxLayout(body_host)
        body_lay.setContentsMargins(0, 0, 0, 0)
        body_lay.setSpacing(12)
        body_lay.addWidget(self._build_section_create())
        body_lay.addWidget(self._build_section_map())
        body_lay.addWidget(self._build_section_generate())
        body_lay.addStretch(1)
        body_scroll.setWidget(body_host)

        # The scroll area gets the lion's share of the stretch; the log
        # pane keeps a fixed, modest footprint so it never starves the
        # sections above it (the old layout let it claim half the dialog).
        outer.addWidget(body_scroll, 1)

        log_pane = self._build_log_pane()
        log_pane.setMinimumHeight(130)
        log_pane.setMaximumHeight(260)
        outer.addWidget(log_pane, 0)

        bottom = QHBoxLayout(); bottom.setContentsMargins(0, 0, 0, 0)
        bottom.addStretch(1)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        bottom.addWidget(close_btn)
        outer.addLayout(bottom)

    def _build_header(self) -> QWidget:
        header = QFrame()
        header.setStyleSheet(
            "QFrame { background:#f1f5f9; border:1px solid #e2e8f0; border-radius:8px; }"
        )
        lay = QHBoxLayout(header); lay.setContentsMargins(14, 10, 14, 10)
        lay.setSpacing(10)

        title = QLabel("Build thesis report")
        f = QFont(); f.setPointSize(14); f.setBold(True); title.setFont(f)
        title.setStyleSheet("color:#0284c7;")
        lay.addWidget(title)

        sub = QLabel(
            "Create the validation runs for Ch. 6, then generate every "
            "table and figure referenced from chapters 6, 7 and 8."
        )
        sub.setStyleSheet("color:#475569; font-size:11px;")
        sub.setWordWrap(True)
        lay.addWidget(sub, 1)

        return header

    # ---- Section 1 ---------------------------------------------------

    def _build_section_create(self) -> QGroupBox:
        box = QGroupBox("1. Create validation runs")
        box.setStyleSheet(self._group_style())
        lay = QVBoxLayout(box); lay.setContentsMargins(14, 16, 14, 12)
        lay.setSpacing(10)

        info = QLabel(
            "Queue the runs that Chapter 6 reports. All runs use 200 ms "
            "windows / 50 ms stride and the same seed for reproducibility. "
            "Each preset writes a new directory under "
            f"<code>{self._validation_runs_root}</code>."
        )
        info.setWordWrap(True)
        info.setTextFormat(Qt.TextFormat.RichText)
        info.setStyleSheet("color:#475569; font-size:11px;")
        lay.addWidget(info)

        # Model + primary-model row
        opts = QFormLayout(); opts.setSpacing(8)
        self._model_checks: Dict[str, QCheckBox] = {}
        model_row = QWidget(); mh = QHBoxLayout(model_row)
        mh.setContentsMargins(0, 0, 0, 0); mh.setSpacing(8)
        for key, label in _THESIS_MODELS:
            cb = QCheckBox(label)
            # All classical models default to checked; deep models off
            # to keep the default suite light. The user can opt in
            # explicitly when their hardware can take it.
            cb.setChecked(key in {"lda", "svm", "random_forest", "catboost"})
            self._model_checks[key] = cb
            mh.addWidget(cb)
        mh.addStretch(1)
        opts.addRow(QLabel("Models:"), model_row)

        self._primary_model_combo = QComboBox()
        for key, label in _THESIS_MODELS:
            self._primary_model_combo.addItem(f"{label}  ({key})", key)
        # Default to catboost — that's the thesis text's "best classical".
        idx = self._primary_model_combo.findData("catboost")
        if idx >= 0:
            self._primary_model_combo.setCurrentIndex(idx)
        opts.addRow(QLabel("Primary model:"), self._primary_model_combo)
        opts.addRow(QLabel(""),
                    QLabel("(used for ablation, cross-domain, and figures)"))

        lay.addLayout(opts)

        # Preset buttons grid
        grid = QGridLayout(); grid.setHorizontalSpacing(8); grid.setVerticalSpacing(6)
        btn_loso_sess = self._primary_button(
            "Create LOSO-session run", "Table 6.3, Figs 6.3–6.5, Table 6.7")
        btn_loso_subj = self._primary_button(
            "Create LOSO-subject run", "Table 6.4")
        btn_abl       = self._primary_button(
            "Create feature ablation suite", "Table 6.5 + Fig 6.6")
        btn_xd        = self._primary_button(
            "Create cross-domain suite", "Table 6.6")
        btn_all       = self._primary_button(
            "Create all of the above", "Queues every preset in order")
        btn_all.setStyleSheet(btn_all.styleSheet() +
                              " QPushButton { background:#0284c7; color:white; }"
                              " QPushButton:hover { background:#0369a1; }")

        btn_loso_sess.clicked.connect(self._on_create_loso_session)
        btn_loso_subj.clicked.connect(self._on_create_loso_subject)
        btn_abl.clicked.connect(self._on_create_ablation)
        btn_xd.clicked.connect(self._on_create_xdomain)
        btn_all.clicked.connect(self._on_create_all)

        grid.addWidget(btn_loso_sess, 0, 0)
        grid.addWidget(btn_loso_subj, 0, 1)
        grid.addWidget(btn_abl,       1, 0)
        grid.addWidget(btn_xd,        1, 1)
        grid.addWidget(btn_all,       2, 0, 1, 2)
        lay.addLayout(grid)

        # Progress + cancel for run-suite
        prog_row = QHBoxLayout(); prog_row.setSpacing(8)
        self._suite_progress = QProgressBar()
        self._suite_progress.setRange(0, 1)
        self._suite_progress.setValue(0)
        self._suite_progress.setFormat("idle")
        prog_row.addWidget(self._suite_progress, 1)
        self._suite_cancel = QPushButton("Cancel")
        self._suite_cancel.setEnabled(False)
        self._suite_cancel.clicked.connect(self._on_cancel_suite)
        prog_row.addWidget(self._suite_cancel, 0)
        lay.addLayout(prog_row)

        return box

    # ---- Section 2 ---------------------------------------------------

    def _build_section_map(self) -> QGroupBox:
        box = QGroupBox("2. Map existing runs to thesis sections")
        box.setStyleSheet(self._group_style())
        outer = QVBoxLayout(box); outer.setContentsMargins(14, 16, 14, 12)
        outer.setSpacing(10)

        info = QLabel(
            "Tell the report generator which run on disk fills each "
            "chapter-6 slot. Arrows mark runs whose strategy matches "
            "what the slot expects, but you can pick any run."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color:#475569; font-size:11px;")
        outer.addWidget(info)

        form = QFormLayout(); form.setSpacing(6)

        self._picker_primary = _RunPicker(
            self._validation_runs_root, strategy_hint="loso_session"
        )
        form.addRow(QLabel("§6.3 LOSO-session:"), self._picker_primary)

        self._picker_loso_subj = _RunPicker(
            self._validation_runs_root, strategy_hint="loso_subject"
        )
        form.addRow(QLabel("§6.4 LOSO-subject:"), self._picker_loso_subj)
        outer.addLayout(form)

        # Ablation: 8 feature slots + combined
        abl_box = QGroupBox("§6.5 Feature ablation")
        abl_box.setStyleSheet(self._sub_group_style())
        abl_lay = QVBoxLayout(abl_box); abl_lay.setContentsMargins(8, 8, 8, 8)
        abl_lay.setSpacing(4)
        self._ablation_table = _RoleMappingTable(
            slots=_THESIS_FEATURES + ["combined"],
            validation_runs_root=self._validation_runs_root,
            strategy_hint="loso_session",
        )
        # Scroll wrapper — eight feature rows is too tall to inline.
        abl_scroll = QScrollArea(); abl_scroll.setWidgetResizable(True)
        abl_scroll.setWidget(self._ablation_table)
        abl_scroll.setMaximumHeight(170)
        abl_lay.addWidget(abl_scroll)
        outer.addWidget(abl_box)

        # Cross-domain: 4 directions
        xd_box = QGroupBox("§6.6 Cross-domain")
        xd_box.setStyleSheet(self._sub_group_style())
        xd_lay = QVBoxLayout(xd_box); xd_lay.setContentsMargins(8, 8, 8, 8)
        xd_lay.setSpacing(4)
        self._xdomain_table = _RoleMappingTable(
            slots=["within_pipe", "within_unity", "p2u", "u2p"],
            validation_runs_root=self._validation_runs_root,
        )
        xd_lay.addWidget(self._xdomain_table)
        outer.addWidget(xd_box)

        # Refresh button — discovers new runs after a suite finishes.
        refresh_row = QHBoxLayout(); refresh_row.setSpacing(8)
        refresh_btn = QPushButton("⟳ Refresh run list")
        refresh_btn.clicked.connect(self._refresh_run_pickers)
        refresh_row.addStretch(1); refresh_row.addWidget(refresh_btn)
        outer.addLayout(refresh_row)

        return box

    # ---- Section 3 ---------------------------------------------------

    def _build_section_generate(self) -> QGroupBox:
        box = QGroupBox("3. Generate thesis outputs")
        box.setStyleSheet(self._group_style())
        lay = QVBoxLayout(box); lay.setContentsMargins(14, 16, 14, 12)
        lay.setSpacing(8)

        info = QLabel(
            "Writes every CSV table and PDF/PNG figure referenced from "
            "chapters 6, 7 and 8 to the output directory below. Anything "
            "left unmapped above is silently skipped — you can run "
            "partial generations as soon as the matching runs finish."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color:#475569; font-size:11px;")
        lay.addWidget(info)

        # Output directory + browse
        out_row = QHBoxLayout(); out_row.setSpacing(6)
        out_row.addWidget(QLabel("Output:"))
        self._out_edit = QLineEdit(str(self._default_out))
        out_row.addWidget(self._out_edit, 1)
        out_btn = QPushButton("Browse…")
        out_btn.clicked.connect(self._on_browse_out)
        out_row.addWidget(out_btn)
        lay.addLayout(out_row)

        # Tunables row
        knobs = QFormLayout(); knobs.setSpacing(6)
        from PySide6.QtWidgets import QDoubleSpinBox, QSpinBox

        self._win_spin = QSpinBox(); self._win_spin.setRange(50, 2000)
        self._win_spin.setSingleStep(10); self._win_spin.setValue(_DEFAULT_WINDOW_MS)
        self._win_spin.setSuffix(" ms")
        self._stride_spin = QSpinBox(); self._stride_spin.setRange(5, 500)
        self._stride_spin.setSingleStep(5); self._stride_spin.setValue(_DEFAULT_STRIDE_MS)
        self._stride_spin.setSuffix(" ms")
        win_row = QWidget(); wh = QHBoxLayout(win_row); wh.setContentsMargins(0, 0, 0, 0)
        wh.setSpacing(8); wh.addWidget(self._win_spin); wh.addWidget(QLabel("/"))
        wh.addWidget(self._stride_spin); wh.addStretch(1)
        knobs.addRow("Window / stride:", win_row)

        self._flag_spin = QDoubleSpinBox(); self._flag_spin.setRange(0.0, 1.0)
        self._flag_spin.setSingleStep(0.05); self._flag_spin.setValue(0.5)
        self._flag_spin.setDecimals(2)
        knobs.addRow("Calibration flag threshold:", self._flag_spin)

        self._gate_spin = QDoubleSpinBox(); self._gate_spin.setRange(1.0, 5000.0)
        self._gate_spin.setSingleStep(10.0); self._gate_spin.setValue(150.0)
        self._gate_spin.setSuffix(" ms")
        knobs.addRow("Latency gate (Table 6.7):", self._gate_spin)

        self._drop_rest_cb = QCheckBox("Exclude rest windows from class-distribution counts")
        knobs.addRow("", self._drop_rest_cb)
        lay.addLayout(knobs)

        # ── Healthy / impaired cohort split + game recordings ────────
        cohort_box = QGroupBox("Participant groups & extra evaluations")
        cohort_box.setStyleSheet(self._sub_group_style())
        cohort_lay = QVBoxLayout(cohort_box)
        cohort_lay.setContentsMargins(8, 10, 8, 8)
        cohort_lay.setSpacing(6)

        cohort_info = QLabel(
            "Splits Tables 6.3/6.4 and the per-session variability figure "
            "by cohort, and adds the game-recording performance report "
            "(§6.8–6.9). Both are optional — leave the registry blank to "
            "fall back to metadata inference."
        )
        cohort_info.setWordWrap(True)
        cohort_info.setStyleSheet("color:#475569; font-size:10px;")
        cohort_lay.addWidget(cohort_info)

        # Registry path row + Browse + Edit.
        groups_row = QHBoxLayout(); groups_row.setSpacing(6)
        groups_row.addWidget(QLabel("Groups file:"))
        self._groups_edit = QLineEdit(str(self._groups_path))
        self._groups_edit.setPlaceholderText(
            "participant_groups.json — leave as-is for the default location"
        )
        self._groups_edit.textChanged.connect(self._on_groups_path_edited)
        groups_row.addWidget(self._groups_edit, 1)
        groups_browse = QPushButton("Browse…")
        groups_browse.clicked.connect(self._on_browse_groups)
        groups_row.addWidget(groups_browse)
        groups_edit_btn = QPushButton("Edit…")
        groups_edit_btn.setToolTip(
            "Open the participant-group editor to assign subjects to "
            "the healthy / impaired cohorts."
        )
        groups_edit_btn.clicked.connect(self._on_edit_groups)
        groups_row.addWidget(groups_edit_btn)
        cohort_lay.addLayout(groups_row)

        # Live status line — counts if the file parses, a hint otherwise.
        self._groups_status = QLabel("")
        self._groups_status.setTextFormat(Qt.TextFormat.RichText)
        self._groups_status.setStyleSheet("font-size:10px; padding-left:2px;")
        cohort_lay.addWidget(self._groups_status)

        # Game-recording toggle.
        self._game_cb = QCheckBox(
            "Include game-recording evaluation  (Tables 6.8–6.9, Figs 6.7–6.8)"
        )
        self._game_cb.setChecked(True)
        self._game_cb.setToolTip(
            "Game recordings carry real logged multi-class predictions, so "
            "they are scored directly (no retraining). Uncheck to skip "
            "them — equivalent to the --skip-game flag."
        )
        cohort_lay.addWidget(self._game_cb)

        lay.addWidget(cohort_box)

        # Populate the status line for whatever path we start with.
        self._refresh_groups_status()

        # Action + progress
        action_row = QHBoxLayout(); action_row.setSpacing(8)
        self._generate_btn = self._primary_button(
            "Generate all tables and figures", "Writes to the output directory")
        self._generate_btn.setStyleSheet(self._generate_btn.styleSheet() +
                                         " QPushButton { background:#16a34a; color:white; }"
                                         " QPushButton:hover { background:#15803d; }")
        self._generate_btn.clicked.connect(self._on_generate)
        action_row.addWidget(self._generate_btn)

        self._open_out_btn = QPushButton("Open output folder")
        self._open_out_btn.setEnabled(False)
        self._open_out_btn.clicked.connect(self._on_open_out)
        action_row.addWidget(self._open_out_btn)
        action_row.addStretch(1)
        lay.addLayout(action_row)

        self._report_progress = QProgressBar()
        self._report_progress.setRange(0, 1); self._report_progress.setValue(0)
        self._report_progress.setFormat("idle")
        lay.addWidget(self._report_progress)

        return box

    def _build_log_pane(self) -> QGroupBox:
        box = QGroupBox("Log")
        box.setStyleSheet(self._group_style())
        lay = QVBoxLayout(box); lay.setContentsMargins(8, 12, 8, 8)
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setStyleSheet(
            "QTextEdit { font-family:'Menlo','DejaVu Sans Mono',monospace; "
            "font-size:11px; background:#0f172a; color:#e2e8f0; "
            "border:1px solid #334155; border-radius:6px; padding:6px; }"
        )
        self._log.setMinimumHeight(90)
        lay.addWidget(self._log)
        return box

    # ──────────────────────────────────────────────────────────────────
    # Styles
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _group_style() -> str:
        return ("QGroupBox { font-weight:600; color:#0f172a; "
                "border:1px solid #cbd5e1; border-radius:8px; margin-top:14px; "
                "padding-top:8px; }"
                "QGroupBox::title { subcontrol-origin: margin; left:12px; "
                "padding:0 4px; }")

    @staticmethod
    def _sub_group_style() -> str:
        return ("QGroupBox { color:#475569; border:1px dashed #cbd5e1; "
                "border-radius:6px; margin-top:10px; padding-top:6px; }"
                "QGroupBox::title { left:10px; padding:0 4px; }")

    @staticmethod
    def _primary_button(title: str, tooltip: str = "") -> QPushButton:
        btn = QPushButton(title)
        btn.setMinimumHeight(34)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setStyleSheet(
            "QPushButton { background:#ffffff; border:1px solid #cbd5e1; "
            " border-radius:6px; padding:6px 12px; color:#0f172a; }"
            "QPushButton:hover { background:#f1f5f9; border-color:#0284c7; }"
            "QPushButton:disabled { color:#94a3b8; background:#f8fafc; }"
        )
        if tooltip:
            btn.setToolTip(tooltip)
        return btn

    # ──────────────────────────────────────────────────────────────────
    # Section 1 — slot handlers
    # ──────────────────────────────────────────────────────────────────

    def _checked_models(self) -> List[str]:
        chosen = [k for k, cb in self._model_checks.items() if cb.isChecked()]
        filtered = _filter_available_models(chosen)
        if len(filtered) != len(chosen):
            dropped = sorted(set(chosen) - set(filtered))
            self._append_log(
                f"Note: {len(dropped)} unavailable model(s) skipped: {', '.join(dropped)}"
            )
        return filtered

    def _primary_model(self) -> str:
        return str(self._primary_model_combo.currentData() or "catboost")

    def _on_create_loso_session(self) -> None:
        models = self._checked_models()
        if not models:
            QMessageBox.information(self, "Pick at least one model",
                                    "Tick at least one model before queuing a run.")
            return
        cfg = _make_primary_cfg(models)
        self._launch_suite([("loso_session (primary)", cfg)])

    def _on_create_loso_subject(self) -> None:
        # Per the thesis text, the LOSO-subject table reports classical
        # models only — deep models often don't have enough subjects.
        # Drop them quietly here unless the user has explicitly checked
        # them; the small log line below explains what happened.
        all_checked = self._checked_models()
        classical = [m for m in all_checked
                     if m in {"lda", "svm", "random_forest", "catboost"}]
        if not classical:
            QMessageBox.information(
                self, "Pick at least one classical model",
                "LOSO-subject (Table 6.4) reports classical models. Tick "
                "LDA, SVM, Random Forest or CatBoost.",
            )
            return
        if len(classical) != len(all_checked):
            self._append_log(
                "LOSO-subject: dropped deep models — "
                "Table 6.4 reports classical models only."
            )
        cfg = _make_loso_subj_cfg(classical)
        self._launch_suite([("loso_subject", cfg)])

    def _on_create_ablation(self) -> None:
        pm = self._primary_model()
        features_to_use = self._features_available_or_warn()
        if not features_to_use:
            return
        plans = _make_ablation_cfgs(pm, features_to_use)
        self._launch_suite(plans)

    def _features_available_or_warn(self) -> List[str]:
        """
        Returns the subset of _THESIS_FEATURES that the project's
        feature factory actually knows about. Surfaces the dropped ones
        in the log so the writer knows their Table 6.5 will have fewer
        rows.
        """
        try:
            from playagain_pipeline.features import FeatureExtractor  # type: ignore
            fe = FeatureExtractor()
            known = set(getattr(fe, "available_features", lambda: [])() or [])
        except Exception:  # noqa: BLE001
            known = set(_THESIS_FEATURES)
        usable = [f for f in _THESIS_FEATURES if f in known]
        dropped = [f for f in _THESIS_FEATURES if f not in known]
        if dropped:
            self._append_log(
                f"Note: {len(dropped)} feature(s) unavailable in the pipeline: "
                f"{', '.join(dropped)} — the ablation suite will have "
                f"{len(usable)} single-feature runs + combined."
            )
        if not usable:
            QMessageBox.warning(
                self, "No features available",
                "The feature factory didn't expose any of the thesis "
                "features. Check playagain_pipeline.features.",
            )
        return usable

    def _on_create_xdomain(self) -> None:
        # Need at least one session of each domain — otherwise the runs
        # will produce empty folds. Quick sanity check before we queue.
        corpus = SessionCorpus(self._data_dir); corpus.discover()
        n_pipe = sum(1 for r in corpus.all() if r.source_domain == "pipeline")
        n_unity = sum(1 for r in corpus.all() if r.source_domain == "unity")
        if not n_pipe or not n_unity:
            QMessageBox.warning(
                self, "Need both domains",
                f"Cross-domain needs sessions in both domains "
                f"(found pipeline={n_pipe}, unity={n_unity}).",
            )
            return
        plans = _make_xdomain_cfgs(self._primary_model())
        self._launch_suite(plans)

    def _on_create_all(self) -> None:
        models = self._checked_models()
        if not models:
            QMessageBox.information(self, "Pick at least one model",
                                    "Tick at least one model before queuing.")
            return
        plans: List[Tuple[str, ExperimentConfig]] = []
        plans.append(("loso_session (primary)", _make_primary_cfg(models)))
        classical = [m for m in models
                     if m in {"lda", "svm", "random_forest", "catboost"}] or models
        plans.append(("loso_subject", _make_loso_subj_cfg(classical)))
        features = self._features_available_or_warn()
        if features:
            plans.extend(_make_ablation_cfgs(self._primary_model(), features))
        # Cross-domain only if both domains are present.
        corpus = SessionCorpus(self._data_dir); corpus.discover()
        if any(r.source_domain == "pipeline" for r in corpus.all()) \
           and any(r.source_domain == "unity" for r in corpus.all()):
            plans.extend(_make_xdomain_cfgs(self._primary_model()))
        else:
            self._append_log(
                "Skipping cross-domain suite — need sessions in both domains."
            )
        self._launch_suite(plans)

    # ──────────────────────────────────────────────────────────────────
    # Suite worker plumbing
    # ──────────────────────────────────────────────────────────────────

    def _launch_suite(self, plans: List[Tuple[str, ExperimentConfig]]) -> None:
        if self._suite_worker is not None and self._suite_worker.isRunning():
            QMessageBox.warning(self, "Already running",
                                "A run suite is already in progress. "
                                "Cancel it first or wait for it to finish.")
            return
        if not plans:
            return

        self._suite_bridge = _RunBridge()
        self._suite_bridge.suite_started.connect(self._on_suite_started)
        self._suite_bridge.run_started.connect(self._on_run_started)
        self._suite_bridge.fold_done.connect(self._on_fold_done)
        self._suite_bridge.run_finished.connect(self._on_run_finished)
        self._suite_bridge.suite_finished.connect(self._on_suite_finished)
        self._suite_bridge.suite_failed.connect(self._on_suite_failed)
        self._suite_bridge.log_line.connect(self._append_log)

        self._suite_worker = _SuiteRunWorker(
            self._data_dir, plans, self._suite_bridge, parent=self,
        )
        self._suite_worker.start()
        self._suite_cancel.setEnabled(True)
        self._append_log(f"Queued {len(plans)} validation run(s).")

    @Slot(int)
    def _on_suite_started(self, total: int) -> None:
        self._suite_progress.setRange(0, total)
        self._suite_progress.setValue(0)
        self._suite_progress.setFormat(f"0 / {total} runs")

    @Slot(int, int, str)
    def _on_run_started(self, i: int, total: int, label: str) -> None:
        self._suite_progress.setFormat(f"{i-1} / {total}  ·  running: {label}")

    @Slot(int, int, object)
    def _on_fold_done(self, eval_idx: int, total_evals: int, fr) -> None:
        # Optional — we already log run-start; per-fold here would be
        # too chatty. Keep this slot in case you want to surface fold
        # counts in a future enhancement.
        pass

    @Slot(int, int, str, object)
    def _on_run_finished(self, i: int, total: int, label: str, rr) -> None:
        self._suite_progress.setValue(i)
        self._suite_progress.setFormat(f"{i} / {total} runs")
        status = "ok" if rr is not None else "FAILED"
        self._append_log(f"◉ [{i}/{total}] {label}: {status}")

    @Slot(object)
    def _on_suite_finished(self, results) -> None:
        self._append_log(
            f"✓ Suite complete: {sum(1 for _,rr in results if rr is not None)} "
            f"of {len(results)} runs succeeded."
        )
        self._suite_cancel.setEnabled(False)
        # New run dirs are on disk — refresh the section-2 pickers so
        # they're immediately mappable.
        self._refresh_run_pickers()

    @Slot(str)
    def _on_suite_failed(self, traceback_text: str) -> None:
        self._append_log("⨯ Suite crashed:")
        self._append_log(traceback_text)
        self._suite_cancel.setEnabled(False)

    def _on_cancel_suite(self) -> None:
        if self._suite_bridge is not None:
            self._suite_bridge.request_cancel()
            self._append_log("Cancellation requested.")
            self._suite_cancel.setEnabled(False)

    # ──────────────────────────────────────────────────────────────────
    # Section 2 plumbing
    # ──────────────────────────────────────────────────────────────────

    def _refresh_run_pickers(self) -> None:
        self._picker_primary.refresh()
        self._picker_loso_subj.refresh()
        self._ablation_table.refresh()
        self._xdomain_table.refresh()

    # ──────────────────────────────────────────────────────────────────
    # Section 3 — generate
    # ──────────────────────────────────────────────────────────────────

    def _on_browse_out(self) -> None:
        chosen = QFileDialog.getExistingDirectory(
            self, "Output directory", self._out_edit.text() or str(self._default_out),
        )
        if chosen:
            self._out_edit.setText(chosen)

    # ---- Participant-group registry ----------------------------------

    def _on_groups_path_edited(self, text: str) -> None:
        """Keep the cached path in sync with manual edits + refresh status."""
        self._groups_path = Path(text.strip()) if text.strip() else \
            default_groups_path(self._data_dir)
        self._refresh_groups_status()

    def _on_browse_groups(self) -> None:
        start = str(self._groups_path.parent if self._groups_path
                    else self._data_dir)
        chosen, _ = QFileDialog.getOpenFileName(
            self, "Participant-group registry", start,
            "Group registry (*.json *.csv);;All files (*)",
        )
        if chosen:
            self._groups_edit.setText(chosen)   # triggers _on_groups_path_edited

    def _on_edit_groups(self) -> None:
        """Open the standalone editor; adopt its path when it saves."""
        try:
            from playagain_pipeline.gui.widgets.participant_groups_dialog import (
                ParticipantGroupsDialog,
            )
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(
                self, "Editor unavailable",
                "Could not open the participant-group editor:\n\n"
                f"{exc}\n\nYou can still point the field above at a "
                "participant_groups.json you maintain by hand.")
            return

        initial = self._groups_path if self._groups_path.exists() else None
        dlg = ParticipantGroupsDialog(
            data_dir=self._data_dir, initial_path=initial, parent=self,
        )
        # When the editor saves, repoint our field at the saved file so
        # the next "Generate" picks it up automatically.
        dlg.registry_saved.connect(
            lambda p: self._groups_edit.setText(str(p))
        )
        dlg.show()

    def _refresh_groups_status(self) -> None:
        """Update the little status line under the groups-file field."""
        path = self._groups_path
        if not path or not path.exists():
            self._groups_status.setText(
                "<span style='color:#94a3b8;'>No registry file yet — "
                "cohorts will fall back to session-metadata inference. "
                "Use “Edit…” to create one.</span>"
            )
            return
        try:
            groups = ParticipantGroups.from_file(path)
        except Exception as exc:  # noqa: BLE001
            self._groups_status.setText(
                f"<span style='color:#dc2626;'>⚠ could not read "
                f"{path.name}: {exc}</span>"
            )
            return
        counts = groups.counts()
        n_h = counts.get(GROUP_HEALTHY, 0)
        n_i = counts.get(GROUP_IMPAIRED, 0)
        if n_h == 0 and n_i == 0:
            self._groups_status.setText(
                f"<span style='color:#d97706;'>{path.name} has no cohort "
                "assignments yet — use “Edit…” to fill it in.</span>"
            )
        else:
            self._groups_status.setText(
                f"<span style='color:#16a34a;'>✓ {path.name}: "
                f"<b>{n_h}</b> healthy · <b>{n_i}</b> impaired</span>"
            )

    def _on_generate(self) -> None:
        if self._report_worker is not None and self._report_worker.isRunning():
            QMessageBox.warning(self, "Already running",
                                "Report generation is already in progress.")
            return

        out_dir = Path(self._out_edit.text().strip() or str(self._default_out))
        out_dir.mkdir(parents=True, exist_ok=True)

        primary   = self._picker_primary.selected_path()
        loso_subj = self._picker_loso_subj.selected_path()
        ablation  = self._ablation_table.spec_string()
        xdomain   = self._xdomain_table.spec_string()

        # The corpus report + calibration report always run, so the
        # button is never disabled — even with no runs mapped, the user
        # still gets §6.1 and §6.2. Surface that expectation up front.
        if not (primary or loso_subj or ablation or xdomain):
            answer = QMessageBox.question(
                self, "Generate without runs?",
                "No validation runs are mapped to thesis sections.\n\n"
                "The corpus overview (§6.1) and calibration report "
                "(§6.2) will still be generated, but Tables 6.3–6.7 "
                "and Figures 6.3–6.6 will be skipped. Continue?",
            )
            if answer != QMessageBox.StandardButton.Yes:
                return

        # Only forward an explicit --groups path when the file actually
        # exists; otherwise let the generator do its own default lookup.
        groups_arg: Optional[Path] = (
            self._groups_path
            if self._groups_path and self._groups_path.exists()
            else None
        )

        args = ReportBuildArgs(
            data_dir=self._data_dir,
            out=out_dir,
            primary=primary,
            loso_subj=loso_subj,
            ablation=ablation or None,
            xdomain=xdomain or None,
            primary_model=self._primary_model(),
            window_ms=self._win_spin.value(),
            stride_ms=self._stride_spin.value(),
            flag_threshold=self._flag_spin.value(),
            gate_ms=self._gate_spin.value(),
            drop_rest=self._drop_rest_cb.isChecked(),
            groups=groups_arg,
            include_game=self._game_cb.isChecked(),
            verbose=True,
        )

        self._report_bridge = _ReportBridge()
        self._report_bridge.started.connect(self._on_report_started)
        self._report_bridge.log_line.connect(self._append_log)
        self._report_bridge.finished.connect(self._on_report_finished)
        self._report_bridge.failed.connect(self._on_report_failed)

        self._report_worker = _ReportBuildWorker(args, self._report_bridge, parent=self)
        self._generate_btn.setEnabled(False)
        self._report_progress.setRange(0, 0)  # indeterminate
        self._report_progress.setFormat("generating…")
        self._report_worker.start()
        self._append_log(f"▶ Generating thesis outputs into {out_dir}")
        if groups_arg:
            self._append_log(f"   • cohort split: {groups_arg.name}")
        else:
            self._append_log("   • cohort split: metadata inference "
                             "(no registry file)")
        self._append_log(
            "   • game recordings: "
            + ("included" if self._game_cb.isChecked() else "skipped")
        )

    @Slot()
    def _on_report_started(self) -> None:
        pass

    @Slot(object)
    def _on_report_finished(self, produced: Dict[str, List[Path]]) -> None:
        n_files = sum(len(v) for v in produced.values())
        self._append_log(f"✓ Wrote {n_files} file(s) across "
                         f"{len(produced)} artefact group(s).")
        self._report_progress.setRange(0, 1)
        self._report_progress.setValue(1)
        self._report_progress.setFormat("done")
        self._generate_btn.setEnabled(True)
        self._open_out_btn.setEnabled(True)

    @Slot(str)
    def _on_report_failed(self, traceback_text: str) -> None:
        self._append_log("⨯ Report generation failed:")
        self._append_log(traceback_text)
        self._report_progress.setRange(0, 1)
        self._report_progress.setValue(0)
        self._report_progress.setFormat("failed")
        self._generate_btn.setEnabled(True)

    def _on_open_out(self) -> None:
        path = Path(self._out_edit.text().strip() or str(self._default_out))
        if not path.exists():
            QMessageBox.information(self, "Folder not found",
                                    f"{path} doesn't exist yet.")
            return
        # Platform-specific reveal-in-finder. xdg-open / open / explorer
        # all interpret a directory path the right way.
        try:
            if platform.system() == "Windows":
                subprocess.Popen(["explorer", str(path)])
            elif platform.system() == "Darwin":
                subprocess.Popen(["open", str(path)])
            else:
                subprocess.Popen(["xdg-open", str(path)])
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Could not open folder",
                                f"{path}\n\n{exc}")

    # ──────────────────────────────────────────────────────────────────
    # Log helpers
    # ──────────────────────────────────────────────────────────────────

    def _append_log(self, msg: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        # Auto-scroll only if the user is already at the bottom — that
        # way they can scroll up to inspect older lines while a run is
        # still emitting fresh ones.
        sb = self._log.verticalScrollBar()
        at_bottom = sb.value() >= sb.maximum() - 4
        for line in str(msg).splitlines() or [""]:
            self._log.append(f"[{ts}] {line}")
        if at_bottom:
            sb.setValue(sb.maximum())

    # ──────────────────────────────────────────────────────────────────
    # Lifecycle
    # ──────────────────────────────────────────────────────────────────

    def closeEvent(self, event) -> None:
        """Ask before closing while a worker is still running."""
        running = (
            (self._suite_worker is not None and self._suite_worker.isRunning())
            or (self._report_worker is not None and self._report_worker.isRunning())
        )
        if running:
            answer = QMessageBox.question(
                self, "Tasks in progress",
                "A run or report build is still in progress. Close anyway?",
            )
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            # Best-effort cancellation; the workers honour their cancel
            # flag at the next safe point.
            if self._suite_bridge is not None:
                self._suite_bridge.request_cancel()
        event.accept()