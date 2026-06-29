"""
validation/intra_session_compare.py
───────────────────────────────────
Train several model *types* with **intra-session** validation, aggregate
the per-session scores, and emit thesis-grade tables + figures — broken
out three ways: **overall**, **by cohort** (healthy / impaired), and
**per participant**.

This is the "give me the within-session ceiling for my shortlist of
models and show me how it differs across the cohort" driver. You hand it
a list of model types::

    run_intra_session_comparison(IntraSessionComparisonConfig(
        data_dir=Path("data"),
        output_dir=Path("out/intra_session"),
        models=["random_forest", "catboost", "cnn", "mstnet"],
    ))

and it:

1. discovers every session (optionally filtered to a subject subset);
2. for each model, fits it **fresh inside each session** (train on the
   earlier trials, score the held-out later trials — see
   :mod:`evaluation.intra_session_eval`) and collects one score per
   session;
3. resolves each session's cohort from
   :class:`validation.participant_groups.ParticipantGroups`;
4. aggregates F1 / accuracy / recall (and a few extras) at three levels —
   overall, per group, per subject — and writes them as CSVs;
5. renders the figures listed in :data:`FIGURES`, all in the FAU
   institutional palette (the ``fau`` family from ``fau_colors``).

Why intra-session per model, scored per session
────────────────────────────────────────────────
Intra-session validation is inherently *per recording*: each session is
split into its own train/test halves, so a model produces one score per
session. That gives a natural distribution to put in a box-and-whisker
plot, and a natural unit (the session, nested in a subject, nested in a
cohort) to aggregate over. Pooling all sessions into one number would
throw away exactly the between-participant spread the thesis is about, so
every aggregate here keeps the session as the unit and reports spread.

Heavy work (model fitting) is delegated to
:func:`evaluation.intra_session_eval.evaluate_intra_session_per_recording`;
this module only orchestrates, aggregates, and plots. Matplotlib is
imported in ``Agg`` mode so it is safe under the headless GUI process.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")  # headless; avoids Qt conflicts in the GUI process

import matplotlib.pyplot as plt                              # noqa: E402
import numpy as np                                           # noqa: E402
from matplotlib import gridspec                              # noqa: E402
from matplotlib.colors import LinearSegmentedColormap        # noqa: E402

# Eval engine lives in the sibling ``evaluation`` subpackage. Use the
# absolute ``playagain_pipeline.evaluation`` path (same convention the
# rest of this package uses for cross-package imports, e.g.
# ``from playagain_pipeline.models.classifier import ...`` in runner.py).
# It's imported as a module so the heavy ML imports inside it stay lazy
# and so it's trivially monkeypatchable in tests.
from playagain_pipeline.evaluation import intra_session_eval as _ise   # noqa: E402
from playagain_pipeline.evaluation.core import (                       # noqa: E402
    ConfusionMatrix, EvaluationResult,
)
from playagain_pipeline.evaluation.loaders import discover_sessions    # noqa: E402

from .participant_groups import (                            # noqa: E402
    GROUP_HEALTHY, GROUP_IMPAIRED, GROUP_UNKNOWN,
    ParticipantGroups, group_label,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FAU palette  (the ``fau`` family — graceful fallback if pkg absent)
# ---------------------------------------------------------------------------

try:
    from fau_colors import cmaps as _CMAPS, colors as _COLORS
    _FAU_RAMP: List[str] = list(_CMAPS.fau)        # dark → light, 5 shades
    FAU_BLUE = _COLORS.fau
    HAS_FAU = True
except Exception:                                  # noqa: BLE001
    _FAU_RAMP = ["#04316A", "#617DA1", "#A0B1C6", "#C0CBDA", "#D3DCF2"]
    FAU_BLUE = "#04316A"
    HAS_FAU = False

# Continuous fau map (light → dark) for sampling N distinct model shades
# and for the sequential confusion-matrix colormap.
_FAU_CONT = LinearSegmentedColormap.from_list("fau_cont", _FAU_RAMP[::-1])
_FAU_SEQ = LinearSegmentedColormap.from_list(
    "fau_seq", ["#FFFFFF", _FAU_RAMP[3], _FAU_RAMP[1], _FAU_RAMP[0]],
)

# Cohort colours, both drawn from the fau ramp so the two cohorts stay
# inside the institutional palette: impaired = deepest fau navy, healthy
# = a mid fau shade. Unknown = neutral grey.
GROUP_COLORS: Dict[str, str] = {
    GROUP_HEALTHY:  _FAU_RAMP[2],     # #A0B1C6  mid blue-grey
    GROUP_IMPAIRED: _FAU_RAMP[0],     # #04316A  deep navy
    GROUP_UNKNOWN:  "#B8BEC6",        # neutral grey
}

NEUTRAL_TEXT = "#1f2937"
NEUTRAL_MUTED = "#6b7280"
NEUTRAL_GRID = "#e2e8f0"


def _setup_mpl() -> None:
    matplotlib.rcParams.update({
        "font.family":       "sans-serif",
        "font.sans-serif":   ["DejaVu Sans", "Arial"],
        "font.size":         10,
        "axes.titlesize":    11,
        "axes.labelsize":    10,
        "axes.titleweight":  "semibold",
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.edgecolor":    NEUTRAL_GRID,
        "axes.labelcolor":   NEUTRAL_TEXT,
        "axes.titlepad":     8,
        "xtick.color":       NEUTRAL_TEXT,
        "ytick.color":       NEUTRAL_TEXT,
        "xtick.labelsize":   9,
        "ytick.labelsize":   9,
        "legend.fontsize":   9,
        "legend.frameon":    False,
        "figure.dpi":        150,
        "savefig.dpi":       300,
        "savefig.bbox":      "tight",
        "savefig.facecolor": "white",
        "pdf.fonttype":      42,
        "ps.fonttype":       42,
    })


def _save(fig: plt.Figure, out_path: Path) -> List[Path]:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    for ext in (".pdf", ".png"):
        p = out_path.with_suffix(ext)
        fig.savefig(p)
        written.append(p)
    plt.close(fig)
    return written


def _model_palette(models: Sequence[str]) -> Dict[str, str]:
    """Assign each model a distinct shade sampled from the fau ramp.

    We sample the continuous fau map over its darker 80 % so every shade
    reads on a white background while staying inside the fau family.
    """
    n = max(len(models), 1)
    out: Dict[str, str] = {}
    for i, m in enumerate(models):
        t = 0.18 + 0.82 * (i / max(n - 1, 1)) if n > 1 else 0.85
        out[m] = matplotlib.colors.to_hex(_FAU_CONT(t))
    return out


# ---------------------------------------------------------------------------
# Metrics we report  (EvaluationResult field → display label)
# ---------------------------------------------------------------------------

#: The headline three the request asked for, plus two robustness extras
#: kept in the CSVs (not all plotted).
METRIC_LABELS: Dict[str, str] = {
    "f1_macro":          "Macro F1",
    "accuracy":          "Accuracy",
    "recall_macro":      "Macro recall",
    "balanced_accuracy": "Balanced acc.",
    "mcc":               "MCC",
}
HEADLINE_METRICS: Tuple[str, ...] = ("f1_macro", "accuracy", "recall_macro")


# ---------------------------------------------------------------------------
# Config + result containers
# ---------------------------------------------------------------------------

@dataclass
class IntraSessionComparisonConfig:
    """Everything the driver needs for one comparison run."""
    data_dir:   Path
    output_dir: Path
    models:     List[str]                      # e.g. ["random_forest","catboost","cnn","mstnet"]

    # Intra-session eval knobs (forwarded to IntraSessionSettings).
    test_fraction:    float = 0.25
    window_size_ms:   Optional[int] = None     # None → 200 ms default
    window_stride_ms: int = 50
    split_unit:       str = "trial"            # "trial" (recommended) | "window"
    seed:             int = 42

    # Feature config for *classical* models (RF/CatBoost/SVM/LDA). None
    # lets each model use its own default feature stack; deep models
    # (cnn/mstnet/attention_net) ignore this and take raw windows.
    feature_config:   Optional[Dict[str, Any]] = None

    # Optional per-model hyper-params: {"catboost": {"depth": 6}, ...}.
    model_params:     Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Cohort registry. None → <data_dir>/participant_groups.json if present.
    groups_file:      Optional[Path] = None
    include_unknown_group: bool = True

    # Session selection.
    subjects:         Optional[List[str]] = None   # restrict to these subjects
    include_unity:    bool = False

    # Plot toggles.
    metrics:          Tuple[str, ...] = HEADLINE_METRICS
    primary_metric:   str = "f1_macro"             # used for ordering / subject figure


@dataclass
class _Row:
    """One session's score under one model — the tidy unit we aggregate."""
    model:       str
    subject_id:  str
    session_id:  str
    group:       str                       # "H" | "I" | "?"
    n_test:      int
    metrics:     Dict[str, float]          # field → value
    per_class_f1: Dict[str, float]         # class name → f1
    confusion:   Optional[ConfusionMatrix]
    inference_ms: Optional[float]


@dataclass
class ComparisonResult:
    rows:               List[_Row]
    figures:            Dict[str, List[Path]]
    csvs:               Dict[str, Path]
    models:             List[str]
    groups_present:     List[str]
    notes:              List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_intra_session_comparison(
    cfg: IntraSessionComparisonConfig,
    *,
    progress: Optional[Any] = None,
) -> ComparisonResult:
    """Run the full intra-session comparison and write tables + figures."""
    _setup_mpl()
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    notes: List[str] = []

    # ── Sessions + cohorts ─────────────────────────────────────────────
    sessions = discover_sessions(cfg.data_dir, include_unity=cfg.include_unity)
    if cfg.subjects:
        keep = set(map(str, cfg.subjects))
        sessions = [s for s in sessions if str(s.subject_id) in keep]
    if not sessions:
        raise RuntimeError("No sessions discovered for the comparison.")

    if cfg.groups_file is not None:
        groups = ParticipantGroups.from_file(cfg.groups_file)
    else:
        groups = ParticipantGroups.from_data_dir(cfg.data_dir)
    if groups.is_empty:
        notes.append(
            "No participant-group registry found — every subject resolves "
            "to 'unknown'. Drop a participant_groups.json next to your data "
            "(see participant_groups.write_template) to get the cohort split."
        )

    # ── Train + score each model, session by session ───────────────────
    rows: List[_Row] = []
    n_models = max(len(cfg.models), 1)
    for mi, model in enumerate(cfg.models):
        pct = 100 * (mi / n_models)
        print(f"[{pct:3.0f}%] Intra-session: {model}...", flush=True)
        if progress:
            try:
                progress(mi / n_models, f"Intra-session: {model}…")
            except Exception:                                # noqa: BLE001
                pass
        settings = _ise.IntraSessionSettings(
            model_type=model,
            feature_config=cfg.feature_config,
            model_params=dict(cfg.model_params.get(model, {})),
            refit=True,
            window_size_ms=cfg.window_size_ms,
            window_stride_ms=cfg.window_stride_ms,
            test_fraction=cfg.test_fraction,
            split_unit=cfg.split_unit,
            seed=cfg.seed,
            per_session_breakdown=True,
        )
        per_session = _ise.evaluate_intra_session_per_recording(
            cfg.data_dir, sessions, settings,
        )
        if not per_session:
            notes.append(f"⚠ {model}: produced no per-session results.")
            continue
        for res in per_session:
            rows.append(_row_from_result(model, res, groups))

    if not rows:
        raise RuntimeError("No usable results across any model — see notes.")

    models_present = [m for m in cfg.models if any(r.model == m for r in rows)]
    groups_present = _groups_present(rows, include_unknown=cfg.include_unknown_group)

    # ── Tables ──────────────────────────────────────────────────────────
    csvs: Dict[str, Path] = {}
    csvs["per_session"] = _write_per_session_csv(rows, out / "intra_session_per_session.csv")
    csvs["overall"] = _write_summary_csv(
        _aggregate(rows, by=("model",)),
        out / "intra_session_summary_overall.csv",
        index_cols=("model",),
    )
    csvs["by_group"] = _write_summary_csv(
        _aggregate(rows, by=("model", "group")),
        out / "intra_session_summary_by_group.csv",
        index_cols=("model", "group"),
    )
    csvs["by_subject"] = _write_summary_csv(
        _aggregate(rows, by=("model", "subject_id")),
        out / "intra_session_summary_by_subject.csv",
        index_cols=("model", "subject_id"),
    )
    csvs["confusion"] = _write_confusion_json(
        rows, models_present, groups_present, out / "intra_session_confusion.json",
    )

    # ── Figures ─────────────────────────────────────────────────────────
    figs: Dict[str, List[Path]] = {}
    model_colors = _model_palette(models_present)

    figs["metrics_overall"] = plot_metrics_overall(
        rows, models_present, cfg.metrics, model_colors,
        out / "fig_intra_metrics_overall",
    )
    figs["metrics_by_group"] = plot_metrics_by_group(
        rows, models_present, groups_present, cfg.metrics,
        out / "fig_intra_metrics_by_group",
    )
    figs["summary_bars"] = plot_summary_bars(
        rows, models_present, groups_present, cfg.metrics,
        out / "fig_intra_summary_bars",
    )
    figs["by_subject"] = plot_by_subject(
        rows, models_present, cfg.primary_metric,
        out / "fig_intra_by_subject",
    )
    figs["confusion_overall"] = plot_confusion_grid(
        rows, models_present, group=None,
        out_path=out / "fig_intra_confusion_overall",
        title="Intra-session confusion matrices (pooled over sessions)",
    )
    for g in groups_present:
        if g == GROUP_UNKNOWN:
            continue
        figs[f"confusion_{group_label(g)}"] = plot_confusion_grid(
            rows, models_present, group=g,
            out_path=out / f"fig_intra_confusion_{group_label(g)}",
            title=f"Intra-session confusion matrices — {group_label(g)} cohort",
        )
    # One confusion-matrix figure per participant (cohort named in title).
    for subj, paths in plot_confusion_per_subject(
        rows, models_present, out,
    ).items():
        figs[f"confusion_subject_{subj}"] = paths
    figs["per_class_f1"] = plot_per_class_f1(
        rows, models_present, model_colors,
        out / "fig_intra_per_class_f1",
    )
    if any(r.inference_ms is not None for r in rows):
        figs["latency"] = plot_latency(
            rows, models_present, model_colors,
            out / "fig_intra_latency",
        )

    if progress:
        try:
            progress(1.0, "Done.")
        except Exception:                                    # noqa: BLE001
            pass
    print("[100%] Done.", flush=True)

    return ComparisonResult(
        rows=rows, figures=figs, csvs=csvs,
        models=models_present, groups_present=groups_present, notes=notes,
    )


# ---------------------------------------------------------------------------
# Row extraction + aggregation
# ---------------------------------------------------------------------------

def _row_from_result(
    model: str, res: EvaluationResult, groups: ParticipantGroups,
) -> _Row:
    rec = res.recordings[0] if res.recordings else None
    subject = str(rec.subject_id) if rec else "?"
    session = str(rec.session_id) if rec else "?"
    metrics = {
        k: float(getattr(res, k, float("nan")))
        for k in METRIC_LABELS
    }
    pcf1 = {cm.name: float(cm.f1) for cm in res.per_class}
    return _Row(
        model=model,
        subject_id=subject,
        session_id=session,
        group=groups.group_of(subject),
        n_test=int(res.n_samples),
        metrics=metrics,
        per_class_f1=pcf1,
        confusion=res.confusion,
        inference_ms=res.inference_ms_per_window,
    )


def _groups_present(rows: List[_Row], *, include_unknown: bool) -> List[str]:
    present = {r.group for r in rows}
    ordered = [g for g in (GROUP_HEALTHY, GROUP_IMPAIRED) if g in present]
    if include_unknown and GROUP_UNKNOWN in present:
        ordered.append(GROUP_UNKNOWN)
    return ordered


def _aggregate(
    rows: List[_Row], *, by: Tuple[str, ...],
) -> List[Dict[str, Any]]:
    """Group rows by the given keys and summarise each metric (mean/std/n)."""
    buckets: Dict[Tuple[str, ...], List[_Row]] = {}
    for r in rows:
        key = tuple(getattr(r, k) for k in by)
        buckets.setdefault(key, []).append(r)

    out: List[Dict[str, Any]] = []
    for key, group_rows in buckets.items():
        entry: Dict[str, Any] = {k: v for k, v in zip(by, key)}
        entry["n_sessions"] = len(group_rows)
        entry["n_subjects"] = len({r.subject_id for r in group_rows})
        for metric in METRIC_LABELS:
            vals = np.array([r.metrics.get(metric, np.nan) for r in group_rows],
                            dtype=np.float64)
            vals = vals[~np.isnan(vals)]
            entry[f"{metric}_mean"] = float(np.mean(vals)) if vals.size else float("nan")
            entry[f"{metric}_std"] = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
        out.append(entry)
    # Stable, readable ordering.
    out.sort(key=lambda e: tuple(str(e.get(k, "")) for k in by))
    return out


def _values_for(
    rows: List[_Row], *, model: str, metric: str,
    group: Optional[str] = None, subject: Optional[str] = None,
) -> List[float]:
    vals: List[float] = []
    for r in rows:
        if r.model != model:
            continue
        if group is not None and r.group != group:
            continue
        if subject is not None and r.subject_id != subject:
            continue
        v = r.metrics.get(metric, float("nan"))
        if not math.isnan(v):
            vals.append(v)
    return vals


# ---------------------------------------------------------------------------
# Confusion-matrix aggregation
# ---------------------------------------------------------------------------

def _sum_confusions(
    confs: Sequence[ConfusionMatrix],
) -> Optional[Tuple[List[int], List[str], np.ndarray]]:
    """Sum confusion matrices over the union of their labels.

    Returns ``(labels, label_names, matrix)`` or ``None`` if no usable
    matrices were supplied.
    """
    confs = [c for c in confs if c is not None and c.labels]
    if not confs:
        return None
    label_set: List[int] = sorted({int(l) for c in confs for l in c.labels})
    name_of: Dict[int, str] = {}
    for c in confs:
        for lbl, name in zip(c.labels, c.label_names):
            name_of.setdefault(int(lbl), str(name))
    index = {lbl: i for i, lbl in enumerate(label_set)}
    acc = np.zeros((len(label_set), len(label_set)), dtype=np.int64)
    for c in confs:
        mat = c.to_array()
        for i_true, lt in enumerate(c.labels):
            for i_pred, lp in enumerate(c.labels):
                acc[index[int(lt)], index[int(lp)]] += int(mat[i_true, i_pred])
    names = [name_of.get(lbl, f"class_{lbl}") for lbl in label_set]
    return label_set, names, acc


def _row_normalise(mat: np.ndarray) -> np.ndarray:
    mat = mat.astype(np.float64)
    rs = mat.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1.0
    return mat / rs


# ---------------------------------------------------------------------------
# CSV writers
# ---------------------------------------------------------------------------

def _write_per_session_csv(rows: List[_Row], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (["model", "subject_id", "session_id", "group", "group_label",
               "n_test"]
              + list(METRIC_LABELS) + ["inference_ms"])
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for r in rows:
            w.writerow(
                [r.model, r.subject_id, r.session_id, r.group,
                 group_label(r.group), r.n_test]
                + [f"{r.metrics.get(m, float('nan')):.6f}" for m in METRIC_LABELS]
                + [("" if r.inference_ms is None else f"{r.inference_ms:.4f}")]
            )
    return path


def _write_summary_csv(
    agg: List[Dict[str, Any]], path: Path, *, index_cols: Tuple[str, ...],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    metric_cols: List[str] = []
    for m in METRIC_LABELS:
        metric_cols += [f"{m}_mean", f"{m}_std"]
    fields = list(index_cols) + ["n_sessions", "n_subjects"] + metric_cols
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for e in agg:
            w.writerow([e.get(c, "") for c in index_cols]
                       + [e.get("n_sessions", 0), e.get("n_subjects", 0)]
                       + [f"{e.get(c, float('nan')):.6f}" for c in metric_cols])
    return path


def _write_confusion_json(
    rows: List[_Row], models: List[str], groups: List[str], path: Path,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    blob: Dict[str, Any] = {"overall": {}, "by_group": {}, "by_subject": {}}
    for m in models:
        summed = _sum_confusions([r.confusion for r in rows if r.model == m])
        if summed is None:
            continue
        labels, names, mat = summed
        blob["overall"][m] = {
            "labels": labels, "label_names": names,
            "matrix": mat.tolist(),
            "matrix_norm": _row_normalise(mat).tolist(),
        }
    for g in groups:
        blob["by_group"][group_label(g)] = {}
        for m in models:
            summed = _sum_confusions(
                [r.confusion for r in rows if r.model == m and r.group == g]
            )
            if summed is None:
                continue
            labels, names, mat = summed
            blob["by_group"][group_label(g)][m] = {
                "labels": labels, "label_names": names,
                "matrix": mat.tolist(),
                "matrix_norm": _row_normalise(mat).tolist(),
            }

    # Per-subject — one confusion block per participant, pooled over that
    # participant's sessions, tagged with the cohort it belongs to so a
    # downstream plot can name the group in the title. Subjects are ordered
    # healthy → impaired → unknown then by id, matching the cohort ordering
    # the by-group figures use. Each subject nests its per-model matrices
    # under "models" so the group metadata sits alongside them.
    rank = {GROUP_HEALTHY: 0, GROUP_IMPAIRED: 1, GROUP_UNKNOWN: 2}
    subj_group: Dict[str, str] = {}
    for r in rows:
        subj_group.setdefault(r.subject_id, r.group)
    for subj in sorted(subj_group, key=lambda s: (rank.get(subj_group[s], 9), s)):
        per_model: Dict[str, Any] = {}
        for m in models:
            summed = _sum_confusions(
                [r.confusion for r in rows
                 if r.model == m and r.subject_id == subj]
            )
            if summed is None:
                continue
            labels, names, mat = summed
            per_model[m] = {
                "labels": labels, "label_names": names,
                "matrix": mat.tolist(),
                "matrix_norm": _row_normalise(mat).tolist(),
            }
        if not per_model:
            continue
        g = subj_group[subj]
        blob["by_subject"][subj] = {
            "group": g,
            "group_label": group_label(g),
            "models": per_model,
        }
    with path.open("w", encoding="utf-8") as f:
        json.dump(blob, f, indent=2)
    return path


# ---------------------------------------------------------------------------
# Generic grouped boxplot helper
# ---------------------------------------------------------------------------

def _draw_boxes(
    ax, data_by_box: List[List[float]], positions: List[float],
    color: str, width: float, *, showfliers: bool = True,
) -> None:
    """Draw a set of patch-artist boxes in the house style."""
    keep = [(p, d) for p, d in zip(positions, data_by_box) if d]
    if not keep:
        return
    pos = [p for p, _ in keep]
    dat = [d for _, d in keep]
    ax.boxplot(
        dat, positions=pos, widths=width, patch_artist=True,
        showfliers=showfliers,
        medianprops=dict(color=NEUTRAL_TEXT, linewidth=1.6),
        boxprops=dict(facecolor=color, edgecolor=NEUTRAL_TEXT,
                      linewidth=0.8, alpha=0.88),
        whiskerprops=dict(color=NEUTRAL_TEXT, linewidth=0.9),
        capprops=dict(color=NEUTRAL_TEXT, linewidth=0.9),
        flierprops=dict(marker="o", markersize=2.5, markerfacecolor=color,
                        markeredgecolor=NEUTRAL_TEXT, markeredgewidth=0.4),
    )


def _scatter_points(ax, positions: List[float], data_by_box: List[List[float]],
                    color: str, width: float, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    for pos, vals in zip(positions, data_by_box):
        if not vals:
            continue
        jitter = rng.normal(0, width * 0.12, size=len(vals))
        ax.scatter(np.full(len(vals), pos) + jitter, vals, s=12,
                   color=color, edgecolor="white", linewidth=0.4,
                   zorder=4, alpha=0.8)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_metrics_overall(
    rows: List[_Row], models: List[str], metrics: Sequence[str],
    model_colors: Dict[str, str], out_path: Path,
) -> List[Path]:
    """One panel per metric; one box per model (distribution across sessions)."""
    metrics = [m for m in metrics if m in METRIC_LABELS]
    fig, axes = plt.subplots(
        1, len(metrics), figsize=(3.6 * len(metrics) + 0.5, 4.4), squeeze=False,
    )
    for ax, metric in zip(axes[0], metrics):
        for j, model in enumerate(models):
            vals = _values_for(rows, model=model, metric=metric)
            _draw_boxes(ax, [vals], [j], model_colors[model], 0.6)
            _scatter_points(ax, [j], [vals], model_colors[model], 0.6, seed=j)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=20, ha="right")
        ax.set_ylim(0, 1.03)
        ax.set_title(METRIC_LABELS[metric], loc="left")
        ax.grid(axis="y", linestyle=":", color=NEUTRAL_GRID)
        ax.set_axisbelow(True)
    axes[0][0].set_ylabel("score  (per-session distribution)")
    fig.suptitle("Intra-session performance by model (overall)",
                 fontsize=12.5, fontweight="semibold", x=0.02, ha="left")
    return _save(fig, out_path)


def plot_metrics_by_group(
    rows: List[_Row], models: List[str], groups: List[str],
    metrics: Sequence[str], out_path: Path,
) -> List[Path]:
    """One panel per metric; per model a box per cohort (hue = cohort)."""
    metrics = [m for m in metrics if m in METRIC_LABELS]
    groups = [g for g in groups if g != GROUP_UNKNOWN] or groups
    if not groups:
        return []
    n_g = len(groups)
    width = 0.8 / n_g
    fig, axes = plt.subplots(
        1, len(metrics), figsize=(4.0 * len(metrics) + 0.5, 4.6), squeeze=False,
    )
    for ax, metric in zip(axes[0], metrics):
        for gi, g in enumerate(groups):
            color = GROUP_COLORS.get(g, "#94a3b8")
            positions, data = [], []
            for j, model in enumerate(models):
                positions.append(j + (gi - (n_g - 1) / 2) * width)
                data.append(_values_for(rows, model=model, metric=metric, group=g))
            _draw_boxes(ax, data, positions, color, width * 0.85, showfliers=False)
            _scatter_points(ax, positions, data, color, width, seed=gi)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=20, ha="right")
        ax.set_ylim(0, 1.03)
        ax.set_title(METRIC_LABELS[metric], loc="left")
        ax.grid(axis="y", linestyle=":", color=NEUTRAL_GRID)
        ax.set_axisbelow(True)
        for i in range(1, len(models)):
            ax.axvline(i - 0.5, color=NEUTRAL_GRID, linewidth=0.8, zorder=1)
    axes[0][0].set_ylabel("score  (per-session distribution)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=GROUP_COLORS.get(g, "#94a3b8"),
                             alpha=0.88, ec=NEUTRAL_TEXT, lw=0.6)
               for g in groups]
    fig.legend(handles, [group_label(g) for g in groups],
               loc="lower center", ncol=len(groups), bbox_to_anchor=(0.5, -0.02),
               columnspacing=1.8)
    fig.suptitle("Intra-session performance by model and cohort",
                 fontsize=12.5, fontweight="semibold", x=0.02, ha="left")
    fig.subplots_adjust(bottom=0.2)
    return _save(fig, out_path)


def plot_summary_bars(
    rows: List[_Row], models: List[str], groups: List[str],
    metrics: Sequence[str], out_path: Path,
) -> List[Path]:
    """Grouped bars: mean ± SD per model for each headline metric (overall)."""
    metrics = [m for m in metrics if m in METRIC_LABELS]
    fig, ax = plt.subplots(figsize=(1.9 * len(models) + 3.0, 4.4))
    n_m = len(metrics)
    width = 0.8 / n_m
    # Each metric is a fau shade so the legend stays in-palette.
    metric_colors = {m: matplotlib.colors.to_hex(_FAU_CONT(0.25 + 0.6 * i / max(n_m - 1, 1)))
                     for i, m in enumerate(metrics)}
    for j, metric in enumerate(metrics):
        means, stds = [], []
        for model in models:
            vals = np.array(_values_for(rows, model=model, metric=metric))
            means.append(float(np.mean(vals)) if vals.size else 0.0)
            stds.append(float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0)
        x = np.arange(len(models)) + (j - (n_m - 1) / 2) * width
        ax.bar(x, means, width=width * 0.92, color=metric_colors[metric],
               edgecolor=NEUTRAL_TEXT, linewidth=0.6, label=METRIC_LABELS[metric])
        ax.errorbar(x, means, yerr=stds, fmt="none", ecolor=NEUTRAL_TEXT,
                    elinewidth=0.9, capsize=2.5)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("mean ± SD across sessions")
    ax.grid(axis="y", linestyle=":", color=NEUTRAL_GRID)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", ncol=min(3, n_m))
    ax.set_title("Intra-session headline metrics (overall)", loc="left")
    return _save(fig, out_path)


def plot_by_subject(
    rows: List[_Row], models: List[str], metric: str, out_path: Path,
) -> List[Path]:
    """Faceted per model: each subject's per-session scores, coloured by cohort."""
    if metric not in METRIC_LABELS:
        metric = "f1_macro"
    subjects = sorted({r.subject_id for r in rows},
                      key=lambda s: (_subject_group(rows, s), s))
    n = len(models)
    ncols = min(2, n)
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(max(6.0, 0.5 * len(subjects) + 2.0) * ncols / 2 + 2,
                                      3.4 * nrows),
                             squeeze=False)
    width = 0.6
    for k, model in enumerate(models):
        ax = axes[divmod(k, ncols)[0]][divmod(k, ncols)[1]]
        for xi, subj in enumerate(subjects):
            g = _subject_group(rows, subj)
            color = GROUP_COLORS.get(g, "#94a3b8")
            vals = _values_for(rows, model=model, metric=metric, subject=subj)
            if not vals:
                continue
            if len(vals) >= 3:
                _draw_boxes(ax, [vals], [xi], color, width, showfliers=False)
            ax.scatter(np.full(len(vals), xi), vals, s=26, color=color,
                       edgecolor="white", linewidth=0.5, zorder=4)
        ax.set_xticks(range(len(subjects)))
        ax.set_xticklabels(subjects, rotation=60, ha="right", fontsize=7.5)
        ax.set_ylim(0, 1.03)
        ax.set_title(model, loc="left")
        ax.grid(axis="y", linestyle=":", color=NEUTRAL_GRID)
        ax.set_axisbelow(True)
    for j in range(n, nrows * ncols):
        axes[divmod(j, ncols)[0]][divmod(j, ncols)[1]].set_visible(False)
    # Shared cohort legend.
    seen = [g for g in (GROUP_HEALTHY, GROUP_IMPAIRED, GROUP_UNKNOWN)
            if any(_subject_group(rows, s) == g for s in subjects)]
    handles = [plt.Line2D([0], [0], marker="o", linestyle="", markersize=8,
                          markerfacecolor=GROUP_COLORS.get(g, "#94a3b8"),
                          markeredgecolor="white", label=group_label(g))
               for g in seen]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"Intra-session {METRIC_LABELS[metric]} per participant",
                 fontsize=12.5, fontweight="semibold", x=0.02, ha="left")
    fig.subplots_adjust(bottom=0.14, hspace=0.5)
    return _save(fig, out_path)


def plot_confusion_grid(
    rows: List[_Row], models: List[str], *, group: Optional[str],
    out_path: Path, title: str, ncols: int = 4,
) -> List[Path]:
    """Grid of row-normalised confusion matrices, one per model (fau cmap)."""
    mats: Dict[str, Tuple[List[str], np.ndarray]] = {}
    for m in models:
        summed = _sum_confusions(
            [r.confusion for r in rows
             if r.model == m and (group is None or r.group == group)]
        )
        if summed is None:
            continue
        _, names, mat = summed
        mats[m] = (names, _row_normalise(mat))
    if not mats:
        return []

    keys = [m for m in models if m in mats]
    n = len(keys)
    ncols = min(ncols, n)
    nrows = int(math.ceil(n / ncols))
    fig = plt.figure(figsize=(3.0 * ncols + 0.6, 3.0 * nrows + 0.7))
    gs = gridspec.GridSpec(nrows, ncols + 1,
                           width_ratios=[1] * ncols + [0.06],
                           wspace=0.32, hspace=0.55)
    im = None
    for k, model in enumerate(keys):
        r_idx, c_idx = divmod(k, ncols)
        ax = fig.add_subplot(gs[r_idx, c_idx])
        names, mat = mats[model]
        im = ax.imshow(mat, cmap=_FAU_SEQ, vmin=0, vmax=1, aspect="equal")
        ax.set_xticks(range(len(names)))
        ax.set_yticks(range(len(names)))
        ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
        ax.set_yticklabels(names, fontsize=9)
        ax.tick_params(length=0)
        for i_r in range(mat.shape[0]):
            for i_c in range(mat.shape[1]):
                v = mat[i_r, i_c]
                if v < 0.005:
                    continue
                ax.text(i_c, i_r, f"{v:.2f}", ha="center", va="center",
                        color="white" if v >= 0.55 else NEUTRAL_TEXT,
                        fontsize=9, fontweight="bold" if i_r == i_c else "normal")
        ax.set_title(model, fontsize=10)
        for sp in ax.spines.values():
            sp.set_visible(False)
    for j in range(n, nrows * ncols):
        r_idx, c_idx = divmod(j, ncols)
        fig.add_subplot(gs[r_idx, c_idx]).set_visible(False)
    if im is not None:
        cax = fig.add_subplot(gs[:, -1])
        cb = fig.colorbar(im, cax=cax)
        cb.set_label("row-normalised", rotation=90, labelpad=10)
        cb.outline.set_visible(False)
        cb.ax.tick_params(length=0)
    fig.suptitle(title, fontsize=12.5, fontweight="semibold", x=0.02, ha="left",
                 y=0.995)
    return _save(fig, out_path)


def plot_confusion_per_subject(
    rows: List[_Row], models: List[str], out_dir: Path,
    *, ncols: int = 4, filename_prefix: str = "fig_intra_confusion_subject",
) -> Dict[str, List[Path]]:
    """One confusion-matrix grid **per participant**.

    Each figure pools a single subject's sessions and lays the models out
    across columns (same panel style as :func:`plot_confusion_grid`); the
    figure title names the subject and the cohort it belongs to, e.g.
    ``"VP_03  (impaired cohort)"``. Returns ``{subject_id: [paths]}``;
    subjects with no usable confusions are skipped. Subjects are emitted
    healthy → impaired → unknown then by id, matching the by-group figures.
    """
    out_dir = Path(out_dir)
    rank = {GROUP_HEALTHY: 0, GROUP_IMPAIRED: 1, GROUP_UNKNOWN: 2}
    subj_group: Dict[str, str] = {}
    for r in rows:
        subj_group.setdefault(r.subject_id, r.group)

    out: Dict[str, List[Path]] = {}
    for subj in sorted(subj_group, key=lambda s: (rank.get(subj_group[s], 9), s)):
        g = subj_group[subj]
        subj_rows = [r for r in rows if r.subject_id == subj]
        # group=None → pool the (already subject-filtered) rows as-is.
        paths = plot_confusion_grid(
            subj_rows, models, group=None,
            out_path=out_dir / f"{filename_prefix}_{subj}",
            title=f"Intra-session confusion — {subj}  ({group_label(g)} cohort)",
            ncols=ncols,
        )
        if paths:
            out[subj] = paths
    return out


def plot_per_class_f1(
    rows: List[_Row], models: List[str], model_colors: Dict[str, str],
    out_path: Path,
) -> List[Path]:
    """Grouped box-and-whisker of per-class F1 (x = class, hue = model)."""
    classes = _ordered_classes(rows)
    if not classes:
        return []
    fig, ax = plt.subplots(figsize=(max(8.0, 1.4 * len(classes) + 2.0), 5.0))
    n_m = len(models)
    width = 0.82 / n_m
    for j, model in enumerate(models):
        color = model_colors[model]
        positions, data = [], []
        for ci, cname in enumerate(classes):
            positions.append(ci + (j - (n_m - 1) / 2) * width)
            vals = [r.per_class_f1[cname] for r in rows
                    if r.model == model and cname in r.per_class_f1]
            data.append(vals)
        _draw_boxes(ax, data, positions, color, width * 0.85, showfliers=False)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_ylim(0, 1.03)
    ax.set_ylabel("F1  (per-session distribution)")
    ax.grid(axis="y", linestyle=":", color=NEUTRAL_GRID)
    ax.set_axisbelow(True)
    for i in range(1, len(classes)):
        ax.axvline(i - 0.5, color=NEUTRAL_GRID, linewidth=0.8, zorder=1)
    handles = [plt.Rectangle((0, 0), 1, 1, color=model_colors[m], alpha=0.88,
                             ec=NEUTRAL_TEXT, lw=0.6) for m in models]
    ax.legend(handles, models, ncol=min(4, n_m), loc="lower center",
              bbox_to_anchor=(0.5, -0.22))
    ax.set_title("Intra-session per-class F1 by model", loc="left")
    return _save(fig, out_path)


def plot_latency(
    rows: List[_Row], models: List[str], model_colors: Dict[str, str],
    out_path: Path,
) -> List[Path]:
    """Inference latency per model — real-time relevance for EMG (≈50 ms cadence)."""
    fig, ax = plt.subplots(figsize=(1.5 * len(models) + 2.5, 4.2))
    any_data = False
    for j, model in enumerate(models):
        vals = [r.inference_ms for r in rows
                if r.model == model and r.inference_ms is not None]
        if not vals:
            continue
        any_data = True
        _draw_boxes(ax, [vals], [j], model_colors[model], 0.6)
        _scatter_points(ax, [j], [vals], model_colors[model], 0.6, seed=j)
    if not any_data:
        plt.close(fig)
        return []
    ax.axhline(50.0, color=GROUP_COLORS[GROUP_IMPAIRED], linestyle="--",
               linewidth=1.0, alpha=0.7)
    ax.text(len(models) - 0.5, 51, "≈50 ms decision cadence", ha="right",
            va="bottom", fontsize=8, color=NEUTRAL_MUTED)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.set_ylabel("ms / window")
    ax.grid(axis="y", linestyle=":", color=NEUTRAL_GRID)
    ax.set_axisbelow(True)
    ax.set_title("Intra-session inference latency by model", loc="left")
    return _save(fig, out_path)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _subject_group(rows: List[_Row], subject: str) -> str:
    for r in rows:
        if r.subject_id == subject:
            return r.group
    return GROUP_UNKNOWN


def _ordered_classes(rows: List[_Row]) -> List[str]:
    canonical = ["Rest", "Fist", "Pinch", "Tripod"]
    present = {c for r in rows for c in r.per_class_f1}
    ordered = [c for c in canonical if c in present]
    ordered += sorted(present - set(ordered))
    return ordered


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Intra-session model comparison (overall / group / subject).",
    )
    p.add_argument("--data-dir", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--models", nargs="+", required=True,
                   help="e.g. random_forest catboost cnn mstnet")
    p.add_argument("--test-fraction", type=float, default=0.25)
    p.add_argument("--window-ms", type=int, default=None)
    p.add_argument("--stride-ms", type=int, default=50)
    p.add_argument("--split-unit", choices=["trial", "window"], default="trial")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--groups-file", type=Path, default=None)
    p.add_argument("--subjects", nargs="*", default=None)
    p.add_argument("--include-unity", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    cfg = IntraSessionComparisonConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        models=list(args.models),
        test_fraction=args.test_fraction,
        window_size_ms=args.window_ms,
        window_stride_ms=args.stride_ms,
        split_unit=args.split_unit,
        seed=args.seed,
        groups_file=args.groups_file,
        subjects=args.subjects,
        include_unity=args.include_unity,
    )

    def _progress(frac: float, msg: str) -> None:
        log.info("[%3.0f%%] %s", 100 * frac, msg)

    res = run_intra_session_comparison(cfg, progress=_progress)

    print("\n=== Intra-session comparison complete ===")
    print(f"models:  {', '.join(res.models)}")
    print(f"cohorts: {', '.join(group_label(g) for g in res.groups_present)}")
    print("CSVs:")
    for k, v in res.csvs.items():
        print(f"  {k:12s} {v}")
    print("Figures:")
    for k, paths in res.figures.items():
        png = next((str(p) for p in paths if p.suffix == ".png"), "")
        print(f"  {k:22s} {png}")
    for n in res.notes:
        print(f"NOTE: {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())