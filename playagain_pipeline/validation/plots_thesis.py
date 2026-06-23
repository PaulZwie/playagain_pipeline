"""
validation/plots_thesis.py
──────────────────────────
Publication-grade figure generators for Chapter 6 / 7.

v3 — uses the FAU institutional palette, replaces the per-class F1 bar
chart with grouped box-and-whisker plots, and detects which calibration
metric is stored so axis labels stay honest.

Public API (unchanged — generate_thesis_outputs keeps working):
* plot_calibration_confidence
* plot_calibration_honest
* plot_calibration_vs_f1
* plot_calibration_vs_f1_per_model
* plot_confusion_matrices
* plot_feature_ablation
* plot_per_class_f1
* plot_per_session_variability

If the optional per-fold per-class CSV (``fig_per_class_f1_per_fold.csv``)
is present alongside the summary CSV, :func:`plot_per_class_f1` draws
real grouped boxplots. Otherwise it falls back to a synthetic-box view
(box from mean±SD, median at mean) that visually matches the boxplot
style but is honestly labelled as a summary.
"""
from __future__ import annotations

import csv
import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")  # headless; avoids Qt conflicts in the GUI process

import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402
from matplotlib import gridspec   # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FAU palette — with a graceful fallback for environments without the pkg
# ---------------------------------------------------------------------------

# Five-shade tech palette (light → dark) and the parallel tech_dark
# palette. We pull both from fau_colors.cmaps when available so the
# values stay in sync with the institutional spec, and fall back to
# hard-coded shades when the package is missing.
try:
    from fau_colors import colors as _FAU, colors_dark as _FAU_D, cmaps as _CMAPS
    _TECH_SHADES: List[str] = list(_CMAPS.tech)           # 5 shades
    _TECH_DARK_SHADES: List[str] = list(_CMAPS.tech_dark) # 5 shades
    _PALETTE = {
        # Keep the broad-faculty names available for any future use,
        # but every plot in this module now draws from the tech family.
        "fau":       _FAU.fau,
        "tech":      _FAU.tech,
        "phil":      _FAU.phil,
        "med":       _FAU.med,
        "nat":       _FAU.nat,
        "wiso":      _FAU.wiso,
        "fau_d":     _FAU_D.fau,
        "tech_d":    _FAU_D.tech,
        "phil_d":    _FAU_D.phil,
        "med_d":     _FAU_D.med,
        "nat_d":     _FAU_D.nat,
        "wiso_d":    _FAU_D.wiso,
    }
    HAS_FAU = True
except Exception:                       # noqa: BLE001
    _TECH_SHADES      = ["#8C9FB1", "#B6C2CE", "#D3DAE1", "#E2E7EB", "#EBF5F7"]
    _TECH_DARK_SHADES = ["#2F586E", "#7C96A3", "#B0BFC8", "#CBD5DB", "#E4E9EC"]
    _PALETTE = {
        "fau":    "#04316A", "tech":   "#8C9FB1", "phil":   "#FDB735",
        "med":    "#18B4F1", "nat":    "#7BB725", "wiso":   "#C50F3C",
        "fau_d":  "#041E42", "tech_d": "#2F586E", "phil_d": "#E87722",
        "med_d":  "#005287", "nat_d":  "#266141", "wiso_d": "#971B2F",
    }
    HAS_FAU = False

# Group colours — Healthy is the lightest tech shade, Impaired is the
# darkest tech_dark shade. These are the headline two colours for any
# cohort-split figure.
GROUP_COLORS: Dict[str, str] = {
    "healthy":  _TECH_SHADES[1],       # #B6C2CE
    "impaired": _TECH_DARK_SHADES[0],  # #2F586E
}
# Title-accent colour used on panels for deep models so a reader can
# tell at a glance which panels are classical vs deep.
DEEP_MODEL_ACCENT = _PALETTE["wiso"]    # FAU red
CLASSIC_MODEL_ACCENT = _PALETTE["fau"]  # FAU blue
DEEP_MODEL_TYPES = frozenset({"mlp", "cnn", "attention_net", "mstnet"})


# Model → colour mapping in the tech family. We interleave the two
# tech ramps so adjacent models in the legend stay distinguishable
# while every fill is still inside the institutional palette.
_TECH_RAMP = _TECH_DARK_SHADES + _TECH_SHADES[:3]
MODEL_COLORS: Dict[str, str] = {
    "lda":           _TECH_DARK_SHADES[0],   # darkest slate
    "svm":           _TECH_DARK_SHADES[1],
    "random_forest": _TECH_DARK_SHADES[2],
    "catboost":      _TECH_DARK_SHADES[3],
    "mlp":           _TECH_SHADES[0],
    "cnn":           _TECH_SHADES[1],
    "mstnet":        _TECH_SHADES[2],
    "attention_net": _TECH_SHADES[3],
}

# Subjects cycle through the tech ramp so adjacent subjects in any
# sorted view get visually distinct shades.
_SUBJECT_CYCLE: List[str] = (
    _TECH_DARK_SHADES + _TECH_SHADES
)

NEUTRAL_TEXT  = "#1f2937"
NEUTRAL_MUTED = "#6b7280"
NEUTRAL_GRID  = "#e2e8f0"
BOX_FACE      = "#eef3f8"


def _setup_mpl() -> None:
    matplotlib.rcParams.update({
        "font.family":          "sans-serif",
        "font.sans-serif":      ["DejaVu Sans", "Arial"],
        "font.size":            10,
        "axes.titlesize":       11,
        "axes.labelsize":       10,
        "axes.titleweight":     "semibold",
        "axes.spines.top":      False,
        "axes.spines.right":    False,
        "axes.edgecolor":       NEUTRAL_GRID,
        "axes.labelcolor":      NEUTRAL_TEXT,
        "axes.titlepad":        8,
        "xtick.color":          NEUTRAL_TEXT,
        "ytick.color":          NEUTRAL_TEXT,
        "xtick.labelsize":      9,
        "ytick.labelsize":      9,
        "legend.fontsize":      9,
        "legend.frameon":       False,
        "figure.dpi":           150,
        "savefig.dpi":          300,
        "savefig.bbox":         "tight",
        "savefig.facecolor":    "white",
        "pdf.fonttype":         42,    # editable text in PDF
        "ps.fonttype":          42,
    })


def _model_color(name: str) -> str:
    return MODEL_COLORS.get(name, "#94a3b8")


def _subject_palette(subjects: Sequence[str]) -> Dict[str, str]:
    """Stable subject → colour mapping. Sorted so two figures of the
    same subjects use matching colours."""
    out: Dict[str, str] = {}
    for i, s in enumerate(sorted(subjects)):
        out[s] = _SUBJECT_CYCLE[i % len(_SUBJECT_CYCLE)]
    return out


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


# ---------------------------------------------------------------------------
# Detect which calibration metric the data carries
# ---------------------------------------------------------------------------

def _is_stability_metric(per_session_rows: Iterable[Dict[str, Any]]) -> bool:
    """Detect whether the stored ``rotation_confidence`` is stability.

    Honours an explicit ``rotation_metric_version == 2`` flag when
    present (set by ``recompute_calibration_metrics``). Otherwise looks
    for the stability signature: lots of values at 1.0 (every trial
    agreed) and/or strong clustering at simple rational fractions
    {0, 1/n, …, 1}. The legacy peak-prominence metric tends to live
    in a narrow continuous band like 0.20-0.50 and almost never hits
    exactly 1.0, which makes these heuristics reliable for typical
    corpora — but the explicit flag remains the source of truth.
    """
    rows = list(per_session_rows)

    # Source of truth — explicit flag from recompute_calibration_metrics.
    for row in rows:
        if row.get("rotation_metric_version") == 2:
            return True
        cm = row.get("custom_metadata")
        if isinstance(cm, dict) and cm.get("rotation_metric_version") == 2:
            return True

    vals = [r["confidence"] for r in rows
            if r.get("confidence") is not None]
    if not vals:
        return False

    # Signature 1: a sizeable fraction of values pin to exactly 1.0.
    frac_at_one = sum(1 for v in vals if v >= 0.999) / len(vals)
    if frac_at_one >= 0.20:
        return True

    # Signature 2: most values land on simple rational fractions. The
    # stability metric for n_trials ∈ {2,3,4,5,6,10} only takes values
    # in {0, 1/n, 2/n, …, 1}. We tolerate ±0.01 to allow for rounding.
    rational = {0.0, 1.0,
                1/2,
                1/3, 2/3,
                1/4, 3/4,
                1/5, 2/5, 3/5, 4/5,
                1/6, 5/6,
                1/10, 3/10, 7/10, 9/10}
    near_rational = sum(
        1 for v in vals
        if any(abs(v - q) < 0.01 for q in rational)
    ) / len(vals)
    return near_rational >= 0.80


def _confidence_axis_label(is_stability: bool) -> str:
    return ("Calibration stability  (fraction of trials agreeing on offset)"
            if is_stability else
            "Calibration confidence  (xcorr peak prominence)")


# ---------------------------------------------------------------------------
# Fig 6.1 — Calibration confidence distribution
# ---------------------------------------------------------------------------

def plot_calibration_confidence(
    calibration_report_json: Path,
    out_path: Path,
    *,
    threshold: Optional[float] = None,
    bins: int = 20,
) -> List[Path]:
    """Histogram of rotation calibration confidence + subject strip plot."""
    _setup_mpl()
    with Path(calibration_report_json).open("r", encoding="utf-8") as f:
        data = json.load(f)
    if threshold is None:
        threshold = float(data.get("flag_threshold", 0.5))

    per_sess = data.get("per_session", [])
    confs = [s["confidence"] for s in per_sess if s.get("confidence") is not None]
    if not confs:
        log.warning("No calibration confidence values to plot.")
        return []
    is_stab = _is_stability_metric(per_sess)

    fig = plt.figure(figsize=(7.4, 4.2))
    gs  = gridspec.GridSpec(2, 1, height_ratios=[3.0, 1.0], hspace=0.05)
    ax_h  = fig.add_subplot(gs[0])
    ax_s  = fig.add_subplot(gs[1], sharex=ax_h)

    x_hi = max(max(confs), threshold, 0.4) + 0.05
    edges = np.linspace(0.0, x_hi, bins + 1)
    ax_h.hist(confs, bins=edges,
              color=_TECH_SHADES[1], edgecolor=_TECH_DARK_SHADES[0],
              linewidth=1.1, alpha=0.85)
    ax_h.axvline(threshold, color=NEUTRAL_MUTED, linestyle="--",
                 linewidth=1.0, label=f"flag threshold = {threshold:.2f}")
    if math.isfinite(data.get("median", float("nan"))):
        ax_h.axvline(data["median"], color=DEEP_MODEL_ACCENT, linestyle=":",
                     linewidth=1.4, label=f"median = {data['median']:.2f}")
    ax_h.set_ylabel("Number of sessions")
    title = ("Rotation calibration stability across sessions"
             if is_stab else
             "Rotation calibration confidence across sessions")
    ax_h.set_title(title, loc="left")
    ax_h.legend(loc="upper right")
    ax_h.tick_params(labelbottom=False)
    ax_h.grid(axis="y", linestyle=":", color=NEUTRAL_GRID)
    ax_h.set_axisbelow(True)

    # Subject strip plot beneath
    subjects = sorted({s["subject_id"] for s in per_sess
                       if s.get("confidence") is not None})
    palette  = _subject_palette(subjects)
    rng      = np.random.default_rng(0)
    for subj in subjects:
        xs = [s["confidence"] for s in per_sess
              if s["subject_id"] == subj and s.get("confidence") is not None]
        ys = rng.normal(0.0, 0.08, size=len(xs))
        ax_s.scatter(xs, ys, s=46, color=palette[subj],
                     edgecolor="white", linewidth=0.7,
                     label=subj, zorder=3)
    ax_s.set_yticks([])
    ax_s.set_ylim(-0.4, 0.4)
    ax_s.set_xlim(0.0, x_hi)
    ax_s.set_xlabel(_confidence_axis_label(is_stab))
    ax_s.grid(axis="x", linestyle=":", color=NEUTRAL_GRID)
    for sp in ("top", "left", "right"):
        ax_s.spines[sp].set_visible(False)
    if subjects:
        ncol = min(len(subjects), 6)
        ax_s.legend(loc="upper center", ncol=ncol,
                    bbox_to_anchor=(0.5, -0.45),
                    handletextpad=0.3, columnspacing=1.0)
    return _save(fig, out_path)


# ---------------------------------------------------------------------------
# Fig 6.1b — Three-panel honest calibration view
# ---------------------------------------------------------------------------

def plot_calibration_honest(
    calibration_report_json: Path,
    out_path: Path,
) -> List[Path]:
    """Three-panel re-framing of the §6.2 calibration story."""
    _setup_mpl()
    with Path(calibration_report_json).open("r", encoding="utf-8") as f:
        data = json.load(f)
    rows = [s for s in data.get("per_session", [])
            if s.get("confidence") is not None and s.get("offset") is not None]
    if not rows:
        log.warning("No usable per-session rows for the honest figure.")
        return []
    is_stab = _is_stability_metric(rows)

    subjects = sorted({s["subject_id"] for s in rows})
    palette  = _subject_palette(subjects)
    by_subj: Dict[str, List[Dict[str, Any]]] = {}
    for s in rows:
        by_subj.setdefault(s["subject_id"], []).append(s)

    fig = plt.figure(figsize=(12.0, 4.6))
    gs  = gridspec.GridSpec(1, 3, wspace=0.34, left=0.06, right=0.985,
                            top=0.83, bottom=0.18)

    # (a) Detected offset per session
    ax = fig.add_subplot(gs[0])
    max_offset   = max(s["offset"] for s in rows)
    max_sessions = max(len(v) for v in by_subj.values())
    for subj in subjects:
        sub = by_subj[subj]
        xs  = list(range(1, len(sub) + 1))
        ys  = [s["offset"] for s in sub]
        ax.plot(xs, ys, "-o", color=palette[subj], linewidth=1.6,
                markersize=8, markeredgecolor="white", label=subj)
    ax.set_xlabel("Session index (within subject)")
    ax.set_ylabel("Detected rotation offset (channels)")
    ax.set_title("(a)  Detected offset per session", loc="left", fontsize=10.5)
    ax.set_yticks(range(max_offset + 2))
    ax.set_xticks(range(1, max_sessions + 1))
    ax.grid(axis="y", linestyle=":", color=NEUTRAL_GRID)
    ax.set_axisbelow(True)
    ax.legend(loc="center right", framealpha=0.85, frameon=True,
              edgecolor=NEUTRAL_GRID, facecolor="white")

    # (b) Within-subject stability
    ax = fig.add_subplot(gs[1])
    for i, subj in enumerate(subjects):
        offsets = np.asarray([s["offset"] for s in by_subj[subj]], dtype=int)
        if offsets.size < 2:
            ax.bar(i, 1.0, color=palette[subj], edgecolor="white",
                   linewidth=1.0, alpha=0.6, hatch="//")
            ax.text(i, 0.5, "n = 1\nreference",
                    ha="center", va="center", fontsize=9,
                    color="white", fontweight="semibold")
            continue
        mode  = int(np.bincount(offsets - offsets.min()).argmax()) + int(offsets.min())
        frac  = float(np.mean(offsets == mode))
        n_uniq = int(np.unique(offsets).size)
        ax.bar(i, frac, color=palette[subj], edgecolor="white",
               linewidth=1.0, alpha=0.9)
        ax.text(i, frac + 0.02,
                f"{int(frac * 100)}%\n{n_uniq} unique",
                ha="center", va="bottom", fontsize=9, color=NEUTRAL_TEXT)
    ax.set_xticks(range(len(subjects)))
    ax.set_xticklabels(subjects)
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("Fraction of sessions on the median offset")
    ax.set_title("(b)  Within-subject offset stability",
                 loc="left", fontsize=10.5)
    ax.grid(axis="y", linestyle=":", color=NEUTRAL_GRID)
    ax.set_axisbelow(True)

    # (c) Stored confidence on its real range
    ax = fig.add_subplot(gs[2])
    for subj in subjects:
        xs = [s["offset"]     for s in by_subj[subj]]
        ys = [s["confidence"] for s in by_subj[subj]]
        ax.scatter(xs, ys, s=100, color=palette[subj],
                   edgecolor="white", linewidth=0.9,
                   label=subj, zorder=3)
    ax.set_xlabel("Detected offset (channels)")
    ax.set_ylabel("Stored stability value" if is_stab else
                  'Stored "confidence" value')
    ax.set_xlim(-1, max_offset + 1)
    if is_stab:
        ax.set_ylim(0, 1.18)
    else:
        y_hi = max([s["confidence"] for s in rows]) * 1.25
        ax.set_ylim(0, max(y_hi, 0.5))
    panel_c_title = ("(c)  Stability per session"
                     if is_stab else "(c)  Stored confidence on its real range")
    ax.set_title(panel_c_title, loc="left", fontsize=10.5)
    ax.legend(loc="lower right")
    ax.grid(linestyle=":", color=NEUTRAL_GRID)
    ax.set_axisbelow(True)

    fig.suptitle("Calibration — three views of the same sessions",
                 fontsize=12.5, fontweight="semibold",
                 x=0.06, ha="left", y=0.96)

    if is_stab:
        footer = (
            "(a) is the operational quantity — which channel index the bracelet "
            "is on. (b) is the calibration question: do we agree across sessions "
            "of the same subject? (c) plots the stored stability value (fraction "
            "of sync trials agreeing on the modal offset)."
        )
    else:
        footer = (
            "(a) is the operational quantity — which channel index the bracelet "
            "is on. (b) is the actual calibration question: do we agree across "
            "sessions of the same subject? (c) is the stored 'confidence' (xcorr "
            "peak prominence) — it measures angular peakedness of the sync "
            "gesture, an anatomy property, not detection accuracy."
        )
    fig.text(0.5, 0.04, footer,
             ha="center", va="top", fontsize=8.5, style="italic",
             color=NEUTRAL_MUTED, wrap=True)
    return _save(fig, out_path)


# ---------------------------------------------------------------------------
# Fig 6.3 — Per-class F1 BOXPLOTS
# ---------------------------------------------------------------------------

def _load_per_fold_per_class(path: Path) -> Dict[Tuple[str, str], List[float]]:
    """Load the per-fold per-class CSV when present.

    Expected columns: ``model, class, fold_id, f1``.
    Returns ``{(model, class): [f1_per_fold]}`` — empty when no data.
    """
    out: Dict[Tuple[str, str], List[float]] = {}
    if not Path(path).exists():
        return out
    with Path(path).open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                v = float(r["f1"])
            except (KeyError, TypeError, ValueError):
                continue
            out.setdefault((r["model"], r["class"]), []).append(v)
    return out


def plot_per_class_f1(
    per_class_csv: Path,
    out_path: Path,
    *,
    models: Optional[List[str]] = None,
    classes: Optional[List[str]] = None,
    summary_csv: Optional[Path] = None,
    per_fold_csv: Optional[Path] = None,
) -> List[Path]:
    """Grouped box-and-whisker plot of per-class F1 by model.

    Uses real fold-level boxes when ``per_fold_csv`` is supplied (or a
    file named ``fig_per_class_f1_per_fold.csv`` exists next to the
    summary CSV). Otherwise falls back to a "summary box" — Q1=mean−SD,
    Q3=mean+SD, median at mean — which is visually consistent but
    honestly labelled as a summary.
    """
    _setup_mpl()
    rows: List[Dict[str, Any]] = []
    with Path(per_class_csv).open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append({
                "model": r["model"],
                "class": r["class"],
                "mean":  float(r["mean"]),
                "std":   float(r["std"]),
                "n_folds": int(float(r.get("n_folds", 1))),
            })
    if not rows:
        log.warning("No rows in %s", per_class_csv)
        return []

    # Resolve per-fold data (real boxplots vs synthetic boxes)
    if per_fold_csv is None:
        candidate = Path(per_class_csv).with_name("fig_per_class_f1_per_fold.csv")
        per_fold_csv = candidate if candidate.exists() else None
    fold_data = _load_per_fold_per_class(per_fold_csv) if per_fold_csv else {}
    is_real_box = bool(fold_data)

    if classes is None:
        canonical = ["Rest", "Fist", "Pinch", "Tripod"]
        present   = {r["class"] for r in rows}
        classes = [c for c in canonical if c in present]
        classes += sorted(present - set(classes))

    if summary_csv is not None and Path(summary_csv).exists():
        summary_rows = []
        with Path(summary_csv).open("r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                summary_rows.append((r["model"], float(r["macro_f1_mean"])))
        all_models = [m for m, _ in sorted(summary_rows, key=lambda kv: -kv[1])]
    else:
        means_by_model: Dict[str, List[float]] = {}
        for r in rows:
            means_by_model.setdefault(r["model"], []).append(r["mean"])
        all_models = sorted(means_by_model,
                            key=lambda m: -float(np.mean(means_by_model[m])))
    if models:
        all_models = [m for m in all_models if m in models]
    lookup = {(r["model"], r["class"]): r for r in rows}

    fig, ax = plt.subplots(figsize=(11.2, 5.4))
    n_models = len(all_models)
    width = 0.84 / n_models

    for j, model in enumerate(all_models):
        positions = np.arange(len(classes)) + (j - (n_models - 1) / 2) * width
        color = _model_color(model)

        per_class_data: List[List[float]] = []
        for c in classes:
            if is_real_box and (model, c) in fold_data:
                per_class_data.append(fold_data[(model, c)])
            else:
                rec = lookup.get((model, c), {})
                m, sd = rec.get("mean", np.nan), rec.get("std", 0.0)
                if math.isnan(m):
                    per_class_data.append([])
                else:
                    lo = max(0.0, m - sd)
                    hi = min(1.0, m + sd)
                    per_class_data.append([lo, m, hi])

        plot_positions = [p for p, d in zip(positions, per_class_data) if d]
        plot_data      = [d for d in per_class_data if d]
        if not plot_data:
            continue

        ax.boxplot(
            plot_data,
            positions=plot_positions,
            widths=width * 0.85,
            patch_artist=True,
            showfliers=is_real_box,
            medianprops=dict(color=NEUTRAL_TEXT, linewidth=1.6),
            boxprops=dict(facecolor=color, edgecolor=NEUTRAL_TEXT,
                          linewidth=0.8, alpha=0.85),
            whiskerprops=dict(color=NEUTRAL_TEXT, linewidth=0.9),
            capprops=dict(color=NEUTRAL_TEXT, linewidth=0.9),
            flierprops=dict(marker="o", markersize=2.5,
                            markerfacecolor=color,
                            markeredgecolor=NEUTRAL_TEXT,
                            markeredgewidth=0.4),
        )
        # Overlay individual fold points when real data is present
        if is_real_box:
            rng = np.random.default_rng(j)
            for pos, vals in zip(plot_positions, plot_data):
                if len(vals) <= 1:
                    continue
                jitter = rng.normal(0, width * 0.10, size=len(vals))
                ax.scatter(np.full(len(vals), pos) + jitter, vals,
                           s=10, color=color, edgecolor="white",
                           linewidth=0.4, zorder=4, alpha=0.75)

    handles = [plt.Rectangle((0, 0), 1, 1, color=_model_color(m), alpha=0.85,
                             ec=NEUTRAL_TEXT, lw=0.6)
               for m in all_models]
    ax.legend(handles, all_models, ncol=min(4, n_models),
              loc="lower center", bbox_to_anchor=(0.5, -0.22),
              columnspacing=1.6, handletextpad=0.6)

    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_ylim(0, 1.05)
    if is_real_box:
        ax.set_ylabel("F1 score  (LOSO-session, per-fold distribution)")
        ax.set_title("Per-class F1 by model", loc="left")
    else:
        ax.set_ylabel("F1 score  (summary box: median = mean, IQR ≈ ± SD)")
        ax.set_title("Per-class F1 by model  (summary view)",
                     loc="left")
    ax.grid(axis="y", linestyle=":", color=NEUTRAL_GRID)
    ax.set_axisbelow(True)
    for i in range(1, len(classes)):
        ax.axvline(i - 0.5, color=NEUTRAL_GRID, linewidth=0.8, zorder=1)

    if not is_real_box:
        fig.text(0.5, -0.04,
                 "Per-fold values were not available; boxes summarise "
                 "mean ± SD across folds (median line = mean, box edges = "
                 "mean ± SD, clipped to [0, 1]). Provide "
                 "fig_per_class_f1_per_fold.csv to get true boxplots.",
                 ha="center", va="top", fontsize=8.5, style="italic",
                 color=NEUTRAL_MUTED, wrap=True)
    return _save(fig, out_path)


# ---------------------------------------------------------------------------
# Fig 6.4 — Confusion matrices
# ---------------------------------------------------------------------------

def plot_confusion_matrices(
    confusion_json: Path,
    out_path: Path,
    *,
    models: Optional[List[str]] = None,
    ncols: int = 4,
    summary_csv: Optional[Path] = None,
    macro_f1: Optional[Dict[str, float]] = None,
    title: Optional[str] = None,
) -> List[Path]:
    _setup_mpl()
    with Path(confusion_json).open("r", encoding="utf-8") as f:
        confs = json.load(f)
    if not confs:
        log.warning("No confusion matrices to plot.")
        return []
    if models:
        confs = {m: confs[m] for m in models if m in confs}

    # Macro-F1 per model drives both the panel ordering and the per-panel
    # title annotation. Callers can pass it directly (``macro_f1=``) — the
    # per-cohort figures use this so each panel shows the cohort's own F1
    # rather than the pooled number — or point at the run summary CSV as
    # before. A passed dict takes precedence over the CSV.
    macro_f1 = dict(macro_f1) if macro_f1 else {}
    if not macro_f1 and summary_csv is not None and Path(summary_csv).exists():
        with Path(summary_csv).open("r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["model"] in confs:
                    macro_f1[r["model"]] = float(r["macro_f1_mean"])
    if macro_f1:
        keys = [m for m in sorted(confs, key=lambda m: -macro_f1.get(m, 0.0))]
    else:
        keys = list(confs)

    n = len(keys)
    nrows = int(math.ceil(n / ncols))
    fig = plt.figure(figsize=(3.0 * ncols + 0.6, 3.0 * nrows + 0.7))
    gs  = gridspec.GridSpec(nrows, ncols + 1,
                            width_ratios=[1] * ncols + [0.06],
                            wspace=0.32, hspace=0.55)

    # Tech-palette colormap for the matrix (lightest → darkest tech)
    cmap = LinearSegmentedColormap.from_list(
        "thesis_tech",
        ["#FFFFFF", _TECH_SHADES[3], _TECH_SHADES[1], _TECH_DARK_SHADES[0]],
    )

    im = None
    for k, model in enumerate(keys):
        r_idx, c_idx = divmod(k, ncols)
        ax = fig.add_subplot(gs[r_idx, c_idx])
        c   = confs[model]
        mat = np.asarray(c["matrix_norm"], dtype=np.float64)
        names = c["label_names"]

        im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=1, aspect="equal")
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
                txt_c  = "white" if v >= 0.55 else NEUTRAL_TEXT
                weight = "bold" if i_r == i_c else "normal"
                ax.text(i_c, i_r, f"{v:.2f}", ha="center", va="center",
                        color=txt_c, fontsize=9, fontweight=weight)

        if model in macro_f1:
            ax.set_title(f"{model}\nmacro F1 = {macro_f1[model]:.3f}",
                         fontsize=10)
        else:
            ax.set_title(model, fontsize=10)
        for sp in ax.spines.values():
            sp.set_visible(False)

    for j in range(n, nrows * ncols):
        r_idx, c_idx = divmod(j, ncols)
        fig.add_subplot(gs[r_idx, c_idx]).set_visible(False)

    if im is not None:
        cax = fig.add_subplot(gs[:, -1])
        cb  = fig.colorbar(im, cax=cax)
        cb.set_label("Row-normalised count", rotation=90, labelpad=10)
        cb.outline.set_visible(False)
        cb.ax.tick_params(length=0)

    fig.suptitle(title or "Normalised confusion matrices  (LOSO-session, pooled)",
                 fontsize=12.5, fontweight="semibold",
                 x=0.02, ha="left", y=0.995)
    return _save(fig, out_path)


# ---------------------------------------------------------------------------
# Fig 6.5 — Per-session variability
# ---------------------------------------------------------------------------

def plot_per_session_variability(
    per_session_csv: Path,
    out_path: Path,
    *,
    model: Optional[str] = None,
) -> List[Path]:
    _setup_mpl()
    rows: List[Dict[str, Any]] = []
    with Path(per_session_csv).open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    if not rows:
        log.warning("No rows in %s", per_session_csv)
        return []

    if model:
        rows = [r for r in rows if r["model"] == model]
    else:
        first = rows[0]["model"]
        rows = [r for r in rows if r["model"] == first]
        model = first
    if not rows:
        log.warning("No rows after filtering to model=%r", model)
        return []

    by_subject: Dict[str, List[float]] = {}
    for r in rows:
        by_subject.setdefault(r["subject_id"], []).append(float(r["macro_f1"]))
    subjects = sorted(by_subject)
    palette  = _subject_palette(subjects)
    data     = [by_subject[s] for s in subjects]

    fig, ax = plt.subplots(figsize=(max(5.0, 0.9 * len(subjects) + 2.5), 4.2))
    ax.boxplot(
        data, positions=range(len(subjects)),
        widths=0.5, patch_artist=True,
        medianprops=dict(color=_TECH_DARK_SHADES[0], linewidth=1.6),
        boxprops=dict(facecolor=_TECH_SHADES[2],
                      edgecolor=_TECH_DARK_SHADES[0], linewidth=1.0),
        whiskerprops=dict(color=_TECH_DARK_SHADES[0], linewidth=1.0),
        capprops=dict(color=_TECH_DARK_SHADES[0], linewidth=1.0),
        flierprops=dict(marker="d", markersize=4,
                        markerfacecolor=_TECH_DARK_SHADES[0],
                        markeredgecolor=_TECH_DARK_SHADES[0]),
    )

    rng = np.random.default_rng(0)
    for i, subj in enumerate(subjects):
        vals = np.asarray(by_subject[subj])
        x    = np.full_like(vals, i, dtype=float) + rng.normal(0, 0.06, vals.size)
        ax.scatter(x, vals, s=44, color=palette[subj],
                   edgecolor="white", linewidth=0.7, zorder=4)
        m = float(np.mean(vals))
        ax.hlines(m, i - 0.24, i + 0.24,
                  colors=DEEP_MODEL_ACCENT, linewidth=2.0, zorder=5,
                  label="subject mean" if i == 0 else None)

    ax.set_xticks(range(len(subjects)))
    ax.set_xticklabels(subjects, rotation=0)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Macro F1  (one dot = one held-out session)")
    ax.set_xlabel("Participant")
    ax.set_title(f"Per-session variability — {model}", loc="left")
    ax.grid(axis="y", linestyle=":", color=NEUTRAL_GRID)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right")
    return _save(fig, out_path)


# ---------------------------------------------------------------------------
# Per-subject boxplot grid, faceted by model, coloured by cohort.
# Consumes the by_group CSV that generate_thesis_outputs already writes.
# Matches the institutional "Healthy = light tech, Impaired = slate" look.
# ---------------------------------------------------------------------------

def plot_per_subject_by_group_box(
    by_group_csv: Path,
    out_path: Path,
    *,
    models: Optional[Sequence[str]] = None,
    ncols: int = 4,
) -> List[Path]:
    """
    One panel per model — for each held-out subject, a box-and-whisker
    of the per-fold macro-F1 values for that subject, coloured by cohort
    (Healthy vs Impaired).

    The CSV is the same one ``generate_thesis_outputs`` writes for
    ``fig_6_5_per_session_variability_by_group.csv`` /
    ``fig_6_4b_loso_subject_by_group.csv``. It must have columns
    ``subject_id``, ``group``, ``model``, ``macro_f1``.

    Subjects appear on the x-axis in cohort order (healthy first then
    impaired), each subject keeps the same x-position across panels so a
    reader can compare model-to-model. Impaired-subject labels are bold.
    """
    _setup_mpl()
    rows: List[Dict[str, Any]] = []
    with Path(by_group_csv).open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                val = float(r["macro_f1"])
            except (TypeError, ValueError):
                continue
            rows.append({
                "subject":  str(r.get("subject_id", "")),
                "group":    str(r.get("group", "?")).strip().lower(),
                "model":    str(r.get("model", "")),
                "f1":       val,
            })
    if not rows:
        log.warning("No usable rows in %s", by_group_csv)
        return []

    if models:
        wanted = list(models)
    else:
        # Preserve first-seen order so the layout is deterministic.
        seen: List[str] = []
        for r in rows:
            if r["model"] not in seen:
                seen.append(r["model"])
        wanted = seen
    if not wanted:
        return []

    # Subject ordering: healthy first then impaired then unknown,
    # alphabetical within each cohort. Same across all panels.
    subj_group: Dict[str, str] = {}
    for r in rows:
        subj_group.setdefault(r["subject"], r["group"])
    group_rank = {"healthy": 0, "impaired": 1}
    subjects = sorted(
        subj_group,
        key=lambda s: (group_rank.get(subj_group.get(s, "?"), 9), s),
    )
    if not subjects:
        return []

    # Pivot: {model: {subject: [f1...]}}
    cube: Dict[str, Dict[str, List[float]]] = {m: {s: [] for s in subjects}
                                               for m in wanted}
    for r in rows:
        if r["model"] in cube and r["subject"] in cube[r["model"]]:
            cube[r["model"]][r["subject"]].append(r["f1"])

    n_models = len(wanted)
    ncols    = max(1, min(int(ncols), n_models))
    nrows    = int(math.ceil(n_models / ncols))
    fig_w    = max(4.6, 0.9 * len(subjects) + 1.4) * ncols
    fig_h    = 4.2 * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h),
                             sharey=True, squeeze=False)

    for k, model in enumerate(wanted):
        r_idx, c_idx = divmod(k, ncols)
        ax = axes[r_idx][c_idx]

        # Build per-cohort box data + positions side-by-side per subject.
        positions:  List[float] = []
        box_data:   List[List[float]] = []
        face_colors: List[str] = []
        edge_colors: List[str] = []
        offsets = {"healthy": -0.22, "impaired": +0.22}
        width   = 0.36

        for i, subj in enumerate(subjects):
            vals = cube[model].get(subj, [])
            grp  = subj_group.get(subj, "?")
            if not vals:
                continue
            positions.append(i + offsets.get(grp, 0.0))
            box_data.append(vals)
            face_colors.append(GROUP_COLORS.get(grp, _TECH_SHADES[3]))
            edge_colors.append(_TECH_DARK_SHADES[0]
                               if grp == "healthy"
                               else "#0F2A3A")

        if box_data:
            bp = ax.boxplot(
                box_data, positions=positions, widths=width,
                patch_artist=True,
                medianprops=dict(color=_TECH_DARK_SHADES[0], linewidth=1.6),
                whiskerprops=dict(color=_TECH_DARK_SHADES[0], linewidth=1.0),
                capprops=dict(color=_TECH_DARK_SHADES[0], linewidth=1.0),
                flierprops=dict(marker="d", markersize=4.5,
                                markerfacecolor=_TECH_DARK_SHADES[0],
                                markeredgecolor=_TECH_DARK_SHADES[0]),
            )
            for patch, fc, ec in zip(bp["boxes"], face_colors, edge_colors):
                patch.set_facecolor(fc)
                patch.set_edgecolor(ec)
                patch.set_linewidth(1.0)

            # Mean marker — open circle on the box. Matches the reference.
            for pos, vals in zip(positions, box_data):
                ax.scatter([pos], [float(np.mean(vals))], s=42,
                           marker="o", facecolor="white",
                           edgecolor=_TECH_DARK_SHADES[0],
                           linewidth=1.2, zorder=5)

        ax.set_xticks(range(len(subjects)))
        ax.set_xticklabels(subjects, rotation=25, ha="right")
        for tick_label, subj in zip(ax.get_xticklabels(), subjects):
            grp = subj_group.get(subj, "?")
            if grp == "impaired":
                tick_label.set_fontweight("bold")
                tick_label.set_color(_TECH_DARK_SHADES[0])
        ax.set_xlim(-0.6, len(subjects) - 0.4)
        ax.set_ylim(0.0, 1.0)
        if c_idx == 0:
            ax.set_ylabel("Macro F1")
        ax.set_xlabel("Held-out subject")

        is_deep = str(model).lower() in DEEP_MODEL_TYPES
        title_color = DEEP_MODEL_ACCENT if is_deep else CLASSIC_MODEL_ACCENT
        ax.set_title(_pretty_model(model), color=title_color, fontweight="bold")
        ax.grid(axis="y", linestyle=":", color=NEUTRAL_GRID)
        ax.set_axisbelow(True)

    # Hide any trailing empty axes (when n_models % ncols != 0).
    for k in range(n_models, nrows * ncols):
        r_idx, c_idx = divmod(k, ncols)
        axes[r_idx][c_idx].set_visible(False)

    # Single shared legend at the bottom — matches the reference style.
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=GROUP_COLORS["healthy"],
              edgecolor=_TECH_DARK_SHADES[0], label="Healthy"),
        Patch(facecolor=GROUP_COLORS["impaired"],
              edgecolor="#0F2A3A", label="Impaired"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=2,
               frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return _save(fig, out_path)


def _pretty_model(name: str) -> str:
    return {
        "lda": "LDA",
        "svm": "SVM",
        "random_forest": "Random Forest",
        "catboost": "CatBoost",
        "mlp": "MLP",
        "cnn": "CNN",
        "attention_net": "Attention-Net",
        "mstnet": "MSTNet",
    }.get(str(name).lower(), str(name))


# ---------------------------------------------------------------------------
# Fig 6.6 — Feature ablation
# ---------------------------------------------------------------------------

def plot_feature_ablation(
    ablation_csv: Path,
    out_path: Path,
) -> List[Path]:
    _setup_mpl()
    rows: List[Dict[str, Any]] = []
    with Path(ablation_csv).open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append({
                "condition": r["condition"],
                "mean":      float(r["macro_f1_mean"]),
                "std":       float(r["macro_f1_std"]),
            })
    if not rows:
        log.warning("No rows in %s", ablation_csv)
        return []

    singles  = [r for r in rows
                if not any(t in r["condition"].lower() for t in ("combined", "all"))]
    combined = [r for r in rows
                if any(t in r["condition"].lower() for t in ("combined", "all"))]
    singles.sort(key=lambda r: r["mean"])
    ordered = singles + combined

    labels = [r["condition"] for r in ordered]
    means  = np.asarray([r["mean"] for r in ordered], dtype=float)
    stds   = np.asarray([r["std"]  for r in ordered], dtype=float)
    lower  = np.minimum(stds, means)
    upper  = np.minimum(stds, 1.0 - means)
    colors = [_TECH_DARK_SHADES[0] if r in combined else _TECH_SHADES[1]
              for r in ordered]

    fig, ax = plt.subplots(figsize=(6.6, 0.45 * len(ordered) + 1.6))
    y = np.arange(len(ordered))
    ax.barh(y, means, xerr=[lower, upper], color=colors,
            edgecolor=NEUTRAL_TEXT, linewidth=0.6, capsize=3, alpha=0.9,
            error_kw=dict(elinewidth=0.9, capthick=0.9))
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Macro F1  (mean ± SD across folds)")
    ax.set_title("Feature ablation under intra-subject LOSO-session", loc="left")
    ax.grid(axis="x", linestyle=":", color=NEUTRAL_GRID)
    ax.set_axisbelow(True)
    return _save(fig, out_path)


# ---------------------------------------------------------------------------
# Fig 7.4 — Calibration vs F1
# ---------------------------------------------------------------------------

def _annotate_correlation(ax, x: np.ndarray, y: np.ndarray,
                          *, loc=(0.04, 0.96)) -> None:
    if x.size < 3 or np.std(x) == 0 or np.std(y) == 0:
        ax.text(*loc, f"n = {x.size}", transform=ax.transAxes,
                va="top", fontsize=9, color=NEUTRAL_MUTED)
        return
    try:
        from scipy.stats import pearsonr, spearmanr
        r, _   = pearsonr(x, y)
        rs, _  = spearmanr(x, y)
        ax.text(*loc,
                f"r = {r:+.2f}   ρ = {rs:+.2f}   n = {x.size}",
                transform=ax.transAxes, va="top", fontsize=9,
                color=NEUTRAL_TEXT)
    except Exception:  # noqa: BLE001
        r = float(np.corrcoef(x, y)[0, 1])
        ax.text(*loc, f"r = {r:+.2f}  n = {x.size}",
                transform=ax.transAxes, va="top", fontsize=9,
                color=NEUTRAL_TEXT)


def plot_calibration_vs_f1(
    correlation_json: Path,
    out_path: Path,
    *,
    model: Optional[str] = None,
) -> List[Path]:
    _setup_mpl()
    with Path(correlation_json).open("r", encoding="utf-8") as f:
        data = json.load(f)
    if model is None:
        model = max(data, key=lambda k: data[k].get("n_pairs", 0))
    record = data[model]
    pairs  = record.get("joined") or []
    if not pairs:
        log.warning("No paired (calibration, F1) rows for %s", model)
        return []

    is_stab = _is_stability_metric(pairs)
    subjects = sorted({p["subject_id"] for p in pairs})
    palette  = _subject_palette(subjects)

    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    x = np.asarray([p["confidence"] for p in pairs], dtype=float)
    y = np.asarray([p["macro_f1"]   for p in pairs], dtype=float)

    x_lo = float(x.min()) - 0.02
    x_hi = float(x.max()) + 0.02
    if x_hi - x_lo < 0.05:
        c = (x_lo + x_hi) / 2
        x_lo, x_hi = c - 0.04, c + 0.04
    y_lo = max(0.0, float(y.min()) - 0.04)
    y_hi = min(1.0, float(y.max()) + 0.04)

    for subj in subjects:
        xs = [p["confidence"] for p in pairs if p["subject_id"] == subj]
        ys = [p["macro_f1"]   for p in pairs if p["subject_id"] == subj]
        ax.scatter(xs, ys, s=110, color=palette[subj],
                   edgecolor="white", linewidth=1.0,
                   label=subj, zorder=3)

    if x.size >= 2 and x.std() > 0:
        a, b = np.polyfit(x, y, 1)
        xs   = np.linspace(x_lo, x_hi, 50)
        ax.plot(xs, a * xs + b, "--", color=NEUTRAL_MUTED,
                linewidth=1.0, label="pooled fit", zorder=2)
    _annotate_correlation(ax, x, y)

    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    x_label = ("Calibration stability  (session)" if is_stab else
               "Calibration confidence  (session)")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Macro F1  (held-out session)")
    ax.set_title(f"Calibration vs. classification — {model}", loc="left")
    ax.grid(linestyle=":", color=NEUTRAL_GRID)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left")
    return _save(fig, out_path)


# ---------------------------------------------------------------------------
# Fig 7.4b — Calibration vs F1 per model
# ---------------------------------------------------------------------------

def plot_calibration_vs_f1_per_model(
    correlation_json: Path,
    out_path: Path,
    *,
    ncols: int = 4,
) -> List[Path]:
    _setup_mpl()
    with Path(correlation_json).open("r", encoding="utf-8") as f:
        data = json.load(f)
    models = [m for m, d in data.items() if d.get("joined")]
    if not models:
        log.warning("No models with paired data in %s", correlation_json)
        return []
    models.sort(key=lambda m: -abs(data[m].get("pearson_r") or 0.0))

    all_x: List[float] = []
    all_y: List[float] = []
    all_subjects: set = set()
    is_stab = False
    for m in models:
        for p in data[m]["joined"]:
            all_x.append(float(p["confidence"]))
            all_y.append(float(p["macro_f1"]))
            all_subjects.add(p["subject_id"])
        if not is_stab:
            is_stab = _is_stability_metric(data[m]["joined"])
    palette = _subject_palette(sorted(all_subjects))

    x_lo, x_hi = float(min(all_x)) - 0.01, float(max(all_x)) + 0.01
    if x_hi - x_lo < 0.05:
        c = (x_lo + x_hi) / 2
        x_lo, x_hi = c - 0.04, c + 0.04
    y_lo = max(0.0, float(min(all_y)) - 0.04)
    y_hi = min(1.0, float(max(all_y)) + 0.04)

    nrows = int(math.ceil(len(models) / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(3.2 * ncols, 2.9 * nrows),
                             sharex=True, sharey=True,
                             gridspec_kw=dict(wspace=0.18, hspace=0.45))
    axes = np.atleast_2d(axes)

    for idx, model in enumerate(models):
        r_idx, c_idx = divmod(idx, ncols)
        ax = axes[r_idx, c_idx]
        pairs = data[model]["joined"]
        for subj in sorted({p["subject_id"] for p in pairs}):
            xs = [p["confidence"] for p in pairs if p["subject_id"] == subj]
            ys = [p["macro_f1"]   for p in pairs if p["subject_id"] == subj]
            ax.scatter(xs, ys, s=46, color=palette[subj],
                       edgecolor="white", linewidth=0.6, zorder=3)
        for subj in sorted({p["subject_id"] for p in pairs}):
            xs = [p["confidence"] for p in pairs if p["subject_id"] == subj]
            ys = [p["macro_f1"]   for p in pairs if p["subject_id"] == subj]
            ax.scatter([np.mean(xs)], [np.mean(ys)],
                       s=120, marker="D", color=palette[subj],
                       edgecolor=NEUTRAL_TEXT, linewidth=1.0, zorder=4)
        r_v  = data[model].get("pearson_r")
        rs_v = data[model].get("spearman_r")
        r_s  = f"r = {r_v:+.2f}" if r_v is not None and math.isfinite(r_v) else "r = n/a"
        rs_s = f"ρ = {rs_v:+.2f}" if rs_v is not None and math.isfinite(rs_v) else ""
        ax.set_title(f"{model}\n{r_s}  {rs_s}".rstrip(), fontsize=10)
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
        ax.grid(linestyle=":", color=NEUTRAL_GRID)
        ax.set_axisbelow(True)

    for idx in range(len(models), nrows * ncols):
        r_idx, c_idx = divmod(idx, ncols)
        axes[r_idx, c_idx].set_visible(False)

    x_axis_label = "Calibration stability" if is_stab else "Calibration confidence"
    for ax in axes[-1, :]:
        if ax.get_visible():
            ax.set_xlabel(x_axis_label)
    for ax in axes[:, 0]:
        if ax.get_visible():
            ax.set_ylabel("Macro F1")

    handles, labels = [], []
    for subj in sorted(all_subjects):
        handles.append(plt.Line2D([0], [0], marker="o", linestyle="",
                                  markerfacecolor=palette[subj],
                                  markeredgecolor="white", markersize=8))
        labels.append(subj)
    handles.append(plt.Line2D([0], [0], marker="D", linestyle="",
                              markerfacecolor="white",
                              markeredgecolor=NEUTRAL_TEXT, markersize=10))
    labels.append("per-subject mean")
    fig.legend(handles, labels, ncol=min(7, len(labels)),
               loc="lower center", bbox_to_anchor=(0.5, -0.04),
               frameon=False)
    fig.suptitle("Calibration vs. macro F1 — per model",
                 fontsize=12.5, fontweight="semibold",
                 x=0.02, ha="left", y=1.0)
    return _save(fig, out_path)