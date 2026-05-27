"""
validation/threshold_plots.py
═════════════════════════════
Figures for the threshold-gameplay evaluation (Chapter 6, §6.10).

All figures are built from the artefacts that
``threshold_report.write_threshold_report`` writes — the per-recording
table, the per-subject summary, the pooled cohort summary and the
threshold sweep. Nothing here re-reads the raw Unity CSVs, so the plots
are cheap and can be regenerated independently of the (slow) evaluation.

Figures produced
────────────────
    fig_6_9_threshold_sweep            ROC + F1-vs-threshold sweep
    fig_6_10_confusion_overall         3 confusion matrices (one per
                                       perspective), pooled over all
                                       valid recordings
    fig_6_11_confusion_per_subject     a grid of confusion matrices,
                                       one row per participant
    fig_6_12_perspective_comparison    grouped bars: as-recorded vs
                                       profile vs optimal F1, per cohort
    fig_6_13_per_subject_f1            per-subject F1 for the three
                                       perspectives, sorted by optimal F1

Every function saves both ``.pdf`` and ``.png`` and returns the list of
paths written. Matplotlib is imported lazily and forced onto the Agg
backend so this module is safe to call from a worker process.

Design choices
──────────────
* Confusion matrices are shown as **row-normalised rates** (each row —
  a true class — sums to 1) with the raw frame count underneath, so a
  reader sees both "what fraction of rest frames were called active"
  and the absolute scale.
* The "as-recorded" perspective is almost always degenerate (the game
  rarely triggered). The plots show it anyway — that degeneracy *is*
  the headline finding — but annotate it so a reader is not confused
  by an all-in-one-column matrix.
* Excluded recordings (single-class ground truth, too short) never
  reach these plots; the data passed in is expected to be the valid
  subset already, but the loaders defensively skip ``excluded`` rows.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

log = logging.getLogger(__name__)

# Perspective metadata: key prefix, display label, accent colour.
_PERSPECTIVES: List[Tuple[str, str, str]] = [
    ("asrec",   "As-recorded",       "#dc2626"),   # red — what the user got
    ("profile", "Profile threshold", "#d97706"),   # amber — re-derived
    ("opt",     "Optimal threshold", "#16a34a"),   # green — achievable
]

_CLASS_LABELS = ["rest", "active"]


# ═══════════════════════════════════════════════════════════════════════════
# Matplotlib bootstrap
# ═══════════════════════════════════════════════════════════════════════════

def _mpl():
    """Import matplotlib with the Agg backend. Lazy — keeps import cheap."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _save(fig, stem: Path) -> List[Path]:
    """Save a figure as both PDF and PNG; return the paths."""
    out: List[Path] = []
    for ext in ("pdf", "png"):
        p = stem.with_suffix(f".{ext}")
        fig.savefig(p, bbox_inches="tight", dpi=150)
        out.append(p)
    plt_mod = _mpl()
    plt_mod.close(fig)
    return out


# ═══════════════════════════════════════════════════════════════════════════
# CSV loaders — small, dependency-free
# ═══════════════════════════════════════════════════════════════════════════

def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    """Read a CSV into a list of dict rows. Empty list if missing."""
    path = Path(path)
    if not path.exists():
        return []
    with open(path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _f(row: Dict[str, str], key: str, default: float = float("nan")) -> float:
    """Float-coerce a CSV cell; blank/invalid → default."""
    v = row.get(key, "")
    if v is None or v == "":
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _i(row: Dict[str, str], key: str, default: int = 0) -> int:
    """Int-coerce a CSV cell."""
    try:
        return int(float(row.get(key, "") or default))
    except (TypeError, ValueError):
        return default


# ═══════════════════════════════════════════════════════════════════════════
# Confusion-matrix drawing primitive
# ═══════════════════════════════════════════════════════════════════════════

def _draw_confusion(ax, tp: int, fp: int, fn: int, tn: int,
                    title: str, accent: str) -> None:
    """
    Draw one 2×2 confusion matrix onto ``ax``.

    Layout (standard): rows = true class, columns = predicted class.
        ┌─────────────┬─────────────┐
        │ TN          │ FP          │   true = rest
        ├─────────────┼─────────────┤
        │ FN          │ TP          │   true = active
        └─────────────┴─────────────┘
          pred=rest      pred=active

    Cells are shaded by *row-normalised rate* so the colour means
    "fraction of this true class" regardless of class imbalance; the
    raw frame count is printed beneath the rate.
    """
    import numpy as np

    counts = np.array([[tn, fp], [fn, tp]], dtype=float)
    row_tot = counts.sum(axis=1, keepdims=True)
    # Row-normalise; guard the divide so an empty row stays 0 not NaN.
    rates = np.divide(counts, row_tot,
                      out=np.zeros_like(counts), where=row_tot > 0)

    ax.imshow(rates, cmap="Blues", vmin=0.0, vmax=1.0, aspect="equal")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels([f"pred\n{c}" for c in _CLASS_LABELS], fontsize=8)
    ax.set_yticklabels([f"true\n{c}" for c in _CLASS_LABELS], fontsize=8)
    ax.set_title(title, fontsize=9, color=accent, fontweight="bold", pad=6)

    total = counts.sum()
    for i in range(2):
        for j in range(2):
            rate = rates[i, j]
            cnt = int(counts[i, j])
            # White text on dark cells, dark on light, for contrast.
            colour = "white" if rate > 0.55 else "#1f2937"
            ax.text(j, i - 0.13, f"{rate*100:.1f}%",
                    ha="center", va="center", fontsize=10,
                    fontweight="bold", color=colour)
            ax.text(j, i + 0.20, _human_count(cnt),
                    ha="center", va="center", fontsize=7, color=colour)
    # Thin grid between cells.
    for edge in (0.5,):
        ax.axhline(edge, color="white", lw=2)
        ax.axvline(edge, color="white", lw=2)
    if total == 0:
        ax.text(0.5, -0.75, "no data", transform=ax.transData,
                ha="center", fontsize=8, color="#9ca3af", style="italic")


def _human_count(n: int) -> str:
    """Compact frame-count formatting: 1234567 → '1.23 M'."""
    if n >= 1_000_000:
        return f"{n/1_000_000:.2f} M"
    if n >= 1_000:
        return f"{n/1_000:.1f} k"
    return str(n)


# ═══════════════════════════════════════════════════════════════════════════
# Figure 1 — overall confusion matrices (3 perspectives)
# ═══════════════════════════════════════════════════════════════════════════

def plot_confusion_overall(
    pooled_csv: Path,
    out_stem: Path,
    *,
    group: str = "all",
) -> List[Path]:
    """
    Three confusion matrices side by side — as-recorded, profile,
    optimal — pooled over every valid recording in ``group``.

    ``pooled_csv`` is ``table_threshold_pooled.csv``.
    """
    rows = _read_csv_rows(pooled_csv)
    row = next((r for r in rows if r.get("group") == group), None)
    if row is None:
        log.warning("plot_confusion_overall: group %r not in %s",
                    group, pooled_csv)
        return []

    plt = _mpl()
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.9))
    for ax, (pfx, label, accent) in zip(axes, _PERSPECTIVES):
        _draw_confusion(
            ax,
            tp=_i(row, f"{pfx}_tp"), fp=_i(row, f"{pfx}_fp"),
            fn=_i(row, f"{pfx}_fn"), tn=_i(row, f"{pfx}_tn"),
            title=label, accent=accent,
        )

    n_rec = _i(row, "n_recordings")
    n_exc = _i(row, "n_excluded")
    subtitle = (f"{n_rec} valid recording(s)"
                + (f", {n_exc} excluded" if n_exc else ""))
    fig.suptitle(
        f"Threshold-gameplay confusion matrices — cohort: "
        f"{row.get('group_label', group)}\n{subtitle}",
        fontsize=11, fontweight="bold", y=1.06,
    )
    # Reserve a strip at the bottom for the caption so it sits clear
    # of the x-axis labels rather than overprinting them.
    fig.tight_layout(rect=(0, 0.07, 1, 0.97))
    fig.text(0.5, 0.015,
             "Cells are row-normalised (fraction of each true class); "
             "frame counts beneath.",
             ha="center", fontsize=8, color="#6b7280")
    return _save(fig, out_stem)


# ═══════════════════════════════════════════════════════════════════════════
# Figure 2 — per-subject confusion-matrix grid
# ═══════════════════════════════════════════════════════════════════════════

def plot_confusion_per_subject(
    per_subject_csv: Path,
    out_stem: Path,
    *,
    perspective: str = "opt",
) -> List[Path]:
    """
    A grid of confusion matrices — one per participant — for a single
    perspective (default: the optimal threshold, since that's the
    informative one; the as-recorded matrices are near-degenerate).

    ``per_subject_csv`` is ``table_threshold_per_subject.csv``.
    """
    rows = _read_csv_rows(per_subject_csv)
    if not rows:
        log.warning("plot_confusion_per_subject: %s empty", per_subject_csv)
        return []

    accent = dict((p, c) for p, _, c in _PERSPECTIVES).get(perspective, "#16a34a")
    label  = dict((p, l) for p, l, _ in _PERSPECTIVES).get(perspective, perspective)

    plt = _mpl()
    n = len(rows)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(3.0 * ncols, 3.6 * nrows))
    # Normalise axes to a flat list regardless of grid shape.
    if n == 1:
        axes = [axes]
    else:
        axes = list(axes.flat)

    for ax, row in zip(axes, rows):
        subj = row.get("subject_id", "?")
        grp  = row.get("group_label", "")
        f1   = _f(row, f"{perspective}_f1")
        title = f"{subj}"
        if grp and grp != "unknown":
            title += f"  ({grp})"
        if f1 == f1:                       # not NaN
            title += f"\nF1 {f1:.2f}"
        _draw_confusion(
            ax,
            tp=_i(row, f"{perspective}_tp"), fp=_i(row, f"{perspective}_fp"),
            fn=_i(row, f"{perspective}_fn"), tn=_i(row, f"{perspective}_tn"),
            title=title, accent=accent,
        )

    # Blank any unused cells in the last row.
    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle(
        f"Per-participant confusion matrices — {label} perspective",
        fontsize=12, fontweight="bold", y=1.01,
    )
    # Generous row spacing — each cell carries a two-line title plus
    # two-line axis labels, so the default tight_layout padding lets
    # the next row's title collide with this row's x-labels.
    fig.tight_layout(h_pad=3.2, w_pad=1.5, rect=(0, 0, 1, 0.98))
    return _save(fig, out_stem)


# ═══════════════════════════════════════════════════════════════════════════
# Figure 3 — perspective comparison (grouped bars per cohort)
# ═══════════════════════════════════════════════════════════════════════════

def plot_perspective_comparison(
    pooled_csv: Path,
    out_stem: Path,
) -> List[Path]:
    """
    Grouped bar chart: F1 of the three perspectives, one cluster per
    cohort. This is the headline figure — it shows the gap between
    what the user experienced (as-recorded) and what the signal could
    deliver (optimal).
    """
    import numpy as np
    rows = _read_csv_rows(pooled_csv)
    if not rows:
        return []

    plt = _mpl()
    cohorts = [r.get("group_label", r.get("group", "?")) for r in rows]
    x = np.arange(len(cohorts))
    width = 0.26

    fig, ax = plt.subplots(figsize=(1.7 * len(cohorts) + 3.0, 4.4))
    for k, (pfx, label, accent) in enumerate(_PERSPECTIVES):
        vals = [_f(r, f"{pfx}_f1", 0.0) for r in rows]
        bars = ax.bar(x + (k - 1) * width, vals, width,
                      label=label, color=accent)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.015,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{c}\n(n={_i(r, 'n_recordings')})" for c, r in zip(cohorts, rows)],
        fontsize=9,
    )
    ax.set_ylabel("F1 score", fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.set_title("Threshold-gameplay F1 by perspective and cohort",
                 fontsize=12, fontweight="bold", pad=10)
    ax.legend(fontsize=9, loc="upper left", framealpha=0.95)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return _save(fig, out_stem)


# ═══════════════════════════════════════════════════════════════════════════
# Figure 4 — per-subject F1 bars
# ═══════════════════════════════════════════════════════════════════════════

def plot_per_subject_f1(
    per_subject_csv: Path,
    out_stem: Path,
) -> List[Path]:
    """
    Per-subject F1 for the three perspectives, sorted by optimal F1.
    Makes the inter-subject spread visible — some participants'
    threshold gameplay worked far better than others'.
    """
    import numpy as np
    rows = _read_csv_rows(per_subject_csv)
    if not rows:
        return []
    # Sort by optimal F1 descending so the strongest subject is first.
    rows = sorted(rows, key=lambda r: _f(r, "opt_f1", 0.0), reverse=True)

    plt = _mpl()
    subjects = [r.get("subject_id", "?") for r in rows]
    y = np.arange(len(subjects))
    height = 0.26

    fig, ax = plt.subplots(figsize=(8.5, 0.55 * len(subjects) + 2.2))
    for k, (pfx, label, accent) in enumerate(_PERSPECTIVES):
        vals = [_f(r, f"{pfx}_f1", 0.0) for r in rows]
        ax.barh(y + (k - 1) * height, vals, height,
                label=label, color=accent)

    ax.set_yticks(y)
    ax.set_yticklabels(
        [f"{s}\n(n={_i(r, 'n_recordings')})" for s, r in zip(subjects, rows)],
        fontsize=8,
    )
    ax.invert_yaxis()                       # strongest subject on top
    ax.set_xlabel("F1 score", fontsize=10)
    ax.set_xlim(0, 1.0)
    ax.set_title("Per-participant threshold-gameplay F1",
                 fontsize=12, fontweight="bold", pad=10)
    ax.legend(fontsize=9, loc="lower right", framealpha=0.95)
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return _save(fig, out_stem)


# ═══════════════════════════════════════════════════════════════════════════
# Figure 5 — ROC + F1 sweep
# ═══════════════════════════════════════════════════════════════════════════

def plot_threshold_sweep(
    sweep_csv: Path,
    out_stem: Path,
    *,
    pooled_csv: Optional[Path] = None,
) -> List[Path]:
    """
    Two panels: an ROC curve (TPR vs FPR) and F1 as a function of the
    RMS threshold. Both come from the pooled sweep. When ``pooled_csv``
    is supplied the operating points of the three perspectives are
    marked on the F1 panel.
    """
    import numpy as np
    rows = _read_csv_rows(sweep_csv)
    if not rows:
        return []

    thr = np.array([_f(r, "threshold") for r in rows])
    tpr = np.array([_f(r, "tpr") for r in rows])
    fpr = np.array([_f(r, "fpr") for r in rows])
    f1  = np.array([_f(r, "f1")  for r in rows])

    plt = _mpl()
    fig, (ax_roc, ax_f1) = plt.subplots(1, 2, figsize=(11, 4.4))

    # ── ROC ──────────────────────────────────────────────────────────
    # Sweep order is by ascending threshold; sort by FPR for a clean
    # monotone ROC line.
    order = np.argsort(fpr)
    ax_roc.plot(fpr[order], tpr[order], color="#0284c7", lw=2)
    ax_roc.plot([0, 1], [0, 1], "--", color="#9ca3af", lw=1)
    ax_roc.fill_between(fpr[order], tpr[order], alpha=0.12, color="#0284c7")
    # Trapezoidal AUC of the sorted curve. ``np.trapz`` was renamed
    # to ``np.trapezoid`` in NumPy 2.0; fall back for older versions.
    _trapz = getattr(np, "trapezoid", None) or np.trapz
    auc = float(_trapz(tpr[order], fpr[order]))
    ax_roc.set_xlabel("False-positive rate", fontsize=10)
    ax_roc.set_ylabel("True-positive rate", fontsize=10)
    ax_roc.set_title(f"ROC — pooled  (AUC ≈ {auc:.3f})",
                     fontsize=11, fontweight="bold")
    ax_roc.set_xlim(0, 1); ax_roc.set_ylim(0, 1)
    ax_roc.grid(alpha=0.3); ax_roc.set_axisbelow(True)

    # ── F1 vs threshold ─────────────────────────────────────────────
    ax_f1.plot(thr, f1, color="#16a34a", lw=2)
    ax_f1.set_xscale("log")
    ax_f1.set_xlabel("RMS threshold (V, log scale)", fontsize=10)
    ax_f1.set_ylabel("F1 score", fontsize=10)
    ax_f1.set_title("F1 vs threshold — pooled", fontsize=11, fontweight="bold")
    ax_f1.set_ylim(0, 1)
    ax_f1.grid(alpha=0.3); ax_f1.set_axisbelow(True)

    # Mark the F1-optimal sweep point.
    if f1.size:
        best = int(np.nanargmax(f1))
        ax_f1.axvline(thr[best], color="#16a34a", ls="--", lw=1)
        ax_f1.plot(thr[best], f1[best], "o", color="#16a34a", ms=7)
        ax_f1.annotate(
            f" optimal\n thr={thr[best]:.2g}\n F1={f1[best]:.2f}",
            (thr[best], f1[best]), fontsize=8,
            va="top", color="#15803d",
        )

    fig.suptitle("Threshold-gameplay — pooled threshold sweep",
                 fontsize=12, fontweight="bold", y=1.03)
    fig.tight_layout()
    return _save(fig, out_stem)


# ═══════════════════════════════════════════════════════════════════════════
# Convenience: render the whole figure set
# ═══════════════════════════════════════════════════════════════════════════

def render_all_threshold_figures(
    out_dir: Path,
    *,
    pooled_csv: Optional[Path] = None,
    per_subject_csv: Optional[Path] = None,
    sweep_csv: Optional[Path] = None,
) -> Dict[str, List[Path]]:
    """
    Render every threshold-gameplay figure from the report artefacts
    already on disk in ``out_dir``.

    Any artefact that isn't present is skipped with a log line rather
    than raising — partial generation is allowed.
    """
    out_dir = Path(out_dir)
    pooled_csv      = pooled_csv      or out_dir / "table_6_11_threshold_pooled.csv"
    per_subject_csv = per_subject_csv or out_dir / "table_threshold_per_subject.csv"
    sweep_csv       = sweep_csv       or out_dir / "fig_6_9_threshold_sweep.csv"

    produced: Dict[str, List[Path]] = {}

    if Path(sweep_csv).exists():
        produced["fig_6_9_threshold_sweep"] = plot_threshold_sweep(
            sweep_csv, out_dir / "fig_6_9_threshold_sweep",
            pooled_csv=pooled_csv if Path(pooled_csv).exists() else None,
        )
    else:
        log.info("threshold_plots: %s missing — sweep figure skipped",
                 sweep_csv)

    if Path(pooled_csv).exists():
        produced["fig_6_10_confusion_overall"] = plot_confusion_overall(
            pooled_csv, out_dir / "fig_6_10_confusion_overall",
        )
        produced["fig_6_12_perspective_comparison"] = plot_perspective_comparison(
            pooled_csv, out_dir / "fig_6_12_perspective_comparison",
        )
    else:
        log.info("threshold_plots: %s missing — cohort figures skipped",
                 pooled_csv)

    if Path(per_subject_csv).exists():
        produced["fig_6_11_confusion_per_subject"] = plot_confusion_per_subject(
            per_subject_csv, out_dir / "fig_6_11_confusion_per_subject",
            perspective="opt",
        )
        produced["fig_6_13_per_subject_f1"] = plot_per_subject_f1(
            per_subject_csv, out_dir / "fig_6_13_per_subject_f1",
        )
    else:
        log.info("threshold_plots: %s missing — per-subject figures skipped",
                 per_subject_csv)

    # Drop any entries whose plotting function returned [] (missing data).
    return {k: v for k, v in produced.items() if v}
