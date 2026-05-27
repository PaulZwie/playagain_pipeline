#!/usr/bin/env python3
"""
results_by_group.py
-------------------
Generate per-group (healthy vs. impaired) validation reports and plots
in the visual style of the existing thesis figures
(fig_6_4_confusion_matrices.pdf, fig_6_5_per_subject_variability.pdf).

INPUTS (default locations match the uploaded files):
  --results-csv     results.csv
  --per-class-csv   per_class_f1.csv
  --results-json    results.json
  --participants    table_6_1_participants.csv
  --out-dir         ./results_by_group_out
  --exclude         subject ids to drop, comma-separated (e.g. VP_04)

OUTPUTS (one .pdf and one .png per plot):
  by_group_overall.csv
  by_group_per_class.csv
  by_group_per_subject.csv
  01_overall_by_group_box.pdf/.png      - boxplot per group x model (style of fig_6_5)
  02_per_class_by_group_heatmap.pdf/.png- per-class F1 heatmap (style of fig_6_4)
  03_per_subject_by_group_box.pdf/.png  - per-subject boxplot grid, coloured by group
  04_overall_by_group_bars.pdf/.png     - compact bar chart per model x group
  05_group_gap.pdf/.png                 - signed H - I macro-F1 gap per model
  06_confusion_by_group.pdf/.png        - 2-row confusion grid (healthy / impaired)

USAGE:
  # Full cohort
  python results_by_group.py --out-dir out_full

  # Without VP_04 (noisy recording, per-thesis Limitation)
  python results_by_group.py --exclude VP_04 --out-dir out_no_vp04
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# FAU palette                                                                  #
# --------------------------------------------------------------------------- #
# We use the FAU "tech" palette as the primary encoding, as requested. The
# palette ramps from a deep cool grey-blue to a near-white tint, which works
# well for both (a) categorical encodings with a small number of levels
# (group H vs I) and (b) sequential encodings such as heatmaps. Model titles
# borrow from the full FAU faculty palette so each architecture has a stable
# identifying colour across plots (matches the reference figures).
try:
    from fau_colors import cmaps as _FAU_CMAPS  # type: ignore
    from fau_colors import colors as _FAU_COLORS  # type: ignore
    TECH_RAMP = list(_FAU_CMAPS.tech)              # 5 colours, dark -> light
    TECH_DARK = list(_FAU_CMAPS.tech_dark)         # 5 colours, even darker
    FAU_FACULTIES = list(_FAU_CMAPS.faculties)     # 6 saturated faculty hues
    FAU_DARK = str(_FAU_COLORS.fau)                # FAU navy
except Exception:                                    # fallback if package missing
    TECH_RAMP = ["#8C9FB1", "#B6C2CE", "#D3DAE1", "#E2E7EB", "#EBF5F7"]
    TECH_DARK = ["#2F586E", "#7C96A3", "#B0BFC8", "#CBD5DB", "#E4E9EC"]
    FAU_FACULTIES = ["#04316A", "#8C9FB1", "#FDB735",
                     "#18B4F1", "#7BB725", "#C50F3C"]
    FAU_DARK = "#04316A"

# Two-tone encoding for "healthy" vs "impaired". The mid tech-ramp shade is
# used for the healthy fills (the within-system positive control); the deep
# tech_dark[0] is used for impaired outlines so impaired reads as the
# foreground category.
GROUP_COLORS = {"healthy": TECH_RAMP[1], "impaired": TECH_DARK[0]}
GROUP_FILL   = {"healthy": TECH_RAMP[2], "impaired": TECH_DARK[1]}

# Sequential heatmap colormap built from the tech ramp (light -> dark, so
# high values appear dark - matching the reference confusion-matrix figure).
TECH_CMAP = LinearSegmentedColormap.from_list(
    "fau_tech_seq",
    [TECH_RAMP[4], TECH_RAMP[3], TECH_RAMP[2], TECH_RAMP[1], TECH_DARK[0]],
    N=256,
)

# --------------------------------------------------------------------------- #
# Model / class ordering                                                       #
# --------------------------------------------------------------------------- #
MODEL_ORDER = [
    "lda", "svm", "random_forest", "catboost",
    "mlp", "cnn", "attention_net", "mstnet",
]
MODEL_LABEL = {
    "lda":           "LDA",
    "svm":           "SVM",
    "random_forest": "Random Forest",
    "catboost":      "CatBoost",
    "mlp":           "MLP",
    "attention_net": "Attention-CNN",
    "cnn":           "CNN",
    "mstnet":        "MSTNet",
}
# Model-title colours used in the reference figures (faculty-palette mix).
MODEL_TITLE_COLOR = {
    "lda":           "#1f1f1f",
    "svm":           "#1f1f1f",
    "random_forest": FAU_FACULTIES[0],   # FAU navy
    "catboost":      FAU_FACULTIES[0],   # FAU navy
    "mlp":           FAU_FACULTIES[2],   # FAU orange
    "cnn":           FAU_FACULTIES[4],   # FAU green
    "attention_net": FAU_FACULTIES[3],   # FAU light blue
    "mstnet":        FAU_FACULTIES[5],   # FAU red
}

CLASS_ORDER = ["Rest", "Fist", "Pinch", "Tripod"]

plt.rcParams.update({
    "font.family":      "serif",
    "font.size":         10,
    "axes.titlesize":    11,
    "axes.titleweight":  "bold",
    "axes.labelsize":    10,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "legend.fontsize":    9,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.25,
    "grid.linestyle":     "-",
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "pdf.fonttype":       42,
    "ps.fonttype":        42,
})


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-csv",    type=Path, default=Path("results.csv"))
    p.add_argument("--per-class-csv",  type=Path, default=Path("per_class_f1.csv"))
    p.add_argument("--results-json",   type=Path, default=Path("results.json"))
    p.add_argument("--participants",   type=Path,
                   default=Path("table_6_1_participants.csv"))
    p.add_argument("--out-dir",        type=Path,
                   default=Path("results_by_group_out"))
    p.add_argument("--exclude",        type=str, default="",
                   help="comma-separated subject IDs to drop "
                        "(e.g. 'VP_04' for the noisy recording)")
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Loading & joins                                                              #
# --------------------------------------------------------------------------- #
_SUBJECT_RE = re.compile(r"__([A-Za-z0-9_]+?)__")


def extract_subject(fold_id: str) -> str | None:
    m = _SUBJECT_RE.search(fold_id)
    return m.group(1) if m else None


def load_participants(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["subject_id"] != "TOTAL"].copy()
    df["group_label"] = df["group"].map({"H": "healthy", "I": "impaired"})
    return df[["subject_id", "group", "group_label",
               "total_sessions", "total_windows"]]


def attach_group(df: pd.DataFrame, participants: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["subject"] = df["fold_id"].astype(str).map(extract_subject)
    df = df.merge(
        participants[["subject_id", "group_label"]],
        left_on="subject", right_on="subject_id", how="left",
    )
    if df["group_label"].isna().any():
        missing = df.loc[df["group_label"].isna(), "subject"].unique()
        print(f"[warn] subjects without a group label: {sorted(missing)}",
              file=sys.stderr)
        df["group_label"] = df["group_label"].fillna("unknown")
    return df.drop(columns="subject_id")


# --------------------------------------------------------------------------- #
# Aggregations                                                                 #
# --------------------------------------------------------------------------- #
def overall_by_group(results: pd.DataFrame) -> pd.DataFrame:
    g = (results.groupby(["model_type", "group_label"], as_index=False)
                .agg(n_folds=("fold_id", "count"),
                     accuracy_mean=("accuracy", "mean"),
                     accuracy_std=("accuracy", "std"),
                     macro_f1_mean=("macro_f1", "mean"),
                     macro_f1_std=("macro_f1", "std")))
    g["model_type"] = pd.Categorical(g["model_type"], MODEL_ORDER, ordered=True)
    return g.sort_values(["group_label", "macro_f1_mean"],
                         ascending=[True, False]).reset_index(drop=True)


def per_class_by_group(per_class: pd.DataFrame) -> pd.DataFrame:
    g = (per_class.groupby(["model_type", "group_label", "class"], as_index=False)
                  .agg(n_folds=("f1", "count"),
                       f1_mean=("f1", "mean"),
                       f1_std=("f1", "std")))
    g["model_type"] = pd.Categorical(g["model_type"], MODEL_ORDER, ordered=True)
    g["class"] = pd.Categorical(g["class"], CLASS_ORDER, ordered=True)
    return g.sort_values(["group_label", "model_type", "class"]).reset_index(drop=True)


def per_subject(results: pd.DataFrame) -> pd.DataFrame:
    g = (results.groupby(["model_type", "subject", "group_label"], as_index=False)
                .agg(n_folds=("fold_id", "count"),
                     macro_f1_mean=("macro_f1", "mean"),
                     macro_f1_std=("macro_f1", "std"),
                     accuracy_mean=("accuracy", "mean")))
    g["model_type"] = pd.Categorical(g["model_type"], MODEL_ORDER, ordered=True)
    return g.sort_values(["model_type", "group_label", "subject"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Plot helpers                                                                 #
# --------------------------------------------------------------------------- #
def _save(fig: plt.Figure, out: Path, stem: str) -> None:
    fig.savefig(out / f"{stem}.pdf")
    fig.savefig(out / f"{stem}.png")
    plt.close(fig)


def _style_box(bp: dict, face: str, edge: str) -> None:
    for patch in bp["boxes"]:
        patch.set_facecolor(face)
        patch.set_edgecolor(edge)
        patch.set_linewidth(1.0)
    for med in bp["medians"]:
        med.set_color(edge)
        med.set_linewidth(1.6)
    for whisk in bp["whiskers"]:
        whisk.set_color(edge)
        whisk.set_linewidth(1.0)
    for cap in bp["caps"]:
        cap.set_color(edge)
        cap.set_linewidth(1.0)
    for flier in bp["fliers"]:
        flier.set_marker("D")
        flier.set_markersize(4)
        flier.set_markerfacecolor("#7c96a3")
        flier.set_markeredgecolor("#3a4a55")
        flier.set_alpha(0.9)


# --------------------------------------------------------------------------- #
# Plots                                                                        #
# --------------------------------------------------------------------------- #
def plot_overall_by_group_box(results: pd.DataFrame, out: Path) -> None:
    """Boxplot of fold-level macro-F1 per model, split healthy vs impaired."""
    fig, ax = plt.subplots(figsize=(11, 4.4))
    width = 0.36
    x = np.arange(len(MODEL_ORDER))
    for offset, grp in zip([-width / 2, +width / 2], ["healthy", "impaired"]):
        data = [results[(results["model_type"] == m) &
                        (results["group_label"] == grp)]["macro_f1"].values
                for m in MODEL_ORDER]
        data = [d if len(d) else np.array([np.nan]) for d in data]
        bp = ax.boxplot(data, positions=x + offset, widths=width * 0.95,
                        patch_artist=True, showfliers=True,
                        manage_ticks=False)
        _style_box(bp, face=GROUP_FILL[grp], edge=GROUP_COLORS[grp])
        means = [np.nanmean(d) for d in data]
        ax.scatter(x + offset, means, marker="o", s=36,
                   facecolor="white",
                   edgecolor=GROUP_COLORS[grp],
                   linewidth=1.2,
                   zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABEL[m] for m in MODEL_ORDER],
                       rotation=15, ha="right")
    for tick, m in zip(ax.get_xticklabels(), MODEL_ORDER):
        tick.set_color(MODEL_TITLE_COLOR[m])
        tick.set_fontweight("bold")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Macro F1")
    ax.axhline(0.25, color="grey", linestyle=":", linewidth=0.8)
    ax.text(len(MODEL_ORDER) - 0.5, 0.255, "4-class chance",
            fontsize=8, color="grey", ha="right", va="bottom")
    handles = [
        plt.Rectangle((0, 0), 1, 1,
                      facecolor=GROUP_FILL["healthy"],
                      edgecolor=GROUP_COLORS["healthy"],
                      label="Healthy"),
        plt.Rectangle((0, 0), 1, 1,
                      facecolor=GROUP_FILL["impaired"],
                      edgecolor=GROUP_COLORS["impaired"],
                      label="Impaired"),
    ]
    ax.legend(handles=handles, loc="upper right", frameon=False)
    fig.tight_layout()
    _save(fig, out, "01_overall_by_group_box")


def plot_per_class_by_group_heatmap(g: pd.DataFrame, out: Path) -> None:
    groups = [grp for grp in ["healthy", "impaired"]
              if grp in g["group_label"].unique()]
    if not groups:
        return
    fig, axes = plt.subplots(1, len(groups),
                             figsize=(5.6 * len(groups), 4.2),
                             sharey=True, constrained_layout=True)
    if len(groups) == 1:
        axes = [axes]
    im = None
    for ax, grp in zip(axes, groups):
        sub = g[g["group_label"] == grp]
        piv = (sub.pivot(index="class", columns="model_type", values="f1_mean")
                  .reindex(index=CLASS_ORDER, columns=MODEL_ORDER))
        im = ax.imshow(piv.values, aspect="auto", vmin=0, vmax=1,
                       cmap=TECH_CMAP)
        ax.set_title(f"{grp.capitalize()}", fontweight="bold",
                     color=FAU_DARK)
        ax.set_xticks(range(len(MODEL_ORDER)))
        ax.set_xticklabels([MODEL_LABEL[m] for m in MODEL_ORDER],
                           rotation=30, ha="right")
        for tick, m in zip(ax.get_xticklabels(), MODEL_ORDER):
            tick.set_color(MODEL_TITLE_COLOR[m])
            tick.set_fontweight("bold")
        ax.set_yticks(range(len(CLASS_ORDER)))
        ax.set_yticklabels(CLASS_ORDER)
        ax.grid(False)
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                v = piv.iat[i, j]
                if pd.notna(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            fontsize=8.5,
                            color="white" if v > 0.55 else "#1f1f1f")
    if im is not None:
        cbar = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02)
        cbar.set_label("Per-class F1")
    _save(fig, out, "02_per_class_by_group_heatmap")


def plot_per_subject_by_group_box(results: pd.DataFrame, out: Path) -> None:
    """4-panel boxplot grid showing fold-level macro-F1 by held-out subject,
    one panel per representative model (style of fig_6_5)."""
    focus = ["catboost", "random_forest", "svm", "mstnet"]
    focus = [m for m in focus if m in results["model_type"].unique()]
    if not focus:
        return

    subjects = sorted(results["subject"].dropna().unique())

    fig, axes = plt.subplots(1, len(focus),
                             figsize=(3.6 * len(focus), 4.2),
                             sharey=True)
    if len(focus) == 1:
        axes = [axes]

    for ax, m in zip(axes, focus):
        sub = results[results["model_type"] == m]
        positions, data, face_colors, edge_colors, labels, means = (
            [], [], [], [], [], [])
        for i, s in enumerate(subjects):
            d = sub[sub["subject"] == s]["macro_f1"].values
            if len(d) == 0:
                continue
            grp = sub[sub["subject"] == s]["group_label"].iloc[0]
            positions.append(i)
            data.append(d)
            face_colors.append(GROUP_FILL[grp])
            edge_colors.append(GROUP_COLORS[grp])
            labels.append(s)
            means.append(float(np.nanmean(d)))

        bp = ax.boxplot(data, positions=positions,
                        widths=0.62, patch_artist=True,
                        showfliers=True, manage_ticks=False)
        for patch, fc, ec in zip(bp["boxes"], face_colors, edge_colors):
            patch.set_facecolor(fc)
            patch.set_edgecolor(ec)
            patch.set_linewidth(1.0)
        for med in bp["medians"]:
            med.set_color(FAU_DARK)
            med.set_linewidth(1.6)
        for whisk, cap in zip(bp["whiskers"], bp["caps"]):
            whisk.set_color("#3a4a55")
            whisk.set_linewidth(1.0)
            cap.set_color("#3a4a55")
            cap.set_linewidth(1.0)
        for flier in bp["fliers"]:
            flier.set_marker("D")
            flier.set_markersize(4)
            flier.set_markerfacecolor("#7c96a3")
            flier.set_markeredgecolor("#3a4a55")
            flier.set_alpha(0.9)

        ax.scatter(positions, means, marker="o", s=44,
                   facecolor="white", edgecolor=FAU_DARK,
                   linewidth=1.4, zorder=4)

        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        for tick, lbl in zip(ax.get_xticklabels(), labels):
            grp = sub[sub["subject"] == lbl]["group_label"].iloc[0]
            tick.set_color(GROUP_COLORS[grp])
            tick.set_fontweight("bold")
        ax.set_title(MODEL_LABEL[m], color=MODEL_TITLE_COLOR[m],
                     fontweight="bold")
        ax.set_xlabel("Held-out subject")
        ax.set_ylim(0, 1.0)
        if ax is axes[0]:
            ax.set_ylabel("Macro F1")

    handles = [
        plt.Rectangle((0, 0), 1, 1,
                      facecolor=GROUP_FILL["healthy"],
                      edgecolor=GROUP_COLORS["healthy"],
                      label="Healthy"),
        plt.Rectangle((0, 0), 1, 1,
                      facecolor=GROUP_FILL["impaired"],
                      edgecolor=GROUP_COLORS["impaired"],
                      label="Impaired"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2,
               frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout()
    _save(fig, out, "03_per_subject_by_group_box")


def plot_overall_by_group_bars(g: pd.DataFrame, out: Path) -> None:
    piv = (g.pivot(index="model_type", columns="group_label",
                   values="macro_f1_mean").reindex(MODEL_ORDER))
    err = (g.pivot(index="model_type", columns="group_label",
                   values="macro_f1_std").reindex(MODEL_ORDER))
    x = np.arange(len(MODEL_ORDER))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    if "healthy" in piv.columns:
        ax.bar(x - width / 2, piv["healthy"], width,
               yerr=err["healthy"], capsize=2,
               color=GROUP_FILL["healthy"],
               edgecolor=GROUP_COLORS["healthy"],
               linewidth=1.0, label="Healthy")
    if "impaired" in piv.columns:
        ax.bar(x + width / 2, piv["impaired"], width,
               yerr=err["impaired"], capsize=2,
               color=GROUP_FILL["impaired"],
               edgecolor=GROUP_COLORS["impaired"],
               linewidth=1.0, label="Impaired")
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABEL[m] for m in MODEL_ORDER],
                       rotation=15, ha="right")
    for tick, m in zip(ax.get_xticklabels(), MODEL_ORDER):
        tick.set_color(MODEL_TITLE_COLOR[m])
        tick.set_fontweight("bold")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Macro F1 (mean ± SD across folds)")
    ax.axhline(0.25, color="grey", linestyle=":", linewidth=0.8,
               label="4-class chance")
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    _save(fig, out, "04_overall_by_group_bars")


def plot_group_gap(g: pd.DataFrame, out: Path) -> None:
    piv = (g.pivot(index="model_type", columns="group_label",
                   values="macro_f1_mean").reindex(MODEL_ORDER))
    if "healthy" not in piv.columns or "impaired" not in piv.columns:
        return
    gap = piv["healthy"] - piv["impaired"]
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    bars = ax.bar(range(len(MODEL_ORDER)), gap,
                  color=GROUP_FILL["impaired"],
                  edgecolor=GROUP_COLORS["impaired"],
                  linewidth=1.0)
    ax.set_xticks(range(len(MODEL_ORDER)))
    ax.set_xticklabels([MODEL_LABEL[m] for m in MODEL_ORDER],
                       rotation=15, ha="right")
    for tick, m in zip(ax.get_xticklabels(), MODEL_ORDER):
        tick.set_color(MODEL_TITLE_COLOR[m])
        tick.set_fontweight("bold")
    ax.set_ylabel("Macro F1 gap  (Healthy − Impaired)")
    ax.axhline(0, color="#1f1f1f", linewidth=0.6)
    for b, v in zip(bars, gap):
        ax.text(b.get_x() + b.get_width() / 2,
                v + 0.005,
                f"{v:+.2f}", ha="center", fontsize=8, color=FAU_DARK)
    fig.tight_layout()
    _save(fig, out, "05_group_gap")


def plot_confusion_by_group(results_json: dict[str, Any],
                            participants: pd.DataFrame,
                            keep_subjects: set[str],
                            out: Path) -> None:
    grp = dict(zip(participants["subject_id"], participants["group_label"]))
    K = len(CLASS_ORDER)
    by_grp: dict[tuple[str, str], np.ndarray] = {}
    label_names_map: dict[str, str] = {}

    for fold in results_json.get("folds", []):
        cm = fold.get("confusion")
        if cm is None:
            continue
        cm = np.asarray(cm, dtype=float)
        cls_idx = fold.get("confusion_labels") or list(range(cm.shape[0]))
        for k, v in (fold.get("label_names") or {}).items():
            label_names_map[str(k)] = str(v)

        subj_list = fold.get("test_subjects") or []
        subject = subj_list[0] if subj_list else extract_subject(
            str(fold.get("fold_id", "")))
        if subject not in keep_subjects:
            continue
        g_label = grp.get(subject, "unknown")
        model = fold.get("model_type")
        if model not in MODEL_ORDER:
            continue

        padded = np.zeros((K, K), dtype=float)
        for i_local, i_global in enumerate(cls_idx):
            for j_local, j_global in enumerate(cls_idx):
                if 0 <= i_global < K and 0 <= j_global < K:
                    padded[i_global, j_global] = cm[i_local, j_local]

        key = (model, g_label)
        if key not in by_grp:
            by_grp[key] = np.zeros((K, K), dtype=float)
        by_grp[key] += padded

    if not by_grp:
        return

    labels = ([label_names_map.get(str(i), CLASS_ORDER[i]) for i in range(K)]
              if label_names_map else CLASS_ORDER)
    groups = [g for g in ["healthy", "impaired"] if any(k[1] == g for k in by_grp)]

    fig, axes = plt.subplots(len(groups), len(MODEL_ORDER),
                             figsize=(2.4 * len(MODEL_ORDER),
                                      2.6 * len(groups)),
                             constrained_layout=True)
    if len(groups) == 1:
        axes = np.array([axes])
    im = None
    for i, grp_label in enumerate(groups):
        for j, m in enumerate(MODEL_ORDER):
            ax = axes[i, j]
            ax.grid(False)
            cm = by_grp.get((m, grp_label))
            if cm is None or cm.sum() == 0:
                ax.set_axis_off()
                continue
            row_sums = cm.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            cmn = cm / row_sums
            im = ax.imshow(cmn, vmin=0, vmax=1, cmap=TECH_CMAP, aspect="auto")
            if i == 0:
                ax.set_title(MODEL_LABEL[m], color=MODEL_TITLE_COLOR[m],
                             fontweight="bold", fontsize=10)
            ax.set_xticks(range(K))
            ax.set_yticks(range(K))
            ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7.5)
            ax.set_yticklabels(labels, fontsize=7.5)
            if j == 0:
                ax.set_ylabel(f"{grp_label.capitalize()}\nTrue class",
                              fontweight="bold", color=FAU_DARK)
            if i == len(groups) - 1:
                ax.set_xlabel("Predicted class")
            for ii in range(K):
                for jj in range(K):
                    v = cmn[ii, jj]
                    ax.text(jj, ii, f"{v:.2f}",
                            ha="center", va="center",
                            fontsize=6.8,
                            color="white" if v > 0.55 else "#1f1f1f")
    if im is not None:
        cbar = fig.colorbar(im, ax=axes, shrink=0.78, pad=0.02)
        cbar.set_label("Per-class recall")
    _save(fig, out, "06_confusion_by_group")


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #
def main() -> int:
    args = parse_args()
    out: Path = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    excludes = {s.strip() for s in args.exclude.split(",") if s.strip()}
    if excludes:
        print(f"[info] excluding subjects: {sorted(excludes)}")

    for required in [args.results_csv, args.per_class_csv, args.participants]:
        if not required.exists():
            print(f"missing {required}", file=sys.stderr)
            return 1

    results   = pd.read_csv(args.results_csv)
    per_class = pd.read_csv(args.per_class_csv)
    participants = load_participants(args.participants)
    results_json: dict[str, Any] | None = None
    if args.results_json.exists():
        results_json = json.loads(args.results_json.read_text(encoding="utf-8"))

    results   = attach_group(results, participants)
    per_class = attach_group(per_class, participants)

    if excludes:
        n0r, n0c = len(results), len(per_class)
        results   = results[~results["subject"].isin(excludes)].copy()
        per_class = per_class[~per_class["subject"].isin(excludes)].copy()
        print(f"[info] dropped {n0r - len(results)} fold rows "
              f"and {n0c - len(per_class)} per-class rows")

    keep_subjects = set(results["subject"].dropna().unique())

    overall        = overall_by_group(results)
    class_by_group = per_class_by_group(per_class)
    subj_summary   = per_subject(results)

    overall.to_csv(out / "by_group_overall.csv", index=False)
    class_by_group.to_csv(out / "by_group_per_class.csv", index=False)
    subj_summary.to_csv(out / "by_group_per_subject.csv", index=False)

    plot_overall_by_group_box(results, out)
    plot_per_class_by_group_heatmap(class_by_group, out)
    plot_per_subject_by_group_box(results, out)
    plot_overall_by_group_bars(overall, out)
    plot_group_gap(overall, out)
    if results_json is not None:
        plot_confusion_by_group(results_json, participants, keep_subjects, out)

    print(f"\nOutputs in: {out}\n")
    print("=== Macro-F1 by model and group (mean ± SD) ===")
    summary = (overall.assign(
                  f1=lambda d: d["macro_f1_mean"].map("{:.3f}".format)
                              + " ± "
                              + d["macro_f1_std"].fillna(0).map("{:.3f}".format))
                      .pivot(index="model_type", columns="group_label", values="f1")
                      .reindex(MODEL_ORDER))
    print(summary.to_string())
    print()
    print("=== Per-class F1 by group (mean over folds) ===")
    print(class_by_group.pivot_table(index=["model_type", "class"],
                                     columns="group_label",
                                     values="f1_mean")
                          .reindex(MODEL_ORDER, level=0)
                          .round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
