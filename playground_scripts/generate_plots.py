"""
generate_plots.py
Generates all figures for Chapter 6 (Results) of the master thesis.
Output: PDF figures in ./figures/
Style: academic, matching the example thesis (Computer Modern aesthetics,
       minimal ink, teal/slate/grey palette).
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
import seaborn as sns

try:
    from fau_colors import colors as _FAU, colors_dark as _FAU_D
    TEAL    = _FAU.tech
    SLATE   = _FAU_D.tech
    WARM    = _FAU.phil
    GREEN   = _FAU.nat
    PURPLE  = _FAU_D.nat
    GOLD    = _FAU.med
    PINK    = _FAU.wiso
    NAVY    = _FAU.fau
except ImportError:
    TEAL    = "#2a7b8c"
    SLATE   = "#4a5568"
    WARM    = "#c65d3a"
    GREEN   = "#3a7c52"
    PURPLE  = "#6b4c8a"
    GOLD    = "#b08030"
    PINK    = "#b03060"
    NAVY    = "#1a3a5c"

warnings.filterwarnings("ignore")

# ── output directory ─────────────────────────────────────────────────────────
os.makedirs("figures", exist_ok=True)

# ── global style ─────────────────────────────────────────────────────────────
mpl.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["DejaVu Serif", "Georgia", "Times New Roman"],
    "font.size":          10,
    "axes.titlesize":     10,
    "axes.labelsize":     10,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "legend.fontsize":    8.5,
    "legend.framealpha":  0.9,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "axes.grid.axis":     "x",
    "grid.color":         "#e0e0e0",
    "grid.linewidth":     0.6,
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.05,
})

# ── palette ───────────────────────────────────────────────────────────────────
MODEL_ORDER = ["lda", "svm", "random_forest", "catboost",
               "mlp", "attention_net", "cnn", "mstnet"]
MODEL_LABEL = {
    "lda":          "LDA",
    "svm":          "SVM (RBF)",
    "random_forest":"Random Forest",
    "catboost":     "CatBoost",
    "mlp":          "MLP",
    "attention_net":"Attention-CNN",
    "cnn":          "CNN",
    "mstnet":       "MSTNet",
}
MODEL_COLORS = {
    "lda":          "#718096",
    "svm":          "#4a5568",
    "random_forest":TEAL,
    "catboost":     NAVY,
    "mlp":          GREEN,
    "attention_net":PURPLE,
    "cnn":          WARM,
    "mstnet":       GOLD,
}
CLASS_ORDER  = ["Rest", "Fist", "Pinch", "Tripod"]
CLASS_COLORS = ["#2f586e", "#4e7084", "#6d879b", "#8c9fb1"]


# ═══════════════════════════════════════════════════════════════════════════════
# Load data
# ═══════════════════════════════════════════════════════════════════════════════
df_results   = pd.read_csv("results/results.csv")
df_pcf1      = pd.read_csv("results/per_class_f1.csv")
df_loso_s    = pd.read_csv("results/table_6_3_loso_session.csv")
df_loso_sg   = pd.read_csv("results/table_6_3b_loso_session_by_group.csv")
df_loso_sub  = pd.read_csv("results/table_6_4_loso_subject.csv")
df_loso_subg = pd.read_csv("results/table_6_4b_loso_subject_by_group.csv")
df_ablation  = pd.read_csv("results/table_6_5_feature_ablation.csv")
df_cross     = pd.read_csv("results/table_6_6_cross_domain.csv")
df_latency   = pd.read_csv("results/table_6_7_latency.csv")
df_thresh_ps = pd.read_csv("results/table_threshold_per_subject.csv")
df_thresh_p  = pd.read_csv("results/table_6_11_threshold_pooled.csv")

# extract subject from fold_id
def extract_subject(fold_id):
    parts = fold_id.split("__")
    return parts[1] if len(parts) > 1 else "unknown"

df_results["subject"] = df_results["fold_id"].apply(extract_subject)

# filter out VP_04 and KinderUni globally for relevant subject-level data (for image 5 and onwards)
df_results = df_results[~df_results["subject"].isin(["VP_04", "KinderUni"])]
df_thresh_ps = df_thresh_ps[~df_thresh_ps["subject_id"].isin(["VP_04", "KinderUni"])]


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 1 – LOSO-Session headline performance (horizontal bar, sorted by F1)
# ═══════════════════════════════════════════════════════════════════════════════
def fig_loso_session_headline():
    df = df_loso_s.copy()
    df["model_label"] = df["model"].map(MODEL_LABEL)
    df = df.sort_values("macro_f1_mean")   # ascending so best is top

    fig, ax = plt.subplots(figsize=(6.5, 3.6))

    y = np.arange(len(df))
    colors = [MODEL_COLORS.get(m, SLATE) for m in df["model"]]

    bars = ax.barh(y, df["macro_f1_mean"], xerr=df["macro_f1_std"],
                   color=colors, alpha=0.88, height=0.55,
                   error_kw=dict(ecolor="#555", lw=1.1, capsize=3, capthick=1))

    # accuracy dots
    ax.scatter(df["accuracy_mean"], y, color="white", edgecolors="#333",
               s=28, zorder=5, linewidths=0.8, label="Accuracy (mean)")

    ax.set_yticks(y)
    ax.set_yticklabels(df["model_label"])
    ax.set_xlabel("Score")
    ax.set_title("Leave-One-Session-Out Cross-Validation", pad=6)
    ax.axvline(0, color="#999", lw=0.5)
    ax.set_xlim(0, 1.02)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.tick_params(axis="y", length=0)

    # value labels - moved to not overlap with accuracy dot
    for i, (bar, val) in enumerate(zip(bars, df["macro_f1_mean"])):
        ext = max(val + df["macro_f1_std"].iloc[i], df["accuracy_mean"].iloc[i])
        ax.text(ext + 0.015, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", ha="left", fontsize=8.5, color="#333")

    legend_elems = [
        mpatches.Patch(color="#999", label="Macro F1 (mean ± SD)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="white",
               markeredgecolor="#333", markersize=6, label="Accuracy (mean)"),
    ]
    ax.legend(handles=legend_elems, loc="lower right", framealpha=0.85)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#e0e0e0", linewidth=0.6)
    fig.tight_layout()
    fig.savefig("figures/fig_6_2_loso_session.pdf")
    plt.close()
    print("✓ fig_6_2_loso_session.pdf")

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 2 – Per-class F1 heatmap (model × class)
# ═══════════════════════════════════════════════════════════════════════════════
def fig_per_class_f1():
    # Heatmap needs aggregate dataframe
    agg = (df_pcf1.groupby(["model_type", "class"])["f1"]
           .mean().reset_index()
           .rename(columns={"f1": "mean_f1"}))

    pivot = agg.pivot(index="model_type", columns="class", values="mean_f1")
    pivot = pivot.reindex(index=MODEL_ORDER, columns=CLASS_ORDER)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5),
                             gridspec_kw={"width_ratios": [2.5, 1], "wspace": 0.3})

    # ── left: boxplot ─────────────────────────────────────────────────────────
    ax = axes[0]
    df_box = df_pcf1[df_pcf1["model_type"].isin(MODEL_ORDER)].copy()
    
    sns.boxplot(
        data=df_box, 
        x="model_type", y="f1", hue="class",
        order=MODEL_ORDER, hue_order=CLASS_ORDER,
        palette=CLASS_COLORS, ax=ax, width=0.7,
        linewidth=1, fliersize=2,
        boxprops=dict(alpha=0.9)
    )

    ax.set_xticks(range(len(MODEL_ORDER)))
    ax.set_xticklabels([MODEL_LABEL[m] for m in MODEL_ORDER],
                       rotation=35, ha="right", fontsize=8.5)
    ax.set_ylabel("F1 Score")
    ax.set_ylim(0, 1.05)
    ax.set_title("Per-Class F1 Distribution by Classifier")
    ax.legend(title="Gesture", loc="lower right", fontsize=8, title_fontsize=8, ncol=2)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#e0e0e0")

    # ── right: heatmap ────────────────────────────────────────────────────────
    ax2 = axes[1]
    # Use a solid custom fau palette map starting from white to fau dark
    cmap = sns.light_palette(CLASS_COLORS[0], as_cmap=True)
    sns.heatmap(pivot.astype(float), annot=True, fmt=".2f",
                cmap=cmap, vmin=0, vmax=1,
                linewidths=0.4, linecolor="#ccc",
                xticklabels=CLASS_ORDER,
                yticklabels=[MODEL_LABEL[m] for m in MODEL_ORDER],
                cbar_kws={"shrink": 0.8, "label": "F1"},
                ax=ax2, annot_kws={"size": 8})
    ax2.set_ylabel("")
    ax2.set_xlabel("")
    ax2.set_title("Per-Class F1 Heatmap")
    ax2.tick_params(axis="y", labelsize=8.5)
    ax2.tick_params(axis="x", labelsize=8.5)

    fig.tight_layout(pad=1.5)
    fig.savefig("figures/fig_6_3_per_class_f1.pdf")
    plt.close()
    print("✓ fig_6_3_per_class_f1.pdf")

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 2B – Selected Models Per-Class
# ═══════════════════════════════════════════════════════════════════════════════
def fig_per_class_f1_selected():
    sel_models = ["catboost", "random_forest", "mstnet"]
    df = df_pcf1[df_pcf1["model_type"].isin(sel_models)].copy()
    
    # 1x3 Subplots figure
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.8), sharey=True)

    for i, model in enumerate(sel_models):
        ax = axes[i]
        sns.boxplot(
            data=df[df["model_type"] == model],
            x="class", y="f1",
            order=CLASS_ORDER,
            palette=CLASS_COLORS,
            ax=ax,
            linewidth=1.2,
            fliersize=3,
            boxprops=dict(alpha=0.9)
        )
        ax.set_title(MODEL_LABEL[model])
        ax.set_xlabel("")
        if i == 0:
            ax.set_ylabel("F1 Score")
        else:
            ax.set_ylabel("")
            
        ax.grid(axis='y', color="#e0e0e0")
        ax.set_axisbelow(True)

    fig.suptitle("Performance of Selected Models for 4 Gestures")
    fig.tight_layout(pad=1.5)
    fig.savefig("figures/fig_6_3c_per_class_f1_selected.pdf")
    plt.close()
    print("✓ fig_6_3c_per_class_f1_selected.pdf")

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 3 – LOSO-Session by participant group
# ═══════════════════════════════════════════════════════════════════════════════
def fig_loso_by_group():
    df = df_loso_sg.copy()
    df["model_label"] = df["model"].map(MODEL_LABEL)

    healthy  = df[df["group"] == "healthy"].set_index("model")
    impaired = df[df["group"] == "impaired"].set_index("model")

    order = MODEL_ORDER[::-1]    # best model on top
    y = np.arange(len(order))
    width = 0.38

    fig, ax = plt.subplots(figsize=(6.5, 3.8))

    h_f1 = [healthy.loc[m, "macro_f1_mean"] if m in healthy.index else 0
             for m in order]
    h_sd = [healthy.loc[m, "macro_f1_std"] if m in healthy.index else 0
             for m in order]
    i_f1 = [impaired.loc[m, "macro_f1_mean"] if m in impaired.index else 0
             for m in order]
    i_sd = [impaired.loc[m, "macro_f1_std"] if m in impaired.index else 0
             for m in order]

    ax.barh(y + width / 2, h_f1, xerr=h_sd, height=width, color=TEAL,
            alpha=0.85, label="Healthy (n=4)",
            error_kw=dict(ecolor="#333", lw=0.9, capsize=2.5, capthick=0.9))
    ax.barh(y - width / 2, i_f1, xerr=i_sd, height=width, color=WARM,
            alpha=0.85, label="Impaired (n=3)",
            error_kw=dict(ecolor="#333", lw=0.9, capsize=2.5, capthick=0.9))

    ax.set_yticks(y)
    ax.set_yticklabels([MODEL_LABEL[m] for m in order])
    ax.set_xlabel("Macro F1 Score (mean ± SD)")
    ax.set_title("LOSO-Session Performance by Participant Group")
    ax.set_xlim(0, 1.05)
    ax.legend(loc="lower right")
    ax.tick_params(axis="y", length=0)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#e0e0e0", linewidth=0.6)
    fig.tight_layout()
    fig.savefig("figures/fig_6_3b_loso_by_group.pdf")
    plt.close()
    print("✓ fig_6_3b_loso_by_group.pdf")

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 4 – Per-session variability box plot (CatBoost, best classical model)
# ═══════════════════════════════════════════════════════════════════════════════
def fig_per_session_variability():
    best_model = "catboost"
    df = df_results[df_results["model_type"] == best_model].copy()

    # only subjects with ≥ 3 sessions (meaningful box)
    counts = df.groupby("subject").size()
    keep = counts[counts >= 3].index
    df = df[df["subject"].isin(keep)]

    sub_order = (df.groupby("subject")["macro_f1"]
                 .median().sort_values(ascending=False).index.tolist())

    GROUP = {"VP_01": "H", "VP_02": "H", "VP_04": "I",
             "VP_06": "I", "VP_12": "H", "VP_13": "H", "VP_14": "I"}
    GCOL  = {"H": TEAL, "I": WARM}

    fig, ax = plt.subplots(figsize=(7, 3.8))

    data_by_sub = [df[df["subject"] == s]["macro_f1"].values for s in sub_order]
    bp = ax.boxplot(data_by_sub, positions=range(len(sub_order)),
                    widths=0.5, patch_artist=True,
                    medianprops=dict(color="white", linewidth=2),
                    whiskerprops=dict(linewidth=0.9),
                    capprops=dict(linewidth=0.9),
                    flierprops=dict(marker="o", markersize=3,
                                   markerfacecolor="#888", alpha=0.5))
    for patch, sub in zip(bp["boxes"], sub_order):
        g = GROUP.get(sub, "H")
        patch.set_facecolor(GCOL[g])
        patch.set_alpha(0.80)

    # jitter
    rng = np.random.default_rng(42)
    for xi, vals in enumerate(data_by_sub):
        jx = rng.uniform(-0.18, 0.18, len(vals))
        ax.scatter(xi + jx, vals, s=12, color="#333", alpha=0.45, zorder=4)

    ax.set_xticks(range(len(sub_order)))
    ax.set_xticklabels(sub_order, rotation=20, ha="right")
    ax.set_ylabel("Macro F1 Score")
    ax.set_title(f"Per-Fold F1 Distribution per Participant — {MODEL_LABEL[best_model]}")
    ax.set_ylim(-0.02, 1.05)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#e0e0e0", linewidth=0.6)
    ax.grid(axis="x", visible=False)

    legend_elems = [
        mpatches.Patch(color=TEAL, alpha=0.80, label="Healthy"),
        mpatches.Patch(color=WARM, alpha=0.80, label="Impaired"),
    ]
    ax.legend(handles=legend_elems, loc="lower left")
    fig.tight_layout()
    fig.savefig("figures/fig_6_5_per_session_variability.pdf")
    plt.close()
    print("✓ fig_6_5_per_session_variability.pdf")

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 5 – Feature ablation (horizontal bar chart)
# ═══════════════════════════════════════════════════════════════════════════════
def fig_feature_ablation():
    df = df_ablation.copy()
    df = df.sort_values("macro_f1_mean")

    FEAT_LABEL = {
        "mav": "MAV", "rms": "RMS", "wl": "WL",
        "zc": "ZC", "ssc": "SSC", "var": "VAR",
        "iemg": "IEMG", "ssi": "SSI",
        "combined": "All 8 Combined",
    }
    df["feat_label"] = df["condition"].map(FEAT_LABEL)

    colors = [NAVY if c == "combined" else TEAL for c in df["condition"]]

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    y = np.arange(len(df))
    ax.barh(y, df["macro_f1_mean"], xerr=df["macro_f1_std"],
            color=colors, alpha=0.87, height=0.6,
            error_kw=dict(ecolor="#555", lw=1.1, capsize=3, capthick=1))

    ax.set_yticks(y)
    ax.set_yticklabels(df["feat_label"])
    ax.set_xlabel("Macro F1 Score (mean ± SD)")
    ax.set_title("Feature Ablation — Random Forest, LOSO-Session")
    ax.set_xlim(0, 0.82)
    ax.tick_params(axis="y", length=0)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#e0e0e0", linewidth=0.6)

    # value labels
    for i, (val, sd) in enumerate(zip(df["macro_f1_mean"], df["macro_f1_std"])):
        ax.text(val + sd + 0.008, i, f"{val:.3f}", va="center",
                ha="left", fontsize=8.5, color="#333")

    legend_elems = [
        mpatches.Patch(color=TEAL, alpha=0.87, label="Single feature"),
        mpatches.Patch(color=NAVY, alpha=0.87, label="All 8 combined"),
    ]
    ax.legend(handles=legend_elems, loc="lower right")
    fig.tight_layout()
    fig.savefig("figures/fig_6_6_feature_ablation.pdf")
    plt.close()
    print("✓ fig_6_6_feature_ablation.pdf")

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 6 – Cross-domain transfer
# ═══════════════════════════════════════════════════════════════════════════════
def fig_cross_domain():
    df = df_cross.copy()
    DOMAIN_LABEL = {
        ("pipeline", "pipeline"): "Pipeline → Pipeline\n(within-domain)",
        ("unity",    "unity"):    "Unity → Unity\n(within-domain)",
        ("pipeline", "unity"):    "Pipeline → Unity\n(cross-domain)",
        ("unity",    "pipeline"): "Unity → Pipeline\n(cross-domain)",
    }
    df["label"] = [DOMAIN_LABEL.get((td, te), f"{td}→{te}")
                   for td, te in zip(df["train_domain"], df["test_domain"])]
    colors = [TEAL, TEAL, WARM, WARM]
    hatches = ["", "//", "", "//"]

    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    y = np.arange(len(df))
    bars = ax.barh(y, df["macro_f1_mean"], color=colors, alpha=0.85,
                   height=0.55, hatch=["", "", "///", "///"])

    ax.set_yticks(y)
    ax.set_yticklabels(df["label"], fontsize=9)
    ax.set_xlabel("Macro F1 Score (mean)")
    ax.set_title("Cross-Domain Transfer — Random Forest")
    ax.set_xlim(0, 0.85)
    ax.tick_params(axis="y", length=0)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#e0e0e0", linewidth=0.6)
    ax.axvline(0, color="#999", lw=0.5)

    for bar, val in zip(bars, df["macro_f1_mean"]):
        ax.text(val + 0.012, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", ha="left", fontsize=9, color="#333")

    legend_elems = [
        mpatches.Patch(color=TEAL, alpha=0.85, label="Within-domain"),
        mpatches.Patch(color=WARM, alpha=0.85, hatch="///", label="Cross-domain"),
    ]
    ax.legend(handles=legend_elems, loc="lower right")
    fig.tight_layout()
    fig.savefig("figures/fig_6_7_cross_domain.pdf")
    plt.close()
    print("✓ fig_6_7_cross_domain.pdf")

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 7 – LOSO-Subject performance
# ═══════════════════════════════════════════════════════════════════════════════
def fig_loso_subject():
    df = df_loso_sub.copy()
    df = df.sort_values("macro_f1_mean")
    df["model_label"] = df["model"].map(MODEL_LABEL)
    colors = [MODEL_COLORS.get(m, SLATE) for m in df["model"]]

    fig, ax = plt.subplots(figsize=(5.2, 2.8))
    y = np.arange(len(df))
    ax.barh(y, df["macro_f1_mean"], xerr=df["macro_f1_std"],
            color=colors, alpha=0.87, height=0.5,
            error_kw=dict(ecolor="#555", lw=1.1, capsize=3, capthick=1))

    ax.scatter(df["accuracy_mean"], y, color="white", edgecolors="#333",
               s=28, zorder=5, linewidths=0.8)

    ax.set_yticks(y)
    ax.set_yticklabels(df["model_label"])
    ax.set_xlabel("Score")
    ax.set_title("Leave-One-Subject-Out Cross-Validation")
    ax.set_xlim(0, 0.78)
    ax.tick_params(axis="y", length=0)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#e0e0e0", linewidth=0.6)

    for i, (val,) in enumerate(zip(df["macro_f1_mean"])):
        ax.text(val + df["macro_f1_std"].iloc[i] + 0.01, i,
                f"{val:.3f}", va="center", ha="left", fontsize=8.5)

    legend_elems = [
        mpatches.Patch(color="#999", label="Macro F1 (mean ± SD)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="white",
               markeredgecolor="#333", markersize=6, label="Accuracy (mean)"),
    ]
    ax.legend(handles=legend_elems, loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig("figures/fig_6_4_loso_subject.pdf")
    plt.close()
    print("✓ fig_6_4_loso_subject.pdf")

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 8 – Computational latency scatter (log scale)
# ═══════════════════════════════════════════════════════════════════════════════
def fig_latency():
    df = df_latency.copy()
    # Handle zero inference times so they show up on log plots
    df.loc[df["inference_ms_mean"] <= 0, "inference_ms_mean"] = 0.005

    df["model_label"] = df["model"].map(MODEL_LABEL)
    df["color"]       = [MODEL_COLORS.get(m, SLATE) for m in df["model"]]

    fig, ax = plt.subplots(figsize=(6, 4))
    for _, row in df.iterrows():
        ax.scatter(row["inference_ms_mean"], row["train_seconds_mean"],
                   color=row["color"], s=100, zorder=5, alpha=0.90, edgecolors="white", linewidths=0.5)
        ax.annotate(row["model_label"],
                    xy=(row["inference_ms_mean"], row["train_seconds_mean"]),
                    xytext=(6, 4), textcoords="offset points",
                    fontsize=8.5, color="#222", weight="bold")

    ax.axvline(150, color=WARM, linewidth=1.5, linestyle="--",
               label="150 ms stability gate")
    ax.set_xscale("log")
    ax.set_yscale("log")
    
    # Adjust axes limits to better space out items
    ax.set_xlim(0.002, max(df["inference_ms_mean"])*1000)
    ax.set_ylim(0.1, max(df["train_seconds_mean"])*1.5)

    ax.set_xlabel("Inference Latency (ms / window, log scale)")
    ax.set_ylabel("Training Time (s, log scale)")
    ax.set_title("Computational Performance — All Models")
    ax.legend(fontsize=9, loc="upper left")
    ax.set_axisbelow(True)
    ax.grid(color="#e0e0e0", linewidth=0.6, which="both")
    fig.tight_layout(pad=1.5)
    fig.savefig("figures/fig_6_8_latency.pdf")
    plt.close()
    print("✓ fig_6_8_latency.pdf")

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 9 – EMG Threshold AUC by subject
# ═══════════════════════════════════════════════════════════════════════════════
def fig_threshold_auc():
    df = df_thresh_ps.copy()
    df = df[df["subject_id"] != "02_test_emg"]   # exclude unlabelled test user
    df = df.sort_values("auc")

    GROUP_LABEL = {"H": "Healthy", "I": "Impaired", "?": "Unknown"}
    GROUP_COLOR  = {"H": TEAL, "I": WARM, "?": SLATE}

    fig, ax = plt.subplots(figsize=(6, 3.2))
    y = np.arange(len(df))
    colors = [GROUP_COLOR.get(g, SLATE) for g in df["group"]]

    ax.barh(y, df["auc"], color=colors, alpha=0.85, height=0.55)
    ax.barh(y, df["opt_f1"], left=0, color=colors, alpha=0.35,
            height=0.25)   # overlay opt F1 in lighter shade for reference

    ax.set_yticks(y)
    ax.set_xticklabels
    ax.set_yticklabels(df["subject_id"])
    ax.set_xlabel("AUC / Optimal F1")
    ax.set_title("Activity-Detection Threshold: AUC per Participant")
    ax.set_xlim(0, 1.02)
    ax.tick_params(axis="y", length=0)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#e0e0e0", linewidth=0.6)

    legend_elems = [
        mpatches.Patch(color=TEAL, alpha=0.85, label="Healthy"),
        mpatches.Patch(color=WARM, alpha=0.85, label="Impaired"),
        mpatches.Patch(color=SLATE, alpha=0.85, label="Unknown group"),
        mpatches.Patch(color="grey", alpha=0.40, label="Optimal-threshold F1 (overlay)"),
    ]
    ax.legend(handles=legend_elems, loc="lower right", fontsize=7.5)
    fig.tight_layout()
    fig.savefig("figures/fig_6_9_threshold_auc.pdf")
    plt.close()
    print("✓ fig_6_9_threshold_auc.pdf")

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 10 – Class distribution pie/bar
# ═══════════════════════════════════════════════════════════════════════════════
def fig_class_distribution():
    data = pd.DataFrame({
        "class":  ["Rest", "Fist", "Pinch", "Tripod"],
        "count":  [89253, 51416, 22236, 20674],
        "pct":    [48.62, 28.01, 12.11, 11.26],
    })

    fig, axes = plt.subplots(1, 2, figsize=(7, 3.0),
                             gridspec_kw={"width_ratios": [1, 1.4]})

    # pie
    ax0 = axes[0]
    wedge_colors = CLASS_COLORS
    wedges, texts, autotexts = ax0.pie(
        data["pct"], labels=data["class"], colors=wedge_colors,
        autopct="%1.1f%%", startangle=90,
        pctdistance=0.70, labeldistance=1.15,
        wedgeprops=dict(linewidth=0.8, edgecolor="white"),
        textprops=dict(fontsize=9))
    for at in autotexts:
        at.set_fontsize(8)
    ax0.set_title("Class Distribution")

    # bar
    ax1 = axes[1]
    y = np.arange(len(data))
    ax1.barh(y, data["count"], color=CLASS_COLORS, alpha=0.85, height=0.55)
    ax1.set_yticks(y)
    ax1.set_yticklabels(data["class"])
    ax1.set_xlabel("Window Count")
    ax1.set_title("Absolute Window Counts")
    for i, (n, p) in enumerate(zip(data["count"], data["pct"])):
        ax1.text(n + 500, i, f"{n:,} ({p:.1f}%)",
                 va="center", fontsize=8.5, color="#333")
    ax1.set_xlim(0, 105000)
    ax1.tick_params(axis="y", length=0)
    ax1.set_axisbelow(True)
    ax1.grid(axis="x", color="#e0e0e0", linewidth=0.6)
    ax1.spines["right"].set_visible(False)
    ax1.spines["top"].set_visible(False)

    fig.tight_layout()
    fig.savefig("figures/fig_6_1_class_distribution.pdf")
    plt.close()
    print("✓ fig_6_1_class_distribution.pdf")

# ═══════════════════════════════════════════════════════════════════════════════
# Run all
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating figures …")
    fig_class_distribution()
    fig_loso_session_headline()
    fig_per_class_f1()
    fig_per_class_f1_selected()
    fig_loso_by_group()
    fig_per_session_variability()
    fig_feature_ablation()
    fig_cross_domain()
    fig_loso_subject()
    fig_latency()
    fig_threshold_auc()
    print("\nAll figures written to ./figures/")