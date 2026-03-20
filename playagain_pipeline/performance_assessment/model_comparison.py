"""
Comprehensive Model Comparison & Validation Pipeline.

Trains and evaluates all available gesture classifiers on the same dataset,
producing side-by-side metrics, confusion matrices, and comparison plots.

Usage:
    python -m playagain_pipeline.performance_assessment.model_comparison          # all subjects
    python -m playagain_pipeline.performance_assessment.model_comparison --subjects VP_01 VP_02
    python -m playagain_pipeline.performance_assessment.model_comparison --models svm lda catboost
    python -m playagain_pipeline.performance_assessment.model_comparison --cv-folds 5

Output is saved to  performance_assessment/results/<timestamp>/
"""

import argparse
import json
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Support both module execution and direct script execution.
PACKAGE_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = PACKAGE_ROOT.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

# Reduce native thread contention on macOS before importing numpy/sklearn/catboost.
if sys.platform == "darwin":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold

from playagain_pipeline.core.data_manager_old import DataManager
from playagain_pipeline.models.classifier import ModelManager
from playagain_pipeline.performance_assessment.session_picker_ui import pick_sessions

# ── Style ─────────────────────────────────────────────────────────────────
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")
plt.rcParams.update({
    "figure.dpi": 120,
    "figure.facecolor": "white",
    "savefig.bbox": "tight",
    "savefig.dpi": 150,
})

# ── Constants ─────────────────────────────────────────────────────────────
DATA_DIR = PACKAGE_ROOT / "data"
ALL_MODELS = list(ModelManager.AVAILABLE_MODELS.keys())

DEFAULT_FEATURE_CONFIG = {"mode": "default", "features": []}
# CNN / AttentionNet / MSTNet work on raw windows (3-D), others on features (2-D)
RAW_WINDOW_MODELS = {"cnn", "attention_net", "mstnet"}
RAW_FEATURE_CONFIG_LABEL = "raw_windows"


def _normalize_feature_config(
    feature_config: Optional[Dict[str, Any]] = None,
    *,
    fallback_label: Optional[str] = None,
) -> Dict[str, Any]:
    cfg = dict(feature_config or {})
    mode = str(cfg.get("mode", DEFAULT_FEATURE_CONFIG["mode"])).lower()
    features = cfg.get("features", []) or []
    features = [str(name) for name in features]
    features = list(dict.fromkeys(features))

    if mode not in {"default", "custom", "raw"}:
        mode = DEFAULT_FEATURE_CONFIG["mode"]
    if mode != "custom":
        features = [] if mode != "raw" else features

    label = str(cfg.get("label") or "").strip()
    if not label:
        if mode == "default":
            label = "default_features"
        elif mode == "raw":
            label = RAW_FEATURE_CONFIG_LABEL
        elif len(features) == 1:
            label = features[0]
        else:
            joined = "+".join(features) if features else "custom"
            label = joined if len(joined) <= 48 else f"{joined[:45]}..."
    if not label:
        label = str(fallback_label or "config")

    return {
        "mode": mode,
        "features": features,
        "label": label,
    }


def _resolve_feature_configs(
    feature_config: Optional[Dict[str, Any]] = None,
    feature_configs: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    raw_configs = list(feature_configs or ([] if feature_config is None else [feature_config]))
    if not raw_configs:
        raw_configs = [DEFAULT_FEATURE_CONFIG]

    resolved: List[Dict[str, Any]] = []
    seen_semantic = set()
    used_labels: Dict[str, int] = {}
    for idx, cfg in enumerate(raw_configs, start=1):
        normalized = _normalize_feature_config(cfg, fallback_label=f"config_{idx}")
        semantic_signature = (
            normalized["mode"],
            tuple(normalized["features"]),
        )
        if semantic_signature in seen_semantic:
            continue
        seen_semantic.add(semantic_signature)

        base_label = normalized["label"]
        label_count = used_labels.get(base_label, 0)
        normalized["label"] = base_label if label_count == 0 else f"{base_label}_{label_count + 1}"
        used_labels[base_label] = label_count + 1
        resolved.append(normalized)

    return resolved


def _validate_feature_configs_for_models(
    resolved_feature_configs: List[Dict[str, Any]],
    feature_models: List[str],
) -> None:
    if feature_models and any(cfg["mode"] == "raw" for cfg in resolved_feature_configs):
        raise ValueError(
            "Feature-based models cannot use a feature configuration with mode='raw'. "
            "Choose 'default' or 'custom' features instead."
        )


def _display_model_name(model_type: str, feature_label: Optional[str], include_feature_label: bool) -> str:
    if not include_feature_label or not feature_label:
        return model_type
    return f"{model_type} [{feature_label}]"


def _display_column_name(results_df: pd.DataFrame) -> str:
    return "model_display_name" if "model_display_name" in results_df.columns else "model_type"


def _metadata_for_run(
    dataset_metadata: Dict[str, Any],
    *,
    model_type: str,
    display_name: str,
    feature_cfg: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    metadata = dict(dataset_metadata)
    metadata["comparison_model_type"] = model_type
    metadata["comparison_model_display_name"] = display_name
    metadata["comparison_feature_config"] = dict(feature_cfg) if feature_cfg else None
    metadata["comparison_feature_config_label"] = feature_cfg.get("label") if feature_cfg else RAW_FEATURE_CONFIG_LABEL
    return metadata


def _build_datasets(
    dm: DataManager,
    subject_ids: Optional[List[str]],
    window_size_ms: int,
    window_stride_ms: int,
    sessions: Optional[List[Any]] = None,
    feature_configs: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    """
    Build two versions of the same dataset:
      1. feature-extracted  (2-D) — for traditional ML models
      2. raw windows        (3-D) — for CNN-family models

    Both use the same windowing / sessions so results are directly comparable.

    If `sessions` is provided, it will be used directly instead of selecting by
    `subject_ids`.
    """
    # ── Resolve sessions ──────────────────────────────────────────────
    if sessions is None:
        sids = subject_ids or dm.list_subjects()
        sessions = []
        for sid in sids:
            sessions.extend(dm.get_all_sessions(sid))

    # ── Filter out sessions whose actual data shape doesn't match ─────
    # This prevents the "inhomogeneous shape" error when np.array(windows)
    # is called — e.g. a file with 31 columns while metadata says 32.
    if sessions:
        target_ch = sessions[0].metadata.num_channels
        clean = []
        skipped = []
        for s in sessions:
            try:
                data = s.get_data()
                if data.shape[1] != target_ch:
                    skipped.append((s.metadata.subject_id, s.metadata.session_id,
                                   f"channels mismatch: {data.shape[1]} vs {target_ch}"))
                    continue
                clean.append(s)
            except Exception as e:
                skipped.append((s.metadata.subject_id, s.metadata.session_id, str(e)))

        if skipped:
            print(f"  Skipped {len(skipped)} invalid sessions:")
            for subj, sess, reason in skipped:
                print(f"    - {subj}/{sess}: {reason}")

        sessions = clean

    if not sessions:
        raise ValueError("No valid sessions available after filtering.")

    raw_ds = dm.create_dataset(
        name="comparison_raw",
        sessions=sessions,
        window_size_ms=window_size_ms,
        window_stride_ms=window_stride_ms,
        use_per_session_rotation=True,
    )
    raw_ds["metadata"] = dict(raw_ds.get("metadata", {}))
    raw_ds["metadata"]["comparison_feature_config_label"] = RAW_FEATURE_CONFIG_LABEL
    raw_ds["metadata"]["comparison_feature_config"] = {"mode": "raw", "features": [], "label": RAW_FEATURE_CONFIG_LABEL}

    feature_datasets: Dict[str, Dict[str, Any]] = {}
    for cfg in _resolve_feature_configs(feature_configs=feature_configs):
        if cfg["mode"] == "raw":
            feature_datasets[cfg["label"]] = raw_ds
            continue

        feat_ds = dm.create_dataset(
            name=f"comparison_features_{cfg['label']}",
            sessions=sessions,
            window_size_ms=window_size_ms,
            window_stride_ms=window_stride_ms,
            use_per_session_rotation=True,
            feature_config={"mode": cfg["mode"], "features": cfg["features"]},
        )
        feat_ds["metadata"] = dict(feat_ds.get("metadata", {}))
        feat_ds["metadata"]["comparison_feature_config_label"] = cfg["label"]
        feat_ds["metadata"]["comparison_feature_config"] = dict(cfg)
        feature_datasets[cfg["label"]] = feat_ds

    return raw_ds, feature_datasets


def _train_and_evaluate_fold(
    model_type: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    metadata: Dict[str, Any],
    fold_idx: int,
    X_eval: Optional[np.ndarray] = None,
    y_eval: Optional[np.ndarray] = None,
    eval_split_name: str = "val",
) -> Dict[str, Any]:
    """Train one model on one fold and return per-class + aggregate metrics."""
    mm = ModelManager(models_dir=DATA_DIR / "models" / "_comparison_tmp")
    model = mm.create_model(model_type, name=f"_cmp_{model_type}_f{fold_idx}")

    t0 = time.time()
    try:
        model.train(
            X_train, y_train, X_val, y_val,
            window_size_ms=metadata.get("window_size_ms", 200),
            sampling_rate=metadata.get("sampling_rate", 2000),
            num_channels=metadata.get("num_channels", 0),
        )
    except Exception as e:
        print(f"    [!] {model_type} fold {fold_idx} training failed: {e}")
        return {"error": str(e)}
    train_time = time.time() - t0

    X_metrics = X_eval if X_eval is not None else X_val
    y_metrics = y_eval if y_eval is not None else y_val

    y_pred = model.predict(X_metrics)
    y_proba = None
    try:
        y_proba = model.predict_proba(X_metrics)
    except Exception:
        pass

    label_names = metadata.get("label_names", {})
    unique_labels = sorted(np.unique(np.concatenate([y_train, y_metrics])))
    target_names = [label_names.get(str(lbl), label_names.get(lbl, str(lbl))) for lbl in unique_labels]

    result: Dict[str, Any] = {
        "model_type": metadata.get("comparison_model_type", model_type),
        "model_display_name": metadata.get("comparison_model_display_name", model_type),
        "feature_config_label": metadata.get("comparison_feature_config_label"),
        "feature_config": metadata.get("comparison_feature_config"),
        "fold": fold_idx,
        "evaluation_split": eval_split_name,
        "accuracy": float(accuracy_score(y_metrics, y_pred)),
        "f1_weighted": float(f1_score(y_metrics, y_pred, average="weighted", zero_division=0)),
        "precision_weighted": float(precision_score(y_metrics, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_metrics, y_pred, average="weighted", zero_division=0)),
        "train_time_s": round(train_time, 2),
        "confusion_matrix": confusion_matrix(y_metrics, y_pred, labels=unique_labels).tolist(),
        "labels": [int(l) for l in unique_labels],
        "label_names": target_names,
        "classification_report": classification_report(
            y_metrics, y_pred, labels=unique_labels, target_names=target_names,
            output_dict=True, zero_division=0,
        ),
    }

    if y_proba is not None:
        # Mean confidence on correct vs incorrect predictions
        correct_mask = y_pred == y_metrics
        result["mean_confidence_correct"] = float(np.mean(np.max(y_proba[correct_mask], axis=1))) if np.any(correct_mask) else 0.0
        result["mean_confidence_incorrect"] = float(np.mean(np.max(y_proba[~correct_mask], axis=1))) if np.any(~correct_mask) else 0.0

    # Clean up temp model files
    mm.delete_model(f"_cmp_{model_type}_f{fold_idx}")

    return result


# ═══════════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════════

def plot_metric_comparison(
    results_df: pd.DataFrame,
    out_dir: Path,
) -> None:
    """Bar chart comparing accuracy, F1 etc. across models (mean ± std over folds)."""
    metrics = ["accuracy", "f1_weighted", "precision_weighted", "recall_weighted"]
    nice = ["Accuracy", "F1 (weighted)", "Precision (weighted)", "Recall (weighted)"]
    label_col = _display_column_name(results_df)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for ax, metric, title in zip(axes.ravel(), metrics, nice):
        summary = (
            results_df.groupby(label_col)[metric]
            .agg(["mean", "std"])
            .sort_values("mean", ascending=False)
        )
        bars = ax.bar(summary.index, summary["mean"], yerr=summary["std"],
                       capsize=4, color=sns.color_palette("husl", len(summary)))
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel(title)
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
        ax.tick_params(axis="x", rotation=30)
        # Value labels
        for bar, val in zip(bars, summary["mean"]):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                    f"{val:.1%}", ha="center", va="bottom", fontsize=9)

    fig.suptitle("Model Comparison — Aggregate Metrics (mean ± std over folds)",
                 fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(out_dir / "metric_comparison.png")
    plt.close(fig)


def plot_confusion_matrices(
    all_results: Dict[str, List[Dict]],
    out_dir: Path,
) -> None:
    """Per-model average confusion matrix (summed over folds)."""
    model_types = sorted(all_results.keys())
    n = len(model_types)
    ncols = min(n, 4)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows))
    if n == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)

    for idx, model_type in enumerate(model_types):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]

        folds = all_results[model_type]
        # Sum confusion matrices across folds
        cm_sum = np.zeros_like(np.array(folds[0]["confusion_matrix"]), dtype=float)
        for fold in folds:
            cm_sum += np.array(fold["confusion_matrix"])
        # Normalize rows to get per-class recall
        row_sums = cm_sum.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_norm = cm_sum / row_sums

        label_names = folds[0]["label_names"]
        sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                    xticklabels=label_names, yticklabels=label_names,
                    ax=ax, vmin=0, vmax=1, cbar=False)
        ax.set_title(model_type, fontweight="bold")
        ax.set_ylabel("True")
        ax.set_xlabel("Predicted")

    # Hide unused subplots
    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].set_visible(False)

    fig.suptitle("Normalized Confusion Matrices (summed over CV folds)",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(out_dir / "confusion_matrices.png")
    plt.close(fig)


def plot_per_class_f1(
    all_results: Dict[str, List[Dict]],
    out_dir: Path,
) -> None:
    """Grouped bar chart: per-class F1 across models."""
    rows = []
    for model_type, folds in all_results.items():
        for fold in folds:
            report = fold["classification_report"]
            display_name = fold.get("model_display_name", model_type)
            for cls, metrics in report.items():
                if isinstance(metrics, dict) and "f1-score" in metrics:
                    if cls in ("accuracy", "macro avg", "weighted avg"):
                        continue
                    rows.append({
                        "Model": display_name,
                        "Gesture": cls,
                        "F1": metrics["f1-score"],
                    })

    df = pd.DataFrame(rows)
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.barplot(data=df, x="Gesture", y="F1", hue="Model", ax=ax,
                errorbar="sd", capsize=0.05)
    ax.set_title("Per-Gesture F1 Score by Model", fontsize=14, fontweight="bold")
    ax.set_ylabel("F1 Score")
    ax.set_ylim(0, 1.05)
    ax.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    fig.savefig(out_dir / "per_class_f1.png")
    plt.close(fig)


def plot_training_time(results_df: pd.DataFrame, out_dir: Path) -> None:
    """Bar chart of training time per model."""
    label_col = _display_column_name(results_df)
    summary = (
        results_df.groupby(label_col)["train_time_s"]
        .agg(["mean", "std"])
        .sort_values("mean")
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(summary.index, summary["mean"], xerr=summary["std"],
                    capsize=4, color=sns.color_palette("husl", len(summary)))
    ax.set_xlabel("Training Time (seconds)")
    ax.set_title("Training Time per Model (per fold)", fontweight="bold")
    for bar, val in zip(bars, summary["mean"]):
        ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}s", va="center", fontsize=9)
    plt.tight_layout()
    fig.savefig(out_dir / "training_time.png")
    plt.close(fig)


def plot_confidence_analysis(
    all_results: Dict[str, List[Dict]],
    out_dir: Path,
) -> None:
    """Bar chart: mean confidence on correct vs incorrect predictions."""
    rows = []
    for model_type, folds in all_results.items():
        for fold in folds:
            display_name = fold.get("model_display_name", model_type)
            if "mean_confidence_correct" in fold:
                rows.append({
                    "Model": display_name,
                    "Prediction": "Correct",
                    "Confidence": fold["mean_confidence_correct"],
                })
                rows.append({
                    "Model": display_name,
                    "Prediction": "Incorrect",
                    "Confidence": fold["mean_confidence_incorrect"],
                })

    df = pd.DataFrame(rows)
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.barplot(data=df, x="Model", y="Confidence", hue="Prediction",
                ax=ax, palette={"Correct": "#66BB6A", "Incorrect": "#EF5350"},
                errorbar="sd", capsize=0.05)
    ax.set_title("Prediction Confidence: Correct vs Incorrect", fontweight="bold")
    ax.set_ylabel("Mean Confidence")
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    fig.savefig(out_dir / "confidence_analysis.png")
    plt.close(fig)


def generate_summary_table(
    all_results: Dict[str, List[Dict]],
    out_dir: Path,
) -> pd.DataFrame:
    """Create a summary CSV/DataFrame with mean ± std for each model."""
    rows = []
    for model_type, folds in all_results.items():
        accs = [f["accuracy"] for f in folds]
        f1s = [f["f1_weighted"] for f in folds]
        precs = [f["precision_weighted"] for f in folds]
        recs = [f["recall_weighted"] for f in folds]
        times = [f["train_time_s"] for f in folds]
        display_name = folds[0].get("model_display_name", model_type)
        feature_label = folds[0].get("feature_config_label")

        rows.append({
            "Model": display_name,
            "Base Model": folds[0].get("model_type", model_type),
            "Feature Config": feature_label,
            "Accuracy (mean)": np.mean(accs),
            "Accuracy (std)": np.std(accs),
            "F1 weighted (mean)": np.mean(f1s),
            "F1 weighted (std)": np.std(f1s),
            "Precision (mean)": np.mean(precs),
            "Recall (mean)": np.mean(recs),
            "Train Time (mean s)": np.mean(times),
            "Folds": len(folds),
        })

    df = pd.DataFrame(rows).sort_values("Accuracy (mean)", ascending=False)
    df.to_csv(out_dir / "summary.csv", index=False)
    return df


# ═══════════════════════════════════════════════════════════════════════════
# Thread-safe plot generation  (avoids SIGBUS on macOS)
# ═══════════════════════════════════════════════════════════════════════════

def _plot_worker(results_df_dict: Dict, all_results: Dict, out_dir_str: str) -> None:
    """Target function for the child process — imports matplotlib fresh."""
    import matplotlib as _mpl
    _mpl.use("Agg")
    import matplotlib.pyplot as _plt

    out_dir = Path(out_dir_str)
    results_df = pd.DataFrame(results_df_dict)

    # Overall summary plots (all models × feature configs combined)
    plot_metric_comparison(results_df, out_dir)
    plot_confusion_matrices(all_results, out_dir)
    plot_per_class_f1(all_results, out_dir)
    plot_training_time(results_df, out_dir)
    plot_confidence_analysis(all_results, out_dir)

    # Per-feature-config breakdown when multiple configs are present.
    # Groups results by their feature_config_label and generates a
    # separate set of confusion matrices and per-class F1 plots in
    # a subdirectory for each config so individual plots stay readable.
    feature_labels = set()
    for folds in all_results.values():
        for fold in folds:
            fl = fold.get("feature_config_label")
            if fl:
                feature_labels.add(fl)

    if len(feature_labels) > 1:
        for fl in sorted(feature_labels):
            sub = {k: v for k, v in all_results.items()
                   if v and v[0].get("feature_config_label") == fl}
            if not sub:
                continue
            sub_dir = out_dir / fl
            sub_dir.mkdir(exist_ok=True)
            sub_rows = [fold for folds in sub.values() for fold in folds]
            sub_df = pd.DataFrame(sub_rows)
            plot_metric_comparison(sub_df, sub_dir)
            plot_confusion_matrices(sub, sub_dir)
            plot_per_class_f1(sub, sub_dir)
            plot_training_time(sub_df, sub_dir)
            plot_confidence_analysis(sub, sub_dir)

    _plt.close("all")


def _generate_plots_safe(
    results_df: pd.DataFrame,
    all_results: Dict[str, List[Dict]],
    output_dir: Path,
) -> None:
    """
    Generate all comparison plots in a *child process* so that matplotlib
    cannot collide with the PySide6 / Qt event loop on the main thread.

    On macOS + Apple Silicon the Agg backend can still trigger Metal/MPS
    operations; doing that from a QThread causes SIGBUS.  A separate process
    avoids the problem entirely.

    Falls back to in-process generation when multiprocessing fails (e.g. when
    running from a plain CLI without Qt).
    """
    import multiprocessing as mp

    try:
        ctx = mp.get_context("spawn")          # 'spawn' = clean interpreter
        p = ctx.Process(
            target=_plot_worker,
            args=(results_df.to_dict(orient="list"), all_results, str(output_dir)),
        )
        p.start()
        p.join(timeout=120)
        if p.exitcode != 0:
            print(f"  [warn] Plot subprocess exited with code {p.exitcode}, "
                  f"retrying in-process …")
            _plot_worker(results_df.to_dict(orient="list"), all_results, str(output_dir))
    except Exception as exc:
        print(f"  [warn] Subprocess plot generation failed ({exc}), "
              f"generating in-process …")
        _plot_worker(results_df.to_dict(orient="list"), all_results, str(output_dir))


# ═══════════════════════════════════════════════════════════════════════════
# Helpers — metadata & incremental saving
# ═══════════════════════════════════════════════════════════════════════════


def _collect_session_info(sessions: List[Any]) -> List[Dict[str, Any]]:
    """Collect per-session metadata for logging purposes."""
    info = []
    for s in sessions:
        entry: Dict[str, Any] = {
            "session_id": s.metadata.session_id,
            "subject_id": s.metadata.subject_id,
            "num_channels": s.metadata.num_channels,
        }
        if hasattr(s.metadata, "gesture_set") and s.metadata.gesture_set:
            entry["gesture_set"] = s.metadata.gesture_set
        if hasattr(s.metadata, "protocol") and s.metadata.protocol:
            entry["protocol"] = s.metadata.protocol
        if hasattr(s.metadata, "created_at") and s.metadata.created_at:
            entry["created_at"] = str(s.metadata.created_at)
        try:
            data = s.get_data()
            entry["num_raw_samples"] = data.shape[0]
        except Exception:
            entry["num_raw_samples"] = None
        info.append(entry)
    return info


def _save_partial_results(
    all_results: Dict[str, List[Dict]],
    output_dir: Path,
    status: str = "in_progress",
) -> None:
    """Save incremental results so partial progress is preserved on early stop."""
    partial = {k: v for k, v in all_results.items() if v}
    if not partial:
        return

    with open(output_dir / "results_partial.json", "w") as f:
        json.dump(partial, f, indent=2)

    rows = []
    for model_type, folds in partial.items():
        accs = [fold["accuracy"] for fold in folds]
        f1s = [fold["f1_weighted"] for fold in folds]
        first = folds[0]
        rows.append({
            "Model": first.get("model_display_name", model_type),
            "Base Model": first.get("model_type", model_type),
            "Feature Config": first.get("feature_config_label"),
            "Accuracy (mean)": float(np.mean(accs)),
            "Accuracy (std)": float(np.std(accs)),
            "F1 weighted (mean)": float(np.mean(f1s)),
            "F1 weighted (std)": float(np.std(f1s)),
            "Folds completed": len(folds),
            "Status": status,
        })
    pd.DataFrame(rows).to_csv(output_dir / "summary_partial.csv", index=False)


def _split_indices_by_label(
    y: np.ndarray,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    *,
    shuffle: bool,
    random_seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split sample indices per class to preserve class balance across splits."""
    rng = np.random.default_rng(random_seed)
    train_parts: List[np.ndarray] = []
    val_parts: List[np.ndarray] = []
    test_parts: List[np.ndarray] = []

    for label in np.unique(y):
        class_idx = np.where(y == label)[0]
        if shuffle:
            rng.shuffle(class_idx)

        n = len(class_idx)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))

        if n_train > n:
            n_train = n
        if n_train + n_val > n:
            n_val = max(0, n - n_train)

        n_test = n - n_train - n_val
        if n_test < 0:
            n_test = 0

        train_parts.append(class_idx[:n_train])
        val_parts.append(class_idx[n_train:n_train + n_val])
        test_parts.append(class_idx[n_train + n_val:n_train + n_val + n_test])

    train_idx = np.concatenate(train_parts) if train_parts else np.array([], dtype=int)
    val_idx = np.concatenate(val_parts) if val_parts else np.array([], dtype=int)
    test_idx = np.concatenate(test_parts) if test_parts else np.array([], dtype=int)

    if shuffle:
        rng.shuffle(train_idx)
        rng.shuffle(val_idx)
        rng.shuffle(test_idx)

    return train_idx, val_idx, test_idx


# ═══════════════════════════════════════════════════════════════════════════
# Main pipeline
# ═══════════════════════════════════════════════════════════════════════════

def run_comparison(
    subject_ids: Optional[List[str]] = None,
    model_types: Optional[List[str]] = None,
    cv_folds: int = 5,
    window_size_ms: int = 200,
    window_stride_ms: int = 50,
    test_ratio: float = 0.2,
    output_dir: Optional[Path] = None,
    validate_subject_ids: Optional[List[str]] = None,
    validate_session_ids: Optional[List[str]] = None,
    interactive: bool = False,
    _holdout_sessions: Optional[Tuple[List[Any], ...]] = None,
    _cv_sessions_and_folds: Optional[Tuple[List[Any], int]] = None,
    _intra_session_split: Optional[Dict[str, Any]] = None,
    feature_config: Optional[Dict[str, Any]] = None,
    feature_configs: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Run the full comparison pipeline.

    1. Build datasets (feature-extracted + raw windows)
    2. Run stratified k-fold cross-validation for each model
    3. Aggregate metrics and produce comparison plots
    4. Save results JSON + CSV + figures

    Returns dict with all results.
    """
    if model_types is None:
        model_types = ALL_MODELS
    model_types = [m for m in model_types if m in ModelManager.AVAILABLE_MODELS]
    resolved_feature_configs = _resolve_feature_configs(feature_config=feature_config, feature_configs=feature_configs)
    include_feature_label = len(resolved_feature_configs) > 1
    raw_models = [m for m in model_types if m in RAW_WINDOW_MODELS]
    feature_models = [m for m in model_types if m not in RAW_WINDOW_MODELS]
    _validate_feature_configs_for_models(resolved_feature_configs, feature_models)

    # ── Output directory ──────────────────────────────────────────────
    pipeline_start = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent / "results" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'═' * 60}")
    print(f"  Model Comparison Pipeline")
    print(f"  Output → {output_dir}")
    print(f"{'═' * 60}\n")

    # ── Resolve train / val sessions ─────────────────────────────────
    dm = DataManager(DATA_DIR)
    subjects = subject_ids or dm.list_subjects()
    print(f"Subjects: {subjects}")
    print(f"Models:   {model_types}")
    print(f"CV folds: {cv_folds}\n")

    # Initialize dataset-summary variables for config output
    n_samples = None
    n_samples_train = None
    n_samples_val = None
    n_samples_test = None
    n_classes = None
    label_names = None
    evaluation_mode = "cv"  # will be overridden to "holdout" if applicable
    train_session_info: List[Dict[str, Any]] = []
    val_session_info: List[Dict[str, Any]] = []
    ui_cv_sessions: Optional[List[Any]] = None   # set in interactive CV branch
    val_sessions:   Optional[List[Any]] = None
    test_sessions:  Optional[List[Any]] = None
    train_sessions: Optional[List[Any]] = None

    intra_split_request: Optional[Dict[str, Any]] = None

    # ── Pre-supplied sessions (from GUI/session picker) ───────────────
    if _intra_session_split is not None:
        intra_split_request = dict(_intra_session_split)
        source_sessions = intra_split_request.get("sessions") or []
        subjects = sorted({s.metadata.subject_id for s in source_sessions})

    elif _holdout_sessions is not None:
        if len(_holdout_sessions) == 2:
            train_sessions, val_sessions = _holdout_sessions
            test_sessions = None
        elif len(_holdout_sessions) == 3:
            train_sessions, val_sessions, test_sessions = _holdout_sessions
        else:
            raise ValueError("_holdout_sessions must contain (train, val) or (train, val, test).")

    elif _cv_sessions_and_folds is not None:
        ui_cv_sessions, cv_folds = _cv_sessions_and_folds
        subjects = sorted({s.metadata.subject_id for s in ui_cv_sessions})

    # ── Interactive mode: open the GUI session picker ────────────────
    elif interactive:
        ui_result = pick_sessions(dm)
        if ui_result is None:
            print("Cancelled.")
            return {}
        if ui_result["mode"] == "holdout":
            train_sessions = ui_result["train"]
            val_sessions   = ui_result["val"]
        else:
            # CV mode — fall through to the CV else-branch below
            train_sessions  = None
            val_sessions    = None
            subjects        = sorted({s.metadata.subject_id for s in ui_result["sessions"]})
            cv_folds        = ui_result["cv_folds"]
            ui_cv_sessions  = ui_result["sessions"]   # used in the CV branch

    # ── CLI / programmatic hold-out mode ─────────────────────────────
    elif validate_subject_ids or validate_session_ids:
        all_sessions = dm.get_all_sessions()
        val_id_set: set = set()
        if validate_session_ids:
            val_id_set |= set(validate_session_ids)
        if validate_subject_ids:
            for sid in validate_subject_ids:
                val_id_set |= {s.metadata.session_id for s in dm.get_all_sessions(sid)}

        val_sessions   = [s for s in all_sessions if s.metadata.session_id in val_id_set]
        train_sessions = [s for s in all_sessions if s.metadata.session_id not in val_id_set]
        test_sessions  = val_sessions

    else:
        val_sessions   = None
        train_sessions = None

    if intra_split_request is not None:
        evaluation_mode = "intra_session"
        source_sessions = intra_split_request.get("sessions") or []
        if not source_sessions:
            raise ValueError("Intra-session mode requires a non-empty 'sessions' list.")

        train_ratio = float(intra_split_request.get("train_ratio", 0.6))
        val_ratio = float(intra_split_request.get("val_ratio", 0.2))
        test_ratio = float(intra_split_request.get("test_ratio", 0.2))
        shuffle = bool(intra_split_request.get("shuffle", False))
        random_seed = int(intra_split_request.get("random_seed", 42))

        if train_ratio <= 0 or val_ratio < 0 or test_ratio < 0:
            raise ValueError("Intra-session ratios must satisfy train>0 and val/test>=0.")
        ratio_sum = train_ratio + val_ratio + test_ratio
        if ratio_sum <= 0:
            raise ValueError("Intra-session ratios must sum to a positive value.")
        train_ratio /= ratio_sum
        val_ratio /= ratio_sum
        test_ratio /= ratio_sum

        print("\nBuilding intra-session datasets …")
        raw_ds, feat_ds_map = _build_datasets(
            dm, None, window_size_ms, window_stride_ms,
            sessions=source_sessions,
            feature_configs=resolved_feature_configs if feature_models else None,
        )
        y = raw_ds["y"]
        train_idx, val_idx, test_idx = _split_indices_by_label(
            y,
            train_ratio,
            val_ratio,
            test_ratio,
            shuffle=shuffle,
            random_seed=random_seed,
        )

        if len(train_idx) == 0:
            raise ValueError("Intra-session split produced an empty training set.")
        if len(val_idx) == 0 and len(test_idx) == 0:
            raise ValueError("Intra-session split produced no validation/test samples.")

        val_for_train_idx = val_idx if len(val_idx) > 0 else test_idx
        eval_idx = test_idx if len(test_idx) > 0 else val_idx
        eval_split_name = "test" if len(test_idx) > 0 else "val"

        n_samples = int(raw_ds["X"].shape[0])
        n_samples_train = int(len(train_idx))
        n_samples_val = int(len(val_idx))
        n_samples_test = int(len(test_idx))
        n_classes = raw_ds["metadata"].get("num_classes")
        label_names = raw_ds["metadata"].get("label_names")
        train_session_info = _collect_session_info(source_sessions)
        ui_cv_sessions = source_sessions

        print(
            f"  ✓ windows={n_samples:,} | train={n_samples_train:,} | "
            f"val={n_samples_val:,} | test={n_samples_test:,}"
        )

        all_result_labels = [
            _display_model_name(model_type, RAW_FEATURE_CONFIG_LABEL, include_feature_label)
            for model_type in raw_models
        ] + [
            _display_model_name(model_type, cfg["label"], include_feature_label)
            for cfg in resolved_feature_configs
            for model_type in feature_models
        ]
        all_results: Dict[str, List[Dict]] = {label: [] for label in all_result_labels}
        completed = 0
        total = len(all_result_labels)

        for model_type in raw_models:
            display_name = _display_model_name(model_type, RAW_FEATURE_CONFIG_LABEL, include_feature_label)
            metadata = _metadata_for_run(
                raw_ds["metadata"],
                model_type=model_type,
                display_name=display_name,
                feature_cfg={"mode": "raw", "features": [], "label": RAW_FEATURE_CONFIG_LABEL},
            )
            print(f"\n[{completed+1}/{total}] Training {display_name} … ", end="", flush=True)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = _train_and_evaluate_fold(
                    model_type,
                    raw_ds["X"][train_idx], raw_ds["y"][train_idx],
                    raw_ds["X"][val_for_train_idx], raw_ds["y"][val_for_train_idx],
                    metadata,
                    0,
                    X_eval=raw_ds["X"][eval_idx],
                    y_eval=raw_ds["y"][eval_idx],
                    eval_split_name=eval_split_name,
                )
            if "error" in result:
                print(f"❌ FAILED ({result['error'][:80]})")
            else:
                print(f"✓ acc={result['accuracy']:.3f}  F1={result['f1_weighted']:.3f}  ({result['train_time_s']:.1f}s)")
                all_results[display_name].append(result)
            completed += 1
            _save_partial_results(all_results, output_dir)

        for cfg in resolved_feature_configs:
            ds = feat_ds_map[cfg["label"]]
            for model_type in feature_models:
                display_name = _display_model_name(model_type, cfg["label"], include_feature_label)
                metadata = _metadata_for_run(
                    ds["metadata"],
                    model_type=model_type,
                    display_name=display_name,
                    feature_cfg=cfg,
                )
                print(f"\n[{completed+1}/{total}] Training {display_name} … ", end="", flush=True)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    result = _train_and_evaluate_fold(
                        model_type,
                        ds["X"][train_idx], ds["y"][train_idx],
                        ds["X"][val_for_train_idx], ds["y"][val_for_train_idx],
                        metadata,
                        0,
                        X_eval=ds["X"][eval_idx],
                        y_eval=ds["y"][eval_idx],
                        eval_split_name=eval_split_name,
                    )
                if "error" in result:
                    print(f"❌ FAILED ({result['error'][:80]})")
                else:
                    print(f"✓ acc={result['accuracy']:.3f}  F1={result['f1_weighted']:.3f}  ({result['train_time_s']:.1f}s)")
                    all_results[display_name].append(result)
                completed += 1
                _save_partial_results(all_results, output_dir)

    elif val_sessions is not None:
        if not train_sessions:
            raise ValueError("No training sessions left after excluding validation sessions.")

        evaluation_mode = "holdout"
        train_session_info = _collect_session_info(train_sessions)
        val_session_info = _collect_session_info(val_sessions)

        print(f"  Train sessions ({len(train_sessions)}): "
              f"{sorted({s.metadata.subject_id for s in train_sessions})}")
        print(f"  Val sessions   ({len(val_sessions)}): "
              f"{sorted({s.metadata.subject_id for s in val_sessions})}")
        if test_sessions is not None:
            print(f"  Test sessions  ({len(test_sessions)}): "
                f"{sorted({s.metadata.subject_id for s in test_sessions})}")

        # Build datasets for train and val (raw + features)
        print("\nBuilding training datasets …")
        raw_train_ds, feat_train_ds_map = _build_datasets(
            dm, None, window_size_ms, window_stride_ms,
            sessions=train_sessions,
            feature_configs=resolved_feature_configs if feature_models else None,
        )
        n_samples_train = raw_train_ds['X'].shape[0]
        print(f"   ✓ Training: {n_samples_train:,} windows")

        print("Building validation datasets …")
        raw_val_ds, feat_val_ds_map = _build_datasets(
            dm, None, window_size_ms, window_stride_ms,
            sessions=val_sessions,
            feature_configs=resolved_feature_configs if feature_models else None,
        )
        n_samples_val = raw_val_ds['X'].shape[0]
        print(f"   ✓ Validation: {n_samples_val:,} windows")

        eval_raw_ds = raw_val_ds
        eval_feat_ds_map = feat_val_ds_map
        if test_sessions is not None:
            print("Building test datasets …")
            raw_test_ds, feat_test_ds_map = _build_datasets(
                dm, None, window_size_ms, window_stride_ms,
                sessions=test_sessions,
                feature_configs=resolved_feature_configs if feature_models else None,
            )
            n_samples_test = raw_test_ds['X'].shape[0]
            print(f"   ✓ Test: {n_samples_test:,} windows")
            eval_raw_ds = raw_test_ds
            eval_feat_ds_map = feat_test_ds_map
        else:
            n_samples_test = n_samples_val

        n_samples = n_samples_train + n_samples_val + (n_samples_test or 0)
        n_classes = raw_train_ds["metadata"].get("num_classes", eval_raw_ds["metadata"].get("num_classes"))
        label_names = raw_train_ds["metadata"].get("label_names", eval_raw_ds["metadata"].get("label_names"))

        all_result_labels = [
            _display_model_name(model_type, RAW_FEATURE_CONFIG_LABEL, include_feature_label)
            for model_type in raw_models
        ] + [
            _display_model_name(model_type, cfg["label"], include_feature_label)
            for cfg in resolved_feature_configs
            for model_type in feature_models
        ]
        all_results: Dict[str, List[Dict]] = {label: [] for label in all_result_labels}
        completed = 0
        total = len(all_result_labels)

        for model_type in raw_models:
            display_name = _display_model_name(model_type, RAW_FEATURE_CONFIG_LABEL, include_feature_label)
            metadata = _metadata_for_run(
                eval_raw_ds["metadata"],
                model_type=model_type,
                display_name=display_name,
                feature_cfg={"mode": "raw", "features": [], "label": RAW_FEATURE_CONFIG_LABEL},
            )
            print(f"\n[{completed+1}/{total}] Training {display_name} … ", end="", flush=True)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = _train_and_evaluate_fold(
                    model_type, raw_train_ds["X"], raw_train_ds["y"],
                    raw_val_ds["X"], raw_val_ds["y"],
                        metadata, 0,
                        X_eval=eval_raw_ds["X"], y_eval=eval_raw_ds["y"],
                        eval_split_name="test" if test_sessions is not None else "val",
                )
            if "error" in result:
                print(f"❌ FAILED ({result['error'][:80]})")
            else:
                print(f"✓ acc={result['accuracy']:.3f}  F1={result['f1_weighted']:.3f}  ({result['train_time_s']:.1f}s)")
                all_results[display_name].append(result)
            completed += 1
            _save_partial_results(all_results, output_dir)

        for cfg in resolved_feature_configs:
            feat_train_ds = feat_train_ds_map[cfg["label"]]
            feat_val_ds = feat_val_ds_map[cfg["label"]]
            feat_eval_ds = eval_feat_ds_map[cfg["label"]]
            for model_type in feature_models:
                display_name = _display_model_name(model_type, cfg["label"], include_feature_label)
                metadata = _metadata_for_run(
                    feat_eval_ds["metadata"],
                    model_type=model_type,
                    display_name=display_name,
                    feature_cfg=cfg,
                )
                print(f"\n[{completed+1}/{total}] Training {display_name} … ", end="", flush=True)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    result = _train_and_evaluate_fold(
                        model_type, feat_train_ds["X"], feat_train_ds["y"],
                        feat_val_ds["X"], feat_val_ds["y"],
                        metadata, 0,
                        X_eval=feat_eval_ds["X"], y_eval=feat_eval_ds["y"],
                        eval_split_name="test" if test_sessions is not None else "val",
                    )
                if "error" in result:
                    print(f"❌ FAILED ({result['error'][:80]})")
                else:
                    print(f"✓ acc={result['accuracy']:.3f}  F1={result['f1_weighted']:.3f}  ({result['train_time_s']:.1f}s)")
                    all_results[display_name].append(result)
                completed += 1
                _save_partial_results(all_results, output_dir)

    else:
        # ── Pure cross-validation (no hold-out) ───────────────────────
        evaluation_mode = "cv"
        print("\nBuilding datasets …")
        raw_ds, feat_ds_map = _build_datasets(
            dm, subjects, window_size_ms, window_stride_ms,
            sessions=ui_cv_sessions,
            feature_configs=resolved_feature_configs if feature_models else None,
        )
        n_samples = raw_ds["X"].shape[0]
        n_classes = raw_ds["metadata"]["num_classes"]
        label_names = raw_ds["metadata"]["label_names"]
        print(f"  ✓ {n_samples:,} windows  |  {n_classes} classes  |  labels: {label_names}")

        # Collect session info for CV sessions
        if ui_cv_sessions:
            train_session_info = _collect_session_info(ui_cv_sessions)

        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        all_result_labels = [
            _display_model_name(model_type, RAW_FEATURE_CONFIG_LABEL, include_feature_label)
            for model_type in raw_models
        ] + [
            _display_model_name(model_type, cfg["label"], include_feature_label)
            for cfg in resolved_feature_configs
            for model_type in feature_models
        ]
        all_results: Dict[str, List[Dict]] = {label: [] for label in all_result_labels}
        total_folds = cv_folds
        total_runs_per_fold = len(all_result_labels)

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(raw_ds["X"], raw_ds["y"])):
            print(f"\n═══ Fold {fold_idx + 1}/{total_folds} ═══ "
                  f"(train={len(train_idx):,}, val={len(val_idx):,})")

            run_idx = 0
            for model_type in raw_models:
                run_idx += 1
                display_name = _display_model_name(model_type, RAW_FEATURE_CONFIG_LABEL, include_feature_label)
                metadata = _metadata_for_run(
                    raw_ds["metadata"],
                    model_type=model_type,
                    display_name=display_name,
                    feature_cfg={"mode": "raw", "features": [], "label": RAW_FEATURE_CONFIG_LABEL},
                )
                print(f"  [{run_idx}/{total_runs_per_fold}] {display_name:<28s} … ", end="", flush=True)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    result = _train_and_evaluate_fold(
                        model_type,
                        raw_ds["X"][train_idx], raw_ds["y"][train_idx],
                        raw_ds["X"][val_idx], raw_ds["y"][val_idx],
                        metadata, fold_idx,
                    )
                if "error" in result:
                    print(f"FAILED ({result['error'][:60]})")
                else:
                    print(f"acc={result['accuracy']:.3f}  F1={result['f1_weighted']:.3f}  ({result['train_time_s']:.1f}s)")
                    all_results[display_name].append(result)
                _save_partial_results(all_results, output_dir)

            for cfg in resolved_feature_configs:
                ds = feat_ds_map[cfg["label"]]
                for model_type in feature_models:
                    run_idx += 1
                    display_name = _display_model_name(model_type, cfg["label"], include_feature_label)
                    metadata = _metadata_for_run(
                        ds["metadata"],
                        model_type=model_type,
                        display_name=display_name,
                        feature_cfg=cfg,
                    )
                    print(f"  [{run_idx}/{total_runs_per_fold}] {display_name:<28s} … ", end="", flush=True)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        result = _train_and_evaluate_fold(
                            model_type,
                            ds["X"][train_idx], ds["y"][train_idx],
                            ds["X"][val_idx], ds["y"][val_idx],
                            metadata, fold_idx,
                        )
                    if "error" in result:
                        print(f"FAILED ({result['error'][:60]})")
                    else:
                        print(f"acc={result['accuracy']:.3f}  F1={result['f1_weighted']:.3f}  ({result['train_time_s']:.1f}s)")
                        all_results[display_name].append(result)
                    _save_partial_results(all_results, output_dir)

    # Remove models that had no successful folds
    all_results = {k: v for k, v in all_results.items() if v}

    if not all_results:
        print("\n[!] No models were successfully trained. Aborting.")
        return {}

    # ── Aggregate results ─────────────────────────────────────────────
    flat_rows = []
    for model_type, folds in all_results.items():
        for fold in folds:
            flat_rows.append(fold)

    results_df = pd.DataFrame(flat_rows)

    # ── Generate outputs ──────────────────────────────────────────────
    print(f"\n{'─' * 60}")
    print("Generating plots and summary …")

    summary = generate_summary_table(all_results, output_dir)
    print(f"\n{summary.to_string(index=False)}\n")

    # Generate plots in a child process to avoid SIGBUS on macOS when
    # matplotlib is used from a QThread while PySide6 owns the main
    # thread's graphics context.
    _generate_plots_safe(results_df, all_results, output_dir)

    # Save raw results as JSON
    serializable = {}
    for model_type, folds in all_results.items():
        serializable[model_type] = folds
    with open(output_dir / "results.json", "w") as f:
        json.dump(serializable, f, indent=2)

    # Compute pipeline duration
    pipeline_duration_s = round(time.time() - pipeline_start, 2)

    # Save config with rich metadata
    config = {
        "timestamp": timestamp,
        "evaluation_mode": evaluation_mode,
        "pipeline_duration_s": pipeline_duration_s,
        "subjects": subjects,
        "model_types": model_types,
        "result_labels": list(all_results.keys()),
        "cv_folds": cv_folds,
        "window_size_ms": window_size_ms,
        "window_stride_ms": window_stride_ms,
        # Dataset statistics
        "total_windows": n_samples,
        "n_train_windows": n_samples_train,
        "n_val_windows": n_samples_val,
        "n_test_windows": n_samples_test,
        "n_classes": n_classes,
        "label_names": label_names,
        # Session details
        "n_train_sessions": len(train_sessions) if train_sessions else (
            len(ui_cv_sessions) if ui_cv_sessions else None
        ),
        "n_val_sessions": len(val_sessions) if val_sessions else None,
        "n_test_sessions": len(test_sessions) if test_sessions else None,
        "train_session_ids": (
            [s.metadata.session_id for s in train_sessions]
            if train_sessions else None
        ),
        "train_subjects": (
            sorted({s.metadata.subject_id for s in train_sessions})
            if train_sessions else None
        ),
        "validation_session_ids": (
            [s.metadata.session_id for s in val_sessions]
            if val_sessions else None
        ),
        "validation_subjects": (
            sorted({s.metadata.subject_id for s in val_sessions})
            if val_sessions else validate_subject_ids
        ),
        "test_session_ids": (
            [s.metadata.session_id for s in test_sessions]
            if test_sessions else None
        ),
        "test_subjects": (
            sorted({s.metadata.subject_id for s in test_sessions})
            if test_sessions else None
        ),
        "cv_session_ids": (
            [s.metadata.session_id for s in ui_cv_sessions]
            if ui_cv_sessions else None
        ),
        # Per-session info (samples, channels, etc.)
        "train_session_info": train_session_info if train_session_info else None,
        "val_session_info": val_session_info if val_session_info else None,
        # Status
        "status": "completed",
        "feature_configs": resolved_feature_configs,
        "raw_window_models": raw_models,
        "feature_based_models": feature_models,
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Clean up partial files now that final results are saved
    for partial_file in ["results_partial.json", "summary_partial.csv"]:
        p = output_dir / partial_file
        if p.exists():
            p.unlink()

    print(f"\nResults saved to: {output_dir}")
    print(f"  summary.csv           — aggregate metrics table")
    print(f"  results.json          — full per-fold results")
    print(f"  metric_comparison.png — accuracy/F1/precision/recall bars")
    print(f"  confusion_matrices.png — per-model normalized confusion matrices")
    print(f"  per_class_f1.png      — per-gesture F1 grouped by model")
    print(f"  training_time.png     — wall-clock training time")
    print(f"  confidence_analysis.png — confidence correct vs incorrect")

    return {"results": all_results, "summary": summary, "output_dir": str(output_dir)}


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive Model Comparison & Validation Pipeline"
    )
    parser.add_argument(
        "--subjects", nargs="*", default=None,
        help="Subject IDs to include (default: all)"
    )
    parser.add_argument(
        "--models", nargs="*", default=None,
        choices=ALL_MODELS,
        help=f"Models to compare (default: all). Choices: {ALL_MODELS}"
    )
    parser.add_argument(
        "--cv-folds", type=int, default=5,
        help="Number of cross-validation folds (default: 5)"
    )
    parser.add_argument(
        "--window-size", type=int, default=200,
        help="Window size in ms (default: 200)"
    )
    parser.add_argument(
        "--window-stride", type=int, default=50,
        help="Window stride in ms (default: 50)"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: performance_assessment/results/<timestamp>)"
    )
    parser.add_argument(
        "--validate-subjects", nargs="*", default=None,
        help="Subject IDs whose recordings will be used as the validation set (hold-out)."
    )
    parser.add_argument(
        "--validate-sessions", nargs="*", default=None,
        help="Session IDs (recording IDs) to use as the validation set (hold-out)."
    )
    parser.add_argument(
        "-i", "--interactive", action="store_true",
        help="Launch an interactive menu to pick train/validation sessions before running."
    )

    args = parser.parse_args()
    out = Path(args.output_dir) if args.output_dir else None

    run_comparison(
        subject_ids=args.subjects,
        model_types=args.models,
        cv_folds=args.cv_folds,
        window_size_ms=args.window_size,
        window_stride_ms=args.window_stride,
        output_dir=out,
        validate_subject_ids=args.validate_subjects,
        validate_session_ids=args.validate_sessions,
        interactive=args.interactive,
    )


if __name__ == "__main__":
    main()
