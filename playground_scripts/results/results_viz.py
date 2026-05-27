#!/usr/bin/env python3
"""Generate validation result visualizations from a validation output path.

This script reads `results.csv` (required) and `results.json` (optional)
from either:
  - a validation run directory, or
  - a direct path to a `results.csv` file,
and writes a set of PNG plots.

Note: a true confusion matrix cannot be reconstructed from aggregate metrics
alone; it requires per-window predictions and ground truth labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize validation run metrics")
    parser.add_argument(
        "input_path",
        type=Path,
        help="Path to a validation run directory or directly to results.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory to store PNG outputs (default: <run_dir>/plots)",
    )
    return parser.parse_args()


def _extract_held_out(fold_id: str) -> str:
    if "__" not in fold_id:
        return fold_id
    return fold_id.split("__")[-1]


def resolve_input_paths(input_path: Path) -> tuple[Path, Path, Path]:
    resolved = input_path.resolve()

    if resolved.is_file():
        if resolved.name != "results.csv":
            raise ValueError(f"Expected a results.csv file, got: {resolved}")
        csv_path = resolved
        run_dir = resolved.parent
    else:
        run_dir = resolved
        csv_path = run_dir / "results.csv"

    json_path = run_dir / "results.json"
    return run_dir, csv_path, json_path


def load_inputs(csv_path: Path, json_path: Path) -> tuple[pd.DataFrame, dict[str, Any] | None]:

    if not csv_path.exists():
        raise FileNotFoundError(f"Missing required file: {csv_path}")

    df = pd.read_csv(csv_path)
    if "fold_id" in df.columns:
        df["held_out"] = df["fold_id"].astype(str).map(_extract_held_out)

    data = None
    if json_path.exists():
        data = json.loads(json_path.read_text(encoding="utf-8"))
    return df, data


def plot_accuracy_by_subject(df: pd.DataFrame, out_dir: Path) -> None:
    piv = df.pivot(index="held_out", columns="model_type", values="accuracy")
    ax = piv.plot(kind="bar", figsize=(10, 5))
    ax.set_title("Accuracy by held-out subject")
    ax.set_xlabel("Held-out subject")
    ax.set_ylabel("Accuracy")
    ax.legend(title="Model")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "accuracy_by_subject.png", dpi=160)
    plt.close()


def plot_macro_f1_by_subject(df: pd.DataFrame, out_dir: Path) -> None:
    piv = df.pivot(index="held_out", columns="model_type", values="macro_f1")
    ax = piv.plot(kind="bar", figsize=(10, 5))
    ax.set_title("Macro-F1 by held-out subject")
    ax.set_xlabel("Held-out subject")
    ax.set_ylabel("Macro-F1")
    ax.legend(title="Model")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "macro_f1_by_subject.png", dpi=160)
    plt.close()


def plot_train_time_vs_accuracy(df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for model in sorted(df["model_type"].unique()):
        d = df[df["model_type"] == model]
        ax.scatter(d["train_seconds"], d["accuracy"], label=model, s=70, alpha=0.8)
        for _, row in d.iterrows():
            ax.annotate(str(row["held_out"]), (row["train_seconds"], row["accuracy"]), fontsize=8, alpha=0.8)
    ax.set_title("Training time vs accuracy")
    ax.set_xlabel("Train seconds")
    ax.set_ylabel("Accuracy")
    ax.grid(alpha=0.3)
    ax.legend(title="Model")
    plt.tight_layout()
    plt.savefig(out_dir / "train_time_vs_accuracy.png", dpi=160)
    plt.close()


def plot_metric_heatmap(df: pd.DataFrame, metric: str, out_dir: Path) -> None:
    piv = df.pivot(index="held_out", columns="model_type", values=metric)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    im = ax.imshow(piv.values, aspect="auto")
    ax.set_title(f"{metric} heatmap (held-out subject x model)")
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels(piv.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels(piv.index)

    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            val = piv.iat[i, j]
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=8)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(metric)
    plt.tight_layout()
    plt.savefig(out_dir / f"{metric}_heatmap.png", dpi=160)
    plt.close()


def plot_per_class_f1_heatmap(results_json: dict[str, Any], out_dir: Path) -> bool:
    rows: list[dict[str, Any]] = []
    for fold in results_json.get("folds", []):
        model = fold.get("model_type")
        held_out = _extract_held_out(str(fold.get("fold_id", "")))
        per_class = fold.get("per_class_f1") or {}
        for cls, score in per_class.items():
            rows.append(
                {
                    "model_type": model,
                    "held_out": held_out,
                    "class_name": cls,
                    "f1": float(score),
                }
            )

    if not rows:
        return False

    df = pd.DataFrame(rows)
    agg = (
        df.groupby(["class_name", "model_type"], as_index=False)["f1"]
        .mean()
        .pivot(index="class_name", columns="model_type", values="f1")
        .sort_index()
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(agg.values, aspect="auto")
    ax.set_title("Per-class F1 heatmap (mean across folds)")
    ax.set_xticks(range(len(agg.columns)))
    ax.set_xticklabels(agg.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(agg.index)))
    ax.set_yticklabels(agg.index)

    for i in range(agg.shape[0]):
        for j in range(agg.shape[1]):
            val = agg.iat[i, j]
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=8)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("F1")
    plt.tight_layout()
    plt.savefig(out_dir / "per_class_f1_heatmap.png", dpi=160)
    plt.close()
    return True


def write_confusion_note(out_dir: Path) -> None:
    note = (
        "True confusion matrix is not available from results.csv/results.json alone.\n"
        "You need per-window y_true and y_pred (or a saved confusion matrix payload)\n"
        "for each fold/model to plot confusion matrices.\n"
    )
    (out_dir / "confusion_matrix_note.txt").write_text(note, encoding="utf-8")


def main() -> int:
    args = parse_args()
    run_dir, csv_path, json_path = resolve_input_paths(args.input_path)
    out_dir = (args.out_dir or (run_dir / "plots")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df, results_json = load_inputs(csv_path, json_path)

    plot_accuracy_by_subject(df, out_dir)
    plot_macro_f1_by_subject(df, out_dir)
    plot_train_time_vs_accuracy(df, out_dir)
    plot_metric_heatmap(df, "accuracy", out_dir)
    plot_metric_heatmap(df, "macro_f1", out_dir)

    has_per_class = False
    if results_json is not None:
        has_per_class = plot_per_class_f1_heatmap(results_json, out_dir)

    write_confusion_note(out_dir)

    print(f"Saved plots to: {out_dir}")
    print("- accuracy_by_subject.png")
    print("- macro_f1_by_subject.png")
    print("- train_time_vs_accuracy.png")
    print("- accuracy_heatmap.png")
    print("- macro_f1_heatmap.png")
    if has_per_class:
        print("- per_class_f1_heatmap.png")
    print("- confusion_matrix_note.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
