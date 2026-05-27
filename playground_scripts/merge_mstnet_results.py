"""
merge_mstnet_results.py
-----------------------
Merges a mstnet-only evaluation run into an existing (incomplete) run.

Usage:
    python merge_mstnet_results.py \\
        --base-dir   <path/to/original_run>   \\
        --new-dir    <path/to/mstnet_only_run> \\
        --out-dir    <path/to/merged_output>   \\  (optional; defaults to base-dir)

The script merges these four files:
    results.csv        — appends new mstnet rows, deduplicates by fold_id+model_type
    per_class_f1.csv   — appends new mstnet rows, deduplicates by fold_id+model_type+class
    results.json       — merges folds list, recomputes mstnet aggregate & aggregate_confusion,
                         updates finished_at, sets cancelled=False
    session_index.json — union of records (deduped by subject_id+session_id)

The environment.json and experiment.json from the base run are kept as-is.
"""

import argparse
import json
import copy
import math
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def save_json(obj, path: Path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"  wrote {path}")


def save_csv(df: pd.DataFrame, path: Path):
    df.to_csv(path, index=False)
    print(f"  wrote {path}")


# ---------------------------------------------------------------------------
# aggregate helpers  (replicates what the pipeline computes)
# ---------------------------------------------------------------------------

def compute_aggregate(folds: list, model_type: str) -> dict:
    """Re-compute the aggregate block for one model from its fold list."""
    model_folds = [f for f in folds if f["model_type"] == model_type]
    if not model_folds:
        return {}

    accuracies = [f["accuracy"] for f in model_folds]
    macro_f1s  = [f["macro_f1"]  for f in model_folds]
    train_secs = [f["train_seconds"] for f in model_folds]

    # per-class f1 mean across folds
    all_classes = set()
    for f in model_folds:
        all_classes.update(f["per_class_f1"].keys())

    per_class = {}
    for cls in sorted(all_classes):
        values = [f["per_class_f1"][cls] for f in model_folds if cls in f["per_class_f1"]]
        per_class[cls] = float(np.mean(values)) if values else 0.0

    return {
        "n_folds":             len(model_folds),
        "accuracy_mean":       float(np.mean(accuracies)),
        "accuracy_std":        float(np.std(accuracies, ddof=0)),
        "macro_f1_mean":       float(np.mean(macro_f1s)),
        "macro_f1_std":        float(np.std(macro_f1s, ddof=0)),
        "train_seconds_mean":  float(np.mean(train_secs)),
        "per_class_f1":        per_class,
    }


def compute_aggregate_confusion(folds: list, model_type: str):
    """Sum confusion matrices across all folds for one model."""
    model_folds = [f for f in folds if f["model_type"] == model_type]
    if not model_folds:
        return None

    n = len(model_folds[0]["confusion"])
    total = [[0] * n for _ in range(n)]
    for fold in model_folds:
        for i, row in enumerate(fold["confusion"]):
            for j, val in enumerate(row):
                total[i][j] += val
    return total


# ---------------------------------------------------------------------------
# main merge logic
# ---------------------------------------------------------------------------

def merge(base_dir: Path, new_dir: Path, out_dir: Path):
    print(f"\nBase run : {base_dir}")
    print(f"New run  : {new_dir}")
    print(f"Output   : {out_dir}\n")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. results.csv
    # ------------------------------------------------------------------
    print("--- results.csv ---")
    base_csv = pd.read_csv(base_dir / "results.csv")
    new_csv  = pd.read_csv(new_dir  / "results.csv")

    new_mst = new_csv[new_csv["model_type"] == "mstnet"].copy()
    print(f"  base rows: {len(base_csv)}  |  new mstnet rows: {len(new_mst)}")

    # Remove any partial mstnet rows from base, then append new ones
    base_no_mst = base_csv[base_csv["model_type"] != "mstnet"]
    merged_csv  = pd.concat([base_no_mst, new_mst], ignore_index=True)
    merged_csv  = merged_csv.drop_duplicates(subset=["fold_id", "model_type"])
    print(f"  merged rows: {len(merged_csv)}")
    save_csv(merged_csv, out_dir / "results.csv")

    # ------------------------------------------------------------------
    # 2. per_class_f1.csv
    # ------------------------------------------------------------------
    print("\n--- per_class_f1.csv ---")
    base_pc = pd.read_csv(base_dir / "per_class_f1.csv")
    new_pc  = pd.read_csv(new_dir  / "per_class_f1.csv")

    new_mst_pc = new_pc[new_pc["model_type"] == "mstnet"].copy()
    print(f"  base rows: {len(base_pc)}  |  new mstnet rows: {len(new_mst_pc)}")

    base_no_mst_pc = base_pc[base_pc["model_type"] != "mstnet"]
    merged_pc      = pd.concat([base_no_mst_pc, new_mst_pc], ignore_index=True)
    merged_pc      = merged_pc.drop_duplicates(subset=["fold_id", "model_type", "class"])
    print(f"  merged rows: {len(merged_pc)}")
    save_csv(merged_pc, out_dir / "per_class_f1.csv")

    # ------------------------------------------------------------------
    # 3. results.json
    # ------------------------------------------------------------------
    print("\n--- results.json ---")
    base_rj = load_json(base_dir / "results.json")
    new_rj  = load_json(new_dir  / "results.json")

    # Folds: remove old mstnet, add new mstnet
    base_folds_no_mst = [f for f in base_rj["folds"] if f["model_type"] != "mstnet"]
    new_mst_folds     = [f for f in new_rj["folds"]  if f["model_type"] == "mstnet"]

    # Deduplicate by fold_id (keep new over old)
    seen = set()
    deduped = []
    for fold in new_mst_folds + base_folds_no_mst:
        key = (fold["fold_id"], fold["model_type"])
        if key not in seen:
            seen.add(key)
            deduped.append(fold)

    merged_rj = copy.deepcopy(base_rj)
    merged_rj["folds"] = deduped

    print(f"  base folds: {len(base_rj['folds'])}  |  new mstnet folds: {len(new_mst_folds)}")
    print(f"  merged folds: {len(merged_rj['folds'])}")

    # Recompute mstnet aggregate and aggregate_confusion
    merged_rj["aggregate"]["mstnet"] = compute_aggregate(deduped, "mstnet")
    merged_rj["aggregate_confusion"]["mstnet"] = compute_aggregate_confusion(deduped, "mstnet")

    print(f"  mstnet n_folds in aggregate: {merged_rj['aggregate']['mstnet']['n_folds']}")

    # Update metadata
    merged_rj["finished_at"] = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    merged_rj["cancelled"]   = False

    save_json(merged_rj, out_dir / "results.json")

    # ------------------------------------------------------------------
    # 4. session_index.json
    # ------------------------------------------------------------------
    print("\n--- session_index.json ---")
    base_si = load_json(base_dir / "session_index.json")
    new_si  = load_json(new_dir  / "session_index.json")

    base_records = base_si.get("records", [])
    new_records  = new_si.get("records",  [])

    seen_sessions = set()
    merged_records = []
    for rec in base_records + new_records:
        key = (rec["subject_id"], rec["session_id"])
        if key not in seen_sessions:
            seen_sessions.add(key)
            merged_records.append(rec)

    merged_si = {"records": merged_records}
    print(f"  base: {len(base_records)}  |  new: {len(new_records)}  |  merged: {len(merged_records)}")
    save_json(merged_si, out_dir / "session_index.json")

    # ------------------------------------------------------------------
    # 5. Copy static files unchanged
    # ------------------------------------------------------------------
    print("\n--- copying static files ---")
    for fname in ("environment.json", "experiment.json"):
        src = base_dir / fname
        dst = out_dir  / fname
        if src.exists() and src.resolve() != dst.resolve():
            import shutil
            shutil.copy2(src, dst)
            print(f"  copied {fname}")

    print("\nDone. Verify with:")
    print(f"  python -c \"import pandas as pd, json; df=pd.read_csv('{out_dir}/results.csv'); print(df.groupby('model_type').size())\"")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-dir", required=True,  type=Path, help="Directory of the original (incomplete) run")
    parser.add_argument("--new-dir",  required=True,  type=Path, help="Directory of the mstnet-only run")
    parser.add_argument("--out-dir",  required=False, type=Path, help="Output directory (default: same as --base-dir)")
    args = parser.parse_args()

    base_dir = args.base_dir.resolve()
    new_dir  = args.new_dir.resolve()
    out_dir  = (args.out_dir or args.base_dir).resolve()

    merge(base_dir, new_dir, out_dir)
