#!/usr/bin/env python3
"""
convert_csv_to_npy.py
─────────────────────
One-time conversion of all Quattrocento CSV recordings to NumPy binary
format (.npy).  Run this once on your data directory; afterwards the
loader reads only the .npy files and never touches the CSVs again.

Usage
─────
    python convert_csv_to_npy.py /path/to/quattrocento/root

Options
    --workers N   Number of parallel worker processes (default: all CPU cores)
    --dry-run     Print what would be converted without writing anything
    --force       Re-convert even if a .npy file already exists
    --delete-csv  Delete the original CSV after successful conversion
                  (IRREVERSIBLE — make sure you have a backup first!)

What it does
────────────
For every VHI_Recording_*.csv found under the root directory:
  1. Reads the first row to detect whether a header line is present.
  2. Parses the CSV with numpy (same logic as the loader).
  3. Saves a float32 .npy file next to the CSV with the same stem.

The loader will then find the .npy file and use it directly via
np.load(..., mmap_mode='r'), which is essentially instant.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Worker (runs in a subprocess — no Qt imports here)
# ---------------------------------------------------------------------------

def _is_float_token(token: str) -> bool:
    try:
        float(token)
        return True
    except ValueError:
        return False


def _detect_header_rows(csv_path: Path) -> int:
    """Return 0 if the first line is numeric data, 1 if it is a header."""
    with open(csv_path, "r") as fh:
        first = fh.readline().strip()
    if not first:
        return 0
    tokens = [t.strip().strip('"').strip("'") for t in first.split(",")]
    non_empty = [t for t in tokens if t]
    if not non_empty:
        return 0
    return 0 if all(_is_float_token(t) for t in non_empty) else 1


def convert_one(args: Tuple[Path, bool, bool]) -> Tuple[Path, str]:
    """
    Convert a single CSV to .npy.
    Returns (csv_path, status_string).
    Called in a worker process.
    """
    import numpy as np

    csv_path, force, delete_csv = args
    npy_path = csv_path.with_suffix(".npy")

    if npy_path.exists() and not force:
        return csv_path, "skipped (already exists)"

    try:
        header_rows = _detect_header_rows(csv_path)
        data = np.loadtxt(
            str(csv_path),
            delimiter=",",
            dtype=np.float32,
            skiprows=header_rows,
        )
        if data.ndim == 1:
            data = data.reshape(1, -1)

        np.save(str(npy_path), data)

        if delete_csv:
            csv_path.unlink()
            return csv_path, f"converted + CSV deleted  [{data.shape}]"

        return csv_path, f"converted  [{data.shape}]"

    except Exception as exc:
        # Remove partial .npy so the loader doesn't pick up a corrupt file
        if npy_path.exists():
            try:
                npy_path.unlink()
            except OSError:
                pass
        return csv_path, f"ERROR: {exc}"


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def find_csv_files(root: Path) -> List[Path]:
    """Recursively find all VHI_Recording_*.csv files under root."""
    return sorted(root.rglob("VHI_Recording_*.csv"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert Quattrocento CSV recordings to .npy binary format."
    )
    parser.add_argument("root", type=Path, help="Root data directory to scan")
    parser.add_argument(
        "--workers", type=int, default=os.cpu_count() or 1,
        help="Number of parallel worker processes (default: all CPU cores)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print files that would be converted without writing anything",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-convert even if a .npy already exists",
    )
    parser.add_argument(
        "--delete-csv", action="store_true",
        help="Delete original CSV after successful conversion (IRREVERSIBLE)",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    if not root.is_dir():
        print(f"ERROR: '{root}' is not a directory.", file=sys.stderr)
        return 1

    csv_files = find_csv_files(root)
    if not csv_files:
        print(f"No VHI_Recording_*.csv files found under {root}")
        return 0

    # Filter out files that already have a fresh .npy (unless --force)
    to_convert = []
    already_done = 0
    for csv in csv_files:
        npy = csv.with_suffix(".npy")
        if npy.exists() and not args.force:
            already_done += 1
        else:
            to_convert.append(csv)

    print(f"Found {len(csv_files)} CSV file(s) under {root}")
    if already_done:
        print(f"  {already_done} already converted (use --force to redo)")
    print(f"  {len(to_convert)} to convert  |  workers={args.workers}")

    if args.delete_csv:
        print("\n  ⚠  --delete-csv is set. Original CSVs will be DELETED after conversion.")
        confirm = input("  Type 'yes' to continue: ").strip().lower()
        if confirm != "yes":
            print("Aborted.")
            return 0

    if args.dry_run:
        print("\nDry run — files that would be converted:")
        for csv in to_convert:
            print(f"  {csv.relative_to(root)}")
        return 0

    if not to_convert:
        print("Nothing to do.")
        return 0

    # ── Parallel conversion ──────────────────────────────────────────────────
    t0 = time.time()
    worker_args = [(p, args.force, args.delete_csv) for p in to_convert]

    errors: List[str] = []
    done = 0

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(convert_one, a): a[0] for a in worker_args}
        for future in as_completed(futures):
            csv_path, status = future.result()
            done += 1
            rel = csv_path.relative_to(root)
            tag = "✗" if status.startswith("ERROR") else "✓"
            print(f"  [{done:>{len(str(len(to_convert)))}}/{len(to_convert)}] {tag}  {rel}  —  {status}")
            if status.startswith("ERROR"):
                errors.append(f"{rel}: {status}")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s  |  {done - len(errors)} converted"
          f"  |  {len(errors)} error(s)")

    if errors:
        print("\nErrors:")
        for e in errors:
            print(f"  {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
