#!/usr/bin/env python3
"""
import_recordings.py
════════════════════
One-stop import script that does three things:

  1. MUOVI NPY REPAIR
     Scans all existing pipeline sessions that have a data.csv but no data.npy
     (or a stale one) and writes the binary .npy alongside them so that the
     session-replay device and performance-review tab never have to re-parse CSV.

  2. UNITY → PIPELINE CONVERSION
     Walks /Users/paul/Coding_Projects/PlayAgain-Game2/playagain-game-2/
     RecordedData/Users, finds every EMG CSV, and converts each one into a
     proper RecordingSession stored under
         <pipeline_data_dir>/unity_sessions/<subject_id>/<session_id>/
     Unity sessions are always clearly separated from Muovi sessions so the
     two never mix in the UI or in dataset creation.

     Gesture labelling:
       • Fixed to threshold-crossing only: "fist" during activity,
         "rest" during silence — because the Unity game method uses RMS based
         activity and fist was the gesture used.

  3. QUATTROCENTO NPY CACHE
     Runs the same fast-cache conversion as convert_csv_to_npy.py but only
     for Quattrocento files inside <pipeline_data_dir>/quattrocento/ (or a
     path you supply with --quattrocento-root).

Usage
─────
    python import_recordings.py                          # use all defaults
    python import_recordings.py --dry-run                # preview only
    python import_recordings.py --unity-root /other/path
    python import_recordings.py --pipeline-root /other/pipeline/data
    python import_recordings.py --skip-muovi --skip-unity
    python import_recordings.py --workers 8

All three tasks run sequentially with a summary table at the end.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ── locate the pipeline package ─────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
for _candidate in (_SCRIPT_DIR, _SCRIPT_DIR.parent, _SCRIPT_DIR.parent.parent, _SCRIPT_DIR.parent.parent.parent):
    if (_candidate / "playagain_pipeline").is_dir():
        if str(_candidate) not in sys.path:
            sys.path.insert(0, str(_candidate))
        break


# ── defaults ────────────────────────────────────────────────────────────────
_DEFAULT_UNITY_ROOT = Path(
    "/Users/paul/Coding_Projects/PlayAgain-Game2/"
    "playagain-game-2/RecordedData/Users"
)
_DEFAULT_PIPELINE_ROOT: Optional[Path] = None  # auto-detected below
_UNITY_SESSION_SUBDIR  = "unity_sessions"       # inside pipeline sessions_dir
_UNITY_TAG             = "UNITY"                # prefix on session folder names
_MIN_ACTIVE_RATIO      = 0.005                  # ignore recordings with < 0.5% active


# ── colour helpers (no external deps) ───────────────────────────────────────
_USE_COLOR = sys.stdout.isatty()

def _c(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text

def _green(t):  return _c(t, "32")
def _yellow(t): return _c(t, "33")
def _red(t):    return _c(t, "31")
def _cyan(t):   return _c(t, "36")
def _bold(t):   return _c(t, "1")


# ══════════════════════════════════════════════════════════════════════════════
# 1. MUOVI NPY REPAIR
# ══════════════════════════════════════════════════════════════════════════════

def _repair_session_npy(session_dir: Path, force: bool) -> Tuple[Path, str]:
    """
    Worker: ensure session_dir/data.npy exists and is up-to-date.
    Called in a subprocess — no Qt, no pipeline imports.
    """
    # Ensure stdlib path is intact in subprocess (defensive; no pipeline imports here)
    import sys as _sys, os as _os
    _here = Path(__file__).resolve().parent
    for _candidate in (_here, _here.parent, _here.parent.parent, _here.parent.parent.parent):
        if (_candidate / "playagain_pipeline").is_dir():
            if str(_candidate) not in _sys.path:
                _sys.path.insert(0, str(_candidate))
            break
    csv_path = session_dir / "data.csv"
    npy_path = session_dir / "data.npy"
    meta_path = session_dir / "metadata.json"

    if not csv_path.exists() or not meta_path.exists():
        return session_dir, "skip (no data.csv or metadata.json)"

    if npy_path.exists() and not force:
        if os.path.getmtime(npy_path) >= os.path.getmtime(csv_path):
            return session_dir, "skip (npy already fresh)"

    try:
        # Detect header: first row all-numeric → no header
        with open(csv_path, "r") as fh:
            first = fh.readline().strip()
        tokens = [t.strip() for t in first.split(",")]
        skip_rows = 0 if all(_is_numeric(t) for t in tokens if t) else 1

        data = np.loadtxt(str(csv_path), delimiter=",", dtype=np.float32,
                          skiprows=skip_rows)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        np.save(str(npy_path), data)
        return session_dir, f"ok  [{data.shape[0]} samples × {data.shape[1]} ch]"
    except Exception as exc:
        return session_dir, f"ERROR: {exc}"


def _is_numeric(s: str) -> bool:
    try:
        float(s); return True
    except ValueError:
        return False


def repair_muovi_npy(sessions_root: Path, force: bool = False,
                     workers: int = 4, dry_run: bool = False) -> List[str]:
    """Find all pipeline sessions missing fresh .npy files and create them."""
    print(_bold("\n── 1. MUOVI NPY REPAIR ────────────────────────────────────────"))
    to_fix: List[Path] = []

    for subj_dir in sorted(sessions_root.iterdir()):
        if not subj_dir.is_dir():
            continue
        # Skip unity_sessions folder — those have their own npy
        if subj_dir.name == _UNITY_SESSION_SUBDIR:
            continue
        for sess_dir in sorted(subj_dir.iterdir()):
            if not sess_dir.is_dir():
                continue
            csv = sess_dir / "data.csv"
            npy = sess_dir / "data.npy"
            meta = sess_dir / "metadata.json"
            if not csv.exists() or not meta.exists():
                continue
            if npy.exists() and not force:
                if os.path.getmtime(npy) >= os.path.getmtime(csv):
                    continue
            to_fix.append(sess_dir)

    if not to_fix:
        print("  All Muovi sessions already have fresh .npy files. ✓")
        return []

    print(f"  Found {len(to_fix)} session(s) needing .npy update.")
    if dry_run:
        for p in to_fix:
            print(f"    {_yellow('DRY')} {p}")
        return []

    errors: List[str] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_repair_session_npy, p, force): p for p in to_fix}
        for i, fut in enumerate(as_completed(futs), 1):
            p, status = fut.result()
            tag = _green("✓") if "ok" in status else (_yellow("→") if "skip" in status else _red("✗"))
            print(f"  [{i:>{len(str(len(to_fix)))}}/{len(to_fix)}] {tag} {p.parent.name}/{p.name} — {status}")
            if "ERROR" in status:
                errors.append(f"{p}: {status}")

    return errors


# ══════════════════════════════════════════════════════════════════════════════
# 2. UNITY → PIPELINE CONVERSION
# ══════════════════════════════════════════════════════════════════════════════

# Patterns for auto-detecting EMG columns
_EMG_COL_PREFIXES = ("emg_ch", "ch_")
_TIMESTAMP_COL    = "Timestamp"
_ACTIVITY_COLS    = ("GroundTruthActive", "GroundTruth", "GestureActive")
_REQUESTED_COL    = "RequestedGesture"
_RMS_COL          = "RMS"


def _is_emg_csv(path: Path) -> bool:
    """Cheap check: look for EMG column names in the first header line."""
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            header = fh.readline().lower()
        return any(p in header for p in _EMG_COL_PREFIXES)
    except OSError:
        return False


def _find_unity_csvs(root: Path) -> List[Path]:
    return sorted(p for p in root.rglob("*.csv") if _is_emg_csv(p))


def _safe_name(text: str) -> str:
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', str(text).strip()).strip('._') or "unnamed"


def _guess_subject_id(csv_path: Path) -> str:
    """Walk up the directory tree to find a VP_NN-style folder name."""
    for part in reversed(csv_path.parts):
        if re.fullmatch(r"VP[_-]?\d+", part, re.IGNORECASE):
            return part.replace("-", "_")
    # fallback: use parent folder name
    return _safe_name(csv_path.parent.name) or "VP_UNITY"


def _infer_sampling_rate_csv(rows: List[List[str]],
                              col_idx: Optional[int],
                              default: float = 2000.0) -> float:
    """Estimate fs from timestamp column differences."""
    if col_idx is None or len(rows) < 4:
        return default
    try:
        ts = []
        for row in rows[1:50]:        # sample first 50 rows
            if col_idx < len(row):
                try:
                    ts.append(float(row[col_idx]))
                except ValueError:
                    pass
        if len(ts) < 2:
            return default
        dts = [abs(ts[i+1] - ts[i]) for i in range(len(ts)-1) if abs(ts[i+1]-ts[i]) > 1e-9]
        if not dts:
            return default
        median_dt = sorted(dts)[len(dts)//2]
        return max(1.0, 1.0 / median_dt)
    except Exception:
        return default


@dataclass
class _UnityConversionResult:
    csv_path: Path
    status: str           # "ok" | "skip" | "error"
    detail: str = ""
    subject_id: str = ""
    session_id: str = ""
    n_samples: int = 0
    n_trials: int = 0
    output_path: Optional[Path] = None


def _convert_unity_csv(
    csv_path: Path,
    destination_root: Path,
    force: bool,
    min_trial_sec: float = 0.2,
) -> _UnityConversionResult:
    """
    Convert a single Unity EMG CSV into a RecordingSession folder.

    This function runs inside a worker subprocess spawned by ProcessPoolExecutor.
    sys.path is NOT inherited from the parent process, so we must locate and
    insert the pipeline package root here before attempting any pipeline imports.
    """
    # ── Fix sys.path in this subprocess ──────────────────────────────────────
    import sys as _sys
    _here = Path(__file__).resolve().parent
    for _candidate in (_here, _here.parent, _here.parent.parent, _here.parent.parent.parent):
        if (_candidate / "playagain_pipeline").is_dir():
            if str(_candidate) not in _sys.path:
                _sys.path.insert(0, str(_candidate))
            break

    try:
        import pandas as pd
    except ImportError:
        return _UnityConversionResult(csv_path, "error",
                                      "pandas not installed — run: pip install pandas")

    subject_id = _guess_subject_id(csv_path)

    # Build a session_id from the stem + UNITY tag so it is unmistakeable
    stem = _safe_name(csv_path.stem)
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_id = f"{_UNITY_TAG}_{stem}"

    # Check if already converted
    out_dir = destination_root / _safe_name(subject_id) / _safe_name(session_id)
    if out_dir.exists() and not force:
        return _UnityConversionResult(csv_path, "skip",
                                      "already converted (use --force to redo)",
                                      subject_id, session_id,
                                      output_path=out_dir)

    # ── Load CSV ─────────────────────────────────────────────────────────────
    try:
        table = pd.read_csv(csv_path)
    except Exception as exc:
        return _UnityConversionResult(csv_path, "error", f"CSV read failed: {exc}")

    # Identify EMG signal columns
    emg_cols = [c for c in table.columns if c.lower().startswith("emg_ch")]
    if not emg_cols:
        emg_cols = [c for c in table.columns if c.lower().startswith("ch_")]
    if not emg_cols:
        return _UnityConversionResult(csv_path, "error", "No EMG columns found")

    # Coerce to float, interpolate NaNs
    signal_df = table[emg_cols].copy().apply(pd.to_numeric, errors="coerce")
    if signal_df.isna().any().any():
        signal_df = signal_df.interpolate(limit_direction="both").fillna(0.0)
    signal = signal_df.to_numpy(dtype=np.float32)   # (n_samples, n_ch)

    # Sampling rate
    if _TIMESTAMP_COL in table.columns:
        ts_arr = pd.to_numeric(table[_TIMESTAMP_COL], errors="coerce") \
                   .interpolate(limit_direction="both").fillna(0.0).to_numpy(float)
        dts = np.diff(ts_arr)
        dts = dts[(dts > 1e-9) & np.isfinite(dts)]
        sampling_rate = float(1.0 / np.median(dts)) if dts.size > 0 else 2000.0
    else:
        ts_arr = np.arange(len(signal), dtype=float) / 2000.0
        sampling_rate = 2000.0

    # ── Activity / label detection ────────────────────────────────────────────
    # Priority 1: explicit GroundTruthActive column
    activity: Optional[np.ndarray] = None
    activity_source = "none"
    for col in _ACTIVITY_COLS:
        if col in table.columns:
            activity = (pd.to_numeric(table[col], errors="coerce")
                        .fillna(0.0).to_numpy(float)) > 0.5
            activity_source = col
            break

    # Priority 2: RMS threshold
    if activity is None and _RMS_COL in table.columns:
        rms = (pd.to_numeric(table[_RMS_COL], errors="coerce")
               .interpolate(limit_direction="both").fillna(0.0).to_numpy(float))
        baseline = float(np.median(rms))
        mad      = float(np.median(np.abs(rms - baseline))) + 1e-12
        threshold = baseline + 2.5 * mad
        if float(np.mean(rms >= threshold)) < _MIN_ACTIVE_RATIO:
            threshold = float(np.quantile(rms, 0.90))
        activity = rms >= threshold
        activity_source = f"RMS_threshold={threshold:.4f}"

    # Priority 3: compute RMS from signal and threshold
    if activity is None:
        rms = np.sqrt(np.mean(signal ** 2, axis=1))
        baseline = float(np.median(rms))
        mad      = float(np.median(np.abs(rms - baseline))) + 1e-12
        threshold = baseline + 2.5 * mad
        if float(np.mean(rms >= threshold)) < _MIN_ACTIVE_RATIO:
            threshold = float(np.quantile(rms, 0.90))
        activity = rms >= threshold
        activity_source = f"computed_RMS_threshold={threshold:.4f}"

    # Sanity check: need at least 0.5% active
    active_ratio = float(np.mean(activity))
    if active_ratio < _MIN_ACTIVE_RATIO:
        return _UnityConversionResult(
            csv_path, "skip",
            f"only {active_ratio:.1%} active — too little signal, skipped",
            subject_id, session_id,
        )

    # Per-sample gesture label
    # Threshold-crossing only: fist vs rest (Unity uses RMS based activity corresponding to fist)
    gesture_labels = np.where(activity, "fist", "rest")
    label_source = f"fist_or_rest_via_{activity_source}"

    # ── Build spans (contiguous label regions) ────────────────────────────────
    spans: List[Dict[str, Any]] = []
    i = 0
    n = len(gesture_labels)
    while i < n:
        cur = str(gesture_labels[i])
        j = i + 1
        while j < n and str(gesture_labels[j]) == cur:
            j += 1
        duration = float(ts_arr[j-1] - ts_arr[i]) if j > i else 0.0
        if duration >= min_trial_sec:
            spans.append({
                "label":  cur,
                "start":  float(ts_arr[i]),
                "end":    float(ts_arr[j-1]),
                "start_sample": i,
                "end_sample":   j,
            })
        i = j

    gesture_classes = sorted({sp["label"] for sp in spans})
    if not gesture_classes:
        return _UnityConversionResult(csv_path, "error",
                                      "No spans found after minimum duration filter",
                                      subject_id, session_id)

    # ── Build RecordingSession ────────────────────────────────────────────────
    try:
        from playagain_pipeline.core.gesture import (
            Gesture, GestureCategory, GestureSet, create_default_gesture_set
        )
        from playagain_pipeline.core.session import RecordingSession, RecordingTrial

        base_set = create_default_gesture_set()
        gesture_set = GestureSet(name="unity_derived")
        for lbl in sorted(gesture_classes, key=lambda x: (x != "rest", x)):
            known = base_set.get_gesture(lbl)
            if known is not None:
                gesture_set.add_gesture(Gesture(
                    name=known.name, display_name=known.display_name,
                    description=known.description, category=known.category,
                    image_path=known.image_path, emoji=known.emoji,
                    label_id=known.label_id, duration_hint=known.duration_hint,
                    metadata=dict(known.metadata),
                ))
            else:
                gesture_set.add_gesture(Gesture(
                    name=lbl,
                    display_name=lbl.replace("_", " ").title(),
                    category=GestureCategory.CUSTOM,
                ))

        session = RecordingSession(
            session_id=session_id,
            subject_id=subject_id,
            device_name="UNITY",
            num_channels=signal.shape[1],
            sampling_rate=int(round(sampling_rate)),
            gesture_set=gesture_set,
            protocol_name="unity_game",
        )
        session._data_chunks = [signal]
        session._current_sample = signal.shape[0]

        for idx, sp in enumerate(spans):
            g = gesture_set.get_gesture(sp["label"])
            if g is None:
                continue
            session.trials.append(RecordingTrial(
                trial_id=idx,
                gesture_name=sp["label"],
                gesture_label=int(g.label_id),
                start_sample=int(sp["start_sample"]),
                end_sample=int(sp["end_sample"]),
                start_time=sp["start"],
                end_time=sp["end"],
                is_valid=True,
                trial_type="gesture",
            ))

        session.metadata.notes = (
            f"Imported from Unity game recording. "
            f"Label source: {label_source}. "
            f"Original CSV: {csv_path}"
        )
        session.metadata.custom_metadata = {
            "source_type":     "unity_game",
            "original_csv":    str(csv_path),
            "label_source":    label_source,
            "activity_source": activity_source,
            "active_ratio":    float(active_ratio),
            "import_time":     datetime.now().isoformat(),
            # Mark clearly so UI can distinguish
            "is_unity_recording": True,
        }

        out_dir.mkdir(parents=True, exist_ok=True)
        session.save(out_dir)

        # ── Guarantee data.npy exists for the pipeline loader ─────────────
        # session.save() may write data.csv only (depending on the version).
        # The pipeline's _prepare_session_data / npy-repair path expects a
        # data.npy file.  Write it directly so we never depend on the CSV path.
        npy_out = out_dir / "data.npy"
        if not npy_out.exists() or force:
            np.save(str(npy_out), signal)

        return _UnityConversionResult(
            csv_path, "ok",
            f"{signal.shape[0]} samples × {signal.shape[1]} ch  |  "
            f"{len(session.trials)} trials  |  gestures: {gesture_classes}",
            subject_id, session_id,
            n_samples=signal.shape[0], n_trials=len(session.trials),
            output_path=out_dir,
        )

    except ImportError as e:
        return _UnityConversionResult(csv_path, "error",
                                      f"Pipeline import failed: {e}", subject_id)
    except Exception as e:
        import traceback
        return _UnityConversionResult(csv_path, "error",
                                      f"{e}\n{traceback.format_exc()}", subject_id)


def convert_unity_recordings(
    unity_root: Path,
    pipeline_sessions_root: Path,
    force: bool = False,
    workers: int = 4,
    dry_run: bool = False,
) -> List[str]:
    """Discover and convert all Unity EMG CSVs."""
    print(_bold("\n── 2. UNITY → PIPELINE CONVERSION ────────────────────────────"))

    destination = pipeline_sessions_root / _UNITY_SESSION_SUBDIR
    destination.mkdir(parents=True, exist_ok=True)

    csv_files = _find_unity_csvs(unity_root)
    if not csv_files:
        print(f"  No EMG CSV files found under {unity_root}")
        return []

    print(f"  Found {len(csv_files)} EMG CSV file(s) under {unity_root}")
    print(f"  Output → {destination}")

    if dry_run:
        for p in csv_files:
            sid = _guess_subject_id(p)
            stem = _safe_name(p.stem)
            print(f"    {_yellow('DRY')} {sid}  ←  {p.name}")
        return []

    errors: List[str] = []
    n = len(csv_files)

    # Use a process pool BUT keep pipeline imports inside the worker
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = {
            pool.submit(_convert_unity_csv, p, destination, force): p
            for p in csv_files
        }
        for i, fut in enumerate(as_completed(futs), 1):
            r: _UnityConversionResult = fut.result()
            if r.status == "ok":
                tag = _green("✓")
            elif r.status == "skip":
                tag = _yellow("→")
            else:
                tag = _red("✗")

            label_width = len(str(n))
            print(f"  [{i:>{label_width}}/{n}] {tag} {r.subject_id} / "
                  f"{r.session_id or _safe_name(r.csv_path.stem)}")
            if r.detail:
                detail_lines = r.detail.strip().splitlines()
                for dl in detail_lines[:3]:        # cap verbose tracebacks
                    print(f"       {dl}")
            if r.status == "error":
                errors.append(f"{r.csv_path}: {r.detail[:200]}")

    return errors


# ══════════════════════════════════════════════════════════════════════════════
# 3. QUATTROCENTO NPY CACHE
# ══════════════════════════════════════════════════════════════════════════════

def _cache_quattrocento_one(args: Tuple[Path, bool]) -> Tuple[Path, str]:
    """Worker: convert one Quattrocento CSV to .npy (same logic as loader)."""
    import numpy as np

    csv_path, force = args
    npy_path = csv_path.with_suffix(".npy")

    if npy_path.exists() and not force:
        if os.path.getmtime(npy_path) >= os.path.getmtime(csv_path):
            return csv_path, "skip (fresh)"

    try:
        with open(csv_path, "r") as fh:
            first = fh.readline().strip()
        tokens = [t.strip() for t in first.split(",")]
        skip = 0 if all(_is_numeric(t) for t in tokens if t) else 1

        data = np.loadtxt(str(csv_path), delimiter=",", dtype=np.float32,
                          skiprows=skip)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        np.save(str(npy_path), data)
        return csv_path, f"ok  [{data.shape[0]} × {data.shape[1]}]"
    except Exception as exc:
        if npy_path.exists():
            try: npy_path.unlink()
            except OSError: pass
        return csv_path, f"ERROR: {exc}"


def cache_quattrocento_npy(quattrocento_root: Optional[Path],
                           force: bool = False,
                           workers: int = 4,
                           dry_run: bool = False) -> List[str]:
    """Build .npy caches for all Quattrocento CSVs."""
    print(_bold("\n── 3. QUATTROCENTO NPY CACHE ──────────────────────────────────"))

    if quattrocento_root is None or not quattrocento_root.exists():
        print("  No Quattrocento root found — skipping.")
        return []

    csv_files = sorted(quattrocento_root.rglob("VHI_Recording_*.csv"))
    if not csv_files:
        print(f"  No VHI_Recording_*.csv found under {quattrocento_root}")
        return []

    need_cache = [p for p in csv_files
                  if force or not p.with_suffix(".npy").exists()
                  or os.path.getmtime(p.with_suffix(".npy")) < os.path.getmtime(p)]

    print(f"  {len(csv_files)} CSV files  |  {len(need_cache)} need caching")

    if not need_cache:
        print("  All Quattrocento CSVs already cached. ✓")
        return []

    if dry_run:
        for p in need_cache:
            print(f"    {_yellow('DRY')} {p.name}")
        return []

    errors: List[str] = []
    n = len(need_cache)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_cache_quattrocento_one, (p, force)): p
                for p in need_cache}
        for i, fut in enumerate(as_completed(futs), 1):
            p, status = fut.result()
            tag = _green("✓") if "ok" in status else (
                  _yellow("→") if "skip" in status else _red("✗"))
            print(f"  [{i:>{len(str(n))}}/{n}] {tag} {p.name} — {status}")
            if "ERROR" in status:
                errors.append(f"{p}: {status}")

    return errors


# ══════════════════════════════════════════════════════════════════════════════
# AUTO-DETECT PIPELINE ROOT
# ══════════════════════════════════════════════════════════════════════════════

def _find_pipeline_data_dir() -> Optional[Path]:
    """Look for the pipeline data directory near the script."""
    candidates = [
        _SCRIPT_DIR / "data",
        _SCRIPT_DIR.parent / "data",
        _SCRIPT_DIR.parent.parent / "data",
        Path.home() / "playagain_data",
    ]
    for c in candidates:
        if (c / "sessions").exists() or (c / "datasets").exists():
            return c
    return None


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Import Muovi / Unity / Quattrocento recordings into the PlayAgain pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--pipeline-root", type=Path, default=None,
        help="Pipeline data directory (auto-detected if omitted)",
    )
    parser.add_argument(
        "--unity-root", type=Path, default=_DEFAULT_UNITY_ROOT,
        help=f"Unity recordings root (default: {_DEFAULT_UNITY_ROOT})",
    )
    parser.add_argument(
        "--quattrocento-root", type=Path, default=None,
        help="Quattrocento data root for NPY caching (auto-detected if omitted)",
    )
    parser.add_argument(
        "--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1),
        help="Parallel worker processes (default: CPU cores - 1)",
    )
    parser.add_argument("--force",      action="store_true",
                        help="Re-process even if output already exists")
    parser.add_argument("--dry-run",    action="store_true",
                        help="Print what would be done without writing anything")
    parser.add_argument("--skip-muovi", action="store_true",
                        help="Skip Muovi NPY repair step")
    parser.add_argument("--skip-unity", action="store_true",
                        help="Skip Unity conversion step")
    parser.add_argument("--skip-quattrocento", action="store_true",
                        help="Skip Quattrocento NPY cache step")

    args = parser.parse_args()

    # ── Resolve pipeline root ────────────────────────────────────────────────
    pipeline_root = args.pipeline_root or _find_pipeline_data_dir()
    if pipeline_root is None:
        print(_red(
            "ERROR: Could not auto-detect the pipeline data directory.\n"
            "Pass --pipeline-root /path/to/your/pipeline/data"
        ))
        return 1

    sessions_root = pipeline_root / "sessions"
    sessions_root.mkdir(parents=True, exist_ok=True)

    # Quattrocento root auto-detect
    q4_root = args.quattrocento_root
    if q4_root is None:
        for candidate in [pipeline_root / "quattrocento",
                          pipeline_root / "Quattrocento",
                          pipeline_root / "quattrocento_data"]:
            if candidate.is_dir():
                q4_root = candidate
                break

    print(_bold("═" * 60))
    print(_bold("  PlayAgain Recording Importer"))
    print(_bold("═" * 60))
    print(f"  Pipeline root : {pipeline_root}")
    print(f"  Sessions dir  : {sessions_root}")
    print(f"  Unity root    : {args.unity_root}")
    print(f"  Q4 root       : {q4_root or '(not found)'}")
    print(f"  Workers       : {args.workers}")
    print(f"  Force         : {args.force}")
    print(f"  Dry-run       : {args.dry_run}")

    t0 = time.time()
    all_errors: List[str] = []

    # ── 1. Muovi NPY repair ──────────────────────────────────────────────────
    if not args.skip_muovi:
        errs = repair_muovi_npy(sessions_root, force=args.force,
                                workers=args.workers, dry_run=args.dry_run)
        all_errors.extend(errs)

    # ── 2. Unity conversion ──────────────────────────────────────────────────
    if not args.skip_unity:
        if not args.unity_root.exists():
            print(_yellow(f"\n  Unity root not found: {args.unity_root}  — skipping Unity step."))
        else:
            errs = convert_unity_recordings(
                args.unity_root, sessions_root,
                force=args.force, workers=args.workers, dry_run=args.dry_run,
            )
            all_errors.extend(errs)

    # ── 3. Quattrocento NPY cache ────────────────────────────────────────────
    if not args.skip_quattrocento:
        errs = cache_quattrocento_npy(q4_root, force=args.force,
                                      workers=args.workers, dry_run=args.dry_run)
        all_errors.extend(errs)

    # ── Summary ──────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print(_bold(f"\n═" * 60))
    print(_bold(f"  Done in {elapsed:.1f}s"))
    if all_errors:
        print(_red(f"  {len(all_errors)} error(s):"))
        for e in all_errors[:10]:
            print(f"    {e[:120]}")
        if len(all_errors) > 10:
            print(f"    … and {len(all_errors) - 10} more")
        return 1
    else:
        print(_green("  All tasks completed without errors. ✓"))

    if not args.dry_run:
        print()
        print("  Next steps:")
        print("   • Open the main GUI → Training tab → Create Dataset")
        print(f"   • Unity sessions are in:  {sessions_root / _UNITY_SESSION_SUBDIR}")
        print("   • Unity sessions are tagged with 'UNITY_' prefix so you can")
        print("     easily include/exclude them in the dataset builder.")

    return 0


if __name__ == "__main__":
    sys.exit(main())