#!/usr/bin/env python3
"""
convert_pkl_to_csv_and_npy.py
──────────────────────────────
Convert PKL recordings to both CSV and NPY formats in one pass.
Splits EMG data into left and right channels and saves both formats.

Usage
─────
    # CLI with file paths
    python convert_pkl_to_csv_and_npy.py /path/to/file.pkl /path/to/another.pkl
    
    # CLI with directories (scans recursively)
    python convert_pkl_to_csv_and_npy.py /path/to/recordings/
    
    # GUI file picker
    python convert_pkl_to_csv_and_npy.py --gui
    
    # Full UI with batch selection
    python convert_pkl_to_csv_and_npy.py --ui

Options
    --workers N   Number of parallel worker processes (default: all CPU cores)
    --gui         Open file picker when no CLI paths provided
    --ui          Open full Tk UI for file selection and conversion
    --csv-only    Save only CSV files (skip NPY)
    --npy-only    Save only NPY files (skip CSV)

What it does
────────────
For every PKL file:
  1. Loads the recording data
  2. Splits EMG into left (192 channels) and right (192 channels)
  3. Computes RMS reference signal and processes trigger
  4. Saves CSV with time, sample rate, trigger, ref_signal, and all EMG channels
  5. Saves NPY with the EMG data as float32 array
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np


# ---------------------------------------------------------------------------
# EMG Processing Functions
# ---------------------------------------------------------------------------

def compute_ref_signal(signal: np.ndarray, fs: float) -> np.ndarray:
    """Compute RMS reference signal from multi-channel EMG."""
    window_size = int(0.2 * fs)
    num_channels, num_samples = signal.shape
    rms_signal = np.zeros((num_channels, num_samples))

    for ch in range(num_channels):
        squared = signal[ch, :] ** 2
        # Moving average (MATLAB's movmean)
        cumsum = np.cumsum(np.insert(squared, 0, 0))
        mov_avg = (cumsum[window_size:] - cumsum[:-window_size]) / window_size

        # Pad to keep same length
        pad_left = window_size // 2
        pad_right = num_samples - len(mov_avg) - pad_left
        mov_avg = np.pad(mov_avg, (pad_left, pad_right), mode='edge')

        rms_signal[ch, :] = np.sqrt(mov_avg)

    # Mean across channels
    ref_signal = np.mean(rms_signal, axis=0)
    return ref_signal


def process_trigger_and_labels(num_samples: int, trigger: np.ndarray) -> np.ndarray:
    """Resample and normalize trigger signal to 0..1."""
    trigger = np.mean(trigger, axis=0)
    
    # Resample trigger if length != EMG
    if len(trigger) != num_samples:
        trigger_processed = np.interp(
            np.linspace(0, len(trigger) - 1, num_samples),
            np.arange(len(trigger)),
            trigger
        )
    else:
        trigger_processed = trigger.copy()

    # Normalize
    t_min = np.min(trigger_processed)
    t_max = np.max(trigger_processed)
    if t_max - t_min > 0:
        trigger_processed = (trigger_processed - t_min) / (t_max - t_min)
    else:
        trigger_processed = np.zeros_like(trigger_processed)

    return trigger_processed


def estimate_gesture_count(trigger: np.ndarray, threshold: float = 0.5) -> int:
    """Count number of gestures based on trigger rising edges."""
    trigger_binary = trigger >= threshold
    rising_edges = np.diff(trigger_binary.astype(np.int8), prepend=0) == 1
    return int(np.sum(rising_edges))


# ---------------------------------------------------------------------------
# File Saving Functions
# ---------------------------------------------------------------------------

def save_csv(csv_path: Path, emg: np.ndarray, fsamp: float, 
             trigger: np.ndarray, ref_signal: np.ndarray, 
             time: np.ndarray) -> None:
    """Save EMG data to CSV with metadata columns."""
    num_channels = emg.shape[0]
    channel_headers = [f"emg_ch_{i + 1:03d}" for i in range(num_channels)]
    header = ["time_s", "sample_rate_hz", "trigger_manual", "ref_signal_rms", *channel_headers]

    data = np.column_stack([
        time,
        np.full(time.shape, fsamp, dtype=float),
        trigger,
        ref_signal,
        emg.T,
    ])
    np.savetxt(csv_path, data, delimiter=",", header=",".join(header), comments="")


def save_npy(npy_path: Path, emg: np.ndarray) -> None:
    """Save EMG data to NPY format as float32."""
    np.save(str(npy_path), emg.astype(np.float32))


# ---------------------------------------------------------------------------
# Worker Function (runs in subprocess)
# ---------------------------------------------------------------------------

def convert_one_pkl(args: Tuple[Path, bool, bool]) -> Tuple[Path, str]:
    """
    Convert a single PKL file to CSV and/or NPY.
    Returns (pkl_path, status_string).
    """
    pkl_path, csv_only, npy_only = args
    
    try:
        # Load PKL
        with open(pkl_path, "rb") as f:
            recording = pickle.load(f)

        # Extract EMG data (4Cento, 3 grids - remove last 24 channels)
        emg = recording["biosignal"][:-24]
        
        # Reshape and concatenate
        emg = np.transpose(emg, (0, 2, 1))
        new_emg = [np.concatenate(channel) for channel in emg]
        emg = np.array(new_emg)
        
        # Remove mean from each channel
        for i, ch in enumerate(emg):
            emg[i, :] = ch - np.mean(ch)
        
        # Compute time and metadata
        time = np.linspace(0, recording['recording_time'], emg.shape[1])
        num_samples = emg.shape[1]
        fsamp = num_samples / recording['recording_time']
        
        # Split into left and right
        right_emg = emg[:192, :]
        left_emg = emg[192:, :]
        
        # Process trigger
        kinematics = recording["ground_truth"]
        trigger = process_trigger_and_labels(num_samples, kinematics)
        gestures_detected = estimate_gesture_count(trigger)
        
        # Compute reference signals
        left_ref_signal = compute_ref_signal(left_emg, fsamp)
        right_ref_signal = compute_ref_signal(right_emg, fsamp)
        
        # Prepare output paths
        base_path = pkl_path.with_suffix('')
        left_csv = Path(f"{base_path}_left.csv")
        left_npy = Path(f"{base_path}_left.npy")
        right_csv = Path(f"{base_path}_right.csv")
        right_npy = Path(f"{base_path}_right.npy")
        
        # Save files
        if not npy_only:
            save_csv(left_csv, left_emg, fsamp, trigger, left_ref_signal, time)
            save_csv(right_csv, right_emg, fsamp, trigger, right_ref_signal, time)
        
        if not csv_only:
            save_npy(left_npy, left_emg)
            save_npy(right_npy, right_emg)
        
        # Build status message
        shape_info = f"L:{left_emg.shape}, R:{right_emg.shape}"
        gesture_info = f"{gestures_detected} gestures"
        format_info = []
        if not npy_only:
            format_info.append("CSV")
        if not csv_only:
            format_info.append("NPY")
        
        status = f"✓ {'+'.join(format_info)} | {shape_info} | {gesture_info}"
        
        if gestures_detected != 5:
            status += f" [WARN: expected 5]"
        
        return pkl_path, status

    except Exception as exc:
        return pkl_path, f"ERROR: {exc}"


# ---------------------------------------------------------------------------
# File Discovery
# ---------------------------------------------------------------------------

def normalize_pkl_paths(paths: List[str]) -> List[Path]:
    """Normalize and deduplicate PKL file paths."""
    normalized = []
    seen = set()

    def add_if_new(candidate: Path) -> None:
        if candidate not in seen:
            seen.add(candidate)
            normalized.append(candidate)

    for path in paths:
        if not path:
            continue
        full_path = Path(path).resolve()
        if full_path.is_dir():
            for pkl_file in sorted(full_path.rglob("*.pkl")):
                add_if_new(pkl_file)
        elif full_path.suffix.lower() == ".pkl" and full_path.exists():
            add_if_new(full_path)
    
    return sorted(normalized)


def try_pick_with_tk() -> List[Path]:
    """Open Tk file picker for PKL selection."""
    try:
        from tkinter import Tk, filedialog
        Tk().withdraw()
        selected = filedialog.askopenfilenames(
            title="Select one or more PKL files",
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")],
        )
        return normalize_pkl_paths(list(selected))
    except Exception as exc:
        print(f"[WARN] Tk file dialog unavailable: {exc}")
        return []


# ---------------------------------------------------------------------------
# UI Mode
# ---------------------------------------------------------------------------

def launch_conversion_ui() -> int:
    """Launch full Tkinter UI for batch conversion."""
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox
    except Exception as exc:
        print(f"[WARN] Tk UI unavailable: {exc}", file=sys.stderr)
        return 2

    root = tk.Tk()
    root.title("PKL to CSV/NPY Converter")
    root.geometry("900x550")

    paths = []
    csv_only_var = tk.BooleanVar(value=False)
    npy_only_var = tk.BooleanVar(value=False)

    container = tk.Frame(root, padx=10, pady=10)
    container.pack(fill=tk.BOTH, expand=True)

    title = tk.Label(container, text="Convert PKL recordings to CSV and NPY formats", 
                     font=("", 11, "bold"))
    title.pack(anchor="w")

    # Options frame
    options_frame = tk.Frame(container)
    options_frame.pack(anchor="w", pady=(8, 0))
    tk.Checkbutton(options_frame, text="CSV only", variable=csv_only_var).pack(side=tk.LEFT)
    tk.Checkbutton(options_frame, text="NPY only", variable=npy_only_var).pack(side=tk.LEFT, padx=(10, 0))

    # File list
    list_frame = tk.Frame(container)
    list_frame.pack(fill=tk.BOTH, expand=True, pady=(8, 8))

    listbox = tk.Listbox(list_frame, selectmode=tk.EXTENDED)
    listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    scrollbar = tk.Scrollbar(list_frame, orient=tk.VERTICAL, command=listbox.yview)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    listbox.config(yscrollcommand=scrollbar.set)

    status_var = tk.StringVar(value="Ready - Add PKL files or folders to begin")
    status_label = tk.Label(container, textvariable=status_var, fg="gray")
    status_label.pack(anchor="w", pady=(0, 8))

    def refresh_listbox():
        listbox.delete(0, tk.END)
        for p in paths:
            listbox.insert(tk.END, str(p))

    def add_files():
        selected = filedialog.askopenfilenames(
            title="Select one or more PKL files",
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")],
        )
        for p in normalize_pkl_paths(list(selected)):
            if p not in paths:
                paths.append(p)
        refresh_listbox()
        status_var.set(f"Selected {len(paths)} file(s)")

    def add_folder():
        folder = filedialog.askdirectory(title="Select folder containing PKL files")
        if not folder:
            return
        for p in normalize_pkl_paths([folder]):
            if p not in paths:
                paths.append(p)
        refresh_listbox()
        status_var.set(f"Selected {len(paths)} file(s)")

    def remove_selected():
        indices = list(listbox.curselection())
        for idx in reversed(indices):
            del paths[idx]
        refresh_listbox()
        status_var.set(f"Selected {len(paths)} file(s)")

    def clear_all():
        paths.clear()
        refresh_listbox()
        status_var.set("Selection cleared")

    def run_conversion():
        if not paths:
            messagebox.showwarning("No files", "Please select at least one PKL file.")
            return
        
        csv_only = csv_only_var.get()
        npy_only = npy_only_var.get()
        
        if csv_only and npy_only:
            messagebox.showwarning("Invalid options", "Cannot select both CSV-only and NPY-only.")
            return
        
        try:
            status_var.set("Converting...")
            root.update_idletasks()
            
            # Run conversion
            worker_args = [(p, csv_only, npy_only) for p in paths]
            errors = []
            
            for i, args in enumerate(worker_args, 1):
                pkl_path, result = convert_one_pkl(args)
                status_var.set(f"Converting {i}/{len(paths)}: {pkl_path.name}")
                root.update_idletasks()
                
                if result.startswith("ERROR"):
                    errors.append(f"{pkl_path.name}: {result}")
            
            if errors:
                status_var.set(f"Completed with {len(errors)} error(s)")
                messagebox.showwarning("Errors", "\n".join(errors[:5]))
            else:
                status_var.set(f"Successfully converted {len(paths)} file(s)")
                messagebox.showinfo("Done", f"Converted {len(paths)} PKL file(s) to CSV/NPY.")
        
        except Exception as exc:
            status_var.set("Conversion failed")
            messagebox.showerror("Error", str(exc))

    button_row = tk.Frame(container)
    button_row.pack(fill=tk.X)
    tk.Button(button_row, text="Add Files", command=add_files).pack(side=tk.LEFT, padx=(0, 6))
    tk.Button(button_row, text="Add Folder", command=add_folder).pack(side=tk.LEFT, padx=(0, 6))
    tk.Button(button_row, text="Remove Selected", command=remove_selected).pack(side=tk.LEFT, padx=(0, 6))
    tk.Button(button_row, text="Clear", command=clear_all).pack(side=tk.LEFT, padx=(0, 6))
    tk.Button(button_row, text="Convert", bg="#4CAF50", fg="white", 
              command=run_conversion).pack(side=tk.RIGHT)

    root.mainloop()
    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert PKL recordings to CSV and NPY formats."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help="PKL files or directories (scanned recursively)",
    )
    parser.add_argument(
        "--workers", type=int, default=os.cpu_count() or 1,
        help="Number of parallel workers (default: all CPU cores)",
    )
    parser.add_argument(
        "--gui", action="store_true",
        help="Open file picker when no CLI paths provided",
    )
    parser.add_argument(
        "--ui", action="store_true",
        help="Open full Tk UI for file selection and conversion",
    )
    parser.add_argument(
        "--csv-only", action="store_true",
        help="Save only CSV files (skip NPY)",
    )
    parser.add_argument(
        "--npy-only", action="store_true",
        help="Save only NPY files (skip CSV)",
    )
    args = parser.parse_args()

    # Launch UI mode if requested
    if args.ui:
        return launch_conversion_ui()

    # Validate options
    if args.csv_only and args.npy_only:
        print("ERROR: Cannot use both --csv-only and --npy-only", file=sys.stderr)
        return 1

    # Get PKL paths
    pkl_paths = normalize_pkl_paths(args.paths)
    
    if not pkl_paths and args.gui:
        pkl_paths = try_pick_with_tk()
    
    if not pkl_paths:
        print(
            "No PKL files selected. Provide paths via CLI, e.g.\n"
            "  python convert_pkl_to_csv_and_npy.py /path/to/file.pkl\n"
            "  python convert_pkl_to_csv_and_npy.py /path/to/recordings/\n"
            "Or use --gui for file picker or --ui for full UI.",
            file=sys.stderr,
        )
        return 2

    print(f"Found {len(pkl_paths)} PKL file(s)")
    
    format_msg = []
    if not args.npy_only:
        format_msg.append("CSV")
    if not args.csv_only:
        format_msg.append("NPY")
    print(f"Output formats: {' + '.join(format_msg)}")
    print(f"Workers: {args.workers}")

    # Parallel conversion
    t0 = time.time()
    worker_args = [(p, args.csv_only, args.npy_only) for p in pkl_paths]
    errors = []
    done = 0

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(convert_one_pkl, a): a[0] for a in worker_args}
        for future in as_completed(futures):
            pkl_path, status = future.result()
            done += 1
            print(f"[{done:>{len(str(len(pkl_paths)))}}/{len(pkl_paths)}] {pkl_path.name}")
            print(f"  {status}")
            
            if status.startswith("ERROR"):
                errors.append(f"{pkl_path.name}: {status}")

    elapsed = time.time() - t0
    print(f"\nCompleted in {elapsed:.1f}s | {done - len(errors)} successful | {len(errors)} error(s)")

    if errors:
        print("\nErrors:")
        for e in errors:
            print(f"  {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
