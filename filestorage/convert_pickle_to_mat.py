import argparse
import pickle
import numpy as np
import os
import sys

def compute_ref_signal(signal, fs):
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


def process_trigger_and_labels(num_samples, trigger):
    """
    Resample trigger and normalize it

    Args:
        num_samples: length of the emg_signal 
        trigger (np.ndarray): Trigger-Signal, 1D-Array

    Returns:
        trigger_processed (np.ndarray): Trigger, resampled & normalized, 0..1
    """

    trigger = np.mean(trigger, axis=0)
    # Resample trigger, if length != EMG
    if len(trigger) != num_samples:
        trigger_processed = np.interp(
            np.linspace(0, len(trigger)-1, num_samples),
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


def estimate_gesture_count(trigger, threshold=0.5):
    trigger_binary = trigger >= threshold
    rising_edges = np.diff(trigger_binary.astype(np.int8), prepend=0) == 1
    return int(np.sum(rising_edges))


def save_side_csv(csv_path, emg, fsamp, trigger, ref_signal, time):
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


def _normalize_pkl_paths(paths):
    normalized = []
    seen = set()

    def add_if_new(candidate):
        if candidate not in seen:
            seen.add(candidate)
            normalized.append(candidate)

    for path in paths:
        if not path:
            continue
        full_path = os.path.abspath(os.path.expanduser(path))
        if os.path.isdir(full_path):
            for root, _, files in os.walk(full_path):
                for name in sorted(files):
                    if name.lower().endswith(".pkl"):
                        add_if_new(os.path.join(root, name))
        elif full_path.lower().endswith(".pkl") and os.path.exists(full_path):
            add_if_new(full_path)
    return sorted(normalized)


def _try_pick_with_tk():
    try:
        from tkinter import Tk, filedialog

        Tk().withdraw()
        return list(
            filedialog.askopenfilenames(
                title="Select one or more PKL files",
                filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")],
            )
        )
    except Exception as exc:
        print(f"[WARN] Tk file dialog unavailable: {exc}")
        return []


def launch_conversion_ui():
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox
    except Exception as exc:
        print(f"[WARN] Tk UI unavailable: {exc}", file=sys.stderr)
        return 2

    root = tk.Tk()
    root.title("PKL to CSV Converter")
    root.geometry("900x500")

    paths = []

    container = tk.Frame(root, padx=10, pady=10)
    container.pack(fill=tk.BOTH, expand=True)

    title = tk.Label(container, text="Select PKL files or folders and run conversion")
    title.pack(anchor="w")

    list_frame = tk.Frame(container)
    list_frame.pack(fill=tk.BOTH, expand=True, pady=(8, 8))

    listbox = tk.Listbox(list_frame, selectmode=tk.EXTENDED)
    listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    scrollbar = tk.Scrollbar(list_frame, orient=tk.VERTICAL, command=listbox.yview)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    listbox.config(yscrollcommand=scrollbar.set)

    status_var = tk.StringVar(value="Ready")
    status_label = tk.Label(container, textvariable=status_var, fg="gray")
    status_label.pack(anchor="w", pady=(0, 8))

    def refresh_listbox():
        listbox.delete(0, tk.END)
        for p in paths:
            listbox.insert(tk.END, p)

    def add_files():
        selected = filedialog.askopenfilenames(
            title="Select one or more PKL files",
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")],
        )
        for p in _normalize_pkl_paths(selected):
            if p not in paths:
                paths.append(p)
        refresh_listbox()
        status_var.set(f"Selected files: {len(paths)}")

    def add_folder():
        folder = filedialog.askdirectory(title="Select folder containing PKL files")
        if not folder:
            return
        for p in _normalize_pkl_paths([folder]):
            if p not in paths:
                paths.append(p)
        refresh_listbox()
        status_var.set(f"Selected files: {len(paths)}")

    def remove_selected():
        indices = list(listbox.curselection())
        for idx in reversed(indices):
            del paths[idx]
        refresh_listbox()
        status_var.set(f"Selected files: {len(paths)}")

    def clear_all():
        paths.clear()
        refresh_listbox()
        status_var.set("Selection cleared")

    def run_conversion():
        if not paths:
            messagebox.showwarning("No files", "Please select at least one PKL file.")
            return
        try:
            status_var.set("Converting...")
            root.update_idletasks()
            split_emg_into_left_and_right_conv_to_mat(paths)
            status_var.set("Conversion completed")
            messagebox.showinfo("Done", f"Converted {len(paths)} file(s) to CSV.")
        except Exception as exc:
            status_var.set("Conversion failed")
            messagebox.showerror("Error", str(exc))

    button_row = tk.Frame(container)
    button_row.pack(fill=tk.X)
    tk.Button(button_row, text="Add Files", command=add_files).pack(side=tk.LEFT, padx=(0, 6))
    tk.Button(button_row, text="Add Folder", command=add_folder).pack(side=tk.LEFT, padx=(0, 6))
    tk.Button(button_row, text="Remove Selected", command=remove_selected).pack(side=tk.LEFT, padx=(0, 6))
    tk.Button(button_row, text="Clear", command=clear_all).pack(side=tk.LEFT, padx=(0, 6))
    tk.Button(button_row, text="Convert to CSV", command=run_conversion).pack(side=tk.RIGHT)

    root.mainloop()
    return 0


def get_pkl_paths_from_input():
    parser = argparse.ArgumentParser(description="Convert PKL recordings to CSV files.")
    parser.add_argument(
        "paths",
        nargs="*",
        help="PKL files or directories containing PKL files (directories are scanned recursively)",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Open Tk file picker when no CLI paths are provided",
    )
    parser.add_argument(
        "--ui",
        action="store_true",
        help="Open full Tk UI for file selection and conversion",
    )
    args = parser.parse_args()

    if args.ui:
        return None

    cli_paths = _normalize_pkl_paths(args.paths)
    if cli_paths:
        return cli_paths

    if args.gui:
        gui_paths = _normalize_pkl_paths(_try_pick_with_tk())
        if gui_paths:
            return gui_paths

    return []


def split_emg_into_left_and_right_conv_to_mat(pkl_paths):
        # ------------------------
    # Loop over all files
    # ------------------------
    for pkl_path in pkl_paths:
        print(f"\n=== Processing: {os.path.basename(pkl_path)} ===")

        # Load PKL
        with open(pkl_path, "rb") as f:
            recording = pickle.load(f)

        print("Keys:", recording.keys())

        # ------------------------
        # Choose EMG Setting
        # ------------------------
        # TODO
        # 4Cento, 3 grids
        emg = recording["biosignal"][:-24]  # remove last 24 channels

        # Bracelet
        # emg = recording["biosignal"][:32]


        print(emg.shape)
        emg = np.transpose(emg, (0, 2, 1))
        new_emg = [np.concatenate(channel) for channel in emg]
        emg = np.array(new_emg)
        for i, ch in enumerate(emg):
            emg[i, :] = ch - np.mean(ch)
        time = np.linspace(0, recording['recording_time'], emg.shape[1])
        num_samples = emg.shape[1]
        # -----------------------------
        # Split into left and right
        # -----------------------------

        right_emg = emg[:192, :]
        left_emg = emg[192:]

        kinematics = recording["ground_truth"]
        trigger = process_trigger_and_labels(num_samples, kinematics)
        gestures_detected = estimate_gesture_count(trigger)
        if gestures_detected != 5:
            print(f"[WARN] Expected 5 gestures, detected {gestures_detected} (manual check recommended).")
        else:
            print("[OK] Detected 5 gestures.")
        # ------------------------
        # Compute ref_signal
        # ------------------------
        fsamp = emg.shape[1] / recording['recording_time']
        left_ref_signal = compute_ref_signal(left_emg, fsamp)
        right_ref_signal = compute_ref_signal(right_emg, fsamp)

        # ------------------------
        # Save to .csv
        # ------------------------
        left_csv_path = os.path.splitext(pkl_path)[0] + "_left.csv"
        save_side_csv(left_csv_path, left_emg, fsamp, trigger, left_ref_signal, time)

        right_csv_path = os.path.splitext(pkl_path)[0] + "_right.csv"
        save_side_csv(right_csv_path, right_emg, fsamp, trigger, right_ref_signal, time)

        print(f"Saved to {left_csv_path}")
        print(f"Saved to {right_csv_path}")

        print("\nAll files processed successfully!")
        




if __name__ == "__main__":
    pkl_paths = get_pkl_paths_from_input()
    if pkl_paths is None:
        raise SystemExit(launch_conversion_ui())
    if not pkl_paths:
        print(
            "No PKL files selected. Provide paths via CLI, e.g.\n"
            "python convert_pickle_to_mat.py /path/to/file.pkl\n"
            "Optional: add --gui to use the file picker or --ui for full UI.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    split_emg_into_left_and_right_conv_to_mat(pkl_paths)