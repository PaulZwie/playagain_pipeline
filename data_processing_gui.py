import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import numpy as np
import scipy.io as sio
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.widgets import SpanSelector

class RMS_GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("EMG Processing & Slicing Tool")
        self.root.geometry("1000x800")

        self.df = None
        self.filepath = None
        self.fs = 2000  # Default sampling rate, maybe make adjustable
        self.span = None

        # UI Layout
        self.create_widgets()

    def create_widgets(self):
        # Top container for all control rows
        top_control_frame = ttk.Frame(self.root, padding="5")
        top_control_frame.pack(side=tk.TOP, fill=tk.X)

        # === ROW 1: Load/Reset and Sampling Rate ===
        row1_frame = ttk.Frame(top_control_frame)
        row1_frame.pack(side=tk.TOP, fill=tk.X, pady=2)

        ttk.Button(row1_frame, text="Load CSV/MAT", command=self.load_file).pack(side=tk.LEFT, padx=5)
        ttk.Button(row1_frame, text="Reset Data", command=self.reset_data).pack(side=tk.LEFT, padx=5)

        ttk.Label(row1_frame, text="Fs (Hz):").pack(side=tk.LEFT, padx=(10, 2))
        self.fs_var = tk.StringVar(value="2000")
        ttk.Entry(row1_frame, textvariable=self.fs_var, width=8).pack(side=tk.LEFT)

        # === ROW 2: Bandpass Filter ===
        row2_frame = ttk.Frame(top_control_frame)
        row2_frame.pack(side=tk.TOP, fill=tk.X, pady=2)

        ttk.Label(row2_frame, text="Lowcut (Hz):").pack(side=tk.LEFT, padx=(10, 2))
        self.lowcut_var = tk.StringVar(value="20")
        ttk.Entry(row2_frame, textvariable=self.lowcut_var, width=5).pack(side=tk.LEFT)

        ttk.Label(row2_frame, text="Highcut (Hz):").pack(side=tk.LEFT, padx=(5, 2))
        self.highcut_var = tk.StringVar(value="450")
        ttk.Entry(row2_frame, textvariable=self.highcut_var, width=5).pack(side=tk.LEFT)

        ttk.Button(row2_frame, text="Apply Filter", command=self.apply_filter).pack(side=tk.LEFT, padx=10)

        # === ROW 3: Outlier Cleaning and RMS ===
        row3_frame = ttk.Frame(top_control_frame)
        row3_frame.pack(side=tk.TOP, fill=tk.X, pady=2)

        ttk.Label(row3_frame, text="Outlier (Std):").pack(side=tk.LEFT, padx=(10, 2))
        self.outlier_std_var = tk.StringVar(value="5")
        ttk.Entry(row3_frame, textvariable=self.outlier_std_var, width=3).pack(side=tk.LEFT)
        ttk.Button(row3_frame, text="Clean Outliers", command=self.remove_outliers).pack(side=tk.LEFT, padx=5)

        ttk.Label(row3_frame, text="RMS Window (ms):").pack(side=tk.LEFT, padx=(10, 2))
        self.rms_window_var = tk.StringVar(value="250")
        ttk.Entry(row3_frame, textvariable=self.rms_window_var, width=5).pack(side=tk.LEFT)

        ttk.Button(row3_frame, text="Calc RMS", command=self.calc_rms).pack(side=tk.LEFT, padx=10)
        ttk.Button(row3_frame, text="Clear RMS", command=self.clear_rms).pack(side=tk.LEFT, padx=5)

        # === ROW 4: Moving Average ===
        row4_frame = ttk.Frame(top_control_frame)
        row4_frame.pack(side=tk.TOP, fill=tk.X, pady=2)

        ttk.Label(row4_frame, text="MA Window (ms):").pack(side=tk.LEFT, padx=(10, 2))
        self.ma_window_var = tk.StringVar(value="50")
        ttk.Entry(row4_frame, textvariable=self.ma_window_var, width=5).pack(side=tk.LEFT)
        ttk.Button(row4_frame, text="Apply MA", command=self.apply_moving_average).pack(side=tk.LEFT, padx=5)

        # Slicing
        slice_frame = ttk.Frame(self.root, padding="5")
        slice_frame.pack(side=tk.BOTTOM, fill=tk.X)

        ttk.Label(slice_frame, text="Start (s):").pack(side=tk.LEFT, padx=5)
        self.start_var = tk.StringVar(value="0")
        self.start_entry = ttk.Entry(slice_frame, textvariable=self.start_var, width=10)
        self.start_entry.pack(side=tk.LEFT)

        ttk.Label(slice_frame, text="End (s):").pack(side=tk.LEFT, padx=5)
        self.end_var = tk.StringVar(value="10")
        self.end_entry = ttk.Entry(slice_frame, textvariable=self.end_var, width=10)
        self.end_entry.pack(side=tk.LEFT)

        ttk.Button(slice_frame, text="Slice & Save", command=self.slice_and_save).pack(side=tk.LEFT, padx=10)

        # Plot Area
        self.plot_frame = ttk.Frame(self.root)
        self.plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.toolbar = NavigationToolbar2Tk(self.canvas, self.plot_frame)
        self.toolbar.update()

        # Toggle Slice Mode
        self.slice_mode = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.plot_frame, text="Enable Slice Selector", variable=self.slice_mode, command=self.toggle_selector).pack(side=tk.TOP)

    def toggle_selector(self):
        # Toggle the interactive span selector. Ensure we only have one active selector.
        if self.slice_mode.get():
            # If an existing selector is present, disconnect it first
            if getattr(self, 'span', None) is not None:
                try:
                    self.span.disconnect_events()
                except Exception:
                    pass
                self.span = None
            # Create a new interactive selector
            self.span = SpanSelector(self.ax, self.onselect, 'horizontal', useblit=True,
                                     props=dict(alpha=0.5, facecolor='red'), interactive=True)
        else:
            # Disable selector: disconnect and remove
            if getattr(self, 'span', None) is not None:
                try:
                    self.span.disconnect_events()
                except Exception:
                    pass
                self.span = None
            # Redraw plot to clear any selection artists
            self.plot_data()

    def onselect(self, xmin, xmax):
        # Ensure xmin <= xmax
        if xmin > xmax:
            xmin, xmax = xmax, xmin

        # Clamp to data time axis if present
        try:
            time_col = [c for c in self.df.columns if 'time' in c.lower()][0]
            tmin = float(self.df[time_col].min())
            tmax = float(self.df[time_col].max())
            xmin_clamped = max(tmin, min(tmax, xmin))
            xmax_clamped = max(tmin, min(tmax, xmax))
        except Exception:
            xmin_clamped, xmax_clamped = xmin, xmax

        # Update the entry fields (seconds). Use reasonable formatting.
        formatted_start = f"{xmin_clamped:.3f}"
        formatted_end = f"{xmax_clamped:.3f}"
        self.start_var.set(formatted_start)
        self.end_var.set(formatted_end)
        # Also explicitly update the Entry widgets to ensure visible change
        try:
            self.start_entry.delete(0, tk.END)
            self.start_entry.insert(0, formatted_start)
            self.end_entry.delete(0, tk.END)
            self.end_entry.insert(0, formatted_end)
        except Exception:
            pass
        # Force immediate UI update
        try:
            self.root.update()
        except Exception:
            pass

    def load_file(self):
        filetypes = (("All files", "*.*"), ("CSV files", "*.csv"), ("MAT files", "*.mat"))
        filename = filedialog.askopenfilename(title="Open Data File", initialdir="/Users/paul/Coding_Projects", filetypes=filetypes)
        if filename:
            self.filepath = filename
            try:
                if filename.endswith('.csv'):
                    self.df = pd.read_csv(filename)
                elif filename.endswith('.mat'):
                    mat = sio.loadmat(filename)
                    # Try to find EMG key
                    keys = [k for k in mat.keys() if not k.startswith('__')]
                    if 'EMG' in keys:
                        data = mat['EMG']
                        if data.shape[0] < data.shape[1]: # Assume channels x samples
                             data = data.T
                        self.df = pd.DataFrame(data, columns=[f'ch{i+1}' for i in range(data.shape[1])])
                    else:
                        # Fallback: take the largest array
                        largest_key = max(keys, key=lambda k: mat[k].size if hasattr(mat[k], 'size') else 0)
                        data = mat[largest_key]
                        if hasattr(data, 'shape') and len(data.shape) == 2:
                             if data.shape[0] < data.shape[1]:
                                 data = data.T
                             self.df = pd.DataFrame(data, columns=[f'ch{i+1}' for i in range(data.shape[1])])
                        else:
                            messagebox.showerror("Error", "Could not interpret MAT file structure.")
                            return

                # Identify numeric columns for plotting
                self.numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()

                # Create time axis if not exists
                if 'time' not in [c.lower() for c in self.df.columns]:
                    self.df['Time'] = np.arange(len(self.df)) / float(self.fs_var.get())

                self.plot_data()
                self.update_slice_ranges()

            except Exception as e:
                messagebox.showerror("Error", f"Failed to load file: {e}")

    def plot_data(self):
        if self.df is None:
            return

        # Clear axis but maybe keep reference if selector needs it?
        # Actually clearing axis destroys selector artists usually.
        self.ax.clear()

        # Determine what to plot
        # For simplicity, plot the first few channels or the summed signal if available
        # But user wants to process.

        # Check if we have processed data
        plot_cols = []
        if 'RMS' in self.df.columns:
            plot_cols = ['RMS']
        else:
            # Plot a subset of channels to avoid clutter
            # Filter out Time column
            cols_to_plot = [c for c in self.numeric_cols if 'time' not in c.lower()]
            plot_cols = cols_to_plot[:5] # Plot max 5 channels initially

        time_col = [c for c in self.df.columns if 'time' in c.lower()][0]

        for col in plot_cols:
            self.ax.plot(self.df[time_col], self.df[col], label=col, alpha=0.7)

        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Amplitude")
        self.ax.legend(loc='upper right')
        self.ax.grid(True)
        self.canvas.draw()

        # Only (re)create selector if slice_mode is enabled and we don't already have one
        if self.slice_mode.get():
            if getattr(self, 'span', None) is None:
                try:
                    self.span = SpanSelector(self.ax, self.onselect, 'horizontal', useblit=True,
                                             props=dict(alpha=0.5, facecolor='red'), interactive=True)
                except Exception:
                    # If selector creation fails, avoid crashing the GUI
                    self.span = None

    def update_slice_ranges(self):
        if self.df is None:
            return
        time_col = [c for c in self.df.columns if 'time' in c.lower()][0]
        max_time = self.df[time_col].max()
        self.end_var.set(f"{max_time:.2f}")

    def reset_data(self):
        if self.filepath:
            # Re-run load_file logic with current filepath
            filename = self.filepath
            try:
                if filename.endswith('.csv'):
                    self.df = pd.read_csv(filename)
                elif filename.endswith('.mat'):
                    mat = sio.loadmat(filename)
                    keys = [k for k in mat.keys() if not k.startswith('__')]
                    if 'EMG' in keys:
                        data = mat['EMG']
                        if data.shape[0] < data.shape[1]: data = data.T
                        self.df = pd.DataFrame(data, columns=[f'ch{i+1}' for i in range(data.shape[1])])
                    else:
                        largest_key = max(keys, key=lambda k: mat[k].size if hasattr(mat[k], 'size') else 0)
                        data = mat[largest_key]
                        if data.shape[0] < data.shape[1]: data = data.T
                        self.df = pd.DataFrame(data, columns=[f'ch{i+1}' for i in range(data.shape[1])])

                self.numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
                self.df['Time'] = np.arange(len(self.df)) / float(self.fs_var.get())
                self.plot_data()
                messagebox.showinfo("Success", "Data Reset to Original")
            except Exception as e:
                messagebox.showerror("Error", f"Reset failed: {e}")

    def apply_filter(self):
        if self.df is None:
            return

        try:
            fs = float(self.fs_var.get())
            low = float(self.lowcut_var.get())
            high = float(self.highcut_var.get())

            # Bandpass Filter
            nyq = 0.5 * fs
            low_norm = low / nyq
            high_norm = high / nyq
            # Explicitly specify output='ba' to avoid ambiguity warnings
            b, a = butter(4, [low_norm, high_norm], btype='band', output='ba')

            cols_to_filter = [c for c in self.numeric_cols if 'time' not in c.lower()]

            for col in cols_to_filter:
                self.df[col] = filtfilt(b, a, self.df[col])

            self.plot_data()
            messagebox.showinfo("Success", "Filter Applied")

        except Exception as e:
            messagebox.showerror("Error", f"Filter error: {e}")

    def apply_moving_average(self):
        """
        Apply a centered moving average (rolling mean) to all numeric channels (excluding time).
        The window is specified in milliseconds in `self.ma_window_var` and converted to samples
        using the sampling rate in `self.fs_var`.
        Edge NaNs produced by the centered window are filled by nearest valid values.
        """
        if self.df is None:
            return
        try:
            window_ms = float(self.ma_window_var.get())
            fs = float(self.fs_var.get())
            window_samples = int(max(1, (window_ms/1000.0) * fs))

            cols_to_smooth = [c for c in self.numeric_cols if 'time' not in c.lower()]
            if len(cols_to_smooth) == 0:
                messagebox.showwarning("Warning", "No numeric channels found to apply MA.")
                return

            # Use pandas rolling mean (centered)
            rolled = self.df[cols_to_smooth].rolling(window=window_samples, center=True).mean()
            # Fill edge NaNs by nearest valid values (backfill then forward-fill)
            rolled = rolled.bfill().ffill()

            # Replace original columns with smoothed data
            for col in cols_to_smooth:
                self.df[col] = rolled[col]

            self.plot_data()
            messagebox.showinfo("Success", f"Applied moving average with window {window_ms} ms ({window_samples} samples)")
        except Exception as e:
            messagebox.showerror("Error", f"Moving average error: {e}")

    def remove_outliers(self):
        if self.df is None:
            return

        try:
            threshold = float(self.outlier_std_var.get())
            cols_to_clean = [c for c in self.numeric_cols if 'time' not in c.lower()]

            for col in cols_to_clean:
                data = self.df[col]
                mean = data.mean()
                std = data.std()

                # Clip values exceeding mean +/- threshold * std
                lower = mean - threshold * std
                upper = mean + threshold * std

                # Count how many were affected (optional but nice)
                outliers_count = ((data < lower) | (data > upper)).sum()
                if outliers_count > 0:
                    self.df[col] = data.clip(lower, upper)

            self.plot_data()
            messagebox.showinfo("Success", f"Outliers clipped at {threshold} Std Dev for all channels.")

        except Exception as e:
            messagebox.showerror("Error", f"Outlier correction error: {e}")

    def calc_rms(self):
        if self.df is None:
            return

        try:
            window_ms = float(self.rms_window_var.get())
            fs = float(self.fs_var.get())
            window_samples = int((window_ms / 1000) * fs)

            cols_to_rms = [c for c in self.numeric_cols if 'time' not in c.lower()]

            # Calculate RMS for each channel? Or summed?
            # User notebook used summed. Let's offer summed RMS for simplicity first, or average RMS.
            # "compute RMS of summed channels" in notebook.

            # Sum channels first?
            # Or RMS of each then valid?

            # Let's do RMS of each channel then sum? Or sum then RMS?
            # Notebook: df['sum_channels'] = df[channels].sum(axis=1) -> rms_sum

            # Let's create 'RMS' column which is the RMS of the sum.
            sum_signal = self.df[cols_to_rms].sum(axis=1)

            # Rolling RMS
            self.df['RMS'] = sum_signal.rolling(window=window_samples, center=True).apply(lambda x: np.sqrt(np.mean(x**2)))

            self.plot_data()

        except Exception as e:
            messagebox.showerror("Error", f"RMS error: {e}")

    def clear_rms(self):
        if self.df is not None and 'RMS' in self.df.columns:
            self.df.drop(columns=['RMS'], inplace=True)
            self.plot_data()
            messagebox.showinfo("Success", "RMS Calculation Removed")

    def slice_and_save(self):
        if self.df is None:
            return

        try:
            start = float(self.start_var.get())
            end = float(self.end_var.get())

            time_col = [c for c in self.df.columns if 'time' in c.lower()][0]

            mask = (self.df[time_col] >= start) & (self.df[time_col] <= end)
            sliced_df = self.df[mask]

            save_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
            if save_path:
                sliced_df.to_csv(save_path, index=False)
                messagebox.showinfo("Success", f"Saved sliced data to {save_path}")

        except Exception as e:
            messagebox.showerror("Error", f"Slice error: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = RMS_GUI(root)
    root.mainloop()
