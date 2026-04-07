"""
Quattrocento HD-EMG data loader  (v3 – CSV input)

CSV file layout (confirmed from inspection):
    delimiter : ','
    rows      : N raw samples  (e.g. 122 881 for a ~60 s recording at 2048 Hz)
    columns   : 196
        col 0  – time_s
        col 1  – sample_rate_hz
        col 2  – trigger_manual
        col 3  – ref_signal_rms
        col 4… – emg_ch_001 … emg_ch_192   (192 EMG channels per side)

Filename pattern (unchanged from PKL era):
    VHI_Recording_<timestamp_ms>_<gesture>_<trial_label>_<side>.csv
    e.g.  VHI_Recording_20260131_105513093906_rest_trial1_left.csv

Each recording exists as a *pair* of files (_left / _right).  By default the
loader returns the LEFT-side array; pass ``side="right"`` or ``side="both"``
to change that behaviour.

Public API is intentionally identical to the old PKL-based loader so that
nothing else in the pipeline needs to change.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union, Callable

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Columns that are NOT EMG channels (always at the front of every CSV row)
_METADATA_COLS = 4          # time_s, sample_rate_hz, trigger_manual, ref_signal_rms
_COL_TIME_S        = 0
_COL_SAMPLE_RATE   = 1
_COL_TRIGGER       = 2
_COL_REF_RMS       = 3

_TASK_ALIASES: Dict[str, str] = {
    "rest": "rest", "power grasp": "power_grasp", "powergrasp": "power_grasp",
    "fist": "power_grasp", "pinch": "pinch", "tripod pinch": "tripod_pinch",
    "tripod": "tripod_pinch", "thumb": "thumb", "index": "index",
    "middle": "middle", "ring": "ring", "pinky": "pinky",
    "open hand": "open_hand", "open": "open_hand",
    "wrist flex": "wrist_flex", "wristflex": "wrist_flex",
    "wrist extend": "wrist_extend", "wristextend": "wrist_extend",
    "pronation": "pronation", "supination": "supination",
}

_DEFAULT_LABEL_ORDER = [
    "rest", "power_grasp", "pinch", "tripod_pinch",
    "thumb", "index", "middle", "ring", "pinky",
    "open_hand", "wrist_flex", "wrist_extend", "pronation", "supination",
]

# Regex for CSV filenames
# Captures: (timestamp_str, gesture_raw, trial_label, side)
_FILENAME_RE = re.compile(
    r"^VHI_Recording_((?:\d+_\d+)|\d+)_(.+?)_(trial\S+|default|mvcpowergrasp\S*|fistmvc\S*|holdduration\S*)_(left|right)\.csv$",
    re.IGNORECASE,
)


def normalize_gesture_name(task: str) -> str:
    return _TASK_ALIASES.get(task.strip().lower(), task.strip().lower().replace(" ", "_"))


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class QuattrocentoRecording:
    """Metadata for one CSV recording (one side of one trial)."""
    path: Path
    subject_id: str
    timestamp: datetime
    gesture_raw: str
    gesture: str
    trial_label: str
    task: str
    side: str                        # "left" | "right"
    use_as_classification: bool
    n_samples: int                   # raw sample count in the CSV
    n_channels: int                  # number of EMG channels (columns 4…end)
    sampling_rate: float             # Hz, read from col 1 of first row
    header_rows: int = 0             # number of non-data rows at top (e.g. named CSV header)
    # n_windows / window_size are computed lazily when load_windows() is called
    window_size: int  = 0            # filled in by load_recording()
    n_windows: int    = 0            # filled in by load_recording()
    bad_channels: List[int] = field(default_factory=list)

    def __repr__(self) -> str:
        return (
            f"QuattrocentoRecording(subj={self.subject_id!r}, "
            f"gesture={self.gesture!r}, side={self.side!r}, "
            f"n_samples={self.n_samples}, n_ch={self.n_channels}, "
            f"sr={self.sampling_rate:.0f}Hz)"
        )


# ---------------------------------------------------------------------------
# File-level loader
# ---------------------------------------------------------------------------

class QuattrocentoFileLoader:
    """Loads a single CSV file."""

    # Default windowing parameters (can be overridden in load_windows)
    DEFAULT_WINDOW_MS   = 200     # ms  → samples depend on sr
    DEFAULT_OVERLAP_PCT = 50      # %

    # ------------------------------------------------------------------
    # Filename parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_filename(fn: str) -> Optional[Tuple[datetime, str, str, str]]:
        """Return (timestamp, gesture_raw, trial_label, side) or None."""
        m = _FILENAME_RE.match(fn)
        if m is None:
            return None
        ts_str = m.group(1)
        # Timestamp may be YYYYMMDD_HHMMSSffffff or plain unix-ms.
        ts = datetime.min
        if "_" in ts_str:
            try:
                ts = datetime.strptime(ts_str, "%Y%m%d_%H%M%S%f")
            except ValueError:
                ts = datetime.min
        else:
            try:
                ts = datetime.fromtimestamp(int(ts_str) / 1000.0)
            except (ValueError, OSError):
                ts = datetime.min
        return ts, m.group(2), m.group(3), m.group(4).lower()

    # ------------------------------------------------------------------
    # Header / metadata scan (cheap — reads only the first row)
    # ------------------------------------------------------------------

    @staticmethod
    def _read_header_row(path: Path) -> Tuple[np.ndarray, int]:
        """Read first numeric row and return (row, header_rows_to_skip)."""
        with open(path, "r") as fh:
            first_line = fh.readline().strip()
            first_row = np.fromstring(first_line, dtype=np.float64, sep=",")
            if len(first_row) >= _METADATA_COLS + 1:
                return first_row, 0

            # Common CSV layout: one named header line then numeric rows.
            second_line = fh.readline().strip()
            second_row = np.fromstring(second_line, dtype=np.float64, sep=",")
            if len(second_row) >= _METADATA_COLS + 1:
                return second_row, 1

        raise ValueError(f"Could not parse numeric CSV rows: {path.name}")

    @classmethod
    def _count_rows(cls, path: Path, header_rows: int = 0) -> int:
        """Count non-empty data rows efficiently, excluding known header rows."""
        with open(path, "r") as fh:
            total = sum(1 for line in fh if line.strip())
        return max(0, total - header_rows)

    # ------------------------------------------------------------------
    # Public: load_recording  (metadata only – fast)
    # ------------------------------------------------------------------

    @classmethod
    def load_recording(
        cls,
        path: Path,
        subject_id: str,
        window_ms: int = DEFAULT_WINDOW_MS,
        overlap_pct: int = DEFAULT_OVERLAP_PCT,
        include_sample_count: bool = True,
    ) -> QuattrocentoRecording:
        """
        Parse filename + first row to build a QuattrocentoRecording header.
        Does NOT load the full data array (fast).
        """
        parsed = cls._parse_filename(path.name)
        if parsed is None:
            raise ValueError(f"Filename does not match VHI CSV pattern: {path.name}")
        ts, gesture_raw, trial_label, side = parsed

        first_row, header_rows = cls._read_header_row(path)
        if len(first_row) < _METADATA_COLS + 1:
            raise ValueError(
                f"CSV has only {len(first_row)} columns, expected ≥ {_METADATA_COLS + 1}: {path.name}"
            )

        sr          = float(first_row[_COL_SAMPLE_RATE]) if first_row[_COL_SAMPLE_RATE] > 0 else 2048.0
        n_channels  = len(first_row) - _METADATA_COLS    # EMG columns only
        n_samples = cls._count_rows(path, header_rows=header_rows) if include_sample_count else 0

        # Pre-compute windowing shape so consumers can plan memory
        ws     = max(1, int(round(sr * window_ms / 1000.0)))
        stride = max(1, int(round(ws * (1.0 - overlap_pct / 100.0))))
        n_wins = max(0, (n_samples - ws) // stride + 1) if include_sample_count else 0

        gesture = normalize_gesture_name(gesture_raw)

        return QuattrocentoRecording(
            path=path,
            subject_id=subject_id,
            timestamp=ts,
            gesture_raw=gesture_raw,
            gesture=gesture,
            trial_label=trial_label,
            task=gesture_raw,
            side=side,
            use_as_classification=True,
            n_samples=n_samples,
            n_channels=n_channels,
            sampling_rate=sr,
            header_rows=header_rows,
            window_size=ws,
            n_windows=n_wins,
            bad_channels=[],
        )

    # ------------------------------------------------------------------
    # Public: load_raw_signal  (full data – returns (n_samples, n_ch))
    # ------------------------------------------------------------------

    @classmethod
    def load_raw_signal(
        cls,
        rec: QuattrocentoRecording,
        channel_slice: Optional[slice] = None,
    ) -> np.ndarray:
        """
        Load the full CSV into a float32 array of shape (n_samples, n_channels).

        Metadata columns (time, sr, trigger, ref_rms) are stripped; only the
        EMG channel columns are returned.

        Parameters
        ----------
        rec:
            Recording descriptor (path + channel count already parsed).
        channel_slice:
            Optional slice applied to the channel axis, e.g. ``slice(0, 32)``.

        Returns
        -------
        signal : np.ndarray, shape (n_samples, n_ch_out), dtype float32
        """
        data = np.loadtxt(
            rec.path,
            delimiter=",",
            dtype=np.float32,
            skiprows=max(0, int(getattr(rec, "header_rows", 0))),
        )
        if data.ndim == 1:
            data = data.reshape(1, -1)

        # Drop metadata columns, keep EMG only
        signal = data[:, _METADATA_COLS:]

        if channel_slice is not None:
            signal = signal[:, channel_slice]

        return signal

    @classmethod
    def load_raw_with_meta(
        cls,
        rec: QuattrocentoRecording,
        channel_slice: Optional[slice] = None,
    ) -> Dict[str, np.ndarray]:
        """Load full CSV and return metadata vectors plus EMG signal."""
        data = np.loadtxt(
            rec.path,
            delimiter=",",
            dtype=np.float32,
            skiprows=max(0, int(getattr(rec, "header_rows", 0))),
        )
        if data.ndim == 1:
            data = data.reshape(1, -1)

        signal = data[:, _METADATA_COLS:]
        if channel_slice is not None:
            signal = signal[:, channel_slice]

        return {
            "time_s": data[:, _COL_TIME_S],
            "sample_rate_hz": data[:, _COL_SAMPLE_RATE],
            "trigger": data[:, _COL_TRIGGER],
            "ref_rms": data[:, _COL_REF_RMS],
            "signal": signal,
        }

    @staticmethod
    def extract_trigger_segments(
        trigger: np.ndarray,
        ref_rms: Optional[np.ndarray],
        sampling_rate: float,
        min_gap_ms: int = 250,
        use_rms_selection: bool = False,
    ) -> List[Tuple[int, int]]:
        """Return active trigger segments as [start, end) sample indices."""
        if trigger.size == 0:
            return []

        trig = np.asarray(trigger).reshape(-1)
        active = trig > 0.5
        diff = np.diff(active.astype(np.int8), prepend=0, append=0)
        starts = np.flatnonzero(diff == 1)
        ends = np.flatnonzero(diff == -1)
        segments = [(int(s), int(e)) for s, e in zip(starts, ends) if e > s]
        if not segments:
            return []

        # Merge short trigger gaps to avoid over-segmenting noisy edges.
        min_gap = max(0, int(round((min_gap_ms / 1000.0) * max(1.0, sampling_rate))))
        merged: List[Tuple[int, int]] = [segments[0]]
        for s, e in segments[1:]:
            ps, pe = merged[-1]
            if s - pe <= min_gap:
                merged[-1] = (ps, e)
            else:
                merged.append((s, e))

        if not use_rms_selection or ref_rms is None or len(merged) <= 1:
            return merged

        rms = np.asarray(ref_rms).reshape(-1)
        scored = []
        for s, e in merged:
            win = rms[s:e]
            score = float(np.nanmean(win)) if win.size else 0.0
            scored.append((score, s, e))
        scored.sort(key=lambda t: t[0], reverse=True)
        best = scored[0]
        return [(int(best[1]), int(best[2]))]

    @classmethod
    def count_trigger_segments(
        cls,
        rec: QuattrocentoRecording,
        min_gap_ms: int = 250,
    ) -> int:
        """Fast trigger count for UI previews."""
        # Read only trigger + RMS columns for speed during scans.
        meta = np.loadtxt(
            rec.path,
            delimiter=",",
            dtype=np.float32,
            skiprows=max(0, int(getattr(rec, "header_rows", 0))),
            usecols=(_COL_TRIGGER, _COL_REF_RMS),
        )
        if meta.ndim == 1:
            meta = meta.reshape(1, -1)
        return len(
            cls.extract_trigger_segments(
                trigger=meta[:, 0],
                ref_rms=meta[:, 1],
                sampling_rate=rec.sampling_rate,
                min_gap_ms=min_gap_ms,
                use_rms_selection=False,
            )
        )

    # ------------------------------------------------------------------
    # Public: load_windows  (sliding-window segmentation)
    # ------------------------------------------------------------------

    @classmethod
    def load_windows(
        cls,
        rec: QuattrocentoRecording,
        window_ms: int = DEFAULT_WINDOW_MS,
        overlap_pct: int = DEFAULT_OVERLAP_PCT,
        channel_slice: Optional[slice] = None,
        use_trigger_segmentation: bool = False,
        use_rms_selection: bool = False,
        trigger_min_gap_ms: int = 250,
    ) -> np.ndarray:
        """
        Load the CSV and return a sliding-window array.

        Parameters
        ----------
        rec:
            Recording descriptor.
        window_ms:
            Window length in milliseconds (default 200 ms).
        overlap_pct:
            Window overlap in percent (default 50 %).
        channel_slice:
            Optional channel sub-selection applied before windowing.

        Returns
        -------
        windows : np.ndarray, shape (n_windows, window_size, n_channels), dtype float32
            Consistent with the old PKL loader's output format.
        """
        if use_trigger_segmentation:
            payload = cls.load_raw_with_meta(rec, channel_slice=channel_slice)
            signal = payload["signal"]
            segments = cls.extract_trigger_segments(
                trigger=payload["trigger"],
                ref_rms=payload["ref_rms"],
                sampling_rate=rec.sampling_rate,
                min_gap_ms=trigger_min_gap_ms,
                use_rms_selection=use_rms_selection,
            )
        else:
            signal = cls.load_raw_signal(rec, channel_slice=channel_slice)
            segments = [(0, signal.shape[0])]

        n_samples, n_ch = signal.shape

        sr     = rec.sampling_rate
        ws     = max(1, int(round(sr * window_ms / 1000.0)))
        stride = max(1, int(round(ws * (1.0 - overlap_pct / 100.0))))

        windows_blocks: List[np.ndarray] = []
        for seg_start, seg_end in segments:
            seg = signal[seg_start:seg_end]
            if seg.shape[0] == 0:
                continue
            if seg.shape[0] < ws:
                pad = np.zeros((ws - seg.shape[0], n_ch), dtype=np.float32)
                windows_blocks.append(np.concatenate([seg, pad], axis=0)[np.newaxis, :, :])
                continue

            # Vectorized window extraction is much faster than Python loops.
            view = np.lib.stride_tricks.sliding_window_view(seg, ws, axis=0)
            wins = view[::stride].astype(np.float32, copy=False)
            windows_blocks.append(wins)

        if windows_blocks:
            return np.concatenate(windows_blocks, axis=0)

        if n_samples < ws:
            pad = np.zeros((ws - n_samples, n_ch), dtype=np.float32)
            return np.concatenate([signal, pad], axis=0)[np.newaxis, :, :]

        return np.empty((0, ws, n_ch), dtype=np.float32)


# ---------------------------------------------------------------------------
# Subject / directory loader
# ---------------------------------------------------------------------------

class QuattrocentoSubjectLoader:
    """
    Scan and load Quattrocento CSV recordings from a directory tree.

    Expected layout::

        <root>/VP_01/recordings/VHI_Recording_*_left.csv
                                VHI_Recording_*_right.csv
               VP_02/recordings/...

    Parameters
    ----------
    root_dir:
        Root directory containing per-subject subdirectories.
    side:
        Which electrode side to load – ``"left"``, ``"right"``; or ``"both"``.
        When ``"both"`` is chosen, left and right files are treated as
        independent recordings (their channel counts are summed only if you
        intentionally concatenate them yourself; by default each appears as a
        separate entry with its own n_channels).
    window_ms:
        Default window length in ms used for n_windows estimation and
        ``build_dataset``.
    overlap_pct:
        Default overlap percentage (0–99).
    """

    def __init__(
        self,
        root_dir: Union[str, Path],
        side: str = "left",
        window_ms: int = QuattrocentoFileLoader.DEFAULT_WINDOW_MS,
        overlap_pct: int = QuattrocentoFileLoader.DEFAULT_OVERLAP_PCT,
    ) -> None:
        self.root_dir    = Path(root_dir)
        self.side        = side.lower()
        self.window_ms   = window_ms
        self.overlap_pct = overlap_pct

        self._recordings: List[QuattrocentoRecording] = []
        self._scanned = False

    # ------------------------------------------------------------------
    # Scanning
    # ------------------------------------------------------------------

    def scan(
        self,
        verbose: bool = True,
        include_sample_counts: bool = True,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        should_cancel: Optional[Callable[[], bool]] = None,
    ) -> List[QuattrocentoRecording]:
        """Walk the directory tree and register all matching CSV files."""
        self._recordings.clear()
        errors: List[str] = []
        candidates: List[Tuple[Path, str]] = []

        for child in sorted(self.root_dir.iterdir()):
            if child.is_dir():
                rd = child / "recordings"
                if rd.is_dir():
                    candidates.append((rd, child.name))

        # Fallback: flat layout (no per-subject subdir)
        if not candidates:
            if (self.root_dir / "recordings").is_dir():
                candidates.append((self.root_dir / "recordings", self.root_dir.name))
            else:
                candidates.append((self.root_dir, self.root_dir.name))

        csv_candidates: List[Tuple[Path, str]] = []
        for rd, sid in candidates:
            for csv_path in sorted(rd.glob("VHI_Recording_*.csv")):
                name_lower = csv_path.name.lower()
                if self.side == "left" and not name_lower.endswith("_left.csv"):
                    continue
                if self.side == "right" and not name_lower.endswith("_right.csv"):
                    continue
                csv_candidates.append((csv_path, sid))

        total = len(csv_candidates)
        if progress_callback:
            progress_callback(0, max(1, total), "Scanning files...")

        for idx, (csv_path, sid) in enumerate(csv_candidates, start=1):
            if should_cancel and should_cancel():
                self._recordings.clear()
                self._scanned = False
                raise InterruptedError("Quattrocento scan cancelled")
            try:
                rec = QuattrocentoFileLoader.load_recording(
                    csv_path,
                    sid,
                    window_ms=self.window_ms,
                    overlap_pct=self.overlap_pct,
                    include_sample_count=include_sample_counts,
                )
                self._recordings.append(rec)
            except Exception as e:
                errors.append(f"  {csv_path.name}: {e}")
            if progress_callback:
                progress_callback(idx, max(1, total), f"{sid}/{csv_path.name}")

        self._scanned = True
        if verbose:
            self._print_summary(errors)
        return self._recordings

    def _print_summary(self, errors: List[str]) -> None:
        subjects = sorted({r.subject_id for r in self._recordings})
        print(f"\n{'─'*64}")
        print(f"Quattrocento CSV scan: {self.root_dir}  (side={self.side!r})")
        print(f"  Total: {len(self._recordings)} files, {len(subjects)} subjects")
        for s in subjects:
            recs = [r for r in self._recordings if r.subject_id == s]
            ch_configs = sorted({(r.n_channels, r.window_size) for r in recs})
            gestures   = sorted({r.gesture for r in recs})
            print(f"  {s}: {len(recs):3d} files | (n_ch,ws)={ch_configs} | {gestures}")
        if errors:
            print(f"\n  ⚠ {len(errors)} parse errors (first 5):")
            for e in errors[:5]:
                print(e)
        print(f"{'─'*64}\n")

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    @property
    def recordings(self) -> List[QuattrocentoRecording]:
        if not self._scanned:
            self.scan(verbose=False)
        return self._recordings

    def list_subjects(self) -> List[str]:
        return sorted({r.subject_id for r in self.recordings})

    def list_gestures(self, subject_id: Optional[str] = None) -> List[str]:
        recs = self.recordings
        if subject_id:
            recs = [r for r in recs if r.subject_id == subject_id]
        return sorted({r.gesture for r in recs})

    def list_channel_configs(self) -> List[Tuple[int, int]]:
        return sorted({(r.n_channels, r.window_size) for r in self.recordings})

    def filter(
        self,
        subjects=None,
        gestures=None,
        n_channels: Optional[int] = None,
        use_classification_only: bool = False,
        sides=None,
    ) -> List[QuattrocentoRecording]:
        out = list(self.recordings)
        if subjects:                out = [r for r in out if r.subject_id in subjects]
        if gestures:                out = [r for r in out if r.gesture in gestures]
        if n_channels is not None:  out = [r for r in out if r.n_channels == n_channels]
        if use_classification_only: out = [r for r in out if r.use_as_classification]
        if sides:                   out = [r for r in out if r.side in sides]
        return out

    # ------------------------------------------------------------------
    # Dataset builder
    # ------------------------------------------------------------------

    def build_dataset(
        self,
        name: str,
        subjects=None,
        gestures=None,
        use_classification_only: bool = False,
        window_ms: Optional[int] = None,
        overlap_pct: Optional[int] = None,
        n_channels_target: Optional[int] = None,
        channel_slice: Optional[slice] = None,
        progress_callback=None,
        use_trigger_segmentation: bool = False,
        use_rms_selection: bool = False,
        trigger_min_gap_ms: int = 250,
    ) -> Dict[str, Any]:
        """
        Build a pipeline-compatible dataset dict from CSV recordings.

        Returns
        -------
        dict with keys:
            ``X``          – float32 array (n_windows, window_size, n_channels)
            ``y``          – int64 label array (n_windows,)
            ``trial_ids``  – string array (n_windows,)
            ``metadata``   – dict with dataset provenance

        Mixed channel counts
        --------------------
        Pass ``n_channels_target=<N>`` to keep only files with exactly N EMG
        channels, OR ``channel_slice=slice(0, N)`` to take the first N channels
        from every file regardless of total count.
        """
        if not self._scanned:
            self.scan(verbose=False)

        wms  = window_ms   if window_ms   is not None else self.window_ms
        opct = overlap_pct if overlap_pct is not None else self.overlap_pct

        recs = self.filter(
            subjects=subjects,
            gestures=gestures,
            n_channels=n_channels_target,
            use_classification_only=use_classification_only,
        )
        if not recs:
            raise ValueError(
                f"No recordings matched: subjects={subjects}, "
                f"gestures={gestures}, n_channels_target={n_channels_target}"
            )

        raw_ch = sorted({r.n_channels for r in recs})
        if len(raw_ch) > 1 and channel_slice is None:
            raise ValueError(
                f"Mixed channel counts {raw_ch}.\n"
                "Pass n_channels_target=<N> OR channel_slice=slice(0, N)."
            )

        if channel_slice is not None:
            min_ch   = min(r.n_channels for r in recs)
            n_ch_out = len(range(*channel_slice.indices(min_ch)))
        else:
            n_ch_out = raw_ch[0]

        sr_est = float(np.median([r.sampling_rate for r in recs]))
        ws     = max(1, int(round(sr_est * wms / 1000.0)))
        stride = max(1, int(round(ws * (1.0 - opct / 100.0))))

        present_gestures = sorted(
            {r.gesture for r in recs},
            key=lambda g: _DEFAULT_LABEL_ORDER.index(g) if g in _DEFAULT_LABEL_ORDER else 999,
        )
        g2l = {g: i for i, g in enumerate(present_gestures)}

        X_parts: List[np.ndarray] = []
        y_parts: List[np.ndarray] = []
        trial_ids: List[str] = []

        for idx, rec in enumerate(recs):
            if progress_callback:
                progress_callback(idx, len(recs), f"{rec.subject_id}/{rec.gesture}/{rec.trial_label}")

            wins = QuattrocentoFileLoader.load_windows(
                rec,
                window_ms=wms,
                overlap_pct=opct,
                channel_slice=channel_slice,
                use_trigger_segmentation=use_trigger_segmentation,
                use_rms_selection=use_rms_selection,
                trigger_min_gap_ms=trigger_min_gap_ms,
            )
            # Resize if window shape differs (e.g. different sr per file)
            if wins.shape[1] != ws or wins.shape[2] != n_ch_out:
                wins = np.stack(
                    [_resize_window(w, ws, n_ch_out) for w in wins], axis=0
                )

            lbl = g2l[rec.gesture]
            n   = wins.shape[0]
            if n == 0:
                continue
            X_parts.append(wins)
            y_parts.append(np.full(n, lbl, dtype=np.int64))
            trial_ids.extend(
                [f"{rec.subject_id}_{rec.trial_label}_{rec.gesture}_{rec.side}"] * n
            )

        if X_parts:
            X = np.concatenate(X_parts, axis=0)
            y = np.concatenate(y_parts, axis=0)
            wp = int(X.shape[0])
        else:
            X = np.empty((0, ws, n_ch_out), dtype=np.float32)
            y = np.empty(0, dtype=np.int64)
            wp = 0
        if progress_callback:
            progress_callback(len(recs), len(recs), "Done")

        stride_ms = round(stride / sr_est * 1000, 1)

        return {
            "X": X,
            "y": y,
            "trial_ids": np.array(trial_ids),
            "metadata": {
                "name": name,
                "source": "quattrocento_csv",
                "created_at": datetime.now().isoformat(),
                "num_samples": int(wp),
                "num_classes": len(present_gestures),
                "window_size_ms": round(ws / sr_est * 1000, 1),
                "window_stride_ms": stride_ms,
                "sampling_rate": round(sr_est, 1),
                "num_channels": n_ch_out,
                "window_samples": ws,
                "overlap_pct": opct,
                "label_names": {int(v): k for k, v in g2l.items()},
                "gesture_order": present_gestures,
                "subjects_used": sorted({r.subject_id for r in recs}),
                "recordings_used": [r.path.name for r in recs],
                "raw_channel_configs": sorted({(r.n_channels, r.window_size) for r in recs}),
                "sides_used": sorted({r.side for r in recs}),
                "channel_slice": str(channel_slice),
                "use_trigger_segmentation": bool(use_trigger_segmentation),
                "use_rms_selection": bool(use_rms_selection),
                "trigger_min_gap_ms": int(trigger_min_gap_ms),
                "calibration_applied": False,
                "calibration_rotation_offset": 0,
                "per_session_rotation": False,
                "session_rotation_offsets": {},
                "bad_channel_mode": "none",
                "signal_modes_used": ["monopolar"],
                "features_extracted": False,
                "feature_config": None,
                "feature_dim": 0,
            },
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resize_window(win: np.ndarray, target_ws: int, target_ch: int) -> np.ndarray:
    """Pad or truncate a single window (ws, ch) to (target_ws, target_ch)."""
    ws, ch = win.shape
    # Channel axis
    if ch > target_ch:
        win = win[:, :target_ch]
    elif ch < target_ch:
        win = np.concatenate(
            [win, np.zeros((ws, target_ch - ch), dtype=win.dtype)], axis=1
        )
    # Time axis
    if ws == target_ws:
        return win
    if ws > target_ws:
        return win[:target_ws]
    return np.concatenate(
        [win, np.zeros((target_ws - ws, win.shape[1]), dtype=win.dtype)], axis=0
    )


def discover_quattrocento_root(data_dir: Path) -> Optional[Path]:
    """Locate the quattrocento data directory under *data_dir*."""
    for name in ["quattrocento", "Quattrocento", "quattrocento_data"]:
        p = data_dir / name
        if p.is_dir():
            return p
    for child in data_dir.iterdir():
        if child.is_dir() and list(child.rglob("VHI_Recording_*_left.csv")):
            return child
    return None
