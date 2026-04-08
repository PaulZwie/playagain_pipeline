"""
Quattrocento HD-EMG data loader  (v4 – trigger-aware CSV input)

Key improvements over v3
─────────────────────────
• Trigger-based repetition extraction:
    Each Quattrocento gesture file contains ~5 repetitions separated by rest.
    The trigger column goes HIGH during each active rep.
    ``extract_trigger_segments()`` returns one TriggerSegment per repetition.

• RMS-based onset detection (fallback):
    When trigger is absent/zero, ``rms_based_segments()`` detects activity
    from the reference-signal RMS with a robust threshold.
    onset_delay_ms compensates for children's reaction-time lag.

• Streaming adapter:
    ``QuattrocentoStreamAdapter`` scans a directory and emits data in
    chronological chunks — same interface as Muovi session-replay.

Backward-compatible with v3 public API.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple, Any, Union

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_METADATA_COLS   = 4
_COL_TIME_S      = 0
_COL_SAMPLE_RATE = 1
_COL_TRIGGER     = 2
_COL_REF_RMS     = 3

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

_FILENAME_RE = re.compile(
    r"^VHI_Recording_((?:\d+_\d+)|\d+)_(.+?)_(trial\S+|default|mvcpowergrasp\S*|fistmvc\S*|holdduration\S*)_(left|right)\.csv$",
    re.IGNORECASE,
)


def normalize_gesture_name(task: str) -> str:
    return _TASK_ALIASES.get(task.strip().lower(), task.strip().lower().replace(" ", "_"))


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class TriggerSegment:
    """One gesture repetition extracted from a Quattrocento file."""
    rep_index: int
    start_sample: int
    end_sample: int
    n_samples: int
    source: str       # "trigger" | "rms" | "full"
    signal: np.ndarray  # (n_samples, n_channels)  float32


@dataclass
class QuattrocentoRecording:
    path: Path
    subject_id: str
    timestamp: datetime
    gesture_raw: str
    gesture: str
    trial_label: str
    task: str
    side: str
    use_as_classification: bool
    n_samples: int
    n_channels: int
    sampling_rate: float
    header_rows: int = 0
    window_size: int = 0
    n_windows: int = 0
    bad_channels: List[int] = field(default_factory=list)
    has_trigger: bool = False
    n_repetitions: int = 0
    trigger_col: int = _COL_TRIGGER
    ref_rms_col: int = _COL_REF_RMS
    signal_start_col: int = _METADATA_COLS
    condition: str = ""   # e.g. "healthy" | "paralysed" | ""

    def __repr__(self) -> str:
        cond = f", cond={self.condition!r}" if self.condition else ""
        return (
            f"QuattrocentoRecording(subj={self.subject_id!r}, "
            f"gesture={self.gesture!r}, side={self.side!r}{cond}, "
            f"n_samples={self.n_samples}, n_ch={self.n_channels}, "
            f"sr={self.sampling_rate:.0f}Hz, reps={self.n_repetitions})"
        )


# ---------------------------------------------------------------------------
# Segment extraction
# ---------------------------------------------------------------------------

def extract_trigger_segments(
    emg: np.ndarray,
    trigger: np.ndarray,
    sampling_rate: float,
    min_duration_ms: float = 200.0,
    pre_ms: float = 50.0,
    post_ms: float = 50.0,
) -> List[TriggerSegment]:
    """Find HIGH periods in trigger and return one segment per repetition."""
    binary = (trigger > 0.5).astype(np.int8)
    diff = np.diff(binary, prepend=0)
    rising  = np.where(diff ==  1)[0]
    falling = np.where(diff == -1)[0]

    segments: List[TriggerSegment] = []
    fi = 0
    min_smp  = int(sampling_rate * min_duration_ms / 1000.0)
    pre_smp  = int(sampling_rate * pre_ms  / 1000.0)
    post_smp = int(sampling_rate * post_ms / 1000.0)

    for r in rising:
        while fi < len(falling) and falling[fi] <= r:
            fi += 1
        if fi >= len(falling):
            break
        f = falling[fi]
        fi += 1
        if (f - r) < min_smp:
            continue
        start = max(0, r - pre_smp)
        end   = min(len(emg), f + post_smp)
        seg   = emg[start:end]
        segments.append(TriggerSegment(
            rep_index=len(segments), start_sample=int(start),
            end_sample=int(end), n_samples=seg.shape[0],
            source="trigger", signal=seg.copy(),
        ))
    return segments


def rms_based_segments(
    emg: np.ndarray,
    ref_rms: np.ndarray,
    sampling_rate: float,
    n_reps_expected: int = 5,
    onset_delay_ms: float = 150.0,
    min_duration_ms: float = 800.0,
    smooth_ms: float = 100.0,
) -> List[TriggerSegment]:
    """Detect repetitions from reference-signal RMS when trigger is absent."""
    smooth_smp = max(1, int(sampling_rate * smooth_ms / 1000.0))
    kernel = np.ones(smooth_smp) / smooth_smp
    smoothed = np.convolve(ref_rms.astype(float), kernel, mode="same")

    baseline = float(np.median(smoothed))
    mad = float(np.median(np.abs(smoothed - baseline))) + 1e-12

    active = np.zeros(len(smoothed), dtype=np.int8)
    for k in (3.0, 2.5, 2.0, 1.5):
        threshold = baseline + k * mad
        active = (smoothed > threshold).astype(np.int8)
        n_rising = int(np.sum(np.diff(active, prepend=0) == 1))
        if n_rising >= n_reps_expected:
            break

    diff = np.diff(active, prepend=0)
    rising  = np.where(diff ==  1)[0]
    falling = np.where(diff == -1)[0]

    delay_smp = int(sampling_rate * onset_delay_ms / 1000.0)
    min_smp   = int(sampling_rate * min_duration_ms / 1000.0)

    segments: List[TriggerSegment] = []
    fi = 0
    for r in rising:
        while fi < len(falling) and falling[fi] <= r:
            fi += 1
        if fi >= len(falling):
            break
        f = falling[fi]
        fi += 1
        r_adj = max(0, min(r + delay_smp, f))
        if (f - r_adj) < min_smp:
            continue
        seg = emg[r_adj : min(len(emg), f)]
        segments.append(TriggerSegment(
            rep_index=len(segments), start_sample=int(r_adj),
            end_sample=int(min(len(emg), f)), n_samples=seg.shape[0],
            source="rms", signal=seg.copy(),
        ))
    return segments


def detect_segments(
    emg: np.ndarray,
    trigger: np.ndarray,
    ref_rms: np.ndarray,
    sampling_rate: float,
    n_reps_expected: int = 5,
    onset_delay_ms: float = 150.0,
) -> Tuple[List[TriggerSegment], str]:
    """Auto-select trigger vs RMS detection. Returns (segments, method)."""
    binary = trigger > 0.5
    has_trigger = bool(binary.any() and not binary.all())
    if has_trigger:
        segs = extract_trigger_segments(emg, trigger, sampling_rate)
        if segs:
            return segs, "trigger"
    segs = rms_based_segments(
        emg, ref_rms, sampling_rate,
        n_reps_expected=n_reps_expected,
        onset_delay_ms=onset_delay_ms,
    )
    return segs, "rms"


# ---------------------------------------------------------------------------
# File-level loader
# ---------------------------------------------------------------------------

class QuattrocentoFileLoader:
    DEFAULT_WINDOW_MS   = 200
    DEFAULT_OVERLAP_PCT = 50

    @staticmethod
    def _is_float_token(token: str) -> bool:
        try:
            float(token)
            return True
        except Exception:
            return False

    @classmethod
    def _read_header_tokens(cls, path: Path) -> Optional[List[str]]:
        with open(path, "r") as fh:
            first_line = fh.readline().strip()
        if not first_line:
            return None
        raw_tokens = [t.strip().strip('"').strip("'") for t in first_line.split(",")]
        if not raw_tokens:
            return None
        if all(cls._is_float_token(tok) for tok in raw_tokens if tok != ""):
            return None
        norm = [tok.lower().lstrip("\ufeff") for tok in raw_tokens]
        return norm

    @staticmethod
    def _resolve_columns(header_tokens: Optional[List[str]], n_cols: int) -> Tuple[int, int, int]:
        if not header_tokens:
            return _COL_TRIGGER, _COL_REF_RMS, _METADATA_COLS

        def _find(*aliases: str) -> Optional[int]:
            for alias in aliases:
                if alias in header_tokens:
                    return header_tokens.index(alias)
            return None

        trigger_col = _find("trigger_manual", "trigger", "trigger_signal")
        ref_col = _find("ref_signal_rms", "ref_rms", "rms", "reference_rms")

        signal_start = None
        for i, tok in enumerate(header_tokens):
            if tok.startswith("emg") or tok.startswith("ch_") or tok.startswith("channel"):
                signal_start = i
                break

        if trigger_col is None:
            trigger_col = _COL_TRIGGER
        if ref_col is None:
            ref_col = _COL_REF_RMS
        if signal_start is None:
            signal_start = max(_METADATA_COLS, trigger_col + 1, ref_col + 1)

        trigger_col = min(max(0, trigger_col), n_cols - 1)
        ref_col = min(max(0, ref_col), n_cols - 1)
        signal_start = min(max(0, signal_start), n_cols)
        return trigger_col, ref_col, signal_start

    @staticmethod
    def _quick_has_trigger(path: Path, header_rows: int, trigger_col: int, n_samples: int) -> bool:
        max_checks = 200
        stride = max(1, n_samples // max_checks)
        with open(path, "r") as fh:
            for _ in range(max(0, header_rows)):
                fh.readline()
            for i, line in enumerate(fh):
                if i % stride != 0:
                    continue
                parts = line.split(",")
                if len(parts) <= trigger_col:
                    continue
                try:
                    if float(parts[trigger_col]) > 0.5:
                        return True
                except ValueError:
                    continue
        return False

    @staticmethod
    def _parse_filename(fn: str) -> Optional[Tuple[datetime, str, str, str]]:
        m = _FILENAME_RE.match(fn)
        if m is None:
            return None
        ts_str = m.group(1)
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

    @staticmethod
    def _read_header_row(path: Path) -> Tuple[np.ndarray, int]:
        with open(path, "r") as fh:
            first_line = fh.readline().strip()
            first_row = np.fromstring(first_line, dtype=np.float64, sep=",")
            if len(first_row) >= _METADATA_COLS + 1:
                return first_row, 0
            second_line = fh.readline().strip()
            second_row = np.fromstring(second_line, dtype=np.float64, sep=",")
            if len(second_row) >= _METADATA_COLS + 1:
                return second_row, 1
        raise ValueError(f"Could not parse numeric CSV rows: {path.name}")

    @classmethod
    def _count_rows(cls, path: Path, header_rows: int = 0) -> int:
        """Fast row count via newline byte scan — avoids decoding every char."""
        with open(path, "rb") as fh:
            buf_size = 1 << 20  # 1 MiB chunks
            count = 0
            buf = fh.read(buf_size)
            while buf:
                count += buf.count(b"\n")
                buf = fh.read(buf_size)
        # Subtract header rows + possible missing trailing newline correction
        return max(0, count - header_rows)

    @classmethod
    def load_recording(
        cls,
        path: Path,
        subject_id: str,
        window_ms: int = DEFAULT_WINDOW_MS,
        overlap_pct: int = DEFAULT_OVERLAP_PCT,
    ) -> QuattrocentoRecording:
        parsed = cls._parse_filename(path.name)
        if parsed is None:
            raise ValueError(f"Filename does not match VHI CSV pattern: {path.name}")
        ts, gesture_raw, trial_label, side = parsed

        header_tokens = cls._read_header_tokens(path)
        first_row, header_rows = cls._read_header_row(path)
        if len(first_row) < _METADATA_COLS + 1:
            raise ValueError(f"CSV has only {len(first_row)} columns: {path.name}")

        trigger_col, ref_col, signal_start_col = cls._resolve_columns(header_tokens, len(first_row))

        sr         = float(first_row[_COL_SAMPLE_RATE]) if first_row[_COL_SAMPLE_RATE] > 0 else 2048.0
        n_channels = max(1, len(first_row) - signal_start_col)
        n_samples  = cls._count_rows(path, header_rows=header_rows)

        ws     = max(1, int(round(sr * window_ms / 1000.0)))
        stride = max(1, int(round(ws * (1.0 - overlap_pct / 100.0))))
        n_wins = max(0, (n_samples - ws) // stride + 1)
        gesture = normalize_gesture_name(gesture_raw)

        # Trigger detection is deferred to load time to keep scanning fast.
        # We do a cheap check: sample just the first 20 rows of the trigger column.
        has_trigger = False
        n_reps = 0
        try:
            has_trigger = cls._quick_has_trigger(path, int(header_rows), int(trigger_col), int(n_samples))
            # n_reps is estimated from n_samples / typical rep length at this sr
            # ~5 reps, each ~3 s rest+active, so estimate 5 reps as default
            n_reps = 5  # will be refined when data is actually loaded
        except Exception:
            pass

        return QuattrocentoRecording(
            path=path, subject_id=subject_id, timestamp=ts,
            gesture_raw=gesture_raw, gesture=gesture,
            trial_label=trial_label, task=gesture_raw, side=side,
            use_as_classification=True, n_samples=n_samples,
            n_channels=n_channels, sampling_rate=sr,
            header_rows=header_rows, window_size=ws, n_windows=n_wins,
            bad_channels=[], has_trigger=has_trigger, n_repetitions=n_reps,
            trigger_col=trigger_col, ref_rms_col=ref_col, signal_start_col=signal_start_col,
        )

    @classmethod
    def _load_npy(cls, rec: "QuattrocentoRecording") -> np.ndarray:
        """
        Load a pre-converted .npy file for *rec*.

        The .npy file must live next to the original CSV with the same stem
        (e.g. VHI_Recording_…_left.npy).  Run ``convert_csv_to_npy.py``
        once on your data directory to create all cache files.

        Raises
        ------
        FileNotFoundError
            If the .npy file does not exist yet.  The error message includes
            the exact command needed to create it.
        """
        npy_path = rec.path.with_suffix(".npy")
        if not npy_path.exists():
            raise FileNotFoundError(
                f"Pre-converted .npy file not found:\n"
                f"  {npy_path}\n\n"
                f"Run the conversion script once on your data directory:\n"
                f"  python convert_csv_to_npy.py \"{rec.path.parent.parent.parent}\"\n"
                f"(point it at the root folder that contains your VP_XX subjects)"
            )
        return np.load(str(npy_path), mmap_mode="r")

    @classmethod
    def load_raw_data(
        cls,
        rec: QuattrocentoRecording,
        channel_slice: Optional[slice] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns (signal, trigger, ref_rms).  Requires a pre-built .npy file."""
        data = cls._load_npy(rec)
        trigger_col = int(getattr(rec, "trigger_col", _COL_TRIGGER))
        ref_col = int(getattr(rec, "ref_rms_col", _COL_REF_RMS))
        signal_start = int(getattr(rec, "signal_start_col", _METADATA_COLS))
        trigger = data[:, trigger_col]
        ref_rms = data[:, ref_col]
        signal  = data[:, signal_start:]
        if channel_slice is not None:
            signal = signal[:, channel_slice]
        return signal, trigger, ref_rms

    @classmethod
    def load_raw_signal(
        cls,
        rec: QuattrocentoRecording,
        channel_slice: Optional[slice] = None,
    ) -> np.ndarray:
        """Backward-compat: returns EMG signal only."""
        signal, _, _ = cls.load_raw_data(rec, channel_slice)
        return signal

    @classmethod
    def load_repetition_segments(
        cls,
        rec: QuattrocentoRecording,
        channel_slice: Optional[slice] = None,
        onset_delay_ms: float = 150.0,
        n_reps_expected: int = 5,
    ) -> Tuple[List[TriggerSegment], str]:
        """Load file and extract per-repetition segments."""
        signal, trigger, ref_rms = cls.load_raw_data(rec, channel_slice)
        return detect_segments(signal, trigger, ref_rms, rec.sampling_rate,
                               n_reps_expected=n_reps_expected,
                               onset_delay_ms=onset_delay_ms)

    @classmethod
    def load_windows(
        cls,
        rec: QuattrocentoRecording,
        window_ms: int = DEFAULT_WINDOW_MS,
        overlap_pct: int = DEFAULT_OVERLAP_PCT,
        channel_slice: Optional[slice] = None,
        use_trigger_segments: bool = False,
        onset_delay_ms: float = 150.0,
    ) -> np.ndarray:
        """
        Load and return sliding-window array (n_windows, ws, n_ch).
        When use_trigger_segments=True, windows are taken within each
        active-gesture segment only (cleaner boundaries).
        """
        signal, trigger, ref_rms = cls.load_raw_data(rec, channel_slice)
        n_samples, n_ch = signal.shape
        sr     = rec.sampling_rate
        ws     = max(1, int(round(sr * window_ms / 1000.0)))
        stride = max(1, int(round(ws * (1.0 - overlap_pct / 100.0))))

        if use_trigger_segments:
            segs, _ = detect_segments(signal, trigger, ref_rms, sr,
                                      onset_delay_ms=onset_delay_ms)
            if segs:
                all_wins = []
                for seg in segs:
                    w = cls._slide(seg.signal, ws, stride)
                    if w is not None:
                        all_wins.append(w)
                if all_wins:
                    return np.concatenate(all_wins, axis=0)

        return cls._slide(signal, ws, stride)

    @staticmethod
    def _slide(signal: np.ndarray, ws: int, stride: int) -> np.ndarray:
        n_samples, n_ch = signal.shape
        if n_samples < ws:
            pad = np.zeros((ws - n_samples, n_ch), dtype=np.float32)
            return np.concatenate([signal, pad], axis=0)[np.newaxis, :, :]
        n_wins  = (n_samples - ws) // stride + 1
        windows = np.empty((n_wins, ws, n_ch), dtype=np.float32)
        for i in range(n_wins):
            windows[i] = signal[i * stride : i * stride + ws]
        return windows


# ---------------------------------------------------------------------------
# Streaming adapter
# ---------------------------------------------------------------------------

class QuattrocentoStreamAdapter:
    """
    Emit Quattrocento EMG data in chronological chunks, mirroring the
    Muovi session-replay interface used by the main pipeline.

    Usage::

        adapter = QuattrocentoStreamAdapter(root_dir, chunk_ms=20)
        adapter.scan()
        for chunk, gesture, is_new_file in adapter.stream():
            ...  # chunk: (n_samples, n_channels)
    """

    def __init__(
        self,
        root_dir: Union[str, Path],
        side: str = "left",
        chunk_ms: float = 20.0,
        channel_slice: Optional[slice] = None,
        use_trigger_segments: bool = True,
        onset_delay_ms: float = 150.0,
        window_ms: int = 200,
        overlap_pct: int = 50,
    ):
        self.root_dir             = Path(root_dir)
        self.side                 = side.lower()
        self.chunk_ms             = chunk_ms
        self.channel_slice        = channel_slice
        self.use_trigger_segments = use_trigger_segments
        self.onset_delay_ms       = onset_delay_ms
        self.window_ms            = window_ms
        self.overlap_pct          = overlap_pct
        self._recordings: List[QuattrocentoRecording] = []
        self._scanned = False

    def scan(self, verbose: bool = False) -> List[QuattrocentoRecording]:
        loader = QuattrocentoSubjectLoader(
            self.root_dir, side=self.side,
            window_ms=self.window_ms, overlap_pct=self.overlap_pct,
        )
        self._recordings = loader.scan(verbose=verbose)
        self._scanned = True
        return self._recordings

    def stream(self) -> Generator[Tuple[np.ndarray, str, bool], None, None]:
        """Yield (chunk, gesture_name, is_new_file) tuples."""
        if not self._scanned:
            self.scan()
        sorted_recs = sorted(self._recordings, key=lambda r: r.timestamp)

        for rec in sorted_recs:
            try:
                signal, trigger, ref_rms = QuattrocentoFileLoader.load_raw_data(
                    rec, channel_slice=self.channel_slice)
            except Exception:
                continue

            sr        = rec.sampling_rate
            chunk_smp = max(1, int(sr * self.chunk_ms / 1000.0))

            if self.use_trigger_segments:
                segs, _ = detect_segments(signal, trigger, ref_rms, sr,
                                          onset_delay_ms=self.onset_delay_ms)
                if not segs:
                    segs = [TriggerSegment(0, 0, len(signal), len(signal), "full", signal)]
            else:
                segs = [TriggerSegment(0, 0, len(signal), len(signal), "full", signal)]

            first = True
            for seg in segs:
                pos = 0
                while pos < len(seg.signal):
                    chunk = seg.signal[pos : pos + chunk_smp]
                    pos += chunk_smp
                    yield chunk, rec.gesture, first
                    first = False


# ---------------------------------------------------------------------------
# Subject / directory loader
# ---------------------------------------------------------------------------

class QuattrocentoSubjectLoader:
    """Scan and load Quattrocento CSV recordings from a directory tree."""

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

    def scan(self, verbose: bool = True) -> List[QuattrocentoRecording]:
        """
        Scan the root directory for Quattrocento CSV files.

        Supported layouts
        ─────────────────
        New (condition-aware):
            <root>/VP_XX/healthy/VHI_Recording_*.csv
            <root>/VP_XX/paralysed/VHI_Recording_*.csv

        Legacy (flat):
            <root>/VP_XX/recordings/VHI_Recording_*.csv
            <root>/VP_XX/VHI_Recording_*.csv
            <root>/VHI_Recording_*.csv
        """
        self._recordings.clear()
        errors: List[str] = []

        # Collect (csv_path, subject_id, condition) tuples
        candidates: List[Tuple[Path, str, str]] = []

        _CONDITION_DIRS = {"healthy", "paralysed", "paretic", "impaired"}

        for child in sorted(self.root_dir.iterdir()):
            if not child.is_dir():
                continue
            subj_id = child.name

            # New layout: VP_XX/healthy|paralysed/
            found_condition_dirs = False
            for cond_dir in sorted(child.iterdir()):
                if not cond_dir.is_dir():
                    continue
                cond = cond_dir.name.lower()
                if cond in _CONDITION_DIRS:
                    found_condition_dirs = True
                    for csv_path in sorted(cond_dir.glob("VHI_Recording_*.csv")):
                        candidates.append((csv_path, subj_id, cond))

            if found_condition_dirs:
                continue  # skip legacy layout for this subject

            # Legacy layout: VP_XX/recordings/ or VP_XX/ directly
            rd = child / "recordings"
            search_dir = rd if rd.is_dir() else child
            for csv_path in sorted(search_dir.glob("VHI_Recording_*.csv")):
                candidates.append((csv_path, subj_id, ""))

        # Fallback: CSVs directly in root
        if not candidates:
            if (self.root_dir / "recordings").is_dir():
                for csv_path in sorted((self.root_dir / "recordings").glob("VHI_Recording_*.csv")):
                    candidates.append((csv_path, self.root_dir.name, ""))
            else:
                for csv_path in sorted(self.root_dir.glob("VHI_Recording_*.csv")):
                    candidates.append((csv_path, self.root_dir.name, ""))

        for csv_path, sid, condition in candidates:
            name_lower = csv_path.name.lower()
            if self.side == "left"  and not name_lower.endswith("_left.csv"):
                continue
            if self.side == "right" and not name_lower.endswith("_right.csv"):
                continue
            try:
                rec = QuattrocentoFileLoader.load_recording(
                    csv_path, sid,
                    window_ms=self.window_ms,
                    overlap_pct=self.overlap_pct,
                )
                rec.condition = condition
                self._recordings.append(rec)
            except Exception as e:
                errors.append(f"  {csv_path.name}: {e}")

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
            gestures = sorted({r.gesture for r in recs})
            has_trig = sum(1 for r in recs if r.has_trigger)
            print(f"  {s}: {len(recs):3d} files | {gestures} | trigger={has_trig}/{len(recs)}")
        if errors:
            print(f"\n  ⚠ {len(errors)} errors:")
            for e in errors[:5]:
                print(e)
        print(f"{'─'*64}\n")

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

    def list_conditions(self) -> List[str]:
        """Return the unique condition labels found in the scan (e.g. ['healthy', 'paralysed'])."""
        return sorted({r.condition for r in self.recordings if r.condition})

    def list_channel_configs(self) -> List[Tuple[int, int]]:
        return sorted({(r.n_channels, r.window_size) for r in self.recordings})

    def filter(
        self,
        subjects=None,
        gestures=None,
        n_channels: Optional[int] = None,
        use_classification_only: bool = False,
        sides=None,
        conditions=None,
    ) -> List[QuattrocentoRecording]:
        out = list(self.recordings)
        if subjects:               out = [r for r in out if r.subject_id in subjects]
        if gestures:               out = [r for r in out if r.gesture in gestures]
        if n_channels is not None: out = [r for r in out if r.n_channels == n_channels]
        if use_classification_only:out = [r for r in out if r.use_as_classification]
        if sides:                  out = [r for r in out if r.side in sides]
        if conditions:             out = [r for r in out if r.condition in conditions]
        return out

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
        use_trigger_segments: bool = True,
        onset_delay_ms: float = 150.0,
        progress_callback=None,
    ) -> Dict[str, Any]:
        """Build a pipeline-compatible dataset dict from CSV recordings."""
        if not self._scanned:
            self.scan(verbose=False)

        wms  = window_ms   if window_ms   is not None else self.window_ms
        opct = overlap_pct if overlap_pct is not None else self.overlap_pct

        recs = self.filter(
            subjects=subjects, gestures=gestures,
            n_channels=n_channels_target,
            use_classification_only=use_classification_only,
        )
        if not recs:
            raise ValueError(
                f"No recordings matched: subjects={subjects}, gestures={gestures}")

        raw_ch = sorted({r.n_channels for r in recs})
        if len(raw_ch) > 1 and channel_slice is None:
            raise ValueError(
                f"Mixed channel counts {raw_ch}.\n"
                "Pass n_channels_target=<N> OR channel_slice=slice(0,N).")

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

        total_windows = sum(
            max(0, (r.n_samples - ws) // stride + 1) if r.n_samples >= ws else 1
            for r in recs
        )
        X         = np.empty((total_windows, ws, n_ch_out), dtype=np.float32)
        y         = np.empty(total_windows, dtype=np.int64)
        trial_ids: List[str] = []
        wp = 0

        for idx, rec in enumerate(recs):
            if progress_callback:
                progress_callback(idx, len(recs), f"{rec.subject_id}/{rec.gesture}/{rec.trial_label}")

            wins = QuattrocentoFileLoader.load_windows(
                rec, window_ms=wms, overlap_pct=opct,
                channel_slice=channel_slice,
                use_trigger_segments=use_trigger_segments,
                onset_delay_ms=onset_delay_ms,
            )
            if wins.shape[1] != ws or wins.shape[2] != n_ch_out:
                wins = np.stack([_resize_window(w, ws, n_ch_out) for w in wins], axis=0)

            lbl = g2l[rec.gesture]
            n   = wins.shape[0]
            if wp + n > X.shape[0]:
                extra = wp + n - X.shape[0]
                X = np.concatenate([X, np.empty((extra, ws, n_ch_out), dtype=np.float32)], 0)
                y = np.concatenate([y, np.empty(extra, dtype=np.int64)], 0)

            X[wp : wp + n] = wins
            y[wp : wp + n] = lbl
            trial_ids.extend(
                [f"{rec.subject_id}_{rec.trial_label}_{rec.gesture}_{rec.side}"] * n
            )
            wp += n

        X = X[:wp]
        y = y[:wp]
        if progress_callback:
            progress_callback(len(recs), len(recs), "Done")

        n_with_trigger = sum(1 for r in recs if r.has_trigger)

        return {
            "X": X, "y": y, "trial_ids": np.array(trial_ids),
            "metadata": {
                "name": name, "source": "quattrocento_csv",
                "created_at": datetime.now().isoformat(),
                "num_samples": int(wp),
                "num_classes": len(present_gestures),
                "window_size_ms": round(ws / sr_est * 1000, 1),
                "window_stride_ms": round(stride / sr_est * 1000, 1),
                "sampling_rate": round(sr_est, 1),
                "num_channels": n_ch_out,
                "window_samples": ws, "overlap_pct": opct,
                "label_names": {int(v): k for k, v in g2l.items()},
                "gesture_order": present_gestures,
                "subjects_used": sorted({r.subject_id for r in recs}),
                "recordings_used": [r.path.name for r in recs],
                "raw_channel_configs": sorted({(r.n_channels, r.window_size) for r in recs}),
                "sides_used": sorted({r.side for r in recs}),
                "channel_slice": str(channel_slice),
                "calibration_applied": False, "calibration_rotation_offset": 0,
                "per_session_rotation": False, "session_rotation_offsets": {},
                "bad_channel_mode": "none", "signal_modes_used": ["monopolar"],
                "features_extracted": False, "feature_config": None, "feature_dim": 0,
                # v4 additions
                "use_trigger_segments": use_trigger_segments,
                "onset_delay_ms": onset_delay_ms,
                "files_with_trigger": n_with_trigger,
                "files_total": len(recs),
            },
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resize_window(win: np.ndarray, target_ws: int, target_ch: int) -> np.ndarray:
    ws, ch = win.shape
    if ch > target_ch:
        win = win[:, :target_ch]
    elif ch < target_ch:
        win = np.concatenate([win, np.zeros((ws, target_ch - ch), dtype=win.dtype)], axis=1)
    if ws == target_ws:
        return win
    if ws > target_ws:
        return win[:target_ws]
    return np.concatenate([win, np.zeros((target_ws - ws, win.shape[1]), dtype=win.dtype)], axis=0)


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