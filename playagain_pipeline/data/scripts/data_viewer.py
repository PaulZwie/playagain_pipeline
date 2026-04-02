#!/usr/bin/env python3
"""Standalone PyQt data explorer for sessions and game recordings.

This tool is intentionally standalone (not integrated into the main pipeline runtime).
It loads raw data, applies optional in-memory transforms for visualization, and can
export transformed data to a new sibling folder without modifying the source dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import matplotlib
import numpy as np
import pandas as pd
from scipy import signal as scipy_signal

# Allow running this file directly via `python data_viewer.py`.
WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from playagain_pipeline.core.gesture import Gesture, GestureCategory, GestureSet, create_default_gesture_set
from playagain_pipeline.core.session import RecordingSession, RecordingTrial

# Keep a non-interactive default backend for headless mode; GUI mode switches to QtAgg.
matplotlib.use("Agg")


UNITY_SOURCE_TYPE = "unity_recordings"
UNITY_DEFAULT_ROOT = Path(
    "/Users/paul/Coding_Projects/PlayAgain-Game2/playagain-game-2/RecordedData/Users"
)
SUPPORTED_TYPES = ("sessions", "game_recordings", UNITY_SOURCE_TYPE)


@dataclass
class TransformConfig:
    notch_enabled: bool = True
    notch_freq: float = 50.0
    notch_q: float = 30.0
    bandpass_enabled: bool = True
    bandpass_low: float = 20.0
    bandpass_high: float = 450.0
    detrend_enabled: bool = False
    artifact_enabled: bool = True
    artifact_lowpass_hz: float = 5.0
    artifact_strength: float = 1.0
    clip_enabled: bool = False
    clip_mad_mult: float = 8.0
    normalize_enabled: bool = False
    rectify_enabled: bool = False
    rms_envelope_enabled: bool = False
    rms_window_ms: float = 80.0


@dataclass
class DisplayConfig:
    show_raw: bool = True
    show_processed: bool = True
    show_annotations: bool = True
    normalize_channels: bool = True
    y_spacing: float = 4.0
    downsample_factor: int = 1
    raw_alpha: float = 0.55


@dataclass
class DataBundle:
    source_type: str
    source_path: Path
    sampling_rate: float
    signal: np.ndarray
    signal_columns: list[str]
    full_table: pd.DataFrame
    timestamp: np.ndarray
    annotation_spans: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TransformTrace:
    name: str
    parameters: dict[str, Any]


def script_default_data_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _safe_float(value: Any, fallback: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _safe_int(value: Any, fallback: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _numeric_signal_from_columns(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    values = df[cols].copy().apply(pd.to_numeric, errors="coerce")
    if values.isna().any().any():
        values = values.interpolate(limit_direction="both")
        values = values.fillna(0.0)
    return values.to_numpy(dtype=float)


def _is_emg_csv_candidate(path: Path) -> bool:
    name = path.name.lower()
    parent = path.parent.name.lower()
    if "emg" in name or "emg" in parent:
        return True
    try:
        header = path.open("r", encoding="utf-8", errors="ignore").readline().lower()
    except OSError:
        return False
    return "emg_ch" in header or "ch_" in header


def discover_recordings(data_root: Path, source_type: str) -> list[Path]:
    if source_type not in SUPPORTED_TYPES:
        raise ValueError(f"Unsupported source type '{source_type}'.")
    root = data_root if source_type == UNITY_SOURCE_TYPE else data_root / source_type
    if not root.exists():
        return []

    if source_type == UNITY_SOURCE_TYPE:
        if root.is_file() and root.suffix.lower() == ".csv":
            return [root] if _is_emg_csv_candidate(root) else []
        csv_files: list[Path] = []
        for candidate in root.rglob("*.csv"):
            if _is_emg_csv_candidate(candidate):
                csv_files.append(candidate)
        return sorted(csv_files)

    required_file = "metadata.json" if source_type == "sessions" else "config.json"
    paths: list[Path] = []
    for candidate in root.rglob("*"):
        if candidate.is_dir() and (candidate / required_file).exists():
            paths.append(candidate)
    return sorted(paths)


def _load_session_bundle(path: Path) -> DataBundle:
    metadata_path = path / "metadata.json"
    metadata = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    csv_path = path / "data.csv"
    npy_path = path / "data.npy"

    if csv_path.exists():
        table = pd.read_csv(csv_path)
        signal_cols = [c for c in table.columns if c.startswith("CH_")]
        if not signal_cols:
            signal_cols = [c for c in table.columns if "ch" in c.lower()]
        if not signal_cols:
            raise ValueError(f"No channel columns found in {csv_path}")
        signal = _numeric_signal_from_columns(table, signal_cols)
        full_table = table.copy()
    elif npy_path.exists():
        signal = np.load(npy_path)
        if signal.ndim != 2:
            raise ValueError(f"Expected 2D array in {npy_path}, got shape {signal.shape}")
        signal_cols = [f"CH_{idx + 1}" for idx in range(signal.shape[1])]
        full_table = pd.DataFrame(signal, columns=signal_cols)
    else:
        raise FileNotFoundError(f"Neither data.csv nor data.npy found in {path}")

    md = metadata.get("metadata", metadata)
    sampling_rate = _safe_float(md.get("sampling_rate"), 2000.0)
    timestamp = np.arange(signal.shape[0], dtype=float) / max(1e-9, sampling_rate)

    spans: list[dict[str, Any]] = []
    for trial in metadata.get("trials", []):
        spans.append(
            {
                "start": _safe_float(
                    trial.get("start_time"),
                    _safe_float(trial.get("start_sample"), 0.0) / sampling_rate,
                ),
                "end": _safe_float(
                    trial.get("end_time"),
                    _safe_float(trial.get("end_sample"), 0.0) / sampling_rate,
                ),
                "label": str(trial.get("gesture_name", "unknown")),
                "trial_type": str(trial.get("trial_type", "trial")),
            }
        )

    return DataBundle(
        source_type="sessions",
        source_path=path,
        sampling_rate=sampling_rate,
        signal=signal,
        signal_columns=signal_cols,
        full_table=full_table,
        timestamp=timestamp,
        annotation_spans=spans,
        metadata=metadata,
    )


def _load_game_bundle(path: Path) -> DataBundle:
    config_path = path / "config.json"
    config = {}
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))

    csv_path = path / "recording.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"recording.csv missing in {path}")

    table = pd.read_csv(csv_path)
    signal_cols = [c for c in table.columns if c.startswith("EMG_Ch")]
    if not signal_cols:
        signal_cols = [c for c in table.columns if c.lower().startswith("ch_")]
    if not signal_cols:
        raise ValueError(f"No EMG channel columns found in {csv_path}")

    signal = _numeric_signal_from_columns(table, signal_cols)

    rec_cfg = config.get("recording", config)
    sampling_rate = _safe_float(rec_cfg.get("sampling_rate"), 2000.0)

    if "Timestamp" in table.columns:
        timestamp = (
            pd.to_numeric(table["Timestamp"], errors="coerce")
            .interpolate(limit_direction="both")
            .fillna(0.0)
            .to_numpy(float)
        )
    else:
        timestamp = np.arange(signal.shape[0], dtype=float) / max(1e-9, sampling_rate)

    labels, inference_meta = _infer_sample_labels(table)
    spans = _labels_to_spans(labels, timestamp)

    return DataBundle(
        source_type="game_recordings",
        source_path=path,
        sampling_rate=sampling_rate,
        signal=signal,
        signal_columns=signal_cols,
        full_table=table,
        timestamp=timestamp,
        annotation_spans=spans,
        metadata={"recording_config": config, "label_inference": inference_meta},
    )


def _load_unity_bundle(path: Path) -> DataBundle:
    if path.is_dir():
        candidates = discover_recordings(path, UNITY_SOURCE_TYPE)
        if not candidates:
            raise FileNotFoundError(f"No EMG CSV file found in {path}")
        csv_path = candidates[0]
    else:
        csv_path = path

    table = pd.read_csv(csv_path)
    signal_cols = [c for c in table.columns if c.startswith("EMG_Ch")]
    if not signal_cols:
        signal_cols = [c for c in table.columns if c.lower().startswith("ch_")]
    if not signal_cols:
        raise ValueError(f"No EMG channel columns found in {csv_path}")

    signal = _numeric_signal_from_columns(table, signal_cols)
    sampling_rate = _infer_sampling_rate(table)
    if "Timestamp" in table.columns:
        timestamp = (
            pd.to_numeric(table["Timestamp"], errors="coerce")
            .interpolate(limit_direction="both")
            .fillna(0.0)
            .to_numpy(float)
        )
    else:
        timestamp = np.arange(signal.shape[0], dtype=float) / max(1e-9, sampling_rate)

    labels, inference_meta = _infer_sample_labels(table)
    spans = _labels_to_spans(labels, timestamp)

    return DataBundle(
        source_type=UNITY_SOURCE_TYPE,
        source_path=csv_path,
        sampling_rate=sampling_rate,
        signal=signal,
        signal_columns=signal_cols,
        full_table=table,
        timestamp=timestamp,
        annotation_spans=spans,
        metadata={"unity_csv": str(csv_path), "label_inference": inference_meta},
    )


def load_bundle(source_type: str, path: Path) -> DataBundle:
    if source_type == "sessions":
        return _load_session_bundle(path)
    if source_type == "game_recordings":
        return _load_game_bundle(path)
    if source_type == UNITY_SOURCE_TYPE:
        return _load_unity_bundle(path)
    raise ValueError(f"Unsupported source type '{source_type}'")


def _infer_sampling_rate(table: pd.DataFrame, default_fs: float = 2000.0) -> float:
    if "Timestamp" not in table.columns or len(table) < 4:
        return default_fs
    ts = pd.to_numeric(table["Timestamp"], errors="coerce").interpolate(limit_direction="both").fillna(0.0).to_numpy(float)
    dts = np.diff(ts)
    dts = dts[np.isfinite(dts) & (dts > 1e-6)]
    if dts.size == 0:
        return default_fs
    return max(1.0, 1.0 / float(np.median(dts)))


def _normalize_label(value: Any) -> str:
    text = str(value).strip().lower()
    if text in {"", "none", "nan", "null", "unknown", "idle"}:
        return "rest"
    return text.replace(" ", "_")


def _binary_activity_from_table(table: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
    for col in ("GroundTruthActive", "GroundTruth", "GestureActive"):
        if col in table.columns:
            values = pd.to_numeric(table[col], errors="coerce").fillna(0.0).to_numpy(float)
            return values > 0.5, {"activity_source": col}

    if "RMS" in table.columns:
        rms = pd.to_numeric(table["RMS"], errors="coerce").interpolate(limit_direction="both").fillna(0.0).to_numpy(float)
        baseline = float(np.median(rms))
        mad = float(np.median(np.abs(rms - baseline)))
        threshold = baseline + 2.5 * max(1e-12, mad)
        if np.mean(rms >= threshold) < 0.005:
            threshold = float(np.quantile(rms, 0.9))
        return rms >= threshold, {"activity_source": "RMS", "rms_threshold": threshold}

    return np.zeros(len(table), dtype=bool), {"activity_source": "none"}


def _infer_sample_labels(table: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
    n_rows = len(table)
    if n_rows == 0:
        return np.array([], dtype=object), {"label_source": "empty"}

    active, meta = _binary_activity_from_table(table)
    labels = np.full(n_rows, "rest", dtype=object)

    if "RequestedGesture" in table.columns:
        requested = table["RequestedGesture"].astype(str).map(_normalize_label).to_numpy(dtype=object)
        labels = np.where(active, requested, "rest")
        labels = np.where(np.isin(labels, ["none", "unknown", "idle"]), "rest", labels)
        meta["label_source"] = "RequestedGesture+activity"
    else:
        labels = np.where(active, "fist", "rest")
        meta["label_source"] = "activity_proxy_fist_rest"

    return labels, meta


def _labels_to_spans(labels: np.ndarray, timestamp: np.ndarray, min_duration_sec: float = 0.08) -> list[dict[str, Any]]:
    spans: list[dict[str, Any]] = []
    if labels.size == 0 or timestamp.size == 0:
        return spans

    start_idx = 0
    current = str(labels[0])
    for idx in range(1, labels.size + 1):
        at_end = idx == labels.size
        if not at_end and str(labels[idx]) == current:
            continue

        end_idx = idx
        if end_idx > start_idx:
            start_time = float(timestamp[start_idx])
            end_time = float(timestamp[end_idx - 1])
            if (end_time - start_time) >= min_duration_sec:
                spans.append({"start": start_time, "end": end_time, "label": current, "trial_type": "gesture"})

        if not at_end:
            start_idx = idx
            current = str(labels[idx])
    return spans


def _safe_name(text: str) -> str:
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', text.strip()).strip('._') or "unnamed"


def _guess_subject_id(path: Path) -> str:
    for part in [path.name, *path.parts[::-1]]:
        if re.fullmatch(r"VP[_-]?\d+", str(part), flags=re.IGNORECASE):
            return str(part).replace("-", "_")
    return "VP_00"


def _build_gesture_set(labels: set[str]) -> GestureSet:
    base = create_default_gesture_set()
    ordered = sorted(labels, key=lambda x: (x != "rest", x != "fist", x))
    gesture_set = GestureSet(name="viewer_derived")
    for label in ordered:
        known = base.get_gesture(label)
        if known is not None:
            gesture = Gesture(
                name=known.name,
                display_name=known.display_name,
                description=known.description,
                category=known.category,
                image_path=known.image_path,
                emoji=known.emoji,
                label_id=known.label_id,
                duration_hint=known.duration_hint,
                metadata=dict(known.metadata),
            )
        else:
            gesture = Gesture(
                name=label,
                display_name=label.replace("_", " ").title(),
                category=GestureCategory.CUSTOM,
            )
        gesture_set.add_gesture(gesture)
    return gesture_set


def export_bundle_as_session(
    bundle: DataBundle,
    signal_to_export: np.ndarray,
    keep_labels: set[str] | None,
    subject_id: str,
    session_id: str,
    destination_root: Path | None = None,
) -> Path:
    if signal_to_export.shape != bundle.signal.shape:
        raise ValueError("Signal to export must match loaded signal shape.")

    spans = sorted(bundle.annotation_spans, key=lambda sp: (_safe_float(sp.get("start"), 0.0), _safe_float(sp.get("end"), 0.0)))
    if keep_labels is not None:
        spans = [sp for sp in spans if _normalize_label(sp.get("label", "")) in keep_labels]
    if not spans:
        raise ValueError("No spans available after filtering labels.")

    kept_chunks: list[np.ndarray] = []
    exported_trials: list[tuple[str, int, int]] = []
    cursor = 0
    n_samples = signal_to_export.shape[0]

    for sp in spans:
        label = _normalize_label(sp.get("label", "rest"))
        st = _safe_float(sp.get("start"), 0.0)
        en = _safe_float(sp.get("end"), st)
        start_idx = int(np.searchsorted(bundle.timestamp, st, side="left"))
        end_idx = int(np.searchsorted(bundle.timestamp, en, side="right"))
        start_idx = max(0, min(start_idx, n_samples - 1))
        end_idx = max(start_idx + 1, min(end_idx, n_samples))
        chunk = signal_to_export[start_idx:end_idx]
        if chunk.size == 0:
            continue
        kept_chunks.append(chunk)
        exported_trials.append((label, cursor, cursor + chunk.shape[0]))
        cursor += chunk.shape[0]

    if not kept_chunks or not exported_trials:
        raise ValueError("No data remained after span conversion.")

    kept_signal = np.vstack(kept_chunks)
    labels = {lbl for lbl, _, _ in exported_trials}
    gesture_set = _build_gesture_set(labels)

    session = RecordingSession(
        session_id=session_id,
        subject_id=subject_id,
        device_name=str(bundle.metadata.get("recording_config", {}).get("recording", {}).get("device_name", "MUOVI")),
        num_channels=kept_signal.shape[1],
        sampling_rate=int(round(bundle.sampling_rate)),
        gesture_set=gesture_set,
        protocol_name="viewer_derived",
    )
    session._data_chunks = [kept_signal]
    session._current_sample = kept_signal.shape[0]
    session.trials = []

    for idx, (label, start_smp, end_smp) in enumerate(exported_trials):
        gesture = gesture_set.get_gesture(label)
        if gesture is None:
            continue
        session.trials.append(
            RecordingTrial(
                trial_id=idx,
                gesture_name=label,
                gesture_label=int(gesture.label_id),
                start_sample=start_smp,
                end_sample=end_smp,
                start_time=start_smp / max(1e-9, bundle.sampling_rate),
                end_time=end_smp / max(1e-9, bundle.sampling_rate),
                is_valid=True,
                trial_type="gesture",
            )
        )

    session.metadata.notes = (
        f"Derived in data_viewer from {bundle.source_type}: {bundle.source_path}. "
        f"Keep labels: {sorted(keep_labels) if keep_labels is not None else 'all'}"
    )
    session.metadata.custom_metadata = {
        "source_type": bundle.source_type,
        "source_path": str(bundle.source_path),
        "source_sampling_rate": bundle.sampling_rate,
        "label_inference": bundle.metadata.get("label_inference", {}),
    }

    dest_root = script_default_data_root() / "sessions" if destination_root is None else destination_root
    subject_dir = dest_root / _safe_name(subject_id)
    subject_dir.mkdir(parents=True, exist_ok=True)
    base = _safe_name(session_id)
    target = subject_dir / base
    suffix = 1
    while target.exists():
        target = subject_dir / f"{base}_{suffix:02d}"
        suffix += 1
    session.save(target)
    return target


def _apply_notch(data: np.ndarray, fs: float, freq: float, q: float) -> np.ndarray:
    nyq = fs / 2.0
    w0 = freq / max(1e-9, nyq)
    if not (0.0 < w0 < 1.0) or data.shape[0] < 16:
        return data
    b, a = scipy_signal.iirnotch(w0, Q=max(1.0, q))
    try:
        return scipy_signal.filtfilt(b, a, data, axis=0)
    except ValueError:
        return data


def _apply_bandpass(data: np.ndarray, fs: float, low_hz: float, high_hz: float) -> np.ndarray:
    nyq = fs / 2.0
    low = max(0.01, low_hz / max(1e-9, nyq))
    high = min(0.99, high_hz / max(1e-9, nyq))
    if low >= high or data.shape[0] < 24:
        return data
    sos = scipy_signal.butter(4, [low, high], btype="band", output="sos")
    try:
        return scipy_signal.sosfiltfilt(sos, data, axis=0)
    except ValueError:
        return data


def _apply_motion_artifact_removal(data: np.ndarray, fs: float, cutoff_hz: float, strength: float) -> np.ndarray:
    nyq = fs / 2.0
    wn = min(max(cutoff_hz / max(1e-9, nyq), 1e-4), 0.99)
    if data.shape[0] < 24:
        return data
    b, a = scipy_signal.butter(2, wn, btype="low", output="ba")
    try:
        baseline = scipy_signal.filtfilt(b, a, data, axis=0)
    except ValueError:
        return data
    strength = min(max(strength, 0.0), 2.0)
    return data - (strength * baseline)


def _robust_clip(data: np.ndarray, mad_mult: float) -> np.ndarray:
    median = np.median(data, axis=0, keepdims=True)
    mad = np.median(np.abs(data - median), axis=0, keepdims=True)
    sigma = np.maximum(1e-12, 1.4826 * mad)
    limit = max(1.0, mad_mult) * sigma
    return np.clip(data, median - limit, median + limit)


def _normalize_channels(data: np.ndarray) -> np.ndarray:
    center = np.nanmean(data, axis=0, keepdims=True)
    scale = np.nanstd(data, axis=0, keepdims=True)
    scale = np.where(scale < 1e-12, 1.0, scale)
    normalized = (data - center) / scale
    return np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)


def _rms_envelope(data: np.ndarray, fs: float, window_ms: float) -> np.ndarray:
    window = max(1, int(fs * window_ms / 1000.0))
    kernel = np.ones(window, dtype=float) / window
    squared = data * data
    out = np.empty_like(squared)
    for idx in range(squared.shape[1]):
        out[:, idx] = np.sqrt(np.convolve(squared[:, idx], kernel, mode="same"))
    return out


def apply_transform_pipeline(
    raw_signal: np.ndarray,
    sampling_rate: float,
    config: TransformConfig,
) -> tuple[np.ndarray, list[TransformTrace]]:
    data = raw_signal.copy()
    trace: list[TransformTrace] = []

    if config.notch_enabled:
        data = _apply_notch(data, sampling_rate, config.notch_freq, config.notch_q)
        trace.append(TransformTrace("notch", {"freq": config.notch_freq, "q": config.notch_q}))

    if config.bandpass_enabled:
        data = _apply_bandpass(data, sampling_rate, config.bandpass_low, config.bandpass_high)
        trace.append(TransformTrace("bandpass", {"low": config.bandpass_low, "high": config.bandpass_high}))

    if config.detrend_enabled:
        data = scipy_signal.detrend(data, axis=0, type="linear")
        trace.append(TransformTrace("detrend", {"type": "linear"}))

    if config.artifact_enabled:
        data = _apply_motion_artifact_removal(
            data,
            sampling_rate,
            config.artifact_lowpass_hz,
            config.artifact_strength,
        )
        trace.append(
            TransformTrace(
                "movement_artifact_removal",
                {"lowpass_hz": config.artifact_lowpass_hz, "strength": config.artifact_strength},
            )
        )

    if config.clip_enabled:
        data = _robust_clip(data, config.clip_mad_mult)
        trace.append(TransformTrace("robust_clip", {"mad_multiplier": config.clip_mad_mult}))

    if config.normalize_enabled:
        data = _normalize_channels(data)
        trace.append(TransformTrace("normalize", {"mode": "zscore_per_channel"}))

    if config.rectify_enabled:
        data = np.abs(data)
        trace.append(TransformTrace("rectify", {}))

    if config.rms_envelope_enabled:
        data = _rms_envelope(data, sampling_rate, config.rms_window_ms)
        trace.append(TransformTrace("rms_envelope", {"window_ms": config.rms_window_ms}))

    return data, trace


def channel_quality_report(raw_signal: np.ndarray) -> dict[str, Any]:
    if raw_signal.size == 0:
        return {"channels": []}
    median = np.median(raw_signal, axis=0)
    rms = np.sqrt(np.mean(raw_signal ** 2, axis=0))
    p2p = np.ptp(raw_signal, axis=0)
    nan_ratio = np.mean(~np.isfinite(raw_signal), axis=0)
    return {
        "channels": [
            {
                "idx": int(i),
                "median": float(median[i]),
                "rms": float(rms[i]),
                "peak_to_peak": float(p2p[i]),
                "nan_ratio": float(nan_ratio[i]),
            }
            for i in range(raw_signal.shape[1])
        ]
    }


def parse_channel_spec(spec: str, max_channels: int) -> list[int]:
    spec = spec.strip()
    if not spec or spec.lower() in {"all", "*"}:
        return list(range(max_channels))

    channels: set[int] = set()
    chunks = [c.strip() for c in spec.split(",") if c.strip()]
    for chunk in chunks:
        if "-" in chunk:
            start_str, end_str = chunk.split("-", maxsplit=1)
            start = max(1, int(start_str))
            end = min(max_channels, int(end_str))
            if start <= end:
                channels.update(range(start - 1, end))
        else:
            idx = int(chunk)
            if 1 <= idx <= max_channels:
                channels.add(idx - 1)

    if not channels:
        return list(range(min(8, max_channels)))
    return sorted(channels)


def _ask_user_to_pick(paths: Iterable[Path], prompt: str) -> Path:
    candidates = list(paths)
    if not candidates:
        raise ValueError("No datasets found for the requested type.")
    print(prompt)
    for idx, pth in enumerate(candidates, start=1):
        print(f"[{idx:02d}] {pth}")
    while True:
        raw = input("Select index: ").strip()
        if raw.isdigit():
            chosen = int(raw)
            if 1 <= chosen <= len(candidates):
                return candidates[chosen - 1]
        print("Invalid selection, try again.")


def _signal_digest(signal: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(signal.shape).encode("utf-8"))
    digest.update(np.ascontiguousarray(signal[: min(signal.shape[0], 5000)]).tobytes())
    return digest.hexdigest()[:16]


def export_transformed_bundle(
    bundle: DataBundle,
    transformed_signal: np.ndarray,
    trace: list[TransformTrace],
    display: DisplayConfig,
    destination_root: Path | None = None,
) -> Path:
    source = bundle.source_path
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    source_stem = source.stem if source.is_file() else source.name
    new_name = f"{source_stem}_viewer_export_{stamp}"

    destination = source.parent / new_name if destination_root is None else destination_root / new_name
    if source.is_dir():
        shutil.copytree(source, destination)
    else:
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination / source.name)

    if bundle.source_type == "sessions":
        data_csv = destination / "data.csv"
        if data_csv.exists():
            table = pd.read_csv(data_csv)
            for idx, col in enumerate(bundle.signal_columns):
                if col in table.columns and idx < transformed_signal.shape[1]:
                    table[col] = transformed_signal[:, idx]
            table.to_csv(data_csv, index=False)
        npy = destination / "data.npy"
        if npy.exists():
            np.save(npy, transformed_signal)

    elif bundle.source_type == "game_recordings":
        csv_file = destination / "recording.csv"
        table = pd.read_csv(csv_file)
        for idx, col in enumerate(bundle.signal_columns):
            if col in table.columns and idx < transformed_signal.shape[1]:
                table[col] = transformed_signal[:, idx]
        table.to_csv(csv_file, index=False)

    elif bundle.source_type == UNITY_SOURCE_TYPE:
        csv_file = destination / source.name
        table = pd.read_csv(csv_file)
        for idx, col in enumerate(bundle.signal_columns):
            if col in table.columns and idx < transformed_signal.shape[1]:
                table[col] = transformed_signal[:, idx]
        table.to_csv(csv_file, index=False)

    manifest = {
        "created_at": datetime.now().isoformat(),
        "source_path": str(source),
        "source_type": bundle.source_type,
        "sampling_rate": bundle.sampling_rate,
        "raw_signal_shape": list(bundle.signal.shape),
        "raw_signal_digest": _signal_digest(bundle.signal),
        "transform_trace": [asdict(item) for item in trace],
        "display_config": asdict(display),
        "note": "Source data was left untouched; this folder is a derived export.",
    }
    (destination / "viewer_transform_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    return destination


def _import_qt_modules() -> dict[str, Any]:
    try:
        from PyQt6.QtCore import Qt, QTimer
        from PyQt6.QtWidgets import (
            QApplication,
            QCheckBox,
            QComboBox,
            QDoubleSpinBox,
            QFileDialog,
            QFormLayout,
            QGroupBox,
            QHBoxLayout,
            QLabel,
            QLineEdit,
            QListWidget,
            QListWidgetItem,
            QMainWindow,
            QMessageBox,
            QPushButton,
            QSlider,
            QSpinBox,
            QScrollArea,
            QSplitter,
            QStatusBar,
            QTabWidget,
            QTextEdit,
            QVBoxLayout,
            QWidget,
        )
        qt_api = "PyQt6"
    except ImportError:
        from PySide6.QtCore import Qt, QTimer
        from PySide6.QtWidgets import (
            QApplication,
            QCheckBox,
            QComboBox,
            QDoubleSpinBox,
            QFileDialog,
            QFormLayout,
            QGroupBox,
            QHBoxLayout,
            QLabel,
            QLineEdit,
            QListWidget,
            QListWidgetItem,
            QMainWindow,
            QMessageBox,
            QPushButton,
            QSlider,
            QSpinBox,
            QScrollArea,
            QSplitter,
            QStatusBar,
            QTabWidget,
            QTextEdit,
            QVBoxLayout,
            QWidget,
        )
        qt_api = "PySide6"

    matplotlib.use("QtAgg")
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
    from matplotlib.figure import Figure

    return {
        "Qt": Qt,
        "QTimer": QTimer,
        "QApplication": QApplication,
        "QCheckBox": QCheckBox,
        "QComboBox": QComboBox,
        "QDoubleSpinBox": QDoubleSpinBox,
        "QFileDialog": QFileDialog,
        "QFormLayout": QFormLayout,
        "QGroupBox": QGroupBox,
        "QHBoxLayout": QHBoxLayout,
        "QLabel": QLabel,
        "QLineEdit": QLineEdit,
        "QListWidget": QListWidget,
        "QListWidgetItem": QListWidgetItem,
        "QMainWindow": QMainWindow,
        "QMessageBox": QMessageBox,
        "QPushButton": QPushButton,
        "QSlider": QSlider,
        "QSpinBox": QSpinBox,
        "QScrollArea": QScrollArea,
        "QSplitter": QSplitter,
        "QStatusBar": QStatusBar,
        "QTabWidget": QTabWidget,
        "QTextEdit": QTextEdit,
        "QVBoxLayout": QVBoxLayout,
        "QWidget": QWidget,
        "Figure": Figure,
        "FigureCanvasQTAgg": FigureCanvasQTAgg,
        "NavigationToolbar2QT": NavigationToolbar2QT,
        "qt_api": qt_api,
    }


def _build_main_window_class(qt: dict[str, Any]) -> type:
    Qt = qt["Qt"]
    QMainWindow = qt["QMainWindow"]
    QWidget = qt["QWidget"]
    QVBoxLayout = qt["QVBoxLayout"]
    QHBoxLayout = qt["QHBoxLayout"]
    QGroupBox = qt["QGroupBox"]
    QFormLayout = qt["QFormLayout"]
    QComboBox = qt["QComboBox"]
    QLineEdit = qt["QLineEdit"]
    QPushButton = qt["QPushButton"]
    QListWidget = qt["QListWidget"]
    QListWidgetItem = qt["QListWidgetItem"]
    QLabel = qt["QLabel"]
    QSlider = qt["QSlider"]
    QSpinBox = qt["QSpinBox"]
    QScrollArea = qt["QScrollArea"]
    QDoubleSpinBox = qt["QDoubleSpinBox"]
    QCheckBox = qt["QCheckBox"]
    QTabWidget = qt["QTabWidget"]
    QTextEdit = qt["QTextEdit"]
    QStatusBar = qt["QStatusBar"]
    QMessageBox = qt["QMessageBox"]
    QFileDialog = qt["QFileDialog"]
    QSplitter = qt["QSplitter"]
    Figure = qt["Figure"]
    FigureCanvasQTAgg = qt["FigureCanvasQTAgg"]
    NavigationToolbar2QT = qt["NavigationToolbar2QT"]

    class MainWindow(QMainWindow):
        def __init__(self, data_root: Path, source_type: str, initial_path: Path | None):
            super().__init__()
            self.data_root = data_root
            self.source_type = source_type
            self.bundle: DataBundle | None = None
            self.trace: list[TransformTrace] = []
            self.transformed_signal = np.empty((0, 0), dtype=float)
            self.transform_cfg = TransformConfig()
            self.display_cfg = DisplayConfig()
            self.external_plot_window: Any | None = None
            self.external_figure: Any | None = None
            self.external_canvas: Any | None = None
            self.external_ax_raw: Any | None = None
            self.external_ax_proc: Any | None = None
            self._plot_update_pending = False

            self.setWindowTitle(f"Data Viewer ({qt['qt_api']})")
            self.resize(1920, 1120)
            self.setStatusBar(QStatusBar())

            self._plot_update_timer = qt["QTimer"](self)
            self._plot_update_timer.setSingleShot(True)
            self._plot_update_timer.timeout.connect(self._update_plot_only_now)

            root_widget = QWidget()
            root_layout = QVBoxLayout(root_widget)
            self.setCentralWidget(root_widget)

            splitter = QSplitter(Qt.Orientation.Horizontal)
            root_layout.addWidget(splitter)

            def _scroll_wrap(widget: QWidget) -> QWidget:
                scroll = QScrollArea()
                scroll.setWidgetResizable(True)
                scroll.setWidget(widget)
                wrapper = QWidget()
                wrapper_layout = QVBoxLayout(wrapper)
                wrapper_layout.setContentsMargins(0, 0, 0, 0)
                wrapper_layout.addWidget(scroll)
                return wrapper

            left_panel = QWidget()
            left_layout = QVBoxLayout(left_panel)
            left_panel.setMinimumWidth(320)
            splitter.addWidget(_scroll_wrap(left_panel))

            right_panel = QWidget()
            right_layout = QVBoxLayout(right_panel)
            splitter.addWidget(right_panel)
            splitter.setSizes([320, 1600])

            self.controls_tabs = QTabWidget()
            right_layout.addWidget(self.controls_tabs)

            browse_group = QGroupBox("Dataset Browser")
            browse_form = QFormLayout(browse_group)
            self.type_combo = QComboBox()
            self.type_combo.addItems(list(SUPPORTED_TYPES))
            self.type_combo.setCurrentText(self.source_type)
            self.type_combo.currentTextChanged.connect(self._on_type_changed)
            browse_form.addRow("Type", self.type_combo)

            self.root_line = QLineEdit(str(self.data_root))
            browse_form.addRow("Data root", self.root_line)

            browse_buttons = QWidget()
            browse_buttons_layout = QHBoxLayout(browse_buttons)
            browse_buttons_layout.setContentsMargins(0, 0, 0, 0)
            self.refresh_btn = QPushButton("Refresh")
            self.refresh_btn.clicked.connect(self._refresh_datasets)
            self.pick_btn = QPushButton("Pick folder")
            self.pick_btn.clicked.connect(self._pick_dataset_folder)
            browse_buttons_layout.addWidget(self.refresh_btn)
            browse_buttons_layout.addWidget(self.pick_btn)
            browse_form.addRow("", browse_buttons)

            self.dataset_list = QListWidget()
            self.dataset_list.itemSelectionChanged.connect(self._load_selected_dataset)
            browse_form.addRow("Available", self.dataset_list)
            left_layout.addWidget(browse_group)

            channel_group = QGroupBox("Channel Selection")
            channel_layout = QVBoxLayout(channel_group)
            self.channel_list = QListWidget()
            self.channel_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
            self.channel_list.itemSelectionChanged.connect(self._update_plot_only)
            channel_layout.addWidget(self.channel_list)

            channel_buttons = QWidget()
            channel_buttons_layout = QHBoxLayout(channel_buttons)
            channel_buttons_layout.setContentsMargins(0, 0, 0, 0)
            btn_all = QPushButton("All")
            btn_all.clicked.connect(self._select_all_channels)
            btn_top8 = QPushButton("Top 8")
            btn_top8.clicked.connect(self._select_top_channels)
            btn_clear = QPushButton("Clear")
            btn_clear.clicked.connect(self._clear_channels)
            channel_buttons_layout.addWidget(btn_all)
            channel_buttons_layout.addWidget(btn_top8)
            channel_buttons_layout.addWidget(btn_clear)
            channel_layout.addWidget(channel_buttons)
            left_layout.addWidget(channel_group)

            anno_group = QGroupBox("Annotations")
            anno_layout = QVBoxLayout(anno_group)
            self.annotation_list = QListWidget()
            self.annotation_list.itemSelectionChanged.connect(self._jump_to_annotation)
            anno_layout.addWidget(self.annotation_list)
            left_layout.addWidget(anno_group)

            left_layout.addStretch(1)

            controls_group = QGroupBox("Refined Controls")
            controls_form = QFormLayout(controls_group)
            controls_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

            self.window_sec_spin = QDoubleSpinBox()
            self.window_sec_spin.setRange(0.2, 120.0)
            self.window_sec_spin.setValue(8.0)
            self.window_sec_spin.setSingleStep(0.5)
            self.window_sec_spin.valueChanged.connect(self._update_plot_only)
            controls_form.addRow("Window [s]", self.window_sec_spin)

            self.start_slider = QSlider(Qt.Orientation.Horizontal)
            self.start_slider.setRange(0, 1000)
            self.start_slider.setValue(0)
            self.start_slider.valueChanged.connect(self._update_plot_only)
            controls_form.addRow("Start", self.start_slider)

            self.downsample_spin = QSpinBox()
            self.downsample_spin.setRange(1, 50)
            self.downsample_spin.setValue(1)
            self.downsample_spin.valueChanged.connect(self._update_plot_only)
            controls_form.addRow("Downsample", self.downsample_spin)

            self.y_spacing_spin = QDoubleSpinBox()
            self.y_spacing_spin.setRange(1.0, 20.0)
            self.y_spacing_spin.setSingleStep(0.5)
            self.y_spacing_spin.setValue(4.0)
            self.y_spacing_spin.valueChanged.connect(self._update_plot_only)
            controls_form.addRow("Y spacing", self.y_spacing_spin)

            display_flags = QWidget()
            display_layout = QHBoxLayout(display_flags)
            display_layout.setContentsMargins(0, 0, 0, 0)
            self.show_raw_cb = QCheckBox("Raw")
            self.show_raw_cb.setChecked(True)
            self.show_raw_cb.stateChanged.connect(self._update_plot_only)
            self.show_processed_cb = QCheckBox("Processed")
            self.show_processed_cb.setChecked(True)
            self.show_processed_cb.stateChanged.connect(self._update_plot_only)
            self.show_ann_cb = QCheckBox("Annotations")
            self.show_ann_cb.setChecked(True)
            self.show_ann_cb.stateChanged.connect(self._update_plot_only)
            self.normalize_cb = QCheckBox("Normalize")
            self.normalize_cb.setChecked(True)
            self.normalize_cb.setToolTip("Display-only scaling for easier visual comparison; does not change exported data")
            self.normalize_cb.stateChanged.connect(self._update_plot_only)
            for cb in (self.show_raw_cb, self.show_processed_cb, self.show_ann_cb, self.normalize_cb):
                display_layout.addWidget(cb)
            controls_form.addRow("Display", display_flags)

            self.popout_plot_btn = QPushButton("Open Plot Window")
            self.popout_plot_btn.clicked.connect(self._open_plot_window)
            controls_form.addRow("Plot", self.popout_plot_btn)

            self.controls_tabs.addTab(_scroll_wrap(controls_group), "View")

            transform_group = QGroupBox("Transform Pipeline")
            transform_form = QFormLayout(transform_group)
            transform_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

            self.notch_cb = QCheckBox("Enable notch")
            self.notch_cb.setChecked(True)
            transform_form.addRow(self.notch_cb)
            self.notch_freq = QDoubleSpinBox()
            self.notch_freq.setRange(1.0, 500.0)
            self.notch_freq.setValue(50.0)
            transform_form.addRow("Notch Hz", self.notch_freq)

            self.bandpass_cb = QCheckBox("Enable bandpass")
            self.bandpass_cb.setChecked(True)
            transform_form.addRow(self.bandpass_cb)
            self.bp_low = QDoubleSpinBox()
            self.bp_low.setRange(1.0, 1000.0)
            self.bp_low.setValue(20.0)
            self.bp_high = QDoubleSpinBox()
            self.bp_high.setRange(10.0, 1200.0)
            self.bp_high.setValue(450.0)
            bp_widget = QWidget()
            bp_layout = QHBoxLayout(bp_widget)
            bp_layout.setContentsMargins(0, 0, 0, 0)
            bp_layout.addWidget(self.bp_low)
            bp_layout.addWidget(QLabel("to"))
            bp_layout.addWidget(self.bp_high)
            transform_form.addRow("Bandpass Hz", bp_widget)

            self.detrend_cb = QCheckBox("Detrend")
            transform_form.addRow(self.detrend_cb)

            self.artifact_cb = QCheckBox("Artifact removal")
            self.artifact_cb.setChecked(True)
            transform_form.addRow(self.artifact_cb)
            self.artifact_cutoff = QDoubleSpinBox()
            self.artifact_cutoff.setRange(0.1, 40.0)
            self.artifact_cutoff.setValue(5.0)
            self.artifact_strength = QDoubleSpinBox()
            self.artifact_strength.setRange(0.0, 2.0)
            self.artifact_strength.setSingleStep(0.1)
            self.artifact_strength.setValue(1.0)
            artifact_widget = QWidget()
            artifact_layout = QHBoxLayout(artifact_widget)
            artifact_layout.setContentsMargins(0, 0, 0, 0)
            artifact_layout.addWidget(self.artifact_cutoff)
            artifact_layout.addWidget(QLabel("x"))
            artifact_layout.addWidget(self.artifact_strength)
            transform_form.addRow("Artifact Hz/strength", artifact_widget)

            self.clip_cb = QCheckBox("Robust clip")
            transform_form.addRow(self.clip_cb)
            self.clip_mad = QDoubleSpinBox()
            self.clip_mad.setRange(1.0, 20.0)
            self.clip_mad.setValue(8.0)
            transform_form.addRow("Clip MAD", self.clip_mad)

            self.proc_normalize_cb = QCheckBox("Normalize channels (processing)")
            self.proc_normalize_cb.setToolTip("Applies per-channel z-score normalization to processed data and exports")
            transform_form.addRow(self.proc_normalize_cb)

            self.rectify_cb = QCheckBox("Rectify")
            transform_form.addRow(self.rectify_cb)

            self.rms_cb = QCheckBox("RMS envelope")
            transform_form.addRow(self.rms_cb)
            self.rms_window = QDoubleSpinBox()
            self.rms_window.setRange(5.0, 500.0)
            self.rms_window.setValue(80.0)
            transform_form.addRow("RMS window [ms]", self.rms_window)

            actions = QWidget()
            actions_layout = QHBoxLayout(actions)
            actions_layout.setContentsMargins(0, 0, 0, 0)
            self.apply_btn = QPushButton("Apply")
            self.apply_btn.clicked.connect(self._recompute_and_plot)
            self.reset_btn = QPushButton("Reset")
            self.reset_btn.clicked.connect(self._reset_transforms)
            self.export_btn = QPushButton("Export Derived")
            self.export_btn.clicked.connect(self._export_copy)
            self.report_btn = QPushButton("Write Report")
            self.report_btn.clicked.connect(self._write_quality_report)
            for btn in (self.apply_btn, self.reset_btn, self.export_btn, self.report_btn):
                actions_layout.addWidget(btn)
            transform_form.addRow(actions)
            self.controls_tabs.addTab(_scroll_wrap(transform_group), "Transforms")

            fit_group = QGroupBox("Session Fit / Curation")
            fit_form = QFormLayout(fit_group)

            self.fit_subject_line = QLineEdit()
            fit_form.addRow("Subject ID", self.fit_subject_line)

            self.fit_session_line = QLineEdit()
            fit_form.addRow("Session ID", self.fit_session_line)

            self.fit_keep_labels_line = QLineEdit("rest,fist")
            self.fit_keep_labels_line.setToolTip("Comma-separated labels to keep in curated export")
            fit_form.addRow("Keep labels", self.fit_keep_labels_line)

            self.fit_use_processed_cb = QCheckBox("Use processed signal")
            self.fit_use_processed_cb.setChecked(True)
            fit_form.addRow("Signal", self.fit_use_processed_cb)

            fit_actions = QWidget()
            fit_actions_layout = QHBoxLayout(fit_actions)
            fit_actions_layout.setContentsMargins(0, 0, 0, 0)
            self.export_all_session_btn = QPushButton("Export Full Session Fit")
            self.export_all_session_btn.setText("Export Full Session (All Labels)")
            self.export_all_session_btn.setToolTip("Exports all detected spans/labels into a fitted session")
            self.export_all_session_btn.clicked.connect(self._export_full_session_fit)
            self.export_curated_session_btn = QPushButton("Export Curated (Keep Labels)")
            self.export_curated_session_btn.setToolTip(
                "Exports only labels listed in 'Keep labels' (for example: rest,fist)"
            )
            self.export_curated_session_btn.clicked.connect(self._export_curated_session_fit)
            fit_actions_layout.addWidget(self.export_all_session_btn)
            fit_actions_layout.addWidget(self.export_curated_session_btn)
            fit_form.addRow(fit_actions)

            export_help = QLabel(
                "Full Session keeps all inferred labels. Curated keeps only 'Keep labels'. "
                "Both exports use the selected Signal mode (raw or processed)."
            )
            export_help.setWordWrap(True)
            fit_form.addRow(export_help)

            self.controls_tabs.addTab(_scroll_wrap(fit_group), "Session Fit")

            self.tabs = QTabWidget()
            right_layout.addWidget(self.tabs, stretch=1)

            plot_tab = QWidget()
            plot_layout = QVBoxLayout(plot_tab)
            self.figure = Figure(figsize=(16, 10), dpi=100)
            self.canvas = FigureCanvasQTAgg(self.figure)
            self.toolbar = NavigationToolbar2QT(self.canvas, self)
            plot_layout.addWidget(self.toolbar)
            plot_layout.addWidget(self.canvas)
            self.ax_raw = self.figure.add_subplot(211)
            self.ax_proc = self.figure.add_subplot(212, sharex=self.ax_raw)
            self.tabs.addTab(plot_tab, "Signals")

            info_tab = QWidget()
            info_layout = QVBoxLayout(info_tab)
            self.info_text = QTextEdit()
            self.info_text.setReadOnly(True)
            info_layout.addWidget(self.info_text)
            self.tabs.addTab(info_tab, "Metadata / Provenance")

            self._refresh_datasets()
            if initial_path is not None:
                self._select_dataset_by_path(initial_path)
            elif self.dataset_list.count() > 0:
                self.dataset_list.setCurrentRow(0)

        def _selected_channels(self) -> list[int]:
            if self.bundle is None:
                return []
            selected = []
            for i in range(self.channel_list.count()):
                item = self.channel_list.item(i)
                if item.isSelected():
                    selected.append(int(item.data(Qt.ItemDataRole.UserRole)))
            if not selected:
                return list(range(min(8, self.bundle.signal.shape[1])))
            return sorted(selected)

        def _populate_channels(self) -> None:
            self.channel_list.clear()
            if self.bundle is None:
                return
            for idx, name in enumerate(self.bundle.signal_columns):
                item = QListWidgetItem(f"{idx + 1:02d} - {name}")
                item.setData(Qt.ItemDataRole.UserRole, idx)
                item.setSelected(idx < min(8, len(self.bundle.signal_columns)))
                self.channel_list.addItem(item)

        def _populate_annotations(self) -> None:
            self.annotation_list.clear()
            if self.bundle is None:
                return
            for idx, span in enumerate(self.bundle.annotation_spans):
                txt = f"[{idx:02d}] {span['label']} | {span['start']:.2f}s - {span['end']:.2f}s"
                item = QListWidgetItem(txt)
                item.setData(Qt.ItemDataRole.UserRole, float(span["start"]))
                self.annotation_list.addItem(item)

        def _refresh_datasets(self) -> None:
            self.dataset_list.clear()
            self.data_root = Path(self.root_line.text()).expanduser().resolve()
            self.source_type = self.type_combo.currentText()
            for pth in discover_recordings(self.data_root, self.source_type):
                item = QListWidgetItem(str(pth))
                item.setData(Qt.ItemDataRole.UserRole, str(pth))
                self.dataset_list.addItem(item)
            self.statusBar().showMessage(f"Found {self.dataset_list.count()} dataset(s)", 2500)

        def _select_dataset_by_path(self, path: Path) -> None:
            target = str(path.resolve())
            for i in range(self.dataset_list.count()):
                item = self.dataset_list.item(i)
                if item.data(Qt.ItemDataRole.UserRole) == target:
                    self.dataset_list.setCurrentItem(item)
                    return
            item = QListWidgetItem(target)
            item.setData(Qt.ItemDataRole.UserRole, target)
            self.dataset_list.insertItem(0, item)
            self.dataset_list.setCurrentItem(item)

        def _on_type_changed(self, text: str) -> None:
            self.source_type = text
            if text == UNITY_SOURCE_TYPE:
                self.root_line.setText(str(UNITY_DEFAULT_ROOT))
            self._refresh_datasets()

        def _pick_dataset_folder(self) -> None:
            if self.source_type == UNITY_SOURCE_TYPE:
                chosen_file, _ = QFileDialog.getOpenFileName(
                    self,
                    "Select Unity EMG CSV",
                    str(self.data_root),
                    "CSV Files (*.csv);;All Files (*)",
                )
                if chosen_file:
                    self._select_dataset_by_path(Path(chosen_file))
                    return

            chosen = QFileDialog.getExistingDirectory(self, "Select dataset folder", str(self.data_root))
            if chosen:
                self._select_dataset_by_path(Path(chosen))

        def _load_selected_dataset(self) -> None:
            selected = self.dataset_list.selectedItems()
            if not selected:
                return
            path = Path(selected[0].data(Qt.ItemDataRole.UserRole))
            try:
                self.bundle = load_bundle(self.source_type, path)
            except Exception as exc:
                QMessageBox.critical(self, "Load Failed", f"Could not load dataset:\n{exc}")
                return

            self.start_slider.blockSignals(True)
            max_sec = float(self.bundle.timestamp[-1]) if self.bundle.timestamp.size else 0.0
            self.start_slider.setRange(0, max(1, int(max_sec * 1000.0)))
            self.start_slider.setValue(0)
            self.start_slider.blockSignals(False)

            self._populate_channels()
            self._populate_annotations()
            self._update_fit_defaults()
            self._recompute_and_plot()

        def _update_fit_defaults(self) -> None:
            if self.bundle is None:
                return
            self.fit_subject_line.setText(_guess_subject_id(self.bundle.source_path))
            source_name = self.bundle.source_path.stem if self.bundle.source_path.is_file() else self.bundle.source_path.name
            self.fit_session_line.setText(f"{_safe_name(source_name)}_fit")

        def _signal_for_fit_export(self) -> np.ndarray:
            if self.bundle is None:
                return np.empty((0, 0), dtype=float)
            if self.fit_use_processed_cb.isChecked() and self.transformed_signal.shape == self.bundle.signal.shape:
                return self.transformed_signal
            return self.bundle.signal

        def _export_full_session_fit(self) -> None:
            self._export_session_fit(keep_labels=None)

        def _export_curated_session_fit(self) -> None:
            labels = {
                _normalize_label(token)
                for token in self.fit_keep_labels_line.text().split(",")
                if token.strip()
            }
            if not labels:
                QMessageBox.warning(self, "Missing labels", "Provide at least one label to keep.")
                return
            self._export_session_fit(keep_labels=labels)

        def _export_session_fit(self, keep_labels: set[str] | None) -> None:
            if self.bundle is None:
                return
            subject_id = self.fit_subject_line.text().strip() or _guess_subject_id(self.bundle.source_path)
            session_id = self.fit_session_line.text().strip()
            if not session_id:
                QMessageBox.warning(self, "Missing session id", "Please set a target session id.")
                return
            try:
                dst = export_bundle_as_session(
                    bundle=self.bundle,
                    signal_to_export=self._signal_for_fit_export(),
                    keep_labels=keep_labels,
                    subject_id=subject_id,
                    session_id=session_id,
                )
                self.statusBar().showMessage(f"Session exported: {dst}", 7000)
            except Exception as exc:
                QMessageBox.critical(self, "Session export failed", str(exc))

        def _current_transform_config(self) -> TransformConfig:
            return TransformConfig(
                notch_enabled=self.notch_cb.isChecked(),
                notch_freq=float(self.notch_freq.value()),
                notch_q=30.0,
                bandpass_enabled=self.bandpass_cb.isChecked(),
                bandpass_low=float(self.bp_low.value()),
                bandpass_high=float(self.bp_high.value()),
                detrend_enabled=self.detrend_cb.isChecked(),
                artifact_enabled=self.artifact_cb.isChecked(),
                artifact_lowpass_hz=float(self.artifact_cutoff.value()),
                artifact_strength=float(self.artifact_strength.value()),
                clip_enabled=self.clip_cb.isChecked(),
                clip_mad_mult=float(self.clip_mad.value()),
                normalize_enabled=self.proc_normalize_cb.isChecked(),
                rectify_enabled=self.rectify_cb.isChecked(),
                rms_envelope_enabled=self.rms_cb.isChecked(),
                rms_window_ms=float(self.rms_window.value()),
            )

        def _current_display_config(self) -> DisplayConfig:
            return DisplayConfig(
                show_raw=self.show_raw_cb.isChecked(),
                show_processed=self.show_processed_cb.isChecked(),
                show_annotations=self.show_ann_cb.isChecked(),
                normalize_channels=self.normalize_cb.isChecked(),
                y_spacing=float(self.y_spacing_spin.value()),
                downsample_factor=max(1, int(self.downsample_spin.value())),
                raw_alpha=0.55,
            )

        def _recompute_and_plot(self) -> None:
            if self.bundle is None:
                return
            self.transform_cfg = self._current_transform_config()
            self.display_cfg = self._current_display_config()
            self.transformed_signal, self.trace = apply_transform_pipeline(
                self.bundle.signal,
                self.bundle.sampling_rate,
                self.transform_cfg,
            )
            self._update_metadata_tab()
            self._update_plot_only()

        def _reset_transforms(self) -> None:
            self.notch_cb.setChecked(True)
            self.notch_freq.setValue(50.0)
            self.bandpass_cb.setChecked(True)
            self.bp_low.setValue(20.0)
            self.bp_high.setValue(450.0)
            self.detrend_cb.setChecked(False)
            self.artifact_cb.setChecked(True)
            self.artifact_cutoff.setValue(5.0)
            self.artifact_strength.setValue(1.0)
            self.clip_cb.setChecked(False)
            self.clip_mad.setValue(8.0)
            self.proc_normalize_cb.setChecked(False)
            self.rectify_cb.setChecked(False)
            self.rms_cb.setChecked(False)
            self.rms_window.setValue(80.0)
            self._recompute_and_plot()

        def _window_mask(self) -> np.ndarray:
            if self.bundle is None or not self.bundle.timestamp.size:
                return np.array([], dtype=bool)
            start_sec = self.start_slider.value() / 1000.0
            win_sec = float(self.window_sec_spin.value())
            t = self.bundle.timestamp
            end_sec = start_sec + win_sec
            mask = (t >= start_sec) & (t <= end_sec)
            if not np.any(mask):
                mask = t >= max(0.0, end_sec - 1.0)
            return mask

        def _open_plot_window(self) -> None:
            if self.external_plot_window is None:
                popout = QMainWindow(self)
                popout.setWindowTitle("Data Viewer - Plot Window")
                popout.resize(1400, 900)

                container = QWidget(popout)
                layout = QVBoxLayout(container)
                fig = Figure(figsize=(14, 8), dpi=100)
                canvas = FigureCanvasQTAgg(fig)
                toolbar = NavigationToolbar2QT(canvas, popout)
                layout.addWidget(toolbar)
                layout.addWidget(canvas)
                popout.setCentralWidget(container)

                self.external_plot_window = popout
                self.external_figure = fig
                self.external_canvas = canvas
                self.external_ax_raw = fig.add_subplot(211)
                self.external_ax_proc = fig.add_subplot(212, sharex=self.external_ax_raw)
                popout.destroyed.connect(self._clear_external_plot_refs)

            self.external_plot_window.show()
            self.external_plot_window.raise_()
            self.external_plot_window.activateWindow()
            self._update_plot_only()

        def _clear_external_plot_refs(self, *_: Any) -> None:
            self.external_plot_window = None
            self.external_figure = None
            self.external_canvas = None
            self.external_ax_raw = None
            self.external_ax_proc = None

        def _update_plot_only(self) -> None:
            # Debounce UI-driven redraw bursts (slider + channel multiselect).
            self._plot_update_pending = True
            self._plot_update_timer.start(40)

        def _render_signal_axes(
            self,
            ax_raw: Any,
            ax_proc: Any,
            t: np.ndarray,
            raw: np.ndarray,
            proc: np.ndarray,
            channels: list[int],
            display_cfg: DisplayConfig,
        ) -> None:
            ax_raw.clear()
            ax_proc.clear()

            spacing = max(0.5, display_cfg.y_spacing)
            for idx, ch in enumerate(channels):
                raw_ch = raw[:, idx]
                proc_ch = proc[:, idx]
                if display_cfg.normalize_channels:
                    raw_std = np.nanstd(raw_ch) or 1.0
                    proc_std = np.nanstd(proc_ch) or 1.0
                    raw_ch = raw_ch / raw_std
                    proc_ch = proc_ch / proc_std
                offset = idx * spacing

                if display_cfg.show_raw:
                    ax_raw.plot(
                        t,
                        raw_ch + offset,
                        linewidth=0.8,
                        alpha=display_cfg.raw_alpha,
                        label=self.bundle.signal_columns[ch],
                    )
                if display_cfg.show_processed:
                    ax_proc.plot(
                        t,
                        proc_ch + offset,
                        linewidth=0.9,
                        label=self.bundle.signal_columns[ch],
                    )

            if display_cfg.show_annotations and self.bundle is not None:
                start = float(t[0])
                end = float(t[-1])
                for span in self.bundle.annotation_spans:
                    st = float(span["start"])
                    en = float(span["end"])
                    if en < start or st > end:
                        continue
                    ax_raw.axvspan(st, en, alpha=0.10, color="tab:orange")
                    ax_proc.axvspan(st, en, alpha=0.10, color="tab:orange")

            ax_raw.set_title("Raw signals")
            ax_proc.set_title(
                "Processed signals | "
                + ", ".join(step.name for step in self.trace)
                if self.trace
                else "Processed signals"
            )
            ax_proc.set_xlabel("Time [s]")
            ax_raw.set_ylabel("Amplitude + offset")
            ax_proc.set_ylabel("Amplitude + offset")
            ax_raw.grid(alpha=0.25)
            ax_proc.grid(alpha=0.25)

            if len(channels) <= 10:
                if display_cfg.show_raw:
                    ax_raw.legend(loc="upper right", fontsize=8)
                if display_cfg.show_processed:
                    ax_proc.legend(loc="upper right", fontsize=8)

        def _update_plot_only_now(self) -> None:
            if self.bundle is None:
                return
            if not self._plot_update_pending:
                return
            self._plot_update_pending = False

            display_cfg = self._current_display_config()
            channels = self._selected_channels()
            mask = self._window_mask()
            if mask.size == 0 or not np.any(mask):
                return

            t = self.bundle.timestamp[mask]
            raw = self.bundle.signal[mask][:, channels]
            proc = self.transformed_signal[mask][:, channels] if self.transformed_signal.size else raw

            ds = display_cfg.downsample_factor
            # Keep plotting responsive on long windows/high sampling rates.
            if t.size > 4000:
                ds = max(ds, int(np.ceil(t.size / 4000.0)))
            if ds > 1:
                t = t[::ds]
                raw = raw[::ds]
                proc = proc[::ds]

            self._render_signal_axes(self.ax_raw, self.ax_proc, t, raw, proc, channels, display_cfg)
            self.canvas.draw_idle()

            if (
                self.external_figure is not None
                and self.external_canvas is not None
                and self.external_ax_raw is not None
                and self.external_ax_proc is not None
            ):
                self._render_signal_axes(
                    self.external_ax_raw,
                    self.external_ax_proc,
                    t,
                    raw,
                    proc,
                    channels,
                    display_cfg,
                )
                self.external_canvas.draw_idle()

        def _update_metadata_tab(self) -> None:
            if self.bundle is None:
                self.info_text.setText("No dataset loaded.")
                return
            preview = {
                "source_type": self.bundle.source_type,
                "source_path": str(self.bundle.source_path),
                "shape": list(self.bundle.signal.shape),
                "sampling_rate": self.bundle.sampling_rate,
                "channels": self.bundle.signal_columns,
                "transform_trace": [asdict(t) for t in self.trace],
                "display": asdict(self._current_display_config()),
            }
            self.info_text.setText(json.dumps(preview, indent=2))

        def _write_quality_report(self) -> None:
            if self.bundle is None:
                return
            report = channel_quality_report(self.bundle.signal)
            report_path = self.bundle.source_path / "viewer_quality_report_preview.json"
            report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            self.statusBar().showMessage(f"Report written: {report_path}", 5000)

        def _export_copy(self) -> None:
            if self.bundle is None:
                return
            self.display_cfg = self._current_display_config()
            try:
                dst = export_transformed_bundle(
                    self.bundle,
                    self.transformed_signal,
                    self.trace,
                    self.display_cfg,
                )
                self.statusBar().showMessage(f"Exported: {dst}", 6000)
            except Exception as exc:
                QMessageBox.critical(self, "Export Failed", str(exc))

        def _select_all_channels(self) -> None:
            for i in range(self.channel_list.count()):
                self.channel_list.item(i).setSelected(True)
            self._update_plot_only()

        def _select_top_channels(self) -> None:
            for i in range(self.channel_list.count()):
                self.channel_list.item(i).setSelected(i < 8)
            self._update_plot_only()

        def _clear_channels(self) -> None:
            for i in range(self.channel_list.count()):
                self.channel_list.item(i).setSelected(False)
            self._update_plot_only()

        def _jump_to_annotation(self) -> None:
            selected = self.annotation_list.selectedItems()
            if not selected:
                return
            start = _safe_float(selected[0].data(Qt.ItemDataRole.UserRole), 0.0)
            self.start_slider.setValue(int(start * 1000.0))
            self._update_plot_only()

    return MainWindow


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Standalone interactive data viewer")
    parser.add_argument("--type", choices=SUPPORTED_TYPES, default="sessions", help="Data type to browse")
    parser.add_argument("--path", type=Path, help="Specific session/game_recording path")
    parser.add_argument("--data-root", type=Path, default=script_default_data_root(), help="Root containing sessions/ and game_recordings/")
    parser.add_argument(
        "--unity-root",
        type=Path,
        default=UNITY_DEFAULT_ROOT,
        help="Root folder (or CSV file) for Unity EMG recordings",
    )
    parser.add_argument("--channels", default="1-8", help="Used in --no-gui mode")
    parser.add_argument("--no-gui", action="store_true", help="Run transform pipeline and print summary without opening UI")
    parser.add_argument("--export", action="store_true", help="Export transformed data copy")
    return parser


def run_no_gui(args: argparse.Namespace) -> int:
    effective_root = args.unity_root.resolve() if args.type == UNITY_SOURCE_TYPE else args.data_root.resolve()
    if args.path is not None:
        selected_path = args.path.resolve()
    else:
        candidates = discover_recordings(effective_root, args.type)
        selected_path = _ask_user_to_pick(candidates, f"Available {args.type} datasets:")

    bundle = load_bundle(args.type, selected_path)
    cfg = TransformConfig()
    display = DisplayConfig()
    transformed, trace = apply_transform_pipeline(bundle.signal, bundle.sampling_rate, cfg)
    report = channel_quality_report(bundle.signal)

    channels = parse_channel_spec(args.channels, bundle.signal.shape[1])
    print(f"Loaded: {bundle.source_path}")
    print(f"Samples x channels: {bundle.signal.shape}")
    print(f"Sampling rate: {bundle.sampling_rate} Hz")
    print(f"Selected channels: {channels[:16]}{'...' if len(channels) > 16 else ''}")
    print(f"Transforms: {[t.name for t in trace]}")
    print(f"Channel report entries: {len(report.get('channels', []))}")

    if args.export:
        dst = export_transformed_bundle(bundle, transformed, trace, display)
        print(f"Exported transformed copy: {dst}")

    return 0


def run_gui(args: argparse.Namespace) -> int:
    try:
        qt = _import_qt_modules()
    except ImportError as exc:
        print("PyQt6/PySide6 is required for GUI mode. Try --no-gui as fallback.")
        print(f"Import error: {exc}")
        return 2

    QApplication = qt["QApplication"]
    MainWindow = _build_main_window_class(qt)

    app = QApplication(sys.argv)
    window = MainWindow(
        data_root=(args.unity_root.resolve() if args.type == UNITY_SOURCE_TYPE else args.data_root.resolve()),
        source_type=args.type,
        initial_path=args.path.resolve() if args.path else None,
    )
    window.show()
    return app.exec()


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.no_gui:
        raise SystemExit(run_no_gui(args))
    raise SystemExit(run_gui(args))


if __name__ == "__main__":
    main()
