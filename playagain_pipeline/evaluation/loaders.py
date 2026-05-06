"""
evaluation/loaders.py
─────────────────────
Discovery and lightweight loading for the three recording sources.

The UI calls :func:`discover_sessions`, :func:`discover_game_recordings`,
:func:`discover_unity_recordings` to populate its pickers — each of
those returns a list of :class:`RecordingDescriptor` without touching
the heavy EMG matrices.

Heavy data is loaded only by the evaluators on demand via the helpers
``load_session_data``, ``load_game_csv``, ``load_unity_csv``.

Path conventions (matching the existing DataManager):
- Sessions:        ``<data_dir>/sessions/<subject>/<session_id>/``
- Unity sessions:  ``<data_dir>/sessions/unity_sessions/<subject>/<session_id>/``
- Game recordings: ``<data_dir>/game_recordings/<subject>/*.csv``
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .core import RecordingDescriptor, RecordingKind

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sessions  (training recordings)
# ---------------------------------------------------------------------------

def discover_sessions(
    data_dir: Path,
    *,
    include_unity: bool = False,
) -> List[RecordingDescriptor]:
    """
    Walk ``<data_dir>/sessions`` and return descriptors for every
    session that has a ``metadata.json`` file.

    By default Unity-derived sessions (under ``sessions/unity_sessions``)
    are excluded — they are exposed via :func:`discover_unity_recordings`
    instead so the evaluator UI can keep the two cleanly separated.
    """
    out: List[RecordingDescriptor] = []
    sessions_dir = Path(data_dir) / "sessions"
    if not sessions_dir.exists():
        return out

    for subject_dir in sorted(sessions_dir.iterdir()):
        if not subject_dir.is_dir():
            continue
        if subject_dir.name == "unity_sessions" and not include_unity:
            continue
        if subject_dir.name == "unity_sessions" and include_unity:
            # Recurse into unity_sessions to expose them too
            for unity_subj in sorted(subject_dir.iterdir()):
                if not unity_subj.is_dir():
                    continue
                out.extend(_scan_session_dir(unity_subj, force_unity=True))
            continue
        out.extend(_scan_session_dir(subject_dir))

    return out


def _scan_session_dir(
    subject_dir: Path,
    *,
    force_unity: bool = False,
) -> List[RecordingDescriptor]:
    """Yield descriptors for every session folder under ``subject_dir``."""
    out: List[RecordingDescriptor] = []
    subject_id = subject_dir.name

    for sess_dir in sorted(subject_dir.iterdir()):
        if not sess_dir.is_dir():
            continue
        meta_path = sess_dir / "metadata.json"
        if not meta_path.exists():
            continue
        try:
            with open(meta_path, "r") as f:
                raw = json.load(f)
        except Exception as exc:
            log.warning("Could not read %s: %s", meta_path, exc)
            continue

        # metadata.json may be either {"metadata": {...}, "trials": [...]}
        # (RecordingSession.save) or a flat dict — handle both.
        meta = raw.get("metadata", raw)
        is_unity = bool(
            force_unity
            or meta.get("custom_metadata", {}).get("is_unity_recording")
            or meta.get("device_name", "").upper() == "UNITY"
        )

        out.append(RecordingDescriptor(
            kind=RecordingKind.SESSION,
            subject_id=subject_id,
            session_id=sess_dir.name,
            path=sess_dir,
            label=f"{subject_id} · {sess_dir.name}",
            meta={
                "device_name":      meta.get("device_name", "?"),
                "num_channels":     int(meta.get("num_channels", 0) or 0),
                "sampling_rate":    int(meta.get("sampling_rate", 0) or 0),
                "gesture_set_name": meta.get("gesture_set_name", "?"),
                "n_trials":         len(raw.get("trials", []) or []),
                "is_unity_recording": is_unity,
            },
        ))
    return out


def load_session_data(desc: RecordingDescriptor) -> Tuple[np.ndarray, Dict[str, Any], List[Dict[str, Any]]]:
    """
    Load (data, meta, trials) for a session descriptor.

    ``data`` is mmap-loaded from ``data.npy`` if present, else read from
    ``data.csv``. Trials are returned as plain dicts so the evaluator
    doesn't need to import the project's RecordingTrial class.
    """
    if desc.kind != RecordingKind.SESSION:
        raise ValueError(f"Expected SESSION descriptor, got {desc.kind}")

    sess_dir = Path(desc.path)
    meta_path = sess_dir / "metadata.json"
    with open(meta_path, "r") as f:
        raw = json.load(f)
    meta   = raw.get("metadata", raw)
    trials = raw.get("trials", []) or []

    npy_path = sess_dir / "data.npy"
    csv_path = sess_dir / "data.csv"
    if npy_path.exists():
        data = np.array(np.load(npy_path, mmap_mode="r"), dtype=np.float32, copy=True)
    elif csv_path.exists():
        # CSV has a single-row header with channel names.
        data = np.loadtxt(csv_path, delimiter=",", skiprows=1, dtype=np.float32)
    else:
        raise FileNotFoundError(f"No data.npy or data.csv under {sess_dir}")

    return data, meta, trials


def load_session_gesture_set(desc: RecordingDescriptor) -> Dict[int, str]:
    """
    Return a ``{label_id: display_name}`` map from a session's
    gesture_set.json. Empty dict if the file is missing.
    """
    gs_path = Path(desc.path) / "gesture_set.json"
    if not gs_path.exists():
        return {}
    try:
        with open(gs_path, "r") as f:
            raw = json.load(f)
    except Exception:
        return {}
    out: Dict[int, str] = {}
    for g in raw.get("gestures", []):
        try:
            out[int(g["label_id"])] = str(g.get("display_name") or g.get("name") or g["label_id"])
        except (KeyError, TypeError, ValueError):
            continue
    return out


# ---------------------------------------------------------------------------
# Game recordings  (subject-level CSVs from GameRecorder)
# ---------------------------------------------------------------------------

def discover_game_recordings(data_dir: Path) -> List[RecordingDescriptor]:
    """
    Walk ``<data_dir>/game_recordings`` and yield a descriptor per CSV.

    Three layouts are accepted (in priority order):

    1. ``<root>/<subject>/<game_session_dir>/recording.csv``
       — what current ``GameRecorder`` writes. The session dir typically
       contains ``recording.csv`` + ``config.json``. The ``session_id``
       is the directory name; the ``subject`` is its parent.

    2. ``<root>/<subject>/<file>.csv``
       — older / flat dump where each subject folder holds one CSV per
       session. The CSV stem is the ``session_id``.

    3. ``<root>/<file>.csv``
       — loose CSVs at the root; subject is guessed from the filename.

    All three coexist gracefully — duplicates (same path) are filtered.
    """
    out: List[RecordingDescriptor] = []
    root = Path(data_dir) / "game_recordings"
    if not root.exists():
        return out

    seen: set = set()

    def _emit(csv_path: Path, subject: str, session_id: str) -> None:
        if csv_path in seen:
            return
        seen.add(csv_path)
        cfg = _read_game_config(csv_path)
        rec_cfg = cfg.get("recording", {}) or {}
        out.append(RecordingDescriptor(
            kind=RecordingKind.GAME,
            subject_id=subject,
            session_id=session_id,
            path=csv_path,
            label=f"{subject} · {session_id}",
            meta={
                "duration_seconds": rec_cfg.get("duration_seconds"),
                "total_samples":    rec_cfg.get("total_samples"),
                "model_name":       (cfg.get("model", {}) or {}).get("name"),
                "n_classes":        len(rec_cfg.get("class_names", []) or []),
                "class_names":      list(rec_cfg.get("class_names", []) or []),
                "config_path":      str(_config_path_for(csv_path))
                                    if _config_path_for(csv_path) else None,
            },
        ))

    # ── Layout 1: <subject>/<session_dir>/recording.csv ─────────────
    # We look for any CSV whose parent directory has a sibling
    # config.json — that's the GameRecorder convention. Falling back
    # to the literal name "recording.csv" misses edge cases (e.g.
    # the user's `viewer_export_*` siblings), so we use the config.json
    # heuristic instead.
    for csv_path in sorted(root.rglob("*.csv")):
        session_dir = csv_path.parent
        # Must have a config.json beside it for this to count as layout 1
        if not (session_dir / "config.json").exists():
            continue
        # Skip CSVs at the root or one level down (covered by layout 2/3)
        try:
            rel = session_dir.relative_to(root)
        except ValueError:
            continue
        if len(rel.parts) < 2:
            continue
        subject = rel.parts[0]
        session_id = rel.parts[-1]
        _emit(csv_path, subject, session_id)

    # ── Layout 2: <subject>/<file>.csv ──────────────────────────────
    for subject_dir in sorted(root.iterdir()):
        if not subject_dir.is_dir():
            continue
        for csv_path in sorted(subject_dir.glob("*.csv")):
            _emit(csv_path, subject_dir.name, csv_path.stem)

    # ── Layout 3: <root>/<file>.csv ─────────────────────────────────
    for csv_path in sorted(root.glob("*.csv")):
        _emit(csv_path, _guess_subject_from_filename(csv_path) or "?", csv_path.stem)

    return out


def _config_path_for(csv_path: Path) -> Optional[Path]:
    """
    Find the matching config JSON for a game-recording CSV.

    Two conventions are accepted (in priority order):

    1. ``<session_dir>/config.json`` — current GameRecorder layout where
       each game session lives in its own folder with a sibling config.
    2. ``<csv_stem>_config.json`` / ``<csv_stem>.json`` / ``<csv_stem>.config.json``
       — older flat dump conventions.
    """
    candidates = [
        csv_path.parent / "config.json",
        csv_path.with_name(csv_path.stem + "_config.json"),
        csv_path.with_suffix(".json"),
        csv_path.parent / f"{csv_path.stem}.config.json",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _read_game_config(csv_path: Path) -> Dict[str, Any]:
    p = _config_path_for(csv_path)
    if p is None:
        return {}
    try:
        with open(p, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def _guess_subject_from_filename(csv_path: Path) -> str:
    """Best-effort subject extraction (e.g. ``VP_05_2026...csv`` → ``VP_05``)."""
    parts = csv_path.stem.split("_")
    if len(parts) >= 2 and parts[0].upper() in {"VP", "SUBJECT"}:
        return f"{parts[0]}_{parts[1]}"
    return parts[0] if parts else "?"


def load_game_csv(desc: RecordingDescriptor) -> "GameRecording":
    """Load a single game-recording CSV into structured arrays."""
    if desc.kind != RecordingKind.GAME:
        raise ValueError(f"Expected GAME descriptor, got {desc.kind}")
    return GameRecording.from_csv(Path(desc.path), config=_read_game_config(Path(desc.path)))


# ---------------------------------------------------------------------------
# Unity recordings  (raw Unity CSVs OR converted Unity sessions)
# ---------------------------------------------------------------------------

def discover_unity_recordings(
    source_dir: Path,
    *,
    recurse: bool = True,
) -> List[RecordingDescriptor]:
    """
    Find Unity recordings under ``source_dir``. Two layouts are
    accepted:

    1. **Original Unity CSVs** with columns ``Timestamp, RMS,
       GestureActive, GroundTruth, GroundTruthActive, ...``. We just
       sniff the header — anything matching is exposed as a Unity
       descriptor.
    2. **Converted Unity sessions** (the output of the user's import
       script) — directories with ``metadata.json`` whose
       ``custom_metadata.is_unity_recording`` is true, or whose device
       name is ``"UNITY"``.

    Use ``recurse=True`` to walk the whole subtree (slow on huge
    folders but matches Unity's own ``RecordedData/Users/<id>/`` tree).
    """
    out: List[RecordingDescriptor] = []
    source_dir = Path(source_dir)
    if not source_dir.exists():
        return out

    # Layout 2: converted sessions (recurse only one level deep — the
    # standard layout is unity_sessions/<subject>/<session>/).
    for subject_dir in sorted(source_dir.iterdir()):
        if not subject_dir.is_dir():
            continue
        for sess_dir in sorted(subject_dir.iterdir()) if subject_dir.is_dir() else []:
            if not sess_dir.is_dir():
                continue
            meta_path = sess_dir / "metadata.json"
            if not meta_path.exists():
                continue
            try:
                with open(meta_path, "r") as f:
                    raw = json.load(f)
            except Exception:
                continue
            meta = raw.get("metadata", raw)
            is_unity = (
                bool(meta.get("custom_metadata", {}).get("is_unity_recording"))
                or str(meta.get("device_name", "")).upper() == "UNITY"
            )
            if not is_unity:
                continue
            out.append(RecordingDescriptor(
                kind=RecordingKind.UNITY,
                subject_id=subject_dir.name,
                session_id=sess_dir.name,
                path=sess_dir,
                label=f"{subject_dir.name} · {sess_dir.name}  (converted)",
                meta={
                    "format":           "session",
                    "num_channels":     int(meta.get("num_channels", 0) or 0),
                    "sampling_rate":    int(meta.get("sampling_rate", 0) or 0),
                    "n_trials":         len(raw.get("trials", []) or []),
                },
            ))

    # Layout 1: raw Unity CSVs. Sniff the header to verify.
    glob = source_dir.rglob("*.csv") if recurse else source_dir.glob("*.csv")
    for csv_path in sorted(glob):
        try:
            with open(csv_path, "r") as f:
                header = f.readline()
        except Exception:
            continue
        if not _looks_like_unity_csv(header):
            continue
        # Subject from the path: <...>/Users/<subject>/RecordedData/...  is a common Unity layout
        subj = "?"
        for part in csv_path.parts:
            if part.startswith("VP_") or part.upper().startswith("SUBJECT"):
                subj = part
                break
        out.append(RecordingDescriptor(
            kind=RecordingKind.UNITY,
            subject_id=subj,
            session_id=csv_path.stem,
            path=csv_path,
            label=f"{subj} · {csv_path.stem}  (raw csv)",
            meta={"format": "csv"},
        ))

    return out


def _looks_like_unity_csv(header: str) -> bool:
    """Cheap header sniff for the Unity raw-CSV format."""
    cols = {c.strip().lower() for c in header.split(",")}
    needed = {"timestamp", "rms", "groundtruthactive"}
    return needed.issubset(cols)


# ---------------------------------------------------------------------------
# Game recording container
# ---------------------------------------------------------------------------

class GameRecording:
    """
    Structured view of a game-recording CSV.

    Columns of interest:

    * ``Timestamp``  (float seconds since recording start)
    * ``PredictedGesture`` / ``PredictedGestureId`` / ``Confidence``
    * ``Prob_<ClassName>``    (one column per class)
    * ``GroundTruthActive``   (binary, set by the game when the user
                               is actively performing the requested gesture)
    * ``RawGroundTruth``      (numeric class id of the requested gesture
                               while it's active, otherwise 0/-1 — the
                               authoritative multi-class label)
    * ``RequestedGesture``    (string, what the game asked for)
    * ``EMG_Ch0..ChN``        (raw EMG matrix)
    """

    def __init__(
        self,
        df,                          # pandas.DataFrame (kept untyped to avoid forcing pandas at import time)
        config: Dict[str, Any],
        class_names: List[str],
        emg_columns: List[str],
        prob_columns: List[str],
    ):
        self.df            = df
        self.config        = config
        self.class_names   = class_names
        self.emg_columns   = emg_columns
        self.prob_columns  = prob_columns

    # ------------------------------------------------------------------

    @classmethod
    def from_csv(cls, path: Path, *, config: Optional[Dict[str, Any]] = None) -> "GameRecording":
        import pandas as pd
        df = pd.read_csv(path)
        config = config or {}

        emg_cols  = sorted([c for c in df.columns if c.startswith("EMG_Ch")],
                            key=lambda c: int(c.replace("EMG_Ch", "")))
        prob_cols = [c for c in df.columns if c.startswith("Prob_")]

        # Class names: prefer config.recording.class_names, fall back to
        # the order embedded in the Prob_ columns.
        class_names = (
            (config.get("recording") or {}).get("class_names")
            or [c.replace("Prob_", "") for c in prob_cols]
        )

        return cls(
            df=df, config=config,
            class_names=list(class_names),
            emg_columns=list(emg_cols),
            prob_columns=list(prob_cols),
        )

    # ------------------------------------------------------------------

    @property
    def sampling_rate(self) -> Optional[int]:
        rec = self.config.get("recording") or {}
        sr = rec.get("sampling_rate")
        if sr:
            return int(sr)
        # Estimate from timestamps
        ts = self.df["Timestamp"].to_numpy() if "Timestamp" in self.df.columns else None
        if ts is None or ts.size < 2:
            return None
        dt = float(np.median(np.diff(ts)))
        return int(round(1.0 / dt)) if dt > 0 else None

    def emg_matrix(self) -> np.ndarray:
        """(n_samples, n_channels) float32. Empty array if no EMG cols."""
        if not self.emg_columns:
            return np.empty((0, 0), dtype=np.float32)
        return self.df[self.emg_columns].to_numpy(dtype=np.float32)

    def prob_matrix(self) -> np.ndarray:
        """(n_samples, n_classes) — columns ordered by ``class_names``."""
        if not self.prob_columns:
            return np.empty((0, 0), dtype=np.float32)
        # If class_names was derived from prob_columns the order matches;
        # otherwise re-order to ``class_names``.
        cols = [f"Prob_{c}" for c in self.class_names if f"Prob_{c}" in self.df.columns]
        if not cols:
            cols = self.prob_columns
        return self.df[cols].to_numpy(dtype=np.float32)