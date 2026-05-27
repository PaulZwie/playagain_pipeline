"""
validation/feature_cache.py
═══════════════════════════
Per-session feature cache for the thesis-report and evaluation pipelines.

Why this exists
───────────────
The validation runner re-runs the same expensive pipeline — load NPY,
zero/interpolate bad channels, apply rotation, slice into windows,
extract features — once for every fold of every experiment. For an
80-session LOSO-session run that's ~6,400 session-rewinds per
experiment; an 8-feature ablation suite balloons to ~50,000. The
sessions don't change between folds, so this is pure waste.

This module persists the result of that pipeline per session on disk
and serves it back through ``np.load(..., mmap_mode="r")`` on cache
hits. The cached payload is *exactly* what
``DataManager.create_dataset`` would have produced for that session
alone (windowed features + matching labels), so the runner can
``np.concatenate`` cached chunks for the sessions in a fold and
materialise the same ``X`` it would have built from scratch.

The cache key includes everything that affects the output: windowing,
feature config, bad-channel mode, per-session rotation flag, plus the
session's own ``rotation_offset`` / ``channel_mapping`` / bad-channel
list, plus the underlying ``data.npy`` mtime and size as a cheap
content-change proxy. Mismatch on any of these means a fresh extract,
so changing windowing or features doesn't return stale rows.

Layout on disk
──────────────
::

    <data_dir>/.feature_cache/
        <feature_set_hash>/                          # 12-char prefix
            <subject>__<session>__<session_hash>.npz # arrays
            <subject>__<session>__<session_hash>.meta.json

Each .npz contains:
    X            float32  (n_windows, n_features)
    y            int64    (n_windows,)
    trial_ids    str_  (n_windows,)

The .meta.json captures the full key dict so a human can audit what's
cached. Deleting either file invalidates that single session entry.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Public types
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class CachedSessionFeatures:
    """A cache hit. Arrays are mmap'd; copy before mutating."""
    X:         np.ndarray             # (n_windows, n_features) float32
    y:         np.ndarray             # (n_windows,) int64
    trial_ids: np.ndarray             # (n_windows,) <U …
    n_windows: int
    n_features: int


@dataclass(frozen=True)
class FeatureKey:
    """
    The set of inputs that uniquely determine the cached features for
    a given session. Two extracts with the same key MUST be byte-equal.
    """
    # Window geometry
    window_ms:               int
    stride_ms:               int
    # Feature config
    feature_mode:            str               # "default" | "custom" | "raw"
    feature_names:           Tuple[str, ...]   # sorted tuple for "custom"
    # Preprocessing
    bad_channel_mode:        str               # "interpolate" | "zero"
    use_per_session_rotation: bool
    # Per-session bits — pulled from RecordingSession.metadata at fill time
    session_bad_channels:    Tuple[int, ...] = ()
    session_rotation_offset: int = 0
    session_channel_mapping: Tuple[int, ...] = ()
    num_channels:            int = 0
    # Cheap content-change proxy for data.npy
    data_mtime_ns:           int = 0
    data_size_bytes:         int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "window_ms":               self.window_ms,
            "stride_ms":               self.stride_ms,
            "feature_mode":            self.feature_mode,
            "feature_names":           list(self.feature_names),
            "bad_channel_mode":        self.bad_channel_mode,
            "use_per_session_rotation": self.use_per_session_rotation,
            "session_bad_channels":    list(self.session_bad_channels),
            "session_rotation_offset": self.session_rotation_offset,
            "session_channel_mapping": list(self.session_channel_mapping),
            "num_channels":            self.num_channels,
            "data_mtime_ns":           self.data_mtime_ns,
            "data_size_bytes":         self.data_size_bytes,
        }

    @property
    def feature_set_hash(self) -> str:
        """
        Cross-session hash — every cache entry sharing this hash uses
        the same windowing + features + preprocessing knobs, so they
        can live under one folder. The session-specific parts go into
        ``session_hash`` instead.
        """
        h = hashlib.sha256()
        h.update(repr((
            self.window_ms, self.stride_ms, self.feature_mode,
            self.feature_names, self.bad_channel_mode,
            self.use_per_session_rotation,
        )).encode())
        return h.hexdigest()[:12]

    @property
    def session_hash(self) -> str:
        """Per-session content hash — invalidates this one entry only."""
        h = hashlib.sha256()
        h.update(repr((
            self.session_bad_channels, self.session_rotation_offset,
            self.session_channel_mapping, self.num_channels,
            self.data_mtime_ns, self.data_size_bytes,
        )).encode())
        return h.hexdigest()[:16]


# ═══════════════════════════════════════════════════════════════════════════
# Cache
# ═══════════════════════════════════════════════════════════════════════════

class FeatureCache:
    """
    On-disk cache of per-session features. Thread-safe within a process
    because writes go to a temp file and atomically rename; safe across
    processes against torn reads for the same reason. Concurrent writes
    of the *same* (session, key) pair are not guarded — last writer
    wins, which is fine since they would write identical bytes.
    """

    CACHE_DIRNAME = ".feature_cache"

    def __init__(self, data_dir: Path, *, enabled: bool = True):
        self.data_dir = Path(data_dir)
        self.root = self.data_dir / self.CACHE_DIRNAME
        self.enabled = bool(enabled)
        # In-process counters useful for logging / progress reports.
        self.hits = 0
        self.misses = 0
        self.writes = 0
        self.bytes_written = 0

    # ──────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────

    def key_for(
        self,
        session,                       # RecordingSession
        *,
        window_ms: int,
        stride_ms: int,
        feature_config: Optional[Dict[str, Any]],
        bad_channel_mode: str,
        use_per_session_rotation: bool,
    ) -> FeatureKey:
        """
        Build the cache key for a single session. ``feature_config=None``
        is normalised to mode='raw' which the cache refuses (raw windows
        aren't cached — see ``is_cacheable``).
        """
        fmode, fnames = _normalise_feature_config(feature_config)

        # Pull all the per-session bits that influence the output.
        bad_chs = sorted(
            int(c) for c in
            getattr(session.metadata, "bad_channels", None) or []
        )
        rot = int(getattr(session.metadata, "rotation_offset", 0) or 0)
        mapping = list(getattr(session.metadata, "channel_mapping", None) or [])
        n_ch = int(session.metadata.num_channels)

        # data.npy footprint — quick, robust content proxy.
        src = getattr(session, "_source_dir", None)
        mtime_ns = 0
        size = 0
        if src is not None:
            data_path = Path(src) / "data.npy"
            try:
                st = data_path.stat()
                mtime_ns = int(st.st_mtime_ns)
                size = int(st.st_size)
            except OSError:
                pass

        return FeatureKey(
            window_ms=int(window_ms),
            stride_ms=int(stride_ms),
            feature_mode=fmode,
            feature_names=tuple(fnames),
            bad_channel_mode=str(bad_channel_mode or "interpolate"),
            use_per_session_rotation=bool(use_per_session_rotation),
            session_bad_channels=tuple(bad_chs),
            session_rotation_offset=rot,
            session_channel_mapping=tuple(int(x) for x in mapping),
            num_channels=n_ch,
            data_mtime_ns=mtime_ns,
            data_size_bytes=size,
        )

    @staticmethod
    def is_cacheable(feature_config: Optional[Dict[str, Any]]) -> bool:
        """
        Raw-window mode is not cached — 3D window tensors are huge and
        the runner only uses them for CNN/TCN models that are slow for
        unrelated reasons. Cache the cheap, common case: 2D features.
        """
        if feature_config is None:
            return False
        return (feature_config.get("mode") or "default") != "raw"

    def get(
        self,
        session,
        key: FeatureKey,
    ) -> Optional[CachedSessionFeatures]:
        """Return a cache hit or ``None``."""
        if not self.enabled:
            return None
        npz_path = self._npz_path_for(session, key)
        if not npz_path.exists():
            self.misses += 1
            return None
        try:
            # mmap=True keeps memory cost proportional to what the
            # caller actually reads. The runner will concatenate
            # multiple sessions, which copies; that's OK.
            data = np.load(npz_path, mmap_mode="r", allow_pickle=False)
            X = data["X"]
            y = data["y"]
            tids = data["trial_ids"]
        except Exception as exc:  # noqa: BLE001
            # Corrupted entry — drop and recompute.
            log.warning("Feature cache read failed for %s (%s); "
                        "treating as miss and rebuilding.",
                        npz_path.name, exc)
            try:
                npz_path.unlink()
            except OSError:
                pass
            self.misses += 1
            return None
        self.hits += 1
        return CachedSessionFeatures(
            X=X, y=y, trial_ids=tids,
            n_windows=int(X.shape[0]),
            n_features=int(X.shape[1]) if X.ndim == 2 else 0,
        )

    def put(
        self,
        session,
        key: FeatureKey,
        X: np.ndarray,
        y: np.ndarray,
        trial_ids: Sequence[str],
    ) -> None:
        """Write a cache entry atomically. Silent no-op when disabled."""
        if not self.enabled:
            return
        npz_path = self._npz_path_for(session, key)
        meta_path = npz_path.with_suffix(".meta.json")
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = npz_path.with_suffix(npz_path.suffix + ".tmp")
        try:
            # Compact dtypes. Features are already float32 upstream; we
            # enforce it here so a caller mistake doesn't bloat the
            # cache to 2× size.
            X32 = np.asarray(X, dtype=np.float32)
            y64 = np.asarray(y, dtype=np.int64)
            tids = np.asarray(trial_ids, dtype=np.str_)
            with open(tmp_path, "wb") as f:
                np.savez(f, X=X32, y=y64, trial_ids=tids)
            os.replace(tmp_path, npz_path)
        except Exception as exc:  # noqa: BLE001
            log.warning("Feature cache write failed for %s (%s); "
                        "next fold will rebuild.", npz_path.name, exc)
            try:
                tmp_path.unlink()
            except OSError:
                pass
            return

        # Sidecar metadata — purely for humans/tooling.
        try:
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump({
                    "key": key.to_dict(),
                    "session_id": session.metadata.session_id,
                    "subject_id": session.metadata.subject_id,
                    "n_windows": int(X32.shape[0]),
                    "n_features": int(X32.shape[1]) if X32.ndim == 2 else 0,
                    "created_at_unix": time.time(),
                }, f, indent=2)
        except OSError:
            pass

        try:
            self.bytes_written += npz_path.stat().st_size
        except OSError:
            pass
        self.writes += 1

    # ──────────────────────────────────────────────────────────────────
    # Bookkeeping helpers
    # ──────────────────────────────────────────────────────────────────

    def stats(self) -> Dict[str, int]:
        """Snapshot of cache counters for logging."""
        return {
            "hits":          self.hits,
            "misses":        self.misses,
            "writes":        self.writes,
            "bytes_written": self.bytes_written,
        }

    def clear_all(self) -> int:
        """Delete every cache entry. Returns the number of files removed."""
        if not self.root.exists():
            return 0
        removed = 0
        for p in self.root.rglob("*"):
            if p.is_file():
                try:
                    p.unlink(); removed += 1
                except OSError:
                    pass
        return removed

    def disk_usage_bytes(self) -> int:
        if not self.root.exists():
            return 0
        total = 0
        for p in self.root.rglob("*.npz"):
            try:
                total += p.stat().st_size
            except OSError:
                pass
        return total

    # ──────────────────────────────────────────────────────────────────
    # Internals
    # ──────────────────────────────────────────────────────────────────

    def _npz_path_for(self, session, key: FeatureKey) -> Path:
        subj  = _safe(session.metadata.subject_id)
        sess  = _safe(session.metadata.session_id)
        return (self.root
                / key.feature_set_hash
                / f"{subj}__{sess}__{key.session_hash}.npz")


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _safe(s: str) -> str:
    """Filesystem-safe component (matches the runner's safe_fold_id rule)."""
    return "".join(
        c if (c.isalnum() or c in "._-") else "_"
        for c in str(s)
    )


def _normalise_feature_config(
    feature_config: Optional[Dict[str, Any]],
) -> Tuple[str, List[str]]:
    """
    Reduce the loose feature_config dict to (mode, sorted_feature_names).

    Cache-key reduction for ``EMGFeatureExtractor.extract_features``:

      mode='raw'                             → ('raw', [])
      mode='custom' + name list (any form)   → ('custom', sorted(names))
      anything else                          → ('default', CANONICAL_SIX)

    Custom-mode entries may be bare strings (``"rms"``) or dict
    entries (``{"name": "rms", "params": {...}}``) — both forms are
    produced by callers in the codebase, and the extractor accepts
    either post-fix. Per-feature params are NOT part of the cache
    key today: if you sweep a ZC ``threshold`` knob, clear the cache
    or extend this key first.

    Historical note: an older runner emitted ``mode="features"`` with
    dict entries the extractor couldn't read, which silently fell
    through to the default six-feature stack. The "anything else"
    branch below preserves that mapping so old cache entries from
    pre-fix runs remain valid lookups for default-mode configs; the
    fixed runner now emits ``mode="custom"`` with proper names, so
    fresh entries are written under different keys and the §6.5
    ablation rows can finally be distinguished.
    """
    canonical_six = ["mav", "rms", "ssc", "var", "wl", "zc"]
    if feature_config is None:
        return ("raw", [])
    mode = (feature_config.get("mode") or "default").lower()
    if mode == "raw":
        return ("raw", [])
    if mode == "custom":
        names: List[str] = []
        for n in (feature_config.get("features") or []):
            if isinstance(n, str) and n.strip():
                names.append(n.strip().lower())
            elif isinstance(n, dict):
                name = str(n.get("name", "")).strip().lower()
                if name:
                    names.append(name)
        if not names:
            return ("default", canonical_six)
        return ("custom", sorted(set(names)))
    # default, features, or any other mode → six time-domain features
    return ("default", canonical_six)


# ═══════════════════════════════════════════════════════════════════════════
# Fold-level helper — what the runner actually calls
# ═══════════════════════════════════════════════════════════════════════════

def materialise_split_with_cache(
    records: Sequence,            # List[SessionRecord]
    *,
    cache: FeatureCache,
    data_manager,                 # DataManager
    window_ms: int,
    stride_ms: int,
    feature_config: Optional[Dict[str, Any]],
    use_per_session_rotation: bool,
    bad_channel_mode: str = "interpolate",
    name_suffix: str = "",        # tmp-dataset name disambiguator
) -> Tuple[np.ndarray, np.ndarray, Dict[int, str]]:
    """
    Build ``(X, y, label_names)`` for one fold split (train / val / test)
    by consulting the cache per session and only running the heavy
    pipeline for misses. Cached entries are concatenated as ``float32``.

    Falls back to a single ``data_manager.create_dataset`` call if the
    feature config isn't cacheable (raw windows) or every session
    happens to need rebuilding — the latter is roughly the same cost
    as the original code path, plus one quick stat per session.

    Returns empty arrays when no sessions are cacheable.
    """
    records = list(records)
    if not records:
        return (np.empty((0,)), np.empty((0,), dtype=np.int64), {})

    cacheable = FeatureCache.is_cacheable(feature_config)
    if not cacheable:
        # Defer to the original code path for raw-window models.
        return _fallback_via_create_dataset(
            records, data_manager,
            window_ms=window_ms, stride_ms=stride_ms,
            feature_config=feature_config,
            use_per_session_rotation=use_per_session_rotation,
            name_suffix=name_suffix,
        )

    # Load sessions one at a time so we never hold more than one in
    # memory for the duration of the cache miss path. Cache hits don't
    # touch the raw mmap at all.
    chunks_X: List[np.ndarray] = []
    chunks_y: List[np.ndarray] = []
    label_names: Dict[int, str] = {}

    misses: List[Tuple[int, Any]] = []   # (slot_idx, session_record)
    placeholder_X = [None] * len(records)
    placeholder_y = [None] * len(records)

    # Pass 1 — open each session just long enough to compute its key
    # and try the cache. Sessions are released after lookup.
    for idx, rec in enumerate(records):
        try:
            session = data_manager.load_session(rec.subject_id, rec.session_id)
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not load %s/%s: %s",
                        rec.subject_id, rec.session_id, exc)
            continue
        key = cache.key_for(
            session,
            window_ms=window_ms, stride_ms=stride_ms,
            feature_config=feature_config,
            bad_channel_mode=bad_channel_mode,
            use_per_session_rotation=use_per_session_rotation,
        )
        hit = cache.get(session, key)
        # First valid session also fixes the label_names dict.
        if not label_names:
            try:
                label_names = {
                    g.label_id: g.display_name
                    for g in session.gesture_set.gestures
                }
            except Exception:  # noqa: BLE001
                label_names = {}
        if hit is not None:
            placeholder_X[idx] = hit.X
            placeholder_y[idx] = hit.y
            del session
            continue
        # Cache miss — defer the heavy work to a single batched call
        # below so DataManager.create_dataset's pre-allocation logic
        # still benefits from seeing the session in context.
        misses.append((idx, rec, session, key))

    # Pass 2 — rebuild missing sessions individually so each one writes
    # its own cache entry. Per-session create_dataset calls are not free
    # but they're vastly cheaper than re-extracting on every fold.
    for idx, rec, session, key in misses:
        try:
            ds = data_manager.create_dataset(
                name=f"_feature_cache_fill_{_safe(rec.subject_id)}_"
                     f"{_safe(rec.session_id)}_{name_suffix or 'x'}",
                sessions=[session],
                window_size_ms=window_ms,
                window_stride_ms=stride_ms,
                feature_config=feature_config,
                use_per_session_rotation=use_per_session_rotation,
                bad_channel_mode=bad_channel_mode,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("Feature build failed for %s/%s: %s",
                        rec.subject_id, rec.session_id, exc)
            continue
        X = np.asarray(ds["X"], dtype=np.float32)
        y = np.asarray(ds["y"], dtype=np.int64)
        tids = ds.get("trial_ids", np.array([""] * len(X), dtype=np.str_))
        placeholder_X[idx] = X
        placeholder_y[idx] = y
        if not label_names:
            label_names = ds["metadata"].get("label_names", {}) or {}
        cache.put(session, key, X, y, tids)
        # Drop everything held by this session before we move on.
        del session, ds

    # Drop placeholders for sessions that failed to load entirely.
    for idx in range(len(records)):
        if placeholder_X[idx] is None:
            continue
        chunks_X.append(placeholder_X[idx])
        chunks_y.append(placeholder_y[idx])

    if not chunks_X:
        return (np.empty((0,)), np.empty((0,), dtype=np.int64), label_names)

    # Concatenate. This is where the cached mmap rows get pulled into
    # RAM — at this point we know exactly how big the fold is, so
    # there's nothing more to defer.
    X_out = (np.concatenate(chunks_X, axis=0)
             if len(chunks_X) > 1
             else np.asarray(chunks_X[0]))
    y_out = (np.concatenate(chunks_y, axis=0)
             if len(chunks_y) > 1
             else np.asarray(chunks_y[0]))
    return X_out, y_out, label_names


def _fallback_via_create_dataset(
    records,
    data_manager,
    *,
    window_ms: int,
    stride_ms: int,
    feature_config: Optional[Dict[str, Any]],
    use_per_session_rotation: bool,
    name_suffix: str,
) -> Tuple[np.ndarray, np.ndarray, Dict[int, str]]:
    """Original (uncached) path — used for raw-window models."""
    sessions = []
    for rec in records:
        try:
            sessions.append(
                data_manager.load_session(rec.subject_id, rec.session_id)
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not load %s/%s: %s",
                        rec.subject_id, rec.session_id, exc)
    if not sessions:
        return (np.empty((0,)), np.empty((0,), dtype=np.int64), {})
    ds = data_manager.create_dataset(
        name=f"_validation_tmp_{name_suffix}",
        sessions=sessions,
        window_size_ms=window_ms,
        window_stride_ms=stride_ms,
        feature_config=feature_config,
        use_per_session_rotation=use_per_session_rotation,
    )
    return (
        np.asarray(ds["X"]),
        np.asarray(ds["y"]),
        ds["metadata"].get("label_names", {}) or {},
    )