"""
corpus.py
─────────
Uniform discovery of every recording session on disk, regardless of which
recorder produced it.

The Python recording pipeline writes sessions under:

    data/sessions/<subject>/<session_id>/
        data.npy
        data.csv
        metadata.json
        gesture_set.json

The Unity C# game (DataManager.cs / DeviceManager.cs) writes the same
layout — but under data/sessions/unity_sessions/...:

    data/sessions/unity_sessions/<subject>/<session_id>/
        data.npy
        ...

This module discovers both, normalises them into the same `SessionRecord`
dataclass, and tags each with a `source_domain` field so experiments can
ask questions like "train on pipeline, test on unity" without any glue
code.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Single-session descriptor
# ---------------------------------------------------------------------------

@dataclass
class SessionRecord:
    """
    A single recording on disk, normalised across sources.

    Attributes
    ----------
    subject_id      : Subject identifier (e.g. ``"VP_01"``).
    session_id      : Session folder name (e.g. ``"2026-03-30_09-32-13_3rep"``).
    path            : Absolute path to the session directory.
    source_domain   : Either ``"pipeline"`` (Python recorder) or
                      ``"unity"`` (C# DataManager). Used by cross-domain CV.
    sampling_rate   : Sampling rate in Hz read from metadata.json.
    num_channels    : EMG channel count read from metadata.json.
    label_names     : ``{int_label: gesture_name}`` mapping, read from
                      metadata.json or gesture_set.json.
    notes           : Free-form notes from metadata.json (may be empty).
    """

    subject_id: str
    session_id: str
    path: Path
    source_domain: str
    sampling_rate: float = 0.0
    num_channels: int = 0
    label_names: Dict[int, str] = field(default_factory=dict)
    notes: str = ""

    # ------------------------------------------------------------------
    # Loaders — kept lazy so the corpus discovery is cheap.
    # ------------------------------------------------------------------

    def load_signal(self) -> np.ndarray:
        """
        Return the raw EMG signal as a ``(n_samples, n_channels)`` array.

        Prefers ``data.npy`` (fast, lossless) and falls back to
        ``data.csv`` if it is missing.
        """
        npy_path = self.path / "data.npy"
        if npy_path.exists():
            arr = np.load(npy_path)
            return self._ensure_samples_first(arr)

        csv_path = self.path / "data.csv"
        if csv_path.exists():
            # CSV is slow but is the only common output of one of the
            # older Unity recorders. Use a permissive loader and drop
            # any timestamp / label columns by keeping only numeric ones.
            import pandas as pd  # local import to keep the package light
            df = pd.read_csv(csv_path)
            num_cols = [c for c in df.columns
                        if np.issubdtype(df[c].dtype, np.number)
                        and c.lower() not in {"timestamp", "time", "label", "gesture"}]
            return df[num_cols].to_numpy()

        raise FileNotFoundError(f"No data.npy or data.csv in {self.path}")

    def load_labels(self) -> Optional[np.ndarray]:
        """
        Return per-sample integer labels if available, else ``None``.

        The Python recorder embeds labels in ``data.csv`` under a
        ``label`` column; the Unity recorder writes a separate
        ``labels.npy``. This method tries both.
        """
        labels_npy = self.path / "labels.npy"
        if labels_npy.exists():
            return np.load(labels_npy)

        csv_path = self.path / "data.csv"
        if csv_path.exists():
            import pandas as pd
            df = pd.read_csv(csv_path)
            for col in ("label", "gesture_id", "ground_truth"):
                if col in df.columns:
                    return df[col].to_numpy()
        return None

    @staticmethod
    def _ensure_samples_first(arr: np.ndarray) -> np.ndarray:
        """Make sure the array is ``(n_samples, n_channels)``."""
        if arr.ndim != 2:
            return arr
        # Heuristic: more samples than channels.
        if arr.shape[0] < arr.shape[1]:
            return arr.T
        return arr

    def to_dict(self) -> dict:
        d = asdict(self)
        d["path"] = str(self.path)
        return d


# ---------------------------------------------------------------------------
# Corpus discovery
# ---------------------------------------------------------------------------

class SessionCorpus:
    """
    Discovers every recording session under a data directory and exposes
    them as a flat, filterable list of :class:`SessionRecord`.

    The class deliberately makes no assumptions about the recorder. It
    looks for any directory containing both ``data.npy`` (or ``data.csv``)
    *and* ``metadata.json``, then infers the source domain from the path:

        */unity_sessions/*  →  ``"unity"``
        otherwise           →  ``"pipeline"``

    Parameters
    ----------
    data_dir : Path
        Root data directory (typically ``playagain_pipeline/data``).
    """

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self._records: List[SessionRecord] = []
        self._discovered = False

    # ------------------------------------------------------------------

    def discover(self, verbose: bool = False) -> List[SessionRecord]:
        """Walk the sessions tree once and cache the result."""
        if self._discovered:
            return self._records

        sessions_root = self.data_dir / "sessions"
        if not sessions_root.exists():
            log.warning("Sessions directory does not exist: %s", sessions_root)
            self._discovered = True
            return self._records

        for meta_path in sessions_root.rglob("metadata.json"):
            session_dir = meta_path.parent
            if not (session_dir / "data.npy").exists() and not (session_dir / "data.csv").exists():
                continue

            rec = self._build_record(session_dir, meta_path)
            if rec is not None:
                self._records.append(rec)
                if verbose:
                    log.info("Discovered %s/%s  (%s)",
                             rec.subject_id, rec.session_id, rec.source_domain)

        # Stable, reproducible ordering — important for deterministic CV folds.
        self._records.sort(key=lambda r: (r.source_domain, r.subject_id, r.session_id))
        self._discovered = True
        return self._records

    # ------------------------------------------------------------------

    def _build_record(self, session_dir: Path, meta_path: Path) -> Optional[SessionRecord]:
        try:
            with meta_path.open("r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception as e:  # noqa: BLE001
            log.warning("Could not read %s: %s", meta_path, e)
            return None

        # Path layout decides the domain. All the unity sessions live
        # under data/sessions/unity_sessions/... so a substring check is
        # robust against subject naming.
        rel = session_dir.relative_to(self.data_dir / "sessions")
        parts = rel.parts
        source_domain = "unity" if "unity_sessions" in parts else "pipeline"

        # Subject inference. Pipeline layout: <subject>/<session_id>.
        # Unity layout (with subject):     unity_sessions/<subject>/<session_id>.
        # Unity layout (legacy, no subj):  unity_sessions/emg/<session_id>  → use 'unity_emg'.
        if source_domain == "unity":
            try:
                idx = parts.index("unity_sessions")
                subject_id = parts[idx + 1] if idx + 1 < len(parts) - 1 else f"unity_{parts[-2]}"
            except ValueError:
                subject_id = "unity"
        else:
            subject_id = parts[0] if len(parts) >= 2 else "unknown"

        session_id = parts[-1]

        return SessionRecord(
            subject_id=subject_id,
            session_id=session_id,
            path=session_dir,
            source_domain=source_domain,
            sampling_rate=float(meta.get("sampling_rate", 0.0)),
            num_channels=int(meta.get("num_channels", 0)),
            label_names={int(k): v for k, v in (meta.get("label_names") or {}).items()},
            notes=str(meta.get("notes", "")),
        )

    # ------------------------------------------------------------------
    # Filtering / querying helpers
    # ------------------------------------------------------------------

    def all(self) -> List[SessionRecord]:
        return list(self.discover())

    def filter(
        self,
        subjects: Optional[Iterable[str]] = None,
        domains: Optional[Iterable[str]] = None,
        min_channels: Optional[int] = None,
        sampling_rate: Optional[float] = None,
    ) -> List[SessionRecord]:
        """
        Return the subset of sessions matching every supplied predicate.

        Any argument left as ``None`` is treated as "do not filter on
        this attribute".
        """
        out: List[SessionRecord] = []
        subj_set = set(subjects) if subjects else None
        dom_set = set(domains) if domains else None
        for rec in self.discover():
            if subj_set is not None and rec.subject_id not in subj_set:
                continue
            if dom_set is not None and rec.source_domain not in dom_set:
                continue
            if min_channels is not None and rec.num_channels < min_channels:
                continue
            if sampling_rate is not None and rec.sampling_rate != sampling_rate:
                continue
            out.append(rec)
        return out

    def subjects(self, domain: Optional[str] = None) -> List[str]:
        recs = self.discover() if domain is None else self.filter(domains=[domain])
        return sorted({r.subject_id for r in recs})

    def summary(self) -> str:
        recs = self.discover()
        per_domain: Dict[str, int] = {}
        for r in recs:
            per_domain[r.source_domain] = per_domain.get(r.source_domain, 0) + 1
        lines = [f"SessionCorpus @ {self.data_dir}"]
        lines.append(f"  total sessions : {len(recs)}")
        for k, v in sorted(per_domain.items()):
            lines.append(f"  {k:9s}      : {v}")
        lines.append(f"  subjects       : {len(self.subjects())}")
        return "\n".join(lines)
