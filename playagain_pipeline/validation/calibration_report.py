"""
validation/calibration_report.py
────────────────────────────────
Calibration confidence statistics for Chapter 6 §6.2.

The thesis needs:
* histogram of calibration confidence scores across sessions
* median, IQR
* count of sessions below the flag threshold
* count manually corrected vs excluded
* range of detected rotation offsets

The pipeline already writes ``rotation_confidence`` and
``rotation_offset`` into every session's ``metadata.json`` (see
``THESIS_DOCUMENTATION.md`` §metadata). This module reads those numbers
back and aggregates them.

A session is treated as "calibrated" if its metadata carries a usable
``rotation_confidence`` value — anything else (legacy session, manual
override, calibration skipped) drops out of the denominator and is
counted separately so the chapter's denominator-of-N is honest.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .corpus import SessionCorpus, SessionRecord

log = logging.getLogger(__name__)


# The flag threshold below which sessions are surfaced for manual
# review. Mirrors the value the calibration dialog uses for its
# yellow/red banner; keep them in sync if you tune the UI.
DEFAULT_FLAG_THRESHOLD = 0.50


@dataclass
class SessionCalibration:
    subject_id:  str
    session_id:  str
    confidence:  Optional[float]
    offset:      Optional[int]
    domain:      str
    status:      str          # "ok" | "flagged" | "missing" | "excluded" | "manual"
    # Set to 2 when ``rtation_confidence`` is the new stability
    # metric (written by recompute_calibration_metrics). Plots use
    # this to decide whether axis labels should say "stability" or
    # "confidence (peak prominence)".
    rotation_metric_version: Optional[int] = None


@dataclass
class CalibrationStats:
    """Aggregated stats. ``per_session`` is the raw list for plotting."""
    n_total:        int = 0
    n_calibrated:   int = 0   # have a confidence value
    n_flagged:      int = 0   # below threshold
    n_manual:       int = 0   # flagged and manually corrected
    n_excluded:     int = 0   # flagged and excluded
    flag_threshold: float = DEFAULT_FLAG_THRESHOLD

    # Five-number summary of confidences (calibrated sessions only).
    median:   float = float("nan")
    q1:       float = float("nan")
    q3:       float = float("nan")
    iqr:      float = float("nan")
    min:      float = float("nan")
    max:      float = float("nan")

    # Offset range in channels.
    offset_min: Optional[int] = None
    offset_max: Optional[int] = None
    offset_abs_max: Optional[int] = None

    per_session: List[SessionCalibration] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["per_session"] = [asdict(s) for s in self.per_session]
        return d


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def _read_calibration_fields(
    rec: SessionRecord,
) -> Tuple[Optional[float], Optional[int], str, Optional[int]]:
    """
    Return (confidence, offset, status) from a session's metadata.


    Status values
    -------------
    "ok"        : usable confidence and offset
    "missing"   : no calibration was recorded at all
    "manual"    : confidence was below threshold but a manual offset
                  was entered (custom_metadata.calibration_manual=True)
    "excluded"  : session was tagged for exclusion from the
                  evaluation corpus (custom_metadata.exclude_from_eval=True)
    """
    meta_path = rec.path / "metadata.json"
    try:
        with meta_path.open("r", encoding="utf-8") as f:
            blob = json.load(f)
    except Exception as exc:  # noqa: BLE001
        log.debug("Could not read %s: %s", meta_path, exc)
        return None, None, "missing", None  # error path
    inner = blob.get("metadata") if isinstance(blob, dict) else blob
    if not isinstance(inner, dict):
        return None, None, "missing", None  # not-a-dict path

    custom = inner.get("custom_metadata") or {}

    if bool(custom.get("exclude_from_eval")):
        return None, None, "excluded", None  # excluded path

    conf = inner.get("rotation_confidence")
    off  = inner.get("rotation_offset")

    try:
        metric_version = custom.get("rotation_metric_version")
        metric_version = int(metric_version) if metric_version is not None else None
    except (TypeError, ValueError):
        metric_version = None

    if conf is None and off is None:
        return None, None, "missing", None  # nothing recorded

    try:
        conf_f = float(conf) if conf is not None else None
    except (TypeError, ValueError):
        conf_f = None
    try:
        off_i = int(off) if off is not None else None
    except (TypeError, ValueError):
        off_i = None

    if bool(custom.get("calibration_manual")):
        return conf_f, off_i, "manual", metric_version

    # Even sessions without a stored offset get "ok" if confidence
    # passed validation — the offset may legitimately be 0.
    return conf_f, off_i, "ok", metric_version


def calibration_stats(
    corpus: SessionCorpus,
    *,
    flag_threshold: float = DEFAULT_FLAG_THRESHOLD,
) -> CalibrationStats:
    """
    Build :class:`CalibrationStats` for §6.2 of the thesis.

    Sessions below ``flag_threshold`` are counted as flagged. The split
    of flagged sessions into "manual" vs "excluded" comes from each
    session's ``custom_metadata``; sessions without those flags are
    counted as flagged-but-unresolved (and surface as ``status="flagged"``).
    """
    stats = CalibrationStats(flag_threshold=flag_threshold)
    confidences: List[float] = []
    offsets: List[int] = []

    for rec in corpus.all():
        conf, off, status, metric_version = _read_calibration_fields(rec)
        stats.n_total += 1

        # Promote ok-but-low-confidence to "flagged" before recording.
        if status == "ok" and conf is not None and conf < flag_threshold:
            status = "flagged"

        stats.per_session.append(SessionCalibration(
            subject_id=rec.subject_id,
            session_id=rec.session_id,
            confidence=conf,
            offset=off,
            domain=rec.source_domain,
            status=status,
            rotation_metric_version=metric_version,
        ))

        if status == "excluded":
            stats.n_excluded += 1
            continue
        if conf is None:
            # "missing" — drop from the calibrated denominator.
            continue

        stats.n_calibrated += 1
        confidences.append(conf)
        if off is not None:
            offsets.append(off)

        if status == "manual":
            stats.n_manual  += 1
            stats.n_flagged += 1
        elif status == "flagged":
            stats.n_flagged += 1

    if confidences:
        arr = np.asarray(confidences, dtype=np.float64)
        stats.median = float(np.median(arr))
        stats.q1     = float(np.quantile(arr, 0.25))
        stats.q3     = float(np.quantile(arr, 0.75))
        stats.iqr    = float(stats.q3 - stats.q1)
        stats.min    = float(arr.min())
        stats.max    = float(arr.max())
    if offsets:
        oa = np.asarray(offsets, dtype=np.int64)
        stats.offset_min = int(oa.min())
        stats.offset_max = int(oa.max())
        stats.offset_abs_max = int(np.abs(oa).max())

    return stats


# ---------------------------------------------------------------------------
# Bundled writer
# ---------------------------------------------------------------------------

def write_calibration_report(
    corpus: SessionCorpus,
    out_dir: Path,
    *,
    flag_threshold: float = DEFAULT_FLAG_THRESHOLD,
) -> Dict[str, Path]:
    """
    Persist the calibration report as JSON + a per-session CSV.

    The JSON is what :mod:`plots_thesis` consumes to draw the histogram
    in §6.2; the CSV is convenient for spot-checking individual
    sessions that triggered the flag threshold.
    """
    import csv

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stats = calibration_stats(corpus, flag_threshold=flag_threshold)

    json_path = out_dir / "calibration_report.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(stats.to_dict(), f, indent=2)

    csv_path = out_dir / "calibration_per_session.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["subject_id", "session_id", "domain",
                    "confidence", "offset", "status", "metric_version"])
        for s in stats.per_session:
            w.writerow([
                s.subject_id, s.session_id, s.domain,
                "" if s.confidence is None else f"{s.confidence:.6f}",
                "" if s.offset is None else s.offset,
                s.status,
                "" if s.rotation_metric_version is None else s.rotation_metric_version,
            ])

    return {"calibration_json": json_path, "calibration_csv": csv_path}
