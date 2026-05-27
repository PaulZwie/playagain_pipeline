"""
validation/threshold_report.py
══════════════════════════════
Thesis-side aggregator for the **threshold-gameplay** evaluation.

This file is the bridge between
:mod:`playagain_pipeline.evaluation.threshold_eval` and the chapter-6
outputs. For each Unity raw-CSV recording it reports three performance
perspectives:

  • **As-recorded**  — what the user actually experienced (the game's
    own ``GestureActive`` decisions at recording time).
  • **Profile-threshold** — predictions re-derived from RMS using the
    threshold that ``profile.json`` claims was active at the time.
  • **Optimal-threshold** — F1-optimal threshold chosen post-hoc, i.e.
    the upper bound the RMS signal could have reached.

The gap between *as-recorded* and *optimal* is the quantitative
motivation for the gesture-recognition model — it's the headroom that
manual threshold calibration leaves on the table.

Outputs (under the run's out-dir)
─────────────────────────────────
    table_threshold_gameplay.csv       per-recording table
    table_threshold_pooled.csv         per-cohort + overall summary
    fig_threshold_sweep.csv            pooled ROC + F1 sweep data
    threshold_report.json              settings + counts (machine-readable)

Cohort separation is handled exactly like ``game_report.py``: pass a
``ParticipantGroups`` instance and the pooled table grows a per-cohort
row alongside the ``all`` row.
"""

from __future__ import annotations

import csv
import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

log = logging.getLogger(__name__)

GROUP_ALL = "all"


# ═══════════════════════════════════════════════════════════════════════════
# Aggregated data classes
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ThresholdGroupSummary:
    """One row of the pooled / per-cohort summary table."""
    group:           str
    group_label:     str
    n_subjects:      int
    n_recordings:    int
    n_frames:        int
    asrec_accuracy:    float; asrec_precision:   float
    asrec_recall:      float; asrec_f1:          float
    profile_accuracy:  float; profile_precision: float
    profile_recall:    float; profile_f1:        float
    opt_accuracy:      float; opt_precision:     float
    opt_recall:        float; opt_f1:            float
    auc_pooled:        float
    n_suspect:         int    = 0


@dataclass
class ThresholdReportBundle:
    """Everything ``write_threshold_report`` writes to disk."""
    per_recording: List[Dict[str, Any]]       = field(default_factory=list)
    pooled:        List[ThresholdGroupSummary] = field(default_factory=list)
    sweep:         List[Dict[str, float]]     = field(default_factory=list)
    settings:      Dict[str, Any]             = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════
# Building
# ═══════════════════════════════════════════════════════════════════════════

def build_threshold_report(
    csv_paths: Sequence[Path],
    *,
    profile_path: Optional[Path] = None,
    groups: Optional[Any] = None,         # ParticipantGroups (duck-typed)
    settings: Optional[Any] = None,       # ThresholdEvalSettings
    recording_started_at: Optional[Dict[Path, Any]] = None,
) -> ThresholdReportBundle:
    """
    Run threshold-gameplay evaluation and split it by cohort.

    Returns a bundle in the same shape as
    ``game_report.build_game_report`` so the GUI and thesis-side
    writers stay symmetrical.
    """
    # Lazy imports — these modules may not be importable from every
    # call site (the GUI imports this dialog before the runner is
    # reachable), and we don't want a partial install to break import.
    from playagain_pipeline.evaluation.threshold_eval import (  # noqa: WPS433
        evaluate_threshold_gameplay, ThresholdEvalSettings,
    )
    try:
        from playagain_pipeline.validation.participant_groups import (  # noqa: WPS433
            GROUP_HEALTHY, GROUP_IMPAIRED, GROUP_UNKNOWN, group_label,
        )
    except Exception:                                       # noqa: BLE001
        GROUP_HEALTHY, GROUP_IMPAIRED, GROUP_UNKNOWN = "H", "I", "?"
        def group_label(code: str) -> str:
            return {"H": "healthy", "I": "impaired"}.get(code, "unknown")

    csv_paths = [Path(p) for p in csv_paths]
    s = settings or ThresholdEvalSettings()
    rep = evaluate_threshold_gameplay(
        csv_paths, s,
        profile_path=profile_path,
        recording_started_at=recording_started_at,
    )

    per_rec: List[Dict[str, Any]] = []
    by_group: Dict[str, List[Any]] = {}
    for row in rep.rows:
        code = (groups.group_of(row.subject_id)
                if groups is not None and hasattr(groups, "group_of")
                else GROUP_UNKNOWN)
        d = row.to_dict()
        d["group"] = code
        d["group_label"] = group_label(code)
        per_rec.append(d)
        by_group.setdefault(code, []).append(row)

    pooled: List[ThresholdGroupSummary] = []
    if rep.rows:
        pooled.append(_summarise_group(GROUP_ALL, "all", rep.rows))
    for code in (GROUP_HEALTHY, GROUP_IMPAIRED, GROUP_UNKNOWN):
        rows_g = by_group.get(code) or []
        if not rows_g:
            continue
        pooled.append(_summarise_group(code, group_label(code), rows_g))

    return ThresholdReportBundle(
        per_recording=per_rec,
        pooled=pooled,
        sweep=rep.sweep,
        settings=rep.settings,
    )


def _summarise_group(
    code: str,
    label: str,
    rows: Sequence[Any],
) -> ThresholdGroupSummary:
    """Frame-weighted mean over the rows in one cohort."""
    n_frames   = sum(int(r.n_frames) for r in rows)
    n_subjects = len({r.subject_id for r in rows})
    n_recs     = len(rows)
    n_suspect  = sum(1 for r in rows if r.suspect_threshold)

    def _wmean(field_name: str) -> float:
        num = den = 0.0
        for r in rows:
            v = getattr(r, field_name)
            if v is not None and isinstance(v, float) and math.isfinite(v):
                num += v * r.n_frames
                den += r.n_frames
        return float(num / den) if den else float("nan")

    return ThresholdGroupSummary(
        group=code, group_label=label,
        n_subjects=n_subjects,
        n_recordings=n_recs,
        n_frames=n_frames,
        asrec_accuracy=_wmean("asrec_accuracy"),
        asrec_precision=_wmean("asrec_precision"),
        asrec_recall=_wmean("asrec_recall"),
        asrec_f1=_wmean("asrec_f1"),
        profile_accuracy=_wmean("profile_accuracy"),
        profile_precision=_wmean("profile_precision"),
        profile_recall=_wmean("profile_recall"),
        profile_f1=_wmean("profile_f1"),
        opt_accuracy=_wmean("opt_accuracy"),
        opt_precision=_wmean("opt_precision"),
        opt_recall=_wmean("opt_recall"),
        opt_f1=_wmean("opt_f1"),
        auc_pooled=_wmean("auc"),
        n_suspect=n_suspect,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Writing
# ═══════════════════════════════════════════════════════════════════════════

def write_threshold_report(
    bundle: ThresholdReportBundle,
    out_dir: Path,
) -> Dict[str, Path]:
    """Write the four output files. Returns ``{role: path}``."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    written: Dict[str, Path] = {}

    # 1) Per-recording table
    per_rec_path = out_dir / "table_threshold_gameplay.csv"
    if bundle.per_recording:
        columns = list(bundle.per_recording[0].keys())
        with open(per_rec_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=columns)
            w.writeheader()
            for row in bundle.per_recording:
                w.writerow({k: _fmt(row.get(k)) for k in columns})
    written["per_recording"] = per_rec_path

    # 2) Pooled cohort summary
    pooled_path = out_dir / "table_threshold_pooled.csv"
    with open(pooled_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "group", "group_label", "n_subjects", "n_recordings",
            "n_frames", "n_suspect_thresholds",
            "asrec_accuracy", "asrec_precision", "asrec_recall", "asrec_f1",
            "profile_accuracy", "profile_precision", "profile_recall",
            "profile_f1",
            "opt_accuracy", "opt_precision", "opt_recall", "opt_f1",
            "auc_pooled",
        ])
        for s in bundle.pooled:
            w.writerow([
                s.group, s.group_label,
                s.n_subjects, s.n_recordings, s.n_frames, s.n_suspect,
                _fmt(s.asrec_accuracy),    _fmt(s.asrec_precision),
                _fmt(s.asrec_recall),      _fmt(s.asrec_f1),
                _fmt(s.profile_accuracy),  _fmt(s.profile_precision),
                _fmt(s.profile_recall),    _fmt(s.profile_f1),
                _fmt(s.opt_accuracy),      _fmt(s.opt_precision),
                _fmt(s.opt_recall),        _fmt(s.opt_f1),
                _fmt(s.auc_pooled),
            ])
    written["pooled"] = pooled_path

    # 3) Threshold-sweep CSV (pooled across all recordings)
    sweep_path = out_dir / "fig_threshold_sweep.csv"
    if bundle.sweep:
        with open(sweep_path, "w", newline="", encoding="utf-8") as f:
            cols = ["threshold", "accuracy", "precision", "recall",
                    "f1", "tpr", "fpr"]
            w = csv.writer(f); w.writerow(cols)
            for row in bundle.sweep:
                w.writerow([_fmt(row.get(c)) for c in cols])
    written["sweep"] = sweep_path

    # 4) JSON summary
    json_path = out_dir / "threshold_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "n_recordings": len(bundle.per_recording),
            "n_suspect_thresholds": sum(
                1 for r in bundle.per_recording
                if bool(r.get("suspect_threshold"))
            ),
            "settings": bundle.settings,
            "pooled": [s.__dict__ for s in bundle.pooled],
        }, f, indent=2, default=str)
    written["json"] = json_path

    return written


def _fmt(v: Any) -> str:
    """Render a value for CSV. NaN/inf → empty; floats fixed at 6 decimals."""
    if v is None:
        return ""
    if isinstance(v, float):
        if math.isnan(v) or not math.isfinite(v):
            return ""
        return f"{v:.6f}"
    return str(v)
