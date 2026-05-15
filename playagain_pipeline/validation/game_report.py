"""
validation/game_report.py
─────────────────────────
Turn the **game recordings** into thesis-grade performance numbers,
split by participant cohort (healthy vs impaired).

Motivation
──────────
Two things were missing from the original reporting pipeline:

1. **Game recordings were never reported on.** ``generate_thesis_outputs``
   only consumed runner outputs (LOSO folds) and the Unity corpus. But
   game recordings are the only artefact that captures the *deployed*
   system's multi-class behaviour in front of a real user — they log
   ``PredictedGestureId`` against ``RawGroundTruth`` frame by frame.
   The Unity recordings, being RMS-threshold only, can't distinguish
   fist / pinch / tripod at all, so leaning on them for multi-class
   performance is misleading. This module makes the game recordings a
   reported source.

2. **Healthy and impaired participants were pooled.** Every aggregate
   collapsed both cohorts into one number. Here every table is emitted
   per group as well as pooled, using the explicit
   :class:`participant_groups.ParticipantGroups` registry (with
   metadata inference as a fallback).

What this module does *not* do
──────────────────────────────
It does not retrain anything. Game recordings already carry the live
model's predictions, so "evaluating" them is just scoring logged
predictions against logged ground truth — which is exactly what
:func:`evaluation.game_eval.evaluate_games` already does well. This
module is the *aggregation* layer on top: per-recording → per-subject →
per-group → pooled, plus CSV/JSON writers shaped like the rest of
``thesis_reports.py``.

Outputs (via :func:`write_game_report`)
───────────────────────────────────────
    game_report.json                 — everything below, combined
    table_game_performance.csv        — one row per cohort + pooled
    table_game_per_subject.csv        — one row per subject
    fig_game_per_class_f1.csv         — cohort × class F1 (long format)
    fig_game_confusion.json           — per-cohort confusion matrices
                                        (raw + row-normalised / recall)

``generate_thesis_outputs`` renames these to their chapter-specific
filenames; used directly they keep the descriptive names above.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from .participant_groups import (
    GROUP_IMPAIRED, GROUP_HEALTHY, GROUP_UNKNOWN, KNOWN_GROUPS,
    ParticipantGroups, group_label,
)

log = logging.getLogger(__name__)

# Sentinel group id used for the pooled (healthy + impaired) row.
GROUP_ALL = "all"


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class GameGroupSummary:
    """Pooled game-recording performance for one cohort."""
    group:        str                   # "H" / "I" / "?" / "all"
    group_label:  str                   # "healthy" / "impaired" / ...
    n_recordings: int
    n_subjects:   int
    n_frames:     int
    accuracy:            float
    macro_f1:            float
    weighted_f1:         float
    balanced_accuracy:   float
    cohen_kappa:         float
    mcc:                 float
    expected_calibration_error: Optional[float]
    per_class_f1:           Dict[str, float] = field(default_factory=dict)
    confusion_labels:       List[int]        = field(default_factory=list)
    confusion_label_names:  List[str]        = field(default_factory=list)
    confusion_raw:          List[List[int]]  = field(default_factory=list)
    confusion_norm:         List[List[float]] = field(default_factory=list)
    subjects:               List[str]        = field(default_factory=list)
    notes:                  List[str]        = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GameSubjectSummary:
    """Pooled game-recording performance for one subject."""
    subject_id:   str
    group:        str
    group_label:  str
    n_recordings: int
    n_frames:     int
    accuracy:     float
    macro_f1:     float
    model_names:  List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GameReport:
    """The complete game-recording report — pooled, per-group, per-subject."""
    pooled:       Optional[GameGroupSummary]
    per_group:    List[GameGroupSummary] = field(default_factory=list)
    per_subject:  List[GameSubjectSummary] = field(default_factory=list)
    truth_source: str = ""
    notes:        List[str] = field(default_factory=list)
    group_registry_source: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "truth_source": self.truth_source,
            "group_registry_source": self.group_registry_source,
            "pooled":      self.pooled.to_dict() if self.pooled else None,
            "per_group":   [g.to_dict() for g in self.per_group],
            "per_subject": [s.to_dict() for s in self.per_subject],
            "notes":       list(self.notes),
        }


# ---------------------------------------------------------------------------
# Internal: evaluate a set of game descriptors via the existing evaluator
# ---------------------------------------------------------------------------

def _evaluate_descriptors(data_dir: Path, recordings: Sequence[Any], settings):
    """
    Score one bundle of game descriptors with the logged-prediction
    evaluator. Returns the :class:`evaluation.core.EvaluationResult`, or
    ``None`` if nothing usable was produced (the evaluator raises in
    that case — we treat it as "no data" rather than an error).
    """
    if not recordings:
        return None
    from playagain_pipeline.evaluation.game_eval import evaluate_games
    try:
        return evaluate_games(Path(data_dir), list(recordings), settings)
    except RuntimeError as exc:
        # "No usable game recordings" — an empty/odd cohort, not a bug.
        log.info("No game-recording metrics for this bundle: %s", exc)
        return None
    except Exception as exc:  # noqa: BLE001
        log.warning("Game-recording evaluation failed: %s", exc)
        return None


def _row_normalise(matrix: List[List[int]]) -> List[List[float]]:
    """Row-normalise a confusion matrix so the diagonal reads as recall."""
    out: List[List[float]] = []
    for row in matrix:
        total = float(sum(row))
        if total > 0:
            out.append([c / total for c in row])
        else:
            out.append([0.0 for _ in row])
    return out


def _group_summary_from_result(
    group: str,
    result: Any,
    recordings: Sequence[Any],
) -> GameGroupSummary:
    """Fold one :class:`EvaluationResult` into a :class:`GameGroupSummary`."""
    subjects = sorted({str(r.subject_id) for r in recordings})
    per_class = {cm.name: float(cm.f1) for cm in (result.per_class or [])}

    conf_labels: List[int] = []
    conf_names:  List[str] = []
    conf_raw:    List[List[int]] = []
    conf_norm:   List[List[float]] = []
    if result.confusion is not None:
        conf_labels = list(result.confusion.labels)
        conf_names  = list(result.confusion.label_names)
        conf_raw    = [list(map(int, row)) for row in result.confusion.matrix]
        conf_norm   = _row_normalise(conf_raw)

    def _f(x: Optional[float]) -> float:
        return float(x) if (x is not None and math.isfinite(float(x))) else float("nan")

    return GameGroupSummary(
        group=group,
        group_label=group_label(group) if group != GROUP_ALL else "all",
        n_recordings=len(recordings),
        n_subjects=len(subjects),
        n_frames=int(result.n_samples),
        accuracy=_f(result.accuracy),
        macro_f1=_f(result.f1_macro),
        weighted_f1=_f(result.f1_weighted),
        balanced_accuracy=_f(result.balanced_accuracy),
        cohen_kappa=_f(result.cohen_kappa),
        mcc=_f(result.mcc),
        expected_calibration_error=(
            float(result.expected_calibration_error)
            if result.expected_calibration_error is not None else None
        ),
        per_class_f1=per_class,
        confusion_labels=conf_labels,
        confusion_label_names=conf_names,
        confusion_raw=conf_raw,
        confusion_norm=conf_norm,
        subjects=subjects,
        notes=list(result.notes or []),
    )


# ---------------------------------------------------------------------------
# Public builder
# ---------------------------------------------------------------------------

def build_game_report(
    data_dir: Path,
    game_corpus: Any,
    groups: ParticipantGroups,
    *,
    settings: Any = None,
    fallback_resolver: Optional[Callable[[str], str]] = None,
    include_unknown_group: bool = True,
) -> GameReport:
    """
    Build the full healthy/impaired game-recording performance report.

    Parameters
    ----------
    data_dir :
        Pipeline data root (passed through to the evaluator).
    game_corpus :
        A :class:`game_corpus.GameCorpus` (or anything exposing
        ``.all()`` → game ``RecordingDescriptor`` objects).
    groups :
        The :class:`participant_groups.ParticipantGroups` registry.
    settings :
        Optional :class:`evaluation.game_eval.GameEvalSettings`. The
        default scores **logged predictions** against ``RawGroundTruth``
        — the authoritative multi-class label — which is what the
        thesis wants. Pass a custom one only to change the truth source
        or add a confidence filter.
    fallback_resolver :
        Optional ``subject_id -> group_code`` closure used when the
        explicit registry has no entry for a subject. Build one with
        :func:`participant_groups.metadata_group_resolver`.
    include_unknown_group :
        When True (default) a ``"?"`` cohort row is also produced for
        subjects whose group could not be resolved, so nothing is
        silently dropped from the corpus total.

    Returns
    -------
    GameReport
        Pooled + per-group + per-subject summaries. Empty cohorts are
        represented by a zero-count summary rather than omitted, so the
        thesis table always has a healthy *and* an impaired row.
    """
    # Default evaluator settings: logged predictions vs RawGroundTruth.
    if settings is None:
        from playagain_pipeline.evaluation.game_eval import (
            GameEvalSettings, TRUTH_RAW,
        )
        settings = GameEvalSettings(
            mode="logged",
            truth_source=TRUTH_RAW,
            drop_inactive_truth_frames=True,
            per_recording_breakdown=True,
        )

    all_recs = list(game_corpus.all())
    notes: List[str] = []
    if not all_recs:
        notes.append("No game recordings discovered under "
                     f"{Path(data_dir) / 'game_recordings'}.")
        return GameReport(
            pooled=None, per_group=[], per_subject=[],
            truth_source=getattr(settings, "truth_source", ""),
            notes=notes,
            group_registry_source=(str(groups.source) if groups.source else None),
        )

    # ── Bucket recordings by cohort ────────────────────────────────────
    buckets = groups.split_by_group(
        all_recs,
        key=lambda r: r.subject_id,
        fallback=fallback_resolver,
        include_unknown=include_unknown_group,
    )

    # ── Per-group pooled summaries ─────────────────────────────────────
    per_group: List[GameGroupSummary] = []
    group_order = list(KNOWN_GROUPS)
    if include_unknown_group and buckets.get(GROUP_UNKNOWN):
        group_order.append(GROUP_UNKNOWN)

    for code in group_order:
        recs = buckets.get(code, [])
        if not recs:
            # Emit an explicit empty row so the cohort is visibly present.
            per_group.append(GameGroupSummary(
                group=code, group_label=group_label(code),
                n_recordings=0, n_subjects=0, n_frames=0,
                accuracy=float("nan"), macro_f1=float("nan"),
                weighted_f1=float("nan"), balanced_accuracy=float("nan"),
                cohen_kappa=float("nan"), mcc=float("nan"),
                expected_calibration_error=None,
                notes=["No game recordings for this cohort."],
            ))
            continue
        result = _evaluate_descriptors(data_dir, recs, settings)
        if result is None:
            per_group.append(GameGroupSummary(
                group=code, group_label=group_label(code),
                n_recordings=len(recs),
                n_subjects=len({r.subject_id for r in recs}),
                n_frames=0,
                accuracy=float("nan"), macro_f1=float("nan"),
                weighted_f1=float("nan"), balanced_accuracy=float("nan"),
                cohen_kappa=float("nan"), mcc=float("nan"),
                expected_calibration_error=None,
                subjects=sorted({r.subject_id for r in recs}),
                notes=["Evaluator produced no usable frames for this cohort."],
            ))
            continue
        per_group.append(_group_summary_from_result(code, result, recs))

    # ── Pooled (all cohorts) summary ───────────────────────────────────
    pooled_result = _evaluate_descriptors(data_dir, all_recs, settings)
    pooled = (
        _group_summary_from_result(GROUP_ALL, pooled_result, all_recs)
        if pooled_result is not None else None
    )
    if pooled is None:
        notes.append("Pooled evaluation produced no usable frames.")

    # ── Per-subject summaries ──────────────────────────────────────────
    per_subject: List[GameSubjectSummary] = []
    by_subject: Dict[str, List[Any]] = {}
    for r in all_recs:
        by_subject.setdefault(str(r.subject_id), []).append(r)

    for subj in sorted(by_subject):
        recs = by_subject[subj]
        code = groups.group_of(subj, fallback=fallback_resolver)
        result = _evaluate_descriptors(data_dir, recs, settings)
        models = sorted({str(r.meta.get("model_name")) for r in recs
                         if r.meta.get("model_name")})
        if result is None:
            per_subject.append(GameSubjectSummary(
                subject_id=subj, group=code, group_label=group_label(code),
                n_recordings=len(recs), n_frames=0,
                accuracy=float("nan"), macro_f1=float("nan"),
                model_names=models,
            ))
            continue
        per_subject.append(GameSubjectSummary(
            subject_id=subj, group=code, group_label=group_label(code),
            n_recordings=len(recs), n_frames=int(result.n_samples),
            accuracy=float(result.accuracy), macro_f1=float(result.f1_macro),
            model_names=models,
        ))

    # Sort per-subject so cohorts are grouped, healthy first.
    _g_rank = {GROUP_HEALTHY: 0, GROUP_IMPAIRED: 1, GROUP_UNKNOWN: 2}
    per_subject.sort(key=lambda s: (_g_rank.get(s.group, 3), s.subject_id))

    if groups.is_empty:
        notes.append(
            "No explicit participant-group registry was supplied — cohorts "
            "were resolved from session metadata only. Provide a "
            "participant_groups.json for authoritative grouping."
        )

    return GameReport(
        pooled=pooled,
        per_group=per_group,
        per_subject=per_subject,
        truth_source=getattr(settings, "truth_source", ""),
        notes=notes,
        group_registry_source=(str(groups.source) if groups.source else None),
    )


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

def write_game_report(report: GameReport, out_dir: Path) -> Dict[str, Path]:
    """
    Persist the game-recording report as CSV + JSON.

    Returns a ``{key: path}`` dict so the orchestrator can rename the
    files to their chapter-specific names.
    """
    import csv

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}

    # ── table_game_performance.csv ─────────────────────────────────────
    perf_csv = out_dir / "table_game_performance.csv"
    with perf_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["group", "n_subjects", "n_recordings", "n_frames",
                    "accuracy", "macro_f1", "weighted_f1",
                    "balanced_accuracy", "cohen_kappa", "mcc",
                    "expected_calibration_error"])

        def _row(s: GameGroupSummary):
            ece = ("" if s.expected_calibration_error is None
                   else f"{s.expected_calibration_error:.4f}")
            w.writerow([
                s.group_label, s.n_subjects, s.n_recordings, s.n_frames,
                f"{s.accuracy:.4f}", f"{s.macro_f1:.4f}", f"{s.weighted_f1:.4f}",
                f"{s.balanced_accuracy:.4f}", f"{s.cohen_kappa:.4f}",
                f"{s.mcc:.4f}", ece,
            ])

        for s in report.per_group:
            _row(s)
        if report.pooled is not None:
            _row(report.pooled)
    paths["game_performance_csv"] = perf_csv

    # ── table_game_per_subject.csv ─────────────────────────────────────
    subj_csv = out_dir / "table_game_per_subject.csv"
    with subj_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["subject_id", "group", "n_recordings", "n_frames",
                    "accuracy", "macro_f1", "model_names"])
        for s in report.per_subject:
            w.writerow([
                s.subject_id, s.group_label, s.n_recordings, s.n_frames,
                f"{s.accuracy:.4f}", f"{s.macro_f1:.4f}",
                ";".join(s.model_names),
            ])
    paths["game_per_subject_csv"] = subj_csv

    # ── fig_game_per_class_f1.csv ──────────────────────────────────────
    pcf_csv = out_dir / "fig_game_per_class_f1.csv"
    with pcf_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["group", "class", "f1"])
        groups_for_pcf = list(report.per_group)
        if report.pooled is not None:
            groups_for_pcf.append(report.pooled)
        for s in groups_for_pcf:
            for cls_name, v in sorted(s.per_class_f1.items()):
                w.writerow([s.group_label, cls_name, f"{float(v):.4f}"])
    paths["game_per_class_csv"] = pcf_csv

    # ── fig_game_confusion.json ────────────────────────────────────────
    conf_json = out_dir / "fig_game_confusion.json"
    conf_payload: Dict[str, Any] = {}
    groups_for_conf = list(report.per_group)
    if report.pooled is not None:
        groups_for_conf.append(report.pooled)
    for s in groups_for_conf:
        if not s.confusion_labels:
            continue
        conf_payload[s.group_label] = {
            "labels":      s.confusion_labels,
            "label_names": s.confusion_label_names,
            "matrix_raw":  s.confusion_raw,
            "matrix_norm": s.confusion_norm,
            "n_samples":   s.n_frames,
        }
    with conf_json.open("w", encoding="utf-8") as f:
        json.dump(conf_payload, f, indent=2)
    paths["game_confusion_json"] = conf_json

    # ── game_report.json (combined) ────────────────────────────────────
    combined_json = out_dir / "game_report.json"
    with combined_json.open("w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2)
    paths["game_report_json"] = combined_json

    return paths