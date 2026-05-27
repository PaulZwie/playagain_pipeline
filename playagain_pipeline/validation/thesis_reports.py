"""
validation/thesis_reports.py
────────────────────────────
Aggregate :class:`RunResult` objects into the tables and per-fold
breakdowns referenced from Chapters 6, 7 and 8 of the thesis.

The runner already captures the raw per-fold numbers
(``accuracy``, ``macro_f1``, ``per_class_f1``, ``train_seconds``,
``inference_ms``, ``confusion`` …). This module takes one or more
:class:`RunResult` objects and computes the cross-fold statistics, the
per-class error bars, the normalised confusion matrices and the
multi-run comparisons that the chapters refer to.

Design choices
──────────────
* **No GUI**, no Qt, no torch. The chapter generators are pure data
  transformations so they can run on a CI box and produce the
  numbers a thesis editor will paste into LaTeX.
* **Operates on existing data structures.** Nothing here re-runs the
  models; it consumes whatever the runner already wrote.
* **All outputs are JSON / CSV / Python dataclasses.** Figures live in
  :mod:`plots_thesis`, which sits on top of this module.

Inputs
──────
Most functions accept a :class:`RunResult`. To consume previously-saved
runs from disk, use :func:`load_run_result`, which reads
``results.json`` plus ``session_index.json`` (the runner writes both).

What maps to which chapter
──────────────────────────
* §6.3.1 ``loso_session_summary``       → Table 6.3 (headline LOSO)
* §6.3.2 ``per_class_f1_summary``       → Fig. 6.3 data
* §6.3.3 ``aggregate_confusion_norm``   → Fig. 6.4 data (row-normalised)
* §6.3.4 ``per_session_variability``    → Fig. 6.5 data (box plot data)
* §6.4   ``loso_subject_summary``       → Table 6.4
* §6.5   ``feature_ablation``           → Table 6.5 + Fig. 6.6
* §6.6   ``cross_domain_comparison``    → Table 6.6
* §6.7   ``latency_table``              → Table 6.7
* §7.4   ``calibration_f1_correlation`` → Discussion-only correlation
"""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .participant_groups import (
    GROUP_IMPAIRED, GROUP_HEALTHY, GROUP_UNKNOWN, ParticipantGroups, group_label,
)

log = logging.getLogger(__name__)

# Group id used for folds whose held-out subjects span both cohorts.
GROUP_MIXED = "mixed"


# ---------------------------------------------------------------------------
# Lightweight loaders — so this module is usable on saved runs without
# pulling in the runner's heavy ML dependencies.
# ---------------------------------------------------------------------------

@dataclass
class FoldStub:
    """A read-only view of one FoldResult, deserialised from JSON."""
    fold_id:         str
    model_type:      str
    accuracy:        float
    macro_f1:        float
    per_class_f1:    Dict[str, float] = field(default_factory=dict)
    train_seconds:   float = 0.0
    inference_ms:    float = 0.0
    n_train_windows: int = 0
    n_test_windows:  int = 0
    confusion:        Optional[List[List[int]]] = None
    confusion_labels: Optional[List[int]] = None
    label_names:      Dict[int, str] = field(default_factory=dict)
    # Optional provenance, useful for per-session breakdowns. Populated
    # by the runner patch in this bundle; absent on older runs.
    test_subjects: List[str] = field(default_factory=list)
    test_sessions: List[Tuple[str, str]] = field(default_factory=list)
    split_kind:    str = ""


@dataclass
class RunStub:
    """The minimum we need from a run on disk."""
    name:        str
    folds:       List[FoldStub] = field(default_factory=list)
    records:     List[Dict[str, Any]] = field(default_factory=list)  # session_index.json
    experiment:  Dict[str, Any] = field(default_factory=dict)
    output_dir:  Optional[Path] = None


def load_run_result(run_dir: Path) -> RunStub:
    """
    Read a runner output folder (``results.json`` + ``session_index.json``)
    back into a :class:`RunStub`.
    """
    run_dir = Path(run_dir)
    res_path = run_dir / "results.json"
    if not res_path.exists():
        raise FileNotFoundError(f"No results.json in {run_dir}")
    with res_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    folds: List[FoldStub] = []
    for fr in raw.get("folds", []) or []:
        # ``label_names`` keys are JSON strings — coerce back to int.
        ln = fr.get("label_names") or {}
        try:
            label_names = {int(k): str(v) for k, v in ln.items()}
        except (TypeError, ValueError):
            label_names = {}
        folds.append(FoldStub(
            fold_id=str(fr.get("fold_id", "")),
            model_type=str(fr.get("model_type", "")),
            accuracy=float(fr.get("accuracy", float("nan"))),
            macro_f1=float(fr.get("macro_f1", float("nan"))),
            per_class_f1=dict(fr.get("per_class_f1") or {}),
            train_seconds=float(fr.get("train_seconds", 0.0)),
            inference_ms=float(fr.get("inference_ms", 0.0)),
            n_train_windows=int(fr.get("n_train_windows", 0)),
            n_test_windows=int(fr.get("n_test_windows", 0)),
            confusion=fr.get("confusion"),
            confusion_labels=fr.get("confusion_labels"),
            label_names=label_names,
            test_subjects=list(fr.get("test_subjects") or []),
            test_sessions=[tuple(p) for p in (fr.get("test_sessions") or [])],
            split_kind=str(fr.get("split_kind") or ""),
        ))

    idx_path = run_dir / "session_index.json"
    records: List[Dict[str, Any]] = []
    if idx_path.exists():
        with idx_path.open("r", encoding="utf-8") as f:
            records = (json.load(f) or {}).get("records") or []

    exp_path = run_dir / "experiment.json"
    experiment: Dict[str, Any] = {}
    if exp_path.exists():
        with exp_path.open("r", encoding="utf-8") as f:
            experiment = json.load(f) or {}

    return RunStub(
        name=str(experiment.get("name") or run_dir.name),
        folds=folds,
        records=records,
        experiment=experiment,
        output_dir=run_dir,
    )


# ---------------------------------------------------------------------------
# Fold-id parsing — pulls subject/session out of legacy folds that
# didn't carry test_subjects/test_sessions explicitly.
# ---------------------------------------------------------------------------

# fold_id grammar produced by cv_strategies.py:
#   "within__{subject}__{session}"
#   "loso_session__{subject}__{session}"
#   "intra_loso_session__{subject}__{session}"
#   "loso_subject__{subject}"
#   "crossdomain__{train_domain}_to_{test_domain}"
#   "kfold_subjects__k{k}_seed{seed}__fold{fi}"
#   "holdout__val{V}_test{T}__seed{S}__strat-{X}"

# The session-level regex accepts the pooled (``loso_session``),
# domain-restricted (``within``) and per-participant
# (``intra_loso_session``) variants — all three carry a
# ``…__{subject}__{session}`` tail, so the same subject/session
# extraction works for every one.
_LOSO_SESS_RE = re.compile(
    r"^(?:within|loso_session|intra_loso_session)__"
    r"(?P<subj>[^_].*?)__(?P<sess>.+)$"
)
_LOSO_SUBJ_RE = re.compile(r"^loso_subject__(?P<subj>.+)$")


def parse_fold_subjects(f: FoldStub) -> List[str]:
    """
    Best-effort extraction of the held-out subject(s) for one fold.

    Falls back from the explicit ``test_subjects`` list (new runs)
    to the ``fold_id`` parse (legacy runs).
    """
    if f.test_subjects:
        return list(dict.fromkeys(f.test_subjects))  # de-dup keep order

    m = _LOSO_SESS_RE.match(f.fold_id)
    if m:
        return [m.group("subj")]
    m = _LOSO_SUBJ_RE.match(f.fold_id)
    if m:
        return [m.group("subj")]
    return []


def parse_fold_session(f: FoldStub) -> Optional[Tuple[str, str]]:
    """Return ``(subject_id, session_id)`` for single-session folds, else None."""
    if f.test_sessions:
        return f.test_sessions[0]
    m = _LOSO_SESS_RE.match(f.fold_id)
    if m:
        return (m.group("subj"), m.group("sess"))
    return None


# ---------------------------------------------------------------------------
# Per-model summary  →  Tables 6.3, 6.4
# ---------------------------------------------------------------------------

@dataclass
class ModelSummary:
    """Cross-fold summary statistics for one model in one run."""
    model_type:           str
    n_folds:              int
    accuracy_mean:        float
    accuracy_std:         float
    macro_f1_mean:        float
    macro_f1_std:         float
    train_seconds_mean:   float
    train_seconds_std:    float
    inference_ms_mean:    float
    inference_ms_std:     float
    n_test_windows_total: int

    def fmt_mean_sd(self, key: str, *, digits: int = 3) -> str:
        m = getattr(self, f"{key}_mean")
        s = getattr(self, f"{key}_std")
        if not math.isfinite(m):
            return "—"
        return f"{m:.{digits}f} ± {s:.{digits}f}"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _stats(xs: Sequence[float]) -> Tuple[float, float]:
    if not xs:
        return float("nan"), float("nan")
    arr = np.asarray(xs, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=0))


def summarise_run(run: RunStub) -> Dict[str, ModelSummary]:
    """
    Compute mean ± SD across folds for every model in the run.

    Used to fill Table 6.3 (LOSO-session) and Table 6.4 (LOSO-subject)
    — the same function, applied to differently-folded runs.
    """
    by_model: Dict[str, List[FoldStub]] = {}
    for fr in run.folds:
        by_model.setdefault(fr.model_type, []).append(fr)

    out: Dict[str, ModelSummary] = {}
    for model, fs in by_model.items():
        accs = [f.accuracy for f in fs]
        f1s  = [f.macro_f1 for f in fs]
        ts   = [f.train_seconds for f in fs]
        ifs  = [f.inference_ms  for f in fs]
        am, asd = _stats(accs)
        fm, fsd = _stats(f1s)
        tm, tsd = _stats(ts)
        im, isd = _stats(ifs)
        out[model] = ModelSummary(
            model_type=model,
            n_folds=len(fs),
            accuracy_mean=am, accuracy_std=asd,
            macro_f1_mean=fm, macro_f1_std=fsd,
            train_seconds_mean=tm, train_seconds_std=tsd,
            inference_ms_mean=im,  inference_ms_std=isd,
            n_test_windows_total=int(sum(f.n_test_windows for f in fs)),
        )
    return out


# ---------------------------------------------------------------------------
# Per-class F1 with SD across folds  →  Fig 6.3
# ---------------------------------------------------------------------------

@dataclass
class PerClassPoint:
    model_type:  str
    class_name:  str
    mean:        float
    std:         float
    n_folds:     int


def per_class_f1_summary(run: RunStub) -> List[PerClassPoint]:
    """
    For every (model, class) pair, return mean and SD of F1 across the
    folds in which that class was present. Missing values are dropped
    rather than counted as zero — penalising a model for a class that
    happened not to appear in a particular held-out fold would be a
    misleading way to compute the error bar.
    """
    by_model_class: Dict[Tuple[str, str], List[float]] = {}
    for f in run.folds:
        for cls, v in f.per_class_f1.items():
            if v is None or (isinstance(v, float) and math.isnan(v)):
                continue
            by_model_class.setdefault((f.model_type, cls), []).append(float(v))

    out: List[PerClassPoint] = []
    for (model, cls), values in by_model_class.items():
        m, s = _stats(values)
        out.append(PerClassPoint(
            model_type=model, class_name=cls,
            mean=m, std=s, n_folds=len(values),
        ))
    # Stable order: model first, then class alphabetically.
    out.sort(key=lambda p: (p.model_type, p.class_name))
    return out


# ---------------------------------------------------------------------------
# Aggregate normalised confusion matrix  →  Fig 6.4
# ---------------------------------------------------------------------------

@dataclass
class NormalisedConfusion:
    """Per-class recall normalised confusion matrix accumulated across folds."""
    model_type: str
    labels:      List[int]
    label_names: List[str]
    matrix_raw:  List[List[int]]
    matrix_norm: List[List[float]]
    n_samples:   int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def aggregate_confusion(run: RunStub) -> Dict[str, NormalisedConfusion]:
    """
    Sum per-fold confusion matrices then row-normalise so diagonal cells
    show per-class recall. Matches the wording in §6.3.3 ("cell values
    are the fraction of test windows in each row classified into each
    column" → that's recall).

    Folds that didn't see every class are padded with zero rows/cols
    before summing.
    """
    by_model: Dict[str, List[FoldStub]] = {}
    for f in run.folds:
        if f.confusion is None or f.confusion_labels is None:
            continue
        by_model.setdefault(f.model_type, []).append(f)

    out: Dict[str, NormalisedConfusion] = {}
    for model, fs in by_model.items():
        all_labels = sorted({int(l) for f in fs for l in f.confusion_labels})
        if not all_labels:
            continue
        idx = {l: i for i, l in enumerate(all_labels)}
        acc = np.zeros((len(all_labels), len(all_labels)), dtype=np.int64)
        for fr in fs:
            cm = np.asarray(fr.confusion, dtype=np.int64)
            for i, li in enumerate(fr.confusion_labels):
                for j, lj in enumerate(fr.confusion_labels):
                    acc[idx[int(li)], idx[int(lj)]] += cm[i, j]

        # Row-normalise (per-class recall). Empty rows → all zeros.
        row_sums = acc.sum(axis=1, keepdims=True).astype(np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            norm = np.where(row_sums > 0, acc / row_sums, 0.0)

        # Name resolution: take the first non-empty label_names dict.
        names_dict: Dict[int, str] = {}
        for f in fs:
            if f.label_names:
                names_dict.update({int(k): str(v) for k, v in f.label_names.items()})
                break
        names = [names_dict.get(l, f"class_{l}") for l in all_labels]

        out[model] = NormalisedConfusion(
            model_type=model,
            labels=all_labels,
            label_names=names,
            matrix_raw=acc.tolist(),
            matrix_norm=norm.tolist(),
            n_samples=int(acc.sum()),
        )
    return out


# ---------------------------------------------------------------------------
# Per-session variability  →  Fig 6.5
# ---------------------------------------------------------------------------

@dataclass
class PerSubjectFolds:
    """All fold F1 values that belong to one subject, for one model."""
    subject_id: str
    model_type: str
    fold_ids:   List[str]
    f1_values:  List[float]


def per_session_variability(
    run: RunStub,
    *,
    model: Optional[str] = None,
) -> List[PerSubjectFolds]:
    """
    Group per-fold macro F1 by held-out subject, for plotting Fig 6.5.

    Only meaningful for LOSO-session and similar strategies where each
    fold holds out a single session of a single subject. For folds that
    span multiple subjects (cross-domain, k-fold subjects with k<N)
    the subject list is collapsed by ``parse_fold_subjects`` which
    yields the full set; those folds end up in every relevant subject's
    bucket. The function exposes the raw fold lists so callers can
    filter further.
    """
    out: Dict[Tuple[str, str], PerSubjectFolds] = {}
    for f in run.folds:
        if model and f.model_type != model:
            continue
        subjects = parse_fold_subjects(f)
        if not subjects:
            continue
        for s in subjects:
            key = (s, f.model_type)
            if key not in out:
                out[key] = PerSubjectFolds(
                    subject_id=s, model_type=f.model_type,
                    fold_ids=[], f1_values=[],
                )
            out[key].fold_ids.append(f.fold_id)
            out[key].f1_values.append(float(f.macro_f1))
    return sorted(out.values(), key=lambda p: (p.model_type, p.subject_id))


# ---------------------------------------------------------------------------
# Feature ablation  →  Table 6.5
# ---------------------------------------------------------------------------

@dataclass
class FeatureAblationRow:
    condition:        str       # e.g. "rms", "mav", ..., "combined"
    vector_length:    int
    model_type:       str
    n_folds:          int
    macro_f1_mean:    float
    macro_f1_std:     float
    accuracy_mean:    float
    accuracy_std:     float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def feature_ablation(
    runs:    Sequence[Tuple[str, RunStub]],
    *,
    model:   str,
) -> List[FeatureAblationRow]:
    """
    Build Table 6.5 from a list of ``(condition_name, RunStub)`` pairs.

    The convention is that each run contains exactly one feature
    condition. ``model`` selects which model's results to read (the
    thesis uses the best-performing classical model — typically
    CatBoost). The vector length is read from the experiment config's
    feature list when available, falling back to a heuristic count of
    ``n_features × 32`` channels.
    """
    rows: List[FeatureAblationRow] = []
    for name, run in runs:
        summaries = summarise_run(run)
        s = summaries.get(model)
        if s is None:
            log.warning("Run %r has no results for model %r", run.name, model)
            continue
        feats = run.experiment.get("features") or []
        n_channels = int((run.experiment.get("windowing") or {}).get("num_channels") or 32)
        vec_len = sum(int((f.get("params") or {}).get("vector_length") or n_channels)
                      for f in feats) or len(feats) * n_channels
        rows.append(FeatureAblationRow(
            condition=name,
            vector_length=int(vec_len),
            model_type=model,
            n_folds=s.n_folds,
            macro_f1_mean=s.macro_f1_mean,
            macro_f1_std=s.macro_f1_std,
            accuracy_mean=s.accuracy_mean,
            accuracy_std=s.accuracy_std,
        ))
    # Sort: single features first by descending F1, combined last.
    def _sort_key(r: FeatureAblationRow):
        is_combined = "combined" in r.condition.lower() or "all" in r.condition.lower()
        return (1 if is_combined else 0, -r.macro_f1_mean)
    rows.sort(key=_sort_key)
    return rows


# ---------------------------------------------------------------------------
# Cross-domain  →  Table 6.6
# ---------------------------------------------------------------------------

@dataclass
class CrossDomainCell:
    train_domain: str        # "pipeline" | "unity"
    test_domain:  str
    model_type:   str
    n_folds:      int
    macro_f1_mean: float
    macro_f1_std:  float


def cross_domain_comparison(
    within_pipeline: Optional[RunStub],
    within_unity:    Optional[RunStub],
    pipeline_to_unity: Optional[RunStub],
    unity_to_pipeline: Optional[RunStub],
    *,
    model: str,
) -> List[CrossDomainCell]:
    """
    Pivot four RunStubs into one comparison table.

    Any of the four arguments may be ``None`` if that condition wasn't
    run; the returned list simply omits missing rows. The model name
    must match one of the model_types present in the runs (typically
    the best classical model from §6.3).
    """
    out: List[CrossDomainCell] = []

    def _add(run: Optional[RunStub], train: str, test: str) -> None:
        if run is None:
            return
        s = summarise_run(run).get(model)
        if s is None:
            log.warning("Cross-domain run for %s→%s has no %s results", train, test, model)
            return
        out.append(CrossDomainCell(
            train_domain=train, test_domain=test, model_type=model,
            n_folds=s.n_folds,
            macro_f1_mean=s.macro_f1_mean,
            macro_f1_std=s.macro_f1_std,
        ))

    _add(within_pipeline,   "pipeline", "pipeline")
    _add(within_unity,      "unity",    "unity")
    _add(pipeline_to_unity, "pipeline", "unity")
    _add(unity_to_pipeline, "unity",    "pipeline")
    return out


# ---------------------------------------------------------------------------
# Latency / training-time table  →  Table 6.7
# ---------------------------------------------------------------------------

@dataclass
class LatencyRow:
    model_type:        str
    train_seconds_mean: float
    inference_ms_mean:  float
    inference_ms_max:   float
    passes_gate:       bool


def latency_table(run: RunStub, *, gate_ms: float = 150.0) -> List[LatencyRow]:
    """
    One row per model with training time and the worst-case per-fold
    inference latency. ``passes_gate`` is the boolean the thesis prints
    in the rightmost column of Table 6.7 — we compare the **mean** per
    the chapter wording, but also report the maximum observed because
    that's the figure that actually matters for real-time use.
    """
    by_model: Dict[str, List[FoldStub]] = {}
    for f in run.folds:
        by_model.setdefault(f.model_type, []).append(f)

    rows: List[LatencyRow] = []
    for model, fs in by_model.items():
        ts = [f.train_seconds for f in fs]
        ifs = [f.inference_ms for f in fs]
        tm, _ = _stats(ts)
        im, _ = _stats(ifs)
        im_max = float(max(ifs)) if ifs else float("nan")
        rows.append(LatencyRow(
            model_type=model,
            train_seconds_mean=tm,
            inference_ms_mean=im,
            inference_ms_max=im_max,
            passes_gate=bool(math.isfinite(im) and im < gate_ms),
        ))
    rows.sort(key=lambda r: r.inference_ms_mean
              if math.isfinite(r.inference_ms_mean) else float("inf"))
    return rows


# ---------------------------------------------------------------------------
# Cohort-split aggregation  (healthy vs impaired)  →  §6.3 / §6.4 split tables
# ---------------------------------------------------------------------------
#
# The runner records ``test_subjects`` on every fold. For subject-level
# strategies (LOSO-session, LOSO-subject, within-session) every fold holds
# out exactly one subject, so a fold can be attributed to that subject's
# cohort and the per-model means recomputed per cohort. This is what lets
# the thesis report "macro-F1 was X on healthy and Y on impaired
# participants" instead of a single pooled number that hides the gap.

def fold_group(
    f: FoldStub,
    groups: ParticipantGroups,
    *,
    fallback: Optional[Any] = None,
) -> str:
    """
    Resolve the cohort a fold belongs to.

    Returns a single cohort code (``"H"`` / ``"I"`` / ``"?"``) when all
    of the fold's held-out subjects share a cohort, :data:`GROUP_MIXED`
    when they don't, or ``"?"`` when the held-out subjects can't be
    determined at all.

    ``fallback`` is an optional ``subject_id -> code`` closure (see
    :func:`participant_groups.metadata_group_resolver`).
    """
    subjects = parse_fold_subjects(f)
    if not subjects:
        return GROUP_UNKNOWN
    codes = {groups.group_of(s, fallback=fallback) for s in subjects}
    codes.discard(GROUP_UNKNOWN)
    if not codes:
        return GROUP_UNKNOWN
    if len(codes) == 1:
        return next(iter(codes))
    return GROUP_MIXED


def split_folds_by_group(
    run: RunStub,
    groups: ParticipantGroups,
    *,
    fallback: Optional[Any] = None,
    include_mixed: bool = False,
) -> Dict[str, List[FoldStub]]:
    """
    Bucket a run's folds by the cohort of their held-out subject(s).

    The healthy and impaired buckets are always present (possibly
    empty). ``"?"`` collects folds whose cohort is unknown; ``"mixed"``
    collects folds spanning both cohorts and is only kept when
    ``include_mixed`` is True (otherwise such folds are dropped from the
    split, since attributing them to one cohort would be wrong).
    """
    buckets: Dict[str, List[FoldStub]] = {GROUP_HEALTHY: [], GROUP_IMPAIRED: []}
    for f in run.folds:
        code = fold_group(f, groups, fallback=fallback)
        if code == GROUP_MIXED and not include_mixed:
            continue
        buckets.setdefault(code, []).append(f)
    return buckets


def summarise_run_by_group(
    run: RunStub,
    groups: ParticipantGroups,
    *,
    fallback: Optional[Any] = None,
    include_mixed: bool = False,
) -> Dict[str, Dict[str, ModelSummary]]:
    """
    Per-cohort version of :func:`summarise_run`.

    Returns ``{group_code: {model_type: ModelSummary}}``. Reuses the
    exact same cross-fold statistics as the pooled summary — only the
    set of folds entering each mean changes — so a cohort row is
    directly comparable to the pooled Table 6.3 / 6.4 row.
    """
    out: Dict[str, Dict[str, ModelSummary]] = {}
    for code, folds in split_folds_by_group(
        run, groups, fallback=fallback, include_mixed=include_mixed,
    ).items():
        # Reuse summarise_run by wrapping the fold subset in a throwaway
        # RunStub — keeps the statistics in exactly one place.
        sub = RunStub(name=f"{run.name}::{code}", folds=folds,
                      records=run.records, experiment=run.experiment,
                      output_dir=run.output_dir)
        out[code] = summarise_run(sub)
    return out


@dataclass
class GroupModelRow:
    """One row of a cohort-split model-summary table."""
    group:        str          # "H" / "I" / "?" / "mixed"
    group_label:  str          # "healthy" / "impaired" / ...
    model_type:   str
    n_folds:      int
    n_subjects:   int
    accuracy_mean:  float
    accuracy_std:   float
    macro_f1_mean:  float
    macro_f1_std:   float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def group_model_rows(
    run: RunStub,
    groups: ParticipantGroups,
    *,
    fallback: Optional[Any] = None,
    include_mixed: bool = False,
) -> List[GroupModelRow]:
    """
    Flatten :func:`summarise_run_by_group` into a list of rows suitable
    for a CSV — one row per (cohort, model). Cohorts are ordered healthy,
    impaired, then anything else; models are ordered by descending
    macro-F1 within each cohort.
    """
    by_group = summarise_run_by_group(
        run, groups, fallback=fallback, include_mixed=include_mixed,
    )
    folds_by_group = split_folds_by_group(
        run, groups, fallback=fallback, include_mixed=include_mixed,
    )

    rank = {GROUP_HEALTHY: 0, GROUP_IMPAIRED: 1, GROUP_UNKNOWN: 2, GROUP_MIXED: 3}
    rows: List[GroupModelRow] = []
    for code in sorted(by_group, key=lambda c: rank.get(c, 9)):
        # Count distinct held-out subjects in this cohort.
        subj: set = set()
        for f in folds_by_group.get(code, []):
            subj.update(parse_fold_subjects(f))
        summaries = by_group[code]
        for model, s in sorted(summaries.items(),
                               key=lambda kv: -kv[1].macro_f1_mean):
            rows.append(GroupModelRow(
                group=code,
                group_label=group_label(code) if code != GROUP_MIXED else "mixed",
                model_type=model,
                n_folds=s.n_folds,
                n_subjects=len(subj),
                accuracy_mean=s.accuracy_mean, accuracy_std=s.accuracy_std,
                macro_f1_mean=s.macro_f1_mean, macro_f1_std=s.macro_f1_std,
            ))
    return rows


def write_group_summary(
    run: RunStub,
    groups: ParticipantGroups,
    out_dir: Path,
    *,
    filename: str = "table_loso_by_group.csv",
    fallback: Optional[Any] = None,
    include_mixed: bool = False,
) -> Path:
    """
    Persist the cohort-split model summary as CSV (healthy vs impaired
    rows for every model in the run).
    """
    import csv
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / filename
    rows = group_model_rows(run, groups, fallback=fallback,
                            include_mixed=include_mixed)
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["group", "model", "n_subjects", "n_folds",
                    "accuracy_mean", "accuracy_std",
                    "macro_f1_mean", "macro_f1_std"])
        for r in rows:
            w.writerow([r.group_label, r.model_type, r.n_subjects, r.n_folds,
                        f"{r.accuracy_mean:.4f}", f"{r.accuracy_std:.4f}",
                        f"{r.macro_f1_mean:.4f}", f"{r.macro_f1_std:.4f}"])
    return p


def annotate_per_session_groups(
    rows: Sequence["PerSubjectFolds"],
    groups: ParticipantGroups,
    *,
    fallback: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """
    Attach a cohort code to each :class:`PerSubjectFolds` row produced
    by :func:`per_session_variability`, so the Fig 6.5 box-plot data can
    be coloured / faceted by healthy vs impaired.

    Returns plain dicts (the dataclass stays group-agnostic on purpose —
    not every caller has a registry).
    """
    out: List[Dict[str, Any]] = []
    for row in rows:
        code = groups.group_of(row.subject_id, fallback=fallback)
        out.append({
            "subject_id": row.subject_id,
            "model_type": row.model_type,
            "group":      code,
            "group_label": group_label(code),
            "fold_ids":   list(row.fold_ids),
            "f1_values":  [float(v) for v in row.f1_values],
        })
    return out


# ---------------------------------------------------------------------------
# Calibration / F1 correlation  →  §7.4
# ---------------------------------------------------------------------------

@dataclass
class CalibrationF1Joined:
    """Per-fold joined record for the §7.4 correlation."""
    subject_id:  str
    session_id:  str
    confidence:  float
    macro_f1:    float
    model_type:  str


@dataclass
class CalibrationF1Correlation:
    model_type: str
    n_pairs:    int
    pearson_r:  float
    pearson_p:  Optional[float]
    spearman_r: float
    spearman_p: Optional[float]
    joined:     List[CalibrationF1Joined] = field(default_factory=list)


def calibration_f1_correlation(
    run: RunStub,
    calibration_per_session: Dict[Tuple[str, str], float],
    *,
    model: Optional[str] = None,
) -> Dict[str, CalibrationF1Correlation]:
    """
    Per-model Pearson + Spearman correlation between a session's
    calibration confidence and the macro F1 the model achieved when
    that session was the held-out test set.

    ``calibration_per_session`` is a ``{(subject_id, session_id):
    confidence}`` map — exactly what :func:`calibration_report.calibration_stats`
    produces if you index its ``per_session`` list. Only folds
    identifiable down to a single (subject, session) pair contribute
    (i.e. LOSO-session and within-session). The runner's optional
    ``test_sessions`` field gives this directly; otherwise we parse it
    out of the fold id.
    """
    by_model_joined: Dict[str, List[CalibrationF1Joined]] = {}
    for f in run.folds:
        if model and f.model_type != model:
            continue
        sess = parse_fold_session(f)
        if sess is None:
            continue
        conf = calibration_per_session.get(sess)
        if conf is None or not math.isfinite(conf):
            continue
        by_model_joined.setdefault(f.model_type, []).append(CalibrationF1Joined(
            subject_id=sess[0], session_id=sess[1],
            confidence=float(conf),
            macro_f1=float(f.macro_f1),
            model_type=f.model_type,
        ))

    out: Dict[str, CalibrationF1Correlation] = {}
    for m, pairs in by_model_joined.items():
        if len(pairs) < 3:
            out[m] = CalibrationF1Correlation(
                model_type=m, n_pairs=len(pairs),
                pearson_r=float("nan"),  pearson_p=None,
                spearman_r=float("nan"), spearman_p=None,
                joined=pairs,
            )
            continue
        x = np.asarray([p.confidence for p in pairs], dtype=np.float64)
        y = np.asarray([p.macro_f1   for p in pairs], dtype=np.float64)
        pr, pp = _pearson(x, y)
        sr, sp = _spearman(x, y)
        out[m] = CalibrationF1Correlation(
            model_type=m, n_pairs=len(pairs),
            pearson_r=pr, pearson_p=pp,
            spearman_r=sr, spearman_p=sp,
            joined=pairs,
        )
    return out


def _pearson(x: np.ndarray, y: np.ndarray) -> Tuple[float, Optional[float]]:
    if x.size < 2 or x.std() == 0 or y.std() == 0:
        return float("nan"), None
    try:
        from scipy.stats import pearsonr
        r, p = pearsonr(x, y)
        return float(r), float(p)
    except Exception:
        # SciPy isn't required — fall back to numpy correlation only.
        r = float(np.corrcoef(x, y)[0, 1])
        return r, None


def _spearman(x: np.ndarray, y: np.ndarray) -> Tuple[float, Optional[float]]:
    if x.size < 2:
        return float("nan"), None
    try:
        from scipy.stats import spearmanr
        r, p = spearmanr(x, y)
        return float(r), float(p)
    except Exception:
        # Rank-correlate by hand if scipy is missing.
        rx = _rank(x); ry = _rank(y)
        if rx.std() == 0 or ry.std() == 0:
            return float("nan"), None
        return float(np.corrcoef(rx, ry)[0, 1]), None


def _rank(a: np.ndarray) -> np.ndarray:
    """Average-rank ties (Spearman convention)."""
    order = a.argsort()
    ranks = np.empty_like(order, dtype=np.float64)
    n = a.size
    i = 0
    while i < n:
        j = i
        while j + 1 < n and a[order[j + 1]] == a[order[i]]:
            j += 1
        avg = 0.5 * (i + j) + 1.0  # 1-based ranks
        ranks[order[i:j + 1]] = avg
        i = j + 1
    return ranks


# ---------------------------------------------------------------------------
# Bundled writer
# ---------------------------------------------------------------------------

def write_run_report(
    run: RunStub,
    out_dir: Path,
    *,
    primary_model: Optional[str] = None,
    gate_ms: float = 150.0,
) -> Dict[str, Path]:
    """
    Persist every chapter-6 deliverable derivable from one run as CSV.

    Intended for the primary LOSO-session run; for ablation / cross-
    domain bundles see :func:`write_feature_ablation` and
    :func:`write_cross_domain`.
    """
    import csv

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}

    # Table 6.3 / 6.4
    summaries = summarise_run(run)
    t_csv = out_dir / "table_model_summary.csv"
    with t_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "n_folds",
                    "accuracy_mean", "accuracy_std",
                    "macro_f1_mean", "macro_f1_std",
                    "train_seconds_mean", "inference_ms_mean"])
        for m, s in sorted(summaries.items(), key=lambda kv: -kv[1].macro_f1_mean):
            w.writerow([m, s.n_folds,
                        f"{s.accuracy_mean:.4f}", f"{s.accuracy_std:.4f}",
                        f"{s.macro_f1_mean:.4f}", f"{s.macro_f1_std:.4f}",
                        f"{s.train_seconds_mean:.2f}", f"{s.inference_ms_mean:.2f}"])
    paths["model_summary_csv"] = t_csv

    # Fig 6.3 data — per-class F1
    pc = per_class_f1_summary(run)
    pc_csv = out_dir / "fig_per_class_f1.csv"
    with pc_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "class", "mean", "std", "n_folds"])
        for p in pc:
            w.writerow([p.model_type, p.class_name,
                        f"{p.mean:.4f}", f"{p.std:.4f}", p.n_folds])
    paths["per_class_csv"] = pc_csv

    # Fig 6.3 data — per-fold per-class F1 (one row per fold × class × model)
    # Powers true boxplots in plots_thesis.plot_per_class_f1.
    pcf_csv = out_dir / "fig_per_class_f1_per_fold.csv"
    with pcf_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "class", "fold_id", "f1"])
        for fold in run.folds:
            for cls, v in (fold.per_class_f1 or {}).items():
                w.writerow([fold.model_type, cls, fold.fold_id, f"{float(v):.6f}"])
    paths["per_class_per_fold_csv"] = pcf_csv

    # Fig 6.4 data — normalised confusion matrices
    confs = aggregate_confusion(run)
    cf_json = out_dir / "fig_confusion_matrices.json"
    with cf_json.open("w", encoding="utf-8") as f:
        json.dump({m: c.to_dict() for m, c in confs.items()}, f, indent=2)
    paths["confusion_json"] = cf_json

    # Fig 6.5 data — per-session variability
    ps = per_session_variability(run, model=primary_model)
    ps_csv = out_dir / "fig_per_session_variability.csv"
    with ps_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["subject_id", "model", "fold_id", "macro_f1"])
        for row in ps:
            for fid, v in zip(row.fold_ids, row.f1_values):
                w.writerow([row.subject_id, row.model_type, fid, f"{v:.4f}"])
    paths["per_session_csv"] = ps_csv

    # Table 6.7 — latency
    lat = latency_table(run, gate_ms=gate_ms)
    lat_csv = out_dir / "table_latency.csv"
    with lat_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "train_seconds_mean", "inference_ms_mean",
                    "inference_ms_max", f"passes_gate_{int(gate_ms)}ms"])
        for r in lat:
            w.writerow([r.model_type,
                        f"{r.train_seconds_mean:.2f}",
                        f"{r.inference_ms_mean:.2f}",
                        f"{r.inference_ms_max:.2f}",
                        "Yes" if r.passes_gate else "No"])
    paths["latency_csv"] = lat_csv

    return paths


def write_feature_ablation(
    rows:    Sequence[FeatureAblationRow],
    out_dir: Path,
) -> Path:
    """Persist Table 6.5."""
    import csv
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "table_feature_ablation.csv"
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["condition", "model", "vector_length",
                    "n_folds", "macro_f1_mean", "macro_f1_std",
                    "accuracy_mean", "accuracy_std"])
        for r in rows:
            w.writerow([r.condition, r.model_type, r.vector_length, r.n_folds,
                        f"{r.macro_f1_mean:.4f}", f"{r.macro_f1_std:.4f}",
                        f"{r.accuracy_mean:.4f}", f"{r.accuracy_std:.4f}"])
    return p


def write_cross_domain(
    rows:    Sequence[CrossDomainCell],
    out_dir: Path,
) -> Path:
    """Persist Table 6.6."""
    import csv
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "table_cross_domain.csv"
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["train_domain", "test_domain", "model",
                    "n_folds", "macro_f1_mean", "macro_f1_std"])
        for c in rows:
            w.writerow([c.train_domain, c.test_domain, c.model_type, c.n_folds,
                        f"{c.macro_f1_mean:.4f}", f"{c.macro_f1_std:.4f}"])
    return p