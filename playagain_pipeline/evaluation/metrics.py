"""
evaluation/metrics.py
─────────────────────
Thin metric helpers that wrap sklearn into our :class:`EvaluationResult`
objects. Centralising this avoids each evaluator hand-rolling its own
classification report.

Two entry points:

* :func:`fill_classification_metrics` — multi-class. Accepts ``y_true``,
  ``y_pred``, optional ``y_proba`` and a ``label_names`` map. Mutates
  the result in place.
* :func:`fill_binary_metrics` — binary specialisation that fills the
  ``specificity`` field on top of the multi-class fields.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .core import (
    ClassMetrics,
    ConfusionMatrix,
    EvaluationResult,
    ThresholdSweepPoint,
    safe_div,
)


# ---------------------------------------------------------------------------
# Multi-class
# ---------------------------------------------------------------------------

def fill_classification_metrics(
    result: EvaluationResult,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    label_names: Optional[Dict[int, str]] = None,
    y_proba:    Optional[np.ndarray] = None,
) -> None:
    """
    Compute aggregate + per-class metrics and write them into ``result``.

    ``label_names`` maps numeric label ids to display names. Labels that
    appear in either y_true or y_pred but not in the dict get a
    fallback name like ``"class_3"``.
    """
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score, recall_score,
        confusion_matrix, classification_report,
        cohen_kappa_score, matthews_corrcoef, balanced_accuracy_score,
        top_k_accuracy_score,
    )

    y_true = np.asarray(y_true).astype(np.int64).ravel()
    y_pred = np.asarray(y_pred).astype(np.int64).ravel()

    if y_true.size == 0:
        result.add_note("No samples — metrics left as NaN.")
        return

    # Stable label order: sorted union of (true ∪ pred), so an unseen
    # class still shows up in the confusion matrix.
    label_set: List[int] = sorted(np.unique(np.concatenate([y_true, y_pred])).tolist())
    name_lookup: Dict[int, str] = {}
    for k, v in (label_names or {}).items():
        try:
            name_lookup[int(k)] = str(v)
        except (TypeError, ValueError):
            pass
    target_names = [name_lookup.get(int(l), f"class_{int(l)}") for l in label_set]
    # If a gesture_set or model metadata accidentally maps two label_ids
    # to the same display name (yes, this happens — e.g. "Tripod" twice),
    # sklearn's classification_report errors out on duplicate target_names.
    # Suffix collisions with " (id N)" so each label keeps a unique header.
    seen_names: Dict[str, int] = {}
    deduped: List[str] = []
    for lbl, name in zip(label_set, target_names):
        if name in seen_names:
            seen_names[name] += 1
            deduped.append(f"{name}  (id {int(lbl)})")
        else:
            seen_names[name] = 1
            deduped.append(name)
    target_names = deduped

    # Aggregate
    result.n_samples       = int(y_true.size)
    result.accuracy        = float(accuracy_score(y_true, y_pred))
    result.f1_macro        = float(f1_score(y_true, y_pred, average="macro",    labels=label_set, zero_division=0))
    result.f1_weighted     = float(f1_score(y_true, y_pred, average="weighted", labels=label_set, zero_division=0))
    result.precision_macro = float(precision_score(y_true, y_pred, average="macro", labels=label_set, zero_division=0))
    result.recall_macro    = float(recall_score(y_true, y_pred, average="macro", labels=label_set, zero_division=0))

    # Robustness/agreement metrics. Cohen's kappa and MCC both reduce
    # to a meaningful single number even on heavy class imbalance, and
    # balanced accuracy is just the macro-recall under another name —
    # but it's so commonly cited it deserves its own field rather than
    # making readers do the translation.
    try:
        result.cohen_kappa = float(cohen_kappa_score(y_true, y_pred))
    except Exception:
        # cohen_kappa_score blows up when there's only one class in
        # both arrays. Leave NaN and add a note rather than crashing.
        result.add_note("Cohen's κ undefined (single-class predictions).")
    try:
        result.mcc = float(matthews_corrcoef(y_true, y_pred))
    except Exception:
        result.add_note("MCC undefined (single-class predictions).")
    try:
        result.balanced_accuracy = float(balanced_accuracy_score(y_true, y_pred))
    except Exception:
        result.add_note("Balanced accuracy undefined.")

    # Per-class
    rep = classification_report(
        y_true, y_pred, labels=label_set, target_names=target_names,
        output_dict=True, zero_division=0,
    )
    result.per_class = []
    for lbl, name in zip(label_set, target_names):
        row = rep.get(name) or {}
        result.per_class.append(ClassMetrics(
            label=int(lbl),
            name=str(name),
            precision=float(row.get("precision", 0.0)),
            recall   =float(row.get("recall",    0.0)),
            f1       =float(row.get("f1-score",  0.0)),
            support  =int  (row.get("support",   0)),
        ))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=label_set)
    result.confusion = ConfusionMatrix(
        labels=[int(l) for l in label_set],
        label_names=list(target_names),
        matrix=cm.tolist(),
    )

    # Confidence stats (only if probabilities supplied)
    if y_proba is not None and y_proba.ndim == 2 and y_proba.shape[0] == y_true.size:
        top_conf = np.max(y_proba, axis=1)
        correct  = (y_pred == y_true)
        if np.any(correct):
            result.mean_confidence_correct = float(np.mean(top_conf[correct]))
        if np.any(~correct):
            result.mean_confidence_incorrect = float(np.mean(top_conf[~correct]))
        # Expected Calibration Error (15-bin, classic Guo et al. 2017).
        result.expected_calibration_error = float(_ece(top_conf, correct, n_bins=15))

        # Top-k accuracy. Only meaningful when n_classes > k — top-2 on
        # a binary task is always 100% and top-3 on a 2-class task is
        # impossible. We also guard against y_proba columns not lining
        # up with the label_set, which happens when a model dropped a
        # class at training time (rare but the runner has produced it).
        n_classes = int(y_proba.shape[1])
        if n_classes >= 3:
            try:
                # We need the proba columns to correspond to label IDs in
                # ``label_set`` order. sklearn's top_k_accuracy_score lets
                # you pass `labels=...` to disambiguate; do that to stay
                # robust to dropped-class models.
                proba_labels = list(range(n_classes))
                if n_classes >= 3:
                    result.top_2_accuracy = float(top_k_accuracy_score(
                        y_true, y_proba, k=2, labels=proba_labels,
                    ))
                if n_classes >= 4:
                    result.top_3_accuracy = float(top_k_accuracy_score(
                        y_true, y_proba, k=3, labels=proba_labels,
                    ))
            except Exception as exc:
                # Don't fail the whole eval over a top-k edge case
                result.add_note(f"Top-k accuracy skipped: {exc}")

        # Confidence histogram for the correct/incorrect overlay plot.
        # 20 equal-width bins between 0 and 1 — fine enough to expose
        # over- and under-confidence, coarse enough to plot at any size.
        edges = np.linspace(0.0, 1.0, 21)
        c_counts, _ = np.histogram(top_conf[correct],  bins=edges)
        i_counts, _ = np.histogram(top_conf[~correct], bins=edges)
        result.confidence_histogram = {
            "bin_edges":        [float(x) for x in edges],
            "correct_counts":   [int(x)   for x in c_counts],
            "incorrect_counts": [int(x)   for x in i_counts],
        }


def _ece(confidences: np.ndarray, correct: np.ndarray, *, n_bins: int = 15) -> float:
    """Expected Calibration Error with equal-width confidence bins."""
    if confidences.size == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi == 1.0:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences <  hi)
        if not np.any(mask):
            continue
        bin_acc  = float(np.mean(correct[mask]))
        bin_conf = float(np.mean(confidences[mask]))
        weight   = float(np.sum(mask)) / confidences.size
        ece += weight * abs(bin_acc - bin_conf)
    return ece


# ---------------------------------------------------------------------------
# Binary  (Unity threshold)
# ---------------------------------------------------------------------------

def fill_binary_metrics(
    result: EvaluationResult,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    label_names: Tuple[str, str] = ("inactive", "active"),
) -> None:
    """
    Fill binary-only fields (``specificity``) **plus** the multi-class
    fields, treating ``y_true`` / ``y_pred`` as 0/1 labels.
    """
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score, recall_score,
        confusion_matrix,
        cohen_kappa_score, matthews_corrcoef, balanced_accuracy_score,
    )

    y_true = (np.asarray(y_true) > 0).astype(np.int64).ravel()
    y_pred = (np.asarray(y_pred) > 0).astype(np.int64).ravel()

    if y_true.size == 0:
        result.add_note("No samples — metrics left as NaN.")
        return

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(cm[0, 0]), int(cm[0, 1]),
                      int(cm[1, 0]), int(cm[1, 1]))

    result.n_samples       = int(y_true.size)
    result.accuracy        = float(accuracy_score(y_true, y_pred))
    result.f1_macro        = float(f1_score(y_true, y_pred, average="macro",    labels=[0, 1], zero_division=0))
    result.f1_weighted     = float(f1_score(y_true, y_pred, average="weighted", labels=[0, 1], zero_division=0))
    result.precision_macro = float(precision_score(y_true, y_pred, average="macro", labels=[0, 1], zero_division=0))
    result.recall_macro    = float(recall_score(y_true, y_pred, average="macro", labels=[0, 1], zero_division=0))
    result.specificity     = safe_div(tn, tn + fp)

    # Same agreement/robustness metrics as in the multi-class path.
    # Cohen's κ on a binary task is sometimes called "Heidke skill score";
    # MCC on a binary task is also known as the phi coefficient.
    try:
        result.cohen_kappa = float(cohen_kappa_score(y_true, y_pred))
    except Exception:
        pass
    try:
        result.mcc = float(matthews_corrcoef(y_true, y_pred))
    except Exception:
        pass
    try:
        result.balanced_accuracy = float(balanced_accuracy_score(y_true, y_pred))
    except Exception:
        pass

    p_pos = safe_div(tp, tp + fp); r_pos = safe_div(tp, tp + fn)
    p_neg = safe_div(tn, tn + fn); r_neg = safe_div(tn, tn + fp)
    result.per_class = [
        ClassMetrics(label=0, name=label_names[0],
                     precision=p_neg, recall=r_neg,
                     f1=safe_div(2 * p_neg * r_neg, p_neg + r_neg),
                     support=int(tn + fp)),
        ClassMetrics(label=1, name=label_names[1],
                     precision=p_pos, recall=r_pos,
                     f1=safe_div(2 * p_pos * r_pos, p_pos + r_pos),
                     support=int(fn + tp)),
    ]

    result.confusion = ConfusionMatrix(
        labels=[0, 1],
        label_names=list(label_names),
        matrix=cm.tolist(),
    )


def threshold_sweep(
    rms: np.ndarray,
    y_true: np.ndarray,
    *,
    n_thresholds: int = 200,
) -> List[ThresholdSweepPoint]:
    """
    Sweep RMS thresholds across the observed range and return one
    :class:`ThresholdSweepPoint` per threshold, sorted ascending.
    """
    rms    = np.asarray(rms,    dtype=np.float64).ravel()
    y_true = (np.asarray(y_true) > 0).astype(np.int64).ravel()

    if rms.size == 0 or rms.size != y_true.size:
        return []

    lo, hi = float(np.min(rms)), float(np.max(rms))
    if lo >= hi:
        thresholds = np.array([lo])
    else:
        thresholds = np.linspace(lo, hi, n_thresholds)

    out: List[ThresholdSweepPoint] = []
    for t in thresholds:
        y_pred = (rms > t).astype(np.int64)
        tp = int(np.sum((y_pred == 1) & (y_true == 1)))
        fp = int(np.sum((y_pred == 1) & (y_true == 0)))
        tn = int(np.sum((y_pred == 0) & (y_true == 0)))
        fn = int(np.sum((y_pred == 0) & (y_true == 1)))
        prec = safe_div(tp, tp + fp)
        rec  = safe_div(tp, tp + fn)
        spec = safe_div(tn, tn + fp)
        f1   = safe_div(2 * prec * rec, prec + rec)
        acc  = safe_div(tp + tn, tp + fp + tn + fn)
        out.append(ThresholdSweepPoint(
            threshold=float(t), accuracy=acc, precision=prec, recall=rec,
            specificity=spec, f1=f1, tp=tp, fp=fp, tn=tn, fn=fn,
        ))
    return out


def pick_optimal_threshold(
    sweep: Sequence[ThresholdSweepPoint],
    *,
    objective: str = "f1",
) -> Optional[ThresholdSweepPoint]:
    """
    Pick the sweep row that maximises an objective.

    Supported: ``"f1"`` (default), ``"youden"`` (sensitivity+specificity-1),
    ``"accuracy"``.
    """
    if not sweep:
        return None
    if objective == "youden":
        key = lambda p: p.recall + p.specificity - 1.0
    elif objective == "accuracy":
        key = lambda p: p.accuracy
    else:
        key = lambda p: p.f1
    return max(sweep, key=key)


def auroc_binary(scores: np.ndarray, y_true: np.ndarray) -> Optional[float]:
    """
    AUROC for a binary task. Returns ``None`` (not NaN) when only one
    class is present so callers can blank the field rather than show
    a misleading ``nan``.
    """
    try:
        from sklearn.metrics import roc_auc_score
    except Exception:
        return None
    y_true = (np.asarray(y_true) > 0).astype(np.int64).ravel()
    scores = np.asarray(scores, dtype=np.float64).ravel()
    if y_true.size == 0 or len(np.unique(y_true)) < 2:
        return None
    try:
        return float(roc_auc_score(y_true, scores))
    except Exception:
        return None