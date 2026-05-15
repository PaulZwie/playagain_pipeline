"""
evaluation/core.py
──────────────────
Shared types for the evaluation pipeline.

The three recording sources (training **sessions**, **game recordings**,
**unity recordings**) each have their own evaluator module, but they
all return the same :class:`EvaluationResult` shape so the UI can
render them uniformly.

This module deliberately has no Qt or sklearn imports so it stays
cheap to load from background workers.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class RecordingKind(str, Enum):
    """Three top-level recording sources understood by the pipeline."""
    SESSION = "session"          # data/sessions/<subject>/<session>
    GAME    = "game_recording"   # data/game_recordings/<subject>/*.csv
    UNITY   = "unity_recording"  # data/sessions/unity_sessions/... or raw Unity CSVs


class EvaluationMode(str, Enum):
    """How predictions are produced for a given evaluation."""
    MODEL_INFERENCE  = "model"     # run a saved model over windows
    LOGGED           = "logged"    # use predictions stored in the recording
    RMS_THRESHOLD    = "threshold" # binary RMS > threshold (Unity)


# ---------------------------------------------------------------------------
# Descriptors — lightweight handles the UI passes around
# ---------------------------------------------------------------------------

@dataclass
class RecordingDescriptor:
    """
    A pointer to one recording on disk.

    The descriptor is intentionally cheap — we don't load the EMG matrix
    until the evaluator actually needs it. The UI uses these to populate
    pickers; the evaluator uses ``path`` to resolve the data.
    """
    kind:       RecordingKind
    subject_id: str
    session_id: str           # filename stem for game/unity, session dir for sessions
    path:       Path          # the directory or .csv file
    label:      str           # short human label for UI
    meta:       Dict[str, Any] = field(default_factory=dict)

    @property
    def is_unity_session(self) -> bool:
        """True if this is a SESSION descriptor that originated from Unity."""
        if self.kind != RecordingKind.SESSION:
            return False
        return bool(self.meta.get("is_unity_recording")) or \
               "unity" in str(self.path).lower()


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class ConfusionMatrix:
    """A simple labelled confusion matrix."""
    labels:       List[int]               # numeric class IDs in matrix order
    label_names:  List[str]               # display names matching ``labels``
    matrix:       List[List[int]]         # [true][pred]

    def to_array(self) -> np.ndarray:
        return np.asarray(self.matrix, dtype=np.int64)


@dataclass
class ClassMetrics:
    """Per-class precision/recall/F1/support."""
    label:     int
    name:      str
    precision: float
    recall:    float
    f1:        float
    support:   int


@dataclass
class ThresholdSweepPoint:
    """One row of the RMS-threshold sweep used by the Unity evaluator."""
    threshold:   float
    accuracy:    float
    precision:   float
    recall:      float       # == sensitivity
    specificity: float
    f1:          float
    tp: int; fp: int; tn: int; fn: int


@dataclass
class EvaluationResult:
    """
    What every evaluator returns.

    ``per_class`` is empty for binary RMS-threshold runs (we use the
    binary fields below instead). ``threshold_sweep`` is populated only
    for the Unity evaluator. ``per_feature`` is populated only when the
    feature-importance evaluator runs.
    """
    # Identity
    title:           str
    mode:            EvaluationMode
    kind:            RecordingKind
    created_at:      datetime              = field(default_factory=datetime.now)

    # Inputs (echoed for provenance)
    recordings:      List[RecordingDescriptor] = field(default_factory=list)
    model_name:      Optional[str]         = None
    settings:        Dict[str, Any]        = field(default_factory=dict)

    # Aggregate metrics  (NaN where not applicable, never None)
    n_samples:       int                   = 0
    accuracy:        float                 = float("nan")
    f1_macro:        float                 = float("nan")
    f1_weighted:     float                 = float("nan")
    precision_macro: float                 = float("nan")
    recall_macro:    float                 = float("nan")

    # Robustness / agreement metrics  (added in v3)
    # All three of these are scalar summaries that scale better than raw
    # accuracy when classes are imbalanced — Cohen's kappa subtracts the
    # agreement you'd expect by chance, MCC is robust under skewed
    # support, and balanced accuracy averages per-class recall so a
    # 95% rest / 5% fist split can't hide a model that ignores fist.
    cohen_kappa:        float = float("nan")
    mcc:                float = float("nan")
    balanced_accuracy:  float = float("nan")

    # Top-k accuracy  (multi-class with predicted probabilities only).
    # ``None`` when the predictor doesn't expose probas, when the
    # number of classes is too small for k to be informative (top-2 on
    # a binary task is always 100%), or when the classifier is binary.
    top_2_accuracy: Optional[float] = None
    top_3_accuracy: Optional[float] = None

    # Inference latency hint, in milliseconds per window. Populated by
    # evaluators that time the prediction step; left as ``None`` when
    # the evaluator didn't measure it. Useful for "is this model fast
    # enough for real-time use" — EMG decisions land at ~50 ms cadence.
    inference_ms_per_window: Optional[float] = None

    # Binary fields (Unity / one-vs-rest views)
    specificity:     float                 = float("nan")
    auroc:           Optional[float]       = None
    chosen_threshold: Optional[float]      = None

    # Confidence (multi-class only — needs predicted probabilities)
    mean_confidence_correct:    Optional[float] = None
    mean_confidence_incorrect:  Optional[float] = None
    expected_calibration_error: Optional[float] = None

    # Confidence histogram for the correct/incorrect overlay plot.
    # When the predictor exposes probabilities, ``fill_classification_metrics``
    # bins the top-1 confidences into 20 equal-width bins and records
    # how many landed in each bin among correct / incorrect predictions.
    # The dict is JSON-safe and small; absent when no probas were given.
    #   {"bin_edges": [...], "correct_counts": [...], "incorrect_counts": [...]}
    confidence_histogram: Optional[Dict[str, List[float]]] = None

    # Detail
    per_class:       List[ClassMetrics]    = field(default_factory=list)
    confusion:       Optional[ConfusionMatrix] = None
    threshold_sweep: List[ThresholdSweepPoint] = field(default_factory=list)
    per_feature:     Dict[str, float]      = field(default_factory=dict)
    notes:           List[str]             = field(default_factory=list)

    # Optional time series for the UI to plot
    timeline:        Optional[Dict[str, np.ndarray]] = None  # {"t": ..., "y_true": ..., "y_pred": ...}

    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable view (without numpy arrays)."""
        d = asdict(self)
        d["created_at"] = self.created_at.isoformat()
        d["mode"] = self.mode.value
        d["kind"] = self.kind.value
        d["recordings"] = [
            {**asdict(r), "path": str(r.path), "kind": r.kind.value}
            for r in self.recordings
        ]
        # Drop numpy timeline for json compactness
        d.pop("timeline", None)
        return d

    def add_note(self, text: str) -> None:
        self.notes.append(text)


# ---------------------------------------------------------------------------
# Convenience helpers used by all evaluators
# ---------------------------------------------------------------------------

def safe_div(num: float, den: float) -> float:
    """Float division that returns 0.0 instead of raising on zero den."""
    return float(num) / float(den) if den else 0.0