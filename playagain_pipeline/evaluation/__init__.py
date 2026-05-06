"""
playagain_pipeline.evaluation
─────────────────────────────
Unified evaluation pipeline for three recording sources:

* **Sessions**         — labelled training recordings; evaluated by running
                          a saved model over windows.
* **Game recordings**  — gameplay logs that already contain predictions and
                          probabilities; can either re-use those directly or
                          replay any saved model on the EMG channels.
* **Unity recordings** — RMS-thresholded recordings with a binary
                          active/rest ground truth; evaluated by sweeping
                          thresholds and picking the best one.

Public API
──────────
    from playagain_pipeline.evaluation import (
        EvaluationResult, RecordingDescriptor, RecordingKind, EvaluationMode,
        evaluate_sessions,  SessionEvalSettings,
        evaluate_games,     GameEvalSettings,
        evaluate_unity,     UnityEvalSettings,
        evaluate_features_lda,
        discover_sessions, discover_game_recordings, discover_unity_recordings,
    )

The :class:`EvaluationResult` shape is identical across all three so the
GUI can render any result with the same widgets.
"""

from .core import (
    EvaluationMode,
    EvaluationResult,
    ClassMetrics,
    ConfusionMatrix,
    RecordingDescriptor,
    RecordingKind,
    ThresholdSweepPoint,
)

from .loaders import (
    discover_sessions,
    discover_game_recordings,
    discover_unity_recordings,
    load_session_data,
    load_session_gesture_set,
    load_game_csv,
    GameRecording,
)

from .session_eval import (
    SessionEvalSettings,
    evaluate_sessions,
    evaluate_features_lda,
)

from .game_eval import (
    GameEvalSettings,
    evaluate_games,
    TRUTH_RAW, TRUTH_REQUESTED, TRUTH_ACTIVE,
)

from .unity_eval import (
    UnityEvalSettings,
    evaluate_unity,
)

__all__ = [
    # core
    "EvaluationMode", "EvaluationResult", "ClassMetrics", "ConfusionMatrix",
    "RecordingDescriptor", "RecordingKind", "ThresholdSweepPoint",
    # loaders
    "discover_sessions", "discover_game_recordings", "discover_unity_recordings",
    "load_session_data", "load_session_gesture_set", "load_game_csv",
    "GameRecording",
    # evaluators
    "SessionEvalSettings", "evaluate_sessions", "evaluate_features_lda",
    "GameEvalSettings", "evaluate_games",
    "UnityEvalSettings", "evaluate_unity",
    "TRUTH_RAW", "TRUTH_REQUESTED", "TRUTH_ACTIVE",
]