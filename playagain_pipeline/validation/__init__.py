"""
playagain_pipeline.validation
─────────────────────────────
A reproducible, config-driven validation harness for the EMG pipeline.

Why this package exists
───────────────────────
Until now, evaluating a feature set or model architecture meant clicking
through dialogs, copy-pasting numbers, and hoping the same data was used
the next time. That is fine for exploration but not for a paper or thesis.

This package provides:

    * `SessionRecord`        — a uniform descriptor of *one* recording on
                               disk, regardless of whether it was captured
                               by the Python pipeline or by the Unity C#
                               game (DataManager.cs / DeviceManager.cs).
                               Both writers already produce the same
                               on-disk layout (data.npy + metadata.json +
                               gesture_set.json), so they can be evaluated
                               in a single corpus.

    * `SessionCorpus`        — discovers all sessions under data/sessions
                               (Python *and* unity_sessions/...) and tags
                               each with a `source_domain` ∈ {"pipeline",
                               "unity"} for cross-domain experiments.

    * `cv_strategies`        — within-session, leave-one-session-out
                               (LOSO-Session), leave-one-subject-out
                               (LOSO-Subject), and cross-domain
                               train-on-pipeline / test-on-unity (and
                               vice versa).

    * `ExperimentConfig`     — pydantic-style dataclass describing one
                               experiment: which sessions, which feature
                               set, which model, which CV strategy, which
                               random seed.

    * `ValidationRunner`     — runs an experiment, captures the
                               environment (git SHA, python version,
                               package versions), persists results +
                               config + git state side-by-side so any
                               run can be reproduced bit-for-bit later.

    * CLI                    — `python -m playagain_pipeline.validation
                               run experiments/baseline.yaml` produces a
                               timestamped directory with results.

See `README.md` in this folder for the full methodology.
"""

from .corpus import SessionRecord, SessionCorpus
from .config import ExperimentConfig, load_experiment, dump_experiment
from .runner import ValidationRunner, RunResult
from . import cv_strategies

__all__ = [
    "SessionRecord",
    "SessionCorpus",
    "ExperimentConfig",
    "load_experiment",
    "dump_experiment",
    "ValidationRunner",
    "RunResult",
    "cv_strategies",
]
