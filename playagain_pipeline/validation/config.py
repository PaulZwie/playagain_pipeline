"""
config.py
─────────
ExperimentConfig — the single, declarative description of an experiment.

A validation experiment is fully captured by:

    * which sessions (subjects, domains, optional explicit allow-list)
    * which feature pipeline (named features + their parameters)
    * which model architecture + hyperparameters
    * which CV strategy + its arguments
    * which preprocessing (window size, stride, channel selection)
    * a single random seed

Anything not in the config must NOT influence the result. That is the
whole point of having a config: paste it into a paper, hand it to a
colleague, and they get the same numbers.

Format
──────
YAML on disk for human readability. Round-trips losslessly to JSON for
storage alongside results, so a result directory is fully
self-describing without needing the original YAML file.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Sub-sections
# ---------------------------------------------------------------------------

@dataclass
class DataSelection:
    """Which sessions to feed into the experiment."""
    subjects:       Optional[List[str]] = None     # None = all
    domains:        Optional[List[str]] = None     # ["pipeline"], ["unity"], or both
    min_channels:   Optional[int] = None
    sampling_rate:  Optional[float] = None
    # Explicit allow-list of "subject/session_id" strings; takes precedence
    # over the predicate filters above when non-empty.
    explicit:       Optional[List[str]] = None


@dataclass
class WindowingConfig:
    """How the continuous EMG is sliced into windows."""
    window_ms:    int = 200
    stride_ms:    int = 50
    drop_rest:    bool = False


@dataclass
class FeatureConfig:
    """A single feature with its parameters."""
    name:    str
    params:  Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelConfig:
    """A single model with its hyperparameters."""
    type:    str                                # e.g. "RandomForest", "CatBoost", "AttentionNet"
    params:  Dict[str, Any] = field(default_factory=dict)


@dataclass
class CVConfig:
    """Cross-validation strategy + its kwargs."""
    strategy: str                               # one of cv_strategies.STRATEGIES
    kwargs:   Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Top-level experiment
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    """
    Declarative description of one validation experiment.

    Two experiments with byte-identical configs (and identical input
    data) MUST produce byte-identical results. The runner enforces this
    by also serialising git SHA, package versions, and the seed.
    """
    name:        str
    description: str = ""
    seed:        int = 42

    data:        DataSelection = field(default_factory=DataSelection)
    windowing:   WindowingConfig = field(default_factory=WindowingConfig)
    features:    List[FeatureConfig] = field(default_factory=list)
    models:      List[ModelConfig] = field(default_factory=list)
    cv:          CVConfig = field(default_factory=lambda: CVConfig(strategy="loso_subject"))

    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Loader / dumper
# ---------------------------------------------------------------------------

def load_experiment(path: Path) -> ExperimentConfig:
    """Load an experiment from a YAML or JSON file."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")

    if path.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml  # PyYAML; declared as a soft dependency
        except ImportError as e:  # noqa: BLE001
            raise ImportError(
                "PyYAML is required to load YAML experiments. "
                "Install with: pip install pyyaml"
            ) from e
        raw = yaml.safe_load(text) or {}
    else:
        raw = json.loads(text)

    return _from_dict(raw)


def dump_experiment(cfg: ExperimentConfig, path: Path) -> None:
    """Serialise an experiment back to disk (JSON for unambiguity)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg.to_dict(), indent=2), encoding="utf-8")


def _from_dict(raw: Dict[str, Any]) -> ExperimentConfig:
    """Tolerant dict → dataclass conversion (no external pydantic dep)."""
    data = DataSelection(**(raw.get("data") or {}))
    win  = WindowingConfig(**(raw.get("windowing") or {}))
    feats = [FeatureConfig(**f) for f in (raw.get("features") or [])]
    models = [ModelConfig(**m) for m in (raw.get("models") or [])]

    cv_raw = raw.get("cv") or {"strategy": "loso_subject"}
    cv = CVConfig(strategy=cv_raw["strategy"], kwargs=cv_raw.get("kwargs", {}))

    return ExperimentConfig(
        name=raw.get("name", "unnamed"),
        description=raw.get("description", ""),
        seed=int(raw.get("seed", 42)),
        data=data,
        windowing=win,
        features=feats,
        models=models,
        cv=cv,
    )
