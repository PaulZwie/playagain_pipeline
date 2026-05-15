"""
validation/game_corpus.py
─────────────────────────
Discovery of **game recordings** as a first-class corpus, on the same
footing as :class:`corpus.SessionCorpus`.

Why game recordings deserve their own corpus
────────────────────────────────────────────
``SessionCorpus`` walks ``data/sessions/`` and is consumed by the
training :mod:`runner` — every record it yields must be loadable by
``DataManager.load_session`` and must carry trial boundaries. Game
recordings are *not* that shape: a game recording is a single CSV
written by :class:`game_recorder.GameRecorder` while the participant
played, and it already contains, frame by frame:

    * the raw EMG matrix (``EMG_Ch0..ChN``),
    * the gesture the live model predicted (``PredictedGestureId``)
      together with its per-class probabilities (``Prob_<Class>``),
    * the gesture the game asked for (``RequestedGesture``) and the
      authoritative active-gesture label (``RawGroundTruth`` /
      ``GroundTruthActive``).

In other words a game recording is the only source that captures what
the deployed system *actually did in front of a user*. The Unity
recordings, by contrast, only ever stored an RMS value and a binary
threshold crossing — so a "gesture" there is just "muscle active vs
not", which collapses fist / pinch / tripod into one class. For
genuine multi-class performance reporting the game recordings are the
better evidence, and the thesis pipeline should not ignore them.

So we keep the two discoveries separate:

    SessionCorpus  →  data/sessions/...        (feeds the runner)
    GameCorpus     →  data/game_recordings/... (feeds game_report.py)

and let the reporting layer combine them where it makes sense (corpus
overview, per-group breakdowns).

Relationship to ``evaluation/``
───────────────────────────────
The actual filesystem walk and CSV parsing already live in
``evaluation/loaders.py`` (``discover_game_recordings``, ``GameRecording``)
and the logged-prediction extraction in ``evaluation/game_eval.py``.
This module is a thin, corpus-flavoured wrapper around those so the
validation package gets a uniform ``.all() / .filter() / .subjects() /
.summary()`` surface without duplicating the parser.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional

log = logging.getLogger(__name__)

# These imports are from the sibling ``evaluation`` package. They are
# light (no torch / sklearn at import time).
from playagain_pipeline.evaluation.core import RecordingDescriptor, RecordingKind
from playagain_pipeline.evaluation.loaders import discover_game_recordings


# Marker so callers that mix session and game records can branch on a
# single attribute, mirroring ``SessionRecord.source_domain``.
GAME_SOURCE_DOMAIN = "game"


class GameCorpus:
    """
    Discovers every game recording under ``<data_dir>/game_recordings``
    and exposes them as a flat, filterable list of
    :class:`evaluation.core.RecordingDescriptor` (kind ``GAME``).

    The descriptors are exactly what
    :func:`evaluation.game_eval.evaluate_games` expects, so the
    reporting layer can hand them straight to the existing evaluator
    without any glue.

    Parameters
    ----------
    data_dir : Path
        Pipeline data root (the directory that contains
        ``game_recordings/``).
    """

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self._records: List[RecordingDescriptor] = []
        self._discovered = False

    # ------------------------------------------------------------------

    def discover(self, verbose: bool = False) -> List[RecordingDescriptor]:
        """Walk the game-recordings tree once and cache the result."""
        if self._discovered:
            return self._records

        root = self.data_dir / "game_recordings"
        if not root.exists():
            log.warning("Game-recordings directory does not exist: %s", root)
            self._discovered = True
            return self._records

        try:
            recs = discover_game_recordings(self.data_dir)
        except Exception as exc:  # noqa: BLE001
            log.warning("Game-recording discovery failed under %s: %s", root, exc)
            recs = []

        # Tag every descriptor with a source domain so downstream code
        # can treat session and game records polymorphically. The
        # RecordingDescriptor.meta dict is the natural place for it.
        for r in recs:
            r.meta.setdefault("source_domain", GAME_SOURCE_DOMAIN)
            if verbose:
                log.info("Discovered game recording %s/%s  (%s)",
                         r.subject_id, r.session_id,
                         r.meta.get("model_name") or "no-model")

        # Stable, reproducible ordering — keeps any derived CSV diff-able.
        recs.sort(key=lambda r: (str(r.subject_id), str(r.session_id)))
        self._records = recs
        self._discovered = True
        return self._records

    # ------------------------------------------------------------------
    # Filtering / querying helpers — deliberately mirror SessionCorpus.
    # ------------------------------------------------------------------

    def all(self) -> List[RecordingDescriptor]:
        return list(self.discover())

    def filter(
        self,
        subjects: Optional[Iterable[str]] = None,
        with_logged_predictions: Optional[bool] = None,
        models: Optional[Iterable[str]] = None,
    ) -> List[RecordingDescriptor]:
        """
        Return the subset of game recordings matching every supplied
        predicate. ``None`` arguments are not filtered on.

        Parameters
        ----------
        subjects :
            Keep only these subject ids.
        with_logged_predictions :
            When True, keep only recordings whose config advertises a
            model (i.e. the CSV should carry ``PredictedGestureId`` and
            ``Prob_*`` columns). When False, keep only those without.
        models :
            Keep only recordings produced by one of these model names.
        """
        subj_set = set(subjects) if subjects else None
        model_set = set(models) if models else None
        out: List[RecordingDescriptor] = []
        for rec in self.discover():
            if subj_set is not None and rec.subject_id not in subj_set:
                continue
            model_name = rec.meta.get("model_name")
            if with_logged_predictions is True and not model_name:
                continue
            if with_logged_predictions is False and model_name:
                continue
            if model_set is not None and model_name not in model_set:
                continue
            out.append(rec)
        return out

    def subjects(self) -> List[str]:
        return sorted({r.subject_id for r in self.discover()})

    def by_subject(self) -> Dict[str, List[RecordingDescriptor]]:
        """Group the discovered recordings into ``{subject_id: [rec, ...]}``."""
        out: Dict[str, List[RecordingDescriptor]] = {}
        for r in self.discover():
            out.setdefault(r.subject_id, []).append(r)
        return out

    def summary(self) -> str:
        recs = self.discover()
        n_with_model = sum(1 for r in recs if r.meta.get("model_name"))
        models = sorted({r.meta.get("model_name") for r in recs
                         if r.meta.get("model_name")})
        lines = [f"GameCorpus @ {self.data_dir}"]
        lines.append(f"  total recordings : {len(recs)}")
        lines.append(f"  with a model     : {n_with_model}")
        lines.append(f"  subjects         : {len(self.subjects())}")
        lines.append(f"  distinct models  : {len(models)}")
        return "\n".join(lines)