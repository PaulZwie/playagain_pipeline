"""
rest_gap_filler.py
──────────────────
Quick-fix utility: insert synthetic ``rest`` trials into the unlabelled gaps
between consecutive gesture trials of a ``RecordingSession``.

Why this exists
───────────────
The protocol-driven recorder emits explicit ``rest`` trials between each
gesture (e.g. a ``quick`` protocol session has alternating gesture/rest
trials).

The training-game recorder, by contrast, only marks the gesture intervals;
the time between gestures (animal walking off, next animal walking in) is
left unlabelled. Datasets built from such sessions therefore never see
rest examples, and downstream models cannot learn a rest class.

This module patches the trial list **after** recording stops but **before**
saving — the EMG samples themselves are never touched. The fix is:

  * Idempotent       — running it twice on a filled session is a no-op.
  * Conservative     — only fills gaps strictly between two
                       ``trial_type == "gesture"`` trials. Calibration-
                       sync neighbours are left alone.
  * Reversible       — every inserted trial carries the recognisable
                       ``notes`` value below, so they can be filtered
                       out or removed cleanly later.
  * Failure-tolerant — if the gesture set has no ``rest`` entry, or the
                       sampling rate is bad, the function logs and
                       returns 0 instead of raising.

Suggested install path
──────────────────────
    playagain_pipeline/utils/rest_gap_filler.py

It is intentionally a free function on a session, not a method on
``RecordingSession``, to keep the diff to ``core/session.py`` at zero.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:  # avoid circular import at runtime
    from playagain_pipeline.core.session import RecordingSession

log = logging.getLogger(__name__)

# Every trial we insert carries this exact ``notes`` value. Searching for
# it lets you find or remove auto-inserted trials later without ambiguity.
INSERTED_NOTE = "auto-inserted rest (fills inter-trial gap)"


def fill_rest_gaps(
    session: "RecordingSession",
    rest_gesture_name: str = "rest",
    min_gap_samples: int = 1,
) -> int:
    """
    Insert ``rest`` trials into time gaps between consecutive gesture trials.

    Parameters
    ----------
    session
        A live or freshly-loaded ``RecordingSession``. Its ``trials``
        list is replaced in-place; nothing else on the session is
        touched.
    rest_gesture_name
        Which gesture from the session's gesture set to use for the
        inserted trials. Defaults to ``"rest"``, which exists in the
        default gesture set with ``label_id=0``.
    min_gap_samples
        Minimum gap length, in samples, that qualifies for filling. Gaps
        smaller than this are ignored — useful for tolerating a sample
        or two of slop between trials produced by different threads.
        The default of ``1`` fills every non-zero gap.

    Returns
    -------
    int
        Number of rest trials inserted. ``0`` means the session was
        already fully labelled or could not be patched safely.

    Behaviour
    ─────────
    * Trials are sorted by ``start_sample`` before processing. The
      original ``RecordingTrial`` objects are kept; only their order in
      ``session.trials`` and their ``trial_id`` values change.
    * For each consecutive pair (A, B) where both have
      ``trial_type == "gesture"`` and ``B.start_sample - A.end_sample
      >= min_gap_samples``, a new ``RecordingTrial`` is inserted
      covering ``[A.end_sample, B.start_sample)``.
    * Calibration-sync trials are never wrapped in rest, and gaps
      adjacent to them are left alone.
    * After insertion, all trials are renumbered with contiguous
      ``trial_id`` values starting from 0.
    """
    # Local import keeps this module importable from places that haven't
    # already loaded ``core.session`` (e.g. small CLI tools).
    from playagain_pipeline.core.session import RecordingTrial

    if not session.trials:
        return 0

    rest_gesture = session.gesture_set.get_gesture(rest_gesture_name)
    if rest_gesture is None or rest_gesture.label_id is None:
        log.warning(
            "fill_rest_gaps: gesture '%s' missing from set '%s' — "
            "no rest trials will be inserted.",
            rest_gesture_name,
            session.gesture_set.name,
        )
        return 0

    sr = session.metadata.sampling_rate
    if sr <= 0:
        log.warning(
            "fill_rest_gaps: sampling_rate=%s is non-positive — refusing "
            "to compute rest trial timestamps.",
            sr,
        )
        return 0

    # Sort by start_sample so consecutive pairs are also temporally
    # consecutive. Original objects are preserved (no copies).
    ordered = sorted(session.trials, key=lambda t: t.start_sample)

    new_trials: List = []
    inserted = 0

    for i, trial in enumerate(ordered):
        new_trials.append(trial)
        if i + 1 >= len(ordered):
            continue
        nxt = ordered[i + 1]

        # Conservative: only fill gaps between two normal gesture trials.
        # Skip anything adjacent to a calibration_sync trial.
        if trial.trial_type != "gesture" or nxt.trial_type != "gesture":
            continue

        gap = nxt.start_sample - trial.end_sample
        if gap < min_gap_samples:
            # Zero-length gap, accidental overlap, or sub-threshold —
            # leave the trial list as-is.
            continue

        rest_trial = RecordingTrial(
            trial_id=-1,  # renumbered below
            gesture_name=rest_gesture.name,
            gesture_label=rest_gesture.label_id,
            start_sample=trial.end_sample,
            end_sample=nxt.start_sample,
            start_time=trial.end_sample / sr,
            end_time=nxt.start_sample / sr,
            is_valid=True,
            notes=INSERTED_NOTE,
            trial_type="gesture",
        )
        new_trials.append(rest_trial)
        inserted += 1

    if inserted == 0:
        return 0

    # Contiguous trial_id renumbering keeps consumers that index into
    # ``trials`` by id happy, including the JSON written to disk.
    for new_id, t in enumerate(new_trials):
        t.trial_id = new_id

    session.trials = new_trials
    log.info(
        "fill_rest_gaps: inserted %d rest trial(s) into session %s",
        inserted,
        session.metadata.session_id,
    )
    return inserted


def count_inserted_rest_trials(session: "RecordingSession") -> int:
    """
    Return how many trials in ``session`` were created by ``fill_rest_gaps``.

    Useful for sanity checks or for stripping auto-inserted trials back
    out before re-running the filler with different parameters.
    """
    return sum(1 for t in session.trials if t.notes == INSERTED_NOTE)


def remove_inserted_rest_trials(session: "RecordingSession") -> int:
    """
    Reverse a previous ``fill_rest_gaps`` call by removing every trial
    whose ``notes`` match :data:`INSERTED_NOTE`.

    Returns the number of trials removed. Trial IDs are renumbered to
    stay contiguous.
    """
    kept = [t for t in session.trials if t.notes != INSERTED_NOTE]
    removed = len(session.trials) - len(kept)
    if removed == 0:
        return 0
    for new_id, t in enumerate(kept):
        t.trial_id = new_id
    session.trials = kept
    return removed
