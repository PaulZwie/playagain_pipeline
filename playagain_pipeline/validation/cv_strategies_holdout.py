"""
cv_strategies_holdout.py
────────────────────────
Reference implementation of the ``holdout_split`` CV strategy.

Why this is a separate file
───────────────────────────
The runner (v1 and v2) references ``holdout_split`` in comments and the
GUI tab builds configs for it, but I couldn't confirm the strategy is
actually registered in your ``cv_strategies.py`` because that file
wasn't in the uploaded bundle. This file provides a drop-in reference.

How to use it
─────────────
EITHER copy the ``holdout_split`` function body into your existing
``playagain_pipeline/validation/cv_strategies.py`` and register it in
whatever ``STRATEGIES`` dict ``get_strategy`` looks up …

OR place this file at ``playagain_pipeline/validation/cv_strategies_holdout.py``
and add one line to ``cv_strategies.py``:

    from .cv_strategies_holdout import holdout_split
    STRATEGIES["holdout_split"] = holdout_split   # or equivalent

Contract
────────
A CV strategy is a callable:

    strategy(records: List[SessionRecord], **kwargs) -> Iterable[dict]

yielding fold dicts with at least:

    {
        "id":         str,                     # unique, human-readable
        "idx":        int,                     # 0-based, for seeding
        "train":      List[SessionRecord],
        "test":       List[SessionRecord],
        "val":        List[SessionRecord],     # optional; [] if absent
        "split_kind": str,                     # see runner for special-cases
    }

For ``holdout_split`` we yield exactly one fold.

Behaviour
─────────
• ``val_ratio``, ``test_ratio`` are fractions of the *session* list, not
  window counts. Sessions are never mixed across splits — this prevents
  window-level leakage of the same session into train and test.
• ``stratify_by="subject"`` (default) makes sure each subject's sessions
  are distributed across train/val/test proportionally. If a subject
  has fewer sessions than splits, we put them all into train.
• ``stratify_by="none"`` does a global shuffle.
• ``seed`` is respected; default 42.

Edge cases
──────────
• ``val_ratio == 0`` → no val split (caller sees ``val=[]``), which the
  runner translates to "let the model do its own internal split".
• ``test_ratio == 0`` → ValueError; a holdout with no test set is
  meaningless.
• Fewer than ~3 sessions total → ValueError with a clear message; the
  GUI's preview panel catches this before a click but belt-and-braces
  in the backend too.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, Iterable, List

# These imports assume the file lives alongside corpus.py inside the
# validation package. Adjust if placing elsewhere.
from .corpus import SessionRecord


def holdout_split(
    records: List[SessionRecord],
    *,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    stratify_by: str = "subject",
    seed: int = 42,
    **_ignored,
) -> Iterable[Dict]:
    """
    Yield one fold with an explicit train / val / test session split.

    See module docstring for the full contract.
    """
    records = list(records)
    if not records:
        return
    if test_ratio <= 0:
        raise ValueError(
            "holdout_split: test_ratio must be > 0 — a holdout without "
            "a test set doesn't evaluate anything."
        )
    if val_ratio < 0 or val_ratio + test_ratio >= 1.0:
        raise ValueError(
            f"holdout_split: need 0 ≤ val_ratio + test_ratio < 1, "
            f"got val={val_ratio}, test={test_ratio}."
        )
    if len(records) < 3:
        raise ValueError(
            f"holdout_split: need at least 3 sessions for a usable "
            f"train/val/test split (got {len(records)})."
        )

    rng = random.Random(int(seed))

    if stratify_by == "subject":
        train, val, test = _split_per_subject(records, val_ratio, test_ratio, rng)
    else:
        train, val, test = _split_global(records, val_ratio, test_ratio, rng)

    yield {
        "id":         "holdout",
        "idx":        0,
        "train":      train,
        "val":        val,
        "test":       test,
        "split_kind": "holdout",
    }


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

def _split_global(
    records: List[SessionRecord],
    val_ratio: float,
    test_ratio: float,
    rng: random.Random,
) -> tuple:
    """Shuffle all records then slice."""
    items = list(records)
    rng.shuffle(items)
    n = len(items)
    n_test = max(1, int(round(n * test_ratio)))
    n_val  = int(round(n * val_ratio))
    # Layout:  [TEST | VAL | TRAIN]  so small edge-cases never eat train
    test  = items[:n_test]
    val   = items[n_test : n_test + n_val] if n_val > 0 else []
    train = items[n_test + n_val :]
    return train, val, test


def _split_per_subject(
    records: List[SessionRecord],
    val_ratio: float,
    test_ratio: float,
    rng: random.Random,
) -> tuple:
    """
    For each subject, carve off proportional val/test chunks and leave
    the rest in train. Small-subject protection: if a subject has fewer
    than 1/(test_ratio) sessions, their sessions all go to train — we'd
    rather under-represent them in test than have a 1-session 'test' set
    dominated by one subject's idiosyncrasies.
    """
    by_subject: Dict[str, List[SessionRecord]] = defaultdict(list)
    for r in records:
        by_subject[r.subject_id].append(r)

    train, val, test = [], [], []
    min_for_test = int(1.0 / max(test_ratio, 1e-6))

    for subject in sorted(by_subject):
        items = list(by_subject[subject])
        rng.shuffle(items)
        if len(items) < min_for_test:
            train.extend(items)
            continue
        n = len(items)
        n_test = max(1, int(round(n * test_ratio)))
        n_val  = int(round(n * val_ratio))
        # Same TEST | VAL | TRAIN layout as global.
        test.extend(items[:n_test])
        if n_val > 0:
            val.extend(items[n_test : n_test + n_val])
        train.extend(items[n_test + n_val:])

    # Protect against the pathological case where every subject was
    # too small: if test ended up empty, promote a random session.
    if not test and train:
        promoted = rng.choice(train)
        train.remove(promoted)
        test.append(promoted)

    return train, val, test
