"""
cv_strategies.py
────────────────
Cross-validation splitters that operate at the *session* level rather
than the *window* level.

Why session-level instead of window-level?
──────────────────────────────────────────
Windows extracted from the same continuous recording are highly
correlated: muscles fatigue, the bracelet drifts, the participant
relaxes. Splitting random windows from the same session into train and
test is the single biggest source of inflated accuracy in EMG papers.
This module enforces splits at recording granularity so that no two
windows from the same physical recording are ever in both train and
test.

All splitters follow the same protocol:

    fold_iter = strategy(records, **kwargs)
    for fold in fold_iter:
        train_records: List[SessionRecord] = fold["train"]
        test_records:  List[SessionRecord] = fold["test"]
        fold_id:       str                 = fold["id"]

That uniform interface lets the runner swap strategies via config
without any branching.
"""

from __future__ import annotations

import random
from typing import Dict, Iterable, Iterator, List, Optional

from .corpus import SessionRecord


Fold = Dict[str, object]   # {"id": str, "train": [SessionRecord], "test": [SessionRecord]}


# ---------------------------------------------------------------------------
# Within-session  (kept honest with a configurable cut-point)
# ---------------------------------------------------------------------------

def within_session(
    records: List[SessionRecord],
    *,
    test_fraction: float = 0.2,
) -> Iterator[Fold]:
    """
    The optimistic baseline: each session is a fold; the *temporal* tail
    of the session becomes its own test set.

    Important: the split is **temporal**, not random — the last
    ``test_fraction`` of the recording is used for testing. Random
    window splits within a session are NOT supported here on purpose, to
    discourage the inflated-accuracy trap described in the module
    docstring. If you really want a random split, do it at window
    extraction time inside your feature pipeline.

    The runner is expected to read the temporal cut from the fold
    metadata and apply it after extracting windows.
    """
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be in (0, 1)")
    for rec in records:
        yield {
            "id": f"within__{rec.subject_id}__{rec.session_id}",
            "train": [rec],
            "test": [rec],
            "test_fraction": test_fraction,   # consumed by the runner
            "split_kind": "temporal_tail",
        }


# ---------------------------------------------------------------------------
# Leave-One-Session-Out
# ---------------------------------------------------------------------------

def leave_one_session_out(records: List[SessionRecord]) -> Iterator[Fold]:
    """
    For each session, hold it out as the test set and train on the
    union of all *other* sessions. This reports how well a model
    generalises across recording sessions of the same subject as well
    as across subjects.
    """
    for i, held_out in enumerate(records):
        train = [r for j, r in enumerate(records) if j != i]
        if not train:
            continue
        yield {
            "id": f"loso_session__{held_out.subject_id}__{held_out.session_id}",
            "train": train,
            "test": [held_out],
            "split_kind": "loso_session",
        }


# ---------------------------------------------------------------------------
# Leave-One-Subject-Out
# ---------------------------------------------------------------------------

def leave_one_subject_out(records: List[SessionRecord]) -> Iterator[Fold]:
    """
    For each subject, hold out *all* of their sessions as the test set
    and train on every session from every other subject. This is the
    canonical generalisation test reported in the EMG literature and
    the most honest single number you can put in a paper.
    """
    subjects = sorted({r.subject_id for r in records})
    if len(subjects) < 2:
        return
    for held_subject in subjects:
        train = [r for r in records if r.subject_id != held_subject]
        test  = [r for r in records if r.subject_id == held_subject]
        if not train or not test:
            continue
        yield {
            "id": f"loso_subject__{held_subject}",
            "train": train,
            "test": test,
            "split_kind": "loso_subject",
        }


# ---------------------------------------------------------------------------
# Cross-domain: pipeline ↔ unity
# ---------------------------------------------------------------------------

def cross_domain(
    records: List[SessionRecord],
    *,
    train_domain: str,
    test_domain: str,
) -> Iterator[Fold]:
    """
    Train on every session from one source domain and test on every
    session from the other.

    This is the experiment that answers "does a model trained on the
    Python pipeline still work when the user plays the Unity game?",
    which directly motivates the Unity recording infrastructure.

    Two folds are emitted: one with the requested orientation and one
    swapped, so a single config produces a symmetric result.
    """
    valid = {"pipeline", "unity"}
    if train_domain not in valid or test_domain not in valid:
        raise ValueError(f"domain must be one of {valid}")
    if train_domain == test_domain:
        raise ValueError("cross_domain requires different train/test domains")

    train = [r for r in records if r.source_domain == train_domain]
    test  = [r for r in records if r.source_domain == test_domain]
    if not train or not test:
        return

    yield {
        "id": f"crossdomain__{train_domain}_to_{test_domain}",
        "train": train,
        "test": test,
        "split_kind": "cross_domain",
    }


# ---------------------------------------------------------------------------
# K-fold over subjects (for larger studies)
# ---------------------------------------------------------------------------

def k_fold_subjects(
    records: List[SessionRecord],
    *,
    k: int = 5,
    seed: int = 42,
) -> Iterator[Fold]:
    """
    K-fold cross-validation where the *unit* of splitting is a subject,
    not a window. Each subject lives entirely in one fold.

    Useful when there are too many subjects for full LOSO to be
    practical (LOSO with N=20 subjects = 20 retrains).
    """
    subjects = sorted({r.subject_id for r in records})
    if len(subjects) < k:
        # fall back to LOSO if there are not enough subjects
        yield from leave_one_subject_out(records)
        return

    rng = random.Random(seed)
    shuffled = subjects[:]
    rng.shuffle(shuffled)

    folds: List[List[str]] = [[] for _ in range(k)]
    for i, subj in enumerate(shuffled):
        folds[i % k].append(subj)

    for fi, test_subjects in enumerate(folds):
        test_set = set(test_subjects)
        train = [r for r in records if r.subject_id not in test_set]
        test  = [r for r in records if r.subject_id in test_set]
        if not train or not test:
            continue
        yield {
            "id": f"kfold_subjects__k{k}_seed{seed}__fold{fi}",
            "train": train,
            "test": test,
            "split_kind": "k_fold_subjects",
            "test_subjects": sorted(test_subjects),
        }


# ---------------------------------------------------------------------------
# Explicit train / val / test ratios
# ---------------------------------------------------------------------------

def holdout_split(
    records: List[SessionRecord],
    *,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    stratify_by: str = "subject",
) -> Iterator[Fold]:
    """
    A single fold with explicit train / val / test ratios.

    The split is performed at session granularity — entire sessions go
    into one bucket each, never windows from the same session into two
    different buckets. The order in which sessions are drawn is
    deterministic given ``seed``.

    Parameters
    ----------
    val_ratio : fraction of sessions used as the validation set.
    test_ratio : fraction of sessions used as the test set.
    seed : RNG seed for the deterministic shuffle.
    stratify_by : ``"subject"`` (default) draws each subject's sessions
        independently, so the val and test sets always contain *some*
        sessions from every subject — the most useful default for small
        cohorts. ``"none"`` shuffles all sessions globally and slices
        them, which is appropriate when the experiment cares about
        cross-subject generalisation.

    Emits a single fold with three populated session lists. The
    ``ValidationRunner`` knows how to consume the ``"val"`` slot and
    pass it as ``X_val`` / ``y_val`` to the model's ``train()`` method
    when the model supports it (MLP, CNN, AttentionNet, MSTNet).
    """
    if not 0.0 <= val_ratio < 1.0 or not 0.0 <= test_ratio < 1.0:
        raise ValueError("val_ratio and test_ratio must be in [0, 1)")
    if val_ratio + test_ratio >= 1.0:
        raise ValueError(
            f"val_ratio + test_ratio must be < 1.0 "
            f"(got {val_ratio} + {test_ratio} = {val_ratio + test_ratio})"
        )
    if not records:
        return

    rng = random.Random(seed)

    train: List[SessionRecord] = []
    val:   List[SessionRecord] = []
    test:  List[SessionRecord] = []

    if stratify_by == "subject":
        per_subject: dict = {}
        for r in records:
            per_subject.setdefault(r.subject_id, []).append(r)
        for subj in sorted(per_subject):
            recs = per_subject[subj][:]
            rng.shuffle(recs)
            n = len(recs)
            n_test = max(1, int(round(n * test_ratio))) if test_ratio > 0 and n > 1 else 0
            n_val  = max(1, int(round(n * val_ratio))) if val_ratio  > 0 and (n - n_test) > 1 else 0
            # Edge case: very few sessions — favour train, then test, then val.
            n_test = min(n_test, max(0, n - 1))
            n_val  = min(n_val,  max(0, n - n_test - 1))
            test.extend(recs[:n_test])
            val.extend(recs[n_test:n_test + n_val])
            train.extend(recs[n_test + n_val:])
    else:
        shuffled = records[:]
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_test = int(round(n * test_ratio))
        n_val  = int(round(n * val_ratio))
        test  = shuffled[:n_test]
        val   = shuffled[n_test:n_test + n_val]
        train = shuffled[n_test + n_val:]

    if not train:
        # Shouldn't happen with the validations above, but guard anyway.
        return

    yield {
        "id": (f"holdout__val{int(val_ratio*100)}_test{int(test_ratio*100)}"
               f"__seed{seed}__strat-{stratify_by}"),
        "train": train,
        "val":   val,
        "test":  test if test else train[:0],
        "split_kind": "holdout",
        "ratios": {
            "val": val_ratio, "test": test_ratio,
            "train": 1.0 - val_ratio - test_ratio,
        },
        "stratify_by": stratify_by,
    }


# ---------------------------------------------------------------------------
# Registry — used by the runner so YAML can name a strategy by string.
# ---------------------------------------------------------------------------

STRATEGIES = {
    "within_session":         within_session,
    "loso_session":           leave_one_session_out,
    "loso_subject":           leave_one_subject_out,
    "cross_domain":           cross_domain,
    "k_fold_subjects":        k_fold_subjects,
    "holdout_split":          holdout_split,
}


def get_strategy(name: str):
    """Look up a CV strategy by the name used in experiment YAML files."""
    if name not in STRATEGIES:
        raise KeyError(
            f"Unknown CV strategy '{name}'. "
            f"Available: {sorted(STRATEGIES)}"
        )
    return STRATEGIES[name]
