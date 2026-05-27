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
    for i, rec in enumerate(records):
        yield {
            "id": f"within__{rec.subject_id}__{rec.session_id}",
            "idx": i,          # stable fold index for per-fold seeding
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
    Pooled-subject LOSO-session.

    For each session, hold it out as the test set and train on the
    union of *all* other sessions, including sessions from other
    subjects. This reports cross-subject + cross-session generalisation
    in a single number, but it does *not* match the most common
    deployment story (use a participant's earlier sessions to fit a
    model that runs on their next session). Use
    :func:`intra_subject_loso_session` for that.
    """
    for i, held_out in enumerate(records):
        train = [r for j, r in enumerate(records) if j != i]
        if not train:
            continue
        yield {
            "id": f"loso_session__{held_out.subject_id}__{held_out.session_id}",
            "idx": i,          # stable fold index for per-fold seeding
            "train": train,
            "test": [held_out],
            "split_kind": "loso_session",
        }


def intra_subject_loso_session(
    records: List[SessionRecord],
) -> Iterator[Fold]:
    """
    Per-participant LOSO-session — the everyday deployment scenario.

    For each subject with two or more sessions, hold each of that
    subject's sessions out in turn and train on the *remaining
    sessions of the same subject only*. No other subject's data is
    used. This answers the question the deployed system actually
    cares about: "given that I have N earlier recordings from this
    participant, how well does a model fit on those generalise to
    their next recording session?"

    Subjects with a single session are silently skipped (there's
    nothing to leave out for them). The fold-id grammar is the same
    as ``leave_one_session_out`` so the thesis aggregators (Table 6.3,
    per-session variability figure) light up unchanged; the
    ``split_kind`` differs so consumers that care can branch on it.

    Empirically this is the cheapest of the LOSO variants too: each
    fold's train set is one subject's sessions rather than the whole
    pooled corpus, so per-fold matrices are ~N_subjects× smaller in
    RAM than pooled LOSO and SVM/RF/CatBoost fits proportionally
    faster.
    """
    by_subject: Dict[str, List[SessionRecord]] = {}
    for r in records:
        by_subject.setdefault(r.subject_id, []).append(r)

    # Global fold counter so idx is unique across all subjects, which
    # keeps the per-fold seed stable regardless of how many models are
    # in the experiment config (the runner falls back to eval_idx when
    # "idx" is absent, making seeds model-order-dependent without this).
    global_idx = 0
    for subject in sorted(by_subject):
        subj_records = by_subject[subject]
        if len(subj_records) < 2:
            continue
        for i, held_out in enumerate(subj_records):
            train = [r for j, r in enumerate(subj_records) if j != i]
            if not train:
                continue
            yield {
                "id": (
                    f"intra_loso_session__{held_out.subject_id}"
                    f"__{held_out.session_id}"
                ),
                "idx": global_idx,   # stable; independent of model count
                "train": train,
                "test": [held_out],
                "split_kind": "intra_subject_loso_session",
            }
            global_idx += 1


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
    for i, held_subject in enumerate(subjects):
        train = [r for r in records if r.subject_id != held_subject]
        test  = [r for r in records if r.subject_id == held_subject]
        if not train or not test:
            continue
        yield {
            "id": f"loso_subject__{held_subject}",
            "idx": i,          # stable fold index for per-fold seeding
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

def k_fold_sessions(
    records: List[SessionRecord],
    *,
    k: int = 10,
    seed: int = 42,
) -> Iterator[Fold]:
    """
    K-fold cross-validation where the *unit* of splitting is a session,
    not a window or a subject. Each session lives entirely in one
    fold's test set; train is the union of all other folds.

    This is the standard substitute for ``leave_one_session_out`` when
    LOSO is impractical. With ~100 sessions, LOSO produces 100 folds
    (one retrain each); K=10 produces 10 folds with ten sessions held
    out per fold. Total compute drops ~10× while still preserving the
    session-boundary guarantee that no two windows from the same
    recording end up in both train and test.

    Subject balance is *not* enforced — a fold may contain multiple
    sessions from the same subject, which is the right behaviour when
    the goal is to estimate per-session generalisation. Use
    ``k_fold_subjects`` instead when the goal is cross-subject
    generalisation.

    Falls back to ``leave_one_session_out`` when ``k >= len(records)``
    so a misconfigured K never silently degrades to "train on
    everything".
    """
    if k < 2:
        raise ValueError(f"k must be >= 2 (got {k})")
    if len(records) < k:
        # Honestly: there aren't enough sessions for K-fold to mean
        # anything different from LOSO. Yield LOSO instead.
        yield from leave_one_session_out(records)
        return

    rng = random.Random(seed)
    shuffled = list(records)
    rng.shuffle(shuffled)

    folds: List[List[SessionRecord]] = [[] for _ in range(k)]
    for i, rec in enumerate(shuffled):
        folds[i % k].append(rec)

    for fi, test_records in enumerate(folds):
        test_ids = {(r.subject_id, r.session_id) for r in test_records}
        train = [r for r in shuffled
                 if (r.subject_id, r.session_id) not in test_ids]
        if not train or not test_records:
            continue
        yield {
            "id": f"kfold_sessions__k{k}_seed{seed}__fold{fi}",
            "idx": fi,
            "train": train,
            "test": test_records,
            "split_kind": "k_fold_sessions",
            # Subjects represented in this fold's test set — useful for
            # per-subject aggregation tables that still want a "which
            # subjects fell here" column.
            "test_subjects": sorted({r.subject_id for r in test_records}),
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
    "within_session":              within_session,
    "loso_session":                leave_one_session_out,
    "intra_subject_loso_session":  intra_subject_loso_session,
    "loso_subject":                leave_one_subject_out,
    "cross_domain":                cross_domain,
    "k_fold_sessions":             k_fold_sessions,
    "k_fold_subjects":             k_fold_subjects,
    "holdout_split":               holdout_split,
}


def get_strategy(name: str):
    """Look up a CV strategy by the name used in experiment YAML files."""
    if name not in STRATEGIES:
        raise KeyError(
            f"Unknown CV strategy '{name}'. "
            f"Available: {sorted(STRATEGIES)}"
        )
    return STRATEGIES[name]