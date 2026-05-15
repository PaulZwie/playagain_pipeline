"""
validation/corpus_report.py
───────────────────────────
Dataset-overview tables for Chapter 6 §6.1.

Three deliverables map directly to thesis tables:

* :func:`participant_summary`  → Table 6.1 (per-participant corpus summary).
  One row per subject, listing group (H/C), session-type counts (S/T/O),
  total sessions and total analysis windows after windowing.

* :func:`class_distribution`  → Table 6.2 (window count per gesture class).
  Pooled across the whole corpus, with raw count and percentage.

* :func:`recording_origins`  → §6.1.3 numerical summary.
  Breakdown into pipeline / Unity / outreach sources.

Everything here is pure-Python and operates on the existing
:class:`SessionCorpus` from ``corpus.py``. We deliberately don't touch
the corpus discovery itself — every signal we need is already in
``metadata.json`` (``protocol_name``, ``custom_metadata.source``,
``custom_metadata.participant_info`` …); we just have to read it.

Session-type taxonomy
─────────────────────
The thesis distinguishes three session origins:

    S  Structured protocol  → recorded with the cued protocol engine
                              (default ``"structured_4gesture"`` or any
                              custom protocol that isn't a training game)
    T  Training-game        → recorded through the game's data manager
                              while the participant was playing
    O  Outreach event       → flagged via custom_metadata.outreach = True
                              or protocol_name containing "outreach"

Sessions whose origin can't be inferred fall back to ``"S"`` — that's
how the recording dialog defaulted before the training-game flag was
introduced.

Participant group taxonomy
──────────────────────────
Each subject belongs to the **healthy** (``H``) or **impaired** (``I``)
cohort. The authoritative source is the explicit
:class:`participant_groups.ParticipantGroups` registry, which the
experimenter maintains by hand and passes in as ``groups=``. When no
registry entry exists the code falls back to
``custom_metadata.participant_info.group`` inside ``metadata.json``
(``healthy`` / ``control`` → ``H``; ``clinical`` / ``cp`` / ``stroke`` /
``hemiparetic`` → ``I``). Anything still unresolved is ``"?"``.

Game recordings
───────────────
``data/game_recordings/`` is a separate corpus (see
:mod:`game_corpus`). It is *not* discovered by ``SessionCorpus`` — the
runner could not train on it — but it *is* counted here when a
:class:`game_corpus.GameCorpus` is passed to :func:`participant_summary`
/ :func:`recording_origins` / :func:`write_corpus_report`, so the
Chapter-6 corpus overview reflects every recording the study collected.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .corpus import SessionCorpus, SessionRecord
from .participant_groups import (
    GROUP_UNKNOWN, ParticipantGroups, normalise_group,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Session-type / group inference
# ---------------------------------------------------------------------------

# Substrings inside protocol_name or custom_metadata.source that
# unambiguously mark a session as a training-game recording. Conservative
# list — we'd rather miss a few than mis-tag a structured session.
_TRAINING_GAME_MARKERS = ("training_game", "trainingsspiel", "exergame", "fuchs_")

# Same for outreach events. These are typically tagged explicitly in
# custom_metadata, but we also catch protocol names like
# "outreach_lange_nacht_2025" that we've seen in practice.
_OUTREACH_MARKERS = ("outreach", "lange_nacht", "open_day", "public_demo")


def infer_session_type(rec: SessionRecord) -> str:
    """
    Return ``"S"`` (structured) / ``"T"`` (training game) / ``"O"`` (outreach).

    Defaults to ``"S"`` when no marker matches.
    """
    meta = _read_full_metadata(rec)
    custom = (meta.get("custom_metadata") or {}) if isinstance(meta, dict) else {}

    # Explicit flags win — these are what new recordings write.
    if bool(custom.get("outreach")):
        return "O"
    if str(custom.get("source", "")).lower() == "training_game":
        return "T"

    # Heuristic fallback for legacy sessions that pre-date the source flag.
    protocol = str(meta.get("protocol_name") or "").lower()
    if any(m in protocol for m in _OUTREACH_MARKERS):
        return "O"
    if any(m in protocol for m in _TRAINING_GAME_MARKERS):
        return "T"

    return "S"


def infer_group(
    rec: SessionRecord,
    *,
    groups: Optional[ParticipantGroups] = None,
) -> str:
    """
    Return the cohort code ``"H"`` (healthy) / ``"I"`` (impaired) /
    ``"?"`` (unknown) for a session's subject.

    Resolution order:

    1. the explicit :class:`participant_groups.ParticipantGroups`
       registry, when one is passed in — this is the authoritative
       source the experimenter maintains by hand;
    2. ``custom_metadata.participant_info.group`` inside the session's
       ``metadata.json``, normalised through
       :func:`participant_groups.normalise_group` (so ``clinical`` /
       ``stroke`` / ``cp`` all resolve to ``"I"``);
    3. ``"?"`` when neither is available.

    Note: the impaired cohort is coded ``"I"`` (previously coded ``"A"``;
    cohort used ``"C"``). The two reporting layers now share one
    vocabulary — see :mod:`participant_groups`.
    """
    # 1) explicit registry wins
    if groups is not None:
        code = groups._map.get(str(rec.subject_id))  # noqa: SLF001
        if code:
            return code

    # 2) metadata inference
    meta = _read_full_metadata(rec)
    custom = (meta.get("custom_metadata") or {}) if isinstance(meta, dict) else {}
    pinfo = custom.get("participant_info") or {}
    raw = pinfo.get("group") or pinfo.get("cohort")
    return normalise_group(raw)


def _read_full_metadata(rec: SessionRecord) -> Dict[str, Any]:
    """
    Re-read ``metadata.json`` for a session, returning the inner
    ``metadata`` dict. SessionCorpus only keeps a flat subset; this
    reaches for the rest. Returns ``{}`` on any error so callers can
    treat missing data as "unknown".
    """
    meta_path = rec.path / "metadata.json"
    try:
        with meta_path.open("r", encoding="utf-8") as f:
            blob = json.load(f)
    except Exception as exc:  # noqa: BLE001
        log.debug("Could not re-read %s: %s", meta_path, exc)
        return {}
    # Two layouts in the wild: the new one has a top-level "metadata"
    # key; very old recorder versions flattened everything.
    inner = blob.get("metadata") if isinstance(blob, dict) else None
    return inner if isinstance(inner, dict) else (blob if isinstance(blob, dict) else {})


# ---------------------------------------------------------------------------
# Window-count estimate per session
# ---------------------------------------------------------------------------

def estimate_window_count(
    rec: SessionRecord,
    *,
    window_ms: int = 200,
    stride_ms: int = 50,
    drop_rest: bool = False,
) -> int:
    """
    Estimate the number of analysis windows this session contributes.

    We do not load the EMG matrix — instead we read trial boundaries
    from ``metadata.json`` and count strided windows that fit inside each
    valid trial. That matches what the windowing pipeline actually
    produces (windows are extracted *per trial*, never across boundaries)
    without paying the I/O cost of loading every ``data.npy``.

    Falls back to a duration-based estimate when no trial info is
    available, which is the case for some legacy Unity captures.
    """
    fs = rec.sampling_rate or 2000.0
    win = max(1, int(round(window_ms * fs / 1000.0)))
    stride = max(1, int(round(stride_ms * fs / 1000.0)))

    meta_path = rec.path / "metadata.json"
    try:
        with meta_path.open("r", encoding="utf-8") as f:
            blob = json.load(f)
    except Exception:  # noqa: BLE001
        return 0
    trials = blob.get("trials") or []

    total = 0
    for t in trials:
        if not t.get("is_valid", True):
            continue
        # Skip the calibration-sync wave-out, it never feeds the classifier.
        if t.get("trial_type") == "calibration_sync":
            continue
        # Skip rest trials when drop_rest is enabled. Rest may be
        # encoded as gesture_label = 0 or gesture_name = "rest".
        if drop_rest:
            if int(t.get("gesture_label", -1)) == 0:
                continue
            if str(t.get("gesture_name", "")).lower() == "rest":
                continue
        s = int(t.get("start_sample", 0))
        e = int(t.get("end_sample", 0))
        if e <= s + win:
            continue
        # Number of full windows that fit between s and e at this stride.
        total += 1 + (e - s - win) // stride

    if total == 0 and not trials:
        # Duration fallback. Only fires when no trials at all — keep
        # the answer rough and label it so the caller can filter.
        n_samples_npy = rec.path / "data.npy"
        if n_samples_npy.exists():
            try:
                import numpy as np  # local import
                arr = np.load(n_samples_npy, mmap_mode="r")
                n = arr.shape[0] if arr.ndim == 2 else len(arr)
                total = max(0, 1 + (n - win) // stride)
            except Exception:  # noqa: BLE001
                pass
    return int(total)


# ---------------------------------------------------------------------------
# Per-participant summary  →  Table 6.1
# ---------------------------------------------------------------------------

@dataclass
class ParticipantRow:
    """One row of Table 6.1 (per-participant corpus summary)."""
    subject_id:   str
    group:        str             # "H", "I", or "?"
    n_structured: int = 0         # column "S"
    n_training:   int = 0         # column "T"
    n_outreach:   int = 0         # column "O"
    n_game:       int = 0         # column "G" — game_recordings/ CSVs
    n_sessions:   int = 0         # S + T + O  (training-corpus sessions)
    n_windows:    int = 0         # analysis windows from the training corpus

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def participant_summary(
    corpus: SessionCorpus,
    *,
    window_ms: int = 200,
    stride_ms: int = 50,
    drop_rest: bool = False,
    groups: Optional[ParticipantGroups] = None,
    game_corpus: Optional[Any] = None,
) -> List[ParticipantRow]:
    """
    Build Table 6.1: one :class:`ParticipantRow` per subject.

    Window counts use the same window/stride the experiment will use
    so the table number matches what the runner produces; pass the
    values from ``cfg.windowing`` to keep them aligned.

    Parameters
    ----------
    groups :
        Optional explicit healthy/impaired registry. When supplied it
        overrides metadata-inferred groups (see :func:`infer_group`).
    game_corpus :
        Optional :class:`game_corpus.GameCorpus`. When supplied, each
        subject's ``n_game`` column is filled with the number of game
        recordings they contributed. Game recordings are *not* folded
        into ``n_sessions`` / ``n_windows`` — those stay defined as the
        training corpus only — so the columns remain unambiguous.
    """
    rows: Dict[str, ParticipantRow] = {}
    for rec in corpus.all():
        row = rows.get(rec.subject_id)
        if row is None:
            row = ParticipantRow(
                subject_id=rec.subject_id,
                group=infer_group(rec, groups=groups),
            )
            rows[rec.subject_id] = row
        elif row.group == GROUP_UNKNOWN:
            # If we discover the group on a later session of the same
            # subject, upgrade — first-seen wins otherwise.
            g = infer_group(rec, groups=groups)
            if g != GROUP_UNKNOWN:
                row.group = g

        kind = infer_session_type(rec)
        if   kind == "S": row.n_structured += 1
        elif kind == "T": row.n_training   += 1
        elif kind == "O": row.n_outreach   += 1
        row.n_sessions += 1

        try:
            n_win = estimate_window_count(
                rec, window_ms=window_ms, stride_ms=stride_ms, drop_rest=drop_rest,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("Window count failed for %s: %s", rec.path, exc)
            n_win = 0
        row.n_windows += n_win

    # Fold in the game-recording counts when a GameCorpus is supplied.
    if game_corpus is not None:
        try:
            game_recs = game_corpus.all()
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not enumerate game corpus: %s", exc)
            game_recs = []
        for grec in game_recs:
            subj = str(grec.subject_id)
            row = rows.get(subj)
            if row is None:
                # A subject that only ever produced game recordings.
                code = GROUP_UNKNOWN
                if groups is not None:
                    code = groups.group_of(subj)
                row = ParticipantRow(subject_id=subj, group=code)
                rows[subj] = row
            elif row.group == GROUP_UNKNOWN and groups is not None:
                code = groups.group_of(subj)
                if code != GROUP_UNKNOWN:
                    row.group = code
            row.n_game += 1

    # Deterministic order: by subject id.
    return [rows[k] for k in sorted(rows)]


# ---------------------------------------------------------------------------
# Class distribution  →  Table 6.2
# ---------------------------------------------------------------------------

@dataclass
class ClassDistributionRow:
    label:        int
    name:         str
    n_windows:    int
    fraction_pct: float


def class_distribution(
    corpus: SessionCorpus,
    *,
    window_ms: int = 200,
    stride_ms: int = 50,
    drop_rest: bool = False,
) -> List[ClassDistributionRow]:
    """
    Build Table 6.2: window counts per gesture class across the corpus.

    Resolves class names from each session's ``label_names`` first and
    falls back to the per-trial ``gesture_name`` when a label is absent
    from the table.
    """
    counts: Dict[int, int] = {}
    names:  Dict[int, str] = {}

    for rec in corpus.all():
        fs = rec.sampling_rate or 2000.0
        win = max(1, int(round(window_ms * fs / 1000.0)))
        stride = max(1, int(round(stride_ms * fs / 1000.0)))

        meta_path = rec.path / "metadata.json"
        try:
            with meta_path.open("r", encoding="utf-8") as f:
                blob = json.load(f)
        except Exception:  # noqa: BLE001
            continue
        trials = blob.get("trials") or []

        # Build a name lookup local to this session — class IDs are
        # the same across sessions if they use the same gesture_set,
        # but defensive coding is cheap and a tester ran a mixed-set
        # corpus once that flushed out this exact bug.
        local_names: Dict[int, str] = dict(rec.label_names or {})

        for t in trials:
            if not t.get("is_valid", True):
                continue
            if t.get("trial_type") == "calibration_sync":
                continue
            lbl = int(t.get("gesture_label", -1))
            if lbl < 0:
                continue
            if drop_rest and (
                lbl == 0 or str(t.get("gesture_name", "")).lower() == "rest"
            ):
                continue
            s = int(t.get("start_sample", 0))
            e = int(t.get("end_sample", 0))
            if e <= s + win:
                continue
            n = 1 + (e - s - win) // stride
            counts[lbl] = counts.get(lbl, 0) + n
            if lbl not in names:
                names[lbl] = local_names.get(lbl) or str(t.get("gesture_name") or f"class_{lbl}")

    total = sum(counts.values())
    out: List[ClassDistributionRow] = []
    for lbl in sorted(counts):
        n = counts[lbl]
        out.append(ClassDistributionRow(
            label=lbl,
            name=names.get(lbl, f"class_{lbl}"),
            n_windows=n,
            fraction_pct=(100.0 * n / total) if total else 0.0,
        ))
    return out


# ---------------------------------------------------------------------------
# Recording origins  →  §6.1.3
# ---------------------------------------------------------------------------

@dataclass
class RecordingOrigins:
    n_total:    int
    n_pipeline: int
    n_unity:    int
    n_outreach: int       # subset of pipeline that are outreach events
    n_game:     int = 0   # game_recordings/ CSVs (counted when a GameCorpus
                          # is supplied; 0 otherwise)

    def to_dict(self) -> Dict[str, int]:
        return asdict(self)


def recording_origins(
    corpus: SessionCorpus,
    *,
    game_corpus: Optional[Any] = None,
) -> RecordingOrigins:
    """
    Count sessions by source. ``n_outreach`` is a subset of
    ``n_pipeline`` (matches the thesis wording: outreach sessions are
    pipeline-origin sessions tagged separately).

    When ``game_corpus`` is supplied, ``n_game`` counts the
    game-recording CSVs and they are added into ``n_total`` — the
    game recordings are a distinct origin alongside pipeline and Unity,
    not a subset of either.
    """
    n_pipe = n_unity = n_outreach = 0
    for rec in corpus.all():
        if rec.source_domain == "unity":
            n_unity += 1
        else:
            n_pipe += 1
            if infer_session_type(rec) == "O":
                n_outreach += 1

    n_game = 0
    if game_corpus is not None:
        try:
            n_game = len(game_corpus.all())
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not count game corpus: %s", exc)
            n_game = 0

    return RecordingOrigins(
        n_total=n_pipe + n_unity + n_game,
        n_pipeline=n_pipe,
        n_unity=n_unity,
        n_outreach=n_outreach,
        n_game=n_game,
    )


# ---------------------------------------------------------------------------
# Bundled writer
# ---------------------------------------------------------------------------

def write_corpus_report(
    corpus: SessionCorpus,
    out_dir: Path,
    *,
    window_ms: int = 200,
    stride_ms: int = 50,
    drop_rest: bool = False,
    groups: Optional[ParticipantGroups] = None,
    game_corpus: Optional[Any] = None,
) -> Dict[str, Path]:
    """
    Write the three corpus tables as both CSV (for direct inclusion in
    LaTeX via ``csvsimple`` or copy-paste) and JSON (for plotting code
    in :mod:`plots_thesis`).

    Parameters
    ----------
    groups :
        Optional explicit healthy/impaired registry. When supplied the
        ``group`` column is authoritative rather than metadata-inferred.
    game_corpus :
        Optional :class:`game_corpus.GameCorpus`. When supplied, the
        participant table gains a populated ``G`` column and the
        recording-origins JSON gains ``n_game``.
    """
    import csv

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}

    # Table 6.1
    part_rows = participant_summary(
        corpus, window_ms=window_ms, stride_ms=stride_ms, drop_rest=drop_rest,
        groups=groups, game_corpus=game_corpus,
    )
    p_csv = out_dir / "table_6_1_participants.csv"
    with p_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["subject_id", "group", "S", "T", "O", "G",
                    "total_sessions", "total_windows"])
        total_s = total_w = total_g = 0
        for r in part_rows:
            w.writerow([r.subject_id, r.group, r.n_structured, r.n_training,
                        r.n_outreach, r.n_game, r.n_sessions, r.n_windows])
            total_s += r.n_sessions
            total_w += r.n_windows
            total_g += r.n_game
        w.writerow(["TOTAL", "", "", "", "", total_g, total_s, total_w])
    paths["participants_csv"] = p_csv

    # Table 6.2
    cls_rows = class_distribution(
        corpus, window_ms=window_ms, stride_ms=stride_ms, drop_rest=drop_rest,
    )
    c_csv = out_dir / "table_6_2_class_distribution.csv"
    with c_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["label", "name", "n_windows", "fraction_pct"])
        for r in cls_rows:
            w.writerow([r.label, r.name, r.n_windows, f"{r.fraction_pct:.2f}"])
    paths["class_distribution_csv"] = c_csv

    # Section 6.1.3
    ro = recording_origins(corpus, game_corpus=game_corpus)
    json_path = out_dir / "section_6_1_3_recording_origins.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(ro.to_dict(), f, indent=2)
    paths["recording_origins_json"] = json_path

    # Per-cohort headcount — small but it's the number the thesis quotes
    # when it says "N healthy and M impaired participants".
    group_counts: Dict[str, int] = {}
    for r in part_rows:
        group_counts[r.group] = group_counts.get(r.group, 0) + 1

    # Single combined JSON for the plotting/reporting tools.
    combined = {
        "participants": [r.to_dict() for r in part_rows],
        "class_distribution": [asdict(r) for r in cls_rows],
        "recording_origins": ro.to_dict(),
        "group_counts": group_counts,
        "group_registry_source": (str(groups.source)
                                  if (groups and groups.source) else None),
        "windowing": {"window_ms": window_ms, "stride_ms": stride_ms,
                      "drop_rest": drop_rest},
    }
    cj = out_dir / "corpus_report.json"
    with cj.open("w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2)
    paths["combined_json"] = cj

    return paths