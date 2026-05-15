"""
validation/participant_groups.py
────────────────────────────────
Authoritative healthy / impaired participant grouping.

Why this module exists
──────────────────────
The study mixes two cohorts — **healthy** controls and **impaired**
participants (post-stroke / hemiparetic / otherwise motor-impaired).
Pooling them into one number hides the effect that actually matters for
the thesis, so every aggregate in the reporting layer needs to be able
to split by cohort.

The group of a subject *can* be inferred from
``custom_metadata.participant_info.group`` inside each session's
``metadata.json`` (see :func:`corpus_report.infer_group`), but in
practice that field was not filled in consistently across the whole
recording campaign. This module therefore provides an **explicit,
file-backed registry** that the experimenter maintains by hand and that
takes precedence over metadata inference.

Providing the mapping
─────────────────────
Point the registry at a small file you maintain yourself. Three
formats are accepted — pick whichever is least annoying to keep
up to date:

1. Grouped lists (recommended)::

       {
         "healthy":  ["VP_01", "VP_02", "VP_04", "VP_06", "VP_09", "VP_12"],
         "impaired": ["VP_03", "VP_11", "VP_13", "VP_14"]
       }

2. Flat subject → group map::

       {"VP_01": "healthy", "VP_13": "impaired", "VP_14": "impaired"}

3. CSV with a header ``subject_id,group``::

       subject_id,group
       VP_01,healthy
       VP_13,impaired

Default location is ``<data_dir>/participant_groups.json``. Call
:func:`write_template` once to drop a starter file next to your data.

Group vocabulary
────────────────
Canonical group ids are the short codes ``"H"`` (healthy), ``"I"``
(impaired) and ``"?"`` (unknown). A generous alias table maps the
words people actually type — ``control``, ``clinical``, ``cp``,
``stroke``, ``patient``, ``affected`` … — onto those codes, so the
file stays human-friendly while everything downstream compares short
codes.

This module has no Qt / sklearn / torch imports and never loads an EMG
matrix — it is safe to import from headless reporting jobs.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple, TypeVar

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Canonical vocabulary
# ---------------------------------------------------------------------------

GROUP_HEALTHY  = "H"
GROUP_IMPAIRED = "I"
GROUP_UNKNOWN  = "?"

#: Codes that represent a *resolved* group (i.e. not ``"?"``).
KNOWN_GROUPS: Tuple[str, str] = (GROUP_HEALTHY, GROUP_IMPAIRED)

#: Long, human-readable label for each code — used in table headers.
GROUP_LABELS: Dict[str, str] = {
    GROUP_HEALTHY:  "healthy",
    GROUP_IMPAIRED: "impaired",
    GROUP_UNKNOWN:  "unknown",
}

# Everything on the left normalises to the code on the right. Keys are
# compared lower-cased and stripped, so add new spellings freely.
_ALIASES: Dict[str, str] = {
    # healthy / control cohort
    "h": GROUP_HEALTHY,
    "healthy": GROUP_HEALTHY,
    "control": GROUP_HEALTHY,
    "controls": GROUP_HEALTHY,
    "able-bodied": GROUP_HEALTHY,
    "able_bodied": GROUP_HEALTHY,
    "unaffected": GROUP_HEALTHY,
    "non-clinical": GROUP_HEALTHY,
    # impaired / clinical cohort. "affected" and the legacy code "a"
    # are kept as accepted input aliases so registry files written
    # under the old vocabulary still load.
    "a": GROUP_IMPAIRED,
    "affected": GROUP_IMPAIRED,
    "impaired": GROUP_IMPAIRED,
    "c": GROUP_IMPAIRED,
    "clinical": GROUP_IMPAIRED,
    "patient": GROUP_IMPAIRED,
    "patients": GROUP_IMPAIRED,
    "cp": GROUP_IMPAIRED,
    "stroke": GROUP_IMPAIRED,
    "post-stroke": GROUP_IMPAIRED,
    "hemiparetic": GROUP_IMPAIRED,
    "hemiparesis": GROUP_IMPAIRED,
}


def normalise_group(raw: Any) -> str:
    """
    Map any user-supplied group spelling onto a canonical code.

    Returns :data:`GROUP_UNKNOWN` for empty / unrecognised values
    rather than raising — an unknown cohort is a reportable state, not
    an error.
    """
    if raw is None:
        return GROUP_UNKNOWN
    key = str(raw).strip().lower()
    if not key:
        return GROUP_UNKNOWN
    if key in (GROUP_HEALTHY.lower(), GROUP_IMPAIRED.lower(), GROUP_UNKNOWN):
        # already a code
        return key.upper() if key != GROUP_UNKNOWN else GROUP_UNKNOWN
    return _ALIASES.get(key, GROUP_UNKNOWN)


def group_label(code: str) -> str:
    """Long label (``"healthy"`` / ``"impaired"`` / ``"unknown"``) for a code."""
    return GROUP_LABELS.get(str(code), GROUP_LABELS[GROUP_UNKNOWN])


# ---------------------------------------------------------------------------
# Default file location
# ---------------------------------------------------------------------------

DEFAULT_GROUPS_FILENAME = "participant_groups.json"


def default_groups_path(data_dir: Path) -> Path:
    """Conventional location of the registry file for a given data dir."""
    return Path(data_dir) / DEFAULT_GROUPS_FILENAME


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_T = TypeVar("_T")


class ParticipantGroups:
    """
    A subject → cohort lookup, loaded from an explicit registry file
    with optional fallback to per-session metadata inference.

    Construct it with one of the classmethods (:meth:`from_file`,
    :meth:`from_data_dir`, :meth:`from_mapping`) or directly from a
    ``{subject_id: code}`` dict.

    The object is intentionally tiny and immutable-ish: callers treat it
    as a frozen lookup table for the duration of a report build.
    """

    def __init__(
        self,
        mapping: Optional[Mapping[str, str]] = None,
        *,
        source: Optional[Path] = None,
    ):
        # Stored already-normalised so every read is a plain dict.get().
        self._map: Dict[str, str] = {}
        for subj, grp in (mapping or {}).items():
            code = normalise_group(grp)
            if code != GROUP_UNKNOWN:
                self._map[str(subj)] = code
        self.source: Optional[Path] = Path(source) if source else None

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "ParticipantGroups":
        """Build directly from an in-memory ``{subject: group}`` mapping."""
        return cls(mapping)

    @classmethod
    def from_file(cls, path: Path) -> "ParticipantGroups":
        """
        Load a registry file. Accepts the three formats described in the
        module docstring (grouped-lists JSON, flat JSON, or CSV).

        Raises ``FileNotFoundError`` if the path doesn't exist — callers
        that want "use it if present" semantics should go through
        :meth:`from_data_dir` instead.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Participant-group file not found: {path}")

        if path.suffix.lower() == ".csv":
            mapping = cls._parse_csv(path)
        else:
            mapping = cls._parse_json(path)
        log.info("Loaded participant groups for %d subject(s) from %s",
                 len(mapping), path)
        return cls(mapping, source=path)

    @classmethod
    def from_data_dir(
        cls,
        data_dir: Path,
        *,
        filename: str = DEFAULT_GROUPS_FILENAME,
    ) -> "ParticipantGroups":
        """
        Load ``<data_dir>/<filename>`` if it exists; otherwise return an
        empty registry (every subject resolves to ``"?"`` until metadata
        fallback kicks in). Never raises on a missing file.
        """
        path = Path(data_dir) / filename
        if path.exists():
            try:
                return cls.from_file(path)
            except Exception as exc:  # noqa: BLE001
                log.warning("Could not read %s (%s) — continuing with no "
                            "explicit group registry.", path, exc)
        else:
            log.info("No participant-group file at %s — groups will fall "
                     "back to session-metadata inference.", path)
        return cls({}, source=path)

    # ------------------------------------------------------------------
    # Parsers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_json(path: Path) -> Dict[str, str]:
        with path.open("r", encoding="utf-8") as f:
            blob = json.load(f)
        if not isinstance(blob, dict):
            raise ValueError(f"{path.name}: expected a JSON object at the top level")

        # Optional wrapper key so the file can carry comments / metadata.
        if "groups" in blob and isinstance(blob["groups"], dict):
            blob = blob["groups"]

        out: Dict[str, str] = {}
        # Heuristic: grouped-lists if every value is a list; otherwise a
        # flat subject → group map.
        if blob and all(isinstance(v, (list, tuple)) for v in blob.values()):
            for grp, subjects in blob.items():
                code = normalise_group(grp)
                for subj in subjects:
                    out[str(subj)] = code
        else:
            for subj, grp in blob.items():
                out[str(subj)] = normalise_group(grp)
        return out

    @staticmethod
    def _parse_csv(path: Path) -> Dict[str, str]:
        out: Dict[str, str] = {}
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                return out
            # Be forgiving about column naming.
            lower = {c.lower().strip(): c for c in reader.fieldnames}
            subj_col = (lower.get("subject_id") or lower.get("subject")
                        or lower.get("vp") or reader.fieldnames[0])
            grp_col = (lower.get("group") or lower.get("cohort")
                       or (reader.fieldnames[1] if len(reader.fieldnames) > 1 else None))
            if grp_col is None:
                raise ValueError(f"{path.name}: need a 'group' column")
            for row in reader:
                subj = str(row.get(subj_col, "")).strip()
                if not subj:
                    continue
                out[subj] = normalise_group(row.get(grp_col))
        return out

    # ------------------------------------------------------------------
    # Lookups
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._map)

    def __contains__(self, subject_id: object) -> bool:
        return str(subject_id) in self._map

    @property
    def is_empty(self) -> bool:
        """True when no explicit assignments are loaded."""
        return not self._map

    def group_of(
        self,
        subject_id: str,
        *,
        fallback: Optional[Callable[[str], str]] = None,
    ) -> str:
        """
        Resolve a subject's cohort code.

        Resolution order:

        1. the explicit registry, if the subject is listed;
        2. ``fallback(subject_id)`` if supplied (e.g. a metadata-based
           inference closure — see :func:`metadata_group_resolver`);
        3. :data:`GROUP_UNKNOWN`.
        """
        code = self._map.get(str(subject_id))
        if code:
            return code
        if fallback is not None:
            try:
                inferred = normalise_group(fallback(str(subject_id)))
            except Exception:  # noqa: BLE001
                inferred = GROUP_UNKNOWN
            return inferred
        return GROUP_UNKNOWN

    def label_of(self, subject_id: str, **kw) -> str:
        """Long label for a subject's cohort (``group_of`` + ``group_label``)."""
        return group_label(self.group_of(subject_id, **kw))

    def subjects_in_group(self, group: str) -> List[str]:
        """
        Sorted list of subjects explicitly assigned to ``group``.

        ``group`` is normalised, so ``"healthy"``, ``"H"`` and
        ``"control"`` all return the same list. Only the explicit
        registry is consulted here — metadata fallback can't enumerate
        subjects it has never been asked about.
        """
        code = normalise_group(group)
        return sorted(s for s, g in self._map.items() if g == code)

    def as_subject_map(self) -> Dict[str, str]:
        """A plain ``{subject_id: code}`` copy of the explicit registry."""
        return dict(self._map)

    def counts(self) -> Dict[str, int]:
        """``{code: n_subjects}`` over the explicit registry."""
        out: Dict[str, int] = {GROUP_HEALTHY: 0, GROUP_IMPAIRED: 0}
        for code in self._map.values():
            out[code] = out.get(code, 0) + 1
        return out

    # ------------------------------------------------------------------
    # Splitting helpers — the workhorses for the reporting layer
    # ------------------------------------------------------------------

    def split_by_group(
        self,
        items: Iterable[_T],
        key: Callable[[_T], str],
        *,
        fallback: Optional[Callable[[str], str]] = None,
        include_unknown: bool = True,
    ) -> Dict[str, List[_T]]:
        """
        Bucket an iterable of arbitrary objects by the cohort of the
        subject each one belongs to.

        Parameters
        ----------
        items :
            Anything — ``SessionRecord``, ``RecordingDescriptor``,
            ``FoldStub``, plain dicts …
        key :
            Callable returning the *subject id* for one item.
        fallback :
            Optional per-subject inference used when the registry has no
            entry (see :func:`metadata_group_resolver`).
        include_unknown :
            When False, items whose group resolves to ``"?"`` are
            dropped instead of collected under the ``"?"`` key.

        Returns a dict keyed by group code. The healthy and impaired
        keys are always present (possibly empty); ``"?"`` appears only
        when ``include_unknown`` is True and at least one item lands
        there.
        """
        buckets: Dict[str, List[_T]] = {GROUP_HEALTHY: [], GROUP_IMPAIRED: []}
        for it in items:
            try:
                subj = str(key(it))
            except Exception:  # noqa: BLE001
                subj = ""
            code = self.group_of(subj, fallback=fallback)
            if code == GROUP_UNKNOWN and not include_unknown:
                continue
            buckets.setdefault(code, []).append(it)
        return buckets

    def filter_group(
        self,
        items: Iterable[_T],
        group: str,
        key: Callable[[_T], str],
        *,
        fallback: Optional[Callable[[str], str]] = None,
    ) -> List[_T]:
        """Return only the items belonging to ``group`` (normalised)."""
        code = normalise_group(group)
        out: List[_T] = []
        for it in items:
            try:
                subj = str(key(it))
            except Exception:  # noqa: BLE001
                continue
            if self.group_of(subj, fallback=fallback) == code:
                out.append(it)
        return out

    # ------------------------------------------------------------------

    def merged_with_inference(
        self,
        subjects: Iterable[str],
        resolver: Callable[[str], str],
    ) -> "ParticipantGroups":
        """
        Return a *new* registry that keeps every explicit assignment and
        additionally pins down any of ``subjects`` that were unknown,
        using ``resolver`` (typically :func:`metadata_group_resolver`).

        Handy when you want a single fully-resolved table to serialise
        into a report so the output is self-describing.
        """
        merged = dict(self._map)
        for subj in subjects:
            subj = str(subj)
            if subj in merged:
                continue
            code = normalise_group(resolver(subj))
            if code != GROUP_UNKNOWN:
                merged[subj] = code
        return ParticipantGroups(merged, source=self.source)


# ---------------------------------------------------------------------------
# Metadata-inference fallback
# ---------------------------------------------------------------------------

def metadata_group_resolver(corpus: Any) -> Callable[[str], str]:
    """
    Build a ``subject_id -> group_code`` closure backed by per-session
    metadata inference.

    ``corpus`` is anything exposing ``.all()`` → an iterable of records
    with ``.subject_id`` (i.e. a :class:`corpus.SessionCorpus`, or the
    :class:`game_corpus.GameCorpus` added in this bundle). The first
    session of each subject whose ``metadata.json`` carries a
    recognisable ``participant_info.group`` wins.

    This is deliberately a closure rather than eager work: callers pass
    it as the ``fallback=`` argument to :meth:`ParticipantGroups.group_of`
    / :meth:`split_by_group`, and it only does I/O for subjects the
    explicit registry didn't already cover.
    """
    # Imported lazily so this module has no hard dependency on
    # corpus_report (and vice-versa — corpus_report imports nothing from
    # here at module load time).
    try:
        from .corpus_report import infer_group as _infer_group
    except Exception:  # noqa: BLE001
        _infer_group = None  # type: ignore[assignment]

    # Group records by subject once.
    by_subject: Dict[str, List[Any]] = {}
    try:
        for rec in corpus.all():
            by_subject.setdefault(str(rec.subject_id), []).append(rec)
    except Exception as exc:  # noqa: BLE001
        log.debug("metadata_group_resolver: corpus.all() failed (%s)", exc)

    cache: Dict[str, str] = {}

    def _resolve(subject_id: str) -> str:
        subject_id = str(subject_id)
        if subject_id in cache:
            return cache[subject_id]
        code = GROUP_UNKNOWN
        if _infer_group is not None:
            for rec in by_subject.get(subject_id, []):
                try:
                    code = normalise_group(_infer_group(rec))
                except Exception:  # noqa: BLE001
                    code = GROUP_UNKNOWN
                if code != GROUP_UNKNOWN:
                    break
        cache[subject_id] = code
        return code

    return _resolve


# ---------------------------------------------------------------------------
# Template writer
# ---------------------------------------------------------------------------

def write_template(
    path: Path,
    *,
    healthy: Optional[Iterable[str]] = None,
    impaired: Optional[Iterable[str]] = None,
    overwrite: bool = False,
) -> Path:
    """
    Drop a starter ``participant_groups.json`` at ``path`` so the
    experimenter only has to move subject ids between two lists.

    Pass ``healthy`` / ``impaired`` to pre-fill (e.g. from a quick first
    pass over the corpus); leave them empty for a blank template.
    Refuses to clobber an existing file unless ``overwrite=True``.
    """
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"{path} already exists — pass overwrite=True to replace it."
        )
    payload = {
        "_comment": (
            "Healthy vs impaired participant registry. Move VP_ ids "
            "between the two lists. Recognised group spellings include "
            "healthy/control and impaired/affected/clinical/stroke/cp. "
            "This file takes precedence over metadata inference."
        ),
        "groups": {
            "healthy":  sorted(set(map(str, healthy or []))),
            "impaired": sorted(set(map(str, impaired or []))),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    log.info("Wrote participant-group template to %s", path)
    return path