"""
validation/recompute_calibration_metrics.py
───────────────────────────────────────────
Walk a saved corpus and rewrite each session's ``metadata.json`` with
the new stability-based ``rotation_confidence`` value.

v2 (the v1 version skipped every session because it tried to parse
trial annotations out of ``metadata.json`` directly. Trials actually
live on the ``RecordingSession`` object and are loaded by
``RecordingSession.load()``. This version uses that API — the same
loader the live calibrator uses — so every session that the GUI can
calibrate is also re-processable from the CLI.)

What it does
────────────
For each session under ``data/sessions/<subject>/<session_id>/``:

  1. Load it via ``RecordingSession.load(session_dir)`` — the proper
     API. This handles signal + trials + metadata in one go.
  2. Pick sync-gesture trials in the same priority order as the live
     calibrator:
        * ``session.get_calibration_trials()`` (dedicated waveout
          recordings at the start of newer sessions) if present;
        * otherwise the highest-priority sync gesture available in
          ``session.get_valid_trials()`` —
          waveout > fist > open > waveIn > pinch > tripod.
  3. Read the matching reference pattern from the saved reference
     calibration (per signal mode — monopolar vs bipolar).
  4. Compute the stability metric across those trials.
  5. Rewrite ``metadata.json`` in place:
        * ``rotation_confidence``        ← stability  (new primary)
        * ``rotation_peak_prominence``   ← legacy z-score (preserved)
        * ``rotation_metric_version``    ← 2
        * ``rotation_stability_metrics`` ← full diagnostic block

The detected ``rotation_offset`` is NOT changed unless
``--update-offset`` is passed, because every downstream dataset that
used the old channel mapping depends on it staying stable.

Safety
──────
* ``--dry-run`` prints what would change but writes nothing.
* Each rewrite uses temp-file + rename so a crash mid-script cannot
  corrupt your data.
* Every overwritten metadata.json is backed up alongside as
  ``metadata.json.bak`` (first run only — subsequent runs find an
  existing .bak and leave it alone).
* The script is idempotent — running it twice in a row is a no-op
  on the second pass.

Usage
─────
    python -m playagain_pipeline.validation.recompute_calibration_metrics \\
        --data-dir data/ \\
        --reference calibrations/reference_calibration.json \\
        [--bootstrap 200] [--dry-run] [--verbose]
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from playagain_pipeline.calibration.calibration_stability import (
    StabilityResult, compute_stability_metrics,
)

# The proper way to load a session — same path DataManager.load_session
# uses. Importing this top-level rather than lazily so a misconfigured
# environment fails fast with a clear traceback.
from playagain_pipeline.core.session import RecordingSession

log = logging.getLogger(__name__)


# Same priority list the live calibrator uses
# (CalibrationProcessor._select_sync_pattern).
SYNC_PRIORITY = ("waveout", "fist", "open", "wavein", "pinch", "tripod")


# ---------------------------------------------------------------------------
# Trial extraction — mirrors what CalibrationProcessor does internally
# ---------------------------------------------------------------------------

def _signal_from_session(session: RecordingSession) -> Optional[np.ndarray]:
    """Tolerant signal extraction, matching CalibrationValidator's helper."""
    getter = getattr(session, "get_data", None)
    if callable(getter):
        data = getter()
        return np.asarray(data) if data is not None else None
    for attr in ("data", "signal", "raw_data"):
        val = getattr(session, attr, None)
        if val is not None:
            return np.asarray(val)
    return None


def _valid_trials(session: RecordingSession) -> List[Any]:
    """Prefer ``get_valid_trials()``, fall back to ``trials``."""
    getter = getattr(session, "get_valid_trials", None)
    if callable(getter):
        try:
            trials = getter()
        except Exception:  # noqa: BLE001
            trials = None
        if trials:
            return list(trials)
    trials = getattr(session, "trials", None)
    return list(trials) if trials else []


def _calibration_trials(session: RecordingSession) -> List[Any]:
    """Dedicated calibration-sync trials, if the session has any.

    Newer sessions record a few waveout trials at the very start of
    each session for use as the calibration sync gesture. The live
    calibrator prefers these over the regular gesture trials — we
    mirror that.
    """
    getter = getattr(session, "get_calibration_trials", None)
    if not callable(getter):
        return []
    try:
        out = getter()
    except Exception:  # noqa: BLE001
        return []
    return list(out) if out else []


def _select_sync_trials(
    session: RecordingSession, signal: np.ndarray,
) -> Tuple[str, List[np.ndarray]]:
    """
    Return ``(gesture_name, list of (n_samples, n_channels) chunks)``.

    Order of preference:
      1. ``session.get_calibration_trials()`` if non-empty → "waveout"
      2. Highest-priority sync gesture among the valid trials.
      3. Any non-rest gesture (so we degrade gracefully on legacy data
         that doesn't follow the protocol).
      4. ``("", [])`` if nothing usable.
    """
    n_samples = signal.shape[0]

    def _chunks_for(trials: List[Any]) -> Dict[str, List[np.ndarray]]:
        by_gesture: Dict[str, List[np.ndarray]] = {}
        for t in trials:
            name = str(getattr(t, "gesture_name", "")).strip()
            if not name:
                continue
            try:
                s = int(getattr(t, "start_sample", 0))
                e = int(getattr(t, "end_sample", s))
            except (TypeError, ValueError):
                continue
            if e <= s or s < 0 or e > n_samples:
                continue
            chunk = signal[s:e]
            if chunk.shape[0] < 10:
                continue
            by_gesture.setdefault(name, []).append(chunk)
        return by_gesture

    # 1. Dedicated calibration sync trials, if any
    cal = _calibration_trials(session)
    if cal:
        by_gesture = _chunks_for(cal)
        # Calibration trials almost always carry the "waveout" gesture
        # name; if not, pick the most frequent one.
        if by_gesture:
            best = max(by_gesture, key=lambda k: len(by_gesture[k]))
            return best, by_gesture[best]

    # 2 + 3. Fall back to valid trials, priority-ordered substring match
    valid = _valid_trials(session)
    if not valid:
        return "", []
    by_gesture = _chunks_for(valid)
    if not by_gesture:
        return "", []
    for token in SYNC_PRIORITY:
        for k in by_gesture:
            if token in k.lower():
                return k, by_gesture[k]
    for k, v in by_gesture.items():
        if "rest" not in k.lower():
            return k, v
    return "", []


# ---------------------------------------------------------------------------
# Reference loading (loaded ONCE at startup, cached per signal mode)
# ---------------------------------------------------------------------------

class ReferenceBank:
    """
    Holds reference energy patterns for every signal mode we'll need.

    Loaded once at the start of ``walk_corpus`` from disk, then consulted
    per-session. The reason this is a separate object — and not just two
    functions — is that we want one place to validate the reference at
    startup (and log clearly what's in it) rather than mysteriously
    failing on every individual session.
    """

    def __init__(self) -> None:
        self._patterns_per_mode: Dict[str, Dict[str, np.ndarray]] = {}
        self._sources: Dict[str, Path] = {}

    @property
    def modes(self) -> List[str]:
        return sorted(self._patterns_per_mode)

    def load_from(self, reference_arg: Path, mode: str) -> bool:
        """Load the reference for one mode, return whether it succeeded."""
        ref_path = _resolve_reference_for_mode(reference_arg, mode)
        if not ref_path.exists():
            log.warning("No reference file found at %s (mode=%s)",
                        ref_path.resolve(), mode)
            return False
        try:
            with ref_path.open("r", encoding="utf-8") as f:
                blob = json.load(f)
        except Exception as exc:  # noqa: BLE001
            log.error("Could not read reference %s: %s", ref_path, exc)
            return False
        patterns = blob.get("reference_patterns")
        if not isinstance(patterns, dict) or not patterns:
            log.error(
                "Reference %s has no usable 'reference_patterns' "
                "(top-level keys: %s).",
                ref_path, sorted(blob.keys()),
            )
            return False
        # Coerce to numpy + drop empty/malformed entries
        clean: Dict[str, np.ndarray] = {}
        for k, v in patterns.items():
            try:
                arr = np.asarray(v, dtype=np.float64)
                if arr.size:
                    clean[k] = arr
            except Exception:  # noqa: BLE001
                continue
        if not clean:
            log.error("Reference %s has no numeric pattern arrays.", ref_path)
            return False
        self._patterns_per_mode[mode] = clean
        self._sources[mode] = ref_path
        ref_mode = _normalize_signal_mode(
            (blob.get("metadata") or {}).get("signal_mode", mode)
        )
        log.info("Loaded reference for mode=%s from %s", mode, ref_path)
        log.info("  declared signal_mode in file:  %s", ref_mode)
        log.info("  num_channels (top-level):     %s",
                 blob.get("num_channels", "n/a"))
        log.info("  patterns available:           %s", sorted(clean.keys()))
        return True

    def lookup(
        self, mode: str, sync_gesture: str,
    ) -> Tuple[Optional[np.ndarray], str, str]:
        """
        Return ``(energy, used_key, reason)`` for one (mode, gesture).

        ``reason`` is ``""`` on success; otherwise a short human
        description of why the lookup failed.
        """
        patterns = self._patterns_per_mode.get(mode)
        if patterns is None:
            return None, "", f"reference not loaded for mode={mode}"
        target = (sync_gesture or "").lower()
        # 1. Direct / substring match against the actual sync gesture
        for k, v in patterns.items():
            if k == "__combined__":
                continue
            if target and (target in k.lower() or k.lower() in target):
                return v, k, ""
        # 2. Priority-list fallback (sync gesture might be named oddly,
        #    e.g. cal_waveout vs waveout)
        for token in SYNC_PRIORITY:
            for k, v in patterns.items():
                if k != "__combined__" and token in k.lower():
                    return v, k, ""
        # 3. Combined energy as the last resort
        if "__combined__" in patterns:
            return patterns["__combined__"], "__combined__", ""
        return None, "", (
            f"no pattern matches '{sync_gesture}' and no __combined__ "
            f"in {sorted(patterns.keys())}"
        )


def _normalize_signal_mode(mode: Optional[str]) -> str:
    """Match AutoCalibrator._normalize_signal_mode exactly."""
    m = str(mode or "monopolar").strip().lower()
    return "bipolar" if m == "bipolar" else "monopolar"


def _session_signal_mode(session: RecordingSession) -> str:
    """Read the signal mode (monopolar / bipolar) off the session.

    Mirrors ``AutoCalibrator._extract_session_signal_mode``: looks in
    ``session.metadata.custom_metadata`` for either a ``signal_mode``
    string or a ``bipolar_mode`` boolean flag. Defaults to monopolar
    when nothing else is known.
    """
    meta = getattr(session, "metadata", None)
    if meta is None:
        return "monopolar"
    custom = getattr(meta, "custom_metadata", None) or {}
    if isinstance(custom, dict):
        m = custom.get("signal_mode")
        if isinstance(m, str):
            return _normalize_signal_mode(m)
        if bool(custom.get("bipolar_mode", False)):
            return "bipolar"
    # Direct attributes too — be tolerant of older recorders.
    for attr in ("signal_mode", "mode"):
        v = getattr(meta, attr, None)
        if isinstance(v, str) and v:
            return _normalize_signal_mode(v)
    return "monopolar"


def _resolve_reference_for_mode(
    reference_arg: Path, signal_mode: str,
) -> Path:
    """
    Allow either a single reference file or a per-mode reference layout.

    If ``reference_arg`` is a file, return it as-is.
    If it's a directory, look for the AutoCalibrator naming scheme:
        reference_calibration.json                (default / monopolar)
        reference_calibration_<mode>.json
    """
    reference_arg = Path(reference_arg)
    if reference_arg.is_file():
        return reference_arg
    if reference_arg.is_dir():
        # Mode-specific first
        cand = reference_arg / f"reference_calibration_{signal_mode}.json"
        if cand.exists():
            return cand
        cand = reference_arg / "reference_calibration.json"
        if cand.exists():
            return cand
    return reference_arg   # let downstream report the missing path


# ---------------------------------------------------------------------------
# I/O for metadata.json
# ---------------------------------------------------------------------------

# Fields that the previous (buggy) version of this script placed at the
# top level of metadata.json. ``RecordingMetadata`` is a strict dataclass
# that rejects unknown keyword arguments, so once these were there,
# every subsequent ``RecordingSession.load()`` call crashed. The
# sanitizer below detects them and moves them into ``custom_metadata``
# where free-form fields belong.
MISPLACED_TOP_LEVEL_FIELDS = (
    "rotation_peak_prominence",
    "rotation_metric_version",
    "rotation_stability_metrics",
)


def _atomic_write_json(path: Path, payload: Any) -> None:
    """Write JSON to a temp file in the same directory, then rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with open(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)
        Path(tmp).replace(path)
    except Exception:
        Path(tmp).unlink(missing_ok=True)
        raise


def _backup_metadata(meta_path: Path) -> None:
    """Create metadata.json.bak the first time we touch a file. Idempotent."""
    backup = meta_path.with_suffix(".json.bak")
    if not backup.exists():
        shutil.copy2(meta_path, backup)


def _load_metadata_blob(meta_path: Path) -> Optional[Dict[str, Any]]:
    if not meta_path.exists():
        return None
    with meta_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _sanitize_misplaced_fields(meta_path: Path) -> bool:
    """
    Move any fields a previous run wrongly placed at the top level of
    ``metadata`` into ``custom_metadata`` and rewrite the file. Returns
    ``True`` if anything was changed.

    This is a healer for the bug in the very first stability-write
    version of this script, which put ``rotation_peak_prominence`` etc.
    at the top level. ``RecordingMetadata.__init__`` doesn't accept
    those as known fields, which means every subsequent
    ``RecordingSession.load()`` call would crash with an
    ``unexpected keyword argument`` error. Running this fixes that.
    """
    blob = _load_metadata_blob(meta_path)
    if blob is None:
        return False
    inner = blob.get("metadata") if isinstance(blob.get("metadata"), dict) else blob

    misplaced = {k: inner[k] for k in MISPLACED_TOP_LEVEL_FIELDS if k in inner}
    if not misplaced:
        return False

    custom = inner.get("custom_metadata")
    if not isinstance(custom, dict):
        custom = {}
        inner["custom_metadata"] = custom
    for k, v in misplaced.items():
        # Prefer the top-level value — it's the most recent one we wrote.
        custom[k] = v
        inner.pop(k, None)

    _backup_metadata(meta_path)
    _atomic_write_json(meta_path, blob)
    return True


# ---------------------------------------------------------------------------
# Per-session update
# ---------------------------------------------------------------------------

def update_session(
    session_dir: Path,
    bank: ReferenceBank,
    *,
    bootstrap_n: int = 200,
    update_offset: bool = False,
    dry_run: bool = False,
) -> Tuple[str, Optional[StabilityResult]]:
    """
    Recompute the stability metric for one session and update its
    ``metadata.json`` in place.

    Returns ``(status, result)`` — status is one of
    ``"updated" | "skipped:<reason>" | "error:<reason>"``.
    """
    meta_path = session_dir / "metadata.json"
    blob = _load_metadata_blob(meta_path)
    if blob is None:
        return "skipped:no metadata", None

    # FIRST: heal any misplaced fields from previous buggy runs. Without
    # this, RecordingSession.load() crashes on files we wrote ourselves.
    healed = False
    if not dry_run:
        healed = _sanitize_misplaced_fields(meta_path)
        if healed:
            # Re-read the now-fixed blob
            blob = _load_metadata_blob(meta_path)

    try:
        session = RecordingSession.load(session_dir)
    except Exception as exc:  # noqa: BLE001
        log.debug("RecordingSession.load failed for %s: %s", session_dir, exc)
        return f"error:load failed ({exc})", None

    signal = _signal_from_session(session)
    if signal is None or signal.size == 0:
        return "skipped:no signal", None

    sync_gesture, sync_chunks = _select_sync_trials(session, signal)
    if not sync_chunks:
        return "skipped:no sync trials", None

    signal_mode = _session_signal_mode(session)
    ref_energy, ref_key, reason = bank.lookup(signal_mode, sync_gesture)
    if ref_energy is None:
        return f"skipped:{reason or 'no reference pattern'} (mode={signal_mode})", None

    n_channels = int(getattr(session.metadata, "num_channels",
                             signal.shape[1] if signal.ndim == 2 else 0))
    if ref_energy.size != n_channels:
        return (f"skipped:reference ch={ref_energy.size} != session ch={n_channels}",
                None)

    sampling_rate = float(getattr(session.metadata, "sampling_rate", 2000.0)
                          or 2000.0)

    try:
        result = compute_stability_metrics(
            sync_chunks, ref_energy,
            num_channels=n_channels,
            sampling_rate=sampling_rate,
            bootstrap_n=bootstrap_n,
        )
    except Exception as exc:  # noqa: BLE001
        return f"error:stability failed ({exc})", None

    inner = blob.get("metadata") if isinstance(blob.get("metadata"), dict) else blob
    legacy_conf   = inner.get("rotation_confidence")
    legacy_offset = inner.get("rotation_offset")

    # Make sure custom_metadata exists — it's the right place for all
    # the new diagnostic fields. RecordingMetadata accepts it.
    custom = inner.get("custom_metadata")
    if not isinstance(custom, dict):
        custom = {}
        inner["custom_metadata"] = custom

    # Preserve the legacy peak prominence under custom_metadata the
    # first time we touch this file. Idempotent on subsequent runs.
    if "rotation_peak_prominence" not in custom and legacy_conf is not None:
        # Only preserve if this looks like the legacy z-score number,
        # not the new stability we wrote ourselves earlier.
        if custom.get("rotation_metric_version") != 2:
            try:
                custom["rotation_peak_prominence_legacy"] = float(legacy_conf)
            except (TypeError, ValueError):
                pass

    # The first-class fields that the rest of the codebase reads.
    # ``rotation_confidence`` is a known field of RecordingMetadata, so
    # it stays at top level. Everything new goes under custom_metadata.
    inner["rotation_confidence"] = float(result.stability)

    custom["rotation_peak_prominence"]   = float(result.peak_prominence)
    custom["rotation_metric_version"]    = 2
    custom["rotation_stability_metrics"] = {
        "stability":           float(result.stability),
        "peak_prominence":     float(result.peak_prominence),
        "top2_ratio":          float(result.top2_ratio)
                               if np.isfinite(result.top2_ratio) else None,
        "n_trials_used":       int(result.n_trials_used),
        "per_trial_offsets":   list(result.per_trial_offsets),
        "bootstrap_offsets":   list(result.bootstrap_offsets),
        "sync_gesture":        sync_gesture,
        "ref_pattern_key":     ref_key,
        "signal_mode":         signal_mode,
    }
    if update_offset:
        inner["rotation_offset"] = int(result.offset)

    if dry_run:
        log.info(
            "  [dry-run] %s  trials=%d  stability=%.2f  "
            "prominence=%.2f  (legacy_offset=%s, new_offset=%d, sync=%s)",
            session_dir.name, result.n_trials_used,
            result.stability, result.peak_prominence,
            legacy_offset, result.offset, sync_gesture,
        )
    else:
        _backup_metadata(meta_path)
        _atomic_write_json(meta_path, blob)
        old_conf = (f"{float(legacy_conf):.2f}"
                    if legacy_conf is not None else "—")
        log.info(
            "  %s  → stability=%.2f  (n=%d trials of %s; was %s%s)",
            session_dir.name, result.stability,
            result.n_trials_used, sync_gesture, old_conf,
            " [healed]" if healed else "",
        )
    return "updated", result


# ---------------------------------------------------------------------------
# Corpus walk
# ---------------------------------------------------------------------------

def walk_corpus(
    data_dir: Path, reference_arg: Path,
    *,
    subjects: Optional[Sequence[str]] = None,
    bootstrap_n: int = 200,
    update_offset: bool = False,
    dry_run: bool = False,
) -> Dict[str, StabilityResult]:
    """Update every session under ``<data_dir>/sessions/``."""
    sessions_root = data_dir / "sessions"
    if not sessions_root.exists():
        raise FileNotFoundError(f"No sessions directory at {sessions_root}")

    # ── Load reference patterns ONCE, up-front, with clear diagnostics ─
    log.info("=" * 70)
    log.info("Reference resolution")
    log.info("  --reference argument:  %s", reference_arg)
    log.info("  absolute path:         %s", Path(reference_arg).resolve())
    log.info("  exists as file:        %s", Path(reference_arg).is_file())
    log.info("  exists as directory:   %s", Path(reference_arg).is_dir())
    bank = ReferenceBank()
    # Always try monopolar (legacy default). Also try bipolar if we can
    # find a separate file for it — sessions with bipolar mode then
    # automatically get the right pattern.
    bank.load_from(reference_arg, "monopolar")
    bank.load_from(reference_arg, "bipolar")
    if not bank.modes:
        log.error(
            "No usable reference patterns loaded. Aborting. "
            "Pass an absolute path to your reference JSON, or a "
            "directory that contains "
            "reference_calibration[_<mode>].json files."
        )
        return {}
    log.info("=" * 70)

    subj_set = set(subjects) if subjects else None
    out: Dict[str, StabilityResult] = {}
    n_seen = 0
    n_updated = 0
    skip_counts: Dict[str, int] = {}

    for meta_path in sorted(sessions_root.rglob("metadata.json")):
        session_dir = meta_path.parent
        rel = session_dir.relative_to(sessions_root)
        parts = rel.parts
        if not parts:
            continue
        if parts[0] == "unity_sessions":
            subject_guess = parts[1] if len(parts) > 1 else "unity"
        else:
            subject_guess = parts[0]
        if subj_set is not None and subject_guess not in subj_set:
            continue
        n_seen += 1

        status, result = update_session(
            session_dir, bank,
            bootstrap_n=bootstrap_n,
            update_offset=update_offset,
            dry_run=dry_run,
        )
        if status == "updated" and result is not None:
            out[f"{subject_guess}/{session_dir.name}"] = result
            n_updated += 1
        else:
            reason = status.split(":", 1)[1] if ":" in status else status
            skip_counts[reason] = skip_counts.get(reason, 0) + 1
            log.debug("  %s: %s", session_dir.name, status)

    log.info("-" * 70)
    log.info("Processed %d/%d sessions  (%s).",
             n_updated, n_seen, "dry-run" if dry_run else "written")
    if skip_counts:
        log.info("Skip reasons:")
        for reason, count in sorted(skip_counts.items(), key=lambda kv: -kv[1]):
            log.info("  %4d × %s", count, reason)
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(
        prog="recompute_calibration_metrics",
        description=__doc__.splitlines()[1],
    )
    p.add_argument("--data-dir", type=Path, required=True,
                   help="Pipeline data root (contains sessions/).")
    p.add_argument("--reference", type=Path, required=True,
                   help=("Reference calibration JSON. May be a file (used "
                         "directly) or a directory containing "
                         "reference_calibration[_<mode>].json files."))
    p.add_argument("--subjects", nargs="*", default=None,
                   help="Limit to these subject IDs. Default: all.")
    p.add_argument("--bootstrap", type=int, default=200,
                   help="Bootstrap sample count for diagnostic plots "
                        "(0 disables).")
    p.add_argument("--update-offset", action="store_true",
                   help="Also overwrite rotation_offset with the recomputed "
                        "mode (default: leave the stored offset alone so "
                        "downstream channel mappings keep working).")
    p.add_argument("--dry-run", action="store_true",
                   help="Print what would change but don't touch any file.")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
    )

    walk_corpus(
        args.data_dir, args.reference,
        subjects=args.subjects,
        bootstrap_n=args.bootstrap,
        update_offset=args.update_offset,
        dry_run=args.dry_run,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())