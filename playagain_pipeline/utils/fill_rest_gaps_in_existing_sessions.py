"""
fill_rest_gaps_in_existing_sessions.py
──────────────────────────────────────
One-off CLI that retroactively patches already-saved training-game sessions
by inserting ``rest`` trials into the unlabelled gaps between consecutive
gesture trials.

Use this when you have data on disk from an earlier training-game run and
need it labelled correctly *without* re-recording.

Behaviour
─────────
* Operates on each session directory's ``metadata.json`` only. The raw
  ``data.npy`` and ``data.csv`` files are never modified.
* Writes a backup of the original metadata to
  ``metadata.json.bak`` next to the file the first time it patches a
  session. Existing backups are never overwritten (so you never lose the
  true original by re-running this script).
* Idempotent — running it twice on the same session produces the same
  result. The second run reports ``0`` insertions.
* Conservative — only fills gaps strictly between two gesture trials
  (``trial_type == "gesture"``). Calibration-sync trials are left alone.
* Dry-run by default. Pass ``--apply`` to write changes.

Examples
────────
    # See what would change for one subject, no writes:
    python fill_rest_gaps_in_existing_sessions.py \\
        /Users/paul/Coding_Projects/Master/Dataprocessing/playagain_pipeline/data/sessions/VP_01

    # Actually apply, recursing into all subjects:
    python fill_rest_gaps_in_existing_sessions.py \\
        /Users/paul/Coding_Projects/Master/Dataprocessing/playagain_pipeline/data/sessions \\
        --apply

    # Only patch sessions whose protocol is the training game:
    python fill_rest_gaps_in_existing_sessions.py \\
        .../data/sessions --apply --only-protocol training_game

The script is dependency-free (stdlib only) so it can run with the same
Python you use for the GUI without activating any venv.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

log = logging.getLogger("fill_rest_gaps")

INSERTED_NOTE = "auto-inserted rest (fills inter-trial gap)"
DEFAULT_REST_NAME = "rest"
BACKUP_SUFFIX = ".bak"


# ---------------------------------------------------------------------------
# Core: gap-filling on a JSON dict (no dependency on the playagain package)
# ---------------------------------------------------------------------------

@dataclass
class FillResult:
    session_dir: Path
    inserted: int
    skipped_reason: Optional[str] = None  # set when no patch was attempted

    @property
    def patched(self) -> bool:
        return self.inserted > 0 and self.skipped_reason is None


def _resolve_rest_label(gesture_set_path: Path, rest_name: str) -> Optional[int]:
    """
    Look up the rest gesture's ``label_id`` from ``gesture_set.json``.
    Returns ``None`` if the file is missing or the gesture is absent.
    """
    if not gesture_set_path.exists():
        return None
    try:
        with open(gesture_set_path, "r", encoding="utf-8") as f:
            gset = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("Could not read %s: %s", gesture_set_path, exc)
        return None

    for g in gset.get("gestures", []):
        if g.get("name") == rest_name:
            label = g.get("label_id")
            if isinstance(label, int):
                return label
    return None


def _fill_trials(trials: List[dict], rest_label: int, sampling_rate: int,
                 min_gap_samples: int) -> List[dict]:
    """
    Pure-data version of fill_rest_gaps. Operates on a list of trial
    dicts (as serialised in metadata.json) and returns a NEW list with
    rest trials inserted and trial_ids renumbered contiguously.
    """
    if not trials:
        return trials

    ordered = sorted(trials, key=lambda t: t["start_sample"])
    out: List[dict] = []

    for i, t in enumerate(ordered):
        out.append(t)
        if i + 1 >= len(ordered):
            continue
        nxt = ordered[i + 1]

        # Conservative: skip anything adjacent to calibration-sync.
        if t.get("trial_type", "gesture") != "gesture":
            continue
        if nxt.get("trial_type", "gesture") != "gesture":
            continue

        gap = nxt["start_sample"] - t["end_sample"]
        if gap < min_gap_samples:
            continue

        rest_trial = {
            "trial_id": -1,  # renumbered below
            "gesture_name": DEFAULT_REST_NAME,
            "gesture_label": rest_label,
            "start_sample": t["end_sample"],
            "end_sample": nxt["start_sample"],
            "start_time": t["end_sample"] / sampling_rate,
            "end_time": nxt["start_sample"] / sampling_rate,
            "is_valid": True,
            "notes": INSERTED_NOTE,
            "trial_type": "gesture",
        }
        out.append(rest_trial)

    # Contiguous renumber.
    for new_id, t in enumerate(out):
        t["trial_id"] = new_id
    return out


# ---------------------------------------------------------------------------
# Per-session worker
# ---------------------------------------------------------------------------

def patch_session_dir(
    session_dir: Path,
    *,
    rest_name: str = DEFAULT_REST_NAME,
    min_gap_samples: int = 1,
    only_protocol: Optional[str] = None,
    apply: bool = False,
) -> FillResult:
    """
    Patch a single session directory. Returns a ``FillResult``.
    """
    meta_path = session_dir / "metadata.json"
    gset_path = session_dir / "gesture_set.json"

    if not meta_path.exists():
        return FillResult(session_dir, 0, skipped_reason="no metadata.json")

    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        return FillResult(session_dir, 0, skipped_reason=f"unreadable: {exc}")

    md = meta.get("metadata", {})
    trials = meta.get("trials", [])

    # Optional protocol filter — useful for "only patch training_game".
    if only_protocol is not None and md.get("protocol_name") != only_protocol:
        return FillResult(
            session_dir, 0,
            skipped_reason=f"protocol={md.get('protocol_name')!r} != {only_protocol!r}",
        )

    sr = md.get("sampling_rate", 0)
    if not isinstance(sr, int) or sr <= 0:
        return FillResult(session_dir, 0, skipped_reason=f"bad sampling_rate={sr!r}")

    rest_label = _resolve_rest_label(gset_path, rest_name)
    if rest_label is None:
        return FillResult(
            session_dir, 0,
            skipped_reason=f"no '{rest_name}' gesture in gesture_set.json",
        )

    new_trials = _fill_trials(trials, rest_label, sr, min_gap_samples)
    inserted = len(new_trials) - len(trials)

    if inserted <= 0:
        return FillResult(session_dir, 0)

    if not apply:
        return FillResult(session_dir, inserted)

    # Backup once. Subsequent re-runs never overwrite the first backup.
    backup_path = meta_path.with_suffix(meta_path.suffix + BACKUP_SUFFIX)
    if not backup_path.exists():
        shutil.copy2(meta_path, backup_path)

    meta["trials"] = new_trials
    # Atomic-ish write: dump to a temp file then rename.
    tmp_path = meta_path.with_suffix(meta_path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    tmp_path.replace(meta_path)

    return FillResult(session_dir, inserted)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def iter_session_dirs(root: Path) -> Iterable[Path]:
    """
    Yield every directory under ``root`` that contains both a
    ``metadata.json`` and a ``data.npy``. Works whether ``root`` is a
    single session, a subject directory, or the top-level sessions
    folder.
    """
    if (root / "metadata.json").exists() and (root / "data.npy").exists():
        yield root
        return

    for meta in sorted(root.rglob("metadata.json")):
        sess_dir = meta.parent
        if (sess_dir / "data.npy").exists():
            yield sess_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Insert 'rest' trials into the gaps between gesture "
                    "trials of already-recorded sessions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("path", type=Path,
                   help="Session dir, subject dir, or top-level sessions dir.")
    p.add_argument("--apply", action="store_true",
                   help="Actually write changes. Without this flag, the "
                        "script only previews what would happen.")
    p.add_argument("--rest-name", default=DEFAULT_REST_NAME,
                   help="Name of the rest gesture in gesture_set.json "
                        f"(default: {DEFAULT_REST_NAME!r}).")
    p.add_argument("--min-gap-samples", type=int, default=1,
                   help="Minimum gap, in samples, that qualifies for "
                        "filling (default: 1).")
    p.add_argument("--only-protocol", default=None,
                   help="Only patch sessions whose protocol_name equals "
                        "this value (e.g. 'training_game').")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="Verbose logging.")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )

    root = args.path.expanduser().resolve()
    if not root.exists():
        log.error("Path does not exist: %s", root)
        return 2

    session_dirs = list(iter_session_dirs(root))
    if not session_dirs:
        log.error("No sessions found under %s", root)
        return 1

    mode = "APPLY" if args.apply else "DRY-RUN"
    log.info("[%s] Inspecting %d session(s) under %s", mode, len(session_dirs), root)

    total_inserted = 0
    patched_count = 0
    skipped_count = 0

    for sess in session_dirs:
        result = patch_session_dir(
            sess,
            rest_name=args.rest_name,
            min_gap_samples=args.min_gap_samples,
            only_protocol=args.only_protocol,
            apply=args.apply,
        )
        rel = sess.relative_to(root) if sess != root else sess.name
        if result.skipped_reason is not None:
            log.debug("  skipped %s: %s", rel, result.skipped_reason)
            skipped_count += 1
        elif result.inserted == 0:
            log.info("  ok  %s: already contiguous (0 inserted)", rel)
        else:
            verb = "would insert" if not args.apply else "inserted"
            log.info("  +   %s: %s %d rest trial(s)", rel, verb, result.inserted)
            patched_count += 1
            total_inserted += result.inserted

    log.info("")
    log.info("Summary: %d session(s) %s, %d skipped, %d rest trial(s) %s",
             patched_count,
             "patched" if args.apply else "would be patched",
             skipped_count,
             total_inserted,
             "inserted" if args.apply else "would be inserted")
    if not args.apply and total_inserted > 0:
        log.info("Re-run with --apply to write the changes "
                 "(originals are backed up to metadata.json.bak).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
