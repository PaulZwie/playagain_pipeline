#!/usr/bin/env python3
"""
Remove Invalid Recordings
=========================
Reads quality_review.json produced by check_recording_quality.py and
permanently deletes every recording folder marked as INVALID.

Usage
-----
    python remove_invalid_recordings.py /path/to/data
    python remove_invalid_recordings.py /path/to/data --dry-run   # preview only
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

REVIEW_FILE = "quality_review.json"


def main():
    parser = argparse.ArgumentParser(
        description="Delete recording folders marked INVALID in quality_review.json."
    )
    parser.add_argument("data_dir", type=Path,
                        help="Root data directory (same one used for the quality check)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be deleted without actually deleting anything")
    args = parser.parse_args()

    data_dir = args.data_dir.expanduser().resolve()
    review_path = data_dir / REVIEW_FILE

    if not review_path.exists():
        print(f"No {REVIEW_FILE} found in {data_dir}. Run check_recording_quality.py first.")
        sys.exit(1)

    with open(review_path) as f:
        reviews = json.load(f)

    invalid = {key: rev for key, rev in reviews.items() if rev.get("status") == "INVALID"}

    if not invalid:
        print("No recordings marked INVALID — nothing to do.")
        return

    # Resolve each key  →  folder path
    # Key format:  "<type>:<subject_id>:<session_id>"
    # type is "session" or "game"
    to_delete: list[tuple[str, Path]] = []
    not_found: list[str] = []

    for key in invalid:
        parts = key.split(":", 2)
        if len(parts) != 3:
            not_found.append(key)
            continue
        rec_type, subject_id, session_id = parts

        if rec_type == "session":
            # Handle unity_sessions nesting: subject stored as "VP_xx/unity"
            if subject_id.endswith("/unity"):
                base_subject = subject_id[: -len("/unity")]
                candidate = data_dir / "sessions" / "unity_sessions" / base_subject / session_id
            else:
                candidate = data_dir / "sessions" / subject_id / session_id
        elif rec_type == "game":
            candidate = data_dir / "game_recordings" / subject_id / session_id
        else:
            not_found.append(key)
            continue

        if candidate.exists():
            to_delete.append((key, candidate))
        else:
            not_found.append(key)

    # ── Summary ──────────────────────────────────────────────────────────
    prefix = "[DRY-RUN] " if args.dry_run else ""

    print(f"\n{prefix}Recordings marked INVALID: {len(invalid)}")
    print(f"  Found on disk : {len(to_delete)}")
    print(f"  Not found     : {len(not_found)}")

    if not_found:
        print("\n  Could not locate (already deleted or path mismatch):")
        for key in not_found:
            print(f"    {key}")

    if not to_delete:
        print("\nNothing to delete.")
        return

    print(f"\n{'Would delete' if args.dry_run else 'Will delete'} {len(to_delete)} folder(s):\n")
    for key, path in to_delete:
        comment = invalid[key].get("comment", "")
        note = f"  # {comment}" if comment else ""
        print(f"  {path}{note}")

    if args.dry_run:
        print("\n[DRY-RUN] No files were deleted. Remove --dry-run to proceed.")
        return

    # ── Confirmation ─────────────────────────────────────────────────────
    print(f"\n⚠  This will permanently delete {len(to_delete)} folder(s).")
    answer = input("Type 'yes' to confirm: ").strip().lower()
    if answer != "yes":
        print("Aborted.")
        return

    # ── Delete ────────────────────────────────────────────────────────────
    deleted, failed = 0, 0
    for key, path in to_delete:
        try:
            shutil.rmtree(path)
            print(f"  ✓ Deleted  {path}")
            deleted += 1
        except Exception as e:
            print(f"  ✗ Failed   {path}  ({e})")
            failed += 1

    # ── Update review file (remove deleted entries) ───────────────────────
    deleted_keys = {key for key, path in to_delete}
    updated = {k: v for k, v in reviews.items() if k not in deleted_keys}
    with open(review_path, "w") as f:
        json.dump(updated, f, indent=2)

    print(f"\nDone — deleted {deleted}, failed {failed}.")
    print(f"Updated {REVIEW_FILE} ({len(deleted_keys)} entries removed).")


if __name__ == "__main__":
    main()
