"""
playagain_pipeline.utils.migrate_requested_gesture
══════════════════════════════════════════════════
One-shot migration that updates the ``RequestedGesture`` column inside
existing game recordings.

Older recordings (written before the GameRecorder change of May 2026)
used the literal string ``"none"`` to mark frames where Unity was not
asking the user for any specific gesture. The user is at rest during
those intervals, so the new convention is to write ``"rest"`` instead.
That makes the column directly usable as a multi-class ground-truth
label without an extra "drop the none rows" step downstream.

This script walks one or more directories looking for game-recording
CSV files (the GameRecorder layout is
``<root>/<subject>/<game_session_dir>/recording.csv`` with a sibling
``config.json``; legacy flat layouts are also handled), and rewrites
the ``RequestedGesture`` column of each one in place.

It is **streaming** — CSVs are read row by row and written to a sibling
temp file, then atomically swapped in. This means it works fine on
multi-hundred-megabyte recordings without ever loading them into RAM.

It is **idempotent** — re-running it only touches files that still have
old-style values to replace.

It is **safe** — by default a backup ``recording.csv.bak`` is dropped
next to each modified file before overwriting, and you can revert with
``--revert``.

Usage
─────
    # Dry run, see what would change
    python -m playagain_pipeline.utils.migrate_requested_gesture \\
        path/to/data --dry-run

    # Actually rewrite (creates .bak files)
    python -m playagain_pipeline.utils.migrate_requested_gesture \\
        path/to/data

    # Roll everything back
    python -m playagain_pipeline.utils.migrate_requested_gesture \\
        path/to/data --revert

    # Custom mapping (e.g. "idle" → "rest")
    python -m playagain_pipeline.utils.migrate_requested_gesture \\
        path/to/data --from idle --to rest

The script auto-detects ``RequestedGesture`` as the target column by
header name. If the column is missing (e.g. a foreign CSV with a
similar name), the file is skipped and reported.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence


# ──────────────────────────────────────────────────────────────────────
# Discovery
# ──────────────────────────────────────────────────────────────────────

def find_recording_csvs(root: Path) -> List[Path]:
    """
    Walk ``root`` and return every game-recording CSV.

    Three layouts are accepted (in priority order):

    1. ``<root>/<subject>/<game_session_dir>/recording.csv``  — current
       GameRecorder layout. The session dir typically contains a sibling
       ``config.json``; we use that as the existence test, which lets us
       skip arbitrary ``.csv`` files that just happen to live nearby.

    2. ``<root>/<subject>/<file>.csv``                         — older
       flat dump where each subject folder has one CSV per session.

    3. ``<root>/<file>.csv``                                   — loose
       CSVs at the root.

    If ``root`` itself points at a single CSV, that one file is returned.
    Duplicates are filtered.
    """
    root = Path(root)
    if root.is_file() and root.suffix.lower() == ".csv":
        return [root]
    if not root.exists():
        return []

    found: List[Path] = []
    seen: set = set()

    def _add(p: Path) -> None:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp); found.append(p)

    # Layout 1: nested with sibling config.json
    for csv_path in sorted(root.rglob("*.csv")):
        if (csv_path.parent / "config.json").exists():
            _add(csv_path)

    # Layout 2 & 3: any CSV under root (already-found ones are deduped)
    for csv_path in sorted(root.rglob("*.csv")):
        _add(csv_path)

    return found


# ──────────────────────────────────────────────────────────────────────
# Per-file migration
# ──────────────────────────────────────────────────────────────────────

@dataclass
class MigrationStats:
    path: Path
    column: str = "RequestedGesture"
    rows: int = 0
    changed: int = 0
    skipped_reason: Optional[str] = None
    dry_run: bool = False
    backup_path: Optional[Path] = None
    reverted: bool = False

    @property
    def changed_anything(self) -> bool:
        return self.changed > 0 and not self.dry_run


def migrate_one(
    csv_path: Path,
    *,
    from_value: str,
    to_value: str,
    column: str = "RequestedGesture",
    dry_run: bool = False,
    make_backup: bool = True,
) -> MigrationStats:
    """
    Stream-rewrite ``csv_path`` replacing ``column == from_value`` with
    ``to_value``. Returns a populated :class:`MigrationStats`.

    The rewrite is atomic: data is written to a temporary file in the
    same directory, fsync'd, and ``os.replace``-d into place. If the
    target column isn't present in the header, the file is skipped and
    ``skipped_reason`` is set.
    """
    stats = MigrationStats(path=csv_path, column=column, dry_run=dry_run)

    if not csv_path.exists():
        stats.skipped_reason = "file does not exist"
        return stats

    # Open with newline="" so the csv module handles line endings
    # consistently across platforms (Windows recordings included).
    try:
        f_in = open(csv_path, "r", newline="", encoding="utf-8")
    except OSError as exc:
        stats.skipped_reason = f"cannot read: {exc}"
        return stats

    with f_in:
        reader = csv.reader(f_in)
        try:
            header = next(reader)
        except StopIteration:
            stats.skipped_reason = "empty file"
            return stats

        if column not in header:
            stats.skipped_reason = f"missing column '{column}'"
            return stats
        col_idx = header.index(column)

        # Dry-run path: just count, don't write
        if dry_run:
            for row in reader:
                stats.rows += 1
                if col_idx < len(row) and row[col_idx] == from_value:
                    stats.changed += 1
            return stats

        # Streaming write to a sibling temp file, then atomic rename
        tmp_fd, tmp_name = tempfile.mkstemp(
            prefix=csv_path.stem + ".",
            suffix=".tmp",
            dir=str(csv_path.parent),
        )
        try:
            with os.fdopen(tmp_fd, "w", newline="", encoding="utf-8") as f_out:
                writer = csv.writer(f_out)
                writer.writerow(header)
                for row in reader:
                    stats.rows += 1
                    if col_idx < len(row) and row[col_idx] == from_value:
                        row[col_idx] = to_value
                        stats.changed += 1
                    writer.writerow(row)
                f_out.flush()
                # Force durability before swapping in
                os.fsync(f_out.fileno())
        except Exception:
            # Clean up partial temp file on any error before rethrowing
            try: os.unlink(tmp_name)
            except OSError: pass
            raise

    # No-op fast path — leave the original file untouched.
    if stats.changed == 0:
        try: os.unlink(tmp_name)
        except OSError: pass
        return stats

    # Make a backup before overwrite, unless suppressed
    if make_backup:
        bak = csv_path.with_suffix(csv_path.suffix + ".bak")
        # Don't clobber an existing backup — preserve the *original*
        # pre-migration version if the user runs the script twice for
        # whatever reason.
        if not bak.exists():
            shutil.copy2(csv_path, bak)
        stats.backup_path = bak

    # Atomic swap (POSIX + Windows: os.replace overwrites)
    os.replace(tmp_name, csv_path)
    return stats


def revert_one(csv_path: Path) -> MigrationStats:
    """Restore ``csv_path`` from its sibling ``.bak`` if one exists."""
    stats = MigrationStats(path=csv_path)
    bak = csv_path.with_suffix(csv_path.suffix + ".bak")
    if not bak.exists():
        stats.skipped_reason = "no .bak file"
        return stats
    # We read a single line to count nothing meaningful; revert is a swap.
    shutil.copy2(bak, csv_path)
    stats.reverted = True
    stats.backup_path = bak
    return stats


# ──────────────────────────────────────────────────────────────────────
# Bulk runner
# ──────────────────────────────────────────────────────────────────────

def run(
    paths: Sequence[Path],
    *,
    from_value: str = "none",
    to_value: str = "rest",
    column: str = "RequestedGesture",
    dry_run: bool = False,
    make_backup: bool = True,
    revert: bool = False,
) -> List[MigrationStats]:
    """Discover all CSVs under ``paths`` and migrate (or revert) each one."""
    all_csvs: List[Path] = []
    for p in paths:
        all_csvs.extend(find_recording_csvs(Path(p)))

    results: List[MigrationStats] = []
    for csv_path in all_csvs:
        if revert:
            results.append(revert_one(csv_path))
        else:
            results.append(migrate_one(
                csv_path,
                from_value=from_value, to_value=to_value, column=column,
                dry_run=dry_run, make_backup=make_backup,
            ))
    return results


def print_summary(results: Iterable[MigrationStats], *, dry_run: bool, revert: bool) -> None:
    """Pretty-print a one-line-per-file summary."""
    total_files = total_rows = total_changed = 0
    skipped: List[MigrationStats] = []
    touched: List[MigrationStats] = []
    untouched: List[MigrationStats] = []

    for r in results:
        total_files += 1
        if r.skipped_reason:
            skipped.append(r); continue
        total_rows += r.rows
        total_changed += r.changed
        if r.reverted or r.changed > 0:
            touched.append(r)
        else:
            untouched.append(r)

    verb = "WOULD CHANGE" if dry_run else ("REVERTED" if revert else "CHANGED")
    print()
    print(f"  {'Path':<70}  {'Rows':>10}  {verb:>15}")
    print(f"  {'-'*70}  {'-'*10}  {'-'*15}")
    for r in touched:
        if revert:
            tag = "from .bak"
        else:
            tag = f"{r.changed:,}"
        print(f"  {_short(r.path):<70}  {r.rows:>10,}  {tag:>15}")
    for r in untouched:
        print(f"  {_short(r.path):<70}  {r.rows:>10,}  {'(no change)':>15}")
    for r in skipped:
        print(f"  {_short(r.path):<70}  {'-':>10}  skipped: {r.skipped_reason}")

    print()
    if revert:
        n_rev = sum(1 for r in results if r.reverted)
        print(f"Summary: reverted {n_rev} file(s) from .bak  ({total_files} scanned)")
    elif dry_run:
        n_would = sum(1 for r in results if r.changed > 0)
        print(f"Summary: would change {total_changed:,} cell(s) "
              f"across {n_would} file(s)  ({total_files} scanned)")
    else:
        n_changed = sum(1 for r in results if r.changed > 0)
        print(f"Summary: changed {total_changed:,} cell(s) "
              f"across {n_changed} file(s)  ({total_files} scanned)")


def _short(p: Path) -> str:
    """Shorten a path to its last three components for display."""
    parts = p.parts
    if len(parts) <= 4:
        return str(p)
    return os.path.join("…", *parts[-3:])


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m playagain_pipeline.utils.migrate_requested_gesture",
        description=(
            "Rewrite the RequestedGesture column of existing game recordings "
            "(default: replace 'none' with 'rest'). Streams CSVs so it works "
            "on huge recordings; makes .bak backups and supports --revert."
        ),
    )
    p.add_argument(
        "paths", nargs="+", type=Path,
        help="One or more directories (or single CSV files) to migrate. "
             "Typically a project's `data/` directory, or just "
             "`data/game_recordings/`.",
    )
    p.add_argument("--from", dest="from_value", default="none",
                   help="Value to replace (default: 'none').")
    p.add_argument("--to",   dest="to_value",   default="rest",
                   help="Replacement value (default: 'rest').")
    p.add_argument("--column", default="RequestedGesture",
                   help="Column to operate on (default: 'RequestedGesture').")
    p.add_argument("--dry-run", action="store_true",
                   help="Just report what would change, don't write anything.")
    p.add_argument("--no-backup", action="store_true",
                   help="Skip writing .bak files before overwriting. "
                        "Faster but irreversible — use with care.")
    p.add_argument("--revert", action="store_true",
                   help="Restore each recording.csv from its sibling .bak file. "
                        "All other options are ignored when this is set.")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.dry_run and args.revert:
        print("error: --dry-run and --revert are mutually exclusive",
              file=sys.stderr)
        return 2

    results = run(
        args.paths,
        from_value=args.from_value, to_value=args.to_value,
        column=args.column,
        dry_run=args.dry_run,
        make_backup=not args.no_backup,
        revert=args.revert,
    )
    print_summary(results, dry_run=args.dry_run, revert=args.revert)
    return 0


if __name__ == "__main__":
    sys.exit(main())