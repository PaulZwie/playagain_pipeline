#!/usr/bin/env python3
"""
Recording Quality Checker
=========================
Scans all sessions/ and game_recordings/ under a data root and produces
an interactive quality-review terminal UI.

Usage
-----
    python check_recording_quality.py /path/to/data

Controls (interactive mode)
---------------------------
    n / p          next / previous recording
    v              mark current as VALID (good)
    i              mark current as INVALID (bad)
    q              flag as QUESTIONABLE
    c              add / edit a comment for the current recording
    s              save review state to  quality_review.json
    e              export summary CSV to  quality_review_summary.csv
    x              exit

The script can also be run in --report mode to print a plain summary
without the interactive reviewer:

    python check_recording_quality.py /path/to/data --report
"""

import argparse
import json
import csv
import os
import sys
import math
import textwrap
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple


# ──────────────────────────────────────────────────────────────────────────────
# Colour helpers (graceful fallback when terminal has no colour support)
# ──────────────────────────────────────────────────────────────────────────────

def _supports_colour() -> bool:
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()

_COLOUR = _supports_colour()

def _c(code: str, text: str) -> str:
    if not _COLOUR:
        return text
    return f"\033[{code}m{text}\033[0m"

RED    = lambda t: _c("91", t)
GREEN  = lambda t: _c("92", t)
YELLOW = lambda t: _c("93", t)
CYAN   = lambda t: _c("96", t)
BOLD   = lambda t: _c("1",  t)
DIM    = lambda t: _c("2",  t)


# ──────────────────────────────────────────────────────────────────────────────
# Data loading helpers
# ──────────────────────────────────────────────────────────────────────────────

def _load_json(path: Path) -> Optional[Dict]:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def _load_csv_head(path: Path, n_rows: int = 500) -> Optional[List[List[str]]]:
    """Read the first n_rows of a CSV file without loading the whole file."""
    try:
        rows = []
        with open(path, newline="") as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                if i >= n_rows + 1:   # +1 for header
                    break
                rows.append(row)
        return rows
    except Exception:
        return None


def _csv_row_count(path: Path) -> int:
    """Count data rows in a CSV (excluding header) efficiently."""
    try:
        count = 0
        with open(path, newline="") as f:
            reader = csv.reader(f)
            next(reader, None)   # skip header
            for _ in reader:
                count += 1
        return count
    except Exception:
        return -1


# ──────────────────────────────────────────────────────────────────────────────
# Quality metrics
# ──────────────────────────────────────────────────────────────────────────────

def _emg_stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {}
    n = len(values)
    mean = sum(values) / n
    var  = sum((v - mean) ** 2 for v in values) / n
    std  = math.sqrt(var)
    rms  = math.sqrt(sum(v ** 2 for v in values) / n)
    mn   = min(values)
    mx   = max(values)
    return {"mean": mean, "std": std, "rms": rms, "min": mn, "max": mx, "n": n}


def _fraction_zeros(values: List[float]) -> float:
    if not values:
        return 0.0
    return sum(1 for v in values if v == 0.0) / len(values)


def _fraction_saturated(values: List[float], threshold: float = 4.0) -> float:
    if not values:
        return 0.0
    return sum(1 for v in values if abs(v) >= threshold) / len(values)


# ──────────────────────────────────────────────────────────────────────────────
# Session analyser  (sessions/VP_xx/session_name/)
# ──────────────────────────────────────────────────────────────────────────────

class SessionQuality:
    """Quality report for one training session."""

    def __init__(self, path: Path):
        self.path            = path
        self.subject_id      = path.parent.name   # may be "unity_sessions" → handled below
        self.session_id      = path.name
        self.recording_type  = "session"

        # Adjust for unity_sessions nesting
        if self.subject_id == "unity_sessions":
            self.subject_id = path.parent.parent.name + "/unity"

        self.metadata        = _load_json(path / "metadata.json") or {}
        self.gesture_set     = _load_json(path / "gesture_set.json") or {}
        self.has_data_csv    = (path / "data.csv").exists()
        self.has_data_npy    = (path / "data.npy").exists()

        self.issues: List[str]   = []
        self.warnings: List[str] = []
        self.info: Dict[str, Any] = {}

        self._analyse()

    # ── private ────────────────────────────────────────────────────────────

    def _analyse(self):
        meta = self.metadata.get("metadata", self.metadata)
        trials = self.metadata.get("trials", [])

        # ── basic metadata checks ──────────────────────────────────────────
        if not self.metadata:
            self.issues.append("metadata.json missing or unreadable")
        if not self.has_data_csv:
            self.issues.append("data.csv missing")
        if not self.has_data_npy:
            self.warnings.append("data.npy missing (CSV-only session)")

        # ── trial summary ──────────────────────────────────────────────────
        n_trials        = len(trials)
        valid_trials    = [t for t in trials if t.get("is_valid", True)]
        invalid_trials  = [t for t in trials if not t.get("is_valid", True)]
        n_valid         = len(valid_trials)
        n_invalid       = len(invalid_trials)
        self.info["n_trials"]   = n_trials
        self.info["n_valid"]    = n_valid
        self.info["n_invalid"]  = n_invalid

        if n_trials == 0:
            self.issues.append("No trials found in metadata")
        elif n_invalid > 0:
            pct = 100 * n_invalid / n_trials
            msg = f"{n_invalid}/{n_trials} trials marked invalid ({pct:.0f}%)"
            (self.issues if pct > 50 else self.warnings).append(msg)

        # Collect notes from invalid trials
        invalid_notes = [t.get("notes", "") for t in invalid_trials if t.get("notes")]
        if invalid_notes:
            self.info["invalid_notes"] = invalid_notes

        # ── session metadata ───────────────────────────────────────────────
        sr              = meta.get("sampling_rate", 0)
        n_ch            = meta.get("num_channels",  0)
        calibrated      = meta.get("calibration_applied", False)
        bad_channels    = meta.get("bad_channels", [])
        notes           = meta.get("notes", "")
        protocol        = meta.get("protocol_name", "unknown")

        self.info["sampling_rate"]  = sr
        self.info["num_channels"]   = n_ch
        self.info["calibrated"]     = calibrated
        self.info["bad_channels"]   = bad_channels
        self.info["protocol"]       = protocol
        if notes:
            self.info["session_notes"] = notes

        if sr not in (0, 2000):
            self.warnings.append(f"Unexpected sampling rate: {sr} Hz")
        if n_ch not in (0, 16, 32):
            self.warnings.append(f"Unexpected channel count: {n_ch}")
        if bad_channels:
            self.warnings.append(f"Bad channels: {bad_channels}")
        if notes and any(kw in notes.lower() for kw in ("bad", "fail", "error", "wrong", "broken")):
            self.issues.append(f"Session note contains warning keyword: \"{notes}\"")

        # ── duration check ─────────────────────────────────────────────────
        if n_valid > 0 and sr > 0:
            last_trial    = max(valid_trials, key=lambda t: t.get("end_sample", 0))
            duration_s    = last_trial.get("end_sample", 0) / sr
            self.info["estimated_duration_s"] = round(duration_s, 1)
            if duration_s < 10:
                self.issues.append(f"Very short session: {duration_s:.1f}s")
            elif duration_s < 30:
                self.warnings.append(f"Short session: {duration_s:.1f}s")

        # ── CSV spot-check ─────────────────────────────────────────────────
        if self.has_data_csv:
            self._check_data_csv(n_ch)

        # ── class balance ──────────────────────────────────────────────────
        if valid_trials:
            from collections import Counter
            label_counts = Counter(t.get("gesture_label") for t in valid_trials)
            self.info["class_distribution"] = dict(label_counts)
            counts = list(label_counts.values())
            if counts:
                imbalance = max(counts) / max(min(counts), 1)
                if imbalance > 5:
                    self.warnings.append(f"Strong class imbalance (ratio {imbalance:.1f}x)")

    def _check_data_csv(self, expected_channels: int):
        csv_path = self.path / "data.csv"
        rows = _load_csv_head(csv_path, n_rows=200)
        if not rows or len(rows) < 2:
            self.issues.append("data.csv is empty or unreadable")
            return

        header = rows[0]
        n_cols = len(header)
        self.info["csv_columns"] = n_cols

        if expected_channels > 0 and n_cols != expected_channels:
            self.warnings.append(
                f"CSV has {n_cols} columns but metadata says {expected_channels} channels"
            )

        # Collect all numeric values from sample rows for basic signal checks
        all_vals = []
        for row in rows[1:]:
            for cell in row:
                try:
                    all_vals.append(float(cell))
                except ValueError:
                    pass

        if all_vals:
            stats = _emg_stats(all_vals)
            self.info["signal_rms"]  = round(stats.get("rms", 0), 4)
            self.info["signal_std"]  = round(stats.get("std", 0), 4)
            frac_zero = _fraction_zeros(all_vals)
            frac_sat  = _fraction_saturated(all_vals)
            self.info["frac_zeros"]     = round(frac_zero, 3)
            self.info["frac_saturated"] = round(frac_sat, 3)

            if frac_zero > 0.30:
                self.warnings.append(f"{frac_zero*100:.0f}% of sampled values are exactly zero")
            if frac_sat > 0.05:
                self.warnings.append(f"{frac_sat*100:.1f}% of sampled values appear saturated (|v|≥4)")
            if stats.get("rms", 1) < 1e-6:
                self.issues.append("Signal RMS is near zero — possible flat-line recording")

    # ── public ─────────────────────────────────────────────────────────────

    @property
    def status(self) -> str:
        if self.issues:
            return "BAD"
        if self.warnings:
            return "WARN"
        return "OK"

    def summary_line(self) -> str:
        s = self.status
        colour = RED if s == "BAD" else (YELLOW if s == "WARN" else GREEN)
        label  = colour(f"[{s:4s}]")
        trials = f"{self.info.get('n_valid','?')}✓/{self.info.get('n_trials','?')} trials"
        dur    = f"{self.info.get('estimated_duration_s', '?')}s"
        return f"{label} {self.subject_id:12s} | {self.session_id:35s} | {trials:15s} | {dur}"


# ──────────────────────────────────────────────────────────────────────────────
# Game recording analyser  (game_recordings/VP_xx/game_yyyy-mm-dd/)
# ──────────────────────────────────────────────────────────────────────────────

class GameRecordingQuality:
    """Quality report for one game recording."""

    def __init__(self, path: Path):
        self.path           = path
        self.subject_id     = path.parent.name
        self.session_id     = path.name
        self.recording_type = "game"

        self.config         = _load_json(path / "config.json") or {}
        self.has_csv        = (path / "recording.csv").exists()
        self.has_bak        = (path / "recording.csv.bak").exists()

        self.issues: List[str]   = []
        self.warnings: List[str] = []
        self.info: Dict[str, Any] = {}

        self._analyse()

    def _analyse(self):
        rec   = self.config.get("recording", {})
        model = self.config.get("model",     {})
        calib = self.config.get("calibration", {})

        # ── config checks ─────────────────────────────────────────────────
        if not self.config:
            self.warnings.append("config.json missing or unreadable")
        if not self.has_csv:
            self.issues.append("recording.csv missing")

        # ── basic recording metadata ───────────────────────────────────────
        started  = rec.get("started_at",  "")
        finished = rec.get("finished_at", "")
        duration = rec.get("duration_seconds", None)
        n_samples= rec.get("total_samples", None)
        sr       = rec.get("sampling_rate", 2000)
        n_ch     = rec.get("num_channels", 32)
        classes  = rec.get("class_names", [])

        self.info["started_at"]       = started
        self.info["duration_s"]       = duration
        self.info["total_samples"]    = n_samples
        self.info["sampling_rate"]    = sr
        self.info["num_channels"]     = n_ch
        self.info["class_names"]      = classes
        self.info["has_backup_csv"]   = self.has_bak

        if duration is not None:
            if duration < 10:
                self.issues.append(f"Recording very short: {duration:.1f}s")
            elif duration < 30:
                self.warnings.append(f"Recording short: {duration:.1f}s")

        if not finished:
            self.warnings.append("No finished_at timestamp — recording may be incomplete")

        # ── model info ────────────────────────────────────────────────────
        model_name = model.get("name", "")
        val_acc    = model.get("validation_accuracy", None)
        self.info["model_name"] = model_name
        self.info["model_val_acc"] = val_acc

        if val_acc is not None and val_acc < 0.70:
            self.warnings.append(f"Model validation accuracy low: {val_acc:.1%}")

        # ── calibration ───────────────────────────────────────────────────
        if calib:
            rot_conf = calib.get("rotation_confidence", None)
            self.info["rotation_confidence"] = rot_conf
            if rot_conf is not None and rot_conf < 0.3:
                self.warnings.append(f"Low rotation confidence: {rot_conf:.3f}")

        # ── CSV analysis ──────────────────────────────────────────────────
        if self.has_csv:
            self._check_recording_csv(n_ch, sr, duration)

    def _check_recording_csv(self, expected_channels: int,
                              sampling_rate: int, expected_duration: Optional[float]):
        csv_path = self.path / "recording.csv"
        rows = _load_csv_head(csv_path, n_rows=300)
        if not rows or len(rows) < 2:
            self.issues.append("recording.csv empty or unreadable")
            return

        header = rows[0]

        # Identify EMG columns
        emg_cols = [i for i, h in enumerate(header) if h.startswith("EMG_Ch")]
        n_emg = len(emg_cols)
        self.info["csv_emg_channels"] = n_emg

        if expected_channels > 0 and n_emg != expected_channels:
            self.warnings.append(
                f"CSV has {n_emg} EMG channels but config says {expected_channels}"
            )

        # GroundTruth column check
        gt_col = next((i for i, h in enumerate(header) if h == "GroundTruthActive"), None)
        req_col = next((i for i, h in enumerate(header) if h == "RequestedGesture"), None)

        if gt_col is not None and req_col is not None:
            gt_values    = []
            req_gestures = set()
            for row in rows[1:]:
                if gt_col < len(row):
                    try:
                        gt_values.append(int(float(row[gt_col])))
                    except ValueError:
                        pass
                if req_col < len(row):
                    req_gestures.add(row[req_col].strip().lower())

            if gt_values:
                frac_active = sum(gt_values) / len(gt_values)
                self.info["gt_active_fraction_sample"] = round(frac_active, 3)
                if frac_active == 0.0:
                    self.warnings.append("Ground truth never active in sampled rows (first 300)")

            self.info["gestures_seen"] = sorted(req_gestures - {""})

        # Confidence column check
        conf_col = next((i for i, h in enumerate(header) if h == "Confidence"), None)
        if conf_col is not None:
            conf_vals = []
            for row in rows[1:]:
                if conf_col < len(row):
                    try:
                        conf_vals.append(float(row[conf_col]))
                    except ValueError:
                        pass
            if conf_vals:
                avg_conf = sum(conf_vals) / len(conf_vals)
                self.info["avg_confidence_sample"] = round(avg_conf, 3)
                if avg_conf < 0.5:
                    self.warnings.append(f"Low average model confidence: {avg_conf:.2f}")

        # EMG signal quality from sample rows
        emg_vals = []
        for row in rows[1:]:
            for ci in emg_cols[:8]:    # check first 8 channels for speed
                if ci < len(row):
                    try:
                        emg_vals.append(float(row[ci]))
                    except ValueError:
                        pass

        if emg_vals:
            stats    = _emg_stats(emg_vals)
            frac_z   = _fraction_zeros(emg_vals)
            frac_sat = _fraction_saturated(emg_vals)
            self.info["signal_rms"]      = round(stats.get("rms", 0), 5)
            self.info["frac_zeros"]      = round(frac_z, 3)
            self.info["frac_saturated"]  = round(frac_sat, 3)

            if frac_z > 0.40:
                self.warnings.append(f"{frac_z*100:.0f}% of sampled EMG values are zero")
            if frac_sat > 0.05:
                self.warnings.append(f"{frac_sat*100:.1f}% of sampled EMG values appear saturated")
            if stats.get("rms", 1) < 1e-6:
                self.issues.append("EMG RMS is near zero — possible flat-line signal")

        # Actual row count (slower, but gives ground truth for expected duration)
        actual_rows = _csv_row_count(csv_path)
        if actual_rows >= 0:
            self.info["actual_rows"] = actual_rows
            if sampling_rate > 0 and expected_duration:
                expected_rows = int(expected_duration * sampling_rate)
                ratio = actual_rows / max(expected_rows, 1)
                self.info["row_coverage"] = round(ratio, 3)
                if ratio < 0.80:
                    self.warnings.append(
                        f"CSV has only {ratio:.0%} of expected rows "
                        f"({actual_rows} vs ~{expected_rows})"
                    )

    @property
    def status(self) -> str:
        if self.issues:
            return "BAD"
        if self.warnings:
            return "WARN"
        return "OK"

    def summary_line(self) -> str:
        s = self.status
        colour = RED if s == "BAD" else (YELLOW if s == "WARN" else GREEN)
        label  = colour(f"[{s:4s}]")
        dur    = self.info.get("duration_s")
        dur_s  = f"{dur:.1f}s" if dur is not None else "?s"
        acc    = self.info.get("model_val_acc")
        acc_s  = f"val_acc={acc:.0%}" if acc is not None else ""
        return (
            f"{label} {self.subject_id:12s} | {self.session_id:35s} | "
            f"{dur_s:7s} | {acc_s}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Scanner
# ──────────────────────────────────────────────────────────────────────────────

def scan_data_dir(data_dir: Path):
    """Walk the data directory and return all quality report objects."""
    reports = []

    # ── sessions ──────────────────────────────────────────────────────────
    sessions_root = data_dir / "sessions"
    if sessions_root.exists():
        for subject_dir in sorted(sessions_root.iterdir()):
            if not subject_dir.is_dir():
                continue
            if subject_dir.name == "unity_sessions":
                # One more level of nesting
                for unity_subj in sorted(subject_dir.iterdir()):
                    if unity_subj.is_dir():
                        for sess_dir in sorted(unity_subj.iterdir()):
                            if sess_dir.is_dir() and (sess_dir / "metadata.json").exists():
                                reports.append(SessionQuality(sess_dir))
            else:
                for sess_dir in sorted(subject_dir.iterdir()):
                    if sess_dir.is_dir() and (sess_dir / "metadata.json").exists():
                        reports.append(SessionQuality(sess_dir))

    # ── game recordings ───────────────────────────────────────────────────
    game_root = data_dir / "game_recordings"
    if game_root.exists():
        for subject_dir in sorted(game_root.iterdir()):
            if not subject_dir.is_dir():
                continue
            for game_dir in sorted(subject_dir.iterdir()):
                if game_dir.is_dir() and (game_dir / "recording.csv").exists():
                    reports.append(GameRecordingQuality(game_dir))

    return reports


# ──────────────────────────────────────────────────────────────────────────────
# Review state persistence
# ──────────────────────────────────────────────────────────────────────────────

REVIEW_FILE = "quality_review.json"
EXPORT_FILE = "quality_review_summary.csv"

def _review_key(r) -> str:
    return f"{r.recording_type}:{r.subject_id}:{r.session_id}"


def load_reviews(path: Path) -> Dict[str, Dict]:
    if path.exists():
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_reviews(reviews: Dict, path: Path):
    with open(path, "w") as f:
        json.dump(reviews, f, indent=2, default=str)
    print(GREEN(f"  ✓ Saved to {path}"))


def export_csv(reports, reviews: Dict, path: Path):
    fieldnames = [
        "type", "subject_id", "session_id", "auto_status",
        "review_status", "comment",
        "n_trials", "n_valid", "n_invalid",
        "duration_s", "sampling_rate", "num_channels",
        "bad_channels", "signal_rms", "frac_zeros", "frac_saturated",
        "model_val_acc", "rotation_confidence",
        "issues", "warnings", "path",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in reports:
            key = _review_key(r)
            rev = reviews.get(key, {})
            row = {
                "type":               r.recording_type,
                "subject_id":         r.subject_id,
                "session_id":         r.session_id,
                "auto_status":        r.status,
                "review_status":      rev.get("status", ""),
                "comment":            rev.get("comment", ""),
                "path":               str(r.path),
                "issues":             " | ".join(r.issues),
                "warnings":           " | ".join(r.warnings),
            }
            row.update(r.info)
            # Stringify list fields
            for k in ("bad_channels", "class_names", "gestures_seen",
                      "invalid_notes", "class_distribution"):
                if k in row and isinstance(row[k], (list, dict)):
                    row[k] = str(row[k])
            w.writerow(row)
    print(GREEN(f"  ✓ Exported to {path}"))


# ──────────────────────────────────────────────────────────────────────────────
# Report-only mode
# ──────────────────────────────────────────────────────────────────────────────

def print_report(reports):
    sessions = [r for r in reports if r.recording_type == "session"]
    games    = [r for r in reports if r.recording_type == "game"]

    def _section(title, items):
        print(BOLD(f"\n{'─'*70}"))
        print(BOLD(f"  {title}  ({len(items)} total)"))
        print(BOLD(f"{'─'*70}"))
        by_status = {"BAD": [], "WARN": [], "OK": []}
        for r in items:
            by_status[r.status].append(r)
        for status in ("BAD", "WARN", "OK"):
            group = by_status[status]
            if group:
                for r in group:
                    print(r.summary_line())
                    for iss in r.issues:
                        print(RED(f"      ✗ {iss}"))
                    for wrn in r.warnings:
                        print(YELLOW(f"      ⚠ {wrn}"))

        counts = {s: len(v) for s, v in by_status.items()}
        print(DIM(f"\n  OK: {counts['OK']}  |  WARN: {counts['WARN']}  |  BAD: {counts['BAD']}"))

    _section("TRAINING SESSIONS", sessions)
    _section("GAME RECORDINGS",   games)
    print()


# ──────────────────────────────────────────────────────────────────────────────
# Interactive reviewer
# ──────────────────────────────────────────────────────────────────────────────

def _clear():
    os.system("cls" if os.name == "nt" else "clear")


def _render_detail(r, review: Dict, idx: int, total: int):
    _clear()
    is_session = r.recording_type == "session"
    type_label = "SESSION" if is_session else "GAME REC"
    status_colour = RED if r.status == "BAD" else (YELLOW if r.status == "WARN" else GREEN)

    print(BOLD(f"\n  {type_label}  [{idx+1}/{total}]"))
    print(f"  Subject  : {CYAN(r.subject_id)}")
    print(f"  Session  : {CYAN(r.session_id)}")
    print(f"  Path     : {DIM(str(r.path))}")
    print(f"  Auto     : {status_colour(r.status)}")

    rev_status = review.get("status", "")
    if rev_status:
        col = GREEN if rev_status == "VALID" else (RED if rev_status == "INVALID" else YELLOW)
        print(f"  Review   : {col(rev_status)}")
    comment = review.get("comment", "")
    if comment:
        print(f"  Comment  : {YELLOW(comment)}")

    # ── Issues & Warnings ──────────────────────────────────────────────────
    if r.issues:
        print(RED(f"\n  ✗ Issues ({len(r.issues)}):"))
        for iss in r.issues:
            print(RED(f"    • {iss}"))
    if r.warnings:
        print(YELLOW(f"\n  ⚠ Warnings ({len(r.warnings)}):"))
        for wrn in r.warnings:
            print(YELLOW(f"    • {wrn}"))
    if not r.issues and not r.warnings:
        print(GREEN("\n  ✓ No issues detected"))

    # ── Info ───────────────────────────────────────────────────────────────
    print(DIM(f"\n  {'─'*60}"))
    print(BOLD("  Info:"))
    exclude = {"class_names", "gestures_seen"}  # too verbose in detail view
    for k, v in r.info.items():
        if k in exclude:
            continue
        vstr = str(v)
        if len(vstr) > 60:
            vstr = vstr[:57] + "..."
        print(f"    {k:30s}: {vstr}")

    # ── Key hints ─────────────────────────────────────────────────────────
    print(DIM(f"\n  {'─'*60}"))
    print(DIM("  v=valid  i=invalid  q=questionable  c=comment  "
              "n=next  p=prev  s=save  e=export  x=exit"))


def interactive_review(reports, data_dir: Path):
    review_path = data_dir / REVIEW_FILE
    export_path = data_dir / EXPORT_FILE
    reviews = load_reviews(review_path)

    idx   = 0
    total = len(reports)

    if total == 0:
        print("No recordings found.")
        return

    # Try to use getch for single-keypress (Unix) or msvcrt (Windows)
    try:
        import tty, termios
        def _getch():
            fd = sys.stdin.fileno()
            old = termios.tcgetattr(fd)
            try:
                tty.setraw(fd)
                ch = sys.stdin.read(1)
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old)
            return ch
    except ImportError:
        def _getch():
            return input("\n  > ").strip().lower()[:1]

    while True:
        r   = reports[idx]
        key = _review_key(r)
        rev = reviews.get(key, {})

        _render_detail(r, rev, idx, total)

        ch = _getch()

        if ch in ("n", "\r", "\n", " "):
            idx = (idx + 1) % total
        elif ch == "p":
            idx = (idx - 1) % total
        elif ch == "v":
            reviews.setdefault(key, {})["status"] = "VALID"
            reviews[key]["reviewed_at"] = datetime.now().isoformat()
            idx = (idx + 1) % total
        elif ch == "i":
            reviews.setdefault(key, {})["status"] = "INVALID"
            reviews[key]["reviewed_at"] = datetime.now().isoformat()
            idx = (idx + 1) % total
        elif ch == "q":
            reviews.setdefault(key, {})["status"] = "QUESTIONABLE"
            reviews[key]["reviewed_at"] = datetime.now().isoformat()
            idx = (idx + 1) % total
        elif ch == "c":
            _clear()
            print(f"\n  Comment for {r.session_id} (leave blank to clear):")
            comment = input("  > ").strip()
            reviews.setdefault(key, {})["comment"] = comment
        elif ch == "s":
            save_reviews(reviews, review_path)
        elif ch == "e":
            export_csv(reports, reviews, export_path)
        elif ch in ("x", "X", "\x03", "\x04"):   # x, Ctrl-C, Ctrl-D
            print("\n  Saving before exit …")
            save_reviews(reviews, review_path)
            export_csv(reports, reviews, export_path)
            print("  Bye!")
            break
        # Jump directly to first BAD recording on 'b'
        elif ch == "b":
            bad_indices = [i for i, rep in enumerate(reports) if rep.status == "BAD"]
            if bad_indices:
                # find next BAD after current
                nexts = [i for i in bad_indices if i > idx]
                idx = nexts[0] if nexts else bad_indices[0]
            else:
                pass   # no bad recordings


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Scan recording sessions and game recordings for quality issues."
    )
    parser.add_argument(
        "data_dir",
        type=Path,
        help="Root data directory (contains sessions/ and game_recordings/)",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Print a plain text summary and exit (no interactive reviewer)",
    )
    parser.add_argument(
        "--export",
        action="store_true",
        help="Export CSV summary to quality_review_summary.csv and exit",
    )
    args = parser.parse_args()

    data_dir = args.data_dir.expanduser().resolve()
    if not data_dir.exists():
        print(f"Error: data directory not found: {data_dir}")
        sys.exit(1)

    print(f"Scanning {data_dir} …")
    reports = scan_data_dir(data_dir)

    n_sess  = sum(1 for r in reports if r.recording_type == "session")
    n_game  = sum(1 for r in reports if r.recording_type == "game")
    n_bad   = sum(1 for r in reports if r.status == "BAD")
    n_warn  = sum(1 for r in reports if r.status == "WARN")
    n_ok    = sum(1 for r in reports if r.status == "OK")

    print(f"Found {len(reports)} recordings "
          f"({n_sess} sessions, {n_game} game recordings) — "
          f"OK:{n_ok}  WARN:{n_warn}  BAD:{n_bad}")

    if not reports:
        print("Nothing to review.")
        return

    if args.report or args.export:
        print_report(reports)
        if args.export:
            export_csv(reports, {}, data_dir / EXPORT_FILE)
        return

    interactive_review(reports, data_dir)


if __name__ == "__main__":
    main()
