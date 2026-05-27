#!/usr/bin/env python3
"""
plot_threshold_recording.py
═══════════════════════════
Standalone timeline plot for a single Unity threshold-gameplay CSV.

Produces a thesis-quality figure of one recording: the RMS envelope
over time, the ground-truth "should be active" periods shaded behind
it, the RMS threshold(s) drawn as horizontal reference lines, and a
strip showing what the game actually did (``GestureActive``). It is
deliberately self-contained — no dependency on the playagain_pipeline
package — so it can be dropped anywhere and run on its own.

Usage
─────
    python plot_threshold_recording.py RECORDING.csv
    python plot_threshold_recording.py RECORDING.csv --profile profile.json
    python plot_threshold_recording.py RECORDING.csv --threshold 0.00034
    python plot_threshold_recording.py RECORDING.csv -o figure.pdf

What gets drawn
───────────────
* **Main panel** — the RMS signal as a filled area in time. Periods
  where the game asked for a gesture (``GroundTruthActive == 1``) are
  shaded; the requested gesture name (pinch / fist / tripod / …) is
  annotated above each band.
* **Threshold lines** — one for the "optimal" F1-maximising threshold
  (always computed) and, when a profile or an explicit ``--threshold``
  is given, one for the threshold the game actually used. The mV→V
  conversion for profile thresholds is applied automatically.
* **Detection strip** — a thin band beneath the main panel showing
  ``GestureActive``: green where the game registered the gesture,
  blank where it did not. For most real recordings this strip is
  almost entirely blank, which is the point.
* **Caption box** — per-recording numbers: duration, RMS percentiles,
  thresholds, and the as-recorded vs optimal F1.

The figure is saved as both PDF (vector, for the thesis) and PNG.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Matplotlib on a non-interactive backend so this runs headless.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# ── Palette ─────────────────────────────────────────────────────────────────
# A restrained, print-friendly palette. Muted blues for the signal,
# a warm amber for the "active" cue bands, green/red for the binary
# detection strip.
C_RMS_LINE   = "#1d4e89"      # deep blue — RMS trace
C_RMS_FILL   = "#cfe0f3"      # pale blue — RMS area fill
C_CUE_BAND   = "#fcd9a8"      # warm sand — ground-truth active band
C_CUE_EDGE   = "#e8a13a"      # amber — band edge
C_THR_OPT    = "#1f9d55"      # green — optimal threshold
C_THR_GAME   = "#d6336c"      # magenta — threshold the game used
C_DETECT_ON  = "#2f9e44"      # green — game registered the gesture
C_DETECT_OFF = "#f1f3f5"      # near-white — no detection
C_TEXT       = "#1f2933"
C_MUTED      = "#6b7280"
C_GRID       = "#e3e8ef"


# ── Gesture-name prettifier ─────────────────────────────────────────────────
_GESTURE_PRETTY = {
    "none": "rest",
    "fist": "Fist",
    "pinch": "Pinch",
    "tripod": "Tripod",
    "open": "Open hand",
}


def _pretty_gesture(name: str) -> str:
    return _GESTURE_PRETTY.get(str(name).strip().lower(), str(name).title())


# ════════════════════════════════════════════════════════════════════════════
# Data loading
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class Recording:
    """The columns of one Unity gameplay CSV we care about."""
    t:          np.ndarray      # seconds, zero-based
    rms:        np.ndarray      # RMS in volts
    truth:      np.ndarray      # ground-truth active (0/1)
    detected:   np.ndarray      # GestureActive — what the game did (0/1)
    requested:  np.ndarray      # RequestedGesture name per frame (object)
    name:       str             # recording stem, for titles
    duration_s: float


_TRUTH_COLS = ("GroundTruthActive", "GroundTruth")


def load_recording(csv_path: Path) -> Recording:
    """Read a Unity gameplay CSV into a :class:`Recording`."""
    import pandas as pd                            # local import — heavy

    df = pd.read_csv(csv_path)

    # Drop non-EMG rows (Markers etc.) — they carry no RMS sample.
    if "InputSource" in df.columns:
        df = df[df["InputSource"] == "EMG"].reset_index(drop=True)

    if "RMS" not in df.columns:
        raise ValueError(f"{csv_path.name}: no RMS column")

    truth_col = next((c for c in _TRUTH_COLS if c in df.columns), None)
    if truth_col is None:
        raise ValueError(
            f"{csv_path.name}: no GroundTruth / GroundTruthActive column"
        )

    t = df["Timestamp"].to_numpy(dtype=np.float64)
    t = t - t[0]                                   # zero-base the clock
    rms = df["RMS"].to_numpy(dtype=np.float64)
    truth = (df[truth_col].to_numpy(dtype=np.float64) > 0.5).astype(np.int64)

    if "GestureActive" in df.columns:
        detected = (df["GestureActive"].to_numpy(dtype=np.float64) > 0.5
                    ).astype(np.int64)
    else:
        detected = np.zeros(rms.size, dtype=np.int64)

    if "RequestedGesture" in df.columns:
        requested = df["RequestedGesture"].astype(str).to_numpy()
    else:
        requested = np.where(truth == 1, "active", "none").astype(object)

    return Recording(
        t=t, rms=rms, truth=truth, detected=detected, requested=requested,
        name=csv_path.stem,
        duration_s=float(t[-1] - t[0]) if t.size else 0.0,
    )


# ════════════════════════════════════════════════════════════════════════════
# Profile threshold resolution (mirrors evaluation/threshold_eval.py)
# ════════════════════════════════════════════════════════════════════════════

_FNAME_TS_RE = re.compile(
    r"(\d{4})-(\d{2})-(\d{2})[_-](\d{2})-(\d{2})-(\d{2})"
)


def _parse_ts(s: str) -> Optional[datetime]:
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f"):
        try:
            return datetime.strptime(s.strip(), fmt)
        except (ValueError, AttributeError):
            continue
    return None


def resolve_profile_threshold(
    profile_path: Path,
    csv_path: Path,
    *,
    units: str = "mV",
) -> Tuple[Optional[float], str]:
    """
    Pick the threshold from ``profile.json`` that was active when the
    recording started, converting mV → V. Returns (threshold_V, note).

    Mirrors the logic in the main evaluator: prefer the
    ``thresholdHistory`` entry whose timestamp is the latest still
    before the recording start (parsed from the filename); fall back
    to ``currentThreshold``.
    """
    try:
        with open(profile_path, "r", encoding="utf-8") as f:
            profile = json.load(f)
    except Exception as exc:                       # noqa: BLE001
        return None, f"profile unreadable ({exc})"

    # Recording start from the filename.
    started_at = None
    m = _FNAME_TS_RE.search(csv_path.stem)
    if m:
        try:
            started_at = datetime(*map(int, m.groups()))
        except ValueError:
            started_at = None

    raw: Optional[float] = None
    src = "no profile threshold"
    history = profile.get("thresholdHistory") or []
    if started_at is not None and history:
        best, best_ts = None, None
        for entry in history:
            ts = _parse_ts(entry.get("timestamp", ""))
            if ts is None or ts > started_at:
                continue
            if best_ts is None or ts > best_ts:
                best, best_ts = entry, ts
        if best is not None:
            raw = float(best.get("threshold", "nan"))
            src = f"profile history @ {best.get('timestamp')}"
    if raw is None and profile.get("currentThreshold") is not None:
        raw = float(profile["currentThreshold"])
        src = "profile currentThreshold"

    if raw is None or not np.isfinite(raw):
        return None, src

    if units.lower() in ("mv", "millivolt", "millivolts"):
        return raw / 1000.0, f"{src} [{raw:g} mV → {raw/1000.0:g} V]"
    return raw, src


# ════════════════════════════════════════════════════════════════════════════
# Metrics — just enough to annotate the figure
# ════════════════════════════════════════════════════════════════════════════

def _f1(truth: np.ndarray, pred: np.ndarray) -> float:
    tp = int(np.sum((pred == 1) & (truth == 1)))
    fp = int(np.sum((pred == 1) & (truth == 0)))
    fn = int(np.sum((pred == 0) & (truth == 1)))
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    return (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0


def optimal_threshold(rms: np.ndarray, truth: np.ndarray,
                      n: int = 200) -> Tuple[float, float]:
    """Return (threshold, F1) of the F1-maximising RMS threshold."""
    pos = rms[rms > 0]
    if pos.size == 0 or truth.sum() == 0 or truth.sum() == truth.size:
        return float("nan"), float("nan")
    lo = float(np.quantile(pos, 0.01))
    hi = float(np.quantile(pos, 0.995))
    if hi <= lo:
        hi = lo * 10
    best_t, best_f1 = float("nan"), -1.0
    for t in np.geomspace(max(lo, 1e-12), hi, n):
        f1 = _f1(truth, (rms > t).astype(np.int64))
        if f1 > best_f1:
            best_t, best_f1 = float(t), f1
    return best_t, best_f1


# ════════════════════════════════════════════════════════════════════════════
# Helpers for drawing
# ════════════════════════════════════════════════════════════════════════════

def _binary_segments(mask: np.ndarray, t: np.ndarray) -> List[Tuple[float, float]]:
    """
    Convert a 0/1 mask into a list of (t_start, t_end) intervals where
    the mask is 1. Used to shade ground-truth bands and detection runs.
    """
    segs: List[Tuple[float, float]] = []
    if mask.size == 0:
        return segs
    edges = np.diff(mask.astype(np.int8))
    starts = list(np.where(edges == 1)[0] + 1)
    ends   = list(np.where(edges == -1)[0] + 1)
    if mask[0] == 1:
        starts = [0] + starts
    if mask[-1] == 1:
        ends = ends + [mask.size - 1]
    for s, e in zip(starts, ends):
        segs.append((float(t[s]), float(t[min(e, t.size - 1)])))
    return segs


def _gesture_label_for_segment(rec: Recording, t0: float, t1: float) -> str:
    """The dominant requested-gesture name within a cue band."""
    mask = (rec.t >= t0) & (rec.t <= t1)
    names = rec.requested[mask]
    names = [n for n in names if str(n).strip().lower() != "none"]
    if not names:
        return "active"
    # Most frequent non-"none" label.
    vals, counts = np.unique(names, return_counts=True)
    return _pretty_gesture(vals[int(np.argmax(counts))])


def _human_dt(seconds: float) -> str:
    """Compact duration: 81.0 → '1:21 min'."""
    if seconds < 90:
        return f"{seconds:.1f} s"
    m, s = divmod(int(round(seconds)), 60)
    return f"{m}:{s:02d} min"


# ════════════════════════════════════════════════════════════════════════════
# The figure
# ════════════════════════════════════════════════════════════════════════════

def plot_recording(
    rec: Recording,
    out_stem: Path,
    *,
    game_threshold: Optional[float] = None,
    game_threshold_note: str = "",
    title: Optional[str] = None,
    smooth_ms: float = 0.0,
) -> List[Path]:
    """
    Render the timeline figure for one recording and save PDF + PNG.

    Parameters
    ----------
    rec : Recording
    out_stem : Path
        Output path without extension.
    game_threshold : float, optional
        The RMS threshold the game actually used (volts). Drawn as a
        magenta reference line when given.
    game_threshold_note : str
        Provenance string for the caption (e.g. the mV→V conversion).
    title : str, optional
        Figure title; defaults to the recording name.
    smooth_ms : float
        Optional moving-average window for the *displayed* RMS trace
        (purely cosmetic — metrics always use the raw signal). 0 = off.
    """
    t, rms, truth = rec.t, rec.rms, rec.truth

    # Cosmetic smoothing of the trace only.
    rms_plot = rms
    if smooth_ms > 0 and t.size > 2:
        dt = float(np.median(np.diff(t)))
        win = max(1, int(round(smooth_ms / 1000.0 / max(dt, 1e-9))))
        if win > 1:
            kernel = np.ones(win) / win
            rms_plot = np.convolve(rms, kernel, mode="same")

    thr_opt, f1_opt = optimal_threshold(rms, truth)
    f1_asrec = _f1(truth, rec.detected)

    # ── Figure scaffold: main panel + thin detection strip ───────────
    # The bottom margin is generous: the multi-line monospace caption
    # box sits in the reserved strip below the x-axis label so it
    # never overprints the ticks.
    fig = plt.figure(figsize=(12.5, 6.2))
    gs = fig.add_gridspec(
        2, 1, height_ratios=[6, 1], hspace=0.18,
        left=0.085, right=0.80, top=0.84, bottom=0.26,
    )
    ax = fig.add_subplot(gs[0])
    ax_strip = fig.add_subplot(gs[1], sharex=ax)

    # RMS values are tiny (mV-scale signal logged in volts); show the
    # axis in millivolts so the numbers read naturally.
    rms_mv      = rms_plot * 1000.0
    thr_opt_mv  = thr_opt * 1000.0 if np.isfinite(thr_opt) else None
    thr_game_mv = (game_threshold * 1000.0
                   if game_threshold is not None else None)

    # ── Ground-truth cue bands ───────────────────────────────────────
    cue_segments = _binary_segments(truth, t)
    for (t0, t1) in cue_segments:
        ax.axvspan(t0, t1, color=C_CUE_BAND, alpha=0.55, zorder=0)
        ax.axvline(t0, color=C_CUE_EDGE, lw=0.8, alpha=0.7, zorder=1)
        ax.axvline(t1, color=C_CUE_EDGE, lw=0.8, alpha=0.7, zorder=1)

    # Gesture labels above each band — skip very short bands so the
    # labels do not collide.
    y_top = max(rms_mv.max(), (thr_game_mv or 0), (thr_opt_mv or 0)) * 1.18
    if not np.isfinite(y_top) or y_top <= 0:
        y_top = 1.0
    min_label_span = rec.duration_s * 0.035
    for (t0, t1) in cue_segments:
        if (t1 - t0) < min_label_span:
            continue
        label = _gesture_label_for_segment(rec, t0, t1)
        ax.text((t0 + t1) / 2, y_top * 0.97, label,
                ha="center", va="top", fontsize=8.5, color="#9a6b1e",
                fontweight="bold", zorder=5)

    # ── RMS trace ────────────────────────────────────────────────────
    ax.fill_between(t, 0, rms_mv, color=C_RMS_FILL, zorder=2)
    ax.plot(t, rms_mv, color=C_RMS_LINE, lw=0.9, zorder=3)

    # ── Threshold lines ──────────────────────────────────────────────
    if thr_opt_mv is not None:
        ax.axhline(thr_opt_mv, color=C_THR_OPT, lw=1.6, ls="--", zorder=4)
        ax.text(rec.duration_s * 1.005, thr_opt_mv,
                f" optimal\n {thr_opt_mv:.3g} mV",
                va="center", ha="left", fontsize=8,
                color=C_THR_OPT, fontweight="bold")
    if thr_game_mv is not None:
        ax.axhline(thr_game_mv, color=C_THR_GAME, lw=1.6, ls="-.", zorder=4)
        ax.text(rec.duration_s * 1.005, thr_game_mv,
                f" game threshold\n {thr_game_mv:.3g} mV",
                va="center", ha="left", fontsize=8,
                color=C_THR_GAME, fontweight="bold")

    ax.set_ylim(0, y_top)
    ax.set_xlim(0, rec.duration_s)
    ax.set_ylabel("RMS amplitude  (mV)", fontsize=10, color=C_TEXT)
    ax.grid(axis="y", color=C_GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(labelbottom=False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    # ── Detection strip ──────────────────────────────────────────────
    # A full-width pale background, with green runs where GestureActive
    # was 1. For most recordings this strip stays blank — that blankness
    # is the headline finding and the figure should make it obvious.
    ax_strip.axhspan(0, 1, color=C_DETECT_OFF, zorder=0)
    detect_segments = _binary_segments(rec.detected, t)
    for (t0, t1) in detect_segments:
        ax_strip.axvspan(t0, t1, color=C_DETECT_ON, zorder=1)
    # Also overlay faint cue-band edges so the reader can line up the
    # strip with the panel above.
    for (t0, t1) in cue_segments:
        ax_strip.axvspan(t0, t1, facecolor="none",
                         edgecolor=C_CUE_EDGE, lw=0.6, alpha=0.5, zorder=2)

    ax_strip.set_ylim(0, 1)
    ax_strip.set_yticks([0.5])
    ax_strip.set_yticklabels(["game\ndetection"], fontsize=8.5, color=C_TEXT)
    ax_strip.set_xlabel("Time  (s)", fontsize=10, color=C_TEXT)
    ax_strip.grid(False)
    for spine in ("top", "right", "left"):
        ax_strip.spines[spine].set_visible(False)

    detect_frac = float(rec.detected.mean())
    if detect_frac < 0.001:
        ax_strip.text(rec.duration_s / 2, 0.5,
                      "game never registered the gesture",
                      ha="center", va="center", fontsize=8.5,
                      style="italic", color=C_MUTED, zorder=3)

    # ── Title ────────────────────────────────────────────────────────
    fig.suptitle(
        title or f"RMS-threshold gameplay — {rec.name}",
        fontsize=13, fontweight="bold", color=C_TEXT, x=0.085, ha="left",
        y=0.95,
    )

    # ── Legend ───────────────────────────────────────────────────────
    legend_items = [
        Patch(facecolor=C_RMS_FILL, edgecolor=C_RMS_LINE,
              label="RMS amplitude"),
        Patch(facecolor=C_CUE_BAND, edgecolor=C_CUE_EDGE,
              label="ground-truth cue (gesture requested)"),
    ]
    if thr_opt_mv is not None:
        legend_items.append(
            plt.Line2D([0], [0], color=C_THR_OPT, ls="--", lw=1.6,
                       label="optimal threshold (post-hoc)"))
    if thr_game_mv is not None:
        legend_items.append(
            plt.Line2D([0], [0], color=C_THR_GAME, ls="-.", lw=1.6,
                       label="threshold used by the game"))
    legend_items.append(
        Patch(facecolor=C_DETECT_ON, label="game registered the gesture"))
    fig.legend(handles=legend_items, loc="upper right",
               bbox_to_anchor=(0.995, 0.88), fontsize=8.0,
               frameon=True, framealpha=0.96, borderpad=0.8)

    # ── Caption box — the numbers ────────────────────────────────────
    cap_lines = [
        f"Duration:  {_human_dt(rec.duration_s)}   ·   {rec.t.size:,} frames",
        f"RMS:  median {rms.max() and np.median(rms)*1000:.3g} mV"
        f"   ·   p95 {np.quantile(rms, 0.95)*1000:.3g} mV"
        f"   ·   max {rms.max()*1000:.3g} mV",
        f"Cued-active share:  {truth.mean()*100:.0f}%  of the recording",
    ]
    if thr_game_mv is not None:
        cap_lines.append(
            f"Game threshold:  {thr_game_mv:.3g} mV"
            + (f"   ({game_threshold_note})" if game_threshold_note else "")
        )
    if np.isfinite(f1_opt):
        cap_lines.append(
            f"F1:  as-recorded {f1_asrec:.3f}"
            f"   →   optimal threshold {f1_opt:.3f}"
        )
    # Place the caption in the reserved strip beneath the x-axis label,
    # inside a soft rounded box so it reads as a distinct annotation
    # rather than floating text colliding with the ticks.
    fig.text(
        0.085, 0.085, "\n".join(cap_lines),
        fontsize=8.0, color=C_TEXT, va="top", ha="left",
        family="DejaVu Sans Mono", linespacing=1.6,
        bbox=dict(boxstyle="round,pad=0.7", facecolor="#f6f8fa",
                  edgecolor=C_GRID, linewidth=1.0),
    )

    # ── Save ─────────────────────────────────────────────────────────
    # No bbox_inches="tight" here — the layout already reserves a
    # bottom strip for the caption, and "tight" would crop it away.
    out: List[Path] = []
    for ext in ("pdf", "png"):
        p = out_stem.with_suffix(f".{ext}")
        fig.savefig(p, dpi=200)
        out.append(p)
    plt.close(fig)
    return out


# ════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Plot Unity threshold-gameplay CSVs as a "
                    "thesis-quality timeline figure.",
    )
    p.add_argument("path", type=Path, help="The recording CSV to plot, or a directory to search for CSVs.")
    p.add_argument("-o", "--out", type=Path, default=None,
                   help="Output path stem (extension is ignored; both "
                        ".pdf and .png are written). "
                        "Default: alongside the CSV.")
    p.add_argument("--profile", type=Path, default=None,
                   help="profile.json to read the game threshold from.")
    p.add_argument("--threshold", type=float, default=None,
                   help="Game threshold in volts, overrides --profile.")
    p.add_argument("--threshold-mv", type=float, default=None,
                   help="Game threshold in millivolts (converted to V).")
    p.add_argument("--smooth-ms", type=float, default=0.0,
                   help="Cosmetic moving-average window for the RMS "
                        "trace, in ms (metrics use the raw signal). "
                        "0 = no smoothing.")
    p.add_argument("--title", type=str, default=None,
                   help="Override the figure title.")
    return p


def process_single_csv(csv_path: Path, args: argparse.Namespace) -> None:
    try:
        rec = load_recording(csv_path)
    except Exception as exc:                       # noqa: BLE001
        print(f"error: could not read {csv_path}: {exc}", file=sys.stderr)
        return

    # Auto-detect profile.json if not provided
    auto_profile = args.profile
    if auto_profile is None and args.threshold is None and args.threshold_mv is None:
        curr = csv_path.parent
        while curr != curr.parent:
            cand = curr / "profile.json"
            if cand.exists():
                auto_profile = cand
                break
            curr = curr.parent

    # Resolve the game threshold (volts) from the most specific source.
    game_thr: Optional[float] = None
    game_note = ""
    if args.threshold is not None:
        game_thr, game_note = args.threshold, "from --threshold"
    elif args.threshold_mv is not None:
        game_thr = args.threshold_mv / 1000.0
        game_note = f"from --threshold-mv [{args.threshold_mv:g} mV]"
    elif auto_profile is not None:
        game_thr, game_note = resolve_profile_threshold(
            auto_profile, csv_path,
        )

    out_stem = (args.out.with_suffix("") if args.out and not args.path.is_dir()
                else csv_path.with_name(csv_path.stem + "_timeline"))

    paths = plot_recording(
        rec, out_stem,
        game_threshold=game_thr,
        game_threshold_note=game_note,
        title=args.title,
        smooth_ms=args.smooth_ms,
    )
    for p in paths:
        print(f"wrote {p}")


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)

    input_path = args.path
    if not input_path.exists():
        print(f"error: {input_path} not found", file=sys.stderr)
        return 2

    if input_path.is_dir():
        csv_paths = list(input_path.rglob("*.csv"))
    else:
        csv_paths = [input_path]

    for csv_path in csv_paths:
        process_single_csv(csv_path, args)
        
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
