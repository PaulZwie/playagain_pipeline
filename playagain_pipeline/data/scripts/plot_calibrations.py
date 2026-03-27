"""
plot_calibrations.py
====================
Load every calibration JSON produced by AutoCalibrator and render a
comprehensive diagnostic figure for each one.

Usage
-----
    python plot_calibrations.py                          # uses ../calibrations/
    python plot_calibrations.py /path/to/calibrations/
    python plot_calibrations.py /path/to/calibrations/ --out ./plots/
    python plot_calibrations.py /path/to/calibrations/ --show

Each figure is saved as  <out_dir>/<stem>.png  (one file per JSON).

Figure layout (3 rows)
----------------------
Row 1  Per-gesture energy bars — one sub-panel per recorded gesture,
       colour-coded by whether each is the primary sync gesture.

Row 2  Left   : Combined energy profile (all active gestures summed),
                with MEC highlighted.
       Centre : Sync-gesture comparison — current session vs reference,
                both L2-normalised so amplitude differences don't distort
                the comparison.  Absent when this session IS the reference.
       Right  : Circular cross-correlation used to find the offset.
                The winning lag (= rotation_offset) is marked in red.
                Absent when there is no reference.

Row 3  Left   : Ring heatmap — energy mapped onto a circular electrode
                layout.  For 32-ch devices the two rings are shown as
                concentric circles; for other counts a single ring.
       Centre : Channel mapping — an arrow diagram showing how the raw
                channel indices are reordered after calibration.
       Right  : Summary text panel (offset, confidence, device, etc.).
"""

import argparse
import json
import math
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy.ndimage import gaussian_filter1d

# ── colour palette ────────────────────────────────────────────────────────────
C_CURRENT  = "#2563eb"   # blue  — current session
C_REF      = "#16a34a"   # green — reference
C_MEC      = "#dc2626"   # red   — maximum energy channel / offset marker
C_XCORR    = "#7c3aed"   # purple — cross-correlation
C_COMBINED = "#0891b2"   # teal  — combined profile
C_SYNC     = "#f59e0b"   # amber — sync-gesture highlight
C_GRID     = "#e5e7eb"   # light grey grid lines
C_TEXT     = "#111827"   # near-black text


# ── helpers ───────────────────────────────────────────────────────────────────

def load_calibration(path: Path) -> dict:
    with open(path) as f:
        data = json.load(f)
    data["reference_patterns"] = {
        k: np.array(v) for k, v in data["reference_patterns"].items()
    }
    data["created_at"] = datetime.fromisoformat(data["created_at"])
    return data


def calibration_output_stem(path: Path, cal: dict) -> str:
    """Prefer source session metadata for exported plot names."""
    if path.name.startswith("reference_calibration"):
        return path.stem
    meta = cal.get("metadata") or {}
    session_id = meta.get("source_session_id")
    if isinstance(session_id, str) and session_id.strip():
        base = "calibration_" + session_id.strip()
        if str(meta.get("signal_mode", "")).lower() == "bipolar":
            base += "_bipolar"
        return "".join(ch if ch.isalnum() or ch in "._-" else "-" for ch in base).strip(". ") or path.stem
    return path.stem


def normalize_energy(energy: np.ndarray) -> np.ndarray:
    """L2-normalise, matching the calibrator's _normalize_energy."""
    norm = np.linalg.norm(energy)
    return energy / norm if norm > 1e-12 else energy.copy()


def smooth_circular(energy: np.ndarray, sigma: float = 0.8) -> np.ndarray:
    """Circular Gaussian smooth, matching the calibrator's helper."""
    n = len(energy)
    if n < 3 or sigma <= 0:
        return energy.copy()
    pad = min(int(4 * sigma) + 1, n)
    ext = np.concatenate([energy[-pad:], energy, energy[:pad]])
    return gaussian_filter1d(ext, sigma=sigma)[pad: pad + n]


def xcorr_circular(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Circular cross-correlation via FFT."""
    return np.real(np.fft.ifft(np.fft.fft(a) * np.conj(np.fft.fft(b))))


def fold_32ch(energy: np.ndarray) -> np.ndarray:
    """Collapse 32-ch split-ring into 16 angular sectors."""
    return energy[:16] + energy[16:]


def angular_profile(energy: np.ndarray) -> np.ndarray:
    """Return the azimuthal profile used for offset detection."""
    if len(energy) == 32:
        return fold_32ch(energy)
    return energy.copy()


# ── per-panel drawing functions ───────────────────────────────────────────────

def draw_gesture_bars(ax, energy: np.ndarray, gesture_name: str,
                      is_sync: bool, is_reference_session: bool):
    """Bar chart of per-channel energy for one gesture."""
    n = len(energy)
    x = np.arange(n)
    mec = int(np.argmax(energy))

    colors = [C_MEC if i == mec else (C_SYNC if is_sync else C_CURRENT)
              for i in range(n)]
    ax.bar(x, energy, color=colors, width=0.8, linewidth=0)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_xticks(x[::max(1, n // 8)])
    ax.tick_params(labelsize=6)
    ax.set_ylabel("Energy", fontsize=6)
    ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    ax.grid(axis="y", color=C_GRID, linewidth=0.5)
    ax.spines[["top", "right"]].set_visible(False)

    title = gesture_name
    if is_sync:
        title += "  ★ sync"
    ax.set_title(title, fontsize=7, fontweight="bold" if is_sync else "normal",
                 color=C_SYNC if is_sync else C_TEXT)


def draw_combined(ax, energy: np.ndarray, num_channels: int):
    """Combined energy profile with MEC annotated."""
    n = len(energy)
    x = np.arange(n)
    mec = int(np.argmax(energy))

    colors = [C_MEC if i == mec else C_COMBINED for i in range(n)]
    ax.bar(x, energy, color=colors, width=0.8, linewidth=0)
    ax.set_xlim(-0.5, n - 0.5)
    ax.tick_params(labelsize=7)
    ax.set_xlabel("Channel", fontsize=7)
    ax.set_ylabel("Σ Energy (active gestures)", fontsize=7)
    ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    ax.grid(axis="y", color=C_GRID, linewidth=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Combined energy profile", fontsize=8)
    ax.annotate(f"MEC = ch {mec}",
                xy=(mec, energy[mec]), xytext=(mec + max(1, n // 10), energy[mec] * 0.9),
                arrowprops=dict(arrowstyle="->", color=C_MEC, lw=1.2),
                fontsize=7, color=C_MEC)


def draw_sync_comparison(ax, current_energy: np.ndarray,
                         ref_energy: np.ndarray, sync_name: str):
    """Normalised current vs reference sync-gesture energy profiles."""
    cur_n = normalize_energy(angular_profile(current_energy))
    ref_n = normalize_energy(angular_profile(ref_energy))
    n = len(cur_n)
    x = np.arange(n)
    w = 0.38

    ax.bar(x - w / 2, ref_n, width=w, color=C_REF,     alpha=0.75, label="Reference", linewidth=0)
    ax.bar(x + w / 2, cur_n, width=w, color=C_CURRENT,  alpha=0.75, label="Current",   linewidth=0)
    ax.set_xlim(-0.5, n - 0.5)
    ax.tick_params(labelsize=7)
    ax.set_xlabel("Angular sector" if n < len(current_energy) else "Channel", fontsize=7)
    ax.set_ylabel("Normalised energy", fontsize=7)
    ax.grid(axis="y", color=C_GRID, linewidth=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=6, frameon=False)
    label = f'Sync gesture: "{sync_name}"'
    if len(current_energy) == 32:
        label += "\n(folded to 16 angular sectors)"
    ax.set_title(label, fontsize=8)


def draw_xcorr(ax, current_energy: np.ndarray, ref_energy: np.ndarray,
               rotation_offset: int, confidence: float):
    """Circular cross-correlation plot with winning lag marked."""
    cur_s = smooth_circular(normalize_energy(angular_profile(current_energy)))
    ref_s = smooth_circular(normalize_energy(angular_profile(ref_energy)))
    xc = xcorr_circular(cur_s, ref_s)
    n  = len(xc)
    lags = np.arange(n)

    ax.plot(lags, xc, color=C_XCORR, linewidth=1.4)
    ax.fill_between(lags, xc, alpha=0.15, color=C_XCORR)
    ax.axvline(rotation_offset, color=C_MEC, linewidth=1.8, linestyle="--",
               label=f"offset = {rotation_offset}")
    ax.scatter([rotation_offset], [xc[rotation_offset]],
               color=C_MEC, zorder=5, s=50)
    ax.set_xlim(0, n - 1)
    ax.tick_params(labelsize=7)
    ax.set_xlabel("Lag (channels)", fontsize=7)
    ax.set_ylabel("Cross-correlation", fontsize=7)
    ax.grid(color=C_GRID, linewidth=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=6, frameon=False)
    ax.set_title(f"Circular cross-correlation  (confidence {confidence:.1%})", fontsize=8)


def draw_ring_heatmap(ax, energy: np.ndarray, rotation_offset: int,
                      num_channels: int):
    """
    Two-part electrode layout visualisation.

    Top section — Flat unrolled band (primary view)
    -----------------------------------------------
    Matches the physical bracelet layout exactly:

        17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32   ← ch 16-31 (outer row)
        01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16   ← ch  0-15 (inner row)

    Each cell is coloured by its energy value.  The column index is the
    angular position around the forearm; both rows at the same column are
    at the same angular position (and so are summed when folding to a
    16-sector profile for offset detection).

    The reference column (col 0) is outlined in green.  If rotation_offset
    != 0, the detected shift column is outlined in red and a ← bracket
    shows the direction and magnitude of the correction.

    Bottom section — Circular ring view (rotation context)
    -------------------------------------------------------
    A tilt-projected ellipse showing both electrode rows as two concentric
    arcs.  The green REF marker and red arrow give an intuitive picture of
    how far the bracelet has rotated from the reference position.

    For non-32-ch devices a single flat row + single ring is shown instead.
    """
    ax.axis("off")

    n = len(energy)
    if n == 0:
        return

    cmap = plt.cm.plasma
    vmin, vmax = energy.min(), energy.max()
    norm_e = Normalize(vmin=vmin, vmax=vmax + 1e-12)

    # ── figure geometry: split ax into top (flat grid) / bottom (ring) ───────
    # We use ax.transAxes coordinates throughout so the two sections always
    # fill the available space regardless of figure size.
    ax_bbox   = ax.get_position()          # in figure coords — used for inset
    fig       = ax.get_figure()

    # Inset axes for the flat grid (top 58 % of this axes)
    ax_grid = ax.inset_axes([0.0, 0.40, 1.0, 0.58])
    ax_grid.axis("off")

    # Inset axes for the circular ring view (bottom 35 %)
    ax_ring = ax.inset_axes([0.05, 0.0, 0.90, 0.37])
    ax_ring.set_aspect("equal")
    ax_ring.axis("off")

    # ── 1. FLAT UNROLLED GRID ────────────────────────────────────────────────
    if num_channels == 32:
        n_cols = 16
        # Physical layout (1-indexed labels in the diagram):
        #   row 0 = outer row: channels 16-31 (0-indexed)
        #   row 1 = inner row: channels  0-15 (0-indexed)
        rows = [
            (list(range(16, 32)), "outer"),   # top row  → ch 16-31
            (list(range(0,  16)), "inner"),   # bottom row → ch 0-15
        ]
    else:
        n_cols = n
        rows = [(list(range(n)), "")]

    n_rows = len(rows)
    cell_w = 1.0 / n_cols
    cell_h = 1.0 / (n_rows + 0.6)   # leave 0.6-cell margin at top for title

    for row_idx, (ch_list, row_label) in enumerate(rows):
        # row 0 is at the top of the grid
        y_top = 1.0 - (row_idx + 0.55) * cell_h

        for col_idx, ch in enumerate(ch_list):
            x_left = col_idx * cell_w
            e_val  = float(energy[ch]) if ch < len(energy) else 0.0
            color  = cmap(norm_e(e_val))

            # Cell rectangle
            rect = mpatches.FancyBboxPatch(
                (x_left + 0.003, y_top - cell_h + 0.003),
                cell_w - 0.006, cell_h - 0.006,
                boxstyle="round,pad=0.005",
                facecolor=color, edgecolor="none",
                transform=ax_grid.transAxes, clip_on=False,
            )
            ax_grid.add_patch(rect)

            # Highlight reference column (col 0) in green
            if col_idx == 0:
                border = mpatches.FancyBboxPatch(
                    (x_left + 0.001, y_top - cell_h + 0.001),
                    cell_w - 0.002, cell_h - 0.002,
                    boxstyle="round,pad=0.002",
                    facecolor="none",
                    edgecolor=C_REF, linewidth=1.8,
                    transform=ax_grid.transAxes, clip_on=False,
                )
                ax_grid.add_patch(border)

            # Highlight rotated-to column in red
            if rotation_offset != 0 and col_idx == rotation_offset % n_cols:
                border2 = mpatches.FancyBboxPatch(
                    (x_left + 0.001, y_top - cell_h + 0.001),
                    cell_w - 0.002, cell_h - 0.002,
                    boxstyle="round,pad=0.002",
                    facecolor="none",
                    edgecolor=C_MEC, linewidth=1.8,
                    transform=ax_grid.transAxes, clip_on=False,
                )
                ax_grid.add_patch(border2)

            # Channel label (1-indexed to match the user-facing layout)
            label = str(ch + 1)
            lum   = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
            txt_c = "white" if lum < 0.55 else "#1f2937"
            ax_grid.text(
                x_left + cell_w / 2, y_top - cell_h / 2,
                label, ha="center", va="center",
                fontsize=max(4.5, 7 - n_cols // 8),
                color=txt_c, fontweight="bold",
                transform=ax_grid.transAxes,
            )

        # Row label on the left
        if row_label:
            ax_grid.text(
                -0.01, y_top - cell_h / 2, row_label,
                ha="right", va="center", fontsize=6,
                color="#6b7280", style="italic",
                transform=ax_grid.transAxes,
            )

    # Column-index tick labels below the bottom row
    y_tick = 1.0 - (n_rows + 0.55) * cell_h
    for col_idx in range(n_cols):
        if col_idx % max(1, n_cols // 8) == 0 or col_idx == rotation_offset % n_cols:
            ax_grid.text(
                (col_idx + 0.5) * cell_w, y_tick,
                str(col_idx), ha="center", va="top",
                fontsize=5.5, color="#6b7280",
                transform=ax_grid.transAxes,
            )

    # Rotation offset bracket annotation
    if rotation_offset != 0:
        col_ref = 0
        col_cur = rotation_offset % n_cols
        x_ref = (col_ref + 0.5) * cell_w
        x_cur = (col_cur + 0.5) * cell_w
        y_ann = 1.0 - 0.30 * cell_h          # above the top row
        # Bracket line
        ax_grid.annotate(
            "",
            xy=(x_cur, y_ann), xytext=(x_ref, y_ann),
            xycoords="axes fraction", textcoords="axes fraction",
            arrowprops=dict(arrowstyle="<->", color=C_MEC, lw=1.3),
        )
        ax_grid.text(
            (x_ref + x_cur) / 2, y_ann + 0.02,
            f"+{rotation_offset} col", ha="center", va="bottom",
            fontsize=6, color=C_MEC, fontweight="bold",
            transform=ax_grid.transAxes,
        )

    # Colourbar anchored to the right of the grid inset
    sm = ScalarMappable(cmap=cmap, norm=norm_e)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax_grid, fraction=0.06, pad=0.03,
                      aspect=12, location="right")
    cb.ax.tick_params(labelsize=5.5)
    cb.set_label("Energy", fontsize=6)

    ax_grid.set_title("Electrode energy  (unrolled bracelet view)",
                      fontsize=8, pad=4)

    # ── 2. CIRCULAR RING VIEW ─────────────────────────────────────────────────
    tilt  = 20.0
    cos_t = math.cos(math.radians(tilt))

    def ellipse_pt(col_idx, n_sectors, radius):
        """Angular position of column col_idx on a tilted ring."""
        theta = 2 * math.pi * col_idx / n_sectors - math.pi / 2
        return radius * math.cos(theta), radius * math.sin(theta) * cos_t

    n_sectors = 16 if num_channels == 32 else num_channels

    t_ell = np.linspace(0, 2 * math.pi, 300)

    if num_channels == 32:
        radii_rows = [
            (list(range(16, 32)), 0.90, "outer"),
            (list(range(0,  16)), 0.65, "inner"),
        ]
        outline_radii = [0.98, 0.73]
    else:
        radii_rows  = [(list(range(n)), 0.78, "")]
        outline_radii = [0.88]

    for r_outline in outline_radii:
        ax_ring.plot(
            r_outline * np.cos(t_ell),
            r_outline * np.sin(t_ell) * cos_t,
            color="#d1d5db", linewidth=0.8, zorder=0,
        )

    dot_r = 0.065
    for ch_list, r_ring, _ in radii_rows:
        for col_idx, ch in enumerate(ch_list):
            cx, cy = ellipse_pt(col_idx, n_sectors, r_ring)
            e_val  = float(energy[ch]) if ch < len(energy) else 0.0
            color  = cmap(norm_e(e_val))
            depth  = math.sin(2 * math.pi * col_idx / n_sectors - math.pi / 2)
            # Only label front-facing electrodes to avoid clutter
            circle = plt.Circle((cx, cy), dot_r * (0.7 + 0.5 * (depth + 1) / 2),
                                 color=color, zorder=3 + int(depth * 2))
            ax_ring.add_patch(circle)
            if depth > 0.15:
                ax_ring.text(cx, cy, str(ch + 1),
                             ha="center", va="center",
                             fontsize=max(3.5, 5.5 * (0.5 + 0.5 * depth)),
                             color="white", fontweight="bold",
                             zorder=5 + int(depth * 2))

    # REF marker
    r_mark = max(outline_radii) + 0.10
    rx, ry = ellipse_pt(0, n_sectors, r_mark)
    ax_ring.scatter([rx], [ry], s=40, color=C_REF, marker="^",
                    zorder=10, linewidths=0)
    ax_ring.text(rx, ry + 0.07, "REF", ha="center", va="bottom",
                 fontsize=5.5, color=C_REF, fontweight="bold")

    # Rotation arrow
    if rotation_offset != 0:
        cx, cy = ellipse_pt(rotation_offset % n_sectors, n_sectors, r_mark)
        ax_ring.annotate(
            "", xy=(cx, cy), xytext=(rx, ry),
            arrowprops=dict(arrowstyle="-|>", color=C_MEC, lw=1.4,
                            connectionstyle="arc3,rad=0.3"),
            zorder=11,
        )
        mx, my = ellipse_pt(rotation_offset / 2, n_sectors, r_mark + 0.18)
        ax_ring.text(mx, my, f"+{rotation_offset}",
                     ha="center", va="center", fontsize=6.5,
                     color=C_MEC, fontweight="bold",
                     bbox=dict(boxstyle="round,pad=0.15", fc="white",
                               ec=C_MEC, alpha=0.85))

    lim = r_mark + 0.28
    ax_ring.set_xlim(-lim, lim)
    ax_ring.set_ylim(-lim * 0.65, lim * 0.85)
    ax_ring.set_title("Circular view  (rotation context)",
                      fontsize=7, color="#6b7280", pad=2)


def draw_channel_mapping(ax, channel_mapping: list, num_channels: int,
                         rotation_offset: int):
    """
    Arrow diagram: raw channel index → remapped index.

    For large channel counts (> 16) only every other channel is labelled
    to keep the plot readable, and an arc arc is drawn instead of individual
    arrows.
    """
    ax.axis("off")
    ax.set_title("Channel mapping", fontsize=8)

    n = num_channels
    step = max(1, n // 16)          # label granularity
    labeled = list(range(0, n, step))

    x_left, x_right = 0.15, 0.85
    y_vals = {i: 1.0 - (i / (n - 1)) if n > 1 else 0.5 for i in range(n)}

    for raw in labeled:
        remapped = channel_mapping[raw] if raw < len(channel_mapping) else raw
        y0 = y_vals[raw]
        y1 = y_vals[remapped] if remapped < n else y_vals[raw]

        changed = (raw != remapped)
        color = C_MEC if changed else "#9ca3af"
        lw    = 1.2 if changed else 0.6

        ax.annotate("",
                    xy=(x_right, y1), xytext=(x_left, y0),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                   connectionstyle="arc3,rad=0.0"))
        ax.text(x_left - 0.04, y0, str(raw), ha="right", va="center",
                fontsize=6, color=C_TEXT)
        ax.text(x_right + 0.04, y1, str(remapped), ha="left", va="center",
                fontsize=6, color=color)

    ax.text(x_left,  1.04, "Raw ch",      ha="center", fontsize=7, fontweight="bold")
    ax.text(x_right, 1.04, "Remapped ch", ha="center", fontsize=7, fontweight="bold")

    if rotation_offset == 0:
        ax.text(0.5, -0.06, "No remapping (offset = 0)",
                ha="center", fontsize=7, color="#6b7280", style="italic",
                transform=ax.transAxes)


def draw_summary(ax, cal: dict, is_reference_file: bool):
    """Plain-text summary panel."""
    ax.axis("off")
    meta = cal.get("metadata", {})

    is_ref_session = meta.get("is_reference", False) or is_reference_file
    sync_gesture   = meta.get("sync_gesture", "n/a")
    mec_channel    = meta.get("mec_channel", "n/a")
    has_ref        = meta.get("has_reference", False)
    n_gestures     = meta.get("num_gestures", "n/a")
    per_conf       = meta.get("per_gesture_confidence", {})

    lines = [
        ("Device",          cal.get("device_name", "unknown")),
        ("Recorded",        cal["created_at"].strftime("%Y-%m-%d  %H:%M:%S")),
        ("Channels",        str(cal["num_channels"])),
        ("",                ""),
        ("Rotation offset", f"{cal['rotation_offset']} channels"),
        ("Confidence",      f"{cal['confidence']:.1%}"),
        ("Sync gesture",    sync_gesture),
        ("MEC channel",     str(mec_channel)),
        ("Gestures used",   str(n_gestures)),
        ("Had reference",   "yes" if has_ref else "no  (this is the reference)"),
        ("",                ""),
        ("Per-gesture confidence", ""),
    ]

    for g, c in sorted(per_conf.items()):
        if g != "__combined__":
            lines.append((f"  {g}", f"{c:.1%}"))

    x_label, x_value = 0.02, 0.52
    y = 0.98
    dy = 0.068

    if is_ref_session:
        ax.add_patch(mpatches.FancyBboxPatch(
            (0, 0.93), 1.0, 0.07, boxstyle="round,pad=0.01",
            facecolor="#dcfce7", edgecolor=C_REF, linewidth=1.2,
            transform=ax.transAxes, clip_on=False))
        ax.text(0.5, 0.965, "★  REFERENCE SESSION",
                ha="center", va="center", fontsize=8,
                color=C_REF, fontweight="bold", transform=ax.transAxes)

    for label, value in lines:
        if label == "":
            y -= dy * 0.4
            continue
        bold = label in ("Rotation offset", "Confidence", "Sync gesture")
        ax.text(x_label, y, label + (":" if label else ""),
                ha="left", va="top", fontsize=7,
                fontweight="bold" if bold else "normal",
                color=C_TEXT, transform=ax.transAxes)
        if value:
            color = C_MEC if label == "Rotation offset" and cal["rotation_offset"] != 0 else C_TEXT
            ax.text(x_value, y, value,
                    ha="left", va="top", fontsize=7, color=color,
                    transform=ax.transAxes)
        y -= dy
        if y < 0.0:
            break


# ── main figure builder ───────────────────────────────────────────────────────

def build_figure(cal: dict, ref_cal: dict | None, is_reference_file: bool) -> plt.Figure:
    patterns    = cal["reference_patterns"]
    n_ch        = cal["num_channels"]
    offset      = cal["rotation_offset"]
    confidence  = cal["confidence"]
    sync_name   = cal.get("metadata", {}).get("sync_gesture", "")
    mapping     = cal["channel_mapping"]

    # Identify gesture sub-panels (exclude the __combined__ pseudo-key)
    gesture_keys = [k for k in patterns if k != "__combined__"]
    n_gestures   = len(gesture_keys)

    # ── figure layout ────────────────────────────────────────────────────────
    # Row 0: gesture bars (up to 8, wrapped to two sub-rows if needed)
    # Row 1: combined | sync comparison | xcorr
    # Row 2: ring heatmap | channel mapping | summary

    cols_per_row = min(n_gestures, 4)
    gesture_rows = math.ceil(n_gestures / cols_per_row) if n_gestures else 1

    fig = plt.figure(figsize=(16, 5 + gesture_rows * 2.2), constrained_layout=False)
    fig.patch.set_facecolor("white")

    outer = gridspec.GridSpec(
        3, 1, figure=fig,
        hspace=0.45,
        height_ratios=[gesture_rows * 2.2, 2.8, 3.2],
        left=0.06, right=0.97, top=0.93, bottom=0.05,
    )

    # ── Row 0: per-gesture bars ───────────────────────────────────────────────
    inner_g = gridspec.GridSpecFromSubplotSpec(
        gesture_rows, cols_per_row, subplot_spec=outer[0],
        hspace=0.55, wspace=0.35,
    )

    for idx, gname in enumerate(gesture_keys):
        r, c = divmod(idx, cols_per_row)
        ax = fig.add_subplot(inner_g[r, c])
        is_sync = sync_name and (sync_name.lower() in gname.lower() or
                                  gname.lower() in sync_name.lower())
        draw_gesture_bars(ax, patterns[gname], gname, bool(is_sync), is_reference_file)

    # Hide unused gesture cells
    for idx in range(n_gestures, gesture_rows * cols_per_row):
        r, c = divmod(idx, cols_per_row)
        fig.add_subplot(inner_g[r, c]).set_visible(False)

    # ── Row 1: combined | sync comparison | xcorr ────────────────────────────
    inner_m = gridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=outer[1], wspace=0.38,
    )

    ax_combined = fig.add_subplot(inner_m[0, 0])
    combined    = patterns.get("__combined__", np.zeros(n_ch))
    draw_combined(ax_combined, combined, n_ch)

    # Sync comparison and xcorr only make sense when a reference exists
    has_ref   = ref_cal is not None and not is_reference_file
    ref_pats  = ref_cal["reference_patterns"] if has_ref else None

    # Find the reference-side sync energy pattern
    ref_sync_energy = None
    if has_ref and ref_pats:
        # Mirror the calibrator's _select_sync_pattern substring logic
        priority = ["waveout", "fist", "open", "wavein", "pinch", "tripod"]
        for token in priority:
            for rk in ref_pats:
                if token in rk.lower() and rk != "__combined__":
                    ref_sync_energy = ref_pats[rk]
                    break
            if ref_sync_energy is not None:
                break
        if ref_sync_energy is None:
            ref_sync_energy = ref_pats.get("__combined__")

    # Current sync energy
    cur_sync_energy = None
    if sync_name:
        for gk in patterns:
            if (sync_name.lower() in gk.lower() or gk.lower() in sync_name.lower()) \
                    and gk != "__combined__":
                cur_sync_energy = patterns[gk]
                break
    if cur_sync_energy is None:
        cur_sync_energy = combined

    ax_sync = fig.add_subplot(inner_m[0, 1])
    if has_ref and ref_sync_energy is not None:
        draw_sync_comparison(ax_sync, cur_sync_energy, ref_sync_energy, sync_name)
    else:
        ax_sync.axis("off")
        msg = ("This session is the reference.\nNo comparison available."
               if is_reference_file else
               "No reference calibration found.")
        ax_sync.text(0.5, 0.5, msg, ha="center", va="center",
                     fontsize=9, color="#6b7280", style="italic",
                     transform=ax_sync.transAxes)
        ax_sync.set_title("Sync-gesture comparison", fontsize=8)

    ax_xcorr = fig.add_subplot(inner_m[0, 2])
    if has_ref and ref_sync_energy is not None:
        draw_xcorr(ax_xcorr, cur_sync_energy, ref_sync_energy, offset, confidence)
    else:
        ax_xcorr.axis("off")
        ax_xcorr.text(0.5, 0.5, "—", ha="center", va="center",
                      fontsize=14, color="#d1d5db",
                      transform=ax_xcorr.transAxes)
        ax_xcorr.set_title("Cross-correlation", fontsize=8)

    # ── Row 2: ring heatmap | channel mapping | summary ───────────────────────
    inner_b = gridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=outer[2], wspace=0.30,
    )

    ax_ring = fig.add_subplot(inner_b[0, 0])
    draw_ring_heatmap(ax_ring, combined, offset, n_ch)

    ax_map = fig.add_subplot(inner_b[0, 1])
    draw_channel_mapping(ax_map, mapping, n_ch, offset)

    ax_sum = fig.add_subplot(inner_b[0, 2])
    draw_summary(ax_sum, cal, is_reference_file)

    # ── super-title ───────────────────────────────────────────────────────────
    ts   = cal["created_at"].strftime("%Y-%m-%d %H:%M:%S")
    tag  = "  ★ REFERENCE" if is_reference_file else f"  offset={offset}  conf={confidence:.1%}"
    fig.suptitle(
        f"Calibration  —  {cal.get('device_name', 'unknown')}  —  {ts}{tag}",
        fontsize=11, fontweight="bold", color=C_TEXT, y=0.975,
    )

    return fig


# ── CLI entry point ────────────────────────────────────────────────────────────

def main():
    default_cal_dir = Path(__file__).resolve().parent.parent / "calibrations"

    parser = argparse.ArgumentParser(
        description="Plot diagnostic figures for all calibration JSON files."
    )
    parser.add_argument(
        "cal_dir", nargs="?", default=None,
        help=(
            "Directory containing calibration JSON files "
            f"(default: {default_cal_dir})"
        ),
    )
    parser.add_argument(
        "--out", default=None,
        help="Output directory for PNG files (default: <cal_dir>/plots/)",
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Display each figure interactively instead of saving",
    )
    parser.add_argument(
        "--dpi", type=int, default=150,
        help="Output DPI (default: 150)",
    )
    args = parser.parse_args()

    cal_dir = Path(args.cal_dir).expanduser() if args.cal_dir else default_cal_dir
    if not cal_dir.exists():
        sys.exit(f"Error: calibration directory not found: {cal_dir}")

    out_dir = Path(args.out) if args.out else cal_dir / "plots"
    if not args.show:
        out_dir.mkdir(parents=True, exist_ok=True)

    # Load reference calibration files (legacy + mode-specific)
    ref_files = sorted(cal_dir.glob("reference_calibration*.json"))
    ref_by_mode = {}
    ref_loaded = {}
    for ref_path in ref_files:
        try:
            ref_cal = load_calibration(ref_path)
            mode = str((ref_cal.get("metadata") or {}).get("signal_mode", "monopolar")).lower()
            ref_by_mode[mode] = ref_cal
            ref_loaded[ref_path] = ref_cal
            print(f"Loaded reference calibration: {ref_path}")
        except Exception as e:
            print(f"Warning: could not load reference calibration {ref_path.name}: {e}")

    # Collect all calibration JSON files (legacy and session-based naming).
    json_files = [p for p in cal_dir.glob("*.json") if not p.name.startswith("reference_calibration")]
    loaded = []
    for path in json_files:
        try:
            cal = load_calibration(path)
            loaded.append((path, cal))
        except Exception as e:
            print(f"  Skipping {path.name}: {e}")

    loaded.sort(key=lambda item: item[1].get("created_at", datetime.min))
    for ref_path in ref_files:
        ref_cal = ref_loaded.get(ref_path)
        if ref_cal is not None:
            loaded.append((ref_path, ref_cal))

    if not loaded:
        sys.exit(f"No calibration JSON files found in {cal_dir}")

    print(f"Found {len(loaded)} calibration file(s).")

    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "axes.titlepad": 6,
    })

    for path, cal in loaded:

        is_ref_file = path.name.startswith("reference_calibration")
        mode = str((cal.get("metadata") or {}).get("signal_mode", "monopolar")).lower()
        ref_cal = ref_by_mode.get(mode)
        print(f"  Plotting {path.name}  "
              f"(offset={cal['rotation_offset']}, conf={cal['confidence']:.1%})"
              + ("  [REFERENCE]" if is_ref_file else ""))

        fig = build_figure(cal, ref_cal, is_ref_file)

        if args.show:
            plt.show()
        else:
            out_path = out_dir / f"{calibration_output_stem(path, cal)}.png"
            fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight",
                        facecolor="white")
            print(f"    → saved {out_path}")

        plt.close(fig)

    if not args.show:
        print(f"\nAll plots saved to {out_dir}/")


if __name__ == "__main__":
    main()
