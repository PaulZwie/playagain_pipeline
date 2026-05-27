"""
validation/pipeline_figures.py
──────────────────────────────
Pipeline / system-structure illustrations for the PlayAgain thesis.

Six diagrams, all rendered with matplotlib so they ship as crisp
vector PDF (+PNG for slides) and stay reproducible from source:

* ``draw_system_architecture``   — four-layer system overview (§4.1)
* ``draw_recording_protocol``    — standard + training-game session
                                   timeline (§5.2)
* ``draw_communication_protocol``— Python ↔ Unity sequence diagram (§5.3)
* ``draw_cv_strategies``         — LOSO-session / LOSO-subject /
                                   cross-domain visualisation (§5.4)
* ``draw_ml_pipeline``           — raw EMG → features → model →
                                   smoother → prediction flow (§4.3)
* ``draw_sliding_window``        — 200 ms window / 50 ms stride
                                   visual (§4.3.2)

CLI
───
    python -m playagain_pipeline.validation.pipeline_figures \\
        --out figures/

Renders all six into ``figures/`` as ``.pdf`` and ``.png``.

Design notes
────────────
* Palette: FAU 2024 (via ``fau_colors``); falls back to hard-coded
  hexes if the package isn't installed.
* Each layer / lane has a single hue with a darker accent for its
  prominent boxes. Sub-components are filled boxes labelled with a
  title plus an optional short subtitle.
* Spines are off everywhere; figures rely on whitespace and weight
  contrast for hierarchy rather than borders. This keeps them at home
  in a Springer-style thesis layout.
"""
from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt           # noqa: E402
from matplotlib.patches import (          # noqa: E402
    FancyArrowPatch, FancyBboxPatch, Rectangle,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FAU palette — same fallback contract as plots_thesis.py
# ---------------------------------------------------------------------------

try:
    from fau_colors import colors as _FAU, colors_dark as _FAU_D
    PALETTE = {
        "fau":       _FAU.fau,        "tech":      _FAU.tech,
        "phil":      _FAU.phil,       "med":       _FAU.med,
        "nat":       _FAU.nat,        "wiso":      _FAU.wiso,
        "fau_d":     _FAU_D.fau,      "tech_d":    _FAU_D.tech,
        "phil_d":    _FAU_D.phil,     "med_d":     _FAU_D.med,
        "nat_d":     _FAU_D.nat,      "wiso_d":    _FAU_D.wiso,
    }
    HAS_FAU = True
except Exception:                       # noqa: BLE001
    PALETTE = {
        "fau":    "#04316A", "tech":   "#8C9FB1", "phil":   "#FDB735",
        "med":    "#18B4F1", "nat":    "#7BB725", "wiso":   "#C50F3C",
        "fau_d":  "#041E42", "tech_d": "#2F586E", "phil_d": "#E87722",
        "med_d":  "#005287", "nat_d":  "#266141", "wiso_d": "#971B2F",
    }
    HAS_FAU = False

NEUTRAL_TEXT  = "#1f2937"
NEUTRAL_MUTED = "#6b7280"
NEUTRAL_GRID  = "#e2e8f0"


def _setup_mpl() -> None:
    matplotlib.rcParams.update({
        "font.family":          "sans-serif",
        "font.sans-serif":      ["DejaVu Sans", "Arial"],
        "font.size":            10,
        "pdf.fonttype":         42,
        "ps.fonttype":          42,
        "savefig.bbox":         "tight",
        "savefig.facecolor":    "white",
    })


def _save(fig: plt.Figure, out_path: Path) -> List[Path]:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    for ext in (".pdf", ".png"):
        p = out_path.with_suffix(ext)
        fig.savefig(p, dpi=300)
        written.append(p)
    plt.close(fig)
    return written


def _hex_to_rgb(hexstr: str) -> Tuple[float, float, float]:
    h = hexstr.lstrip("#")
    return tuple(int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4))


def _tint(color: str, alpha: float) -> Tuple[float, float, float, float]:
    """Return colour with given alpha — used for layer backgrounds."""
    r, g, b = _hex_to_rgb(color)
    return (r, g, b, alpha)


def _text_color_for(bg_hex: str) -> str:
    """Pick black or white text against a fill for readable contrast."""
    r, g, b = _hex_to_rgb(bg_hex)
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return "white" if luminance < 0.55 else NEUTRAL_TEXT


# ---------------------------------------------------------------------------
# Building blocks — reusable across all six diagrams
# ---------------------------------------------------------------------------

def _box(
    ax, x: float, y: float, w: float, h: float,
    title: str, *, subtitle: Optional[str] = None,
    fill: str = "#ffffff", border: str = NEUTRAL_TEXT,
    text_color: Optional[str] = None,
    radius: float = 0.05, title_size: int = 10,
    subtitle_size: int = 8, lw: float = 1.0,
    title_weight: str = "semibold",
) -> None:
    """Draw a rounded box with a bold title and optional subtitle below."""
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=fill, edgecolor=border, linewidth=lw,
    )
    ax.add_patch(patch)
    tc = text_color if text_color else _text_color_for(fill)
    if subtitle:
        ax.text(x + w / 2, y + h * 0.62, title,
                ha="center", va="center",
                fontsize=title_size, fontweight=title_weight, color=tc)
        ax.text(x + w / 2, y + h * 0.30, subtitle,
                ha="center", va="center",
                fontsize=subtitle_size, color=tc, alpha=0.92)
    else:
        ax.text(x + w / 2, y + h / 2, title,
                ha="center", va="center",
                fontsize=title_size, fontweight=title_weight, color=tc)


def _arrow(ax, x1, y1, x2, y2, *,
           color: str = NEUTRAL_TEXT, lw: float = 1.4,
           style: str = "-|>", mut_scale: float = 14,
           ls: str = "-") -> None:
    arr = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style, color=color, linewidth=lw,
        mutation_scale=mut_scale, linestyle=ls,
        shrinkA=0, shrinkB=0,
    )
    ax.add_patch(arr)


# ===========================================================================
# 1. System architecture (§4.1)
# ===========================================================================

def draw_system_architecture(out_path: Path) -> List[Path]:
    """Four-layer system overview with the TCP channel between game and rest."""
    _setup_mpl()
    fig, ax = plt.subplots(figsize=(14.0, 8.0))

    # Layer definitions (top to bottom in reading order, but we draw the
    # acquisition layer at the top of the canvas — y increases upward).
    layers = [
        dict(
            key="game", color=PALETTE["nat"], color_d=PALETTE["nat_d"],
            title="Game & Interaction Layer",
            subtitle="Unity — therapeutic exergame, participant profiles, "
                     "game-state recording",
            children=[
                ("Fox Adventure",     "Exergame"),
                ("Participant",       "Profiles"),
                ("Game-State",        "Recorder"),
                ("Difficulty",        "Calibration"),
            ],
        ),
        dict(
            key="val", color=PALETTE["wiso_d"], color_d=PALETTE["wiso_d"],
            title="Validation & Experiment Layer",
            subtitle="Reproducible cross-validation · leakage-free splits · "
                     "experiment persistence",
            children=[
                ("Corpus",         "Discovery"),
                ("CV Strategies",  "(Within · LOSO · Cross-domain)"),
                ("Experiment",     "Runner"),
                ("Result",         "Persistence"),
            ],
        ),
        dict(
            key="ml", color=PALETTE["fau"], color_d=PALETTE["fau_d"],
            title="Machine Learning Layer",
            subtitle="Calibration · feature extraction · model training · "
                     "real-time inference",
            children=[
                ("Bracelet",          "Calibration"),
                ("Feature Extraction","(8 × 32 = 256)"),
                ("Classifiers",       "LDA · SVM · RF\nCatBoost · CNN"),
                ("Prediction Smoother","(EMA + gate)"),
                ("Real-Time",         "Inference Server"),
            ],
        ),
        dict(
            key="sig", color=PALETTE["phil_d"], color_d=PALETTE["phil_d"],
            title="Signal Acquisition Layer",
            subtitle="OTBioelettronica Muovi · 32-channel monopolar sEMG · "
                     "2 kHz · synthetic generator",
            children=[
                ("Muovi Bracelet", "32 ch · 2 kHz"),
                ("Synthetic",      "Signal Generator"),
                ("Device",         "Abstraction"),
            ],
        ),
    ]

    # Canvas coordinates — single-page, tight layout
    x0, x1 = 0.3, 13.7
    layer_height = 1.95          # taller bands → room for header + child row
    gap = 0.36
    n = len(layers)
    total_h = n * layer_height + (n - 1) * gap
    y_top = 7.55                 # top edge of topmost layer band
    y_bottom = y_top - total_h   # bottom edge of bottommost layer band

    child_h = 0.92

    for i, layer in enumerate(layers):
        # Layer band at y_band (bottom of this layer)
        y_band = y_top - i * (layer_height + gap) - layer_height
        # Background tint
        bg = _tint(layer["color"], 0.15)
        ax.add_patch(Rectangle(
            (x0, y_band), x1 - x0, layer_height,
            facecolor=bg, edgecolor=layer["color"], linewidth=1.2,
        ))
        # Header text — top portion of the band
        ax.text(x0 + 0.22, y_band + layer_height - 0.20, layer["title"],
                ha="left", va="top", fontsize=12.5, fontweight="bold",
                color=layer["color_d"])
        ax.text(x0 + 0.22, y_band + layer_height - 0.55, layer["subtitle"],
                ha="left", va="top", fontsize=9, color=NEUTRAL_MUTED,
                style="italic")

        # Children — evenly spaced row in the lower portion of the band,
        # clear of the subtitle line above.
        k = len(layer["children"])
        avail_w = (x1 - x0) - 0.60
        child_w = (avail_w - 0.26 * (k - 1)) / k
        for j, (title, sub) in enumerate(layer["children"]):
            cx = x0 + 0.30 + j * (child_w + 0.26)
            cy = y_band + 0.18
            _box(ax, cx, cy, child_w, child_h,
                 title=title, subtitle=sub,
                 fill=layer["color"], border=layer["color_d"],
                 radius=0.07, title_size=10.5, subtitle_size=8.5)

    # Stack-arrows on the LEFT side: signal flows upward from acquisition
    # (bottom) to game (top). One arrow per inter-layer gap.
    arrow_x = x0 - 0.18
    for i in range(n - 1):
        y_upper = y_top - i * (layer_height + gap) - layer_height
        y_lower = y_top - (i + 1) * (layer_height + gap)
        _arrow(ax, arrow_x, y_lower - 0.02, arrow_x, y_upper + 0.02,
               color=NEUTRAL_MUTED, lw=1.2, mut_scale=12)

    # TCP communication channel bridges Game ↔ ML/Validation: place it in
    # the gap between the Game layer and the Validation layer below it.
    y_game_bot = y_top - layer_height                       # bottom of Game band
    y_val_top  = y_top - layer_height - gap                 # top of Validation band
    gap_center_y = (y_game_bot + y_val_top) / 2
    pill_cx = (x0 + x1) / 2
    pill_w, pill_h = 3.0, 0.34
    _box(ax, pill_cx - pill_w / 2, gap_center_y - pill_h / 2, pill_w, pill_h,
         title="TCP Communication Channel",
         fill=PALETTE["phil"], border=PALETTE["phil_d"],
         radius=0.17, title_size=9.5, title_weight="bold")

    # Legend — top-right corner, *inside* the figure area (just above the
    # top layer). Order matches the reading order of the diagram from top.
    legend_items = [
        ("Game & Interaction Layer",       PALETTE["nat"]),
        ("Validation & Experiment Layer",  PALETTE["wiso_d"]),
        ("Machine Learning Layer",         PALETTE["fau"]),
        ("Signal Acquisition Layer",       PALETTE["phil_d"]),
    ]
    leg_w, leg_row_h = 3.05, 0.24
    leg_h = leg_row_h * len(legend_items) + 0.20
    lx = x1 - leg_w
    ly = y_top + 0.15            # tight above the top band
    ax.add_patch(Rectangle((lx, ly), leg_w, leg_h,
                           facecolor="white", edgecolor=NEUTRAL_GRID,
                           linewidth=0.8, zorder=10))
    for i, (label, c) in enumerate(legend_items):
        yy = ly + leg_h - 0.16 - (i + 1) * leg_row_h + 0.06
        ax.add_patch(Rectangle((lx + 0.14, yy), 0.22, 0.15,
                               facecolor=c, edgecolor=NEUTRAL_TEXT,
                               linewidth=0.6, zorder=11))
        ax.text(lx + 0.44, yy + 0.075, label, ha="left", va="center",
                fontsize=8.5, color=NEUTRAL_TEXT, zorder=11)

    ax.set_xlim(-0.5, x1 + 0.4)
    ax.set_ylim(y_bottom - 0.30, ly + leg_h + 0.35)
    ax.set_aspect("auto")
    ax.set_axis_off()
    return _save(fig, out_path)


# ===========================================================================
# 2. Recording protocol (§5.2)
# ===========================================================================

def draw_recording_protocol(out_path: Path) -> List[Path]:
    """Two-panel figure: (a) standard recording, (b) training-game loop."""
    _setup_mpl()
    fig = plt.figure(figsize=(13.5, 7.5))
    # 2 rows: (a) is short, (b) is taller (has three lanes)
    gs = fig.add_gridspec(2, 1, height_ratios=[0.85, 2.0], hspace=0.10,
                          left=0.04, right=0.985, top=0.96, bottom=0.04)
    ax_a = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1])

    # ─── (a) Standard session ─────────────────────────────────────────
    ax_a.set_xlim(0, 14.5)
    ax_a.set_ylim(-0.1, 2.0)
    ax_a.set_aspect("equal")
    ax_a.set_axis_off()
    ax_a.text(0.05, 1.95, "(a)  Standard Recording Session Structure",
              fontsize=12, fontweight="bold", color=NEUTRAL_TEXT,
              ha="left", va="top")

    sequence = [
        ("Wave-Out Calibration", "(sync gesture · not a class)",
                                      PALETTE["phil_d"], 1.95),
        ("Rest",   None,     PALETTE["tech"], 0.85),
        ("Fist",   "[hold]", PALETTE["fau"],  1.10),
        ("Rest",   None,     PALETTE["tech"], 0.85),
        ("Pinch",  "[hold]", PALETTE["fau"],  1.10),
        ("Rest",   None,     PALETTE["tech"], 0.85),
        ("Tripod", "[hold]", PALETTE["fau"],  1.10),
        ("Rest",   None,     PALETTE["tech"], 0.85),
        ("… (N repetitions, pseudo-randomised order)", None,
                                      PALETTE["tech"], 3.55),
    ]
    x = 0.20
    y_pill = 0.78
    h_pill = 0.90
    for title, sub, color, w in sequence:
        _box(ax_a, x, y_pill, w, h_pill,
             title=title, subtitle=sub,
             fill=color, border=color, radius=0.30,
             title_size=9, subtitle_size=7.5, title_weight="bold")
        x += w + 0.08

    # Timeline axis below
    ax_a.annotate("", xy=(13.2, 0.40), xytext=(0.20, 0.40),
                  arrowprops=dict(arrowstyle="->", color=NEUTRAL_MUTED, lw=1.0))
    ax_a.text(13.35, 0.40, "Time →", ha="left", va="center",
              fontsize=9, color=NEUTRAL_MUTED)
    # Calibration callout points up to the wave-out pill
    cal_cx = 0.20 + 1.95 / 2
    ax_a.annotate("calibration only\n(not in dataset)",
                  xy=(cal_cx, y_pill - 0.02), xytext=(cal_cx, -0.05),
                  ha="center", va="bottom", fontsize=7.5,
                  color=PALETTE["phil_d"],
                  arrowprops=dict(arrowstyle="-",
                                  color=PALETTE["phil_d"], lw=0.8))

    # ─── (b) Training-game session: three lanes ─────────────────────
    ax_b.set_xlim(0, 14.5)
    ax_b.set_ylim(0, 6.2)
    ax_b.set_aspect("equal")
    ax_b.set_axis_off()
    ax_b.text(0.05, 6.10, "(b)  Training-Game Session Structure",
              fontsize=12, fontweight="bold", color=NEUTRAL_TEXT,
              ha="left", va="top")

    lanes = [
        ("Pipeline", PALETTE["fau"],     PALETTE["fau_d"],     4.10),
        ("TCP",      PALETTE["phil"],    PALETTE["phil_d"],    2.60),
        ("Game",     PALETTE["nat"],     PALETTE["nat_d"],     1.10),
    ]
    lane_h = 1.20
    x_lane_start = 1.55             # more left margin for the lane labels
    x_lane_end   = 14.30
    for label, color, color_d, y in lanes:
        ax_b.add_patch(Rectangle((x_lane_start, y),
                                 x_lane_end - x_lane_start, lane_h,
                                 facecolor=_tint(color, 0.10),
                                 edgecolor=color, linewidth=1.1))
        ax_b.text(x_lane_start - 0.15, y + lane_h / 2, label,
                  ha="right", va="center", fontsize=10.5,
                  fontweight="bold", color=color_d)

    # "Recording begins" — sits BEFORE the Pipeline boxes start, on the
    # left edge of the second column, so it never overlaps box text.
    rec_x = 3.05
    ax_b.plot([rec_x, rec_x], [1.10, 5.65], color=PALETTE["fau"],
              linestyle="--", linewidth=1.0, alpha=0.7)
    ax_b.text(rec_x, 5.75, "Recording begins", ha="center", va="bottom",
              fontsize=9, color=PALETTE["fau"], fontweight="semibold")

    # Pipeline lane content
    pipeline_steps = [
        ("Wait for level",    "(game starts)",       2.30, 1.55),
        ("Schedule next",     "target gesture",      4.05, 1.65),
        ("Monitor EMG",       "(easy mode)",         5.90, 1.65),
        ("Inject synthetic",  "prediction",          7.75, 1.65),
        ("Advance to",        "next trial",          9.60, 1.65),
        ("… (repeat for all trials) …", None,        11.85, 2.40),
        ("Stop &",            "save dataset",        13.90, 1.50),
    ]
    y_pipe = 4.30
    for title, sub, cx, w in pipeline_steps:
        _box(ax_b, cx - w / 2, y_pipe, w, 0.85,
             title=title, subtitle=sub,
             fill=PALETTE["fau"], border=PALETTE["fau_d"],
             radius=0.10, title_size=9, subtitle_size=7.5,
             title_weight="bold")

    # TCP message pills
    tcp_msgs = [
        ("← game_level_started", 4.05),
        ("target_gesture →",     5.90),
        ("synth. predict →",     7.75),
        ("← game_state_update",  9.60),
    ]
    y_tcp = 2.85
    for title, cx in tcp_msgs:
        _box(ax_b, cx - 0.92, y_tcp, 1.84, 0.60,
             title=title,
             fill=PALETTE["phil"], border=PALETTE["phil_d"],
             radius=0.12, title_size=8.5,
             title_weight="bold",
             text_color=NEUTRAL_TEXT)

    # Game lane content
    game_steps = [
        ("Loading…", None,                       2.30, 1.55),
        ("Auto-spawn animal",  "(mapped from cue)", 4.05, 1.95),
        ("Child performs gesture",
         "+ game renders feedback",               6.55, 2.65),
        ("Animal walks",      "away",             9.60, 1.65),
        ("… (next animals) …", None,             12.10, 2.55),
    ]
    y_game = 1.35
    for title, sub, cx, w in game_steps:
        _box(ax_b, cx - w / 2, y_game, w, 0.85,
             title=title, subtitle=sub,
             fill=PALETTE["nat"], border=PALETTE["nat_d"],
             radius=0.10, title_size=9, subtitle_size=7.5,
             title_weight="bold")

    # Vertical alignment guides between lanes — only over the inner four
    # columns where all three lanes have aligned content.
    guide_x = [4.05, 5.90, 7.75, 9.60]
    for gx in guide_x:
        for y_seg in [(1.75 + 0.85, 2.85), (2.85 + 0.60, 4.30)]:
            ax_b.plot([gx, gx], list(y_seg),
                      color=NEUTRAL_MUTED, linestyle=":", linewidth=0.6,
                      alpha=0.7)

    # Bottom time arrow
    ax_b.annotate("", xy=(14.20, 0.55), xytext=(0.50, 0.55),
                  arrowprops=dict(arrowstyle="->", color=NEUTRAL_MUTED, lw=1.0))
    ax_b.text(14.35, 0.55, "Time →", ha="left", va="center",
              fontsize=9, color=NEUTRAL_MUTED)

    return _save(fig, out_path)


# ===========================================================================
# 3. Communication protocol (§5.3)
# ===========================================================================

def draw_communication_protocol(out_path: Path) -> List[Path]:
    """Python ↔ Unity sequence diagram."""
    _setup_mpl()
    fig, ax = plt.subplots(figsize=(12.5, 8.0))

    # Three vertical lanes — Python | TCP | Unity
    lane_w = 3.0
    gap_w  = 3.4
    x_py   = 0.5
    x_tcp  = x_py + lane_w + 0.6
    x_unity = x_tcp + gap_w + 0.6
    y_top, y_bot = 8.4, 0.5

    def lane(x: float, color: str, color_d: str, title: str,
             style: str = "solid") -> None:
        if style == "solid":
            ax.add_patch(Rectangle((x, y_bot), lane_w, y_top - y_bot,
                                   facecolor=_tint(color, 0.12),
                                   edgecolor=color, linewidth=1.1))
        else:  # dashed for the TCP gap
            ax.add_patch(Rectangle((x, y_bot), gap_w, y_top - y_bot,
                                   facecolor=_tint(color, 0.10),
                                   edgecolor=color, linewidth=1.0,
                                   linestyle=(0, (4, 3))))
        w = lane_w if style == "solid" else gap_w
        ax.text(x + w / 2, y_top - 0.30, title,
                ha="center", va="center", fontsize=13,
                fontweight="bold", color=color_d)

    lane(x_py,    PALETTE["fau"],   PALETTE["fau_d"],   "Python Pipeline")
    lane(x_tcp,   PALETTE["phil"],  PALETTE["phil_d"],  "TCP Channel", style="dashed")
    lane(x_unity, PALETTE["nat"],   PALETTE["nat_d"],   "Unity Game")

    messages = [
        ("Session Handshake",  "(gesture vocabulary)",
         "one-time, on connect", "py_to_unity"),
        ("game_level_started", "+ session_config",
         "game confirms level loaded", "unity_to_py"),
        ("target_gesture",     "(next gesture to cue)",
         "scheduler picks next trial", "py_to_unity"),
        ("Gesture Prediction", "(class + probability)",
         "continuous during inference", "py_to_unity"),
        ("game_state_update",  "(ground-truth label)",
         "on every gameplay event", "unity_to_py"),
        ("Synthetic Prediction", "(easy-mode injection)",
         "on threshold crossing", "py_to_unity"),
    ]
    n = len(messages)
    # Evenly distribute vertical positions
    avail = (y_top - 0.60) - (y_bot + 0.30)
    step = avail / n
    box_h = 0.62
    box_w = lane_w - 0.40

    for i, (title, sub, note, direction) in enumerate(messages):
        y_center = y_top - 0.95 - i * step
        py_active = direction == "py_to_unity"
        unity_active = direction == "unity_to_py"
        _box(ax, x_py + 0.20, y_center - box_h / 2, box_w, box_h,
             title=title, subtitle=sub,
             fill=PALETTE["fau"] if py_active else _tint(PALETTE["tech"], 0.65),
             border=PALETTE["fau_d"] if py_active else PALETTE["tech_d"],
             radius=0.10, title_size=9.5, subtitle_size=7.5,
             title_weight="bold",
             text_color="white" if py_active else NEUTRAL_TEXT)
        _box(ax, x_unity + 0.20, y_center - box_h / 2, box_w, box_h,
             title=title, subtitle=sub,
             fill=PALETTE["nat_d"] if unity_active else _tint(PALETTE["tech"], 0.65),
             border=PALETTE["nat_d"] if unity_active else PALETTE["tech_d"],
             radius=0.10, title_size=9.5, subtitle_size=7.5,
             title_weight="bold",
             text_color="white" if unity_active else NEUTRAL_TEXT)

        # Arrow on TCP channel
        if direction == "py_to_unity":
            x_from, x_to = x_py + lane_w, x_unity
            arrow_color = PALETTE["fau_d"]
        else:
            x_from, x_to = x_unity, x_py + lane_w
            arrow_color = PALETTE["nat_d"]
        _arrow(ax, x_from, y_center, x_to, y_center,
               color=arrow_color, lw=1.6, mut_scale=16)
        # Note
        ax.text((x_tcp + x_tcp + gap_w) / 2, y_center - 0.30, note,
                ha="center", va="top", fontsize=8.5, style="italic",
                color=PALETTE["phil_d"])

    # Footnote — sits directly under the lanes, kept close so it doesn't
    # orphan at the page bottom.
    ax.text(x_py, y_bot - 0.35,
            "Note: target_gesture is a scheduling cue, not a spawn command. "
            "Unity's spawner runs on its own timeline; the cue only sets which "
            "animal the next automatic spawn uses.",
            ha="left", va="top", fontsize=8, style="italic",
            color=NEUTRAL_MUTED, wrap=True)

    ax.set_xlim(0, x_unity + lane_w + 0.5)
    ax.set_ylim(-0.55, y_top + 0.25)
    ax.set_aspect("equal")
    ax.set_axis_off()
    return _save(fig, out_path)


# ===========================================================================
# 4. CV strategy diagram (§5.4)
# ===========================================================================

def draw_cv_strategies(out_path: Path) -> List[Path]:
    """LOSO-session vs LOSO-subject vs cross-domain CV layouts.

    Each panel draws a grid of session cells per subject and shades the
    train / test partitioning for one example fold. Reader can see at a
    glance what unit (a single session, a whole subject, a whole
    domain) is held out by each strategy.
    """
    _setup_mpl()
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 3.6),
                             gridspec_kw=dict(wspace=0.18,
                                              left=0.04, right=0.99,
                                              top=0.96, bottom=0.04))

    subjects = [("VP_01", 3, "pipe"), ("VP_02", 3, "pipe"),
                ("VP_03", 3, "pipe"), ("VP_04", 2, "unity"),
                ("VP_05", 2, "unity")]
    train_color = PALETTE["med"]
    test_color  = PALETTE["wiso"]

    def draw_grid(ax, title: str, subtitle: str,
                  highlight_fn, *, legend_items=None) -> None:
        ax.set_xlim(0, 5.4)
        ax.set_ylim(-0.5, 4.85)
        ax.set_aspect("equal")
        ax.set_axis_off()
        ax.text(2.7, 4.65, title, ha="center", va="center",
                fontsize=12, fontweight="bold", color=NEUTRAL_TEXT)
        ax.text(2.7, 4.32, subtitle, ha="center", va="center",
                fontsize=9.5, style="italic", color=NEUTRAL_MUTED)

        # x positions: 1 cell per session
        max_sessions = max(s[1] for s in subjects)
        cell_w = 0.55
        cell_h = 0.55
        x0     = 0.55
        y0     = 0.45

        # Subject labels on left
        for r, (subj, n_sess, domain) in enumerate(subjects):
            yy = y0 + (len(subjects) - 1 - r) * (cell_h + 0.10)
            ax.text(x0 - 0.10, yy + cell_h / 2, subj,
                    ha="right", va="center", fontsize=9,
                    color=NEUTRAL_TEXT, fontweight="semibold")
            tag = "pipeline" if domain == "pipe" else "unity"
            tag_color = PALETTE["fau"] if domain == "pipe" else PALETTE["nat_d"]
            ax.text(x0 - 0.10, yy + cell_h / 2 - 0.20, tag,
                    ha="right", va="top", fontsize=7,
                    color=tag_color, style="italic")
            for c in range(n_sess):
                xx = x0 + c * (cell_w + 0.10)
                is_test = highlight_fn(subj, c, domain)
                fill = test_color if is_test else train_color
                border = PALETTE["wiso_d"] if is_test else PALETTE["med_d"]
                ax.add_patch(FancyBboxPatch(
                    (xx, yy), cell_w, cell_h,
                    boxstyle="round,pad=0,rounding_size=0.08",
                    facecolor=fill, edgecolor=border, linewidth=0.7,
                ))
                ax.text(xx + cell_w / 2, yy + cell_h / 2, f"S{c + 1}",
                        ha="center", va="center", fontsize=8,
                        color="white", fontweight="bold")
        ax.text(x0 + (max_sessions / 2) * (cell_w + 0.10) - 0.05, y0 - 0.20,
                "Sessions →", ha="center", va="top",
                fontsize=8, color=NEUTRAL_MUTED, style="italic")

        # Legend below — wider spacing so the (c) labels don't crowd
        leg = legend_items or [
            ("train fold", train_color, PALETTE["med_d"]),
            ("held out",   test_color,  PALETTE["wiso_d"]),
        ]
        for i, (lbl, fc, bc) in enumerate(leg):
            ly = -0.35
            lx = 0.55 + i * 2.10
            ax.add_patch(Rectangle((lx, ly), 0.30, 0.18,
                                   facecolor=fc, edgecolor=bc, linewidth=0.6))
            ax.text(lx + 0.40, ly + 0.09, lbl,
                    ha="left", va="center", fontsize=8.5,
                    color=NEUTRAL_TEXT)

    # (a) LOSO-session: hold out a single (subj, sess) cell
    def fn_loso_sess(subj, c, domain):
        return subj == "VP_02" and c == 1
    draw_grid(axes[0], "(a)  LOSO-session",
              "One session held out · others train",
              fn_loso_sess)

    # (b) LOSO-subject: hold out every cell of one subject
    def fn_loso_subj(subj, c, domain):
        return subj == "VP_03"
    draw_grid(axes[1], "(b)  LOSO-subject",
              "All sessions of one subject held out",
              fn_loso_subj)

    # (c) Cross-domain: train on pipeline, test on unity
    def fn_xdom(subj, c, domain):
        return domain == "unity"
    draw_grid(axes[2], "(c)  Cross-domain",
              "Train on pipeline · test on Unity",
              fn_xdom,
              legend_items=[
                  ("pipeline (train)", train_color, PALETTE["med_d"]),
                  ("unity (test)",     test_color,  PALETTE["wiso_d"]),
              ])

    return _save(fig, out_path)


# ===========================================================================
# 5. ML pipeline data flow (§4.3)
# ===========================================================================

def draw_ml_pipeline(out_path: Path) -> List[Path]:
    """Raw EMG → preprocessing → windowing → features → model → smoother
    → prediction. Single-row pipeline diagram with a feedback arrow for
    the calibration step."""
    _setup_mpl()
    canvas_w = 17.0
    fig, ax = plt.subplots(figsize=(canvas_w, 3.7))

    stages = [
        ("Raw EMG",           "32 ch · 2 kHz",         PALETTE["phil_d"], PALETTE["phil_d"]),
        ("Bandpass",          "20 – 450 Hz",           PALETTE["tech_d"], PALETTE["tech_d"]),
        ("Sliding Window",    "200 ms · 50 ms stride", PALETTE["tech_d"], PALETTE["tech_d"]),
        ("Channel Rotation",  "(calibration offset)",  PALETTE["med_d"],  PALETTE["med_d"]),
        ("Feature Extraction","8 × 32 = 256",          PALETTE["fau"],    PALETTE["fau_d"]),
        ("Classifier",        "LDA · SVM · RF\nCatBoost · CNN",
                                                       PALETTE["fau"],    PALETTE["fau_d"]),
        ("Prob. Smoother",    "EMA + stability gate",  PALETTE["wiso"],   PALETTE["wiso_d"]),
        ("Predicted Class",   "+ confidence",          PALETTE["nat_d"],  PALETTE["nat_d"]),
    ]
    n      = len(stages)
    box_w  = 1.85
    box_h  = 1.15
    gap    = 0.28
    total_w = n * box_w + (n - 1) * gap
    x_start = (canvas_w - total_w) / 2
    y_box   = 1.05

    centers_x = []
    for i, (title, sub, fill, border) in enumerate(stages):
        x = x_start + i * (box_w + gap)
        _box(ax, x, y_box, box_w, box_h, title=title, subtitle=sub,
             fill=fill, border=border, radius=0.10,
             title_size=10, subtitle_size=8, title_weight="bold")
        centers_x.append(x + box_w / 2)
        if i < n - 1:
            x_to = x_start + (i + 1) * (box_w + gap)
            _arrow(ax, x + box_w + 0.02, y_box + box_h / 2,
                   x_to - 0.02, y_box + box_h / 2,
                   color=NEUTRAL_TEXT, lw=1.4, mut_scale=14)

    # Calibration block sits above "Channel Rotation" and feeds the offset.
    cal_w = 2.2
    cal_x = centers_x[3] - cal_w / 2
    cal_y = y_box + box_h + 0.55
    _box(ax, cal_x, cal_y, cal_w, 0.65,
         title="Bracelet Calibration",
         subtitle="(detect offset · stability)",
         fill=PALETTE["med"], border=PALETTE["med_d"], radius=0.10,
         title_size=9.5, subtitle_size=7.5, title_weight="bold")
    _arrow(ax, cal_x + cal_w / 2, cal_y, centers_x[3], y_box + box_h,
           color=PALETTE["med_d"], lw=1.4, mut_scale=14)

    # Inference-server side note: anchored just below the smoother + class
    # output, kept close so no dead space drifts under it.
    note_x = (centers_x[6] + centers_x[7]) / 2
    note_y = y_box - 0.55
    ax.text(note_x, note_y,
            "Real-time inference server\nstreams predictions to Unity",
            ha="center", va="center", fontsize=9, style="italic",
            color=NEUTRAL_MUTED)
    _arrow(ax, note_x, note_y + 0.28, centers_x[7], y_box - 0.05,
           color=NEUTRAL_MUTED, lw=0.8, mut_scale=10, ls=":")

    ax.set_xlim(0, canvas_w)
    ax.set_ylim(note_y - 0.30, cal_y + 0.75 + 0.10)
    ax.set_aspect("equal")
    ax.set_axis_off()
    return _save(fig, out_path)


# ===========================================================================
# 6. Sliding window diagram (§4.3.2)
# ===========================================================================

def draw_sliding_window(out_path: Path) -> List[Path]:
    """Visual of the 200 ms window / 50 ms stride pattern."""
    _setup_mpl()
    fig, ax = plt.subplots(figsize=(13.0, 3.6))

    import numpy as np
    rng = np.random.default_rng(3)
    t = np.linspace(0, 1.0, 2000)        # 1 second, 2 kHz
    sig = np.zeros_like(t)
    for centre in (0.30, 0.55, 0.80):
        sig += np.exp(-((t - centre) ** 2) / 0.005) * 0.6
    sig += rng.normal(0, 0.07, t.size)
    ax.plot(t * 1000, sig, color=NEUTRAL_TEXT, linewidth=0.6, alpha=0.85,
            zorder=2)

    # Windows: start every 50 ms, length 200 ms. Draw the first 5.
    win_len = 200.0
    stride  = 50.0
    starts  = [0.0, 50.0, 100.0, 150.0, 200.0]
    palette = [PALETTE["fau"], PALETTE["med"], PALETTE["nat"],
               PALETTE["phil_d"], PALETTE["wiso"]]
    y0 = -1.20
    win_h = 0.32
    for i, (s, color) in enumerate(zip(starts, palette)):
        y = y0 - i * (win_h + 0.10)
        ax.add_patch(Rectangle(
            (s, y), win_len, win_h,
            facecolor=color, edgecolor=color, alpha=0.85,
        ))
        ax.text(s + win_len / 2, y + win_h / 2,
                f"w{i + 1}  [{int(s)} – {int(s + win_len)} ms]",
                ha="center", va="center", fontsize=9,
                color="white", fontweight="bold")

    # Window-1 marker on the signal — shaded region
    ax.add_patch(Rectangle((0, -1.0), win_len, 2.0,
                           facecolor=PALETTE["fau"], alpha=0.08,
                           edgecolor="none", zorder=1))
    # Stride + window length annotations: placed on a dedicated row
    # ABOVE the signal so they never overlap window pills below.
    annot_y = 1.20
    _arrow(ax, 0, annot_y, stride, annot_y,
           color=NEUTRAL_TEXT, lw=1.0, mut_scale=10, style="<|-|>")
    ax.text(stride / 2, annot_y + 0.10, f"stride = {int(stride)} ms",
            ha="center", va="bottom", fontsize=9, color=NEUTRAL_TEXT)
    _arrow(ax, 250, annot_y, 250 + win_len, annot_y,
           color=NEUTRAL_TEXT, lw=1.0, mut_scale=10, style="<|-|>")
    ax.text(250 + win_len / 2, annot_y + 0.10,
            f"window = {int(win_len)} ms", ha="center", va="bottom",
            fontsize=9, color=NEUTRAL_TEXT)

    # Axis cosmetics
    ax.set_xlim(-40, 480)
    ax.set_ylim(-3.55, 1.65)
    ax.set_xlabel("Time (ms)")
    ax.set_yticks([])
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(NEUTRAL_GRID)
    ax.grid(axis="x", linestyle=":", color=NEUTRAL_GRID)

    fig.tight_layout()
    return _save(fig, out_path)


# ===========================================================================
# CLI
# ===========================================================================

ALL_DIAGRAMS = {
    "system_architecture":   draw_system_architecture,
    "recording_protocol":    draw_recording_protocol,
    "communication_protocol": draw_communication_protocol,
    "cv_strategies":         draw_cv_strategies,
    "ml_pipeline":           draw_ml_pipeline,
    "sliding_window":        draw_sliding_window,
}


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(
        prog="pipeline_figures",
        description="Render the PlayAgain pipeline / structure diagrams.",
    )
    p.add_argument("--out", type=Path, required=True,
                   help="Output directory for the PDFs and PNGs.")
    p.add_argument("--only", nargs="*", default=None,
                   choices=list(ALL_DIAGRAMS),
                   help="Render only the listed diagrams. "
                        "Default: render all six.")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)s  %(message)s")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    targets = args.only or list(ALL_DIAGRAMS)
    log.info("Rendering %d diagram(s) to %s", len(targets), out_dir)
    for key in targets:
        fn = ALL_DIAGRAMS[key]
        out_path = out_dir / f"fig_{key}"
        files = fn(out_path)
        for f in files:
            log.info("  • %s", f.relative_to(out_dir.parent))
    return 0


if __name__ == "__main__":
    sys.exit(main())