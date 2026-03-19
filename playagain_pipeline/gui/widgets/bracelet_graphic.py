"""
Matplotlib-based 3D bracelet visualization widget.

Shows electrode positions as a 3D-projected ring with depth shading,
rotation offset indicator, and channel numbering. Designed to be
embedded in PySide6 via FigureCanvasQTAgg.
"""

import math
import numpy as np

from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from PySide6.QtWidgets import QWidget, QVBoxLayout, QSizePolicy


class BraceletGraphicWidget(QWidget):
    """
    A PySide6 widget embedding a matplotlib 3D-projected bracelet diagram.

    Features:
    - 3D perspective view of electrode ring (inner + outer rows)
    - Depth-based shading (front electrodes bright, back ones dark)
    - Rotation offset arrow with reference marker
    - Channel numbers on front-facing electrodes
    - Tilt for natural viewing angle
    """

    def __init__(
        self,
        num_electrodes: int = 32,
        rotation_offset: int = 0,
        parent=None,
    ):
        super().__init__(parent)
        self._num_electrodes = num_electrodes
        self._rotation_offset = rotation_offset
        self._tilt = 25.0  # degrees of tilt for 3D perspective

        self._fig = Figure(figsize=(4, 4), dpi=100, facecolor="white")
        self._ax = self._fig.add_subplot(111)
        self._canvas = FigureCanvasQTAgg(self._fig)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._canvas)

        self._canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )

        self._draw()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_rotation_offset(self, offset: int) -> None:
        """Update the displayed rotation offset and redraw."""
        self._rotation_offset = offset
        self._draw()

    def set_num_electrodes(self, n: int) -> None:
        """Update the number of electrodes and redraw."""
        self._num_electrodes = n
        self._draw()

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def _draw(self) -> None:
        """Render the bracelet diagram."""
        ax = self._ax
        ax.clear()

        N = self._num_electrodes
        n_pairs = N // 2  # electrode pairs (inner + outer per position)
        tilt_rad = math.radians(self._tilt)
        cos_tilt = math.cos(tilt_rad)

        # Radii for inner / outer electrode rows
        r_outer = 1.0
        r_inner = 0.78

        # Rotation angle per channel-offset
        rot_rad = (2.0 * math.pi / n_pairs) * self._rotation_offset

        # -- Helper: project an electrode at angular position theta ------
        def project(theta: float, radius: float):
            """Return (x, y_proj, depth) for a point on the ring."""
            x = radius * math.cos(theta)
            y = radius * math.sin(theta)
            y_proj = y * cos_tilt            # squash y for tilt
            depth = math.sin(theta)          # +1 = front, -1 = back
            return x, y_proj, depth

        # -- Collect electrode data --------------------------------------
        elec_x, elec_y, elec_depth, elec_idx, elec_ring = [], [], [], [], []

        for pair in range(n_pairs):
            theta = 2.0 * math.pi * pair / n_pairs - math.pi / 2  # start at bottom (approx)

            for ring_i, (radius, idx_offset) in enumerate(
                [(r_outer, 0), (r_inner, 1)]
            ):
                if N == 32:
                    # Split ring topology for 32-ch bracelet
                    # Inner ring (ring_i=1): 0-15
                    # Outer ring (ring_i=0): 16-31
                    if ring_i == 1:
                        idx = pair
                    else:
                        idx = pair + 16
                else:
                    # Default interleaved topology
                    idx = pair * 2 + idx_offset

                x, y, d = project(theta, radius)
                elec_x.append(x)
                elec_y.append(y)
                elec_depth.append(d)
                elec_idx.append(idx)
                elec_ring.append(ring_i)  # 0 = outer, 1 = inner

        elec_x = np.array(elec_x)
        elec_y = np.array(elec_y)
        elec_depth = np.array(elec_depth)

        # -- Draw bracelet outline (ellipse) -----------------------------
        t_ell = np.linspace(0, 2 * math.pi, 200)
        outline_r = (r_outer + r_inner) / 2 + 0.15
        ax.plot(
            outline_r * np.cos(t_ell),
            outline_r * np.sin(t_ell) * cos_tilt,
            color="#bbbbbb", linewidth=1.5, zorder=0,
        )

        # -- Sort by depth so back electrodes are drawn first ------------
        order = np.argsort(elec_depth)

        # Normalise depth → alpha / size
        d_min, d_max = elec_depth.min(), elec_depth.max()
        d_range = d_max - d_min if d_max != d_min else 1.0

        # Colour map: blue → bright for front, dark for back
        for i in order:
            t = (elec_depth[i] - d_min) / d_range  # 0 = back, 1 = front
            alpha = 0.25 + 0.75 * t
            size = 80 + 220 * t  # bigger when closer
            color_val = 0.15 + 0.60 * t  # darker when further

            is_rotated = False
            idx = elec_idx[i]
            # Check if this electrode has been shifted by the rotation
            if self._rotation_offset != 0 and idx < N:
                is_rotated = True

            # Use blue for electrodes, orange tint for rotated marker area
            if is_rotated and self._rotation_offset != 0:
                face_color = (0.2, 0.45, 0.85, alpha)
            else:
                face_color = (0.2, 0.45, 0.85, alpha)

            ax.scatter(
                elec_x[i], elec_y[i],
                s=size, c=[face_color], edgecolors="black",
                linewidths=0.5 * alpha, zorder=2 + int(t * 10),
            )

            # Label front-facing electrodes
            if t > 0.35:
                fs = max(5, int(6 + 4 * t))
                ax.text(
                    elec_x[i], elec_y[i], str(idx),
                    fontsize=fs, ha="center", va="center",
                    color="white", fontweight="bold",
                    zorder=3 + int(t * 10),
                    alpha=min(1.0, alpha + 0.1),
                )

        # -- Draw rotation offset arrow ----------------------------------
        if self._rotation_offset != 0:
            # Arrow from reference (top) rotated by offset amount
            ref_theta = -math.pi / 2  # top
            cur_theta = ref_theta + rot_rad

            r_arrow = outline_r + 0.08
            # Reference marker (green dot at top)
            rx, ry, _ = project(ref_theta, r_arrow)
            ax.scatter(
                rx, ry, s=100, c="limegreen", edgecolors="darkgreen",
                linewidths=1.5, zorder=20, marker="^",
            )
            ax.text(
                rx, ry + 0.12, "REF",
                fontsize=7, ha="center", va="bottom",
                color="darkgreen", fontweight="bold", zorder=20,
            )

            # Current position marker (red arrow)
            cx, cy, _ = project(cur_theta, r_arrow)
            ax.annotate(
                "", xy=(cx, cy), xytext=(rx, ry),
                arrowprops=dict(
                    arrowstyle="-|>", color="red", lw=2,
                    connectionstyle="arc3,rad=0.3",
                ),
                zorder=19,
            )

            # Label the offset
            mid_theta = (ref_theta + cur_theta) / 2
            mx, my, _ = project(mid_theta, r_arrow + 0.22)
            ax.text(
                mx, my, f"+{self._rotation_offset}",
                fontsize=9, ha="center", va="center",
                color="red", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="red", alpha=0.8),
                zorder=20,
            )
        else:
            # No rotation — show reference marker at top
            r_arrow = outline_r + 0.08
            rx, ry, _ = project(-math.pi / 2, r_arrow)
            ax.scatter(
                rx, ry, s=80, c="limegreen", edgecolors="darkgreen",
                linewidths=1.2, zorder=20, marker="^",
            )
            ax.text(
                rx, ry + 0.10, "REF",
                fontsize=7, ha="center", va="bottom",
                color="darkgreen", fontweight="bold", zorder=20,
            )

        # -- Title -------------------------------------------------------
        title = f"Bracelet  ({N} ch)"
        if self._rotation_offset != 0:
            title += f"  —  Rotation: {self._rotation_offset} ch"
        else:
            title += "  —  Aligned"
        ax.set_title(title, fontsize=10, pad=8)

        # -- Clean up axes ----------------------------------------------
        ax.set_aspect("equal")
        ax.set_xlim(-1.55, 1.55)
        ax.set_ylim(-1.55, 1.55)
        ax.axis("off")
        self._fig.tight_layout()
        self._canvas.draw_idle()