"""
home_tab.py
───────────
The Home landing page for the redesigned main window.

Layout (top to bottom):

  ┌─────────────────────────────────────────────┐
  │ Welcome banner                              │  ← fades for 2nd+ visit
  ├─────────────────────────────────────────────┤
  │ ┌─────────┐ ┌─────────┐ ┌─────────┐         │
  │ │  1      │ │  2      │ │  3      │         │  ← three primary task cards
  │ │ Record  │ │ Train   │ │ Use     │         │     (click to jump tab)
  │ └─────────┘ └─────────┘ └─────────┘         │
  ├─────────────────────────────────────────────┤
  │ Recent activity                             │  ← last 5 sessions + runs
  │  • VP_01 / 2026-04-17 (recorded 3 min ago)  │
  │  • lda_pinch_v1 (trained yesterday)         │
  │  …                                          │
  ├─────────────────────────────────────────────┤
  │ [ Run the quickstart ↗ ]                    │  ← only shown on first run
  └─────────────────────────────────────────────┘

The tab is stateless in the strict sense — it re-reads state from the
data manager and corpus on each refresh(). The main window calls
refresh() when something relevant changes (new session, new model).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QPalette, QColor
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QPushButton, QScrollArea, QSizePolicy,
    QVBoxLayout, QWidget,
)

log = logging.getLogger(__name__)


# Jump targets emitted by the cards — the main window routes these to
# the corresponding tabs / dialogs. Using strings (not ints) keeps the
# HomeTab uncoupled from the tab index layout.
TARGET_RECORD   = "record"
TARGET_TRAIN    = "train"
TARGET_USE      = "use"
TARGET_TOOLS    = "tools"


@dataclass
class _RecentActivity:
    kind: str        # "session", "model", "run"
    title: str       # human-readable short description
    subtitle: str    # when/who
    icon: str        # single emoji / glyph
    target: str      # where clicking jumps to


# ---------------------------------------------------------------------------
# A single clickable task card
# ---------------------------------------------------------------------------

class _TaskCard(QFrame):
    """
    Large clickable card with a step number, title, one-line
    description, and a hint about what needs to be done first.
    """

    clicked = Signal(str)   # emits the target string

    def __init__(
        self,
        step_number: int,
        title: str,
        description: str,
        target: str,
        accent: str,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self._target = target
        self._accent = accent
        self._enabled_state = True

        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setSizePolicy(QSizePolicy.Policy.Expanding,
                           QSizePolicy.Policy.Fixed)
        self.setMinimumHeight(160)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(18, 16, 18, 16)
        lay.setSpacing(4)

        # Step badge — small coloured circle with number
        step_row = QHBoxLayout()
        step_row.setContentsMargins(0, 0, 0, 0)
        badge = QLabel(str(step_number))
        badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        badge.setFixedSize(28, 28)
        bf = QFont(); bf.setBold(True); bf.setPointSize(11)
        badge.setFont(bf)
        badge.setStyleSheet(
            f"QLabel {{ background: {accent}; color: white; "
            f"border-radius: 14px; }}"
        )
        step_row.addWidget(badge)
        step_row.addStretch()
        lay.addLayout(step_row)

        # Title
        self._title_label = QLabel(title)
        tf = QFont(); tf.setBold(True); tf.setPointSize(14)
        self._title_label.setFont(tf)
        self._title_label.setStyleSheet("color: #0f172a;")
        lay.addWidget(self._title_label)

        # Description
        desc = QLabel(description)
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #475569; font-size: 11px;")
        lay.addWidget(desc)

        lay.addStretch()

        # Status line — filled in by set_state()
        self._status_label = QLabel("")
        self._status_label.setStyleSheet("color: #64748b; font-size: 10px;")
        self._status_label.setWordWrap(True)
        lay.addWidget(self._status_label)

        self._apply_style(hover=False)

    def _apply_style(self, hover: bool) -> None:
        base_bg = "#ffffff"
        hover_bg = "#f8fafc"
        bg = hover_bg if hover else base_bg
        border = self._accent if hover else "#e2e8f0"
        self.setStyleSheet(
            f"_TaskCard {{"
            f"  background: {bg};"
            f"  border: 2px solid {border};"
            f"  border-radius: 10px;"
            f"}}"
        )

    def enterEvent(self, event):
        if self._enabled_state:
            self._apply_style(hover=True)
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._apply_style(hover=False)
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        if self._enabled_state and event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self._target)
            event.accept()
        else:
            super().mousePressEvent(event)

    def set_status(self, text: str) -> None:
        self._status_label.setText(text)

    def set_enabled_state(self, enabled: bool) -> None:
        """
        Visually indicate whether the card's prerequisites are met.
        A disabled card is still clickable — we don't want to hide
        the path to the user — but it's rendered muted.
        """
        self._enabled_state = True   # always clickable
        if enabled:
            self._title_label.setStyleSheet("color: #0f172a;")
        else:
            self._title_label.setStyleSheet("color: #94a3b8;")


# ---------------------------------------------------------------------------
# Recent-activity row
# ---------------------------------------------------------------------------

class _RecentRow(QFrame):
    """Single row in the recent-activity list."""

    clicked = Signal(str)

    def __init__(self, item: _RecentActivity, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._target = item.target
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet(
            "_RecentRow { border-bottom: 1px solid #e2e8f0; }"
            "_RecentRow:hover { background: #f8fafc; }"
        )

        lay = QHBoxLayout(self)
        lay.setContentsMargins(10, 8, 10, 8)
        lay.setSpacing(10)

        icon = QLabel(item.icon)
        f = QFont(); f.setPointSize(14)
        icon.setFont(f)
        lay.addWidget(icon, 0, Qt.AlignmentFlag.AlignVCenter)

        text_col = QVBoxLayout()
        text_col.setContentsMargins(0, 0, 0, 0)
        text_col.setSpacing(0)

        title = QLabel(item.title)
        tf = QFont(); tf.setBold(True); tf.setPointSize(10)
        title.setFont(tf)
        title.setStyleSheet("color: #0f172a;")
        text_col.addWidget(title)

        sub = QLabel(item.subtitle)
        sub.setStyleSheet("color: #64748b; font-size: 9px;")
        text_col.addWidget(sub)

        lay.addLayout(text_col, 1)

        chev = QLabel("›")
        chev.setStyleSheet("color: #cbd5e1; font-size: 18px;")
        lay.addWidget(chev)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self._target)
            event.accept()
        else:
            super().mousePressEvent(event)


# ---------------------------------------------------------------------------
# The Home tab
# ---------------------------------------------------------------------------

class HomeTab(QWidget):
    """
    Landing page. Emits ``jump_to_target(str)`` when the user clicks a
    card — the main window routes to the appropriate tab/dialog.

    Constructor arguments
    ─────────────────────
    data_dir : Path
        Root pipeline data directory. Used by refresh() to scan
        sessions/, models/, and validation_runs/ for recent activity.
    """

    jump_to_target = Signal(str)           # card or activity-row target
    quickstart_requested = Signal()        # "Run the quickstart" link

    def __init__(self, data_dir: Path, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.data_dir = Path(data_dir)
        self._is_first_run = not self._any_sessions_exist()

        self.setAutoFillBackground(True)
        pal = self.palette()
        pal.setColor(QPalette.ColorRole.Window, QColor("#f8fafc"))
        self.setPalette(pal)

        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 20)
        root.setSpacing(18)

        # ── Welcome banner ─────────────────────────────────────────────
        self._banner = self._build_banner()
        root.addWidget(self._banner)

        # ── Three task cards ──────────────────────────────────────────
        cards_row = QHBoxLayout()
        cards_row.setSpacing(14)
        self._record_card = _TaskCard(
            1, "Record new data",
            "Connect a device and capture gesture sessions for one subject.",
            TARGET_RECORD, "#0284c7",
        )
        self._train_card = _TaskCard(
            2, "Train a model",
            "Build a dataset from your recordings and train a classifier.",
            TARGET_TRAIN, "#7c3aed",
        )
        self._use_card = _TaskCard(
            3, "Use a model live",
            "Run a trained model on the live EMG stream or stream "
            "predictions to Unity.",
            TARGET_USE, "#16a34a",
        )
        for card in (self._record_card, self._train_card, self._use_card):
            card.clicked.connect(self.jump_to_target.emit)
            cards_row.addWidget(card, 1)
        root.addLayout(cards_row)

        # ── Recent activity ───────────────────────────────────────────
        self._recent_container = self._build_recent_container()
        root.addWidget(self._recent_container, 1)

        # ── Quickstart CTA ────────────────────────────────────────────
        self._quickstart_button = QPushButton("  Run the quickstart  ↗")
        self._quickstart_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._quickstart_button.setStyleSheet(
            "QPushButton {"
            "  background: #0284c7; color: white; border: none;"
            "  border-radius: 8px; padding: 12px 20px;"
            "  font-size: 12px; font-weight: 600;"
            "}"
            "QPushButton:hover { background: #0369a1; }"
        )
        self._quickstart_button.clicked.connect(self.quickstart_requested.emit)
        cta_row = QHBoxLayout()
        cta_row.addStretch()
        cta_row.addWidget(self._quickstart_button)
        cta_row.addStretch()
        root.addLayout(cta_row)

        # Initial population
        self.refresh()

    # ------------------------------------------------------------------

    def _build_banner(self) -> QFrame:
        banner = QFrame()
        banner.setStyleSheet(
            "QFrame {"
            "  background: qlineargradient(x1:0, y1:0, x2:1, y2:0,"
            "    stop:0 #0284c7, stop:1 #7c3aed);"
            "  border-radius: 10px;"
            "}"
        )
        lay = QVBoxLayout(banner)
        lay.setContentsMargins(24, 16, 24, 16)
        lay.setSpacing(4)

        title = QLabel("EMG Gesture Pipeline")
        tf = QFont(); tf.setBold(True); tf.setPointSize(16)
        title.setFont(tf)
        title.setStyleSheet("color: white; background: transparent;")
        lay.addWidget(title)

        sub = QLabel(
            "Record EMG recordings, train a gesture classifier, and run it "
            "live. Each step saves its work to disk — you can stop and "
            "resume at any time."
        )
        sub.setWordWrap(True)
        sub.setStyleSheet(
            "color: rgba(255,255,255,0.85); background: transparent;"
            " font-size: 11px;"
        )
        lay.addWidget(sub)

        return banner

    def _build_recent_container(self) -> QFrame:
        box = QFrame()
        box.setStyleSheet(
            "QFrame {"
            "  background: white;"
            "  border: 1px solid #e2e8f0;"
            "  border-radius: 10px;"
            "}"
        )
        lay = QVBoxLayout(box)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        # Header row
        header = QFrame()
        header.setStyleSheet(
            "QFrame { border-bottom: 1px solid #e2e8f0; background: #f8fafc; "
            "  border-top-left-radius: 10px; border-top-right-radius: 10px; }"
        )
        hlay = QHBoxLayout(header)
        hlay.setContentsMargins(14, 10, 14, 10)
        hlbl = QLabel("Recent activity")
        hf = QFont(); hf.setBold(True); hf.setPointSize(10)
        hlbl.setFont(hf)
        hlbl.setStyleSheet("color: #475569;")
        hlay.addWidget(hlbl)
        hlay.addStretch()
        lay.addWidget(header)

        # Scrollable list of rows
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(
            "QScrollArea { border: none; background: white; "
            "  border-bottom-left-radius: 10px; "
            "  border-bottom-right-radius: 10px; }"
        )
        self._recent_host = QWidget()
        self._recent_host.setStyleSheet("background: white;")
        self._recent_lay = QVBoxLayout(self._recent_host)
        self._recent_lay.setContentsMargins(0, 0, 0, 0)
        self._recent_lay.setSpacing(0)
        self._recent_lay.addStretch(1)
        scroll.setWidget(self._recent_host)
        lay.addWidget(scroll, 1)
        return box

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def refresh(self) -> None:
        """Re-scan the data dir and update cards + recent activity."""
        sessions_count = self._count_sessions()
        models_count   = self._count_models()

        # Card statuses give at-a-glance progress signals.
        if sessions_count == 0:
            self._record_card.set_status("No sessions yet — start here.")
        else:
            self._record_card.set_status(
                f"{sessions_count} session(s) on disk."
            )

        if sessions_count == 0:
            self._train_card.set_status(
                "Record data first, then you can train a model."
            )
            self._train_card.set_enabled_state(False)
        elif models_count == 0:
            self._train_card.set_status(
                f"Ready — {sessions_count} session(s) available."
            )
            self._train_card.set_enabled_state(True)
        else:
            self._train_card.set_status(
                f"{models_count} model(s) trained."
            )
            self._train_card.set_enabled_state(True)

        if models_count == 0:
            self._use_card.set_status(
                "Train a model first."
            )
            self._use_card.set_enabled_state(False)
        else:
            self._use_card.set_status(
                f"{models_count} model(s) ready to run."
            )
            self._use_card.set_enabled_state(True)

        # Recent activity list
        self._populate_recent(self._gather_recent())

        # Quickstart CTA visibility
        self._quickstart_button.setVisible(self._is_first_run)

    def mark_no_longer_first_run(self) -> None:
        """Hide the quickstart CTA once the user has completed it."""
        self._is_first_run = False
        self._quickstart_button.setVisible(False)

    # ------------------------------------------------------------------
    # Internal scanning
    # ------------------------------------------------------------------

    def _any_sessions_exist(self) -> bool:
        sessions_root = self.data_dir / "sessions"
        if not sessions_root.exists():
            return False
        for _ in sessions_root.rglob("metadata.json"):
            return True
        return False

    def _count_sessions(self) -> int:
        sessions_root = self.data_dir / "sessions"
        if not sessions_root.exists():
            return 0
        return sum(1 for _ in sessions_root.rglob("metadata.json"))

    def _count_models(self) -> int:
        models_root = self.data_dir / "models"
        if not models_root.exists():
            return 0
        # Treat each immediate subdir with a metadata.json as a model.
        return sum(
            1 for p in models_root.iterdir()
            if p.is_dir() and (p / "metadata.json").exists()
        )

    def _gather_recent(self) -> List[_RecentActivity]:
        items: List[_RecentActivity] = []
        items.extend(self._scan_recent_sessions(limit=3))
        items.extend(self._scan_recent_models(limit=2))
        items.extend(self._scan_recent_validation_runs(limit=2))
        # Sort by last-modified timestamp (descending) and cap.
        items.sort(key=lambda it: it.subtitle, reverse=True)
        return items[:6]

    def _scan_recent_sessions(self, limit: int) -> List[_RecentActivity]:
        root = self.data_dir / "sessions"
        if not root.exists():
            return []
        out: List[_RecentActivity] = []
        for meta in root.rglob("metadata.json"):
            try:
                mtime = datetime.fromtimestamp(meta.stat().st_mtime)
                session_dir = meta.parent
                rel = session_dir.relative_to(root)
                parts = rel.parts
                subject = parts[0] if parts else "unknown"
                if "unity_sessions" in parts:
                    subject = f"unity/{parts[1] if len(parts) > 1 else '?'}"
                session_id = parts[-1]
                out.append(_RecentActivity(
                    kind="session",
                    title=f"{subject}  /  {session_id}",
                    subtitle=_human_time(mtime),
                    icon="📼",
                    target=TARGET_RECORD,
                ))
            except Exception:  # noqa: BLE001
                continue
        out.sort(key=lambda it: it.subtitle, reverse=True)
        return out[:limit]

    def _scan_recent_models(self, limit: int) -> List[_RecentActivity]:
        root = self.data_dir / "models"
        if not root.exists():
            return []
        out: List[_RecentActivity] = []
        for child in root.iterdir():
            meta = child / "metadata.json"
            if not meta.exists():
                continue
            try:
                mtime = datetime.fromtimestamp(meta.stat().st_mtime)
                model_type = "?"
                try:
                    d = json.loads(meta.read_text())
                    model_type = d.get("model_type", "?")
                except Exception:
                    pass
                out.append(_RecentActivity(
                    kind="model",
                    title=f"{child.name}  ({model_type})",
                    subtitle=_human_time(mtime),
                    icon="🧠",
                    target=TARGET_USE,
                ))
            except Exception:  # noqa: BLE001
                continue
        out.sort(key=lambda it: it.subtitle, reverse=True)
        return out[:limit]

    def _scan_recent_validation_runs(self, limit: int) -> List[_RecentActivity]:
        root = self.data_dir / "validation_runs"
        if not root.exists():
            return []
        out: List[_RecentActivity] = []
        for child in root.iterdir():
            if not child.is_dir():
                continue
            exp = child / "experiment.json"
            if not exp.exists():
                continue
            try:
                mtime = datetime.fromtimestamp(exp.stat().st_mtime)
                try:
                    d = json.loads(exp.read_text())
                    name = d.get("name", child.name)
                except Exception:
                    name = child.name
                out.append(_RecentActivity(
                    kind="run",
                    title=f"Validation: {name}",
                    subtitle=_human_time(mtime),
                    icon="📊",
                    target=TARGET_TOOLS,
                ))
            except Exception:  # noqa: BLE001
                continue
        out.sort(key=lambda it: it.subtitle, reverse=True)
        return out[:limit]

    def _populate_recent(self, items: List[_RecentActivity]) -> None:
        # Clear existing rows (keep the trailing stretch at the end).
        while self._recent_lay.count() > 1:
            row = self._recent_lay.takeAt(0)
            if row and row.widget():
                row.widget().deleteLater()

        if not items:
            empty = QLabel(
                "  No activity yet — record your first session to get started."
            )
            empty.setStyleSheet("color: #94a3b8; padding: 20px; font-style: italic;")
            self._recent_lay.insertWidget(0, empty)
            return

        for item in items:
            row = _RecentRow(item)
            row.clicked.connect(self.jump_to_target.emit)
            self._recent_lay.insertWidget(self._recent_lay.count() - 1, row)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _human_time(when: datetime) -> str:
    """Render a datetime as a short relative label like '3 min ago'."""
    delta = datetime.now() - when
    sec = int(delta.total_seconds())
    if sec < 60:
        return "just now"
    if sec < 3600:
        m = sec // 60
        return f"{m} min ago"
    if sec < 86_400:
        h = sec // 3600
        return f"{h} hr ago"
    days = sec // 86_400
    if days < 7:
        return f"{days} day{'s' if days != 1 else ''} ago"
    return when.strftime("%Y-%m-%d")
