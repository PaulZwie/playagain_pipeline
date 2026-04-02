"""Shared GUI style helper for consistent tab/dialog theming."""

from __future__ import annotations

from typing import Any

_THEMES = {
    "bright": {
        "bg": "#f6f8fc",
        "panel": "#ffffff",
        "card": "#eef2ff",
        "accent": "#6d28d9",
        "accent2": "#0284c7",
        "text": "#111827",
        "muted": "#4b5563",
        "border": "#c7d2fe",
    },
    "dark": {
        "bg": "#1e1e2e",
        "panel": "#2a2a3e",
        "card": "#313145",
        "accent": "#7c3aed",
        "accent2": "#06b6d4",
        "text": "#e2e8f0",
        "muted": "#94a3b8",
        "border": "#3f3f5c",
    },
}


def _qss_for(theme: str) -> str:
    c = _THEMES.get((theme or "bright").lower(), _THEMES["bright"])
    return f"""
QMainWindow, QDialog, QWidget {{
    background: {c['bg']};
    color: {c['text']};
}}
QTabWidget::pane {{
    border: 1px solid {c['border']};
    border-radius: 6px;
    background: {c['panel']};
}}
QTabBar::tab {{
    background: {c['bg']};
    color: {c['muted']};
    padding: 6px 14px;
    border: 1px solid {c['border']};
    border-bottom: none;
    border-top-left-radius: 5px;
    border-top-right-radius: 5px;
    margin-right: 2px;
    font-size: 11px;
}}
QTabBar::tab:selected {{
    background: {c['panel']};
    color: {c['accent2']};
    border-bottom: 2px solid {c['accent2']};
}}
QGroupBox {{
    background: {c['panel']};
    border: 1px solid {c['border']};
    border-radius: 7px;
    font-weight: 600;
    color: {c['text']};
    padding-top: 14px;
    margin-top: 5px;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 5px;
    color: {c['accent2']};
    font-size: 11px;
}}
QPushButton {{
    background: {c['card']};
    color: {c['text']};
    border: 1px solid {c['border']};
    border-radius: 5px;
    padding: 5px 12px;
    font-size: 12px;
}}
QPushButton:hover {{ border-color: {c['accent2']}; color: {c['accent2']}; }}
QPushButton:pressed {{ background: {c['border']}; }}
QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox, QTextEdit {{
    background: {c['bg']};
    color: {c['text']};
    border: 1px solid {c['border']};
    border-radius: 4px;
    padding: 3px 6px;
    selection-background-color: {c['accent']};
}}
QComboBox:focus, QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
    border-color: {c['accent']};
}}
QComboBox::drop-down {{ border: none; }}
QListWidget, QTableWidget {{
    background: {c['bg']};
    color: {c['text']};
    border: 1px solid {c['border']};
    border-radius: 5px;
    selection-background-color: {c['accent']};
}}
QProgressBar {{
    background: {c['bg']};
    border: 1px solid {c['border']};
    border-radius: 4px;
    text-align: center;
    color: {c['text']};
}}
QProgressBar::chunk {{
    background: {c['accent2']};
    border-radius: 3px;
}}
QCheckBox::indicator, QRadioButton::indicator {{
    border: 1px solid {c['border']};
    background: {c['bg']};
    border-radius: 3px;
    width: 13px;
    height: 13px;
}}
QCheckBox::indicator:checked, QRadioButton::indicator:checked {{
    background: {c['accent']};
    border-color: {c['accent']};
}}
QHeaderView::section {{
    background: {c['panel']};
    color: {c['muted']};
    border: none;
    padding: 4px 8px;
    font-size: 11px;
}}
"""


def apply_app_style(widget: Any, theme: str = "bright") -> None:
    """Apply the shared app style to a top-level widget/dialog."""
    widget.setStyleSheet(_qss_for(theme))
