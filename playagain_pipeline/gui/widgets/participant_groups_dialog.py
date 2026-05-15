"""
gui/widgets/participant_groups_dialog.py
════════════════════════════════════════
A small, self-contained editor for the healthy / impaired participant
registry (``participant_groups.json``).

Why this exists
───────────────
The validation + thesis-report pipeline can separate participants into
cohorts (healthy vs. impaired), but until now the only way to tell it
who belongs where was to hand-write a JSON file next to the data
directory. This dialog surfaces that file as a normal piece of UI:

  • discover every subject the project knows about (training sessions
    *and* game recordings),
  • assign each one to a cohort with a dropdown,
  • see live healthy / impaired / unknown counts,
  • save back to the canonical ``participant_groups.json`` (grouped-list
    format, which :class:`ParticipantGroups` reads natively).

The dialog is deliberately a *pure editor* — it never runs an
evaluation. It reads and writes one JSON file and emits
:data:`registry_saved` so the caller can refresh whatever depends on it.

Integration
───────────

    dlg = ParticipantGroupsDialog(data_dir=self._data_dir, parent=self)
    dlg.registry_saved.connect(self._on_groups_saved)   # Path
    dlg.exec()

or, non-modal, opened from a button next to a "groups file" field:

    dlg = ParticipantGroupsDialog(data_dir, initial_path=chosen, parent=self)
    dlg.registry_saved.connect(picker.set_path)
    dlg.show()

The dialog has no hard dependency on the rest of the GUI; if
``gui_style`` or the corpus discovery helpers are unavailable it simply
degrades (no theme / no auto-scan button) rather than failing to open.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QComboBox, QDialog, QFileDialog, QFrame, QHBoxLayout, QHeaderView,
    QLabel, QLineEdit, QMessageBox, QPushButton, QSizePolicy,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

# ── Registry backend (Qt-free) ────────────────────────────────────────────
from playagain_pipeline.validation.participant_groups import (
    GROUP_HEALTHY, GROUP_IMPAIRED, GROUP_LABELS, GROUP_UNKNOWN,
    ParticipantGroups, default_groups_path, group_label, normalise_group,
)

# ── Optional app style (same convention as the other dialogs) ─────────────
try:
    from playagain_pipeline.gui.gui_style import apply_app_style  # type: ignore
except Exception:  # noqa: BLE001
    apply_app_style = None  # type: ignore[assignment]

log = logging.getLogger(__name__)


# The three cohorts the dropdown offers, in display order. Unknown is
# last because it is the "not yet decided" state, not a real cohort.
_COHORT_CHOICES = [
    (GROUP_HEALTHY,  f"Healthy  ({GROUP_HEALTHY})"),
    (GROUP_IMPAIRED, f"Impaired  ({GROUP_IMPAIRED})"),
    (GROUP_UNKNOWN,  "Unknown / unset"),
]


# ═══════════════════════════════════════════════════════════════════════════
# Subject discovery — best-effort, never fatal
# ═══════════════════════════════════════════════════════════════════════════

def _discover_subject_ids(data_dir: Path) -> List[str]:
    """
    Return every subject id the project can see — union of training-
    session subjects and game-recording subjects.

    This is wrapped in broad excepts on purpose: the dialog must still
    open on a half-set-up project. A failed scan just means the user
    types subject ids in by hand.
    """
    found: set[str] = set()

    # Training sessions (the SessionCorpus the runner already uses).
    try:
        from playagain_pipeline.validation import SessionCorpus  # type: ignore
        corpus = SessionCorpus(data_dir)
        corpus.discover()
        for rec in corpus.all():
            sid = getattr(rec, "subject_id", None)
            if sid:
                found.add(str(sid))
    except Exception as exc:  # noqa: BLE001
        log.info("Session corpus scan skipped (%s)", exc)

    # Game recordings (separate discovery — see GameCorpus).
    try:
        from playagain_pipeline.evaluation import (  # type: ignore
            discover_game_recordings,
        )
        for desc in discover_game_recordings(data_dir):
            sid = getattr(desc, "subject_id", None)
            if sid:
                found.add(str(sid))
    except Exception as exc:  # noqa: BLE001
        log.info("Game-recording scan skipped (%s)", exc)

    return sorted(found)


# ═══════════════════════════════════════════════════════════════════════════
# Dialog
# ═══════════════════════════════════════════════════════════════════════════

class ParticipantGroupsDialog(QDialog):
    """
    Editor for ``participant_groups.json``.

    Parameters
    ----------
    data_dir : Path
        Project data directory. Used to default the registry path and to
        scan for subject ids.
    initial_path : Path, optional
        Registry file to open on launch. Defaults to
        ``<data_dir>/participant_groups.json``.
    parent : QWidget, optional
        Parent window.

    Signals
    -------
    registry_saved(Path)
        Emitted after a successful save, carrying the file path so the
        caller can point its own "groups file" field at it.
    """

    registry_saved = Signal(Path)

    DEFAULT_TITLE = "Participant groups — healthy / impaired"

    def __init__(
        self,
        data_dir: Path,
        initial_path: Optional[Path] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.setWindowTitle(self.DEFAULT_TITLE)
        self.setModal(False)
        self.resize(560, 620)

        if apply_app_style is not None:
            try:
                apply_app_style(self, theme="bright")
            except Exception:  # noqa: BLE001
                pass

        self._data_dir: Path = Path(data_dir)
        self._path: Path = Path(initial_path) if initial_path else \
            default_groups_path(self._data_dir)

        # subject_id -> cohort code, the single source of truth the table
        # is rendered from and the file is written from.
        self._assignments: Dict[str, str] = {}

        self._build_ui()
        self._load_from_path(self._path, quiet=True)
        # On first open, also pull in any subjects the project knows
        # about so the table isn't empty on a fresh project.
        self._scan_corpus(announce=False)

    # ──────────────────────────────────────────────────────────────────
    # UI construction
    # ──────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(14, 14, 14, 14)
        outer.setSpacing(10)

        # ── Header ───────────────────────────────────────────────────
        header = QFrame()
        header.setStyleSheet(
            "QFrame { background:#f1f5f9; border:1px solid #e2e8f0; "
            "border-radius:8px; }"
        )
        h_lay = QVBoxLayout(header)
        h_lay.setContentsMargins(14, 10, 14, 10)
        h_lay.setSpacing(2)
        title = QLabel("Participant groups")
        tf = QFont(); tf.setPointSize(13); tf.setBold(True)
        title.setFont(tf)
        title.setStyleSheet("color:#0284c7;")
        h_lay.addWidget(title)
        sub = QLabel(
            "Assign each subject to a cohort. This registry is read by "
            "the evaluation tab and the thesis-report generator to keep "
            "healthy and impaired participants separated. It takes "
            "precedence over session-metadata inference."
        )
        sub.setWordWrap(True)
        sub.setStyleSheet("color:#475569; font-size:11px;")
        h_lay.addWidget(sub)
        outer.addWidget(header)

        # ── File row ─────────────────────────────────────────────────
        file_row = QHBoxLayout(); file_row.setSpacing(6)
        file_row.addWidget(QLabel("File:"))
        self._path_edit = QLineEdit(str(self._path))
        self._path_edit.setReadOnly(True)
        self._path_edit.setStyleSheet("color:#475569;")
        file_row.addWidget(self._path_edit, 1)
        load_btn = QPushButton("Open…")
        load_btn.setToolTip("Open an existing participant_groups.json / .csv")
        load_btn.clicked.connect(self._on_open_clicked)
        file_row.addWidget(load_btn)
        default_btn = QPushButton("Use default")
        default_btn.setToolTip(
            "Point at <data_dir>/participant_groups.json — the location "
            "the pipeline looks for automatically."
        )
        default_btn.clicked.connect(self._on_use_default_clicked)
        file_row.addWidget(default_btn)
        outer.addLayout(file_row)

        # ── Toolbar above the table ──────────────────────────────────
        tools = QHBoxLayout(); tools.setSpacing(6)
        scan_btn = QPushButton("⟳ Scan project for subjects")
        scan_btn.setToolTip(
            "Discover subject ids from training sessions and game "
            "recordings and add any that aren't in the table yet."
        )
        scan_btn.clicked.connect(lambda: self._scan_corpus(announce=True))
        tools.addWidget(scan_btn)

        add_btn = QPushButton("+ Add subject")
        add_btn.setToolTip("Add a subject id by hand.")
        add_btn.clicked.connect(self._on_add_subject_clicked)
        tools.addWidget(add_btn)

        remove_btn = QPushButton("− Remove selected")
        remove_btn.setToolTip("Remove the selected row(s) from the registry.")
        remove_btn.clicked.connect(self._on_remove_selected_clicked)
        tools.addWidget(remove_btn)

        tools.addStretch(1)

        # Bulk cohort assignment for the current selection.
        tools.addWidget(QLabel("Set selected to:"))
        for code, label in _COHORT_CHOICES:
            b = QPushButton(group_label(code).capitalize()
                            if code != GROUP_UNKNOWN else "Unknown")
            b.setMaximumWidth(110)
            b.clicked.connect(lambda _=False, c=code: self._set_selected_to(c))
            tools.addWidget(b)
        outer.addLayout(tools)

        # ── Table ────────────────────────────────────────────────────
        self._table = QTableWidget(0, 2)
        self._table.setHorizontalHeaderLabels(["Subject", "Cohort"])
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        hh = self._table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        hh.setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        self._table.setColumnWidth(1, 190)
        outer.addWidget(self._table, 1)

        # ── Counts strip ─────────────────────────────────────────────
        self._counts_label = QLabel("")
        self._counts_label.setStyleSheet(
            "color:#0f172a; font-size:11px; padding:4px 2px;"
        )
        outer.addWidget(self._counts_label)

        # ── Bottom buttons ───────────────────────────────────────────
        bottom = QHBoxLayout(); bottom.setSpacing(8)
        hint = QLabel(
            "Subjects left as “Unknown” are written out as a comment, not "
            "a cohort — the pipeline will try metadata inference for them."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#94a3b8; font-size:10px;")
        bottom.addWidget(hint, 1)

        save_as_btn = QPushButton("Save as…")
        save_as_btn.clicked.connect(self._on_save_as_clicked)
        bottom.addWidget(save_as_btn)

        self._save_btn = QPushButton("Save")
        self._save_btn.setStyleSheet(
            "QPushButton { background:#16a34a; color:white; border:none; "
            "border-radius:5px; padding:6px 16px; font-weight:600; }"
            "QPushButton:hover { background:#15803d; }"
        )
        self._save_btn.clicked.connect(self._on_save_clicked)
        bottom.addWidget(self._save_btn)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        bottom.addWidget(close_btn)
        outer.addLayout(bottom)

    # ──────────────────────────────────────────────────────────────────
    # Table <-> assignments syncing
    # ──────────────────────────────────────────────────────────────────

    def _rebuild_table(self) -> None:
        """Re-render the whole table from ``self._assignments``."""
        self._table.setRowCount(0)
        for subject in sorted(self._assignments.keys()):
            self._append_row(subject, self._assignments[subject])
        self._refresh_counts()

    def _append_row(self, subject: str, code: str) -> None:
        row = self._table.rowCount()
        self._table.insertRow(row)

        item = QTableWidgetItem(subject)
        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        self._table.setItem(row, 0, item)

        combo = QComboBox()
        for c, label in _COHORT_CHOICES:
            combo.addItem(label, c)
        idx = combo.findData(code if code in GROUP_LABELS else GROUP_UNKNOWN)
        combo.setCurrentIndex(max(0, idx))
        # When the user changes a row's cohort, mirror it straight back
        # into the assignments dict so save/counts stay in sync.
        combo.currentIndexChanged.connect(
            lambda _=0, s=subject, cb=combo: self._on_row_cohort_changed(s, cb)
        )
        self._table.setCellWidget(row, 1, combo)

    def _on_row_cohort_changed(self, subject: str, combo: QComboBox) -> None:
        self._assignments[subject] = combo.currentData()
        self._refresh_counts()

    def _selected_subjects(self) -> List[str]:
        rows = {idx.row() for idx in self._table.selectedIndexes()}
        out: List[str] = []
        for r in sorted(rows):
            item = self._table.item(r, 0)
            if item is not None:
                out.append(item.text())
        return out

    def _set_selected_to(self, code: str) -> None:
        subjects = self._selected_subjects()
        if not subjects:
            QMessageBox.information(
                self, "No selection",
                "Select one or more rows first, then choose a cohort.")
            return
        targets = set(subjects)
        for subject in targets:
            self._assignments[subject] = code
        # Update the combos in place rather than a full rebuild so the
        # user's selection and scroll position survive.
        for r in range(self._table.rowCount()):
            item = self._table.item(r, 0)
            if item is None or item.text() not in targets:
                continue
            combo = self._table.cellWidget(r, 1)
            if isinstance(combo, QComboBox):
                combo.blockSignals(True)
                combo.setCurrentIndex(max(0, combo.findData(code)))
                combo.blockSignals(False)
        self._refresh_counts()

    def _refresh_counts(self) -> None:
        n_h = sum(1 for c in self._assignments.values() if c == GROUP_HEALTHY)
        n_i = sum(1 for c in self._assignments.values() if c == GROUP_IMPAIRED)
        n_u = sum(1 for c in self._assignments.values() if c == GROUP_UNKNOWN)
        self._counts_label.setText(
            f"<b>{n_h}</b> healthy &nbsp;·&nbsp; "
            f"<b>{n_i}</b> impaired &nbsp;·&nbsp; "
            f"<b>{n_u}</b> unknown &nbsp;&nbsp;"
            f"<span style='color:#94a3b8;'>({len(self._assignments)} total)</span>"
        )
        self._counts_label.setTextFormat(Qt.TextFormat.RichText)

    # ──────────────────────────────────────────────────────────────────
    # Loading
    # ──────────────────────────────────────────────────────────────────

    def _load_from_path(self, path: Path, *, quiet: bool = False) -> None:
        """Load a registry file into the table. Missing file = empty start."""
        path = Path(path)
        self._path = path
        self._path_edit.setText(str(path))

        if not path.exists():
            if not quiet:
                QMessageBox.information(
                    self, "New file",
                    f"{path.name} does not exist yet — it will be created "
                    "when you press Save.")
            self._assignments = {}
            self._rebuild_table()
            return

        try:
            groups = ParticipantGroups.from_file(path)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(
                self, "Could not read file",
                f"Failed to parse {path}:\n\n{exc}")
            return

        self._assignments = dict(groups.as_subject_map())
        self._rebuild_table()
        if not quiet:
            self._append_status_to_counts(
                f"loaded {len(self._assignments)} subject(s) from {path.name}")

    def _append_status_to_counts(self, msg: str) -> None:
        """Briefly tack a status note onto the counts strip."""
        self._refresh_counts()
        self._counts_label.setText(
            self._counts_label.text()
            + f" &nbsp;<span style='color:#16a34a;'>— {msg}</span>"
        )

    def _scan_corpus(self, *, announce: bool) -> None:
        """Add discovered subject ids that aren't already in the table."""
        discovered = _discover_subject_ids(self._data_dir)
        added = 0
        for sid in discovered:
            if sid not in self._assignments:
                self._assignments[sid] = GROUP_UNKNOWN
                added += 1
        if added or announce:
            self._rebuild_table()
        if announce:
            if added:
                self._append_status_to_counts(
                    f"added {added} subject(s) from project scan")
            else:
                self._append_status_to_counts("scan found no new subjects")

    # ──────────────────────────────────────────────────────────────────
    # Button handlers
    # ──────────────────────────────────────────────────────────────────

    def _on_open_clicked(self) -> None:
        chosen, _ = QFileDialog.getOpenFileName(
            self, "Open participant-group registry",
            str(self._path.parent if self._path else self._data_dir),
            "Group registry (*.json *.csv);;All files (*)",
        )
        if chosen:
            self._load_from_path(Path(chosen))

    def _on_use_default_clicked(self) -> None:
        self._load_from_path(default_groups_path(self._data_dir))

    def _on_add_subject_clicked(self) -> None:
        from PySide6.QtWidgets import QInputDialog
        text, ok = QInputDialog.getText(
            self, "Add subject", "Subject id (e.g. VP_07):")
        if not ok:
            return
        sid = (text or "").strip()
        if not sid:
            return
        if sid in self._assignments:
            QMessageBox.information(
                self, "Already listed",
                f"{sid} is already in the table.")
            return
        self._assignments[sid] = GROUP_UNKNOWN
        self._rebuild_table()

    def _on_remove_selected_clicked(self) -> None:
        subjects = self._selected_subjects()
        if not subjects:
            QMessageBox.information(
                self, "No selection", "Select one or more rows to remove.")
            return
        for sid in subjects:
            self._assignments.pop(sid, None)
        self._rebuild_table()

    def _on_save_clicked(self) -> None:
        self._save_to(self._path)

    def _on_save_as_clicked(self) -> None:
        chosen, _ = QFileDialog.getSaveFileName(
            self, "Save participant-group registry",
            str(self._path), "JSON registry (*.json)",
        )
        if chosen:
            path = Path(chosen)
            if path.suffix.lower() != ".json":
                path = path.with_suffix(".json")
            self._save_to(path)

    # ──────────────────────────────────────────────────────────────────
    # Saving
    # ──────────────────────────────────────────────────────────────────

    def _save_to(self, path: Path) -> None:
        """
        Write the registry in the grouped-list JSON format that
        :class:`ParticipantGroups` reads natively. Unknown subjects are
        preserved as a comment list so the file round-trips without
        silently forgetting them, but they are not written as a cohort.
        """
        path = Path(path)
        healthy = sorted(s for s, c in self._assignments.items()
                         if c == GROUP_HEALTHY)
        impaired = sorted(s for s, c in self._assignments.items()
                          if c == GROUP_IMPAIRED)
        unknown = sorted(s for s, c in self._assignments.items()
                         if c not in (GROUP_HEALTHY, GROUP_IMPAIRED))

        payload = {
            "_comment": (
                "Healthy vs impaired participant registry. Edited via the "
                "Participant groups dialog. Recognised group spellings "
                "include healthy/control and impaired/affected/clinical/"
                "stroke/cp. This file takes precedence over metadata "
                "inference."
            ),
            "_unassigned": unknown,
            "groups": {
                "healthy": healthy,
                "impaired": impaired,
            },
        }

        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
                f.write("\n")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(
                self, "Save failed",
                f"Could not write {path}:\n\n{exc}")
            return

        # Re-verify by parsing it straight back — cheap insurance that
        # what we wrote is what the pipeline will read.
        try:
            check = ParticipantGroups.from_file(path)
            verified = check.counts()
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(
                self, "Saved, but unreadable",
                f"The file was written but could not be parsed back:\n\n{exc}")
            return

        self._path = path
        self._path_edit.setText(str(path))
        self._append_status_to_counts(
            f"saved → {path.name} "
            f"({verified.get(GROUP_HEALTHY, 0)} H / "
            f"{verified.get(GROUP_IMPAIRED, 0)} I)")
        log.info("Participant-group registry saved to %s", path)
        self.registry_saved.emit(path)