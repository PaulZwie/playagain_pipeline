"""
Session Picker UI
─────────────────
A small tkinter window that lets you:
  • Choose evaluation mode: Hold-out  or  Cross-Validation
  • Tick which participants (VPs) to include
  • Hold-out: see all sessions with TRAIN/VAL badges, click to toggle, slider for ratio
  • Cross-Val: choose number of folds
  • Click "Run" → returns a result dict

Usage (standalone):
    python performance_assessment/session_picker_ui.py

Usage (from code):
    from performance_assessment.session_picker_ui import pick_sessions
    result = pick_sessions(dm)
    # result["mode"] == "holdout"  → result["train"], result["val"]
    # result["mode"] == "cv"       → result["sessions"], result["cv_folds"]
"""

import sys
import tkinter as tk
from tkinter import ttk
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from playagain_pipeline.core.data_manager import DataManager


# ═══════════════════════════════════════════════════════════════════════════
# Colour palette
# ═══════════════════════════════════════════════════════════════════════════

C = {
    "bg":        "#1e1e2e",
    "panel":     "#2a2a3e",
    "card":      "#313145",
    "accent":    "#7c3aed",
    "accent2":   "#a78bfa",
    "train":     "#22c55e",
    "val":       "#f59e0b",
    "cv":        "#38bdf8",
    "excluded":  "#6b7280",
    "text":      "#e2e8f0",
    "subtext":   "#94a3b8",
    "border":    "#3f3f5a",
    "hover":     "#3d3d5c",
    "btn_run":   "#7c3aed",
    "btn_hover": "#6d28d9",
    "danger":    "#ef4444",
    "seg_on":    "#7c3aed",
    "seg_off":   "#2a2a3e",
}


# ═══════════════════════════════════════════════════════════════════════════
# Main UI class
# ═══════════════════════════════════════════════════════════════════════════

class SessionPickerUI:
    """
    Mode "holdout": choose VPs, auto-split per VP chronologically, click to
                    override individual session badges.
    Mode "cv":      choose VPs (all their sessions are used), choose #folds.
    """

    def __init__(self, dm: DataManager, val_ratio: float = 0.2):
        self.dm = dm
        self.val_ratio = val_ratio
        self.result: Optional[Dict[str, Any]] = None

        # ── Load sessions ─────────────────────────────────────────────
        self._subject_sessions: Dict[str, List[Any]] = {}
        for subj in dm.list_subjects():
            sessions = []
            for sess_id in dm.list_sessions(subj):
                try:
                    s = dm.load_session(subj, sess_id)
                    # Validate data shape matches metadata
                    data = s.get_data()
                    if data.shape[1] != s.metadata.num_channels:
                        continue  # silently skip corrupt sessions
                    sessions.append(s)
                except Exception:
                    pass
            if sessions:
                self._subject_sessions[subj] = sessions

        if not self._subject_sessions:
            raise ValueError("No sessions found in data directory.")

        # ── Create root window ────────────────────────────────────────
        self.root = tk.Tk()
        self.root.title("Session Picker")
        self.root.configure(bg=C["bg"])
        self.root.resizable(True, True)

        # ── Tk state variables ────────────────────────────────────────
        self._mode = tk.StringVar(value="holdout")   # "holdout" | "cv"
        self._vp_vars: Dict[str, tk.BooleanVar] = {
            subj: tk.BooleanVar(value=True)
            for subj in self._subject_sessions
        }
        self._session_role: Dict[str, str] = {}
        self._ratio_var = tk.DoubleVar(value=self.val_ratio)
        self._cv_folds_var = tk.IntVar(value=5)

        self._compute_split()

        # ── Build & show ──────────────────────────────────────────────
        self._build_ui()
        self.root.update_idletasks()
        w, h = 900, 680
        sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        self.root.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")
        self.root.minsize(720, 520)

    # ─────────────────────────────────────────────────────────────────
    # Split logic
    # ─────────────────────────────────────────────────────────────────

    def _compute_split(self, preserve_exclusions: bool = False) -> None:
        """Chronological hold-out split per VP."""
        mode = self._mode.get()

        for subj, sessions in self._subject_sessions.items():
            if not self._vp_vars[subj].get():
                for s in sessions:
                    self._session_role[s.metadata.session_id] = "excluded"
                continue

            # Filter sessions that are already manually excluded if preserving
            active_sessions = []
            excluded_sessions = []

            for s in sessions:
                sid = s.metadata.session_id
                if preserve_exclusions and self._session_role.get(sid) == "excluded":
                    excluded_sessions.append(s)
                else:
                    active_sessions.append(s)

            if mode == "cv":
                # In CV mode, all active sessions are just "use"
                for s in active_sessions:
                    self._session_role[s.metadata.session_id] = "use"
            else:
                # In Hold-out mode, split active sessions
                n = len(active_sessions)
                n_val = max(1, round(n * self.val_ratio)) if n > 0 else 0
                for i, s in enumerate(active_sessions):
                    self._session_role[s.metadata.session_id] = (
                        "train" if i < n - n_val else "val"
                    )

            # Ensure excluded stay excluded
            for s in excluded_sessions:
                self._session_role[s.metadata.session_id] = "excluded"

    def _recompute_and_refresh(self) -> None:
        # Checkbox toggled or ratio changed -> recompute but try to stick to logic
        # Usually when toggling VP, we reset. When changing ratio, we reset split.
        self._compute_split(preserve_exclusions=True)
        self._refresh_session_list()
        self._refresh_stats()

    # ─────────────────────────────────────────────────────────────────
    # Build UI
    # ─────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        # Accent top bar
        tk.Frame(self.root, bg=C["accent"], height=4).pack(fill="x")

        # Header
        hdr = tk.Frame(self.root, bg=C["bg"], pady=10)
        hdr.pack(fill="x", padx=20)
        tk.Label(hdr, text="Session Picker",
                 font=("SF Pro Display", 17, "bold"),
                 bg=C["bg"], fg=C["text"]).pack(side="left")

        # ── Mode toggle (segmented control) ──────────────────────────
        self._build_mode_toggle(hdr)

        # ── Body (left panel + right panel) ──────────────────────────
        self._body = tk.Frame(self.root, bg=C["bg"])
        self._body.pack(fill="both", expand=True, padx=20, pady=(4, 0))
        self._body.columnconfigure(0, weight=0, minsize=190)
        self._body.columnconfigure(1, weight=1)
        self._body.rowconfigure(0, weight=1)

        self._build_left_panel(self._body)

        # Right panel container for session list (always shown now)
        self._right_container = tk.Frame(self._body, bg=C["bg"])
        self._right_container.grid(row=0, column=1, sticky="nsew", pady=(0, 10))
        self._right_container.rowconfigure(0, weight=1)
        self._right_container.columnconfigure(0, weight=1)

        self._build_session_panel(self._right_container)
        # CV panel removed, we use the session list for both modes

        # ── Bottom bar ────────────────────────────────────────────────
        self._build_bottom_bar()

    def _build_mode_toggle(self, parent: tk.Frame) -> None:
        seg = tk.Frame(parent, bg=C["panel"],
                       highlightbackground=C["border"], highlightthickness=1)
        seg.pack(side="right", padx=(0, 0))

        self._seg_btns: Dict[str, tk.Label] = {}
        for mode, label in [("holdout", "Hold-out"), ("cv", "Cross-Validation")]:
            btn = tk.Label(seg, text=label,
                           font=("SF Pro Display", 11),
                           padx=14, pady=5, cursor="hand2")
            btn.pack(side="left")
            self._seg_btns[mode] = btn
            btn.bind("<Button-1>", lambda e, m=mode: self._on_mode(m))

        self._update_seg_style()

    def _on_mode(self, mode: str) -> None:
        if self._mode.get() == mode:
            return
        self._mode.set(mode)
        self._update_seg_style()
        self._reassign_roles_for_mode(mode)
        self._refresh_session_list()
        self._refresh_stats()

    def _update_seg_style(self) -> None:
        for mode, btn in self._seg_btns.items():
            active = self._mode.get() == mode
            btn.configure(
                bg=C["seg_on"] if active else C["seg_off"],
                fg=C["text"] if active else C["subtext"],
            )

    def _reassign_roles_for_mode(self, mode: str) -> None:
        """Translate roles when switching modes, preserving exclusions."""
        # Identify currently excluded sessions
        excluded = {sid for sid, role in self._session_role.items() if role == "excluded"}

        if mode == "cv":
            # Switch to CV: anything not excluded becomes "use"
            for subj, sessions in self._subject_sessions.items():
                if not self._vp_vars[subj].get():
                    continue
                for s in sessions:
                    sid = s.metadata.session_id
                    if sid not in excluded:
                        self._session_role[sid] = "use"
        else:
            # Switch to Hold-out: re-apply split to non-excluded sessions
            self._compute_split(preserve_exclusions=True)

    def _switch_mode(self) -> None:
        pass # No longer needed as we share the panel

    # ── Left panel (VP checkboxes + ratio/fold controls) ──────────────

    def _build_left_panel(self, parent: tk.Frame) -> None:
        frame = tk.Frame(parent, bg=C["panel"],
                         highlightbackground=C["border"], highlightthickness=1)
        frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10), pady=(0, 10))

        tk.Label(frame, text="Participants",
                 font=("SF Pro Display", 12, "bold"),
                 bg=C["panel"], fg=C["text"], pady=10).pack(fill="x", padx=12)
        tk.Frame(frame, bg=C["border"], height=1).pack(fill="x")

        for subj, var in self._vp_vars.items():
            row = tk.Frame(frame, bg=C["panel"])
            row.pack(fill="x", padx=8, pady=2)
            tk.Checkbutton(
                row, variable=var, text=subj,
                font=("SF Pro Display", 12),
                bg=C["panel"], fg=C["text"],
                selectcolor=C["accent"],
                activebackground=C["panel"], activeforeground=C["accent2"],
                bd=0, highlightthickness=0,
                command=self._recompute_and_refresh,
            ).pack(side="left", padx=4, pady=4)
            n = len(self._subject_sessions[subj])
            tk.Label(row, text=str(n), font=("SF Pro Display", 10),
                     bg=C["panel"], fg=C["subtext"]).pack(side="right", padx=8)

        # ── Ratio slider (hold-out only) ──────────────────────────────
        tk.Frame(frame, bg=C["border"], height=1).pack(fill="x", pady=(12, 0))
        self._ratio_section = tk.Frame(frame, bg=C["panel"])
        self._ratio_section.pack(fill="x", padx=10, pady=8)

        tk.Label(self._ratio_section, text="Val ratio  (hold-out)",
                 font=("SF Pro Display", 10),
                 bg=C["panel"], fg=C["subtext"]).pack(anchor="w")
        self._ratio_label = tk.Label(self._ratio_section,
                                     text=f"{self.val_ratio:.0%}",
                                     font=("SF Pro Display", 11, "bold"),
                                     bg=C["panel"], fg=C["accent2"])
        self._ratio_label.pack(anchor="w")
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TScale", troughcolor=C["border"], background=C["accent"])
        ttk.Scale(self._ratio_section, from_=0.1, to=0.5,
                  variable=self._ratio_var, orient="horizontal",
                  command=self._on_ratio_change).pack(fill="x", pady=(4, 0))

        # ── Fold spinner (CV only) ────────────────────────────────────
        tk.Frame(frame, bg=C["border"], height=1).pack(fill="x", pady=(4, 0))
        self._fold_section = tk.Frame(frame, bg=C["panel"])
        self._fold_section.pack(fill="x", padx=10, pady=8)

        tk.Label(self._fold_section, text="Folds  (cross-val)",
                 font=("SF Pro Display", 10),
                 bg=C["panel"], fg=C["subtext"]).pack(anchor="w")
        spin_row = tk.Frame(self._fold_section, bg=C["panel"])
        spin_row.pack(anchor="w", pady=(4, 0))
        tk.Button(spin_row, text="−", font=("SF Pro Display", 13),
                  bg=C["card"], fg=C["text"], relief="flat", bd=0,
                  padx=8, pady=2, cursor="hand2",
                  command=lambda: self._adj_folds(-1)).pack(side="left")
        self._fold_label = tk.Label(spin_row,
                                    text=str(self._cv_folds_var.get()),
                                    font=("SF Pro Display", 13, "bold"),
                                    bg=C["panel"], fg=C["accent2"],
                                    width=3)
        self._fold_label.pack(side="left", padx=6)
        tk.Button(spin_row, text="+", font=("SF Pro Display", 13),
                  bg=C["card"], fg=C["text"], relief="flat", bd=0,
                  padx=8, pady=2, cursor="hand2",
                  command=lambda: self._adj_folds(+1)).pack(side="left")

    def _adj_folds(self, delta: int) -> None:
        new = max(2, min(20, self._cv_folds_var.get() + delta))
        self._cv_folds_var.set(new)
        self._fold_label.config(text=str(new))
        self._refresh_stats()

    # ── Right panel: session list (hold-out) ──────────────────────────

    def _build_session_panel(self, parent: tk.Frame) -> None:
        self._session_outer = tk.Frame(parent, bg=C["bg"])
        self._session_outer.grid(row=0, column=0, sticky="nsew")
        self._session_outer.rowconfigure(0, weight=1)
        self._session_outer.columnconfigure(0, weight=1)

        canvas = tk.Canvas(self._session_outer, bg=C["bg"],
                           bd=0, highlightthickness=0)
        sb = ttk.Scrollbar(self._session_outer, orient="vertical",
                           command=canvas.yview)
        self._session_frame = tk.Frame(canvas, bg=C["bg"])
        self._session_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=self._session_frame, anchor="nw")
        canvas.configure(yscrollcommand=sb.set)
        canvas.grid(row=0, column=0, sticky="nsew")
        sb.grid(row=0, column=1, sticky="ns")
        canvas.bind_all("<MouseWheel>",
                        lambda e: canvas.yview_scroll(int(-1*(e.delta/120)), "units"))
        self._canvas = canvas
        self._refresh_session_list()

    def _refresh_session_list(self) -> None:
        for w in self._session_frame.winfo_children():
            w.destroy()

        mode = self._mode.get()
        for subj, sessions in self._subject_sessions.items():
            included = self._vp_vars[subj].get()
            color = C["text"] if included else C["excluded"]

            hdr = tk.Frame(self._session_frame, bg=C["bg"])
            hdr.pack(fill="x", pady=(12, 4), padx=4)
            tk.Label(hdr, text=subj, font=("SF Pro Display", 13, "bold"),
                     bg=C["bg"], fg=color).pack(side="left")

            if not included:
                tk.Label(hdr, text="excluded", font=("SF Pro Display", 10),
                         bg=C["bg"], fg=C["excluded"]).pack(side="left", padx=8)
            else:
                if mode == "cv":
                    n_use = sum(1 for s in sessions
                                if self._session_role.get(s.metadata.session_id) == "use")
                    tk.Label(hdr, text=f"  {n_use} selected",
                             font=("SF Pro Display", 10),
                             bg=C["bg"], fg=C["subtext"]).pack(side="left")
                else:
                    n_tr = sum(1 for s in sessions
                               if self._session_role.get(s.metadata.session_id) == "train")
                    n_va = sum(1 for s in sessions
                               if self._session_role.get(s.metadata.session_id) == "val")
                    tk.Label(hdr, text=f"  {n_tr} train  ·  {n_va} val",
                             font=("SF Pro Display", 10),
                             bg=C["bg"], fg=C["subtext"]).pack(side="left")

            for s in sessions:
                role = self._session_role.get(s.metadata.session_id, "excluded")
                self._make_session_row(s, role, included, mode)

    def _make_session_row(self, session: Any, role: str, included: bool, mode: str) -> None:
        sid = session.metadata.session_id

        # Badge config
        display_text = "—"
        display_color = C["excluded"]

        if role == "train":
            display_text = "TRAIN"
            display_color = C["train"]
        elif role == "val":
            display_text = "VAL"
            display_color = C["val"]
        elif role == "use":
            display_text = "USE"
            display_color = C["cv"]

        row_bg = C["card"] if included else C["panel"]

        # Helper to toggle logic
        def _toggle(e, s=sid):
            if not included: return
            cur = self._session_role.get(s, "excluded")

            if mode == "cv":
                # Toggle: use <-> excluded
                new = "excluded" if cur == "use" else "use"
            else:
                # Cycle: train -> val -> excluded -> train
                if cur == "train": new = "val"
                elif cur == "val": new = "excluded"
                else: new = "train"

            self._session_role[s] = new
            self._refresh_session_list()
            self._refresh_stats()

        row = tk.Frame(self._session_frame, bg=row_bg,
                       cursor="hand2" if included else "arrow",
                       pady=5,
                       highlightbackground=C["border"], highlightthickness=1)
        row.pack(fill="x", padx=4, pady=2)

        badge = tk.Label(row, text=f"  {display_text}  ",
                         font=("SF Pro Rounded", 9, "bold"),
                         bg=display_color, fg="#0f0f1a",
                         relief="flat", padx=4, pady=2)
        badge.pack(side="right", padx=8, pady=4)

        lbl = tk.Label(row, text=sid, font=("SF Pro Mono", 11),
                 bg=row_bg,
                 fg=C["text"] if (included and role != "excluded") else C["excluded"])
        lbl.pack(side="left", padx=10)

        if included:
            row.bind("<Button-1>", _toggle)
            badge.bind("<Button-1>", _toggle)
            lbl.bind("<Button-1>", _toggle)

    # ── Right panel: CV info removed (integrated into sessions list) ──

    def _build_cv_panel(self, parent: tk.Frame) -> None:
        pass # Removed

    # ── Bottom bar ────────────────────────────────────────────────────

    def _build_bottom_bar(self) -> None:
        bar = tk.Frame(self.root, bg=C["panel"],
                       highlightbackground=C["border"], highlightthickness=1)
        bar.pack(fill="x", side="bottom")
        inner = tk.Frame(bar, bg=C["panel"])
        inner.pack(fill="x", padx=20, pady=10)

        self._stats_label = tk.Label(inner, text="",
                                     font=("SF Pro Display", 10),
                                     bg=C["panel"], fg=C["subtext"])
        self._stats_label.pack(side="left")
        self._refresh_stats()

        self._flash_label = tk.Label(inner, text="",
                                     font=("SF Pro Display", 10),
                                     bg=C["panel"], fg=C["danger"])
        self._flash_label.pack(side="right", padx=16)

        tk.Button(inner, text="Cancel",
                  font=("SF Pro Display", 11),
                  bg=C["card"], fg=C["subtext"],
                  activebackground=C["hover"], activeforeground=C["text"],
                  relief="flat", padx=16, pady=6, bd=0,
                  command=self._on_cancel).pack(side="right", padx=(8, 0))

        tk.Button(inner, text="▶  Run Comparison",
                  font=("SF Pro Display", 12, "bold"),
                  bg=C["btn_run"], fg="white",
                  activebackground=C["btn_hover"], activeforeground="white",
                  relief="flat", padx=20, pady=7, bd=0,
                  cursor="hand2",
                  command=self._on_run).pack(side="right")

    # ─────────────────────────────────────────────────────────────────
    # Callbacks
    # ─────────────────────────────────────────────────────────────────

    def _on_ratio_change(self, value) -> None:
        self.val_ratio = round(float(value) * 10) / 10
        self._ratio_var.set(self.val_ratio)
        self._ratio_label.config(text=f"{self.val_ratio:.0%}")
        self._recompute_and_refresh()

    def _on_cancel(self) -> None:
        self.result = None
        self.root.destroy()

    def _on_run(self) -> None:
        mode = self._mode.get()

        # In holdout, we check if we have train/val.
        # In CV, we check if we have enough sessions.

        if mode == "cv":
            # Collect sessions marked as 'use'
            sessions = []
            for sessions_list in self._subject_sessions.values():
                for s in sessions_list:
                    if self._session_role.get(s.metadata.session_id) == "use":
                        sessions.append(s)

            if not sessions:
                self._flash("Select at least one session!")
                return

            self.result = {
                "mode": "cv",
                "sessions": sessions,
                "cv_folds": self._cv_folds_var.get(),
            }
        else:
            train, val = self._get_sessions()
            if not train:
                self._flash("No TRAIN sessions!")
                return
            if not val:
                self._flash("No VAL sessions!")
                return
            self.result = {"mode": "holdout", "train": train, "val": val}

        self.root.destroy()

    # ─────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────

    def _refresh_stats(self) -> None:
        mode = self._mode.get()

        # Calculate totals based on roles
        n_use = 0
        n_train = 0
        n_val = 0

        participating_subjs = set()

        for subj, sessions in self._subject_sessions.items():
            for s in sessions:
                role = self._session_role.get(s.metadata.session_id, "excluded")
                if role == "use":
                    n_use += 1
                    participating_subjs.add(subj)
                elif role == "train":
                    n_train += 1
                    participating_subjs.add(subj)
                elif role == "val":
                    n_val += 1
                    participating_subjs.add(subj)

        if mode == "cv":
            folds = self._cv_folds_var.get()
            self._stats_label.config(
                text=(f"Participants: {len(participating_subjs)}   "
                      f"Sessions: {n_use}   "
                      f"Folds: {folds}   "
                      f"~{n_use//folds if folds else 0} val sessions per fold")
            )
        else:
            total = n_train + n_val
            pct = f"{n_val/total:.0%}" if total else "—"
            self._stats_label.config(
                text=(f"Train: {n_train} sessions   "
                      f"Val: {n_val} sessions   "
                      f"Val ratio: {pct}")
            )

    def _flash(self, msg: str) -> None:
        self._flash_label.config(text=msg)
        self.root.after(3000, lambda: self._flash_label.config(text=""))

    def _get_sessions(self) -> Tuple[List[Any], List[Any]]:
        train, val = [], []
        for sessions in self._subject_sessions.values():
            for s in sessions:
                role = self._session_role.get(s.metadata.session_id, "excluded")
                if role == "train":
                    train.append(s)
                elif role == "val":
                    val.append(s)
        return train, val

    def show(self) -> Optional[Dict[str, Any]]:
        self.root.mainloop()
        return self.result


# ═══════════════════════════════════════════════════════════════════════════
# Public helper
# ═══════════════════════════════════════════════════════════════════════════

def pick_sessions(
    dm: DataManager,
    val_ratio: float = 0.2,
) -> Optional[Dict[str, Any]]:
    """
    Open the Session Picker UI.

    Returns a dict:
      {"mode": "holdout", "train": [...], "val": [...]}
      {"mode": "cv",      "sessions": [...], "cv_folds": int}
    Returns None if cancelled.
    """
    return SessionPickerUI(dm, val_ratio=val_ratio).show()


# ═══════════════════════════════════════════════════════════════════════════
# Standalone entry-point
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from performance_assessment.model_comparison import run_comparison

    DATA_DIR = PROJECT_ROOT / "playagain_pipeline" / "data"
    dm = DataManager(DATA_DIR)
    result = pick_sessions(dm)

    if result is None:
        print("Cancelled.")
    elif result["mode"] == "holdout":
        train, val = result["train"], result["val"]
        print(f"\nMode: Hold-out")
        print(f"Train ({len(train)}): {sorted({s.metadata.subject_id for s in train})}")
        print(f"Val   ({len(val)}):   {sorted({s.metadata.subject_id for s in val})}")
        run_comparison(interactive=False,
                       validate_subject_ids=None,
                       validate_session_ids=None,
                       _holdout_sessions=(train, val))
    else:
        sessions = result["sessions"]
        cv_folds = result["cv_folds"]
        print(f"\nMode: Cross-Validation  ({cv_folds} folds)")
        print(f"Sessions ({len(sessions)}): "
              f"{sorted({s.metadata.subject_id for s in sessions})}")
        run_comparison(interactive=False,
                       validate_subject_ids=None,
                       validate_session_ids=None,
                       _cv_sessions_and_folds=(sessions, cv_folds))
