"""
unity_launcher.py
─────────────────
Spawn the Unity PlayAgain executable as a subprocess from Python.

Why this is its own file
────────────────────────
Launching Unity sounds trivial — ``subprocess.Popen(path)`` — but in
practice every environment disagrees about:

  • where the build actually lives (different per OS, per lab machine)
  • which CLI flags to pass (``-batchmode`` for silent, a window
    position on dual-monitor setups, a scene name to jump into)
  • what happens when the user closes it mid-session (SIGTERM? just
    wait for them to re-open?)

Containing that mess in a single helper keeps the GUI's Record tab
focused on the training workflow, and means a single place to update
if the Unity build moves or grows new flags.

Design notes
────────────
  • The path is remembered in QSettings under ``unity/exe_path`` so
    subsequent sessions skip the file picker.
  • ``launch()`` never blocks — the returned Popen is given back to
    the caller so they can check ``poll()`` or kill it from their
    "Stop session" button.
  • We never attach a pipe to stdout/stderr. Unity logs go to its
    own files (``~/AppData/LocalLow/...`` on Windows, ``~/Library/Logs``
    on macOS); trying to drain them here would block on buffered
    writes we can't control.
"""
from __future__ import annotations

import logging
import os
import platform
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Optional, Sequence

from PySide6.QtCore import QSettings

log = logging.getLogger(__name__)

# QSettings namespace — keep consistent with the rest of the pipeline.
_SETTINGS_ORG = "playagain"
_SETTINGS_APP = "pipeline"
_SETTINGS_KEY = "unity/exe_path"


class UnityNotFoundError(FileNotFoundError):
    """Raised when no Unity executable has been configured yet."""


class UnityLauncher:
    """
    Spawn and supervise a Unity PlayAgain build.

    Typical use
    ───────────
        launcher = UnityLauncher()
        if not launcher.has_saved_path():
            path, _ = QFileDialog.getOpenFileName(
                parent, "Locate PlayAgain executable",
                str(Path.home()),
                launcher.file_dialog_filter(),
            )
            if not path:
                return
            launcher.remember_path(path)

        proc = launcher.launch(extra_args=["-screen-fullscreen", "0"])
        # ... later:
        launcher.terminate(proc)
    """

    def __init__(self) -> None:
        self._settings = QSettings(_SETTINGS_ORG, _SETTINGS_APP)
        self._processes: list[subprocess.Popen] = []

    # ------------------------------------------------------------------
    # Path persistence
    # ------------------------------------------------------------------

    def saved_path(self) -> Optional[Path]:
        """Return the cached Unity exe path, or None if never set."""
        raw = self._settings.value(_SETTINGS_KEY, "", type=str)
        if not raw:
            return None
        p = Path(raw)
        return p if p.exists() else None

    def has_saved_path(self) -> bool:
        return self.saved_path() is not None

    def remember_path(self, path: str | Path) -> None:
        """Persist the Unity exe path for next time."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Not a real path: {p}")
        self._settings.setValue(_SETTINGS_KEY, str(p))
        # Force flush so a crash after this call doesn't lose the value.
        self._settings.sync()

    def clear_saved_path(self) -> None:
        self._settings.remove(_SETTINGS_KEY)
        self._settings.sync()

    # ------------------------------------------------------------------
    # Platform helpers
    # ------------------------------------------------------------------

    @staticmethod
    def file_dialog_filter() -> str:
        """
        Return a Qt-style file-filter string appropriate for the host
        OS. Used by the GUI when asking the user to locate the exe.
        """
        system = platform.system()
        if system == "Windows":
            return "Unity executable (*.exe);;All files (*)"
        if system == "Darwin":
            # Users pick the .app bundle; we translate internally.
            return "Unity app bundle (*.app);;All files (*)"
        return "Unity executable (*);;All files (*)"

    @staticmethod
    def _resolve_bundle(path: Path) -> Path:
        """
        On macOS the user picks ``PlayAgain.app``; the actual exe is
        ``PlayAgain.app/Contents/MacOS/PlayAgain``. Resolve transparently
        so callers don't have to care.
        """
        if platform.system() != "Darwin" or path.suffix != ".app":
            return path
        macos_dir = path / "Contents" / "MacOS"
        if not macos_dir.is_dir():
            return path
        # The exe is typically named after the app, but fall back to
        # the first executable file in the directory.
        expected = macos_dir / path.stem
        if expected.exists():
            return expected
        for child in macos_dir.iterdir():
            if child.is_file() and os.access(child, os.X_OK):
                return child
        return path

    # ------------------------------------------------------------------
    # Launch / supervise
    # ------------------------------------------------------------------

    def launch(
        self,
        extra_args: Optional[Sequence[str]] = None,
        path_override: Optional[str | Path] = None,
    ) -> subprocess.Popen:
        """
        Start Unity as a detached subprocess.

        Parameters
        ----------
        extra_args : sequence of str, optional
            Extra CLI args to pass to Unity. ``-screen-fullscreen 0``
            is handy during data collection so the clinician can keep
            the pipeline GUI visible on the second monitor.
        path_override : path-like, optional
            Use this Unity build instead of the remembered one, without
            updating QSettings. Useful for "try before you save".

        Returns
        -------
        subprocess.Popen
            The running process. Caller can keep it around to
            ``terminate()`` it later.
        """
        raw = Path(path_override) if path_override else self.saved_path()
        if raw is None:
            raise UnityNotFoundError(
                "Unity executable path is not configured. "
                "Call remember_path(...) first."
            )
        exe = self._resolve_bundle(raw)
        if not exe.exists():
            raise UnityNotFoundError(f"Unity exe missing: {exe}")

        cmd: list[str] = [str(exe)]
        if extra_args:
            cmd.extend(str(a) for a in extra_args)

        log.info("Launching Unity: %s", " ".join(shlex.quote(c) for c in cmd))

        # Detach: no stdin; stdout/stderr go to /dev/null so we don't
        # block on buffered output or leak file descriptors. We do NOT
        # set start_new_session on Windows — Unity handles its own
        # windowing fine and detaching it would prevent CTRL+C from
        # the parent console from reaching it, which is rarely wanted.
        popen_kwargs: dict = dict(
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # On POSIX, starting a new session prevents a Ctrl+C in the
        # launching terminal from killing Unity; that's usually what
        # the user wants during a data-collection session.
        if platform.system() != "Windows":
            popen_kwargs["start_new_session"] = True

        proc = subprocess.Popen(cmd, **popen_kwargs)
        self._processes.append(proc)
        return proc

    # ------------------------------------------------------------------
    # Teardown
    # ------------------------------------------------------------------

    def is_running(self, proc: subprocess.Popen) -> bool:
        return proc.poll() is None

    def terminate(self, proc: subprocess.Popen, timeout: float = 3.0) -> None:
        """
        Politely ask Unity to close; escalate to SIGKILL if it doesn't.

        Unity usually responds to SIGTERM quickly but large builds with
        unsaved data can hold on; the escalation path matches what the
        child's clinician would do by hand (close it, then Force Quit).
        """
        if proc.poll() is not None:
            return
        try:
            proc.terminate()
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            log.warning("Unity did not exit after %.1fs; killing", timeout)
            proc.kill()
            try:
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                log.error("Unity still alive after SIGKILL — giving up")
        finally:
            try:
                self._processes.remove(proc)
            except ValueError:
                pass

    def terminate_all(self) -> None:
        """Convenience: kill every Unity this launcher has ever started."""
        for proc in list(self._processes):
            self.terminate(proc)
