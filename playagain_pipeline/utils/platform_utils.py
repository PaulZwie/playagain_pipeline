"""
Platform detection utilities.

Automatically detects the operating system and provides
platform-appropriate paths, imports, and behaviour throughout
the pipeline.
"""

import os
import platform
import sys
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# OS detection
# ---------------------------------------------------------------------------

_SYSTEM = platform.system()  # "Darwin", "Windows", "Linux"

IS_MACOS   = _SYSTEM == "Darwin"
IS_WINDOWS = _SYSTEM == "Windows"
IS_LINUX   = _SYSTEM == "Linux"


def get_os_name() -> str:
    """Return a human-readable OS name."""
    if IS_MACOS:
        return "macOS"
    if IS_WINDOWS:
        return "Windows"
    if IS_LINUX:
        return "Linux"
    return _SYSTEM or "Unknown"


# ---------------------------------------------------------------------------
# Local package resolution
# Resolves the paths to `device_interfaces` and `gui_custom_elements`
# regardless of where the project is cloned on macOS *or* Windows.
# ---------------------------------------------------------------------------

def _find_sibling_package(package_dir_name: str) -> Optional[Path]:
    """
    Locate a local package by searching sibling directories of the
    project root (i.e. the parent of `playagain_pipeline/`).

    Strategy
    --------
    1. Walk up from this file to the repo root (the directory that
       contains ``playagain_pipeline/``).
    2. Look inside the *parent* of that root for a folder whose name
       matches ``package_dir_name`` (case-insensitive on Windows).
    3. Also check a ``vendor/`` sub-folder inside the repo root.
    """
    # This file lives at  <repo_root>/playagain_pipeline/utils/platform_utils.py
    repo_root = Path(__file__).resolve().parent.parent.parent  # Dataprocessing/
    parent    = repo_root.parent                               # Master/

    candidates = [
        parent / package_dir_name,
        repo_root / "vendor" / package_dir_name,
        repo_root.parent.parent / "PlayAgain-Game1" / package_dir_name,  # common dev layout
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return None


def get_device_interfaces_path() -> Optional[Path]:
    """Return the filesystem path to the ``device-interfaces-main`` package."""
    return _find_sibling_package("device-interfaces-main")


def get_gui_custom_elements_path() -> Optional[Path]:
    """Return the filesystem path to the ``gui-custom-elements-main`` package."""
    return _find_sibling_package("gui-custom-elements-main")


# ---------------------------------------------------------------------------
# sys.path injection  (called once at startup)
# ---------------------------------------------------------------------------

def inject_local_packages() -> dict:
    """
    Add the local packages to ``sys.path`` so they can be imported
    without a hard-coded absolute path in requirements.txt.

    Returns a dict with keys ``device_interfaces`` and
    ``gui_custom_elements`` whose values are the resolved Path or None.
    """
    results = {
        "device_interfaces":   None,
        "gui_custom_elements": None,
    }

    di_path = get_device_interfaces_path()
    if di_path and str(di_path) not in sys.path:
        sys.path.insert(0, str(di_path))
        results["device_interfaces"] = di_path

    ge_path = get_gui_custom_elements_path()
    if ge_path and str(ge_path) not in sys.path:
        sys.path.insert(0, str(ge_path))
        results["gui_custom_elements"] = ge_path

    return results


# ---------------------------------------------------------------------------
# Data-directory helpers
# ---------------------------------------------------------------------------

def get_default_data_dir() -> Path:
    """
    Return the platform-appropriate default data directory.

    • macOS / Linux : ``~/Documents/PlayAgain/data``
    • Windows       : ``%USERPROFILE%/Documents/PlayAgain/data``
    """
    return Path.home() / "Documents" / "PlayAgain" / "data"


def get_app_config_dir() -> Path:
    """
    Return the platform-appropriate config directory.

    • macOS  : ``~/Library/Application Support/PlayAgain``
    • Windows: ``%APPDATA%/PlayAgain``
    • Linux  : ``~/.config/PlayAgain``
    """
    if IS_MACOS:
        base = Path.home() / "Library" / "Application Support"
    elif IS_WINDOWS:
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    else:
        base = Path.home() / ".config"

    config_dir = base / "PlayAgain"
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


# ---------------------------------------------------------------------------
# Diagnostic helper
# ---------------------------------------------------------------------------

def print_platform_info() -> None:
    """Print a summary of the detected platform and package availability."""
    di   = get_device_interfaces_path()
    ge   = get_gui_custom_elements_path()

    print(f"[Platform] OS            : {get_os_name()} ({platform.version()})")
    print(f"[Platform] Python        : {sys.version.split()[0]}")
    #print(f"[Platform] device_interfaces  : {'✓ ' + str(di) if di else '✗ not found'}")
    #print(f"[Platform] gui_custom_elements: {'✓ ' + str(ge) if ge else '✗ not found'}")
