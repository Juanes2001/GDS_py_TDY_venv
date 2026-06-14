"""
lumerical_session.py — Robust, OS-aware import of the Ansys Lumerical Python
API (`lumapi`) and a small helper to open a MODE Solutions session.

This isolates the brittle, platform-specific path/DLL handling that the original
notebook repeated at the top of every Lumerical cell.  Importing `lumapi`
correctly on Python 3.8+ requires `os.add_dll_directory` on Windows
(manipulating PATH no longer works for CDLL); this module does that once.

Typical use
-----------
    from .lumerical_session import open_mode
    mode = open_mode(hide_gui=True)
    try:
        ...                       # build geometry, findmodes(), getdata(), ...
    finally:
        mode.close()
"""
from __future__ import annotations

import os
import platform
import sys
from pathlib import Path

from .config import LUMERICAL_VERSION, log


def _resolve_paths(version: str):
    """Return (root, api, bin) install paths for the current OS."""
    if platform.system() == "Windows":
        root = rf"C:\Program Files\Lumerical\{version}"
        api = rf"{root}\api\python"
        binp = rf"{root}\bin"
    else:
        root = f"/opt/lumerical/{version}"
        api = f"{root}/api/python"
        binp = f"{root}/bin"
    return root, api, binp


def import_lumapi(version: str = LUMERICAL_VERSION):
    """
    Import and return the `lumapi` module.

    Inputs
    ------
    version : str
        Lumerical install folder name (e.g. "v202").

    Outputs
    -------
    module
        The imported `lumapi` module object.

    Notes
    -----
    * Clears any previously failed cached import from `sys.modules`.
    * Adds the API folder to `sys.path` and registers the `bin` folder as a DLL
      search directory on Windows (required on Python 3.8+).
    """
    _root, api, binp = _resolve_paths(version)

    if "lumapi" in sys.modules:
        del sys.modules["lumapi"]
    if api not in sys.path:
        sys.path.insert(0, api)
    if platform.system() == "Windows":
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(str(binp))
        else:
            os.environ["PATH"] = str(binp) + ";" + os.environ.get("PATH", "")

    assert Path(api).exists(), (
        f"Lumerical API path not found:\n  {api}\n"
        f"Check LUMERICAL_VERSION = '{version}' in config.py"
    )
    assert Path(binp).exists(), f"Lumerical bin path not found:\n  {binp}"

    import lumapi  # noqa: E402 — must come after the path setup above
    log.info(f"lumapi imported from: {lumapi.__file__}")
    return lumapi


def open_mode(hide_gui: bool = False, version: str = LUMERICAL_VERSION):
    """
    Open and return a Lumerical MODE Solutions session.

    Inputs
    ------
    hide_gui : bool
        True for headless / HPC runs.
    version : str
        Lumerical install folder name.

    Outputs
    -------
    lumapi.MODE
        An open MODE session.  The caller is responsible for `.close()`,
        ideally in a `finally:` block.
    """
    lumapi = import_lumapi(version)
    log.info("Launching Lumerical MODE …")
    return lumapi.MODE(hide=hide_gui)
