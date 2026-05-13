"""
session_manager.py — Safe INTERCONNECT session lifecycle
=========================================================
Provides a context manager that guarantees the session is always closed,
even if an exception is raised inside the with-block.

Usage:
    from session_manager import SessionManager

    with SessionManager() as ic:
        ic.addelement(...)
        ic.run()
    # ic.close() is called automatically here
"""

import logging
import sys
import time
import traceback
from contextlib import contextmanager
from typing     import Optional

import config

if config.LUMERICAL_API_PATH not in sys.path:
    sys.path.insert(0, config.LUMERICAL_API_PATH)

try:
    import lumapi
except ImportError as e:
    raise ImportError(
        f"Cannot import lumapi from '{config.LUMERICAL_API_PATH}'.\n"
        f"Verify LUMERICAL_API_PATH in config.py.\nOriginal error: {e}"
    )

log = logging.getLogger(__name__)


class SessionManager:
    """
    Context manager for a single INTERCONNECT session.

    Parameters
    ----------
    hide         : True = headless (no GUI). False = GUI visible (for debugging).
    project_file : load an existing .icp file on open.
    save_on_exit : save project to this path before closing.
    max_retries  : number of open attempts before raising.
    retry_delay  : seconds between retry attempts.
    """

    def __init__(
        self,
        hide        : bool          = True,
        project_file: Optional[str] = None,
        save_on_exit: Optional[str] = None,
        max_retries : int           = 3,
        retry_delay : float         = 5.0,
    ):
        self.hide         = hide
        self.project_file = project_file
        self.save_on_exit = save_on_exit
        self.max_retries  = max_retries
        self.retry_delay  = retry_delay
        self._ic          = None

    def _open(self):
        last_exc = None
        for attempt in range(1, self.max_retries + 1):
            try:
                log.info(f"Opening INTERCONNECT session (attempt {attempt}/{self.max_retries}) …")
                ic = lumapi.INTERCONNECT(hide=self.hide)
                if self.project_file:
                    ic.load(self.project_file)
                log.info("Session opened.")
                return ic
            except Exception as exc:
                last_exc = exc
                log.warning(f"Open failed: {exc}")
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay)
        raise RuntimeError(
            f"Could not open INTERCONNECT after {self.max_retries} attempts."
        ) from last_exc

    def _close(self):
        if self._ic is None:
            return
        try:
            if self.save_on_exit:
                self._ic.save(self.save_on_exit)
            self._ic.close()
            log.info("Session closed.")
        except Exception as exc:
            log.error(f"Error closing session: {exc}")
        finally:
            self._ic = None

    def __enter__(self):
        self._ic = self._open()
        return self._ic

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            log.error(f"Exception in session block: {exc_val}")
            log.debug(traceback.format_exc())
        self._close()
        return False   # do not suppress the exception


# ── Smoke test — run this file directly to verify the connection ──────────
if __name__ == "__main__":
    import logger_setup
    print("Testing INTERCONNECT connection …")
    try:
        with SessionManager(hide=config.HIDE_GUI) as ic:
            print(f"  ✓ Session opened")
            ic.addelement("Optical Network Analyzer")
            print(f"  ✓ ONA element added")
        print("  ✓ Session closed\n\nAll checks passed.")
    except Exception as e:
        print(f"\n  ✗ Failed: {e}")
        print("  → Check LUMERICAL_API_PATH in config.py")
