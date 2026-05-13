"""
sweep_runner.py — Parameter sweep engine for the 7-ring cascade
================================================================
Handles:
  - build_param_grid()   : Cartesian product of parameter axes
  - build_linear_sweep() : 1-D sweep of one parameter
  - SweepRunner          : sequential sweep with fault-tolerant registry
  - ParallelSweep        : process-pool sweep (needs multi-seat license)
  - RunRegistry          : JSON-based completed-run tracker

Typical sweeps for this circuit:
  - Coupling coefficient sweep: how do the drop spectra change with κ?
  - Individual ring length perturbations: fabrication variation study
  - Temperature sweep: thermo-optic shift of resonances
"""

import hashlib
import json
import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime           import datetime
from itertools          import product
from pathlib            import Path
from typing             import Any, Callable, Dict, List, Optional

import config

log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Parameter grid builders
# ══════════════════════════════════════════════════════════════════════════════

def build_param_grid(**axes: List[Any]) -> List[Dict[str, Any]]:
    """
    Cartesian product of parameter axes.

    Example
    -------
    >>> grid = build_param_grid(
    ...     coupling = [0.05, 0.10, 0.15, 0.20],
    ...     loss_db  = [0.0, 1.0, 2.0],
    ... )
    >>> # → 12 dicts, one per (coupling, loss_db) combination
    """
    keys, values = list(axes.keys()), list(axes.values())
    return [dict(zip(keys, combo)) for combo in product(*values)]


def build_linear_sweep(param_name: str, values: List[Any], **fixed: Any) -> List[Dict[str, Any]]:
    """
    1-D sweep of one parameter with all others held fixed.

    Example
    -------
    >>> sweep = build_linear_sweep("coupling", [0.05, 0.10, 0.20], loss_db=0.0)
    """
    return [{param_name: v, **fixed} for v in values]


# ══════════════════════════════════════════════════════════════════════════════
# Run registry — fault-tolerant tracking
# ══════════════════════════════════════════════════════════════════════════════

class RunRegistry:
    """
    Persistent JSON file recording which run_ids have completed.
    Enables resume after a crash without re-running finished simulations.
    """

    def __init__(self, registry_path: Path):
        self.path  = Path(registry_path)
        self._data = self._load()

    def _load(self) -> dict:
        if self.path.exists():
            with open(self.path) as f:
                return json.load(f)
        return {"completed": [], "failed": [], "started": str(datetime.now())}

    def _save(self) -> None:
        with open(self.path, "w") as f:
            json.dump(self._data, f, indent=2)

    def is_done(self, run_id: str) -> bool:
        return run_id in self._data["completed"]

    def mark_done(self, run_id: str, params: dict) -> None:
        self._data["completed"].append(run_id)
        self._data.setdefault("log", []).append(
            {"run_id": run_id, "params": params, "time": str(datetime.now())}
        )
        self._save()

    def mark_failed(self, run_id: str, error: str) -> None:
        self._data["failed"].append({"run_id": run_id, "error": error,
                                     "time": str(datetime.now())})
        self._save()

    @property
    def n_completed(self) -> int:
        return len(self._data["completed"])

    @property
    def failed(self) -> list:
        return self._data.get("failed", [])


def _make_run_id(params: dict, prefix: str = "run") -> str:
    """Deterministic 8-char hash of the param dict → stable, human-friendly run ID."""
    canonical = json.dumps(params, sort_keys=True, default=str)
    h = hashlib.md5(canonical.encode()).hexdigest()[:8]
    return f"{prefix}_{h}"


# ══════════════════════════════════════════════════════════════════════════════
# SweepRunner — sequential
# ══════════════════════════════════════════════════════════════════════════════

class SweepRunner:
    """
    Sequential sweep: one INTERCONNECT session opened and closed per run.

    Parameters
    ----------
    simulation_fn : callable(ic, params) → ResultBundle
                    Receives an open ic session and a param dict.
                    Must build the circuit, call ic.run(), extract and return
                    a ResultBundle.
    results_dir   : where to write HDF5/CSV output
    run_id_prefix : prefix for auto-generated run IDs
    resume        : skip already-completed runs (uses RunRegistry)
    """

    def __init__(
        self,
        simulation_fn : Callable,
        results_dir   : Optional[Path] = None,
        run_id_prefix : str            = "run",
        resume        : bool           = True,
    ):
        self.simulation_fn = simulation_fn
        self.results_dir   = Path(results_dir or config.RESULTS_DIR)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.run_id_prefix = run_id_prefix
        self.resume        = resume
        self.registry      = RunRegistry(self.results_dir / "registry.json")

    def run(self, param_grid: List[Dict[str, Any]]) -> List[Any]:
        """
        Execute all parameter combinations and return a list of ResultBundles.
        Failed or skipped runs appear as None in the returned list.
        """
        from session_manager import SessionManager
        from data_extractor  import save_results

        total   = len(param_grid)
        results = []

        log.info(f"SweepRunner: {total} total runs | "
                 f"already done: {self.registry.n_completed} | "
                 f"resume={self.resume}")

        for idx, params in enumerate(param_grid, start=1):
            run_id = _make_run_id(params, prefix=self.run_id_prefix)

            if self.resume and self.registry.is_done(run_id):
                log.info(f"[{idx}/{total}] {run_id} — SKIPPED (already complete)")
                results.append(None)
                continue

            log.info(f"[{idx}/{total}] {run_id} — START  {params}")
            t0 = time.time()

            try:
                with SessionManager(hide=config.HIDE_GUI) as ic:
                    bundle = self.simulation_fn(ic, params)
                    bundle.run_id = run_id

                save_results(bundle, base_dir=self.results_dir)
                self.registry.mark_done(run_id, params)
                log.info(f"[{idx}/{total}] {run_id} — DONE  ({time.time()-t0:.1f}s)")
                results.append(bundle)

            except Exception as exc:
                log.error(f"[{idx}/{total}] {run_id} — FAILED ({time.time()-t0:.1f}s): {exc}")
                self.registry.mark_failed(run_id, str(exc))
                results.append(None)

        n_ok = sum(1 for r in results if r is not None)
        log.info(f"Sweep complete: {n_ok}/{total} succeeded, "
                 f"{len(self.registry.failed)} failed.")
        return results


# ══════════════════════════════════════════════════════════════════════════════
# ParallelSweep — process pool (multi-license)
# ══════════════════════════════════════════════════════════════════════════════

def _parallel_worker(args):
    """Top-level picklable worker for ProcessPoolExecutor."""
    import sys
    sys.path.insert(0, config.LUMERICAL_API_PATH)
    simulation_fn, params, run_id = args
    from session_manager import SessionManager
    from data_extractor  import save_results

    with SessionManager(hide=config.HIDE_GUI) as ic:
        bundle = simulation_fn(ic, params)
        bundle.run_id = run_id
    save_results(bundle)
    return run_id


class ParallelSweep:
    """
    Same as SweepRunner but uses ProcessPoolExecutor for concurrent execution.

    REQUIREMENT: simulation_fn must be a module-level function (not a lambda
    or nested def) so it can be pickled by multiprocessing.

    REQUIREMENT: you must have enough Lumerical license seats for n_workers
    simultaneous sessions.  Default is config.MAX_PARALLEL_SESSIONS = 1.
    """

    def __init__(
        self,
        simulation_fn : Callable,
        n_workers     : int            = None,
        results_dir   : Optional[Path] = None,
        run_id_prefix : str            = "run",
    ):
        self.simulation_fn = simulation_fn
        self.n_workers     = n_workers or config.MAX_PARALLEL_SESSIONS
        self.results_dir   = Path(results_dir or config.RESULTS_DIR)
        self.registry      = RunRegistry(self.results_dir / "registry.json")
        self.run_id_prefix = run_id_prefix

    def run(self, param_grid: List[Dict[str, Any]]) -> List[str]:
        jobs = [
            (self.simulation_fn, params, _make_run_id(params, self.run_id_prefix))
            for params in param_grid
            if not self.registry.is_done(_make_run_id(params, self.run_id_prefix))
        ]
        log.info(f"ParallelSweep: {len(jobs)} jobs on {self.n_workers} workers")
        completed = []

        with ProcessPoolExecutor(max_workers=self.n_workers) as pool:
            future_map = {pool.submit(_parallel_worker, j): j[2] for j in jobs}
            for future in as_completed(future_map):
                run_id = future_map[future]
                try:
                    rid = future.result(timeout=config.SIMULATION_TIMEOUT)
                    self.registry.mark_done(rid, {})
                    completed.append(rid)
                    log.info(f"Worker done: {rid}")
                except Exception as e:
                    log.error(f"Worker failed {run_id}: {e}")
                    self.registry.mark_failed(run_id, str(e))

        return completed
