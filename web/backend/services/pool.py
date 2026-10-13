"""Process pool created on the main thread. Request handlers only submit."""

import logging
import multiprocessing
import os
import threading
import time
from concurrent.futures import ProcessPoolExecutor

logger = logging.getLogger(__name__)
_SHUTDOWN_GRACE_S = 1.0


def pool_workers() -> int:
    cpus = os.cpu_count() or 1
    return max(1, min(4, cpus - 2))


class SimPool:
    def __init__(self) -> None:
        self.ctx = multiprocessing.get_context("spawn")
        self.max_workers = pool_workers()
        self._lock = threading.Lock()
        self._closing = False
        self.executor = self._new_executor()
        self.manager = self.ctx.Manager()

    def _new_executor(self) -> ProcessPoolExecutor:
        return ProcessPoolExecutor(max_workers=self.max_workers, mp_context=self.ctx)

    def recreate(self) -> None:
        """Replace a broken executor. The manager (cancel events, progress queue) stays.

        Safe from a future's done-callback: the stdlib runs those on the executor's
        manager thread while it holds the executor's non-reentrant shutdown lock, so
        the old executor is shut down on a separate thread. Idempotent: every
        observer of one crash calls this, and only the first one swaps.
        """
        with self._lock:
            if self._closing:
                return
            old = self.executor
            if getattr(old, "_broken", False) is False:
                return
            self.executor = self._new_executor()
        threading.Thread(target=_retire, args=(old,), daemon=True).start()

    def shutdown(self) -> None:
        with self._lock:
            self._closing = True
            executor = self.executor
        processes = _processes(executor)
        executor.shutdown(wait=False, cancel_futures=True)
        _stop_processes(processes, grace=_SHUTDOWN_GRACE_S)
        try:
            self.manager.shutdown()
        except Exception:
            logger.exception("sim pool manager shutdown failed")


def _retire(executor: ProcessPoolExecutor) -> None:
    processes = _processes(executor)
    executor.shutdown(wait=False, cancel_futures=True)
    _stop_processes(processes, grace=0.2)


def _processes(executor: ProcessPoolExecutor) -> list:
    raw = getattr(executor, "_processes", None) or {}
    return list(raw.values())


def _stop_processes(processes: list, grace: float) -> None:
    deadline = time.monotonic() + grace
    for proc in processes:
        remaining = deadline - time.monotonic()
        if remaining > 0:
            proc.join(timeout=remaining)
    for proc in processes:
        if proc.is_alive():
            proc.terminate()
    for proc in processes:
        proc.join(timeout=0.5)
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=0.2)
