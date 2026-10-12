"""Process pool created on the main thread. Request handlers only submit."""

import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor


def pool_workers() -> int:
    cpus = os.cpu_count() or 1
    return max(1, min(4, cpus - 2))


class SimPool:
    def __init__(self) -> None:
        self.ctx = multiprocessing.get_context("spawn")
        self.max_workers = pool_workers()
        self.executor = ProcessPoolExecutor(max_workers=self.max_workers, mp_context=self.ctx)
        self.manager = self.ctx.Manager()

    def shutdown(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=True)
        self.manager.shutdown()
