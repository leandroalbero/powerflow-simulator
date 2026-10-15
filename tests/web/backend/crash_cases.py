"""Pool worker stand-ins. A module of its own so spawned workers can import them."""

import os
import time

from web.backend.services.sim_worker import CaseOutput


def crash_or_run(payload):
    if payload.strategy_id == "no_battery":
        time.sleep(0.3)
        os._exit(1)
    time.sleep(1.5)
    return CaseOutput(payload.job_id, payload.strategy_id, "completed", metrics={"total_cost": 1.0})


def slow(payload):
    time.sleep(30)
    return CaseOutput(payload.job_id, payload.strategy_id, "completed", metrics={"total_cost": 1.0})


def quick(payload):
    return CaseOutput(payload.job_id, payload.strategy_id, "completed", metrics={"total_cost": 1.0})
