import numpy as np
import pandas as pd
import pytest

from src.domain.battery.models import Battery
from src.domain.energy_simulator.models import DT_H, run
from src.domain.grid.model import Grid
from src.domain.power_tariff.model import EnergyDirection
from src.domain.strategy.model import (
    ForceChargeAtNightStrategy,
    ForceChargeAtValleyStrategy,
    ForceChargeValleyAndPrePeakStrategy,
    ForecastChargeStrategy,
    NoBatteryStrategy,
    SelfConsumeStrategy,
    SmartDischargeStrategy,
    ValleyChargePeakDischargeStrategy,
)
from tests.domain.strategy.conftest import make_td20
from tests.synthetic import synthetic_frame

HEURISTICS = {
    "no_battery": NoBatteryStrategy,
    "self_consume": SelfConsumeStrategy,
    "charge_night": ForceChargeAtNightStrategy,
    "force_valleys": ForceChargeAtValleyStrategy,
    "force_valleys_pre_peak": ForceChargeValleyAndPrePeakStrategy,
    "valley_charge_peak_discharge": ValleyChargePeakDischargeStrategy,
    "smart_discharge": SmartDischargeStrategy,
    "forecast_charge": ForecastChargeStrategy,
}


def make(cls, initial_soc=0.5, min_soc=0.1):
    battery = Battery(capacity=10.0, max_charge_rate=4.5, max_discharge_rate=4.6, efficiency=0.9,
                      initial_soc=initial_soc)
    tariff = make_td20()
    kwargs = {"daily_forecasts": {}} if cls is ForecastChargeStrategy else {}
    return cls(battery, Grid(max_import=5.5, max_export=5.5), tariff, min_battery_level=min_soc, **kwargs), tariff


@pytest.fixture(scope="module")
def frame():
    return synthetic_frame()


@pytest.mark.parametrize("sid", list(HEURISTICS))
def test_energy_balance_and_limits_every_step(frame, sid):
    strategy, tariff = make(HEURISTICS[sid])
    res = run(frame, strategy, tariff)
    ts = res.timeseries
    v = frame["valid"].to_numpy()
    b = strategy.battery

    solar = ts["solar_kw"][v] * DT_H
    load = ts["load_kw"][v] * DT_H
    imp, exp = ts["grid_import_kw"][v] * DT_H, ts["grid_export_kw"][v] * DT_H
    ch, dis = ts["battery_charge_kw"][v] * DT_H, ts["battery_discharge_kw"][v] * DT_H
    curt = ts["curtailed_kw"][v] * DT_H
    np.testing.assert_allclose(solar + imp + dis, load + exp + ch + curt, atol=1e-9)

    level = ts["battery_level_kwh"]
    prev = np.concatenate(([res.metrics["initial_battery_kwh"]], level[:-1]))
    np.testing.assert_allclose((level - prev)[v], ch * b.charge_efficiency - dis / b.discharge_efficiency, atol=1e-9)
    np.testing.assert_allclose((level - prev)[~v], 0.0, atol=1e-12)

    assert level.min() >= 0.1 * b.capacity - 1e-9
    gc = ts["grid_charge_kw"][v]
    load_import = ts["grid_import_kw"][v] - gc
    assert np.all(gc <= np.maximum(0.0, 5.5 - load_import) + 1e-9)
    assert res.metrics["unserved_kwh"] == 0.0
    assert res.metrics["over_limit_kwh"] > 0  # the 11 kW spike exceeds cap + battery


def test_invalid_rows_are_skipped_but_present(frame):
    strategy, tariff = make(SelfConsumeStrategy)
    res = run(frame, strategy, tariff)
    v = frame["valid"].to_numpy()
    assert len(res.timeseries["ts"]) == len(frame)
    for key in ("grid_import_kw", "grid_export_kw", "solar_kw", "load_kw", "step_cost"):
        assert np.isnan(res.timeseries[key][~v]).all()
        assert not np.isnan(res.timeseries[key][v]).any()
    assert res.metrics["excluded_hours"] == pytest.approx(2.0)
    assert res.metrics["valid_hours"] == pytest.approx(70.0)
    assert res.metrics["total_house_consumption"] == pytest.approx(np.nansum(frame["load_kw"]) / 60)


def test_cost_uses_rate_at_each_step(frame):
    strategy, tariff = make(NoBatteryStrategy)
    res = run(frame, strategy, tariff)
    ts = res.timeseries
    idx = frame.index
    for i in range(0, len(idx), 97):
        assert ts["import_price"][i] == tariff.rate_at(idx[i], EnergyDirection.IMPORT).price
        assert ts["export_price"][i] == tariff.rate_at(idx[i], EnergyDirection.EXPORT).price
    v = frame["valid"].to_numpy()
    net = (frame["load_kw"] - frame["pv_kw"]).to_numpy()[v] * DT_H
    expected = np.sum(np.clip(net, 0, None) * ts["import_price"][v] - np.clip(-net, 0, None) * ts["export_price"][v])
    assert res.metrics["total_cost"] == pytest.approx(expected, abs=1e-6)
    assert res.metrics["total_cost"] == pytest.approx(res.metrics["import_cost"] - res.metrics["export_revenue"])
    assert sum(m["cost"] for m in res.metrics["monthly"]) == pytest.approx(res.metrics["total_cost"], abs=1e-4)


def test_terminal_value_and_adjusted_cost(frame):
    strategy, tariff = make(ForceChargeAtNightStrategy, initial_soc=0.1)
    res = run(frame, strategy, tariff)
    m = res.metrics
    v = frame["valid"].to_numpy()
    mean_rate = res.timeseries["import_price"][v].mean()
    delta = m["final_battery_kwh"] - m["initial_battery_kwh"]
    assert m["stored_energy_delta_kwh"] == pytest.approx(delta, abs=1e-6)
    assert m["terminal_value_eur"] == pytest.approx(delta * mean_rate * strategy.battery.discharge_efficiency,
                                                    abs=1e-6)
    assert m["adjusted_cost"] == pytest.approx(m["total_cost"] - m["terminal_value_eur"], abs=1e-6)
    assert m["equivalent_full_cycles"] == pytest.approx(m["total_battery_out"] / (10.0 * 0.9), abs=1e-6)
    assert m["ev_kwh"] == 0.0


def test_no_battery_costs_at_least_self_consume(frame):
    nb, t1 = make(NoBatteryStrategy)
    sc, t2 = make(SelfConsumeStrategy)
    assert run(frame, nb, t1).metrics["adjusted_cost"] >= run(frame, sc, t2).metrics["adjusted_cost"]


def test_progress_reaches_100(frame):
    seen = []
    strategy, tariff = make(SelfConsumeStrategy)
    run(frame.iloc[:500], strategy, tariff, on_progress=seen.append)
    assert seen[-1] == 100 and seen == sorted(seen)


def test_empty_frame():
    strategy, tariff = make(SelfConsumeStrategy)
    empty = synthetic_frame().iloc[:0]
    res = run(empty, strategy, tariff)
    assert res.metrics["total_cost"] == 0.0
    assert res.metrics["monthly"] == []


def test_dst_day_runs_on_fixed_grid():
    frame = synthetic_frame("2026-03-28", days=3, outage=False)
    assert len(frame) == (24 + 23 + 24) * 60
    strategy, tariff = make(ForceChargeAtNightStrategy)
    res = run(frame, strategy, tariff)
    assert pd.Series(res.timeseries["ts"]).is_monotonic_increasing
    assert res.metrics["valid_hours"] == pytest.approx(71.0)


def _planner(sid):
    from src.domain.strategy.dqn.agent import DqnAgent
    from src.domain.strategy.dqn_strategy import DqnStrategy
    from src.domain.strategy.mpc import LoadForecaster, MpcStrategy, PerfectSolarForecaster
    from src.domain.strategy.oracle import OracleStrategy

    frame = synthetic_frame()
    battery = Battery(capacity=10.0, max_charge_rate=4.5, max_discharge_rate=4.6, efficiency=0.9, initial_soc=0.5)
    grid = Grid(max_import=5.5, max_export=5.5)
    tariff = make_td20()
    if sid == "oracle":
        s = OracleStrategy(battery, grid, tariff, frame)
    elif sid == "mpc":
        history = synthetic_frame("2026-02-01", days=40, outage=False, spike=False)
        s = MpcStrategy(battery, grid, tariff, LoadForecaster(history["load_kw"]),
                        PerfectSolarForecaster(frame["pv_kw"]), step_minutes=15, resolve_minutes=60)
    else:
        s = DqnStrategy(battery, grid, tariff, agent=DqnAgent(state_dim=16, action_dim=9))
    return frame, s, tariff


@pytest.mark.parametrize("sid", ["oracle", "mpc", "dqn"])
def test_energy_balance_for_planning_strategies(sid):
    frame, strategy, tariff = _planner(sid)
    res = run(frame, strategy, tariff)
    ts = res.timeseries
    v = frame["valid"].to_numpy()
    b = strategy.battery
    lhs = (ts["solar_kw"] + ts["grid_import_kw"] + ts["battery_discharge_kw"])[v]
    rhs = (ts["load_kw"] + ts["grid_export_kw"] + ts["battery_charge_kw"] + ts["curtailed_kw"])[v]
    np.testing.assert_allclose(lhs, rhs, atol=1e-9)
    level = ts["battery_level_kwh"]
    prev = np.concatenate(([res.metrics["initial_battery_kwh"]], level[:-1]))
    delta = (ts["battery_charge_kw"] * b.charge_efficiency - ts["battery_discharge_kw"] / b.discharge_efficiency) / 60
    np.testing.assert_allclose((level - prev)[v], delta[v], atol=1e-9)
    assert level.min() >= 0.1 * b.capacity - 1e-9
    gc = ts["grid_charge_kw"][v]
    assert np.all(gc <= np.maximum(0.0, 5.5 - (ts["grid_import_kw"][v] - gc)) + 1e-9)
    assert res.metrics["unserved_kwh"] == 0.0
