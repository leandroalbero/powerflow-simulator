"""Battery strategies.

Every strategy only DECIDES a battery command for the step from the tariff
state at ``self.now``; :meth:`BaseEnergyStrategy._settle` turns that command
into physically consistent flows against the actual solar/load.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date, datetime
from typing import Dict, List, NamedTuple, Optional, cast

from src.domain.battery.models import Battery
from src.domain.grid.model import Grid
from src.domain.power_tariff.model import (
    EnergyDirection,
    PowerTariff,
    Rate,
    RateBlock,
    RateType,
    local_date_hour,
)

GHI_TO_PV_FACTOR = 4.592
DEFAULT_CHARGE_TARGET = 1.0
_EPS = 1e-9


@dataclass
class EnergyFlow:
    """Energy exchanged during one step. Every field is in kWh for that step.

    ``battery_charge`` is AC energy drawn into the battery (from solar surplus
    first, then ``battery_charge_from_grid``); ``battery_discharge`` is AC
    energy delivered by it. Per step:
    solar + grid_import + battery_discharge
        == (load - unserved) + grid_export + battery_charge + curtailed.
    ``over_limit`` is the part of grid_import above the grid's max_import,
    caused by load alone (grid charging is never allowed to exceed the cap).
    """

    direct_solar: float = 0.0
    battery_charge: float = 0.0
    battery_discharge: float = 0.0
    grid_import: float = 0.0
    grid_export: float = 0.0
    remaining_solar: float = 0.0
    remaining_load: float = 0.0
    unserved: float = 0.0
    curtailed: float = 0.0
    battery_charge_from_grid: float = 0.0
    over_limit: float = 0.0


class Decision(NamedTuple):
    battery_cmd_kw: float  # > 0 charge, < 0 discharge
    allow_grid_charge: bool = False
    allow_export_discharge: bool = False


class BaseEnergyStrategy(ABC):
    def __init__(self, battery: Battery, grid: Grid, tariff: PowerTariff,
                 min_battery_level: float = 0.1, max_charge_power: Optional[float] = None):
        self.battery = battery
        self.grid = grid
        self.tariff = tariff
        self.min_battery_level = min_battery_level
        self.max_charge_power = battery.max_charge_rate if max_charge_power is None else max_charge_power
        self.now: Optional[datetime] = None

    def set_timestamp(self, ts: datetime) -> None:
        self.now = ts

    @property
    def grid_charge_power(self) -> float:
        return max(0.0, min(self.max_charge_power, self.battery.max_charge_rate))

    def calculate_energy_flows(self, solar_power: float, load_power: float,
                               hour: int, duration: float) -> EnergyFlow:
        """Decide and settle one step; powers in kW, ``duration`` in hours, result in kWh."""
        if duration == 0:
            raise ZeroDivisionError("Duration cannot be zero")
        decision = self._decide(max(0.0, solar_power), max(0.0, load_power), duration)
        return self._settle(max(0.0, solar_power), max(0.0, load_power), decision.battery_cmd_kw, duration,
                            decision.allow_grid_charge, decision.allow_export_discharge)

    @abstractmethod
    def _decide(self, solar_kw: float, load_kw: float, dt_h: float) -> Decision:
        """Battery command for the step at ``self.now``."""

    # -- the one settle primitive ------------------------------------------------

    def _settle(self, solar_kw: float, load_kw: float, battery_cmd_kw: float, dt_h: float,
                allow_grid_charge: bool, allow_export_discharge: bool = False) -> EnergyFlow:
        if dt_h <= 0:
            raise ZeroDivisionError("Duration must be positive")
        solar = max(0.0, solar_kw)
        load = max(0.0, load_kw)
        direct = min(solar, load)
        surplus = solar - direct
        deficit = load - direct
        max_import = self.grid.max_import
        max_export = self.grid.max_export

        discharged = charged = from_solar = from_grid = 0.0
        if battery_cmd_kw < -_EPS:
            want = -battery_cmd_kw
            if allow_export_discharge:
                want = min(want, deficit + max(0.0, max_export - surplus))
            else:
                want = min(want, deficit)
            if want > _EPS:
                floor = self.min_battery_level * self.battery.capacity
                discharged = self.battery.discharge(want, dt_h, floor_kwh=floor)
        elif battery_cmd_kw > _EPS:
            solar_part = min(battery_cmd_kw, surplus)
            grid_part = 0.0
            if allow_grid_charge:
                # The import cap only constrains controllable import: load is always served.
                grid_part = max(0.0, min(battery_cmd_kw - solar_part, max_import - deficit,
                                         self.grid_charge_power))
            if solar_part + grid_part > _EPS:
                charged = self.battery.charge(solar_part + grid_part, dt_h)
                from_solar = min(charged, solar_part)
                from_grid = charged - from_solar

        to_load = min(discharged, deficit)
        load_import = deficit - to_load
        export_avail = (surplus - from_solar) + (discharged - to_load)
        export = min(export_avail, max_export)
        return EnergyFlow(
            direct_solar=direct * dt_h,
            battery_charge=charged * dt_h,
            battery_discharge=discharged * dt_h,
            grid_import=(load_import + from_grid) * dt_h,
            grid_export=export * dt_h,
            curtailed=(export_avail - export) * dt_h,
            battery_charge_from_grid=from_grid * dt_h,
            over_limit=max(0.0, load_import - max_import) * dt_h,
        )

    # -- common decisions --------------------------------------------------------

    @staticmethod
    def _self_consume(solar_kw: float, load_kw: float) -> Decision:
        return Decision(solar_kw - load_kw)

    @staticmethod
    def _hold(solar_kw: float, load_kw: float) -> Decision:
        """Keep the battery for later: absorb solar surplus, never discharge."""
        return Decision(max(0.0, solar_kw - load_kw))

    def _grid_charge_to(self, target_frac: float, solar_kw: float, load_kw: float, dt_h: float) -> Decision:
        """Charge towards ``target_frac`` of capacity, solar surplus first, topping up from the grid."""
        surplus = max(0.0, solar_kw - load_kw)
        need_kwh = target_frac * self.battery.capacity - self.battery.current_charge
        if need_kwh <= _EPS:
            return Decision(surplus)
        need_kw = need_kwh / (self.battery.charge_efficiency * dt_h)
        grid_kw = min(max(0.0, need_kw - surplus), self.grid_charge_power)
        return Decision(surplus + grid_kw, allow_grid_charge=grid_kw > _EPS)

    def _local_date(self) -> date:
        if self.now is None:
            raise RuntimeError(f"{type(self).__name__} needs strategy.now set before each step")
        return local_date_hour(self.now)[0]


class NoBatteryStrategy(BaseEnergyStrategy):
    """Baseline: the battery never moves."""

    def _decide(self, solar_kw: float, load_kw: float, dt_h: float) -> Decision:
        return Decision(0.0)


class SelfConsumeStrategy(BaseEnergyStrategy):
    def _decide(self, solar_kw: float, load_kw: float, dt_h: float) -> Decision:
        return self._self_consume(solar_kw, load_kw)


class BaseRateAwareStrategy(BaseEnergyStrategy):
    """Strategies driven by the import rate blocks of the CURRENT local day."""

    def __init__(self, battery: Battery, grid: Grid, tariff: PowerTariff,
                 min_battery_level: float = 0.1, max_charge_power: Optional[float] = None,
                 valley_charge_target: float = 1.0):
        super().__init__(battery, grid, tariff, min_battery_level, max_charge_power)
        self.valley_charge_target = valley_charge_target
        self._validate_tariff()

    def _validate_tariff(self) -> None:
        import_rates = [rate for rate in self.tariff.rate_schedule.values()
                        if rate.energy_direction == EnergyDirection.IMPORT]
        if len(import_rates) < 3:
            raise ValueError("Tariff must have at least 3 different rates for this strategy")

    # -- block helpers -----------------------------------------------------------

    def _blocks(self, d: Optional[date] = None) -> List[RateBlock]:
        return self.tariff.day_rate_blocks(d or self._local_date())

    def _current_index(self, blocks: List[RateBlock]) -> int:
        hour = local_date_hour(cast(datetime, self.now))[1]
        for i, block in enumerate(blocks):
            if block["start"] <= hour < block["end"]:
                return i
        return len(blocks) - 1

    def _neighbour_block(self, offset: int) -> RateBlock:
        """Block ``offset`` positions from the current one, crossing into the previous/next day."""
        d = self._local_date()
        blocks = self._blocks(d)
        i = self._current_index(blocks) + offset
        if i >= len(blocks):
            return self._blocks(date.fromordinal(d.toordinal() + 1))[0]
        if i < 0:
            return self._blocks(date.fromordinal(d.toordinal() - 1))[-1]
        return blocks[i]

    def _current_rate(self) -> Rate:
        return self.tariff.rate_at(cast(datetime, self.now), EnergyDirection.IMPORT)

    def _is_type(self, rate_type: RateType) -> bool:
        return self._current_rate().rate_type == rate_type

    def _single_rate_day(self) -> bool:
        return len({b["rate"].price for b in self._blocks()}) == 1

    def _pricier_ahead(self) -> bool:
        """True when a strictly pricier import block follows today, or tomorrow before its first valley block."""
        d = self._local_date()
        blocks = self._blocks(d)
        price = self._current_rate().price
        if any(b["rate"].price > price + _EPS for b in blocks[self._current_index(blocks) + 1:]):
            return True
        for b in self._blocks(date.fromordinal(d.toordinal() + 1)):
            if b["rate"].rate_type == RateType.VALLEY:
                return False
            if b["rate"].price > price + _EPS:
                return True
        return False

    def _valley_charge_due(self) -> bool:
        return self._is_type(RateType.VALLEY) and self._pricier_ahead()

    def _idle(self, solar_kw: float, load_kw: float) -> Decision:
        """Outside a strategy's active windows: hold for later, unless the day has a single price."""
        if self._single_rate_day():
            return self._self_consume(solar_kw, load_kw)
        return self._hold(solar_kw, load_kw)


class ForceChargeAtNightStrategy(BaseRateAwareStrategy):
    """Grid-charge to the target in the valley, self-consume the rest of the day."""

    def _decide(self, solar_kw: float, load_kw: float, dt_h: float) -> Decision:
        if self._valley_charge_due():
            return self._grid_charge_to(self.valley_charge_target, solar_kw, load_kw, dt_h)
        return self._self_consume(solar_kw, load_kw)


class ForceChargeAtValleyStrategy(BaseRateAwareStrategy):
    """Grid-charge in the last hour of a shoulder block right before a peak; discharge only at peak."""

    def _is_pre_peak_hour(self) -> bool:
        if self._single_rate_day() or not self._is_type(RateType.PLAIN):
            return False
        blocks = self._blocks()
        current = blocks[self._current_index(blocks)]
        hour = local_date_hour(cast(datetime, self.now))[1]
        nxt = self._neighbour_block(1)["rate"]
        return (hour == current["end"] - 1
                and nxt.rate_type == RateType.PEAK
                and nxt.price > current["rate"].price + _EPS)

    def _decide(self, solar_kw: float, load_kw: float, dt_h: float) -> Decision:
        if self._single_rate_day():
            return self._self_consume(solar_kw, load_kw)
        if self._is_pre_peak_hour():
            return self._grid_charge_to(self.valley_charge_target, solar_kw, load_kw, dt_h)
        if self._is_type(RateType.PEAK):
            return self._self_consume(solar_kw, load_kw)
        return self._idle(solar_kw, load_kw)


class ForceChargeValleyAndPrePeakStrategy(BaseRateAwareStrategy):
    """Grid-charge in valley blocks and in shoulder blocks that precede a peak; discharge at peak."""

    def _should_charge(self) -> bool:
        if self._single_rate_day():
            return False
        if self._is_type(RateType.VALLEY):
            return self._pricier_ahead()
        # Only shoulders skip after a valley: the previous block can be yesterday's all-day
        # off-peak block (Monday, day after a holiday), which never charged.
        if self._neighbour_block(-1)["rate"].rate_type == RateType.VALLEY:
            return False
        nxt = self._neighbour_block(1)["rate"]
        return (self._is_type(RateType.PLAIN)
                and nxt.rate_type == RateType.PEAK
                and nxt.price > self._current_rate().price + _EPS)

    def _decide(self, solar_kw: float, load_kw: float, dt_h: float) -> Decision:
        if self._single_rate_day():
            return self._self_consume(solar_kw, load_kw)
        if self._should_charge():
            return self._grid_charge_to(self.valley_charge_target, solar_kw, load_kw, dt_h)
        if self._is_type(RateType.PEAK):
            return self._self_consume(solar_kw, load_kw)
        return self._idle(solar_kw, load_kw)


class ValleyChargePeakDischargeStrategy(BaseRateAwareStrategy):
    """Charge at valley rate, discharge only during peak hours."""

    def _decide(self, solar_kw: float, load_kw: float, dt_h: float) -> Decision:
        if self._valley_charge_due():
            return self._grid_charge_to(self.valley_charge_target, solar_kw, load_kw, dt_h)
        if self._is_type(RateType.PEAK):
            return self._self_consume(solar_kw, load_kw)
        return self._idle(solar_kw, load_kw)


class SmartDischargeStrategy(BaseRateAwareStrategy):
    """Charge at valley. Discharge everywhere except a shoulder sandwiched between two peaks.

    Holding through that shoulder preserves the battery for the evening peak
    (highest value, no solar).
    """

    def _is_hold_period(self) -> bool:
        return (self._is_type(RateType.PLAIN)
                and self._neighbour_block(-1)["rate"].rate_type == RateType.PEAK
                and self._neighbour_block(1)["rate"].rate_type == RateType.PEAK)

    def _decide(self, solar_kw: float, load_kw: float, dt_h: float) -> Decision:
        if self._valley_charge_due():
            return self._grid_charge_to(self.valley_charge_target, solar_kw, load_kw, dt_h)
        if self._is_type(RateType.VALLEY) or self._is_hold_period():
            return self._idle(solar_kw, load_kw)
        return self._self_consume(solar_kw, load_kw)


class ForecastChargeStrategy(BaseRateAwareStrategy):
    """Like ForceChargeAtNight, but the valley target shrinks with the day's solar forecast."""

    def __init__(self, battery: Battery, grid: Grid, tariff: PowerTariff,
                 daily_forecasts: Dict[date, float], min_battery_level: float = 0.1,
                 max_charge_power: Optional[float] = None, valley_charge_target: float = 1.0,
                 ghi_to_pv_factor: float = GHI_TO_PV_FACTOR):
        super().__init__(battery, grid, tariff, min_battery_level, max_charge_power, valley_charge_target)
        self.daily_forecasts = daily_forecasts
        self.ghi_to_pv_factor = ghi_to_pv_factor

    def _get_charge_target(self) -> float:
        daily_ghi = self.daily_forecasts.get(self._local_date())
        if daily_ghi is None:
            return DEFAULT_CHARGE_TARGET
        expected_solar_kwh = daily_ghi * self.ghi_to_pv_factor / 1000
        target = 1.0 - (expected_solar_kwh / self.battery.capacity)
        return max(self.min_battery_level, min(1.0, target))

    def _decide(self, solar_kw: float, load_kw: float, dt_h: float) -> Decision:
        if self._valley_charge_due():
            return self._grid_charge_to(self._get_charge_target(), solar_kw, load_kw, dt_h)
        return self._self_consume(solar_kw, load_kw)
