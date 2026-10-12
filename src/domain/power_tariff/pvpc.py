"""PVPC tariff.

Drop-in replacement for ``PowerTariff`` backed by Spain's regulated PVPC
hourly import prices. Export is a flat rate. The public surface mirrors
``PowerTariff`` so ``EnergySimulator`` and every strategy work unchanged.

Prices are keyed by **UTC**, hour-truncated. Incoming ``datetime`` instances
from the simulator are Madrid-local tz-aware; they're converted to UTC on
lookup so DST transitions resolve cleanly to the REE feed.

Rate-type classification is computed **per simulation day** on
``update_datetime``: the day's 24 hourly prices are bucketed into three
equal-count tertiles (cheapest 8 = VALLEY, middle 8 = PLAIN, most
expensive 8 = PEAK). This mirrors strategies that check ``rate_type ==
PEAK`` across several hours — a single min/max hour would break them.
"""

from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

from src.domain.power_tariff.model import EnergyDirection, Rate, RateType

_UTC = timezone.utc


def _to_utc_hour(dt: datetime) -> datetime:
    """Normalise a datetime to an hour-truncated UTC datetime.

    Naive datetimes are treated as UTC. Madrid-local tz-aware datetimes
    convert to UTC before truncation.
    """
    if dt.tzinfo is None:
        aware = dt.replace(tzinfo=_UTC)
    else:
        aware = dt.astimezone(_UTC)
    return aware.replace(minute=0, second=0, microsecond=0)


class PvpcTariff:
    """Hourly PVPC import tariff with flat export rate.

    Mirrors the ``PowerTariff`` public interface used by strategies and the
    simulator: ``get_import_rate``, ``get_export_rate``, ``get_rate``,
    ``update_datetime`` and a ``rate_schedule`` attribute.

    Parameters
    ----------
    prices_eur_kwh:
        Mapping of UTC hour-truncated datetimes to €/kWh import prices.
        Must cover every simulation hour — missing lookups raise.
    export_price:
        Flat €/kWh export price applied to every hour.
    """

    rate_schedule: Dict[Tuple[int, int], Rate]

    def __init__(
        self,
        prices_eur_kwh: Dict[datetime, float],
        export_price: float,
        export_follows_pvpc: bool = False,
        export_pvpc_discount: float = 0.0,
    ) -> None:
        # Normalise price dict keys to UTC hour-truncated for consistent lookup.
        self._prices: Dict[datetime, float] = {
            _to_utc_hour(k): float(v) for k, v in prices_eur_kwh.items()
        }
        self._export_price: float = float(export_price)
        self._export_follows_pvpc: bool = export_follows_pvpc
        self._export_pvpc_discount: float = max(0.0, float(export_pvpc_discount))
        self._current_datetime: Optional[datetime] = None
        self._current_day_key: Optional[Tuple[int, int, int]] = None
        # Per-hour (0-23) Rate for the current Madrid-local day, import only.
        self._day_import_rates: Dict[int, Rate] = {}
        # Synthesized rate_schedule rebuilt per day so rate-aware strategies
        # can iterate it just like PowerTariff.
        self.rate_schedule = {}
        # Seed the schedule and per-hour rates with the first full 24h window
        # available so `BaseRateAwareStrategy.__init__` (which calls
        # `_initialize_rates` before any `update_datetime`) sees a valid 3-tier
        # schedule. These values will be replaced on the first `update_datetime`
        # call with the actual simulation day's prices.
        self._seed_schedule()

    def _seed_schedule(self) -> None:
        if not self._prices:
            return
        first_hour = min(self._prices.keys())
        seed_prices: List[float] = []
        for h in range(24):
            price = self._prices.get(first_hour + timedelta(hours=h))
            if price is None:
                return
            seed_prices.append(price)
        rate_types = _tertile_rate_types(seed_prices)
        self._day_import_rates = {
            h: Rate(
                price=seed_prices[h],
                energy_direction=EnergyDirection.IMPORT,
                _rate_type=rate_types[h],
            )
            for h in range(24)
        }
        self.rate_schedule = self._build_rate_schedule(seed_prices, rate_types)

    # -- public tariff interface -------------------------------------------------

    def update_datetime(self, dt: datetime) -> None:
        """Advance to ``dt`` and refresh the day's rate-type classification.

        Called per simulator step. Does nothing heavy if we're still inside
        the same Madrid-local calendar day.
        """
        self._current_datetime = dt
        day_key = (dt.year, dt.month, dt.day)
        if day_key != self._current_day_key:
            self._rebuild_day_rates(dt)
            self._current_day_key = day_key

    def get_rate(self, hour: int, direction: EnergyDirection) -> Rate:
        """Return the Rate for ``hour`` of the current day in ``direction``.

        IMPORT hours return the PVPC price with the day's tertile rate_type.
        EXPORT hours return the flat export price with rate_type=PLAIN.
        """
        if not 0 <= hour < 24:
            raise ValueError(f"Hour must be between 0 and 23, got {hour}")

        if direction == EnergyDirection.EXPORT:
            if self._export_follows_pvpc:
                import_rate = self._day_import_rates.get(hour)
                if import_rate is None:
                    raise RuntimeError(
                        f"Missing PVPC price for hour {hour}; "
                        "PvpcService should have pre-fetched this range."
                    )
                export_price = max(0.0, import_rate.price - self._export_pvpc_discount)
                return Rate(
                    price=export_price,
                    energy_direction=EnergyDirection.EXPORT,
                    _rate_type=import_rate.rate_type,
                )
            return Rate(
                price=self._export_price,
                energy_direction=EnergyDirection.EXPORT,
                _rate_type=RateType.PLAIN,
            )

        rate = self._day_import_rates.get(hour)
        if rate is None:
            raise RuntimeError(
                f"Missing PVPC price for hour {hour}; "
                "PvpcService should have pre-fetched this range."
            )
        return rate

    def get_import_rate(self, hour: int) -> float:
        return self.get_rate(hour, EnergyDirection.IMPORT).price

    def get_export_rate(self, hour: int) -> float:
        return self.get_rate(hour, EnergyDirection.EXPORT).price

    # -- internals ---------------------------------------------------------------

    def _rebuild_day_rates(self, dt: datetime) -> None:
        """Rebuild per-hour rates and ``rate_schedule`` for the day of ``dt``.

        Uses the calendar date of ``dt`` in its own timezone: for each local
        hour 0..23, converts to UTC and looks up the price. DST boundaries
        are rare and tolerated — if a specific hour has no matching UTC price
        in the cache an error is raised at lookup time.
        """
        base = dt.replace(minute=0, second=0, microsecond=0)
        # Construct a local "midnight" in the same timezone as dt (keeps its
        # tzinfo/offset). For pytz zones this may be off by an hour on DST
        # transition days, but astimezone(UTC) normalises to the correct
        # instant for hours outside the transition window.
        local_midnight = base.replace(hour=0)

        prices_today: List[float] = []
        for h in range(24):
            local_hour = local_midnight.replace(hour=h)
            utc_hour = _to_utc_hour(local_hour)
            price = self._prices.get(utc_hour)
            if price is None:
                raise RuntimeError(
                    f"Missing PVPC price for {utc_hour.isoformat()} "
                    f"(local hour {h} of {local_midnight.date().isoformat()}); "
                    "PvpcService should have pre-fetched this range."
                )
            prices_today.append(price)

        rate_types = _tertile_rate_types(prices_today)
        self._day_import_rates = {
            h: Rate(
                price=prices_today[h],
                energy_direction=EnergyDirection.IMPORT,
                _rate_type=rate_types[h],
            )
            for h in range(24)
        }
        self.rate_schedule = self._build_rate_schedule(prices_today, rate_types)

    @staticmethod
    def _build_rate_schedule(
        prices_today: List[float],
        rate_types: List[RateType],
    ) -> Dict[Tuple[int, int], Rate]:
        """Synthesize a ``(start, end) -> Rate`` schedule for the day.

        Contiguous hours with identical (price, rate_type) collapse into one
        block so strategies iterating ``rate_schedule.values()`` still see a
        useful tariff shape. An EXPORT entry covering 0-24 is not included
        here — exports come from ``get_rate`` directly.
        """
        schedule: Dict[Tuple[int, int], Rate] = {}
        if not prices_today:
            return schedule

        block_start = 0
        block_price = prices_today[0]
        block_type = rate_types[0]
        for h in range(1, 24):
            if prices_today[h] != block_price or rate_types[h] != block_type:
                schedule[(block_start, h)] = Rate(
                    price=block_price,
                    energy_direction=EnergyDirection.IMPORT,
                    _rate_type=block_type,
                )
                block_start = h
                block_price = prices_today[h]
                block_type = rate_types[h]
        schedule[(block_start, 24)] = Rate(
            price=block_price,
            energy_direction=EnergyDirection.IMPORT,
            _rate_type=block_type,
        )
        return schedule


def _tertile_rate_types(prices: List[float]) -> List[RateType]:
    """Classify 24 hourly prices into VALLEY/PLAIN/PEAK by rank tertile.

    Cheapest 8 hours → VALLEY, middle 8 → PLAIN, most expensive 8 → PEAK.
    Ties are broken by original hour order (stable sort), which is fine — the
    boundary choice is arbitrary and rate-aware strategies only need
    *multiple* hours of each tier to operate.
    """
    if len(prices) != 24:
        raise ValueError(f"Expected 24 prices, got {len(prices)}")

    # Rank hours by price ascending; tie-break by hour so it's deterministic.
    ranked = sorted(range(24), key=lambda h: (prices[h], h))
    result: List[RateType] = [RateType.PLAIN] * 24
    for rank, hour in enumerate(ranked):
        if rank < 8:
            result[hour] = RateType.VALLEY
        elif rank < 16:
            result[hour] = RateType.PLAIN
        else:
            result[hour] = RateType.PEAK
    return result
