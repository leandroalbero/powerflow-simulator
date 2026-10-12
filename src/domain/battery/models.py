import math


class Battery:
    """Battery with symmetric per-direction losses and a charge-power taper near full.

    ``efficiency`` is the ROUND-TRIP efficiency; each direction loses
    ``sqrt(efficiency)``. Power arguments and return values are AC-side kW:
    charging draws ``p`` and stores ``p * eta_c * dt``; discharging delivers ``p``
    and removes ``p * dt / eta_d`` from storage.
    """

    def __init__(self, capacity: float, max_charge_rate: float, max_discharge_rate: float,
                 efficiency: float = 0.90, initial_soc: float = 0.5,
                 taper_start: float = 0.9, taper_factor: float = 0.7):
        if not 0 < efficiency <= 1:
            raise ValueError(f"Round-trip efficiency must be in (0, 1], got {efficiency}")
        self.capacity = capacity
        self.max_charge_rate = max_charge_rate
        self.max_discharge_rate = max_discharge_rate
        self.efficiency = efficiency
        self.taper_start = taper_start
        self.taper_factor = taper_factor
        self.current_charge = capacity * initial_soc

    @property
    def charge_efficiency(self) -> float:
        return math.sqrt(self.efficiency)

    @property
    def discharge_efficiency(self) -> float:
        return math.sqrt(self.efficiency)

    def max_charge_power(self) -> float:
        """Charge power limit: full rate below ``taper_start``, linearly down to (1 - taper_factor) at 100 %."""
        level = self.current_charge / self.capacity
        if level < self.taper_start or self.taper_start >= 1.0:
            return self.max_charge_rate
        remaining_ratio = max(0.0, (1 - level) / (1 - self.taper_start))
        return self.max_charge_rate * ((1 - self.taper_factor) + self.taper_factor * remaining_ratio)

    def charge(self, power: float, duration: float) -> float:
        """Charge with up to ``power`` kW for ``duration`` h; returns the kW actually drawn."""
        eta = self.charge_efficiency
        space = max(0.0, self.capacity - self.current_charge)
        drawn = max(0.0, min(power, self.max_charge_power(), space / (eta * duration)))
        self.current_charge = min(self.capacity, self.current_charge + drawn * duration * eta)
        return drawn

    def discharge(self, power: float, duration: float, floor_kwh: float = 0.0) -> float:
        """Deliver up to ``power`` kW without storage dropping below ``floor_kwh``; returns kW delivered."""
        eta = self.discharge_efficiency
        available = max(0.0, self.current_charge - floor_kwh)
        delivered = max(0.0, min(power, self.max_discharge_rate, available * eta / duration))
        self.current_charge = max(min(floor_kwh, self.current_charge),
                                  self.current_charge - delivered * duration / eta)
        return delivered

    @property
    def charge_level(self) -> float:
        return self.current_charge / self.capacity
