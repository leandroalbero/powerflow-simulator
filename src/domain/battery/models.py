class Battery:
    def __init__(self, capacity: float, max_charge_rate: float, max_discharge_rate: float,
                 efficiency: float = 0.95):
        self.capacity = capacity
        self.current_charge = capacity * 0.5
        self.max_charge_rate = max_charge_rate
        self.max_discharge_rate = max_discharge_rate
        self.efficiency = efficiency
        self._charge_taper_start = 0.9
        self._taper_factor = 0.7

    def _get_charge_efficiency(self) -> float:
        charge_level = self.current_charge / self.capacity
        if charge_level < self._charge_taper_start:
            return self.efficiency

        remaining_ratio = (1 - charge_level) / (1 - self._charge_taper_start)
        tapered_efficiency = self.efficiency * (
                (1 - self._taper_factor) +
                self._taper_factor * remaining_ratio
        )
        return max(0.1, tapered_efficiency)

    def _get_max_charge_power(self) -> float:
        charge_level = self.current_charge / self.capacity
        if charge_level < self._charge_taper_start:
            return self.max_charge_rate

        remaining_ratio = (1 - charge_level) / (1 - self._charge_taper_start)
        return self.max_charge_rate * (
                (1 - self._taper_factor) +
                self._taper_factor * remaining_ratio
        )

    def charge(self, power: float, duration: float) -> float:
        efficiency = self._get_charge_efficiency()
        max_charge_power = self._get_max_charge_power()
        energy_space = self.capacity - self.current_charge

        max_energy = min(
            max_charge_power * duration,
            energy_space / efficiency,
            power * duration
        )

        actual_power = max_energy / duration
        self.current_charge += actual_power * duration * efficiency
        return actual_power

    def discharge(self, power: float, duration: float) -> float:
        max_energy = min(
            self.max_discharge_rate * duration,
            self.current_charge * self.efficiency,
            power * duration
        )

        actual_power = max_energy / duration
        self.current_charge -= actual_power * duration / self.efficiency
        return actual_power

    @property
    def charge_level(self) -> float:
        return self.current_charge / self.capacity
