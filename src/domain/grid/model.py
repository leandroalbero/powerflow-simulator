class Grid:
    def __init__(self, max_import: float, max_export: float):
        self.max_import = max_import  # kW
        self.max_export = max_export  # kW
        self.imported_energy = 0.0
        self.exported_energy = 0.0

    def import_power(self, power: float, duration: float) -> float:
        actual_power = min(power, self.max_import)
        self.imported_energy += actual_power * duration
        return actual_power

    def export_power(self, power: float, duration: float) -> float:
        actual_power = min(power, self.max_export)
        self.exported_energy += actual_power * duration
        return actual_power
