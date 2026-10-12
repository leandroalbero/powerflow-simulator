"""Spain 2.0TD off-peak (P3 all day) calendar.

Per CNMC Circular 3/2020, period P3 applies all day on Saturdays, Sundays,
January 6 and national holidays with a fixed date that cannot be substituted by
the regions. Moveable feasts such as Good Friday are NOT included.
"""

from datetime import date

FIXED_NATIONAL_HOLIDAYS = frozenset({
    (1, 1),    # Año Nuevo
    (1, 6),    # Epifanía del Señor
    (5, 1),    # Fiesta del Trabajo
    (8, 15),   # Asunción de la Virgen
    (10, 12),  # Fiesta Nacional de España
    (11, 1),   # Todos los Santos
    (12, 6),   # Día de la Constitución
    (12, 8),   # Inmaculada Concepción
    (12, 25),  # Natividad del Señor
})


def is_offpeak_day(d: date) -> bool:
    return d.weekday() >= 5 or (d.month, d.day) in FIXED_NATIONAL_HOLIDAYS
