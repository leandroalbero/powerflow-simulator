"""Private signal raised from a progress callback when a run is cancelled."""


class RunCancelled(Exception):
    """Cooperative cancel. Callers must not treat this as a strategy failure."""

    def __str__(self) -> str:
        return "cancelled"
