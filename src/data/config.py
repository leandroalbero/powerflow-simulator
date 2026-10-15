"""Influx credentials from the process environment, then a repo-root .env file."""

import os
from dataclasses import dataclass
from pathlib import Path

_KEYS = ("INFLUX_URL", "INFLUX_DB", "INFLUX_USER", "INFLUX_PASSWORD")


class MissingCredentialError(Exception):
    def __init__(self, name: str) -> None:
        self.name = name
        super().__init__(name)


@dataclass(frozen=True, slots=True)
class InfluxConfig:
    url: str
    db: str
    user: str
    password: str


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_env_file(text: str) -> dict[str, str]:
    """Parse KEY=VALUE lines. Does not expand variables or execute anything."""
    values: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        values[key] = value
    return values


def load_influx_config(env_path: Path | None = None) -> InfluxConfig:
    """Process environment wins over the file. The file is read only for missing keys."""
    file_values: dict[str, str] = {}
    needed = [key for key in _KEYS if not os.environ.get(key)]
    if needed:
        path = env_path if env_path is not None else repo_root() / ".env"
        if path.is_file():
            file_values = parse_env_file(path.read_text())

    def pick(name: str) -> str:
        from_env = os.environ.get(name)
        if from_env:
            return from_env
        from_file = file_values.get(name)
        if from_file:
            return from_file
        raise MissingCredentialError(name)

    url = pick("INFLUX_URL").rstrip("/")
    return InfluxConfig(
        url=url,
        db=pick("INFLUX_DB"),
        user=pick("INFLUX_USER"),
        password=pick("INFLUX_PASSWORD"),
    )
