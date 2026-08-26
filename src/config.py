from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


@dataclass(frozen=True)
class Settings:
    values: dict[str, Any]

    def section(self, name: str) -> dict[str, Any]:
        return self.values[name]


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> Settings:
    with Path(path).open(encoding="utf-8") as handle:
        values = yaml.safe_load(handle)
    required = {"databricks", "data", "split", "mlflow", "prediction"}
    missing = required.difference(values or {})
    if missing:
        raise ValueError(f"Missing config sections: {sorted(missing)}")
    return Settings(values)

