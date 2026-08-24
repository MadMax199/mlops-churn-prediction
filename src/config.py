from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DataConfig:
    source_table: str
    target_column: str
    id_columns: list[str] = field(default_factory=list)
    exclude_columns: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class SplitConfig:
    test_size: float = 0.2
    random_state: int = 42


@dataclass(frozen=True)
class PipelineConfig:
    data: DataConfig
    split: SplitConfig


def load_config(path: str | Path) -> PipelineConfig:
    with Path(path).open(encoding="utf-8") as config_file:
        raw: dict[str, Any] = yaml.safe_load(config_file)

    data = DataConfig(**raw["data"])
    split = SplitConfig(**raw.get("split", {}))

    if not 0 < split.test_size < 1:
        raise ValueError("split.test_size must be between 0 and 1")
    if data.source_table.startswith("REPLACE_WITH_"):
        raise ValueError("Set data.source_table in configs/pipeline.yaml")

    return PipelineConfig(data=data, split=split)

