from __future__ import annotations

import os
from pathlib import Path

from churn_prediction.config import load_config
from churn_prediction.session import get_spark_session


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    config = load_config(project_root / "configs" / "pipeline.yaml")
    source_table = os.getenv("CHURN_SOURCE_TABLE", config.data.source_table)

    spark = get_spark_session()
    current_user = spark.sql("SELECT current_user() AS user").first()["user"]
    row_count = spark.table(source_table).count()

    print(f"Connected as: {current_user}")
    print(f"Source table: {source_table}")
    print(f"Rows: {row_count:,}")
    spark.table(source_table).printSchema()


if __name__ == "__main__":
    main()

