from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from churn_prediction.config import DataConfig, SplitConfig


def load_databricks_table(spark: Any | None, config: DataConfig) -> pd.DataFrame:
    """Load the configured Databricks table and return a local Pandas frame."""
    if spark is None:
        from churn_prediction.session import get_spark_session

        spark = get_spark_session()
    spark_frame = spark.table(config.source_table)
    frame = spark_frame.toPandas()
    validate_dataset(frame, config)
    return frame


def validate_dataset(frame: pd.DataFrame, config: DataConfig) -> None:
    if frame.empty:
        raise ValueError("The source table is empty")
    if config.target_column not in frame.columns:
        raise ValueError(f"Target column '{config.target_column}' is missing")
    if frame[config.target_column].isna().any():
        raise ValueError("The target column contains missing values")
    if frame[config.target_column].nunique() != 2:
        raise ValueError("The initial pipeline expects a binary churn target")

    configured_columns = set(config.id_columns + config.exclude_columns)
    missing = configured_columns.difference(frame.columns)
    if missing:
        raise ValueError(f"Configured columns are missing: {sorted(missing)}")


def split_features_target(
    frame: pd.DataFrame,
    data_config: DataConfig,
    split_config: SplitConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    drop_columns = [
        data_config.target_column,
        *data_config.id_columns,
        *data_config.exclude_columns,
    ]
    features = frame.drop(columns=drop_columns)
    target = frame[data_config.target_column]

    if features.shape[1] == 0:
        raise ValueError("No feature columns remain after exclusions")

    return train_test_split(
        features,
        target,
        test_size=split_config.test_size,
        random_state=split_config.random_state,
        stratify=target,
    )
