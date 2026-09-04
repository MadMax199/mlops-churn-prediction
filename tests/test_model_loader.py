"""Tests for model input preparation."""

import pandas as pd

from src.api.model_loader import (
    ModelService,
    STRING_COLUMNS,
)
from src.api.schemas import PredictionRequest
from src.features.schema import FEATURE_COLUMNS


def test_prepare_input_contains_all_features() -> None:
    payload = PredictionRequest(
        canal="web",
        country="US",
        gender=1,
        age_group=3,
        platform="ios",
        order_count=12,
        total_amount=850.5,
        avg_order_amount=70.88,
        total_items=24,
        event_count=130,
        session_count=18,
        days_since_creation=540,
        days_since_last_activity=12,
        days_since_last_transaction=30,
        days_since_last_event=5,
    )

    frame = ModelService.prepare_input(payload)

    assert isinstance(frame, pd.DataFrame)
    assert len(frame) == 1
    assert frame.columns.tolist() == FEATURE_COLUMNS


def test_prepare_input_converts_numeric_features() -> None:
    payload = PredictionRequest(
        order_count=12,
        total_amount=850.5,
    )

    frame = ModelService.prepare_input(payload)

    numeric_columns = [column for column in FEATURE_COLUMNS if column not in STRING_COLUMNS]

    assert all(str(frame[column].dtype) == "float64" for column in numeric_columns)


def test_prepare_input_handles_missing_values() -> None:
    payload = PredictionRequest()

    frame = ModelService.prepare_input(payload)

    assert frame.isna().all().all()
