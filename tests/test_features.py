"""Tests for the feature preprocessing pipeline."""

import numpy as np
import pandas as pd

from src.features.preprocessing import (
    build_preprocessor,
)
from src.features.schema import FEATURE_COLUMNS


CATEGORICAL_COLUMNS = {
    "canal",
    "country",
    "platform",
}


def create_training_features() -> pd.DataFrame:
    """Create representative training features."""

    values = {}

    for feature in FEATURE_COLUMNS:
        if feature in CATEGORICAL_COLUMNS:
            values[feature] = [
                "category_a",
                "category_b",
                None,
            ]
        else:
            values[feature] = [
                1.0,
                None,
                12.0,
            ]

    return pd.DataFrame(
        values,
        columns=FEATURE_COLUMNS,
    )


def create_inference_features() -> pd.DataFrame:
    """Create input containing unknown categories."""

    values = {}

    for feature in FEATURE_COLUMNS:
        if feature in CATEGORICAL_COLUMNS:
            values[feature] = [
                "previously_unknown_category"
            ]
        else:
            values[feature] = [4.0]

    return pd.DataFrame(
        values,
        columns=FEATURE_COLUMNS,
    )


def to_numpy_array(transformed):
    """Convert dense or sparse output to a NumPy array."""

    if hasattr(transformed, "toarray"):
        return transformed.toarray()

    return np.asarray(transformed)


def test_preprocessor_handles_missing_values() -> None:
    train = create_training_features()

    preprocessor = build_preprocessor(train)

    transformed = preprocessor.fit_transform(
        train
    )

    transformed_array = to_numpy_array(
        transformed
    )

    assert transformed.shape[0] == len(train)
    assert np.isfinite(
        transformed_array
    ).all()


def test_preprocessor_handles_unknown_categories() -> None:
    train = create_training_features()
    inference = create_inference_features()

    preprocessor = build_preprocessor(train)
    preprocessor.fit(train)

    transformed = preprocessor.transform(
        inference
    )

    transformed_array = to_numpy_array(
        transformed
    )

    assert transformed.shape[0] == 1
    assert np.isfinite(
        transformed_array
    ).all()


def test_training_and_inference_have_same_width() -> None:
    train = create_training_features()
    inference = create_inference_features()

    preprocessor = build_preprocessor(train)

    transformed_train = (
        preprocessor.fit_transform(train)
    )

    transformed_inference = (
        preprocessor.transform(inference)
    )

    assert (
        transformed_train.shape[1]
        == transformed_inference.shape[1]
    )


def test_preprocessor_preserves_row_count() -> None:
    train = create_training_features()

    preprocessor = build_preprocessor(train)

    transformed = preprocessor.fit_transform(
        train
    )

    assert transformed.shape[0] == train.shape[0]