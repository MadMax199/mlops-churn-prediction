"""Tests for training data validation and splitting."""

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import load_config
from src.data.validation import validate_training_data
from src.features.schema import (
    FEATURE_COLUMNS,
    ID_COLUMN,
    TARGET_COLUMN,
)


CATEGORICAL_COLUMNS = {
    "canal",
    "country",
    "platform",
}


def create_training_frame(
    number_of_rows: int = 20,
) -> pd.DataFrame:
    """Create representative training data."""

    frame = pd.DataFrame(
        {
            ID_COLUMN: range(number_of_rows),
            TARGET_COLUMN: [0, 1] * (number_of_rows // 2),
        }
    )

    for feature in FEATURE_COLUMNS:
        if feature in CATEGORICAL_COLUMNS:
            frame[feature] = [
                "category_a",
                "category_b",
            ] * (number_of_rows // 2)
        else:
            frame[feature] = [float(value) for value in range(number_of_rows)]

    return frame


def split_training_data(
    frame: pd.DataFrame,
):
    """Apply the split used by the training pipeline."""

    config = load_config().values
    split_config = config["split"]

    features = frame[FEATURE_COLUMNS]
    target = frame[TARGET_COLUMN].astype(int)

    return train_test_split(
        features,
        target,
        test_size=split_config["test_size"],
        random_state=split_config["random_state"],
        stratify=target,
    )


def test_training_data_is_valid() -> None:
    frame = create_training_frame()

    validate_training_data(
        frame,
        ID_COLUMN,
        TARGET_COLUMN,
    )


def test_feature_columns_exclude_id_and_target() -> None:
    assert ID_COLUMN not in FEATURE_COLUMNS
    assert TARGET_COLUMN not in FEATURE_COLUMNS


def test_split_is_reproducible() -> None:
    frame = create_training_frame()

    first_split = split_training_data(frame)
    second_split = split_training_data(frame)

    first_X_train = first_split[0]
    first_X_test = first_split[1]
    second_X_train = second_split[0]
    second_X_test = second_split[1]

    assert first_X_train.index.tolist() == second_X_train.index.tolist()

    assert first_X_test.index.tolist() == second_X_test.index.tolist()


def test_split_excludes_id_and_target() -> None:
    frame = create_training_frame()

    X_train, X_test, _, _ = split_training_data(frame)

    assert ID_COLUMN not in X_train.columns
    assert TARGET_COLUMN not in X_train.columns
    assert ID_COLUMN not in X_test.columns
    assert TARGET_COLUMN not in X_test.columns


def test_split_preserves_target_distribution() -> None:
    frame = create_training_frame()

    _, _, y_train, y_test = split_training_data(frame)

    train_distribution = y_train.value_counts(normalize=True).sort_index()

    test_distribution = y_test.value_counts(normalize=True).sort_index()

    pd.testing.assert_series_equal(
        train_distribution,
        test_distribution,
        check_names=False,
    )
