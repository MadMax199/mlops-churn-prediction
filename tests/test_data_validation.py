import pandas as pd
import pytest

from src.data.validation import validate_training_data


def test_valid_training_contract():
    validate_training_data(
        pd.DataFrame({"user_id": ["a", "b"], "churn": [0, 1]}), "user_id", "churn"
    )


def test_duplicate_user_is_rejected():
    frame = pd.DataFrame({"user_id": ["a", "a"], "churn": [0, 1]})
    with pytest.raises(ValueError, match="unique"):
        validate_training_data(frame, "user_id", "churn")


def test_non_binary_target_is_rejected():
    frame = pd.DataFrame({"user_id": ["a", "b"], "churn": [0, 2]})
    with pytest.raises(ValueError, match="binary"):
        validate_training_data(frame, "user_id", "churn")
