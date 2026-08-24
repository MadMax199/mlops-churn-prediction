import pandas as pd

from churn_prediction.config import DataConfig, SplitConfig
from churn_prediction.data import split_features_target, validate_dataset


def test_split_is_reproducible_and_excludes_ids() -> None:
    frame = pd.DataFrame(
        {
            "customer_id": range(20),
            "tenure": range(20),
            "plan": ["basic", "premium"] * 10,
            "churn": [0, 1] * 10,
        }
    )
    data_config = DataConfig(
        source_table="catalog.schema.churn",
        target_column="churn",
        id_columns=["customer_id"],
    )
    split_config = SplitConfig(test_size=0.2, random_state=42)

    validate_dataset(frame, data_config)
    first = split_features_target(frame, data_config, split_config)
    second = split_features_target(frame, data_config, split_config)

    assert "customer_id" not in first[0].columns
    assert first[0].index.tolist() == second[0].index.tolist()
    assert first[2].value_counts().to_dict() == {0: 8, 1: 8}

