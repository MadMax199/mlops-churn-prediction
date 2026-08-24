import pandas as pd

from churn_prediction.features import build_preprocessor


def test_preprocessor_handles_missing_and_unknown_values() -> None:
    train = pd.DataFrame(
        {
            "tenure": [1.0, None, 12.0],
            "plan": ["basic", "premium", None],
        }
    )
    inference = pd.DataFrame({"tenure": [4.0], "plan": ["new_plan"]})

    preprocessor = build_preprocessor(train)
    preprocessor.fit(train)
    transformed = preprocessor.transform(inference)

    assert transformed.shape[0] == 1

