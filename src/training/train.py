import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.config import load_config
from src.data.validation import validate_training_data
from src.features.preprocessing import build_preprocessor
from src.features.schema import FEATURE_COLUMNS, ID_COLUMN, TARGET_COLUMN
from src.session import get_spark_session


def evaluate(y_true, predictions, probabilities) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1": f1_score(y_true, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probabilities),
        "pr_auc": average_precision_score(y_true, probabilities),
    }


def run() -> None:
    config = load_config().values
    data = config["data"]
    split = config["split"]
    frame: pd.DataFrame = get_spark_session().table(data["gold_features_table"]).toPandas()
    validate_training_data(frame, ID_COLUMN, TARGET_COLUMN)
    X, y = frame[FEATURE_COLUMNS], frame[TARGET_COLUMN].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=split["test_size"], random_state=split["random_state"], stratify=y
    )
    models = {
        "dummy": DummyClassifier(strategy="prior"),
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=250,
            max_depth=12,
            class_weight="balanced",
            n_jobs=-1,
            random_state=split["random_state"],
        ),
    }

    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    for name, estimator in models.items():
        pipeline = Pipeline([("preprocessor", build_preprocessor(X_train)), ("model", estimator)])
        with mlflow.start_run(run_name=name):
            pipeline.fit(X_train, y_train)
            predictions = pipeline.predict(X_test)
            probabilities = pipeline.predict_proba(X_test)[:, 1]
            metrics = evaluate(y_test, predictions, probabilities)
            mlflow.log_params(
                {
                    "model": name,
                    "artifact_type": "sklearn_pipeline",
                    "serving_candidate": str(name != "dummy").lower(),
                    "test_size": split["test_size"],
                    "random_state": split["random_state"],
                }
            )
            mlflow.log_metrics(metrics)
            signature = infer_signature(X_train, pipeline.predict(X_train))
            mlflow.sklearn.log_model(
                pipeline,
                name="model",
                signature=signature,
                input_example=X_train.head(3),
                serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_SKOPS,
                skops_trusted_types=["numpy.dtype"],
            )
            print(name, {key: round(value, 4) for key, value in metrics.items()})


if __name__ == "__main__":
    run()
