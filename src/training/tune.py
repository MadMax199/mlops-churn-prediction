"""Hyperparameter tuning for the random forest model."""

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline

from src.config import load_config
from src.data.validation import validate_training_data
from src.features.preprocessing import build_preprocessor
from src.features.schema import (
    FEATURE_COLUMNS,
    ID_COLUMN,
    TARGET_COLUMN,
)
from src.session import get_spark_session
from src.training.train import evaluate


def run() -> None:
    config = load_config().values
    data = config["data"]
    split = config["split"]

    spark = get_spark_session()

    training_columns = [
        ID_COLUMN,
        *FEATURE_COLUMNS,
        TARGET_COLUMN,
    ]

    frame = (
        spark
        .table(data["gold_features_table"])
        .select(*training_columns)
        .toPandas()
    )
    validate_training_data(
        frame,
        ID_COLUMN,
        TARGET_COLUMN,
    )

    X = frame[FEATURE_COLUMNS]
    y = frame[TARGET_COLUMN].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=split["test_size"],
        random_state=split["random_state"],
        stratify=y,
    )

    pipeline = Pipeline([
        (
            "preprocessor",
            build_preprocessor(X_train),
        ),
        (
            "model",
            RandomForestClassifier(
                random_state=split["random_state"],
                n_jobs=1,
            ),
        ),
    ])

    parameter_space = {
        "model__n_estimators": [
            200,
            300,
            500,
        ],
        "model__max_depth": [
            8,
            12,
            16,
            None,
        ],
        "model__min_samples_split": [
            2,
            5,
            10,
            20,
        ],
        "model__min_samples_leaf": [
            1,
            2,
            4,
            8,
        ],
        "model__max_features": [
            "sqrt",
            "log2",
            0.5,
        ],
        "model__class_weight": [
            None,
            "balanced",
            "balanced_subsample",
        ],
    }

    cross_validation = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=split["random_state"],
    )

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=parameter_space,
        n_iter=20,
        scoring="average_precision",
        cv=cross_validation,
        refit=True,
        n_jobs=-1,
        verbose=2,
        random_state=split["random_state"],
        return_train_score=True,
    )

    mlflow.set_tracking_uri(
        config["mlflow"]["tracking_uri"]
    )
    mlflow.set_experiment(
        config["mlflow"]["experiment_name"]
    )

    with mlflow.start_run(
        run_name="random_forest_tuned"
    ):
        search.fit(X_train, y_train)

        best_model = search.best_estimator_

        predictions = best_model.predict(X_test)
        probabilities = best_model.predict_proba(
            X_test
        )[:, 1]

        metrics = evaluate(
            y_test,
            predictions,
            probabilities,
        )

        best_parameters = {
            key.replace("model__", ""): str(value)
            for key, value in search.best_params_.items()
        }

        mlflow.log_params({
            "model": "random_forest_tuned",
            "search_method": "randomized_search",
            "optimization_metric": "average_precision",
            "n_iter": 20,
            "cv_folds": 3,
            "test_size": split["test_size"],
            "random_state": split["random_state"],
            **best_parameters,
        })

        mlflow.log_metrics(metrics)

        mlflow.log_metric(
            "cv_pr_auc",
            search.best_score_,
        )

        cv_results = (
            pd.DataFrame(search.cv_results_)
            .sort_values("rank_test_score")
        )

        mlflow.log_text(
            cv_results.to_csv(index=False),
            "tuning/cv_results.csv",
        )

        # Float-Signatur erlaubt fehlende numerische Werte.
        signature_input = X_train.head(100).copy()

        integer_columns = (
            signature_input
            .select_dtypes(include=["integer"])
            .columns
        )

        signature_input[integer_columns] = (
            signature_input[integer_columns]
            .astype("float64")
        )

        signature = infer_signature(
            signature_input,
            best_model.predict(signature_input),
        )

        mlflow.sklearn.log_model(
            best_model,
            name="model",
            signature=signature,
            input_example=signature_input.head(3),
            serialization_format=(
                mlflow.sklearn
                .SERIALIZATION_FORMAT_SKOPS
            ),
            skops_trusted_types=["numpy.dtype"],
        )

        print("\nBeste Parameter:")
        print(search.best_params_)

        print("\nBeste CV-PR-AUC:")
        print(round(search.best_score_, 4))

        print("\nTestmetriken:")
        print({
            key: round(value, 4)
            for key, value in metrics.items()
        })


if __name__ == "__main__":
    run()