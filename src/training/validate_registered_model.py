"""Validate the registered Champion model."""

import mlflow
import pandas as pd
from mlflow import MlflowClient

from src.config import load_config
from src.features.schema import (
    FEATURE_COLUMNS,
    ID_COLUMN,
)
from src.session import get_spark_session
from src.utils.databricks_auth import configure_databricks_auth


MODEL_ALIAS = "Champion"
SAMPLE_SIZE = 100


def prepare_input(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Prepare input data according to the MLflow signature."""

    features = frame[FEATURE_COLUMNS].copy()

    integer_columns = features.select_dtypes(
        include=["integer"]
    ).columns

    features[integer_columns] = (
        features[integer_columns]
        .astype("float64")
    )

    return features


def validate_predictions(
    predictions,
    probabilities,
) -> None:
    """Validate model outputs."""

    predicted_classes = set(
        pd.Series(predictions)
        .astype(int)
        .unique()
    )

    if not predicted_classes.issubset({0, 1}):
        raise ValueError(
            "Das Modell erzeugt ungültige Klassen: "
            f"{predicted_classes}"
        )

    probability_series = pd.Series(
        probabilities
    )

    if not probability_series.between(0, 1).all():
        raise ValueError(
            "Mindestens eine Wahrscheinlichkeit "
            "liegt außerhalb des Bereichs [0, 1]."
        )

    if probability_series.isna().any():
        raise ValueError(
            "Die Vorhersagen enthalten fehlende "
            "Wahrscheinlichkeiten."
        )


def run() -> None:
    """Load and validate the registered model."""

    configure_databricks_auth()

    config = load_config().values
    mlflow_config = config["mlflow"]

    mlflow.set_tracking_uri(
        mlflow_config["tracking_uri"]
    )

    mlflow.set_registry_uri(
        "databricks-uc"
    )

    mlflow.set_experiment(
        mlflow_config["experiment_name"]
    )

    registered_model_name = (
        mlflow_config["registered_model_name"]
    )

    model_uri = (
        f"models:/{registered_model_name}"
        f"@{MODEL_ALIAS}"
    )

    client = MlflowClient()

    model_version = (
        client.get_model_version_by_alias(
            name=registered_model_name,
            alias=MODEL_ALIAS,
        )
    )

    print("\nRegistriertes Modell:")
    print(f"Name:    {registered_model_name}")
    print(f"Alias:   {MODEL_ALIAS}")
    print(f"Version: {model_version.version}")
    print(f"URI:     {model_uri}")

    spark = get_spark_session()

    sample_frame = (
        spark
        .table(
            config["data"]["gold_features_table"]
        )
        .select(
            ID_COLUMN,
            *FEATURE_COLUMNS,
        )
        .orderBy(ID_COLUMN)
        .limit(SAMPLE_SIZE)
        .toPandas()
    )

    if sample_frame.empty:
        raise ValueError(
            "Die Gold-Tabelle enthält keine Daten."
        )

    model_input = prepare_input(
        sample_frame
    )

    print(
        f"\nValidierungsdatensätze: "
        f"{len(model_input)}"
    )

    sklearn_model = mlflow.sklearn.load_model(
        model_uri
    )

    pyfunc_model = mlflow.pyfunc.load_model(
        model_uri
    )

    sklearn_predictions = (
        sklearn_model.predict(model_input)
    )

    probabilities = (
        sklearn_model
        .predict_proba(model_input)[:, 1]
    )

    pyfunc_predictions = (
        pd.Series(
            pyfunc_model.predict(model_input)
        )
        .astype(int)
        .to_numpy()
    )

    validate_predictions(
        sklearn_predictions,
        probabilities,
    )

    prediction_agreement = float(
        (
            sklearn_predictions
            == pyfunc_predictions
        ).mean()
    )

    if prediction_agreement != 1.0:
        raise ValueError(
            "Sklearn- und PyFunc-Modell erzeugen "
            "unterschiedliche Vorhersagen."
        )

    results = pd.DataFrame({
        "sample_row": range(
            1,
            len(model_input) + 1,
        ),
        "prediction": sklearn_predictions,
        "churn_probability": probabilities,
    })

    validation_summary = {
        "registered_model_name": (
            registered_model_name
        ),
        "model_alias": MODEL_ALIAS,
        "model_version": str(
            model_version.version
        ),
        "model_uri": model_uri,
        "sample_size": len(model_input),
        "prediction_agreement": (
            prediction_agreement
        ),
        "minimum_probability": float(
            results["churn_probability"].min()
        ),
        "maximum_probability": float(
            results["churn_probability"].max()
        ),
        "predicted_churners": int(
            results["prediction"].sum()
        ),
    }

    with mlflow.start_run(
        run_name="registered_model_validation"
    ):
        mlflow.set_tags({
            "pipeline_stage": "model_validation",
            "registered_model": (
                registered_model_name
            ),
            "model_alias": MODEL_ALIAS,
            "model_version": str(
                model_version.version
            ),
        })

        mlflow.log_params({
            "registered_model": (
                registered_model_name
            ),
            "model_alias": MODEL_ALIAS,
            "model_version": str(
                model_version.version
            ),
            "sample_size": SAMPLE_SIZE,
        })

        mlflow.log_metrics({
            "prediction_agreement": (
                prediction_agreement
            ),
            "minimum_probability": (
                validation_summary[
                    "minimum_probability"
                ]
            ),
            "maximum_probability": (
                validation_summary[
                    "maximum_probability"
                ]
            ),
            "predicted_churners": (
                validation_summary[
                    "predicted_churners"
                ]
            ),
        })

        mlflow.log_text(
            results.to_csv(index=False),
            "validation/predictions.csv",
        )

        mlflow.log_dict(
            validation_summary,
            "validation/summary.json",
        )

    print("\nBeispielvorhersagen:")
    print(
        results.head(10).to_string(
            index=False
        )
    )

    print("\nValidierung erfolgreich:")
    print(
        f"Übereinstimmung: "
        f"{prediction_agreement:.0%}"
    )
    print(
        "Alle Klassen und Wahrscheinlichkeiten "
        "sind gültig."
    )


if __name__ == "__main__":
    run()