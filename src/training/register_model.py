"""Register the selected MLflow model in Unity Catalog."""

import mlflow
from mlflow import MlflowClient

from src.config import load_config
from src.utils.databricks_auth import configure_databricks_auth


def get_selected_run_id(
    experiment_id: str,
) -> str:
    """Read the selected model run from the latest selection run."""

    runs = mlflow.search_runs(
        experiment_ids=[experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["attributes.start_time DESC"],
    )

    if runs.empty:
        raise ValueError("Keine erfolgreichen MLflow-Runs gefunden.")

    run_name_column = "tags.mlflow.runName"

    selection_runs = runs[runs[run_name_column] == "model_selection"]

    if selection_runs.empty:
        raise ValueError("Kein erfolgreicher model_selection-Run gefunden.")

    latest_selection = selection_runs.sort_values("start_time", ascending=False).iloc[0]

    selected_run_id = latest_selection.get("params.selected_run_id")

    if not selected_run_id:
        raise ValueError("Im model_selection-Run fehlt selected_run_id.")

    return str(selected_run_id)


def get_logged_model(
    experiment_id: str,
    selected_run_id: str,
):
    """Find the logged model associated with the selected run."""

    logged_models = mlflow.search_logged_models(
        experiment_ids=[experiment_id],
        filter_string="status = 'READY'",
        order_by=[
            {
                "field_name": "creation_time",
                "ascending": False,
            }
        ],
        output_format="list",
    )

    matching_models = [
        model
        for model in logged_models
        if (
            getattr(model, "source_run_id", None) == selected_run_id
            or getattr(model, "run_id", None) == selected_run_id
        )
    ]

    if not matching_models:
        available_models = [
            {
                "model_id": model.model_id,
                "source_run_id": getattr(
                    model,
                    "source_run_id",
                    None,
                ),
                "run_id": getattr(
                    model,
                    "run_id",
                    None,
                ),
                "status": str(model.status),
            }
            for model in logged_models
        ]

        raise ValueError(
            "Zum ausgewählten Run wurde kein "
            "erfolgreiches Logged Model gefunden. "
            f"Gefundene Modelle: {available_models}"
        )

    return matching_models[0]


def run() -> None:
    """Register and alias the selected model."""

    configure_databricks_auth()

    config = load_config().values
    mlflow_config = config["mlflow"]

    mlflow.set_tracking_uri(mlflow_config["tracking_uri"])

    mlflow.set_registry_uri("databricks-uc")

    experiment = mlflow.get_experiment_by_name(mlflow_config["experiment_name"])

    if experiment is None:
        raise ValueError("Das konfigurierte MLflow-Experiment wurde nicht gefunden.")

    selected_run_id = get_selected_run_id(experiment.experiment_id)

    logged_model = get_logged_model(
        experiment.experiment_id,
        selected_run_id,
    )

    registered_model_name = mlflow_config["registered_model_name"]

    model_version = mlflow.register_model(
        model_uri=logged_model.model_uri,
        name=registered_model_name,
    )

    client = MlflowClient()

    client.set_registered_model_alias(
        name=registered_model_name,
        alias="Champion",
        version=model_version.version,
    )

    client.set_model_version_tag(
        name=registered_model_name,
        version=model_version.version,
        key="selection_metric",
        value="pr_auc",
    )

    client.set_model_version_tag(
        name=registered_model_name,
        version=model_version.version,
        key="source_run_id",
        value=selected_run_id,
    )

    print("\nModell erfolgreich registriert:")
    print(f"Name:     {registered_model_name}")
    print(f"Version:  {model_version.version}")
    print("Alias:    Champion")
    print(f"Run-ID:   {selected_run_id}")
    print(f"Model-ID: {logged_model.model_id}")


if __name__ == "__main__":
    run()
