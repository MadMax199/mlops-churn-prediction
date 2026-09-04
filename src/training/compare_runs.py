"""Compare MLflow model runs and persist the selection result."""

"""Compare MLflow model runs and persist the selection result."""

import mlflow
import pandas as pd

from src.config import load_config
from src.utils.databricks_auth import configure_databricks_auth


RUN_NAMES = [
    "random_forest",
    "random_forest_tuned",
]

MODEL_METRICS = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "pr_auc",
]

COMPARISON_METRICS = [
    *MODEL_METRICS,
    "cv_pr_auc",
]


def get_latest_successful_runs(
    experiment_id: str,
) -> pd.DataFrame:
    """Load the latest successful run for each model."""

    runs = mlflow.search_runs(
        experiment_ids=[experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["attributes.start_time DESC"],
    )

    if runs.empty:
        raise ValueError("Im MLflow-Experiment wurden keine erfolgreichen Runs gefunden.")

    run_name_column = "tags.mlflow.runName"

    if run_name_column not in runs.columns:
        raise ValueError("Die MLflow-Runs enthalten keine Run-Namen.")

    runs["run_name"] = runs[run_name_column]

    selected_runs = (
        runs[runs["run_name"].isin(RUN_NAMES)]
        .sort_values(
            "start_time",
            ascending=False,
        )
        .drop_duplicates(
            subset=["run_name"],
            keep="first",
        )
    )

    missing_runs = set(RUN_NAMES) - set(selected_runs["run_name"])

    if missing_runs:
        raise ValueError(f"Folgende erfolgreichen Runs fehlen: {sorted(missing_runs)}")

    return selected_runs.set_index("run_name").loc[RUN_NAMES].reset_index()


def build_comparison(
    selected_runs: pd.DataFrame,
) -> pd.DataFrame:
    """Create a compact model comparison table."""

    metric_columns = [
        f"metrics.{metric}"
        for metric in COMPARISON_METRICS
        if f"metrics.{metric}" in selected_runs.columns
    ]

    comparison = selected_runs[
        [
            "run_name",
            "run_id",
            "start_time",
            *metric_columns,
        ]
    ].copy()

    comparison = comparison.rename(
        columns={f"metrics.{metric}": metric for metric in COMPARISON_METRICS}
    )

    available_metrics = [metric for metric in COMPARISON_METRICS if metric in comparison.columns]

    comparison[available_metrics] = comparison[available_metrics].round(4)

    return comparison


def get_metrics(
    run_data: pd.Series,
) -> dict[str, float]:
    """Extract evaluation metrics from one run."""

    missing_metrics = [
        metric
        for metric in MODEL_METRICS
        if (f"metrics.{metric}" not in run_data.index or pd.isna(run_data[f"metrics.{metric}"]))
    ]

    if missing_metrics:
        raise ValueError(f"Im Run fehlen folgende Metriken: {missing_metrics}")

    return {metric: float(run_data[f"metrics.{metric}"]) for metric in MODEL_METRICS}


def get_tuning_parameters(
    tuned_run: pd.Series,
) -> dict[str, str]:
    """Extract the selected tuning parameters."""

    parameter_names = [
        "n_estimators",
        "max_depth",
        "max_features",
        "min_samples_leaf",
        "min_samples_split",
        "class_weight",
    ]

    return {
        parameter: str(
            tuned_run.get(
                f"params.{parameter}",
                "nicht protokolliert",
            )
        )
        for parameter in parameter_names
    }


def save_model_selection(
    comparison: pd.DataFrame,
    baseline: pd.Series,
    tuned: pd.Series,
    baseline_metrics: dict[str, float],
    tuned_metrics: dict[str, float],
    tuned_parameters: dict[str, str],
    cv_pr_auc: float | None,
) -> tuple[str, str]:
    """Select the model and persist the decision in MLflow."""

    baseline_pr_auc = baseline_metrics["pr_auc"]
    tuned_pr_auc = tuned_metrics["pr_auc"]

    difference = tuned_pr_auc - baseline_pr_auc

    if difference > 0:
        selected_model = "random_forest_tuned"
        selected_run = tuned
        selected_metrics = tuned_metrics
    else:
        selected_model = "random_forest"
        selected_run = baseline
        selected_metrics = baseline_metrics

    selected_run_id = str(selected_run["run_id"])

    baseline_run_id = str(baseline["run_id"])

    tuned_run_id = str(tuned["run_id"])

    selection_summary = {
        "primary_metric": "pr_auc",
        "selected_model": selected_model,
        "selected_run_id": selected_run_id,
        "baseline_run_id": baseline_run_id,
        "tuned_run_id": tuned_run_id,
        "baseline_pr_auc": baseline_pr_auc,
        "tuned_pr_auc": tuned_pr_auc,
        "pr_auc_difference": difference,
        "cv_pr_auc": cv_pr_auc,
        "selected_metrics": selected_metrics,
        "tuned_parameters": tuned_parameters,
    }

    with mlflow.start_run(run_name="model_selection"):
        mlflow.set_tags(
            {
                "pipeline_stage": "model_selection",
                "selected_model": selected_model,
                "selection_metric": "pr_auc",
            }
        )

        mlflow.log_params(
            {
                "selected_model": selected_model,
                "selected_run_id": selected_run_id,
                "baseline_run_id": baseline_run_id,
                "tuned_run_id": tuned_run_id,
                "selection_metric": "pr_auc",
            }
        )

        mlflow.log_metric(
            "baseline_pr_auc",
            baseline_pr_auc,
        )

        mlflow.log_metric(
            "tuned_pr_auc",
            tuned_pr_auc,
        )

        mlflow.log_metric(
            "pr_auc_difference",
            difference,
        )

        if cv_pr_auc is not None:
            mlflow.log_metric(
                "tuned_cv_pr_auc",
                cv_pr_auc,
            )

        for metric, value in selected_metrics.items():
            mlflow.log_metric(
                f"selected_{metric}",
                value,
            )

        mlflow.log_text(
            comparison.to_csv(index=False),
            "model_selection/model_comparison.csv",
        )

        mlflow.log_dict(
            selection_summary,
            "model_selection/selection_summary.json",
        )

    return selected_model, selected_run_id


def run() -> None:
    """Compare runs and save the selection in MLflow."""

    configure_databricks_auth()

    config = load_config().values

    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])

    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    experiment = mlflow.get_experiment_by_name(config["mlflow"]["experiment_name"])

    if experiment is None:
        raise ValueError("Das konfigurierte MLflow-Experiment wurde nicht gefunden.")

    selected_runs = get_latest_successful_runs(experiment.experiment_id)

    comparison = build_comparison(selected_runs)

    print("\nModellvergleich:")
    print(comparison.to_string(index=False))

    baseline = selected_runs[selected_runs["run_name"] == "random_forest"].iloc[0]

    tuned = selected_runs[selected_runs["run_name"] == "random_forest_tuned"].iloc[0]

    baseline_metrics = get_metrics(baseline)

    tuned_metrics = get_metrics(tuned)

    baseline_pr_auc = baseline_metrics["pr_auc"]
    tuned_pr_auc = tuned_metrics["pr_auc"]
    difference = tuned_pr_auc - baseline_pr_auc

    print("\nPR-AUC-Vergleich:")
    print(f"Baseline:  {baseline_pr_auc:.4f}")
    print(f"Getunt:    {tuned_pr_auc:.4f}")
    print(f"Differenz: {difference:+.4f}")

    tuned_parameters = get_tuning_parameters(tuned)

    print("\nBeste Tuning-Parameter:")

    for parameter, value in tuned_parameters.items():
        print(f"{parameter}: {value}")

    cv_pr_auc_value = tuned.get(
        "metrics.cv_pr_auc",
        pd.NA,
    )

    if pd.notna(cv_pr_auc_value):
        cv_pr_auc = float(cv_pr_auc_value)
        print(f"\nCV-PR-AUC: {cv_pr_auc:.4f}")
    else:
        cv_pr_auc = None
        print("\nCV-PR-AUC wurde nicht protokolliert.")

    selected_model, selected_run_id = save_model_selection(
        comparison=comparison,
        baseline=baseline,
        tuned=tuned,
        baseline_metrics=baseline_metrics,
        tuned_metrics=tuned_metrics,
        tuned_parameters=tuned_parameters,
        cv_pr_auc=cv_pr_auc,
    )

    print("\nGespeicherte Modellentscheidung:")
    print(f"Ausgewähltes Modell: {selected_model}")
    print(f"Ausgewählter Run:    {selected_run_id}")

    if selected_model == "random_forest_tuned":
        print("Ergebnis: Das getunte Modell erzielt die höhere Test-PR-AUC.")
    else:
        print("Ergebnis: Die Baseline erzielt mindestens dieselbe Test-PR-AUC.")


if __name__ == "__main__":
    run()
