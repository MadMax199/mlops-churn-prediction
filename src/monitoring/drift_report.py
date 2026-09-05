"""Create a feature drift report using PSI."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT, load_config
from src.features.schema import FEATURE_COLUMNS
from src.session import get_spark_session


def population_stability_index(
    reference: pd.Series,
    current: pd.Series,
    bins: int = 10,
) -> float:
    """Calculate PSI for one numeric feature."""

    reference_values = pd.to_numeric(
        reference,
        errors="coerce",
    ).dropna()

    current_values = pd.to_numeric(
        current,
        errors="coerce",
    ).dropna()

    if reference_values.empty or current_values.empty:
        return float("nan")

    quantile_count = min(
        bins,
        int(reference_values.nunique()),
    )

    if quantile_count < 2:
        return float("nan")

    edges = pd.qcut(
        reference_values,
        q=quantile_count,
        duplicates="drop",
        retbins=True,
    )[1]

    edges = np.unique(edges.astype(float))

    if len(edges) < 2:
        return float("nan")

    # Include current values outside the reference range.
    edges[0] = -np.inf
    edges[-1] = np.inf

    reference_share = pd.cut(
        reference_values,
        bins=edges,
        include_lowest=True,
    ).value_counts(
        normalize=True,
        sort=False,
    )

    current_share = (
        pd.cut(
            current_values,
            bins=edges,
            include_lowest=True,
        )
        .value_counts(
            normalize=True,
            sort=False,
        )
        .reindex(
            reference_share.index,
            fill_value=0.0,
        )
    )

    epsilon = 1e-6

    reference_share = reference_share.clip(lower=epsilon)
    current_share = current_share.clip(lower=epsilon)

    psi = ((current_share - reference_share) * np.log(current_share / reference_share)).sum()

    return float(psi)


def classify_psi(psi: float) -> str:
    """Convert a PSI value into a drift status."""

    if pd.isna(psi):
        return "not_evaluable"

    if psi < 0.10:
        return "stable"

    if psi < 0.25:
        return "warning"

    return "drift"


def run() -> None:
    """Create or update the feature drift report."""

    config = load_config().values

    reference_path = PROJECT_ROOT / "data" / "reference" / "gold_customer_features.parquet"

    current = (
        get_spark_session()
        .table(config["data"]["gold_features_table"])
        .select(*FEATURE_COLUMNS)
        .toPandas()
    )

    # Databricks Connect attaches PlanMetrics that cannot
    # be serialized by pandas when writing Parquet.
    current.attrs.clear()

    if not reference_path.exists():
        reference_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        current.to_parquet(
            reference_path,
            index=False,
        )

        print(f"Reference snapshot created: {reference_path}")
        return

    reference = pd.read_parquet(reference_path)

    rows: list[dict[str, object]] = []

    for column in FEATURE_COLUMNS:
        if not pd.api.types.is_numeric_dtype(reference[column]):
            continue

        psi = population_stability_index(
            reference[column],
            current[column],
        )

        rows.append(
            {
                "feature": column,
                "psi": psi,
                "status": classify_psi(psi),
            }
        )

    report = pd.DataFrame(rows).sort_values(
        by="psi",
        ascending=False,
        na_position="last",
    )

    output_path = PROJECT_ROOT / "reports" / "drift_report.csv"

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report.to_csv(
        output_path,
        index=False,
    )

    print(report.to_string(index=False))
    print(f"Report written: {output_path}")


if __name__ == "__main__":
    run()
