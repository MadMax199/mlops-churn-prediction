from pathlib import Path

import pandas as pd

from src.config import PROJECT_ROOT, load_config
from src.features.schema import FEATURE_COLUMNS
from src.session import get_spark_session


def population_stability_index(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """Small dependency-free PSI implementation for numeric feature monitoring."""
    edges = pd.qcut(reference, q=bins, duplicates="drop", retbins=True)[1]
    reference_share = pd.cut(reference, edges, include_lowest=True).value_counts(
        normalize=True, sort=False
    )
    current_share = pd.cut(current, edges, include_lowest=True).value_counts(
        normalize=True, sort=False
    )
    reference_share = reference_share.clip(lower=1e-6)
    current_share = current_share.reindex(reference_share.index, fill_value=1e-6).clip(lower=1e-6)
    return float(
        (
            (current_share - reference_share)
            * (current_share / reference_share).apply(__import__("math").log)
        ).sum()
    )


def run() -> None:
    config = load_config().values
    reference_path = PROJECT_ROOT / "data" / "reference" / "gold_customer_features.parquet"
    current = get_spark_session().table(config["data"]["gold_features_table"]).toPandas()
    if not reference_path.exists():
        reference_path.parent.mkdir(parents=True, exist_ok=True)
        current.to_parquet(reference_path, index=False)
        print(f"Reference snapshot created: {reference_path}")
        return
    reference = pd.read_parquet(reference_path)
    rows = []
    for column in FEATURE_COLUMNS:
        if pd.api.types.is_numeric_dtype(reference[column]):
            rows.append(
                {
                    "feature": column,
                    "psi": population_stability_index(
                        reference[column].dropna(), current[column].dropna()
                    ),
                }
            )
    report = pd.DataFrame(rows).sort_values("psi", ascending=False)
    output = PROJECT_ROOT / "reports" / "drift_report.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output, index=False)
    print(report.to_string(index=False))
    print(f"Report written: {output}")


if __name__ == "__main__":
    run()
