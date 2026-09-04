import pandas as pd


def validate_training_data(df: pd.DataFrame, id_column: str, target_column: str) -> None:
    """Validiert den Trainingsdatensatz für das Modelltraining.

    Args:
        df: Der Trainingsdatensatz als pandas DataFrame.
        id_column: Der Name der ID-Spalte.
        target_column: Der Name der Zielvariable-Spalte.

    Raises:
        ValueError: Wenn die Validierung fehlschlägt.
    """
    required = {id_column, target_column}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if df.empty:
        raise ValueError("Training data is empty.")
    if df[id_column].isna().any() or df[id_column].duplicated().any():
        raise ValueError(f"{id_column} must be non-null and unique.")
    labels = set(df[target_column].dropna().unique())
    if not labels or not labels.issubset({0, 1}):
        raise ValueError(f"{target_column} must be binary, got {sorted(labels)}")
