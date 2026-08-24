# MLOps Churn Prediction

Initial, reproducible data pipeline for an internal Databricks churn dataset.

## Current scope

- Configuration-based Databricks table loading
- Dataset validation for binary churn classification
- Reproducible stratified train/test split
- Separate numeric and categorical preprocessing
- Initial unit tests

## Configure the data source

Set the actual three-level Databricks table name and schema-specific columns in
`configs/pipeline.yaml`. Customer identifiers belong in `id_columns`; columns
that would leak future information belong in `exclude_columns`.

## Local setup

```bash
python -m pip install -e ".[dev]"
pytest
```

The next implementation step is a training pipeline with a logistic-regression
baseline and MLflow logging.
