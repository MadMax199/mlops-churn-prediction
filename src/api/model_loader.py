"""Load the registered model and create predictions."""

import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowClient

from src.api.schemas import PredictionRequest
from src.config import load_config
from src.features.schema import FEATURE_COLUMNS
from src.utils.databricks_auth import (
    configure_databricks_auth,
)

MODEL_ALIAS = "Champion"

STRING_COLUMNS = [
    "canal",
    "country",
    "platform",
]


class ModelService:
    """Serve predictions from the registered Champion model."""

    def __init__(self) -> None:
        configure_databricks_auth()

        config = load_config().values
        mlflow_config = config["mlflow"]

        mlflow.set_tracking_uri(mlflow_config["tracking_uri"])

        mlflow.set_registry_uri("databricks-uc")

        self.model_name = mlflow_config["registered_model_name"]
        self.model_alias = MODEL_ALIAS

        self.model_uri = f"models:/{self.model_name}@{self.model_alias}"

        client = MlflowClient()

        model_version = client.get_model_version_by_alias(
            name=self.model_name,
            alias=self.model_alias,
        )

        self.model_version = str(model_version.version)

        self.model = mlflow.sklearn.load_model(self.model_uri)

    @staticmethod
    def prepare_input(
        payload: PredictionRequest,
    ) -> pd.DataFrame:
        """Convert the request into model input."""

        values = payload.model_dump()

        frame = pd.DataFrame(
            [values],
            columns=FEATURE_COLUMNS,
        )

        frame = frame.replace({None: np.nan}).infer_objects(copy=False)

        numeric_columns = [column for column in FEATURE_COLUMNS if column not in STRING_COLUMNS]

        frame[numeric_columns] = frame[numeric_columns].astype("float64")

        return frame

    def predict(
        self,
        payload: PredictionRequest,
    ) -> dict:
        """Create a churn prediction."""

        model_input = self.prepare_input(payload)

        prediction = int(self.model.predict(model_input)[0])

        churn_probability = float(self.model.predict_proba(model_input)[0, 1])

        return {
            "prediction": prediction,
            "churn_probability": churn_probability,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "model_alias": self.model_alias,
        }

    def get_model_info(self) -> dict:
        """Return information about the loaded model."""

        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "model_alias": self.model_alias,
            "model_uri": self.model_uri,
        }
