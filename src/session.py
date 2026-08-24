from __future__ import annotations

from databricks.connect import DatabricksSession
from dotenv import load_dotenv


def get_spark_session() -> DatabricksSession:
    """Create a Spark session backed by remote Databricks serverless compute."""
    load_dotenv()
    return DatabricksSession.builder.validateSession(True).getOrCreate()

