import os
from pathlib import Path

from dotenv import load_dotenv


def prepare_environment() -> None:
    load_dotenv()
    cli_path = os.getenv("DATABRICKS_CLI_PATH")
    if cli_path:
        cli_dir = str(Path(cli_path).expanduser().resolve().parent)
        os.environ["PATH"] = cli_dir + os.pathsep + os.environ.get("PATH", "")


def get_spark_session():
    """Arellt die Databricks Verbindung her."""
    prepare_environment()
    from databricks.connect import DatabricksSession

    return DatabricksSession.builder.validateSession(True).getOrCreate()
