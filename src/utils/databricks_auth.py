"""Configure Databricks authentication for the project."""

import os
from pathlib import Path

from dotenv import load_dotenv


def configure_databricks_auth() -> None:
    """Load Databricks authentication settings from .env."""

    project_root = Path(__file__).resolve().parents[2]
    env_path = project_root / ".env"

    if not env_path.is_file():
        raise FileNotFoundError(f".env-Datei wurde nicht gefunden: {env_path}")

    load_dotenv(
        env_path,
        override=True,
    )

    profile = os.getenv("DATABRICKS_CONFIG_PROFILE")

    cli_path_value = os.getenv("DATABRICKS_CLI_PATH")

    if not profile:
        raise ValueError("DATABRICKS_CONFIG_PROFILE fehlt in der .env-Datei.")

    if not cli_path_value:
        raise ValueError("DATABRICKS_CLI_PATH fehlt in der .env-Datei.")

    cli_path = Path(cli_path_value)

    if not cli_path.is_file():
        raise FileNotFoundError(f"Databricks CLI wurde nicht gefunden: {cli_path}")

    cli_directory = str(cli_path.parent)

    os.environ["PATH"] = cli_directory + os.pathsep + os.environ.get("PATH", "")

    print(f"Databricks-Profil: {profile}")
    print(f"Databricks CLI:    {cli_path}")
