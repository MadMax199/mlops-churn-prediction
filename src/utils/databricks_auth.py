"""Configure Databricks authentication for the project."""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

logger = logging.getLogger(__name__)


def configure_databricks_auth() -> None:
    """Configure local or environment-based Databricks authentication."""

    project_root = Path(__file__).resolve().parents[2]
    env_path = project_root / ".env"

    if env_path.is_file():
        load_dotenv(env_path, override=False)

    # Container, CI/CD oder Cloud-Umgebung
    if os.getenv("DATABRICKS_HOST"):
        logger.info("Using environment-based Databricks authentication")
        return

    # Lokale Entwicklung über das Databricks-CLI-Profil
    profile = os.getenv("DATABRICKS_CONFIG_PROFILE")
    cli_path_value = os.getenv("DATABRICKS_CLI_PATH")

    if not profile:
        raise ValueError(
            "Keine Databricks-Authentifizierung konfiguriert. "
            "Setze DATABRICKS_HOST und passende Zugangsdaten oder "
            "DATABRICKS_CONFIG_PROFILE für die lokale CLI-Anmeldung."
        )

    if cli_path_value:
        cli_path = Path(cli_path_value)

        if not cli_path.is_file():
            raise FileNotFoundError(f"Databricks CLI wurde nicht gefunden: {cli_path}")

        cli_directory = str(cli_path.parent)
        current_path = os.environ.get("PATH", "")

        if cli_directory not in current_path.split(os.pathsep):
            os.environ["PATH"] = cli_directory + os.pathsep + current_path

        logger.info(
            "Using Databricks CLI profile '%s' with CLI '%s'",
            profile,
            cli_path,
        )
    else:
        logger.info(
            "Using Databricks configuration profile '%s'",
            profile,
        )
