from __future__ import annotations

from src.config import load_config
from src.session import get_spark_session


def main() -> None:
    config = load_config().values
    spark = get_spark_session()
    identity = spark.sql("SELECT current_user() AS user, current_catalog() AS catalog").first()
    print(f"Connected as: {identity['user']}")
    print(f"Current catalog: {identity['catalog']}")
    for name in ("users_path", "orders_path", "events_path"):
        path = config["data"][name]
        print(f"Configured {name}: {path}")


if __name__ == "__main__":
    main()
