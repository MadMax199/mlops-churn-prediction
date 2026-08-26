from src.config import load_config
from src.data.clean import clean_events, clean_orders, clean_users
from src.data.ingest import load_raw_data
from src.data.write import ensure_schema, write_delta_table
from src.session import get_spark_session


def run() -> None:

    config = load_config().values
    spark = get_spark_session()
    ensure_schema(spark, config["databricks"]["catalog"], config["databricks"]["schema"])
    raw_users, raw_orders, raw_events = load_raw_data(spark, config)
    
    outputs = {
        config["data"]["silver_users_table"]: clean_users(raw_users),
        config["data"]["silver_orders_table"]: clean_orders(raw_orders),
        config["data"]["silver_events_table"]: clean_events(raw_events),
    }
    for table_name, frame in outputs.items():
        write_delta_table(frame, table_name)
        print(f"Written {frame.count():,} rows to {table_name}")


if __name__ == "__main__":
    run()

