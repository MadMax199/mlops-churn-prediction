from pyspark.sql import DataFrame, functions as F

from src.config import load_config
from src.data.write import write_delta_table
from src.session import get_spark_session


def aggregate_orders(df: DataFrame) -> DataFrame:
    """Aggregiert den Orders DataFrame auf User Ebene."""

    return df.groupBy("user_id").agg(
        F.count("order_id").alias("order_count"),
        F.sum("amount").alias("total_amount"),
        F.avg("amount").alias("avg_order_amount"),
        F.sum("item_count").alias("total_items"),
        F.max("creation_date").alias("last_transaction"),
    )


def aggregate_events(df: DataFrame) -> DataFrame:
    """Aggregiert den Events DataFrame auf User Ebene."""

    return df.groupBy("user_id").agg(
        F.count("event_id").alias("event_count"),
        F.countDistinct("session_id").alias("session_count"),
        F.first("platform", ignorenulls=True).alias("platform"),
        F.max("event_time").alias("last_event"),
    )


def build_customer_features(users: DataFrame, orders: DataFrame, events: DataFrame) -> DataFrame:
    """Baut die Feature auf Baisis der gejointen Datennenframes für User, Orders und Events auf."""

    today = F.current_date()
    return (
        users.join(aggregate_orders(orders), "user_id", "left")
        .join(aggregate_events(events), "user_id", "left")
        .fillna({
            "order_count": 0,
            "total_amount": 0.0,
            "avg_order_amount": 0.0,
            "total_items": 0,
            "event_count": 0,
            "session_count": 0,
            "platform": "unknown",
        })
        .withColumn("days_since_creation", F.datediff(today, "creation_date"))
        .withColumn("days_since_last_activity", F.datediff(today, "last_activity_date"))
        .withColumn("days_since_last_transaction", F.datediff(today, "last_transaction"))
        .withColumn("days_since_last_event", F.datediff(today, "last_event"))
        .drop("creation_date", "last_activity_date", "last_transaction", "last_event")
    )


def run() -> None:
    config = load_config().values
    spark = get_spark_session()
    data = config["data"]
    features = build_customer_features(
        spark.table(data["silver_users_table"]),
        spark.table(data["silver_orders_table"]),
        spark.table(data["silver_events_table"]),
    )
    write_delta_table(features, data["gold_features_table"])
    print(f"Written {features.count():,} rows to {data['gold_features_table']}")


if __name__ == "__main__":
    run()

