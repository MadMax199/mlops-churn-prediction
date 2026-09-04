from pyspark.sql import SparkSession


def load_raw_data(spark: SparkSession, config: dict):
    """Funktion zum Laden der Rohdaten."""

    data = config["data"]
    users = spark.read.json(data["users_path"])
    orders = spark.read.json(data["orders_path"])
    events = spark.read.option("header", True).option("inferSchema", True).csv(data["events_path"])

    return users, orders, events
