from pyspark.sql import DataFrame, SparkSession


def ensure_schema(spark: SparkSession, catalog: str, schema: str) -> None:
    """Stellt sicher, dass das angegebene Schema im angegebenen Katalog existiert."""
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS `{catalog}`.`{schema}`")


def write_delta_table(df: DataFrame, table_name: str) -> None:
    """Schreibt den DataFrame als Delta-Tabelle in die angegebene Tabelle."""
    df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(table_name)

