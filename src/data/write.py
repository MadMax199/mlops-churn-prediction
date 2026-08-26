from pyspark.sql import DataFrame, SparkSession


def ensure_schema(spark: SparkSession, catalog: str, schema: str) -> None:
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS `{catalog}`.`{schema}`")


def write_delta_table(df: DataFrame, table_name: str) -> None:
    df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(table_name)

