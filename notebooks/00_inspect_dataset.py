# Databricks notebook source
# MAGIC %md
# MAGIC # Inspect the C360 churn feature table
# MAGIC Run this notebook once before training. It verifies the table name and
# MAGIC provides the information needed to finalize leakage exclusions.

# COMMAND ----------

TABLE_NAME = "dbdemos_retail_c360.gold_churn_features"

df = spark.table(TABLE_NAME)
df.printSchema()
display(df.limit(10))

# COMMAND ----------

from pyspark.sql import functions as F

summary = df.select(
    F.count("*").alias("row_count"),
    F.countDistinct("customer_id").alias("distinct_customers"),
    F.sum(F.col("customer_id").isNull().cast("int")).alias("missing_customer_id"),
    F.sum(F.col("churn").isNull().cast("int")).alias("missing_churn"),
)
display(summary)

# COMMAND ----------

display(df.groupBy("churn").count().orderBy("churn"))

# COMMAND ----------

# Review the feature names for variables that are only known after churn.
display(spark.sql(f"DESCRIBE TABLE {TABLE_NAME}"))
