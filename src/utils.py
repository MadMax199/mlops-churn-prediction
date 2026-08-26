from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import NumericType, StringType


def get_missing_values(df: DataFrame) -> DataFrame:
    """Ermittelt NULL-, NaN- und leere Werte je Spalte."""

    total_rows = df.count()

    if total_rows == 0:
        raise ValueError("Der DataFrame enthält keine Zeilen.")

    expressions = []

    for field in df.schema.fields:
        column = field.name
        condition = F.col(column).isNull()

        if isinstance(field.dataType, NumericType):
            condition = condition | F.isnan(F.col(column))

        elif isinstance(field.dataType, StringType):
            normalized = F.lower(F.trim(F.col(column)))

            condition = condition | normalized.isin(
                "",
                "null",
                "none",
                "nan",
                "na",
                "n/a",
            )

        expressions.append(
            F.sum(condition.cast("int")).alias(column)
        )

    missing_counts = df.select(*expressions).first().asDict()

    result = [
        (
            column,
            int(count or 0),
            round((count or 0) / total_rows, 4),
        )
        for column, count in missing_counts.items()
    ]

    return (
        df.sparkSession
        .createDataFrame(
            result,
            ["column", "missing_count", "missing_share"],
        )
        .orderBy(
            F.desc("missing_count"),
            F.asc("column"),
        )
    )


)


def aggregate_orders(df:DataFrame) -> DataFrame:
    """Aggregiert den Orders DataFrame nach user_id und berechnet die Summe der amount und item_count Spalten"""
    return(df.groupBy("user_id").agg(
        F.count("*").alias("order_count"),
        F.sum("amount").alias("total_amount"),
        F.sum("item_count").alias("total_item"),
        F.max("creation_date").alias("last_transaction"))       
        )

def aggre