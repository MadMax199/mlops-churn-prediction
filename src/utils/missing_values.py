from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import NumericType, StringType


def get_missing_values(df: DataFrame, only_missing: bool = False) -> DataFrame:
    """Counts NULL, NaN and blank/null-like strings for every Spark column."""
    total_rows = df.count()
    if total_rows == 0:
        raise ValueError("The DataFrame contains no rows.")

    expressions = []
    for field in df.schema.fields:
        condition = F.col(field.name).isNull()
        if isinstance(field.dataType, NumericType):
            condition = condition | F.isnan(F.col(field.name))
        elif isinstance(field.dataType, StringType):
            normalized = F.lower(F.trim(F.col(field.name)))
            condition = condition | normalized.isin("", "null", "none", "nan", "na", "n/a")
        expressions.append(F.sum(condition.cast("int")).alias(field.name))

    counts = df.select(*expressions).first().asDict()
    rows = [
        (column, int(count or 0), round((count or 0) / total_rows, 4))
        for column, count in counts.items()
        if not only_missing or count
    ]
    return df.sparkSession.createDataFrame(
        rows, ["column", "missing_count", "missing_share"]
    ).orderBy(F.desc("missing_count"), F.asc("column"))
