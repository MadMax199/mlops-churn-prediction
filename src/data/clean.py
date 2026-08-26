from pyspark.sql import DataFrame, functions as F



def clean_user(df:DataFrame) -> DataFrame:
    
    """Bereinigt den User DataFram nacheem es aus der csv gelesen wurde für weitere Verarbeitungsschritte"""
    return(df.select(
            F.col("id").alias("user_id"),
            F.sha1("email").alias("email"),
            F.to_timestamp(
                "creation_date",
                "MM-dd-yyyy HH:mm:ss",
            ).alias("creation_date"),
            F.to_timestamp(
                "last_activity_date",
                "MM-dd-yyyy HH:mm:ss",
            ).alias("last_activity_date"),
            F.initcap("firstname").alias("firstname"),
            F.initcap("lastname").alias("lastname"),
            "address",
            "canal",
            "country",
            F.col("gender").cast("int").alias("gender"),
            F.col("age_group").cast("int").alias("age_group"),
            F.col("churn").cast("int").alias("churn"))
    )


def clean_orders(df:DataFrame) -> DataFrame:
    """Bereinigt den Orders DataFrame nacheem es aus der csv gelesen wurde für weitere Verarbeitungsschritte"""
    return(df.select(
                    F.col("amount").cast("int").alias("amount"),
                    F.col("id").alias("order_id"),
                    "user_id",
                    F.col("item_count").cast("int").alias("item_count"),
                    F.to_timestamp(
                        "transaction_date",
                        "MM-dd-yyyy HH:mm:ss",
                    ).alias("creation_date"),
                )
                )
    


def clean_events(df: DataFrame) -> DataFrame:
    """Bereinigt den Event DataFrame nacheem es aus der csv gelesen wurde für weitere Verarbeitungsschritte"""
    event_id = "event_id" if "event_id" in df.columns else "id"
    event_time = "event_time" if "event_time" in df.columns else "timestamp"
    return (
        df.select(
            F.col(event_id).alias("event_id"),
            "user_id",
            "session_id",
            F.lower(F.trim("platform")).alias("platform"),
            F.lower(F.trim("action")).alias("action"),
            F.to_timestamp(event_time).alias("event_time"),
        )
        .filter(F.col("event_id").isNotNull() & F.col("user_id").isNotNull())
        .fillna({"platform": "unknown", "action": "unknown"})
        .dropDuplicates(["event_id"])
    )
