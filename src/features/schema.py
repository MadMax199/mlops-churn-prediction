ID_COLUMN = "user_id"
TARGET_COLUMN = "churn"

DIRECT_IDENTIFIER_COLUMNS = [
    "email",
    "firstname",
    "lastname",
    "address",
]

FEATURE_COLUMNS = [
    "age_group",
    "gender",
    "canal",
    "country",
    "order_count",
    "total_amount",
    "avg_order_amount",
    "total_items",
    "event_count",
    "session_count",
    "platform",
    "days_since_creation",
    "days_since_last_activity",
    "days_since_last_transaction",
    "days_since_last_event",
]

GOLD_OUTPUT_COLUMNS = [
    ID_COLUMN,
    *FEATURE_COLUMNS,
    TARGET_COLUMN,
]