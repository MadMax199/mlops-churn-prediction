from src.config import load_config


def test_config_contains_c360_paths_and_own_schema():
    config = load_config().values
    assert config["data"]["users_path"].startswith("/Volumes/main/dbdemos_retail_c360/")
    assert config["databricks"]["schema"] == "mlops_churn"
    assert config["data"]["gold_features_table"] == "main.mlops_churn.gold_customer_features"
