from datetime import datetime

import mlflow
import mlflow.sklearn
import pandas as pd
import streamlit as st

from src.config import load_config
from src.features.schema import FEATURE_COLUMNS
from src.session import get_spark_session, prepare_environment

prepare_environment()
config = load_config().values
mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
st.set_page_config(page_title="Customer Churn Monitor", page_icon="📉", layout="wide")
st.title("📉 Customer Churn Monitor")
st.caption("Risk prioritization based on the Databricks Retail C360 demo data")


@st.cache_resource
def load_model():
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(config["mlflow"]["experiment_name"])
    if experiment is None:
        raise RuntimeError("MLflow experiment not found; run `make train` first.")
    runs = client.search_runs(
        [experiment.experiment_id],
        filter_string="params.artifact_type = 'sklearn_pipeline' AND params.serving_candidate = 'true'",
        order_by=[f"metrics.{config['mlflow']['selection_metric']} DESC"],
        max_results=1,
    )
    if not runs:
        raise RuntimeError("No trained serving candidate found.")
    return mlflow.sklearn.load_model(f"runs:/{runs[0].info.run_id}/model"), runs[0]


@st.cache_data(ttl=900)
def load_customers() -> pd.DataFrame:
    return get_spark_session().table(config["data"]["gold_features_table"]).toPandas()


try:
    model, best_run = load_model()
    customers = load_customers()
except Exception as exc:
    st.error(str(exc))
    st.stop()

customers["churn_probability"] = model.predict_proba(customers[FEATURE_COLUMNS])[:, 1]
threshold = config["prediction"]["threshold"]
customers["predicted_churn"] = (customers["churn_probability"] >= threshold).astype(int)

with st.sidebar:
    st.header("Model")
    st.metric("ROC-AUC", f"{best_run.data.metrics.get('roc_auc', 0):.3f}")
    st.metric("PR-AUC", f"{best_run.data.metrics.get('pr_auc', 0):.3f}")
    st.caption(f"Run: {best_run.info.run_id[:8]}…")

left, middle, right = st.columns(3)
left.metric("Customers", f"{len(customers):,}")
middle.metric("Predicted churn", f"{customers['predicted_churn'].sum():,}")
right.metric("Average risk", f"{customers['churn_probability'].mean():.1%}")

st.subheader("Highest-risk customers")
display_columns = ["user_id", "country", "canal", "order_count", "event_count", "churn_probability"]
st.dataframe(
    customers.sort_values("churn_probability", ascending=False)[display_columns].head(50),
    column_config={"churn_probability": st.column_config.ProgressColumn("Churn risk", min_value=0.0, max_value=1.0)},
    use_container_width=True,
    hide_index=True,
)
st.subheader("Risk by channel")
st.bar_chart(customers.groupby("canal")["churn_probability"].mean().sort_values(ascending=False))
st.caption(f"Updated: {datetime.now():%Y-%m-%d %H:%M:%S}")
