"""Pydantic schemas for the prediction API."""

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Input features for one churn prediction."""

    canal: str | None = None
    country: str | None = None
    gender: float | None = None
    age_group: float | None = None
    platform: str | None = None

    order_count: float | None = Field(
        default=None,
        ge=0,
    )
    total_amount: float | None = Field(
        default=None,
        ge=0,
    )
    avg_order_amount: float | None = Field(
        default=None,
        ge=0,
    )
    total_items: float | None = Field(
        default=None,
        ge=0,
    )
    event_count: float | None = Field(
        default=None,
        ge=0,
    )
    session_count: float | None = Field(
        default=None,
        ge=0,
    )
    days_since_creation: float | None = Field(
        default=None,
        ge=0,
    )
    days_since_last_activity: float | None = Field(
        default=None,
        ge=0,
    )
    days_since_last_transaction: float | None = Field(
        default=None,
        ge=0,
    )
    days_since_last_event: float | None = Field(
        default=None,
        ge=0,
    )


class PredictionResponse(BaseModel):
    """Prediction returned by the API."""

    prediction: int
    churn_probability: float = Field(
        ge=0,
        le=1,
    )
    model_name: str
    model_version: str
    model_alias: str


class HealthResponse(BaseModel):
    """Health status returned by the API."""

    status: str
    model_loaded: bool


class ModelInfoResponse(BaseModel):
    """Information about the loaded model."""

    model_name: str
    model_version: str
    model_alias: str
    model_uri: str


class MonitoringResponse(BaseModel):
    """Aggregated operational prediction metrics."""

    total_requests: int = Field(ge=0)
    successful_predictions: int = Field(ge=0)
    failed_predictions: int = Field(ge=0)
    predicted_churn_count: int = Field(ge=0)

    predicted_churn_rate: float = Field(
        ge=0,
        le=1,
    )
    average_churn_probability: float = Field(
        ge=0,
        le=1,
    )
    average_latency_ms: float = Field(ge=0)

    model_name: str
    model_version: str
    started_at: str
    last_request_at: str | None
