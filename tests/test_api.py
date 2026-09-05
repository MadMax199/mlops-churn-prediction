"""Tests for the churn prediction API."""

import pytest
from fastapi.testclient import TestClient

import src.api.main as api_main

VALID_PAYLOAD = {
    "canal": "web",
    "country": "US",
    "gender": 1,
    "age_group": 3,
    "platform": "ios",
    "order_count": 12,
    "total_amount": 850.5,
    "avg_order_amount": 70.88,
    "total_items": 24,
    "event_count": 130,
    "session_count": 18,
    "days_since_creation": 540,
    "days_since_last_activity": 12,
    "days_since_last_transaction": 30,
    "days_since_last_event": 5,
}


class FakeModelService:
    """Controlled replacement for the real MLflow model."""

    def __init__(self) -> None:
        self.model = object()
        self.model_name = "main.mlops_churn.churn_prediction_model"
        self.model_version = "1"
        self.model_alias = "Champion"
        self.model_uri = "models:/main.mlops_churn.churn_prediction_model@Champion"

    def get_model_info(self) -> dict:
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "model_alias": self.model_alias,
            "model_uri": self.model_uri,
        }

    def predict(self, payload) -> dict:
        return {
            "prediction": 1,
            "churn_probability": 0.8274,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "model_alias": self.model_alias,
        }


@pytest.fixture()
def client(
    monkeypatch: pytest.MonkeyPatch,
):
    """Create a test client without loading Databricks."""

    monkeypatch.setattr(
        api_main,
        "ModelService",
        FakeModelService,
    )

    with TestClient(api_main.app) as test_client:
        yield test_client


def test_root(client: TestClient) -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "name": "Churn Prediction API",
        "documentation": "/docs",
    }


def test_health(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model_loaded": True,
    }


def test_model_info(client: TestClient) -> None:
    response = client.get("/model-info")

    assert response.status_code == 200

    result = response.json()

    assert result["model_name"] == ("main.mlops_churn.churn_prediction_model")
    assert result["model_version"] == "1"
    assert result["model_alias"] == "Champion"
    assert result["model_uri"].endswith("@Champion")


def test_monitoring_starts_empty(
    client: TestClient,
) -> None:
    response = client.get("/monitoring")

    assert response.status_code == 200

    result = response.json()

    assert result["total_requests"] == 0
    assert result["successful_predictions"] == 0
    assert result["failed_predictions"] == 0
    assert result["predicted_churn_count"] == 0
    assert result["predicted_churn_rate"] == 0
    assert result["average_churn_probability"] == 0
    assert result["average_latency_ms"] == 0
    assert result["model_version"] == "1"
    assert result["last_request_at"] is None


def test_predict(client: TestClient) -> None:
    response = client.post(
        "/predict",
        json=VALID_PAYLOAD,
    )

    assert response.status_code == 200

    result = response.json()

    assert result["prediction"] in {0, 1}
    assert 0 <= result["churn_probability"] <= 1
    assert result["model_version"] == "1"
    assert result["model_alias"] == "Champion"


def test_successful_prediction_updates_monitoring(
    client: TestClient,
) -> None:
    prediction_response = client.post(
        "/predict",
        json=VALID_PAYLOAD,
    )

    assert prediction_response.status_code == 200

    monitoring_response = client.get("/monitoring")

    assert monitoring_response.status_code == 200

    result = monitoring_response.json()

    assert result["total_requests"] == 1
    assert result["successful_predictions"] == 1
    assert result["failed_predictions"] == 0
    assert result["predicted_churn_count"] == 1
    assert result["predicted_churn_rate"] == 1
    assert result["average_churn_probability"] == pytest.approx(0.8274)
    assert result["average_latency_ms"] >= 0
    assert result["last_request_at"] is not None


def test_predict_rejects_negative_values(
    client: TestClient,
) -> None:
    invalid_payload = VALID_PAYLOAD.copy()
    invalid_payload["order_count"] = -1

    response = client.post(
        "/predict",
        json=invalid_payload,
    )

    assert response.status_code == 422


def test_predict_handles_model_failure(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def failing_prediction(_payload):
        raise RuntimeError("Internal model failure")

    monkeypatch.setattr(
        client.app.state.model_service,
        "predict",
        failing_prediction,
    )

    response = client.post(
        "/predict",
        json=VALID_PAYLOAD,
    )

    assert response.status_code == 500
    assert response.json() == {"detail": "Die Vorhersage ist fehlgeschlagen."}

    monitoring_response = client.get("/monitoring")
    monitoring_result = monitoring_response.json()

    assert monitoring_result["total_requests"] == 1
    assert monitoring_result["successful_predictions"] == 0
    assert monitoring_result["failed_predictions"] == 1
