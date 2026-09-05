"""FastAPI application for churn predictions."""

import logging
from contextlib import asynccontextmanager
from time import perf_counter
from uuid import uuid4

from fastapi import (
    Depends,
    FastAPI,
    HTTPException,
    Request,
    status,
)

from src.api.model_loader import ModelService
from src.api.schemas import (
    HealthResponse,
    ModelInfoResponse,
    MonitoringResponse,
    PredictionRequest,
    PredictionResponse,
)
from src.monitoring.prediction_monitor import (
    PredictionMonitor,
)

logging.basicConfig(
    level=logging.INFO,
    format=("%(asctime)s | %(levelname)s | %(name)s | %(message)s"),
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model and initialize monitoring."""

    logger.info("Loading registered Champion model")

    model_service = ModelService()

    app.state.model_service = model_service
    app.state.prediction_monitor = PredictionMonitor(
        model_name=model_service.model_name,
        model_version=model_service.model_version,
    )

    logger.info(
        "Model version %s loaded",
        model_service.model_version,
    )

    yield

    logger.info("Stopping prediction API")


app = FastAPI(
    title="Churn Prediction API",
    description=("Prediction API for the registered MLflow Champion model."),
    version="1.1.0",
    lifespan=lifespan,
)


def get_model_service(
    request: Request,
) -> ModelService:
    """Return the initialized model service."""

    model_service = getattr(
        request.app.state,
        "model_service",
        None,
    )

    if model_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Das Modell ist nicht geladen.",
        )

    return model_service


def get_prediction_monitor(
    request: Request,
) -> PredictionMonitor:
    """Return the initialized prediction monitor."""

    prediction_monitor = getattr(
        request.app.state,
        "prediction_monitor",
        None,
    )

    if prediction_monitor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Das Monitoring ist nicht initialisiert.",
        )

    return prediction_monitor


@app.get("/")
def root() -> dict[str, str]:
    """Return basic API information."""

    return {
        "name": "Churn Prediction API",
        "documentation": "/docs",
    }


@app.get(
    "/health",
    response_model=HealthResponse,
)
def health(
    model_service: ModelService = Depends(  # noqa: B008
        get_model_service
    ),
) -> HealthResponse:
    """Return the API health status."""

    return HealthResponse(
        status="ok",
        model_loaded=model_service.model is not None,
    )


@app.get(
    "/model-info",
    response_model=ModelInfoResponse,
)
def model_info(
    model_service: ModelService = Depends(  # noqa: B008
        get_model_service
    ),
) -> ModelInfoResponse:
    """Return information about the loaded model."""

    return ModelInfoResponse(**model_service.get_model_info())


@app.get(
    "/monitoring",
    response_model=MonitoringResponse,
)
def monitoring(
    prediction_monitor: PredictionMonitor = Depends(  # noqa: B008
        get_prediction_monitor
    ),
) -> MonitoringResponse:
    """Return aggregated operational metrics."""

    return MonitoringResponse(**prediction_monitor.snapshot())


@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(
    payload: PredictionRequest,
    model_service: ModelService = Depends(  # noqa: B008
        get_model_service
    ),
    prediction_monitor: PredictionMonitor = Depends(  # noqa: B008
        get_prediction_monitor
    ),
) -> PredictionResponse:
    """Create a churn prediction."""

    request_id = uuid4().hex
    started_at = perf_counter()

    try:
        prediction = model_service.predict(payload)
        response = PredictionResponse(**prediction)

        latency_ms = (perf_counter() - started_at) * 1000

        prediction_monitor.record_success(
            request_id=request_id,
            prediction=response.prediction,
            churn_probability=response.churn_probability,
            latency_ms=latency_ms,
        )

        return response

    except ValueError as error:
        latency_ms = (perf_counter() - started_at) * 1000

        prediction_monitor.record_failure(
            request_id=request_id,
            latency_ms=latency_ms,
            error=error,
        )

        logger.warning(
            "Invalid prediction input request_id=%s",
            request_id,
        )

        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Die Eingabedaten sind ungültig.",
        ) from error

    except Exception as error:
        latency_ms = (perf_counter() - started_at) * 1000

        prediction_monitor.record_failure(
            request_id=request_id,
            latency_ms=latency_ms,
            error=error,
        )

        logger.exception(
            "Prediction failed request_id=%s",
            request_id,
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Die Vorhersage ist fehlgeschlagen.",
        ) from error
