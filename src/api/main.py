"""FastAPI application for churn predictions."""

import logging
from contextlib import asynccontextmanager

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
    PredictionRequest,
    PredictionResponse,
)


logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s | %(levelname)s | "
        "%(name)s | %(message)s"
    ),
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model when the API starts."""

    logger.info("Loading registered Champion model")

    app.state.model_service = ModelService()

    logger.info(
        "Model version %s loaded",
        app.state.model_service.model_version,
    )

    yield

    logger.info("Stopping prediction API")


app = FastAPI(
    title="Churn Prediction API",
    description=(
        "Prediction API for the registered "
        "MLflow Champion model."
    ),
    version="1.0.0",
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
    model_service: ModelService = Depends(
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
    model_service: ModelService = Depends(
        get_model_service
    ),
) -> ModelInfoResponse:
    """Return information about the loaded model."""

    return ModelInfoResponse(
        **model_service.get_model_info()
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(
    payload: PredictionRequest,
    model_service: ModelService = Depends(
        get_model_service
    ),
) -> PredictionResponse:
    """Create a churn prediction."""

    try:
        prediction = model_service.predict(
            payload
        )

        logger.info(
            "Prediction completed with model version %s",
            model_service.model_version,
        )

        return PredictionResponse(
            **prediction
        )

    except ValueError as error:
        logger.warning(
            "Invalid prediction input: %s",
            error,
        )

        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Die Eingabedaten sind ungültig.",
        ) from error

    except Exception as error:
        logger.exception(
            "Prediction failed"
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Die Vorhersage ist fehlgeschlagen.",
        ) from error