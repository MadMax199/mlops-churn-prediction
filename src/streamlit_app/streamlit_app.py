"""Streamlit frontend for the churn prediction API."""

from __future__ import annotations

import os
from typing import Any

import requests
import streamlit as st

API_BASE_URL = os.getenv(
    "CHURN_API_URL",
    "http://127.0.0.1:8000",
).rstrip("/")

REQUEST_TIMEOUT_SECONDS = 30


def call_api(
    method: str,
    endpoint: str,
    **kwargs: Any,
) -> dict[str, Any]:
    """Send a request to the FastAPI service."""

    url = f"{API_BASE_URL}{endpoint}"

    try:
        response = requests.request(
            method=method,
            url=url,
            timeout=REQUEST_TIMEOUT_SECONDS,
            **kwargs,
        )
    except requests.RequestException as exc:
        raise RuntimeError(f"Die Churn API ist unter {API_BASE_URL} nicht erreichbar.") from exc

    if not response.ok:
        try:
            response_body = response.json()
            detail = response_body.get(
                "detail",
                response_body,
            )
        except ValueError:
            detail = response.text

        raise RuntimeError(f"API-Fehler {response.status_code}: {detail}")

    return response.json()


@st.cache_data(ttl=30)
def load_health() -> dict[str, Any]:
    """Load the current API health status."""

    return call_api(
        method="GET",
        endpoint="/health",
    )


@st.cache_data(ttl=300)
def load_model_info() -> dict[str, Any]:
    """Load information about the deployed model."""

    return call_api(
        method="GET",
        endpoint="/model-info",
    )


def predict_churn(
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Request one churn prediction."""

    return call_api(
        method="POST",
        endpoint="/predict",
        json=payload,
    )


st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📉",
    layout="wide",
)

st.title("📉 Customer Churn Prediction")

st.caption(
    "Vorhersage des Abwanderungsrisikos mit dem registrierten Champion-Modell aus Databricks."
)


try:
    health = load_health()
    model_info = load_model_info()
except RuntimeError as exc:
    st.error(str(exc))

    st.info(
        f"Prüfe, ob der Docker-Container läuft und die FastAPI unter {API_BASE_URL} erreichbar ist."
    )

    st.stop()


with st.sidebar:
    st.header("API-Status")

    if health.get("model_loaded"):
        st.success(f"API erreichbar: {health.get('status', 'healthy')}")
        st.caption("Champion-Modell erfolgreich geladen")
    else:
        st.error("API erreichbar, aber das Modell ist nicht geladen")
        st.stop()

    st.header("Modell")

    st.write(
        "**Name:**",
        model_info.get(
            "model_name",
            "Unbekannt",
        ),
    )

    st.write(
        "**Version:**",
        model_info.get(
            "model_version",
            "Unbekannt",
        ),
    )

    st.write(
        "**Alias:**",
        model_info.get(
            "model_alias",
            "Unbekannt",
        ),
    )

    with st.expander("Technische Informationen"):
        st.write(
            "**Model URI:**",
            model_info.get(
                "model_uri",
                "Unbekannt",
            ),
        )
        st.write(
            "**API URL:**",
            API_BASE_URL,
        )


st.subheader("Kundendaten")

st.write(
    "Gib die aggregierten Kundenmerkmale ein. Anschließend sendet "
    "Streamlit die Daten an die lokale FastAPI."
)


with st.form("prediction_form"):
    customer, orders, activity = st.columns(3)

    with customer:
        st.markdown("#### Kundenmerkmale")

        canal = st.text_input(
            "Kanal",
            value="web",
            help="Beispielsweise web, email oder mobile.",
        )

        country = st.text_input(
            "Land",
            value="DE",
            help="Ländercode oder Kategorie aus den Trainingsdaten.",
        )

        platform = st.text_input(
            "Plattform",
            value="ios",
            help="Beispielsweise ios, android oder other.",
        )

        gender = st.number_input(
            "Gender-Code",
            min_value=0.0,
            value=0.0,
            step=1.0,
        )

        age_group = st.number_input(
            "Altersgruppen-Code",
            min_value=0.0,
            value=1.0,
            step=1.0,
        )

    with orders:
        st.markdown("#### Bestellungen")

        order_count = st.number_input(
            "Anzahl Bestellungen",
            min_value=0.0,
            value=2.0,
            step=1.0,
        )

        total_amount = st.number_input(
            "Gesamtumsatz",
            min_value=0.0,
            value=120.0,
            step=10.0,
        )

        avg_order_amount = st.number_input(
            "Durchschnittlicher Bestellwert",
            min_value=0.0,
            value=60.0,
            step=5.0,
        )

        total_items = st.number_input(
            "Anzahl gekaufter Artikel",
            min_value=0.0,
            value=3.0,
            step=1.0,
        )

        event_count = st.number_input(
            "Anzahl Events",
            min_value=0.0,
            value=10.0,
            step=1.0,
        )

        session_count = st.number_input(
            "Anzahl Sessions",
            min_value=0.0,
            value=4.0,
            step=1.0,
        )

    with activity:
        st.markdown("#### Kundenaktivität")

        days_since_creation = st.number_input(
            "Tage seit Kontoerstellung",
            min_value=0.0,
            value=365.0,
            step=1.0,
        )

        days_since_last_activity = st.number_input(
            "Tage seit letzter Aktivität",
            min_value=0.0,
            value=30.0,
            step=1.0,
        )

        days_since_last_transaction = st.number_input(
            "Tage seit letzter Transaktion",
            min_value=0.0,
            value=45.0,
            step=1.0,
        )

        days_since_last_event = st.number_input(
            "Tage seit letztem Event",
            min_value=0.0,
            value=15.0,
            step=1.0,
        )

    submitted = st.form_submit_button(
        "Churn-Risiko berechnen",
        type="primary",
        use_container_width=True,
    )


if submitted:
    payload = {
        "canal": canal.strip() or None,
        "country": country.strip() or None,
        "gender": float(gender),
        "age_group": float(age_group),
        "platform": platform.strip() or None,
        "order_count": float(order_count),
        "total_amount": float(total_amount),
        "avg_order_amount": float(avg_order_amount),
        "total_items": float(total_items),
        "event_count": float(event_count),
        "session_count": float(session_count),
        "days_since_creation": float(days_since_creation),
        "days_since_last_activity": float(days_since_last_activity),
        "days_since_last_transaction": float(days_since_last_transaction),
        "days_since_last_event": float(days_since_last_event),
    }

    try:
        with st.spinner("Churn-Risiko wird berechnet ..."):
            result = predict_churn(payload)

    except RuntimeError as exc:
        st.error(str(exc))

    else:
        probability = float(result["churn_probability"])

        prediction = int(result["prediction"])

        st.divider()
        st.subheader("Vorhersage")

        result_left, result_middle, result_right = st.columns(3)

        result_left.metric(
            "Churn-Wahrscheinlichkeit",
            f"{probability:.1%}",
        )

        result_middle.metric(
            "Vorhersage",
            "Churn" if prediction == 1 else "Kein Churn",
        )

        result_right.metric(
            "Modellversion",
            result.get(
                "model_version",
                "Unbekannt",
            ),
        )

        st.progress(
            min(
                max(probability, 0.0),
                1.0,
            )
        )

        if prediction == 1:
            st.warning("Für diesen Kunden wurde ein erhöhtes Abwanderungsrisiko vorhergesagt.")
        else:
            st.success("Für diesen Kunden wurde aktuell kein Churn vorhergesagt.")

        with st.expander("Modellinformationen"):
            st.write(
                "**Modell:**",
                result.get(
                    "model_name",
                    "Unbekannt",
                ),
            )

            st.write(
                "**Version:**",
                result.get(
                    "model_version",
                    "Unbekannt",
                ),
            )

            st.write(
                "**Alias:**",
                result.get(
                    "model_alias",
                    "Unbekannt",
                ),
            )

        with st.expander("Gesendete API-Daten"):
            st.json(payload)

        with st.expander("Vollständige API-Antwort"):
            st.json(result)
