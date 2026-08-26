# MLOps Churn Prediction

Lokal entwickelbarer MLOps-Prototyp auf Basis des Databricks-Demos **Lakehouse for C360: Reducing Customer Churn**. Databricks Connect liest die internen Volume-Daten und schreibt reproduzierbare Delta-Tabellen. Training, Tests, FastAPI und Streamlit werden lokal ausgeführt; Experimente liegen in Databricks MLflow.

## Datenfluss

```text
C360 Volumes (users, orders, events)
        -> Silver-Tabellen (bereinigt)
        -> Gold-Feature-Tabelle (eine Zeile je user_id)
        -> sklearn-Pipelines + MLflow
        -> FastAPI / Streamlit / Drift-Report
```

Die Pipeline schreibt ausschließlich nach `main.mlops_churn`. Die Quelldaten unter `main.dbdemos_retail_c360` bleiben unverändert. Namen, Anschriften und E-Mail-Adressen werden nicht in die Modelltabellen übernommen.

## Einrichtung unter Windows PowerShell

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e . --no-deps
Copy-Item .env.example .env
```

Falls `Get-Command databricks` auf die VS-Code-Erweiterung zeigt, trage deren vollständigen Pfad in `.env` als `DATABRICKS_CLI_PATH` ein. `.env` wird nicht versioniert.

Verbindung prüfen:

```powershell
python scripts/test_connection.py
```

## Pipeline ausführen

```powershell
# 1. Rohdaten lesen, bereinigen und als Silver speichern
python -m src.data.pipeline

# 2. Nutzer-, Order- und Eventmerkmale zur Gold-Tabelle verbinden
python -m src.features.build_features

# 3. Baseline, logistische Regression und Random Forest trainieren/loggen
python -m src.training.train
```

Mit installiertem GNU Make entsprechen `make silver`, `make features`, `make train` und `make all` demselben Ablauf.

## Tests, API und Dashboard

```powershell
pytest tests -v
uvicorn src.api.main:app --reload --port 8000
streamlit run src/app/streamlit_app.py
```

- API-Dokumentation: <http://127.0.0.1:8000/docs>
- Streamlit: <http://localhost:8501>
- API-Endpunkte: `/health`, `/predict`, `/model-info`, `/metrics`

API und Dashboard laden automatisch den besten MLflow-Run unter den Serving-Kandidaten nach PR-AUC. Die Dummy-Baseline wird geloggt, aber nicht ausgeliefert.

## Monitoring

```powershell
python -m src.monitoring.drift_report
```

Beim ersten Lauf wird ein lokaler, nicht versionierter Referenz-Snapshot angelegt. Weitere Läufe schreiben numerische PSI-Werte nach `reports/drift_report.csv`.

## Projektstruktur

```text
config/                 zentrale Konfiguration
src/data/               Ingestion, Bereinigung, Delta-Persistenz
src/features/           Aggregationen, Featureschema, Preprocessing
src/training/           Modellvergleich und MLflow Tracking
src/api/                FastAPI-Inferenz
src/app/                Streamlit-Dashboard
src/monitoring/         einfacher Drift-Report
src/utils/              Spark-Hilfsfunktionen, u. a. Missing Values
tests/                  schnelle lokale Unit-Tests
```

## Nächster Arbeitsschritt

Führe zuerst `python -m src.data.pipeline` aus. Falls das Demoschema einer Spalte von den erwarteten Namen abweicht, prüfe `raw_users.printSchema()`, `raw_orders.printSchema()` und `raw_events.printSchema()` im Notebook und passe ausschließlich `src/data/clean.py` an. Danach kontrollierst du die Silver-Tabellen mit `get_missing_values(..., only_missing=True)`.
