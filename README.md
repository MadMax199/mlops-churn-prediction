# MLOps Churn Prediction

[![CI](https://github.com/MadMax199/mlops-churn-prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/MadMax199/mlops-churn-prediction/actions/workflows/ci.yml)

Lokal entwickelbarer End-to-End-MLOps-Prototyp zur Vorhersage von Kundenabwanderung auf Basis des Databricks-Demos **Lakehouse for Retail C360: Reducing Customer Churn**.

Databricks Connect verbindet die lokale Entwicklungsumgebung mit Databricks. Die Daten werden reproduzierbar aufbereitet, als Delta-Tabellen gespeichert und für das Training mehrerer Klassifikationsmodelle verwendet. Experimente und Modelle werden mit MLflow verwaltet. Das ausgewählte Champion-Modell wird über FastAPI und Docker bereitgestellt und kann über eine Streamlit-Oberfläche verwendet werden.

## Funktionsumfang

Der Prototyp umfasst:

- reproduzierbare Datenvorverarbeitung mit PySpark und Databricks,
- eine Silver- und Gold-Datenpipeline,
- Feature Engineering und Trainingsdatenvalidierung,
- Training und Evaluation mehrerer Klassifikationsmodelle,
- Hyperparameter-Tuning eines Random Forest,
- Experiment Tracking mit MLflow,
- Modellregistrierung im Unity Catalog,
- lokale Modellbereitstellung über FastAPI und Docker,
- eine Streamlit-Oberfläche für Einzelvorhersagen,
- grundlegende Tests mit Pytest,
- Codequalitätsprüfungen mit Ruff,
- eine CI-Pipeline mit GitHub Actions,
- operatives Prediction-Monitoring,
- Feature-Drift-Monitoring mit dem Population Stability Index.

## Datenfluss

```text
Databricks C360 Volumes
        |
        v
Silver-Tabellen
        |
        v
Gold-Feature-Tabelle
        |
        v
Training und Hyperparameter-Tuning
        |
        v
MLflow Experiment Tracking
        |
        v
Unity Catalog Champion-Modell
        |
        v
FastAPI im Docker-Container
        |
        v
Streamlit-Oberfläche
```

Zusätzlich werden zwei Monitoring-Ansätze verwendet:

```text
FastAPI
    -> Prediction-Logs
    -> Laufzeitmetriken unter /monitoring

Gold-Feature-Tabelle
    -> Referenz-Snapshot
    -> PSI-basierter Drift-Report
```

## Datengrundlage

Die Quelldaten stammen aus dem Databricks-Demo Retail C360.

| Datensatz | Format | Pfad |
|---|---|---|
| Users | JSON | `/Volumes/main/dbdemos_retail_c360/c360/users` |
| Orders | JSON | `/Volumes/main/dbdemos_retail_c360/c360/orders` |
| Events | CSV | `/Volumes/main/dbdemos_retail_c360/c360/events` |

Die Pipeline schreibt ausschließlich in das eigene Schema:

```text
main.mlops_churn
```

Die Quelldaten des Databricks-Demos werden nicht verändert.

## Silver-Tabellen

Die Silver-Pipeline bereinigt und typisiert die Rohdaten.

| Tabelle | Zeilen |
|---|---:|
| `main.mlops_churn.silver_users` | 68.879 |
| `main.mlops_churn.silver_orders` | 178.061 |
| `main.mlops_churn.silver_events` | 170.992 |

## Gold-Feature-Tabelle

Die Gold-Tabelle verbindet Kunden-, Bestell- und Aktivitätsmerkmale:

```text
main.mlops_churn.gold_customer_features
```

Sie enthält eine Zeile pro `user_id`.

Direkte Identifikatoren können zur fachlichen Zuordnung in der Gold-Tabelle enthalten sein. Sie werden jedoch durch die zentrale Definition in `src/features/schema.py` explizit vom Training und von der Inferenz ausgeschlossen.

Verwendete Modellmerkmale:

```text
canal
country
gender
age_group
platform
order_count
total_amount
avg_order_amount
total_items
event_count
session_count
days_since_creation
days_since_last_activity
days_since_last_transaction
days_since_last_event
```

Zielvariable:

```text
churn
```

Technischer Schlüssel:

```text
user_id
```

## Zentrale Konfiguration

Die zentrale Konfiguration befindet sich unter:

```text
config/config.yaml
```

Wichtige Einstellungen:

| Einstellung | Wert |
|---|---|
| Catalog | `main` |
| Schema | `mlops_churn` |
| Testanteil | `0.20` |
| Random State | `42` |
| Auswahlmetrik | `pr_auc` |
| Prediction Threshold | `0.50` |
| MLflow Experiment | `/Shared/mlops-churn-prediction` |
| Registered Model | `main.mlops_churn.churn_prediction_model` |
| Modellalias | `Champion` |

## Voraussetzungen

Benötigt werden:

- Python 3.12,
- Git,
- ein Databricks Workspace,
- das installierte Retail-C360-Demo,
- Zugriff auf die verwendeten Volumes,
- Schreibrechte für `main.mlops_churn`,
- Databricks Connect,
- Databricks CLI beziehungsweise eine unterstützte Authentifizierung,
- Docker Desktop für die containerisierte API.

## Lokale Einrichtung unter Windows

Repository klonen:

```powershell
git clone https://github.com/MadMax199/mlops-churn-prediction.git
cd mlops-churn-prediction
```

Virtuelle Umgebung anlegen und aktivieren:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Abhängigkeiten installieren:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
python -m pip install -e . --no-deps
```

`requirements-dev.txt` bindet die Laufzeitabhängigkeiten aus `requirements.txt` ein und ergänzt unter anderem Pytest, Coverage, HTTPX und Ruff.

Lokale Umgebungsdatei anlegen:

```powershell
Copy-Item .env.example .env
```

## Lokale Databricks-Authentifizierung

Beispiel für die lokale `.env`:

```dotenv
DATABRICKS_CONFIG_PROFILE=max
DATABRICKS_SERVERLESS_COMPUTE_ID=auto
DATABRICKS_CLI_PATH=C:\Users\<BENUTZER>\.vscode\extensions\databricks.databricks-<VERSION>-win32-x64\bin\databricks.exe
```

Der aktuelle Pfad der Databricks CLI kann in PowerShell ermittelt werden:

```powershell
Get-ChildItem "$env:USERPROFILE\.vscode\extensions" `
    -Recurse `
    -Filter databricks.exe `
    -File |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1 -ExpandProperty FullName
```

Die Dateien `.env` und `.env.docker` werden nicht versioniert.

Verbindung testen:

```powershell
python -m scripts.test_connection
```

## Datenpipeline ausführen

### 1. Silver-Tabellen erstellen

```powershell
python -m src.data.pipeline
```

Dabei werden Users, Orders und Events geladen, bereinigt und als Delta-Tabellen geschrieben.

### 2. Gold-Feature-Tabelle erstellen

```powershell
python -m src.features.build_features
```

Die Gold-Tabelle kann anschließend in Databricks SQL geprüft werden:

```sql
SHOW TABLES IN main.mlops_churn;
```

```sql
SELECT *
FROM main.mlops_churn.gold_customer_features
LIMIT 10;
```

## Modelltraining

Die Trainingspipeline lädt die Gold-Tabelle aus Databricks und überführt sie für das lokale Training in einen Pandas DataFrame.

Folgende Modelle werden trainiert:

- Dummy Classifier,
- logistische Regression,
- Random Forest.

Training starten:

```powershell
python -m src.training.train
```

Für jedes Modell werden Parameter, Metriken, Signatur, Input-Beispiel und die vollständige scikit-learn-Pipeline in MLflow gespeichert.

## Hyperparameter-Tuning

Das Tuning des Random Forest wird gestartet mit:

```powershell
python -m src.training.tune
```

Anschließend werden Baseline und getuntes Modell verglichen:

```powershell
python -m src.training.compare_runs
```

## Modellergebnisse

| Modell | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Dummy | 0,6552 | 0,6552 | 1,0000 | 0,7917 | 0,5000 | 0,6552 |
| Logistische Regression | 0,6913 | 0,8478 | 0,6445 | 0,7323 | 0,7479 | 0,8173 |
| Random Forest | 0,8129 | 0,8540 | 0,8618 | 0,8579 | 0,7943 | 0,8357 |
| Random Forest Tuned | **0,8193** | 0,8531 | **0,8749** | **0,8639** | **0,7951** | **0,8363** |

Die Cross-Validation des getunten Random Forest erzielte eine PR-AUC von:

```text
0,8385
```

Die Test-PR-AUC verbesserte sich gegenüber dem ursprünglichen Random Forest um:

```text
+0,0006
```

Ausgewählte Hyperparameter:

| Parameter | Wert |
|---|---:|
| `n_estimators` | 500 |
| `max_depth` | 16 |
| `max_features` | 0,5 |
| `min_samples_leaf` | 4 |
| `min_samples_split` | 5 |
| `class_weight` | `None` |

Da das getunte Modell die höhere Test-PR-AUC erzielte, wurde es als Champion ausgewählt.

## Modellregistrierung

Das ausgewählte Modell wird im Unity Catalog registriert:

```powershell
python -m src.training.register_model
```

Registriertes Modell:

```text
main.mlops_churn.churn_prediction_model
```

Aktueller Stand:

```text
Version: 1
Alias: Champion
```

Das registrierte Modell kann zusätzlich validiert werden:

```powershell
python -m src.training.validate_registered_model
```

## FastAPI

FastAPI lädt beim Start das registrierte Champion-Modell aus dem Unity Catalog.

Lokal starten:

```powershell
uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8000
```

Das Laden des Modells kann ungefähr eine Minute dauern.

API-Dokumentation:

```text
http://127.0.0.1:8000/docs
```

### API-Endpunkte

| Endpoint | Methode | Beschreibung |
|---|---|---|
| `/` | GET | Allgemeine API-Informationen |
| `/health` | GET | API- und Modellstatus |
| `/model-info` | GET | Modellname, Version, Alias und URI |
| `/predict` | POST | Einzelvorhersage |
| `/monitoring` | GET | Aggregierte Prediction-Metriken |
| `/docs` | GET | Interaktive OpenAPI-Dokumentation |

### Beispielvorhersage

```powershell
$payload = @{
    canal = "web"
    country = "US"
    gender = 1
    age_group = 3
    platform = "ios"
    order_count = 12
    total_amount = 850.5
    avg_order_amount = 70.88
    total_items = 24
    event_count = 130
    session_count = 18
    days_since_creation = 540
    days_since_last_activity = 12
    days_since_last_transaction = 30
    days_since_last_event = 5
} | ConvertTo-Json
```

```powershell
Invoke-RestMethod `
    -Uri http://127.0.0.1:8000/predict `
    -Method Post `
    -ContentType "application/json" `
    -Body $payload
```

## Docker-Bereitstellung

Für den Docker-Container wird eine separate `.env.docker` verwendet.

Beispiel:

```dotenv
DATABRICKS_HOST=https://<WORKSPACE>.cloud.databricks.com
DATABRICKS_TOKEN=<TOKEN>
DATABRICKS_AUTH_TYPE=pat
```

Der verwendete Token benötigt Zugriff auf MLflow, Unity Catalog und die Modelldateien.

Secrets dürfen niemals in das Repository oder das Docker-Image übernommen werden.

Docker-Image bauen:

```powershell
docker build -t mlops-churn-api:local .
```

Container starten:

```powershell
docker run -d `
    --name mlops-churn-api `
    --env-file .env.docker `
    --log-driver json-file `
    --log-opt max-size=10m `
    --log-opt max-file=3 `
    -p 8000:8000 `
    mlops-churn-api:local
```

Status prüfen:

```powershell
docker ps --filter "name=mlops-churn-api"
```

Logs anzeigen:

```powershell
docker logs -f mlops-churn-api
```

Health-Endpoint prüfen:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health |
    Format-List
```

## Streamlit

Die Streamlit-Anwendung lädt das Modell nicht selbst. Sie verwendet die lokale FastAPI als Prediction Service.

Die FastAPI muss bereits unter Port 8000 laufen.

Streamlit starten:

```powershell
$env:CHURN_API_URL="http://127.0.0.1:8000"
python -m streamlit run src\streamlit_app\streamlit_app.py
```

Die Oberfläche ist anschließend erreichbar unter:

```text
http://localhost:8501
```

## Tests und Codequalität

Ruff-Linting ausführen:

```powershell
python -m ruff check src tests
```

Formatierung prüfen:

```powershell
python -m ruff format src tests --check
```

Tests mit Coverage ausführen:

```powershell
python -m pytest `
    --cov=src `
    --cov-report=term-missing `
    --cov-report=xml `
    -v
```

Der aktuelle Projektstand umfasst 26 erfolgreiche Tests.

Die Tests prüfen unter anderem:

- API-Endpunkte,
- Fehlerbehandlung der API,
- Aktualisierung der Monitoring-Metriken,
- Trainingsdatenvalidierung,
- reproduzierbare Datensplits,
- Ausschluss von ID und Target aus den Modellfeatures,
- Preprocessing fehlender Werte,
- Behandlung unbekannter Kategorien,
- Vorbereitung der API-Eingaben.

## GitHub Actions

Der Workflow befindet sich unter:

```text
.github/workflows/ci.yml
```

Er wird bei Pushes und Pull Requests auf `main` ausgeführt und umfasst:

1. Checkout des Repositorys,
2. Einrichtung von Python 3.12,
3. Installation der Laufzeit- und Entwicklungsabhängigkeiten,
4. Ruff-Linting,
5. Ruff-Formatierungsprüfung,
6. Pytest mit Coverage,
7. Speicherung von `coverage.xml` als Workflow-Artefakt.

Der aktuelle Workflow bildet Continuous Integration ab. Ein automatisiertes Pushen des Docker-Images oder ein automatisiertes Deployment ist noch nicht enthalten.

## Operatives Prediction-Monitoring

FastAPI erfasst während der Containerlaufzeit:

- Gesamtzahl der Vorhersageanfragen,
- erfolgreiche Vorhersagen,
- fehlgeschlagene Vorhersagen,
- Anzahl vorhergesagter Churn-Fälle,
- Anteil vorhergesagter Churn-Fälle,
- durchschnittliche Churn-Wahrscheinlichkeit,
- durchschnittliche Vorhersagelatenz,
- Modellname und Modellversion,
- Zeitpunkt der letzten Vorhersageanfrage.

Monitoring abrufen:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/monitoring |
    Format-List
```

Prediction-Ereignisse werden strukturiert in die Docker-Logs geschrieben.

Es werden keine Eingabefeatures oder direkten Kundenidentifikatoren protokolliert. Die aggregierten Metriken werden im Arbeitsspeicher gehalten und bei einem Container-Neustart zurückgesetzt.

## Feature-Drift-Monitoring

Drift-Report starten:

```powershell
python -m src.monitoring.drift_report
```

Beim ersten Lauf wird ein lokaler Referenz-Snapshot angelegt:

```text
data/reference/gold_customer_features.parquet
```

Der Snapshot enthält ausschließlich die in `FEATURE_COLUMNS` definierten Modellmerkmale.

Weitere Läufe vergleichen die aktuelle Gold-Tabelle mit der Referenz und schreiben den Bericht nach:

```text
reports/drift_report.csv
```

Bewertung der PSI-Werte:

| PSI | Bewertung |
|---:|---|
| `< 0,10` | stabil |
| `0,10 bis < 0,25` | auffällig |
| `>= 0,25` | deutlicher Drift |

## Datenschutz und Secret Management

- `.env` und `.env.docker` werden nicht versioniert.
- Databricks-Tokens werden nicht in das Docker-Image kopiert.
- Direkte Identifikatoren werden nicht als Modellfeatures verwendet.
- Der Drift-Snapshot enthält nur die definierten Modellfeatures.
- Prediction-Logs enthalten keine Eingabedaten.
- Referenzdaten und generierte Reports werden über `.gitignore` ausgeschlossen.

## Projektstruktur

```text
.github/workflows/         GitHub-Actions-Workflow
config/                    Zentrale YAML-Konfiguration
notebooks/                 Lokale Analyse- und Kontrollnotebooks
scripts/                   Verbindungstests und Hilfsskripte
src/api/                   FastAPI, Schemas und Model Loader
src/data/                  Ingestion, Bereinigung und Silver-Pipeline
src/features/              Gold-Features, Schema und Preprocessing
src/monitoring/            Prediction-Monitoring und Drift-Report
src/streamlit_app/         Streamlit-Oberfläche
src/training/              Training, Tuning, Vergleich und Registrierung
src/utils/                 Authentifizierung und Hilfsfunktionen
tests/                     Automatisierte Tests
Dockerfile                 FastAPI-Container
requirements.txt           Laufzeitabhängigkeiten
requirements-dev.txt       Entwicklungs- und Testabhängigkeiten
```

## Grenzen und mögliche Erweiterungen

- Die Datengrundlage ist ein Demo-Datensatz und keine produktive Kundenpopulation.
- Der Performancegewinn des Hyperparameter-Tunings ist gering.
- Das operative Monitoring ist prozesslokal und besitzt keine dauerhafte Metrikdatenbank.
- Die Metriken werden bei einem Container-Neustart zurückgesetzt.
- Der Drift-Report überwacht aktuell numerische Features.
- Ein späterer Outcome-Join wäre notwendig, um die Modellgüte anhand tatsächlicher Churn-Labels zu überwachen.
- Automatisiertes Continuous Deployment ist noch nicht implementiert.

## Status der Anforderungen

| Anforderung | Umsetzung |
|---|---|
| Reproduzierbare Datenvorverarbeitung | Silver-/Gold-Pipeline und zentrale YAML-Konfiguration |
| Modelltraining und Evaluation | Drei Basismodelle, Tuning und Testmetriken |
| Experiment Tracking | MLflow in Databricks |
| Grundlegende Tests | 26 Pytest-Tests und Coverage-Bericht |
| Continuous Integration | Ruff und Pytest über GitHub Actions |
| Lokale Modellbereitstellung | FastAPI im Docker-Container |
| Benutzeroberfläche | Streamlit als API-Client |
| Logging und Monitoring | Prediction-Logs, Laufzeitmetriken und PSI-Report |

## Lizenz und Nutzung

Dieses Repository wurde als MLOps-Prototyp im Rahmen einer akademischen Prüfungsleistung entwickelt.

Die verwendeten C360-Daten stammen aus einem Databricks-Demo und sind nicht Bestandteil dieses Repositorys.