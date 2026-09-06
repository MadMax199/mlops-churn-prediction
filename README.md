## Vollständigen Prototyp ausführen

Dieser Abschnitt beschreibt den vollständigen Ablauf von der lokalen Einrichtung bis zur Vorhersage und zum Monitoring.

Die Schritte müssen beim erstmaligen Aufbau grundsätzlich in der angegebenen Reihenfolge ausgeführt werden.

### 1. Virtuelle Umgebung aktivieren

Im Projektverzeichnis:

```powershell
cd C:\Users\<BENUTZER>\mlops-churn-prediction
.\.venv\Scripts\Activate.ps1
```

Prüfen, ob die richtige Python-Umgebung aktiv ist:

```powershell
python -c "import sys; print(sys.executable)"
python --version
```

Erwartet wird ein Interpreter innerhalb des Projektordners:

```text
mlops-churn-prediction\.venv\Scripts\python.exe
Python 3.12.x
```

Falls PowerShell weiterhin einen anderen Python-Interpreter verwendet, können alle Befehle explizit mit der virtuellen Umgebung ausgeführt werden:

```powershell
& ".\.venv\Scripts\python.exe" -m pytest -v
```

### 2. Abhängigkeiten installieren

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
python -m pip install -e . --no-deps
```

Die Datei `requirements-dev.txt` bindet `requirements.txt` ein und installiert zusätzlich die Entwicklungs- und Testabhängigkeiten.

Installation prüfen:

```powershell
python -m pip check
```

Erwartete Ausgabe:

```text
No broken requirements found.
```

### 3. Lokale Databricks-Konfiguration vorbereiten

Falls noch nicht vorhanden:

```powershell
Copy-Item .env.example .env
```

Beispiel für `.env`:

```dotenv
DATABRICKS_CONFIG_PROFILE=max
DATABRICKS_SERVERLESS_COMPUTE_ID=auto
DATABRICKS_CLI_PATH=C:\Users\<BENUTZER>\.vscode\extensions\databricks.databricks-<VERSION>-win32-x64\bin\databricks.exe
```

Aktuellen CLI-Pfad ermitteln:

```powershell
$databricksCli = Get-ChildItem "$env:USERPROFILE\.vscode\extensions" `
    -Recurse `
    -Filter databricks.exe `
    -File |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1 -ExpandProperty FullName
$databricksCli
Test-Path $databricksCli
```

Die Databricks-CLI ist Bestandteil der VS-Code-Erweiterung. Der Installationspfad enthält deren Versionsnummer und kann sich nach einem Update ändern. `Test-Path` muss deshalb `True` zurückgeben.

Die Einstellungen für das aktuelle PowerShell-Terminal setzen:

```powershell
$env:DATABRICKS_CLI_PATH = $databricksCli
$env:DATABRICKS_CONFIG_PROFILE = "max"
$env:DATABRICKS_SERVERLESS_COMPUTE_ID = "auto"
```

Den ermittelten CLI-Pfad zusätzlich in der lokalen `.env` unter `DATABRICKS_CLI_PATH` eintragen. Die `.env` wird nicht versioniert.

Anmeldung prüfen:

```powershell
& $databricksCli auth profiles
& $databricksCli current-user me --profile max
```

Falls das Profil nicht mehr gültig ist, die Browser-Anmeldung erneuern:

```powershell
& $databricksCli auth login `
    --host "https://dbc-ffd64f04-ac70.cloud.databricks.com" `
    --profile max
```

Databricks-Verbindung des Projekts prüfen:

```powershell
python -m scripts.test_connection
```

### 4. Silver-Pipeline ausführen

```powershell
python -m src.data.pipeline
```

Die Pipeline liest die drei C360-Quelldatensätze, bereinigt sie und schreibt:

```text
main.mlops_churn.silver_users
main.mlops_churn.silver_orders
main.mlops_churn.silver_events
```

Erwartete Größenordnung:

```text
silver_users     68.879 Zeilen
silver_orders   178.061 Zeilen
silver_events   170.992 Zeilen
```

Die C360-Quelldaten werden nicht verändert.

### 5. Gold-Feature-Tabelle erzeugen

```powershell
python -m src.features.build_features
```

Dadurch wird folgende Tabelle erstellt:

```text
main.mlops_churn.gold_customer_features
```

Die Tabelle enthält eine Zeile pro Kunde und dient als Grundlage für das Modelltraining.

In Databricks SQL kontrollieren:

```sql
SHOW TABLES IN main.mlops_churn;
```

```sql
SELECT *
FROM main.mlops_churn.gold_customer_features
LIMIT 10;
```

Die Gold-Tabelle wird über eine Positivliste auf `user_id`, die definierten Modellfeatures und `churn` begrenzt. Direkte Identifikatoren dürfen nicht enthalten sein.

Das gespeicherte Schema lokal prüfen:

```powershell
python -c "from src.session import get_spark_session; df=get_spark_session().table('main.mlops_churn.gold_customer_features'); forbidden={'email','firstname','lastname','address'}; print(df.columns); print('Zeilen:', df.count()); print('Unerlaubte Spalten:', sorted(forbidden.intersection(df.columns))); assert not forbidden.intersection(df.columns)"
```

Erwartete Abschlussmeldung:

```text
Unerlaubte Spalten: []
```

### 6. Basismodelle trainieren

```powershell
python -m src.training.train
```

Dabei werden folgende Modelle trainiert und in MLflow protokolliert:

```text
Dummy Classifier
Logistische Regression
Random Forest
```

Die Runs werden im Experiment gespeichert:

```text
/Shared/mlops-churn-prediction
```

### 7. Hyperparameter-Tuning ausführen

```powershell
python -m src.training.tune
```

Das Tuning optimiert den Random Forest und speichert den getunten Run ebenfalls in MLflow.

### 8. Modelle vergleichen

```powershell
python -m src.training.compare_runs
```

Der Vergleich verwendet die PR-AUC als Auswahlmetrik und dokumentiert die Modellentscheidung.

Aktueller Vergleich:

```text
Random Forest PR-AUC:         0,8357
Random Forest Tuned PR-AUC:   0,8363
Differenz:                   +0,0006
```

### 9. Ausgewähltes Modell registrieren

```powershell
python -m src.training.register_model
```

Das ausgewählte Modell wird im Unity Catalog registriert:

```text
main.mlops_churn.churn_prediction_model
```

Aktueller Modellstand:

```text
Version: 1
Alias: Champion
```

Ein erneuter Aufruf der Registrierung kann eine neue Modellversion erzeugen. Wenn bereits ein gültiges Champion-Modell registriert ist und kein neues Training durchgeführt wurde, muss dieser Schritt nicht erneut ausgeführt werden.

### 10. Registriertes Modell validieren

```powershell
python -m src.training.validate_registered_model
```

Dabei werden das registrierte Modell, seine Signatur und die Ausführung einer Beispielvorhersage geprüft.

### 11. Tests und Codequalität prüfen

Ruff-Linting:

```powershell
python -m ruff check src tests
```

Ruff-Formatierung:

```powershell
python -m ruff format src tests --check
```

Vollständige Testsuite mit Coverage:

```powershell
python -m pytest `
    --cov=src `
    --cov-report=term-missing `
    --cov-report=xml `
    -v
```

Der aktuelle Projektstand umfasst 26 erfolgreiche Tests.

### 12. Docker-Authentifizierung vorbereiten

Für den Docker-Container wird eine separate `.env.docker` benötigt:

```dotenv
DATABRICKS_HOST=https://<WORKSPACE>.cloud.databricks.com
DATABRICKS_TOKEN=<TOKEN>
DATABRICKS_AUTH_TYPE=pat
```

Der Token benötigt Zugriff auf:

```text
MLflow
Unity Catalog
Modelldateien
```

Die Datei `.env.docker` wird durch `.gitignore` ausgeschlossen und darf nicht committed werden.

### 13. Docker-Image bauen

```powershell
docker build -t mlops-churn-api:local .
```

Prüfen, ob das Image vorhanden ist:

```powershell
docker image ls mlops-churn-api
```

### 14. FastAPI-Container starten

Falls bereits ein alter Container vorhanden ist:

```powershell
docker rm -f mlops-churn-api
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

Containerstatus prüfen:

```powershell
docker ps -a --filter "name=mlops-churn-api"
```

Startprotokoll anzeigen:

```powershell
docker logs -f mlops-churn-api
```

Das Laden des Champion-Modells kann ungefähr eine Minute dauern.

Die API ist bereit, sobald folgende Meldungen erscheinen:

```text
Model version 1 loaded
Application startup complete.
Uvicorn running on http://0.0.0.0:8000
```

Die Loganzeige kann anschließend mit `Strg + C` verlassen werden. Der Container läuft dabei weiter.

### 15. FastAPI prüfen

Health-Endpoint:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health |
    Format-List
```

Erwartete Ausgabe:

```text
status       : ok
model_loaded : True
```

Modellinformationen:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/model-info |
    Format-List
```

Monitoring-Ausgangszustand:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/monitoring |
    Format-List
```

Die interaktive API-Dokumentation ist verfügbar unter:

```text
http://127.0.0.1:8000/docs
```

### 16. Streamlit starten

Streamlit wird in einem zweiten PowerShell-Terminal gestartet. Der FastAPI-Container muss währenddessen weiterlaufen.

```powershell
cd C:\Users\<BENUTZER>\mlops-churn-prediction
.\.venv\Scripts\Activate.ps1
$env:CHURN_API_URL="http://127.0.0.1:8000"
python -m streamlit run src\streamlit_app\streamlit_app.py
```

Die Benutzeroberfläche ist anschließend verfügbar unter:

```text
http://localhost:8501
```

In Streamlit können die Kundenmerkmale eingegeben und über den Button **„Churn-Risiko berechnen“** an FastAPI gesendet werden.

### 17. Prediction-Monitoring prüfen

Nach einer Vorhersage in Streamlit:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/monitoring |
    Format-List
```

Nach einer erfolgreichen Vorhersage sollte mindestens Folgendes erscheinen:

```text
total_requests         : 1
successful_predictions : 1
failed_predictions     : 0
model_version          : 1
```

Strukturierte Prediction-Logs anzeigen:

```powershell
docker logs --tail 30 mlops-churn-api
```

Die Logs enthalten unter anderem:

```text
prediction
churn_probability
latency_ms
model_version
```

Es werden keine Eingabefeatures oder direkten Kundenidentifikatoren protokolliert.

### 18. Feature-Drift-Report ausführen

```powershell
python -m src.monitoring.drift_report
```

Beim ersten Lauf wird der Referenz-Snapshot angelegt:

```text
data/reference/gold_customer_features.parquet
```

Beim zweiten und bei allen weiteren Läufen wird der Drift-Bericht erzeugt:

```powershell
python -m src.monitoring.drift_report
```

Ausgabedatei:

```text
reports/drift_report.csv
```

Die Bewertung erfolgt anhand des Population Stability Index:

| PSI | Bewertung |

|---:|---|

| `< 0,10` | stabil |

| `0,10 bis < 0,25` | auffällig |

| `>= 0,25` | deutlicher Drift |

Direkt aufeinanderfolgende Läufe verwenden zunächst dieselben Gold-Daten. Deshalb werden dabei PSI-Werte nahe `0` erwartet.

### 19. Änderungen über GitHub Actions prüfen

Vor dem Push lokal prüfen:

```powershell
python -m ruff check src tests
python -m ruff format src tests --check
python -m pytest -v
```

Änderungen committen:

```powershell
git status --short
git add .
git commit -m "Update MLOps prototype"
git push origin main
```

Der Push startet automatisch den Workflow:

```text
.github/workflows/ci.yml
```

GitHub Actions führt anschließend Ruff, Pytest, die Coverage-Erstellung sowie den Docker-Build mit Container-Smoke-Test aus.

Erwartete erfolgreiche Jobs:

```text
Ruff and pytest
Docker build and smoke test
```

## Kompakter Ablauf nach abgeschlossener Einrichtung

Wenn Umgebung, Authentifizierung und Modellregistrierung bereits eingerichtet sind, genügt für einen erneuten vollständigen Daten- und Trainingslauf:

```powershell
.\.venv\Scripts\Activate.ps1
python -m scripts.test_connection
python -m src.data.pipeline
python -m src.features.build_features
python -m src.training.train
python -m src.training.tune
python -m src.training.compare_runs
python -m src.training.register_model
python -m src.training.validate_registered_model
python -m src.monitoring.drift_report
python -m ruff check src tests
python -m pytest -v
```

Für die lokale Anwendung müssen anschließend FastAPI und Streamlit gestartet werden:

```text
Terminal 1: Docker-Container mit FastAPI
Terminal 2: Streamlit-Oberfläche
Terminal 3: optionale Status- und Monitoring-Abfragen
```
