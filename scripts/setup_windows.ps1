param(
    [Parameter(Mandatory = $true)]
    [string]$WorkspaceUrl
)

$ErrorActionPreference = "Stop"

if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    throw "Python Launcher 'py' is missing. Install Python 3.12 first."
}

$pythonVersion = & py -3.12 -c "import sys; print('.'.join(map(str, sys.version_info[:2])))"
if ($LASTEXITCODE -ne 0 -or $pythonVersion -ne "3.12") {
    throw "Python 3.12 is required. Install it and rerun this script."
}

if (-not (Test-Path ".venv")) {
    & py -3.12 -m venv .venv
}

& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -e ".[dev]"

if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
}

if (-not (Get-Command databricks -ErrorAction SilentlyContinue)) {
    throw "Databricks CLI is missing. Install the current CLI, then rerun this script."
}

& databricks auth login --host $WorkspaceUrl --profile DEFAULT

Write-Host ""
Write-Host "Environment ready. Activate it with:"
Write-Host ".\.venv\Scripts\Activate.ps1"
Write-Host ""
Write-Host "Then validate Databricks Connect with:"
Write-Host "databricks-connect test"
Write-Host "python scripts/test_connection.py"

