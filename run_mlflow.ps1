$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$DatabasePath = (Join-Path $ProjectRoot "artifacts\mlruns\mlflow.db").Replace("\", "/")
$ArtifactPath = (Join-Path $ProjectRoot "artifacts\mlruns\artifacts").Replace("\", "/")
$TemporaryPath = Join-Path $ProjectRoot "artifacts\temp"
$null = New-Item -ItemType Directory -Path $TemporaryPath -Force
$env:TEMP = $TemporaryPath
$env:TMP = $TemporaryPath
$env:MLFLOW_DISABLE_AGENT_HINT = "1"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "The virtual environment is missing. Run setup.ps1 first."
}
if (-not (Test-Path -LiteralPath (Join-Path $ProjectRoot "artifacts\mlruns\mlflow.db"))) {
    throw "No MLflow database exists. Run run_day2.ps1 first."
}

$PortInUse = [System.Net.NetworkInformation.IPGlobalProperties]::GetIPGlobalProperties().GetActiveTcpListeners() |
    Where-Object { $_.Port -eq 5000 } |
    Select-Object -First 1
if ($PortInUse) {
    throw "MLflow cannot start because port 5000 is already in use. Close the application using it and try again."
}

Write-Host "MLflow is starting at http://127.0.0.1:5000" -ForegroundColor Green
Write-Host "Press Ctrl+C to stop it."
& $Python -m mlflow ui `
    --backend-store-uri "sqlite:///$DatabasePath" `
    --default-artifact-root "file:///$ArtifactPath" `
    --host 127.0.0.1 `
    --port 5000 `
    --workers 1
