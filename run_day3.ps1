$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$TemporaryPath = Join-Path $ProjectRoot "artifacts\temp"
$null = New-Item -ItemType Directory -Path $TemporaryPath -Force
$env:TEMP = $TemporaryPath
$env:TMP = $TemporaryPath

if (-not (Test-Path -LiteralPath $Python)) {
    throw "The virtual environment is missing. Run setup.ps1 first."
}

Push-Location $ProjectRoot
try {
    & $Python -m aeroguard.cli day3
    if ($LASTEXITCODE -ne 0) {
        throw "AeroGuard Day 3 failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
