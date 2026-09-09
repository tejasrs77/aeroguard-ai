$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "The virtual environment is missing. Run setup.ps1 first."
}

Push-Location $ProjectRoot
try {
    & $Python -m aeroguard.cli day2
    if ($LASTEXITCODE -ne 0) {
        throw "AeroGuard Day 2 failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
