$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Run setup.ps1 first. The project virtual environment does not exist."
}

Push-Location $ProjectRoot
try {
    & $Python -m aeroguard.cli day1
}
finally {
    Pop-Location
}

