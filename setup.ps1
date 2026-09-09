$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VirtualEnvironment = Join-Path $ProjectRoot ".venv"
$Python = Join-Path $VirtualEnvironment "Scripts\python.exe"

if (-not (Test-Path -LiteralPath $Python)) {
    $PyLauncher = Get-Command "py" -ErrorAction SilentlyContinue
    $PythonCommand = Get-Command "python" -ErrorAction SilentlyContinue
    if ($PyLauncher) {
        & $PyLauncher.Source -3.11 -m venv $VirtualEnvironment
    }
    elseif ($PythonCommand) {
        & $PythonCommand.Source -m venv $VirtualEnvironment
    }
    else {
        throw "Python 3.11 was not found. Install it, reopen PowerShell, and run setup.ps1 again."
    }
}

& $Python -m pip install --upgrade pip
& $Python -m pip install -e "${ProjectRoot}[dev]"

Write-Host "AeroGuard environment is ready." -ForegroundColor Green
Write-Host "Run Day 1 with: powershell -ExecutionPolicy Bypass -File .\run_day1.ps1"
Write-Host "Run Day 2 with: powershell -ExecutionPolicy Bypass -File .\run_day2.ps1"
Write-Host "Run Day 3 with: powershell -ExecutionPolicy Bypass -File .\run_day3.ps1"
Write-Host "Run Day 4 with: powershell -ExecutionPolicy Bypass -File .\run_day4.ps1"
Write-Host "Run Day 5 and open the dashboard with: powershell -ExecutionPolicy Bypass -File .\run_day5.ps1"
