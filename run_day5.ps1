$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$DashboardUrl = "http://127.0.0.1:8000"
$HealthUrl = "$DashboardUrl/api/health"
$Port = 8000
$ExpectedPython = [System.IO.Path]::GetFullPath($Python)

if (-not (Test-Path -LiteralPath $Python)) {
    throw "The AeroGuard environment is missing. Run setup.ps1 first."
}

function Stop-StaleAeroGuardServer {
    $listeners = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
    foreach ($listener in $listeners) {
        $ownerId = $listener.OwningProcess
        $processInfo = Get-CimInstance Win32_Process -Filter "ProcessId = $ownerId"
        $commandLine = [string]$processInfo.CommandLine
        $executablePath = [string]$processInfo.ExecutablePath
        $isAeroGuard = $processInfo.Name -like "python*" -and
            ($executablePath -eq $ExpectedPython -or $commandLine -like "*$ProjectRoot*") -and
            $commandLine -like "*aeroguard.api.main*"

        if ($isAeroGuard) {
            Write-Host "Stopping a previous AeroGuard server on port $Port..." -ForegroundColor Yellow
            Stop-Process -Id $ownerId -Force
        }
        else {
            throw "Port $Port is being used by $($processInfo.Name) (PID $ownerId). Close it and run Day 5 again."
        }
    }
}

$server = $null
Push-Location $ProjectRoot
try {
    & $Python -m aeroguard.cli day5
    if ($LASTEXITCODE -ne 0) {
        throw "The Day 5 readiness check failed."
    }

    Stop-StaleAeroGuardServer
    $LogDirectory = Join-Path $ProjectRoot "artifacts\logs"
    New-Item -ItemType Directory -Path $LogDirectory -Force | Out-Null
    $RunStamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $StandardOutput = Join-Path $LogDirectory "api-$RunStamp.stdout.log"
    $StandardError = Join-Path $LogDirectory "api-$RunStamp.stderr.log"
    $server = Start-Process `
        -FilePath $Python `
        -ArgumentList @("-m", "uvicorn", "aeroguard.api.main:app", "--host", "127.0.0.1", "--port", "$Port") `
        -WorkingDirectory $ProjectRoot `
        -WindowStyle Hidden `
        -RedirectStandardOutput $StandardOutput `
        -RedirectStandardError $StandardError `
        -PassThru

    $healthy = $false
    for ($attempt = 1; $attempt -le 40; $attempt++) {
        if ($server.HasExited) {
            $details = Get-Content -LiteralPath $StandardError -Raw -ErrorAction SilentlyContinue
            throw "AeroGuard stopped during startup. $details"
        }
        try {
            $health = Invoke-RestMethod -Uri $HealthUrl -TimeoutSec 2
            if ($health.ready) {
                $healthy = $true
                break
            }
        }
        catch {
            Start-Sleep -Milliseconds 500
        }
    }
    if (-not $healthy) {
        throw "AeroGuard did not become healthy within 20 seconds."
    }

    Write-Host "" 
    Write-Host "AeroGuard AI is ready:" -ForegroundColor Green
    Write-Host "  Dashboard  $DashboardUrl"
    Write-Host "  API docs   $DashboardUrl/docs"
    Write-Host "Press Ctrl+C to stop it."
    try {
        Start-Process $DashboardUrl
    }
    catch {
        Write-Host "The browser could not be opened automatically. Open $DashboardUrl manually." -ForegroundColor Yellow
    }

    while (-not $server.HasExited) {
        Start-Sleep -Seconds 1
        $server.Refresh()
    }
    throw "AeroGuard stopped unexpectedly. See $StandardError"
}
finally {
    if ($server -and -not $server.HasExited) {
        Stop-Process -Id $server.Id -Force -ErrorAction SilentlyContinue
    }
    Pop-Location
}
