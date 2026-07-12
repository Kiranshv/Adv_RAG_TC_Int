param(
    [int]$Port = 5050,
    [switch]$NoHealthCheck
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonExe = Join-Path $projectRoot ".venv\Scripts\python.exe"
$appPath = Join-Path $projectRoot "app.py"
$envPath = Join-Path $projectRoot ".env"

if (-not (Test-Path $pythonExe)) {
    throw "Python executable not found at $pythonExe"
}

if (-not (Test-Path $appPath)) {
    throw "App file not found at $appPath"
}

if (Test-Path $envPath) {
    $envText = Get-Content -Path $envPath -Raw
    $providerMatch = [regex]::Match($envText, "(?m)^LLM_PROVIDER\s*=\s*(.*)$")
    $provider = if ($providerMatch.Success) { $providerMatch.Groups[1].Value.Trim().ToLower() } else { "openrouter" }

    $openrouterMatch = [regex]::Match($envText, "(?m)^OPENROUTER_API_KEY\s*=\s*(.*)$")
    $groqMatch = [regex]::Match($envText, "(?m)^GROQ_API_KEY\s*=\s*(.*)$")

    $openrouterKey = if ($openrouterMatch.Success) { $openrouterMatch.Groups[1].Value.Trim() } else { "" }
    $groqKey = if ($groqMatch.Success) { $groqMatch.Groups[1].Value.Trim() } else { "" }

    if (($provider -eq "openrouter" -and [string]::IsNullOrWhiteSpace($openrouterKey)) -or
        ($provider -eq "groq" -and [string]::IsNullOrWhiteSpace($groqKey))) {
        Write-Warning "LLM key for provider '$provider' is empty in .env. Chat will use fallback text until you set it."
    }
} else {
    Write-Warning ".env not found. Create it from .env.example and set API key(s)."
}

Write-Output "Stopping existing app processes..."
$targets = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -eq "python.exe" -and
    $_.CommandLine -match "Advanced_RAG/Advance_RAG_EXPLAIN/app.py|Advanced_RAG_EXPLAIN\\app.py"
}

if ($targets) {
    $ids = $targets | Select-Object -ExpandProperty ProcessId
    $ids | ForEach-Object { Stop-Process -Id $_ -Force }
    Write-Output ("Stopped PID(s): " + ($ids -join ", "))
} else {
    Write-Output "No running app process found."
}

Write-Output "Starting app..."
Start-Process -FilePath $pythonExe -ArgumentList $appPath -WorkingDirectory $projectRoot | Out-Null

if (-not $NoHealthCheck) {
    Write-Output "Waiting for health endpoint..."
    $ok = $false
    for ($i = 0; $i -lt 20; $i++) {
        Start-Sleep -Milliseconds 500
        try {
            $health = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/api/health" -Method Get -TimeoutSec 8
            $ok = $true
            Write-Output "App is up on http://127.0.0.1:$Port"
            Write-Output ("chunks_indexed=" + $health.chunks_indexed)
            Write-Output ("llm_provider=" + $health.llm_provider)
            break
        } catch {
            # retry until timeout loop ends
        }
    }

    if (-not $ok) {
        throw "App did not become healthy in time."
    }
}
