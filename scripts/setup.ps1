$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot
$BundledPython = Join-Path $ProjectRoot "work\python312\python.exe"
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (Test-Path $VenvPython) {
    Write-Host "Using existing .venv"
} elseif (Test-Path $BundledPython) {
    & $BundledPython -m venv .venv
} else {
    $PythonCommand = Get-Command py -ErrorAction SilentlyContinue
    if ($PythonCommand) {
    & py -3.12 -m venv .venv
    } else {
        $PythonCommand = Get-Command python -ErrorAction SilentlyContinue
        if (-not $PythonCommand) {
            throw "Python was not found. Install Python 3.12 first."
        }
        & python -m venv .venv
    }
}

& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt

if (-not (Test-Path .env)) {
    Copy-Item .env.example .env
}
if (-not (Test-Path config\styles.yaml)) {
    Copy-Item config\styles.example.yaml config\styles.yaml
}
if (-not (Test-Path config\training.yaml)) {
    Copy-Item config\training.example.yaml config\training.yaml
}

Write-Host "Setup complete. Fill .env and config\styles.yaml, then run scripts\start.ps1"
