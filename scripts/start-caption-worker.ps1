$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $VenvPython)) {
    throw ".venv was not found. Run scripts/setup.ps1 first."
}

$env:PYTHONPATH = Join-Path $ProjectRoot "src"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
& $VenvPython -m stylebot.caption_main --watch --interval 15 --limit 100
