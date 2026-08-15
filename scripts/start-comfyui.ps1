$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Runtime = Join-Path $ProjectRoot "work\ComfyUI"
$Python = Join-Path $ProjectRoot "work\comfyui-venv\Scripts\python.exe"
$Paths = Join-Path $Runtime "extra_model_paths.yaml"
$Output = Join-Path $ProjectRoot "outputs\comfyui"

if (-not (Test-Path $Python) -or -not (Test-Path $Paths)) {
    throw "ComfyUI is not configured. Run scripts/setup-comfyui.ps1 first."
}
& $Python (Join-Path $Runtime "main.py") --listen 127.0.0.1 --port 8188 --extra-model-paths-config $Paths --output-directory $Output
