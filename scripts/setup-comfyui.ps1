$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Runtime = Join-Path $ProjectRoot "work\ComfyUI"
$Python = Join-Path $ProjectRoot "work\comfyui-venv\Scripts\python.exe"

if (-not (Test-Path $Runtime)) {
    git clone --depth 1 https://github.com/Comfy-Org/ComfyUI.git $Runtime
}
if (-not (Test-Path $Python)) {
    & (Join-Path $ProjectRoot "work\python312\python.exe") -m venv (Join-Path $ProjectRoot "work\comfyui-venv")
    & $Python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
}
& $Python -m pip install -r (Join-Path $Runtime "requirements.txt")

$ModelRoot = Join-Path $ProjectRoot "models\comfyui"
@("checkpoints", "vae", "controlnet", "ipadapter") | ForEach-Object {
    New-Item -ItemType Directory -Force -Path (Join-Path $ModelRoot $_) | Out-Null
}
New-Item -ItemType Directory -Force -Path (Join-Path $ProjectRoot "outputs\comfyui") | Out-Null

$PathConfig = @"
end01:
  base_path: $($ProjectRoot.Replace('\', '/'))
  checkpoints: models/comfyui/checkpoints
  vae: models/comfyui/vae
  loras: models
  controlnet: models/comfyui/controlnet
"@
Set-Content -LiteralPath (Join-Path $Runtime "extra_model_paths.yaml") -Value $PathConfig -Encoding UTF8

& $Python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0)); print('VRAM_GB:', round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1))"
