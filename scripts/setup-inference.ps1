$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot "work\sd-scripts\venv\Scripts\python.exe"

if (-not (Test-Path $Python)) {
    throw "Training environment was not found. Install sd-scripts first."
}

& $Python -m pip install -r (Join-Path $ProjectRoot "requirements-inference.txt")

$BaseModel = Join-Path $ProjectRoot "models\base\sdxl-base-1.0"
if (-not (Test-Path (Join-Path $BaseModel "model_index.json"))) {
    $DownloadScript = @'
from huggingface_hub import snapshot_download

patterns = [
    "model_index.json", "scheduler/*", "tokenizer/*", "tokenizer_2/*",
    "text_encoder/config.json", "text_encoder/model.safetensors",
    "text_encoder_2/config.json", "text_encoder_2/model.safetensors",
    "unet/config.json", "unet/diffusion_pytorch_model.safetensors",
    "vae/config.json", "vae/diffusion_pytorch_model.safetensors",
]
snapshot_download(
    "stabilityai/stable-diffusion-xl-base-1.0",
    local_dir=r"__BASE_MODEL__",
    allow_patterns=patterns,
)
'@
    $DownloadScript = $DownloadScript.Replace("__BASE_MODEL__", $BaseModel)
    & $Python -c $DownloadScript
}
