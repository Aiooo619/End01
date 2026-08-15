# Character Design AI V1 upgrade

End01 remains the source project. Heavy runtimes and weights stay under ignored
`work/` and `models/` directories; Discord ingestion, training history, v001,
and v002 remain intact.

## Verified environment

- Windows 11
- NVIDIA RTX 4070 Ti SUPER, 16GB VRAM
- LoRA runtime: Python 3.10, PyTorch 2.6.0 + CUDA 12.4
- ComfyUI runtime: Python 3.12, PyTorch 2.13.0 + CUDA 13.0
- ComfyUI 0.33.0 at `http://127.0.0.1:8188`

The two Python environments are intentionally isolated.

## Dataset V1 target

Grow from 51 reviewed full-body images to 80–150 high-quality images:

- 60% full-body character illustrations
- 25% half-body clothing-structure references
- 15% clothing/accessory detail references

Do not duplicate or physically move the currently registered files until the
dataset database supports categories. The category folders under
`datasets/_template/` define the next import schema.

## Evaluation

`config/evaluation_prompts.json` is the fixed out-of-dataset evaluation set.
Every checkpoint comparison must keep prompt, negative prompt, seed, sampler,
steps, CFG, resolution, and LoRA strength constant.

## ComfyUI

Run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup-comfyui.ps1
powershell -ExecutionPolicy Bypass -File scripts/start-comfyui.ps1
```

ComfyUI reads checkpoints from `models/comfyui/checkpoints`, LoRAs recursively
from `models/`, and writes images to `outputs/comfyui`.

`comfyui/workflows/CharacterDesignGenerator.json` is an API workflow skeleton.
It intentionally contains a placeholder checkpoint until the Illustrious model
choice and license are confirmed.

## Base-model migration rule

v001 and v002 were trained against Stability AI SDXL Base 1.0. Keep them for
comparison. Do not relabel them as Illustrious-compatible models. A future v003
must be trained from the selected Illustrious XL base after the dataset reaches
the V1 quality and size target.
