from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


DEFAULT_NEGATIVE = (
    "low quality, worst quality, blurry, deformed anatomy, extra limbs, "
    "bad hands, text, watermark, signature"
)
OPENPOSE_CONTROLNET = "xinsir/controlnet-openpose-sdxl-1.0"


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate an SDXL image with LoRA adapters")
    parser.add_argument("--request", required=True)
    args = parser.parse_args()
    request_path = Path(args.request).resolve()
    request = json.loads(request_path.read_text(encoding="utf-8"))

    import torch
    from diffusers import (
        ControlNetModel,
        StableDiffusionXLControlNetPipeline,
        StableDiffusionXLPipeline,
    )
    from PIL import Image

    dtype = torch.bfloat16
    base_model = request.get("base_model", "stabilityai/stable-diffusion-xl-base-1.0")
    pose_path = request.get("pose_path")
    control_image = None
    if pose_path:
        from controlnet_aux import OpenposeDetector

        detector = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
        control_image = detector(
            Image.open(pose_path).convert("RGB"), hand_and_face=True
        ).resize((request["width"], request["height"]))
        controlnet = ControlNetModel.from_pretrained(
            request.get("controlnet_model", OPENPOSE_CONTROLNET),
            torch_dtype=dtype,
            use_safetensors=True,
        )
        pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            base_model, controlnet=controlnet, torch_dtype=dtype, use_safetensors=True
        )
    else:
        pipe = StableDiffusionXLPipeline.from_pretrained(
            base_model, torch_dtype=dtype, use_safetensors=True
        )

    adapter_names: list[str] = []
    adapter_weights: list[float] = []
    for index, adapter in enumerate(request["adapters"]):
        adapter_name = f"adapter_{index}"
        path = Path(adapter["path"])
        pipe.load_lora_weights(
            path.parent.as_posix(), weight_name=path.name, adapter_name=adapter_name
        )
        adapter_names.append(adapter_name)
        adapter_weights.append(float(adapter["strength"]))
    if adapter_names:
        pipe.set_adapters(adapter_names, adapter_weights=adapter_weights)

    pipe.enable_model_cpu_offload()
    pipe.enable_vae_tiling()
    generator = torch.Generator(device="cpu").manual_seed(int(request["seed"]))
    kwargs = {
        "prompt": request["prompt"],
        "negative_prompt": request.get("negative_prompt") or DEFAULT_NEGATIVE,
        "width": int(request["width"]),
        "height": int(request["height"]),
        "num_inference_steps": int(request.get("steps", 30)),
        "guidance_scale": float(request.get("guidance_scale", 6.5)),
        "generator": generator,
    }
    if control_image is not None:
        kwargs.update(
            image=control_image,
            controlnet_conditioning_scale=float(request.get("pose_strength", 0.8)),
        )
    result = pipe(**kwargs).images[0]
    output = Path(request["output_path"]).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    result.save(output, "PNG")
    metadata_path = output.with_suffix(".json")
    metadata_path.write_text(json.dumps(request, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"image_path": output.as_posix(), "metadata_path": metadata_path.as_posix()}))
    return 0


if __name__ == "__main__":
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    raise SystemExit(main())
