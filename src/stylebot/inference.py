from __future__ import annotations

import json
import os
import random
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path

from .config import Settings
from .registry import ModelRecord


@dataclass(frozen=True)
class GenerationResult:
    image_path: Path
    request_path: Path
    seed: int


class InferenceRunner:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.python = (
            settings.project_root / "work" / "sd-scripts" / "venv" / "Scripts" / "python.exe"
        )

    def training_active(self) -> bool:
        for path in (self.settings.data_root / "queues").glob("*.json"):
            job = json.loads(path.read_text(encoding="utf-8"))
            if job.get("status") == "running":
                return True
        return False

    def generate(
        self,
        models: list[tuple[ModelRecord, float]],
        prompt: str,
        negative_prompt: str = "",
        seed: int = -1,
        width: int = 1024,
        height: int = 1024,
        purpose: str = "generate",
        pose_path: Path | None = None,
        pose_strength: float = 0.8,
    ) -> GenerationResult:
        if not self.python.exists():
            raise RuntimeError("生圖環境尚未安裝。")
        if self.training_active():
            raise RuntimeError("GPU 正在訓練；請等待訓練完成後再生圖。")
        if not models:
            raise ValueError("至少需要一個模型。")
        if width not in {768, 832, 896, 1024, 1152, 1216, 1344}:
            raise ValueError("不支援的寬度。")
        if height not in {768, 832, 896, 1024, 1152, 1216, 1344}:
            raise ValueError("不支援的高度。")
        seed = random.randint(0, 2**31 - 1) if seed < 0 else seed
        request_id = uuid.uuid4().hex[:16]
        output_root = self.settings.data_root / "outputs" / (
            "previews" if purpose == "review" else "generated"
        )
        output = output_root / f"{request_id}.png"
        request_root = self.settings.data_root / "queues" / "generation_requests"
        request_root.mkdir(parents=True, exist_ok=True)
        request_path = request_root / f"{request_id}.json"
        local_base = self.settings.data_root / "models" / "base" / "sdxl-base-1.0"
        base_model = (
            local_base.as_posix()
            if (local_base / "model_index.json").exists()
            else "stabilityai/stable-diffusion-xl-base-1.0"
        )
        request = {
            "request_id": request_id,
            "base_model": base_model,
            "adapters": [
                {
                    "model_id": model.model_id,
                    "path": (self.settings.data_root / model.path).as_posix(),
                    "strength": strength,
                }
                for model, strength in models
            ],
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "seed": seed,
            "width": width,
            "height": height,
            "steps": 30,
            "guidance_scale": 6.5,
            "purpose": purpose,
            "pose_path": pose_path.as_posix() if pose_path else None,
            "pose_strength": pose_strength,
            "controlnet_model": "xinsir/controlnet-openpose-sdxl-1.0",
            "output_path": output.as_posix(),
        }
        request_path.write_text(json.dumps(request, ensure_ascii=False, indent=2), encoding="utf-8")
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(self.settings.project_root / "src")
        environment["HF_HOME"] = str(self.settings.data_root / "cache" / "huggingface")
        environment["MPLCONFIGDIR"] = str(self.settings.data_root / "cache" / "matplotlib")
        process = subprocess.run(
            [str(self.python), "-m", "stylebot.generate_main", "--request", str(request_path)],
            cwd=self.settings.project_root,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        if process.returncode != 0:
            raise RuntimeError(process.stderr[-3000:] or process.stdout[-3000:])
        if not output.exists():
            raise RuntimeError("生圖程序完成但找不到輸出圖片。")
        return GenerationResult(output, request_path, seed)
