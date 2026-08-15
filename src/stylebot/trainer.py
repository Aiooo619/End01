from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import yaml

from .config import Settings
from .storage import DatasetStore, IngestError


DEFAULTS = {
    "base_model": "stabilityai/stable-diffusion-xl-base-1.0",
    "resolution": 1024,
    "network_dim": 32,
    "network_alpha": 16,
    "learning_rate": 0.0001,
    "epochs": 10,
    "repeats": 5,
    "seed": 42,
    "mixed_precision": "bf16",
    "save_every_n_epochs": 1,
}


@dataclass(frozen=True)
class PreparedRun:
    job_path: Path
    run_root: Path
    dataset_config: Path
    output_dir: Path
    output_name: str
    command: list[str]
    image_count: int


class TrainingWorker:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.store = DatasetStore(settings)
        self.sd_scripts = settings.project_root / "work" / "sd-scripts"
        self.python = self.sd_scripts / "venv" / "Scripts" / "python.exe"
        self.script = self.sd_scripts / "sdxl_train_network.py"
        self.config = dict(DEFAULTS)
        config_path = settings.project_root / "config" / "training.yaml"
        if not config_path.exists():
            config_path = settings.project_root / "config" / "training.example.yaml"
        if config_path.exists():
            with config_path.open("r", encoding="utf-8") as handle:
                self.config.update(yaml.safe_load(handle) or {})

    def _next_version(self, style_id: str) -> int:
        root = self.settings.data_root / "models" / style_id
        versions = [
            int(path.name[1:]) for path in root.glob("v[0-9][0-9][0-9]")
            if path.name[1:].isdigit()
        ] if root.exists() else []
        return max(versions, default=0) + 1

    def prepare(self, job_path: Path) -> PreparedRun:
        if not self.python.exists() or not self.script.exists():
            raise IngestError("sd-scripts 訓練環境尚未安裝。")
        job = json.loads(job_path.read_text(encoding="utf-8"))
        style = self.settings.styles.get(job["style_id"])
        if not style:
            raise IngestError("訓練工作引用了不存在的風格。")
        items = self.store.approved_items(style.style_id)
        if len(items) < style.minimum_approved_images:
            raise IngestError(
                f"已批准 {len(items)} 張，至少需要 {style.minimum_approved_images} 張。"
            )

        version = self._next_version(style.style_id)
        version_name = f"v{version:03d}"
        run_root = self.settings.project_root / "work" / "training_runs" / job["job_id"]
        image_root = run_root / "images"
        image_root.mkdir(parents=True, exist_ok=True)
        missing_captions: list[str] = []
        for item in items:
            source = self.settings.data_root / str(item["local_path"])
            caption = (
                self.settings.data_root / "datasets" / style.style_id / "captions" / f"{source.stem}.txt"
            )
            if not caption.exists():
                missing_captions.append(source.name)
                continue
            shutil.copy2(source, image_root / source.name)
            shutil.copy2(caption, image_root / caption.name)
        if missing_captions:
            raise IngestError(f"有 {len(missing_captions)} 張已批准圖片缺少 caption。")

        dataset_config = run_root / "dataset.toml"
        dataset_config.write_text(
            "\n".join(
                [
                    "[general]",
                    'caption_extension = ".txt"',
                    "shuffle_caption = false",
                    "keep_tokens = 2",
                    "",
                    "[[datasets]]",
                    f"resolution = {int(self.config['resolution'])}",
                    "batch_size = 1",
                    "enable_bucket = true",
                    "min_bucket_reso = 512",
                    "max_bucket_reso = 1536",
                    "bucket_reso_steps = 64",
                    "",
                    "[[datasets.subsets]]",
                    f'image_dir = "{image_root.as_posix()}"',
                    f"num_repeats = {int(self.config['repeats'])}",
                    "",
                ]
            ),
            encoding="utf-8",
        )

        output_dir = self.settings.data_root / "models" / style.style_id / version_name
        output_dir.mkdir(parents=True, exist_ok=True)
        output_name = f"{style.style_id}_{version_name}"
        command = [
            str(self.python), "-m", "accelerate.commands.launch",
            "--num_cpu_threads_per_process=1",
            f"--mixed_precision={self.config['mixed_precision']}",
            str(self.script),
            f"--pretrained_model_name_or_path={self.config['base_model']}",
            f"--dataset_config={dataset_config}",
            f"--output_dir={output_dir}",
            f"--output_name={output_name}",
            "--save_model_as=safetensors",
            "--network_module=networks.lora",
            f"--network_dim={int(self.config['network_dim'])}",
            f"--network_alpha={int(self.config['network_alpha'])}",
            f"--learning_rate={float(self.config['learning_rate'])}",
            "--optimizer_type=AdamW8bit",
            "--lr_scheduler=cosine",
            f"--max_train_epochs={int(self.config['epochs'])}",
            f"--mixed_precision={self.config['mixed_precision']}",
            f"--save_precision={self.config['mixed_precision']}",
            f"--seed={int(self.config['seed'])}",
            f"--save_every_n_epochs={int(self.config['save_every_n_epochs'])}",
            "--gradient_checkpointing",
            "--cache_latents",
            "--cache_latents_to_disk",
            "--cache_text_encoder_outputs",
            "--cache_text_encoder_outputs_to_disk",
            "--network_train_unet_only",
            "--sdpa",
            "--no_half_vae",
            "--max_data_loader_n_workers=2",
            "--persistent_data_loader_workers",
        ]
        return PreparedRun(
            job_path, run_root, dataset_config, output_dir, output_name, command, len(items)
        )

    @staticmethod
    def _write_job(path: Path, job: dict) -> None:
        path.write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")

    def run(self, job_path: Path, dry_run: bool = False) -> PreparedRun:
        prepared = self.prepare(job_path)
        if dry_run:
            return prepared
        job = json.loads(job_path.read_text(encoding="utf-8"))
        job.update(
            version=prepared.output_name,
            image_count=prepared.image_count,
            dataset_config=prepared.dataset_config.as_posix(),
            output_dir=prepared.output_dir.as_posix(),
            command=prepared.command,
            status="running",
            updated_at=datetime.now(UTC).isoformat(),
        )
        self._write_job(job_path, job)
        environment = os.environ.copy()
        environment["HF_HOME"] = str(self.settings.data_root / "cache" / "huggingface")
        environment["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
        log_path = prepared.run_root / "training.log"
        try:
            with log_path.open("w", encoding="utf-8") as log:
                subprocess.run(
                    prepared.command,
                    cwd=self.sd_scripts,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            job.update(status="completed", finished_at=datetime.now(UTC).isoformat())
        except Exception as exc:
            job.update(status="failed", error=str(exc), finished_at=datetime.now(UTC).isoformat())
            raise
        finally:
            self._write_job(job_path, job)
        return prepared

    def next_job(self) -> Path | None:
        for path in sorted(self.settings.data_root.joinpath("queues").glob("*.json")):
            job = json.loads(path.read_text(encoding="utf-8"))
            if job.get("status") == "queued":
                return path
        return None
