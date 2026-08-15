from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from PIL import Image

from .config import Settings


LOGGER = logging.getLogger(__name__)
MODEL_REPO = "SmilingWolf/wd-vit-tagger-v3"
GENERAL_THRESHOLD = 0.35
DESIGN_ANCHOR = "character costume design"
DESIGN_WORDS = {
    "armor", "belt", "blazer", "boots", "bow", "bracelet", "cape", "cloak",
    "coat", "collar", "dress", "earrings", "eyewear", "fedora", "footwear",
    "gloves", "goggles", "hat", "headwear", "hood", "jacket", "jewelry",
    "pants", "ribbon", "scarf", "shirt", "shoes", "shorts", "skirt", "sleeves",
    "socks", "stockings", "suit", "sweater", "tie", "uniform", "vest",
}
ART_WORDS = {
    "3d", "cel shading", "chibi", "comic", "gradient", "lineart", "monochrome",
    "painting", "pixel art", "realistic", "render", "sketch", "watercolor",
}
KAOMOJI = {
    "0_0", "(o)_(o)", "+_+", "+_-", "._.", "<o>_<o>", "<|>_<|>",
    "=_=", ">_<", "3_3", "6_9", ">_o", "@_@", "^_^", "o_o", "u_u",
    "x_x", "|_|", "||_||",
}


@dataclass(frozen=True)
class Tag:
    name: str
    category: int


class WDTagger:
    def __init__(self, cache_dir: Path):
        cache_dir.mkdir(parents=True, exist_ok=True)
        model_path = hf_hub_download(MODEL_REPO, "model.onnx", cache_dir=cache_dir.as_posix())
        labels_path = hf_hub_download(
            MODEL_REPO, "selected_tags.csv", cache_dir=cache_dir.as_posix()
        )
        available = ort.get_available_providers()
        providers = [
            name for name in ("CUDAExecutionProvider", "CPUExecutionProvider")
            if name in available
        ]
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input = self.session.get_inputs()[0]
        self.output_name = self.session.get_outputs()[0].name
        _, self.height, self.width, _ = self.input.shape
        with open(labels_path, "r", encoding="utf-8", newline="") as handle:
            self.tags = [
                Tag(row["name"], int(row["category"])) for row in csv.DictReader(handle)
            ]
        LOGGER.info("Caption model providers: %s", self.session.get_providers())

    def _prepare(self, image_path: Path) -> np.ndarray:
        with Image.open(image_path) as source:
            image = source.convert("RGBA")
            canvas = Image.new("RGBA", image.size, (255, 255, 255, 255))
            canvas.alpha_composite(image)
            rgb = canvas.convert("RGB")
        max_dim = max(rgb.size)
        square = Image.new("RGB", (max_dim, max_dim), "white")
        square.paste(rgb, ((max_dim - rgb.width) // 2, (max_dim - rgb.height) // 2))
        square = square.resize((self.width, self.height), Image.Resampling.BICUBIC)
        array = np.asarray(square, dtype=np.float32)[:, :, ::-1]
        return np.expand_dims(array, axis=0)

    @staticmethod
    def _display_name(name: str) -> str:
        return name if name in KAOMOJI else name.replace("_", " ")

    def caption(self, image_path: Path, trigger_token: str) -> str:
        probabilities = self.session.run(
            [self.output_name], {self.input.name: self._prepare(image_path)}
        )[0][0]
        selected: list[tuple[str, float]] = []
        for tag, score in zip(self.tags, probabilities, strict=True):
            score = float(score)
            if tag.category == 0 and score >= GENERAL_THRESHOLD:
                selected.append((self._display_name(tag.name), score))

        def priority(item: tuple[str, float]) -> tuple[int, float]:
            name, score = item
            words = set(name.split())
            if words & DESIGN_WORDS or name in DESIGN_WORDS:
                group = 0
            elif words & ART_WORDS or name in ART_WORDS:
                group = 3
            elif any(term in name for term in ("background", "looking", "standing", "sitting", "view")):
                group = 2
            else:
                group = 1
            return group, -score

        selected.sort(key=priority)
        return ", ".join(
            [trigger_token, DESIGN_ANCHOR, *(name for name, _ in selected[:60])]
        )


def process_caption_jobs(settings: Settings, limit: int = 100) -> tuple[int, int]:
    queue_root = settings.data_root / "queues" / "captions"
    queue_root.mkdir(parents=True, exist_ok=True)
    queued: list[tuple[Path, dict]] = []
    for path in sorted(queue_root.glob("caption-*.json"))[:limit]:
        job = json.loads(path.read_text(encoding="utf-8"))
        if job.get("status") == "queued":
            queued.append((path, job))
    if not queued:
        return 0, 0

    tagger = WDTagger(settings.data_root / "cache" / "wd-tagger")
    completed = failed = 0
    for path, job in queued:
        try:
            style = settings.styles[job["style_id"]]
            image_path = settings.data_root / job["image_path"]
            caption = tagger.caption(image_path, style.trigger_token)
            caption_dir = settings.data_root / "datasets" / style.style_id / "captions"
            caption_dir.mkdir(parents=True, exist_ok=True)
            caption_path = caption_dir / f"{image_path.stem}.txt"
            caption_path.write_text(caption + "\n", encoding="utf-8")
            job.update(
                status="completed",
                caption_path=caption_path.relative_to(settings.data_root).as_posix(),
            )
            completed += 1
        except Exception as exc:
            LOGGER.exception("Caption job failed: %s", path.name)
            job.update(status="failed", error=str(exc))
            failed += 1
        path.write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")
    return completed, failed
