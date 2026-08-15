from __future__ import annotations

import json
import time
import urllib.request
import uuid

from .bot import ITERATION_NEGATIVE
from .config import load_settings
from .inference import InferenceRunner
from .registry import ModelRegistry


def upload_message(token: str, channel_id: int, content: str, image_path) -> str:
    boundary = f"----End01{uuid.uuid4().hex}"
    payload = json.dumps({"content": content}, ensure_ascii=False).encode("utf-8")
    image = image_path.read_bytes()
    body = b"".join(
        [
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"payload_json\"\r\n"
            "Content-Type: application/json\r\n\r\n".encode(), payload, b"\r\n",
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"files[0]\"; "
            f"filename=\"{image_path.name}\"\r\nContent-Type: image/png\r\n\r\n".encode(),
            image, b"\r\n", f"--{boundary}--\r\n".encode(),
        ]
    )
    request = urllib.request.Request(
        f"https://discord.com/api/v10/channels/{channel_id}/messages",
        data=body,
        headers={
            "Authorization": f"Bot {token}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "User-Agent": "End01-evaluator",
        },
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))["id"]


def main() -> int:
    settings = load_settings()
    registry = ModelRegistry(settings)
    runner = InferenceRunner(settings)
    models = registry.list_models("arknights_portrait")
    by_checkpoint = {model.checkpoint: model for model in models if model.version == "v003"}
    candidates = [by_checkpoint[name] for name in ("epoch-002", "epoch-004", "epoch-006", "final")]
    prompt = (
        "arknights_portrait_style, 1girl, solo, full body, standing, alpine signal engineer, "
        "insulated asymmetrical coat, layered functional workwear, portable antenna tools, "
        "navy and orange accents, simple background, clean anime game illustration"
    )
    style = settings.styles["arknights_portrait"]
    session_id = registry.create_comparison(style.style_id, "v003", prompt, ITERATION_NEGATIVE, 42)
    for index, model in enumerate(candidates, start=1):
        result = runner.generate([(model, 0.5)], prompt, ITERATION_NEGATIVE, 42, 768, 1024, "comparison")
        generation_id = registry.record_generation(
            model.model_id, prompt, ITERATION_NEGATIVE, 42, 0.5, result.image_path, "comparison"
        )
        registry.add_comparison_candidate(session_id, generation_id, model.model_id, 0.5)
        content = (
                f"v003 候選 {index}/4 · `{model.checkpoint}` · strength `0.5` · seed `42`\n"
                f"比較組 `{session_id}` · generation `{generation_id}`\n"
                "請用右鍵／長按 → Apps → Select review image 選最佳圖；也可用 /feedback 記錄問題。"
        )
        message_id = upload_message(settings.bot_token, style.discord_channel_id, content, result.image_path)
        registry.attach_message(generation_id, message_id)
        time.sleep(1)
    print(json.dumps({"session_id": session_id, "generated": len(candidates)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
