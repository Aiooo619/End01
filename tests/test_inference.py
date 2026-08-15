from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from stylebot.config import Settings, StyleConfig
from stylebot.inference import InferenceRunner
from stylebot.registry import ModelRecord


class InferenceRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        style = StyleConfig("design", "設計", 10, "design_style", 10, "design")
        self.settings = Settings(
            project_root=root,
            data_root=root,
            bot_token="test",
            guild_id=1,
            forum_channel_id=2,
            allowed_user_ids=frozenset({3}),
            max_attachment_mb=25,
            styles={"design": style},
        )
        python = root / "work" / "sd-scripts" / "venv" / "Scripts" / "python.exe"
        python.parent.mkdir(parents=True)
        python.touch()
        model_path = root / "models" / "design" / "v001" / "model.safetensors"
        model_path.parent.mkdir(parents=True)
        model_path.touch()
        self.base_path = root / "models" / "base" / "sdxl-base-1.0"
        self.base_path.mkdir(parents=True)
        (self.base_path / "model_index.json").write_text("{}", encoding="utf-8")
        self.model = ModelRecord(
            "design:v001:final", "design", "v001", "final", "design",
            "production", model_path.relative_to(root).as_posix(), "job-1"
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_builds_reproducible_generation_request(self) -> None:
        captured: dict = {}

        def fake_run(command, **kwargs):
            request_path = Path(command[-1])
            request = json.loads(request_path.read_text(encoding="utf-8"))
            captured.update(request)
            output = Path(request["output_path"])
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(b"png")
            return subprocess.CompletedProcess(command, 0, "ok", "")

        with patch("stylebot.inference.subprocess.run", side_effect=fake_run):
            result = InferenceRunner(self.settings).generate(
                [(self.model, 0.8)], "design_style, full body", seed=123
            )
        self.assertTrue(result.image_path.exists())
        self.assertEqual(result.seed, 123)
        self.assertEqual(captured["adapters"][0]["strength"], 0.8)
        self.assertEqual(captured["prompt"], "design_style, full body")
        self.assertEqual(captured["base_model"], self.base_path.as_posix())


if __name__ == "__main__":
    unittest.main()
