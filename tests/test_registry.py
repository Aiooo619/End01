from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from stylebot.config import Settings, StyleConfig
from stylebot.registry import ModelRegistry
from stylebot.trainer import TrainingWorker


class ModelRegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        style = StyleConfig("outfit", "服飾", 123, "outfit_style", 10, "design")
        self.settings = Settings(
            project_root=root,
            data_root=root,
            bot_token="test",
            guild_id=1,
            forum_channel_id=2,
            allowed_user_ids=frozenset({3}),
            max_attachment_mb=25,
            styles={"outfit": style},
        )
        self.registry = ModelRegistry(self.settings)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_register_promote_generate_select_and_feedback(self) -> None:
        model_path = self.settings.data_root / "models" / "outfit" / "v001" / "model.safetensors"
        model_path.parent.mkdir(parents=True)
        model_path.write_bytes(b"test")
        model = self.registry.register_model(
            "outfit", "v001", "epoch-01", model_path, "job-1"
        )
        self.assertEqual(model.model_type, "design")
        production = self.registry.promote(model.model_id)
        self.assertEqual(production.status, "production")
        self.assertEqual(len(self.registry.list_models(production_only=True)), 1)

        image = self.settings.data_root / "outputs" / "generated" / "result.png"
        image.parent.mkdir(parents=True)
        image.write_bytes(b"image")
        generation_id = self.registry.record_generation(
            model.model_id, "prompt", "bad", 42, 0.8, image, "review"
        )
        self.registry.attach_message(generation_id, "message-1")
        self.assertEqual(self.registry.select_by_message("message-1"), generation_id)
        feedback_id = self.registry.add_feedback(
            model.model_id, "user-1", "clothing", "增加配件", generation_id
        )
        self.assertTrue(feedback_id)
        iteration_id = self.registry.create_iteration(model.model_id, "增加服飾層次")
        self.assertTrue(iteration_id.startswith("iter-"))

    def test_register_existing_job_outputs(self) -> None:
        output_dir = self.settings.data_root / "models" / "outfit" / "v001"
        output_dir.mkdir(parents=True)
        (output_dir / "outfit_v001-000001.safetensors").write_bytes(b"epoch")
        (output_dir / "outfit_v001.safetensors").write_bytes(b"final")
        job_path = self.settings.data_root / "queues" / "job-1.json"
        job_path.parent.mkdir(parents=True)
        job_path.write_text(
            '{"job_id":"job-1","style_id":"outfit","version":"outfit_v001",'
            f'"output_dir":"{output_dir.as_posix()}"}}',
            encoding="utf-8",
        )

        count = TrainingWorker(self.settings).register_job_outputs(job_path)

        self.assertEqual(count, 2)
        checkpoints = {model.checkpoint for model in self.registry.list_models("outfit")}
        self.assertEqual(checkpoints, {"epoch-001", "final"})

    def test_comparison_selection_tags_and_report(self) -> None:
        model_path = self.settings.data_root / "models" / "outfit" / "v001" / "model.safetensors"
        model_path.parent.mkdir(parents=True)
        model_path.write_bytes(b"model")
        model = self.registry.register_model("outfit", "v001", "epoch-006", model_path, "job-1")
        image = self.settings.data_root / "outputs" / "generated" / "candidate.png"
        image.parent.mkdir(parents=True)
        image.write_bytes(b"image")
        generation_id = self.registry.record_generation(
            model.model_id, "prompt", "negative", 42, 0.55, image, "comparison"
        )
        session_id = self.registry.create_comparison(
            "outfit", "v001", "prompt", "negative", 42
        )
        self.registry.add_comparison_candidate(
            session_id, generation_id, model.model_id, 0.55
        )
        self.registry.choose_candidate(session_id, generation_id)
        self.registry.tag_candidate(
            session_id, generation_id, "user-1", "extra_limbs"
        )

        report = self.registry.comparison_report("outfit")

        self.assertEqual(report["sessions"], 1)
        self.assertEqual(report["candidates"][0]["wins"], 1)
        self.assertEqual(report["tags"][0]["tag"], "extra_limbs")


if __name__ == "__main__":
    unittest.main()
