from __future__ import annotations

import json
import unittest
from pathlib import Path

from stylebot.bot import estimated_clip_tokens


class V1ConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(__file__).resolve().parents[1]

    def test_evaluation_prompts_fit_clip_budget(self) -> None:
        config = json.loads(
            (self.root / "config" / "evaluation_prompts.json").read_text(encoding="utf-8")
        )
        self.assertGreaterEqual(len(config["prompts"]), 4)
        for item in config["prompts"]:
            prompt = item["prompt"].replace("{trigger}", "game_style")
            self.assertLessEqual(estimated_clip_tokens(prompt), 70, item["id"])

    def test_comfy_workflow_uses_core_nodes_and_batch_four(self) -> None:
        workflow = json.loads(
            (self.root / "comfyui" / "workflows" / "CharacterDesignGenerator.json")
            .read_text(encoding="utf-8")
        )
        classes = {node["class_type"] for node in workflow.values()}
        self.assertTrue({"CheckpointLoaderSimple", "LoraLoader", "KSampler", "SaveImage"} <= classes)
        latent = next(node for node in workflow.values() if node["class_type"] == "EmptyLatentImage")
        self.assertEqual(latent["inputs"]["batch_size"], 4)

    def test_illustrious_workflows_do_not_attach_legacy_lora(self) -> None:
        workflows = self.root / "comfyui" / "workflows"
        generator_text = (workflows / "CharacterDesignGenerator.json").read_text(encoding="utf-8")
        baseline = json.loads((workflows / "IllustriousBaseline.json").read_text(encoding="utf-8"))

        self.assertNotIn("arknights_portrait_v001", generator_text)
        self.assertNotIn("arknights_portrait_v002", generator_text)
        checkpoint = next(
            node for node in baseline.values() if node["class_type"] == "CheckpointLoaderSimple"
        )
        self.assertEqual(
            checkpoint["inputs"]["ckpt_name"], "Illustrious-XL-v2.0.safetensors"
        )


if __name__ == "__main__":
    unittest.main()
