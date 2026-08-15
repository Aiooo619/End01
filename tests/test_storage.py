from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from stylebot.config import Settings, StyleConfig
from stylebot.storage import DatasetStore, IngestError
from stylebot.trainer import TrainingWorker


class DatasetStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.style = StyleConfig("film", "底片", None, "film_style", 1)
        settings = Settings(
            project_root=self.root,
            data_root=self.root,
            bot_token="test",
            guild_id=None,
            forum_channel_id=None,
            allowed_user_ids=frozenset(),
            max_attachment_mb=25,
            styles={"film": self.style},
        )
        self.store = DatasetStore(settings)

    def tearDown(self) -> None:
        self.temp.cleanup()

    @staticmethod
    def image_bytes(
        size: tuple[int, int] = (1024, 1024), color: str = "#cb9b73"
    ) -> bytes:
        buffer = io.BytesIO()
        Image.new("RGB", size, color).save(buffer, "PNG")
        return buffer.getvalue()

    def test_ingest_duplicate_approve_and_queue(self) -> None:
        payload = self.image_bytes()
        first = self.store.ingest(self.style, payload, "my photo.png", "m1", "u1")
        duplicate = self.store.ingest(self.style, payload, "copy.png", "m2", "u1")
        self.assertEqual(first.status, "received")
        self.assertEqual(duplicate.status, "duplicate")
        self.assertEqual(self.store.status("film")["incoming"], 1)
        self.assertEqual(self.store.approve("film"), 1)
        self.assertEqual(self.store.status("film")["approved"], 1)
        self.assertEqual(len(list((self.root / "queues" / "captions").glob("*.json"))), 1)
        self.assertEqual(self.store.queue_missing_captions(), 0)
        caption_path = self.root / "datasets" / "film" / "captions" / f"{Path(first.filename).stem}.txt"
        caption_path.write_text("film_style, solo, full body\n", encoding="utf-8")
        job = self.store.queue_training(self.style)
        self.assertTrue(job.exists())
        scripts_root = self.root / "work" / "sd-scripts"
        (scripts_root / "venv" / "Scripts").mkdir(parents=True)
        (scripts_root / "venv" / "Scripts" / "python.exe").touch()
        (scripts_root / "sdxl_train_network.py").touch()
        prepared = TrainingWorker(self.store.settings).prepare(job)
        self.assertEqual(prepared.image_count, 1)
        self.assertTrue(prepared.dataset_config.exists())
        self.assertIn("--network_train_unet_only", prepared.command)

    def test_pending_and_reject_by_message(self) -> None:
        self.store.ingest(
            self.style, self.image_bytes(color="#000000"), "black.png", "reject-me", "u1"
        )
        self.store.ingest(
            self.style, self.image_bytes(color="#ffffff"), "white.png", "keep-me", "u1"
        )
        self.assertEqual(len(self.store.pending("film")), 2)
        self.assertEqual(self.store.reject("film", message_id="reject-me"), 1)
        self.assertEqual(self.store.status("film")["rejected"], 1)
        self.assertEqual(self.store.status("film")["incoming"], 1)

    def test_rejects_small_image(self) -> None:
        with self.assertRaises(IngestError):
            self.store.ingest(self.style, self.image_bytes((256, 256)), "small.png", "m1", "u1")


if __name__ == "__main__":
    unittest.main()
