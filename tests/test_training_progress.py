from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from stylebot.bot import StyleBot


class TrainingProgressTests(unittest.TestCase):
    def test_parses_latest_step_and_builds_bar(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            log = Path(root) / "training.log"
            log.write_text(
                "steps:  25%|xx| 100/400 [01:00<03:00, 1.2it/s, avr_loss=0.123]\r"
                "steps:  50%|xx| 200/400 [02:00<02:00, 1.2it/s, avr_loss=0.087]\r",
                encoding="utf-8",
            )
            progress = StyleBot.training_progress(log)
        self.assertIsNotNone(progress)
        step, total, percent, eta, loss = progress
        self.assertEqual((step, total), (200, 400))
        self.assertEqual(percent, 50)
        self.assertEqual(eta, "02:00")
        self.assertEqual(loss, "0.087")
        self.assertEqual(StyleBot.progress_bar(percent), "██████████░░░░░░░░░░")


if __name__ == "__main__":
    unittest.main()
