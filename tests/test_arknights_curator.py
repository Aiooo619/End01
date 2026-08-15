from __future__ import annotations

import unittest

from stylebot.arknights_curator import design_draft, normalized


class ArknightsCuratorTests(unittest.TestCase):
    def test_normalizes_diacritics_and_punctuation(self) -> None:
        self.assertEqual(normalized("Pozëmka"), "pozemka")
        self.assertEqual(normalized("Kal'tsit"), "kaltsit")
        self.assertEqual(normalized("Młynar"), "mlynar")
        self.assertEqual(normalized("丰川祥子"), "丰川祥子")

    def test_builds_design_draft_from_tags(self) -> None:
        draft = design_draft(["long coat", "collared shirt", "black pants", "boots", "belt", "asymmetrical sleeves"])
        self.assertIn("外層", draft["structure"])
        self.assertIn("belt", draft["materials"])
        self.assertIn("asymmetrical sleeves", draft["design_points"])


if __name__ == "__main__":
    unittest.main()
