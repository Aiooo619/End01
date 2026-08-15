from __future__ import annotations

import unittest

from stylebot.arknights_curator import normalized


class ArknightsCuratorTests(unittest.TestCase):
    def test_normalizes_diacritics_and_punctuation(self) -> None:
        self.assertEqual(normalized("Pozëmka"), "pozemka")
        self.assertEqual(normalized("Kal'tsit"), "kaltsit")
        self.assertEqual(normalized("Młynar"), "mlynar")
        self.assertEqual(normalized("丰川祥子"), "丰川祥子")


if __name__ == "__main__":
    unittest.main()
