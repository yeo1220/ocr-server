"""Tests for deterministic OCR rule fixes."""

import unittest

from ocr_rules import apply_rules_to_table, fix_cell_text


class OcrRulesTests(unittest.TestCase):
    def test_fix_hangul_typos(self):
        fixed, changed = fix_cell_text("소제지 화성시", "소재지")
        self.assertTrue(changed)
        self.assertIn("소재지", fixed)

    def test_fix_numeric_o_to_zero(self):
        fixed, changed = fix_cell_text("1O,000", "금액")
        self.assertTrue(changed)
        self.assertIn("10,000", fixed)

    def test_apply_rules_to_table(self):
        table = {
            "cols": 2,
            "header_rows": 1,
            "headers": ["소재지", "금액"],
            "data": [["소제지", "5OO"]],
            "cells": [
                {"row": 1, "col": 0, "text": "소제지", "score": 0.9},
                {"row": 1, "col": 1, "text": "5OO", "score": 0.7},
            ],
        }
        out, n = apply_rules_to_table(table)
        self.assertGreaterEqual(n, 1)
        self.assertEqual(out["data"][0][0], "소재지")


if __name__ == "__main__":
    unittest.main()
