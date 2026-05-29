"""Tests for table structure cleanup (header repeat, row merge)."""

import unittest

from table_builder import build_table
from table_structure import (
    apply_rule_structure_fix,
    detect_structure_issues,
    drop_repeated_header_rows,
    is_duplicate_header_row,
    merge_wrapped_data_rows,
)


def _block(text: str, x0: float, y0: float, x1: float, y1: float, score: float = 0.95):
    box = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
    return {"text": text, "score": score, "box": box}


class TableStructureTests(unittest.TestCase):
    def test_duplicate_header_row_detected(self):
        headers = ["소재지", "구조 및 규격", "금액", "소유자"]
        row = ["소재지", "구조 및 규격", "금액", "소유자"]
        self.assertTrue(is_duplicate_header_row(row, headers))

    def test_drop_repeated_headers(self):
        headers = ["A", "B", "C"]
        data = [
            ["x1", "y1", "z1"],
            ["A", "B", "C"],
            ["x2", "y2", "z2"],
        ]
        cleaned, n = drop_repeated_header_rows(data, headers)
        self.assertEqual(n, 1)
        self.assertEqual(len(cleaned), 2)
        self.assertEqual(cleaned[0][0], "x1")

    def test_merge_wrapped_data_rows(self):
        data = [
            ["경기", "긴규격첫줄", "100", "홍길동"],
            ["", "긴규격둘째줄", "", ""],
        ]
        merged = merge_wrapped_data_rows(data, 4)
        self.assertEqual(len(merged), 1)
        self.assertIn("긴규격첫줄", merged[0][1])
        self.assertIn("긴규격둘째줄", merged[0][1])

    def test_build_table_strips_mid_header(self):
        bounds = [0, 100, 200, 300, 400]
        blocks = [
            _block("소재지", 10, 5, 90, 25),
            _block("금액", 210, 5, 290, 25),
            _block("경기", 10, 40, 90, 60),
            _block("1000", 210, 40, 290, 60),
            _block("소재지", 10, 85, 90, 105),
            _block("금액", 210, 85, 290, 105),
            _block("서울", 10, 120, 90, 140),
            _block("2000", 210, 120, 290, 140),
        ]
        table = build_table(
            blocks, num_cols=2, header_rows=1, col_boundaries=bounds
        )
        self.assertEqual(len(table["data"]), 2)
        self.assertEqual(table["data"][0][0], "경기")
        self.assertEqual(table["data"][1][0], "서울")

    def test_detect_structure_issues(self):
        table = {
            "cols": 2,
            "headers": ["A", "B"],
            "data": [["A", "B"], ["1", "2"]],
        }
        issues = detect_structure_issues(table)
        self.assertTrue(issues["needs_fix"])
        self.assertGreaterEqual(issues["duplicate_header_rows"], 1)

    def test_apply_rule_structure_fix(self):
        table = {
            "cols": 2,
            "header_rows": 1,
            "headers": ["A", "B"],
            "data": [["A", "B"], ["1", "2"]],
            "all_rows": [["A", "B"], ["A", "B"], ["1", "2"]],
            "cells": [
                {"row": 1, "col": 0, "text": "A", "score": 0.9},
                {"row": 1, "col": 1, "text": "B", "score": 0.9},
                {"row": 2, "col": 0, "text": "1", "score": 0.9},
                {"row": 2, "col": 1, "text": "2", "score": 0.9},
            ],
        }
        fixed, meta = apply_rule_structure_fix(table)
        self.assertEqual(len(fixed["data"]), 1)
        self.assertEqual(meta["dropped_header_rows"], 1)


if __name__ == "__main__":
    unittest.main()
