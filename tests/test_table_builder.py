"""Tests for fixed-column table reconstruction from OCR blocks."""

import unittest

from table_builder import build_table, export_table_aliases, table_to_text


def _block(text: str, x0: float, y0: float, x1: float, y1: float, score: float = 0.95):
    box = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
    return {"text": text, "score": score, "box": box}


class TableBuilderTests(unittest.TestCase):
    def test_build_table_fixed_columns(self):
        blocks = [
            _block("품목", 10, 10, 90, 30),
            _block("수량", 110, 10, 190, 30),
            _block("금액", 210, 10, 290, 30),
            _block("볼트", 10, 50, 90, 70),
            _block("1O", 110, 50, 190, 70, score=0.72),
            _block("5OO", 210, 50, 290, 70, score=0.68),
            _block("너트", 10, 90, 90, 110),
            _block("20", 110, 90, 190, 110),
            _block("600", 210, 90, 290, 110),
        ]
        table = build_table(blocks, num_cols=3, header_row=0)

        self.assertEqual(table["cols"], 3)
        self.assertEqual(table["headers"], ["품목", "수량", "금액"])
        self.assertEqual(len(table["data"]), 2)
        self.assertEqual(table["data"][0], ["볼트", "1O", "5OO"])
        self.assertEqual(table["data"][1], ["너트", "20", "600"])
        self.assertEqual(len(table["cells"]), 9)

    def test_table_to_text(self):
        table = {
            "headers": ["A", "B"],
            "data": [["1", "2"], ["3", "4"]],
        }
        self.assertEqual(table_to_text(table), "A\tB\n1\t2\n3\t4")

    def test_col_boundaries_override(self):
        blocks = [
            _block("H1", 0, 0, 50, 20),
            _block("H2", 100, 0, 150, 20),
            _block("a", 0, 40, 50, 60),
            _block("b", 100, 40, 150, 60),
        ]
        bounds = [0, 75, 200]
        table = build_table(
            blocks, num_cols=2, header_row=0, col_boundaries=bounds
        )
        self.assertEqual(table["headers"], ["H1", "H2"])
        self.assertEqual(table["data"], [["a", "b"]])

    def test_two_row_headers(self):
        """재결서 2단 헤더: 상단·하단 행을 열별로 병합한다."""
        blocks = [
            _block("구분지상권", 0, 0, 80, 18),
            _block("보상액", 200, 0, 280, 18),
            _block("소재지", 0, 22, 80, 40),
            _block("단가", 200, 22, 280, 40),
            _block("경기도", 0, 50, 80, 68),
            _block("1000", 200, 50, 280, 68),
        ]
        table = build_table(blocks, num_cols=2, header_rows=2)
        self.assertEqual(table["header_rows"], 2)
        self.assertEqual(len(table["data"]), 1)
        self.assertEqual(table["headers"][0], "구분지상권 소재지")
        self.assertEqual(table["headers"][1], "보상액 단가")
        self.assertEqual(len(table["all_rows"]), 3)

    def test_export_table_aliases(self):
        table = build_table(
            [
                _block("A", 0, 0, 40, 20),
                _block("B", 60, 0, 100, 20),
                _block("1", 0, 40, 40, 60),
                _block("2", 60, 40, 100, 60),
            ],
            num_cols=2,
            header_rows=1,
        )
        table["data_refined"] = [["1x", "2x"]]
        export_table_aliases(table)
        self.assertEqual(len(table["rows"]), 2)
        self.assertEqual(table["rows_refined"], [["A", "B"], ["1x", "2x"]])


if __name__ == "__main__":
    unittest.main()
