"""Page chrome vs real table header detection."""

import unittest

from table_structure import (
    apply_rule_structure_fix,
    canonical_headers,
    detect_structure_issues,
    find_table_region_start,
    is_collapsed_table_row,
    is_invalid_headers,
    normalize_table_headers,
    row_has_embedded_header_labels,
)


class TableRegionTests(unittest.TestCase):
    def test_invalid_page_chrome_headers(self):
        headers = [
            "",
            "[별지제1목록]인덕원-동탄 복선전철 건설사업",
            "",
            "",
            "보 상 금 내역(구분지상권)",
            "",
            "",
            "",
            "Page 4/35",
        ]
        self.assertTrue(is_invalid_headers(headers))

    def test_normalize_headers_9_cols(self):
        table = {
            "cols": 9,
            "header_rows": 1,
            "headers": ["", "별지", "", "", "Page 2/10", "", "", "", ""],
            "data": [["경기", "", "", "1000", "", "", "", "", ""]],
            "all_rows": [],
            "cells": [],
        }
        fixed = normalize_table_headers(table)
        self.assertEqual(fixed["headers"], canonical_headers(9))

    def test_find_table_region_skips_title(self):
        rows = [
            ["[별지제1목록]인덕원동탄", "", "보상금내역", "", "Page 4/35"],
            ["소재지", "지번", "구조 및 규격", "단가", "금액", "성명", "권리", "주소", ""],
            ["소재지", "", "면적", "", "", "", "", "", ""],
            ["경기도", "138", "388", "1000", "", "", "", "", ""],
        ]
        start = find_table_region_start(rows, 2)
        self.assertEqual(start, 1)

    def test_collapsed_row_dropped_before_strip(self):
        row = [
            "소재지 경기도수원시팔달구",
            "지번 구분지상권 138",
            "수량 면적 388m2",
            "단가/ 금액 436,995 540,860",
            "성명 김미영(101-1703) 김민정(상가-101)",
            "주소 충청남도",
            "성명/",
            "권리의종류",
            "주소",
        ]
        headers = canonical_headers(9)
        self.assertTrue(is_collapsed_table_row(row, headers, 9))
        table = {
            "cols": 9,
            "header_rows": 1,
            "headers": headers,
            "data": [row, ["경기", "138", "1", "", "", "", "", "", ""]],
            "all_rows": [headers, row],
            "cells": [],
        }
        fixed, meta = apply_rule_structure_fix(table)
        self.assertEqual(meta["dropped_collapsed_rows"], 1)
        self.assertEqual(len(fixed["data"]), 1)

    def test_detect_issues_with_chrome_headers(self):
        table = {
            "cols": 9,
            "headers": ["Page 1/10", "", "", "", "", "", "", "", ""],
            "data": [["소재지 경기", "지번 138", "", "", "", "", "", "", ""]],
        }
        issues = detect_structure_issues(table)
        self.assertTrue(issues["invalid_headers"])
        self.assertTrue(issues["needs_fix"])

    def test_embedded_labels_detected(self):
        self.assertTrue(
            row_has_embedded_header_labels(
                ["소재지 경기도", "지번 138", "", "", "", "", "", "", ""]
            )
        )


if __name__ == "__main__":
    unittest.main()
