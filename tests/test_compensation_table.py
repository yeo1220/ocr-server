"""Regression tests for 재결서-style merged / fragmented OCR rows."""

import unittest

from table_structure import (
    apply_rule_structure_fix,
    is_collapsed_table_row,
    is_fragment_continuation_row,
    split_multivalue_rows,
    strip_cell_header_pollution,
)


class CompensationTableTests(unittest.TestCase):
    def test_strip_embedded_header_labels(self):
        cell = "소재지 경기도수원시팔달구 우만 동 경기도수원시팔달구우만 동"
        out = strip_cell_header_pollution(cell, 0, ["소재지", "지번"])
        self.assertFalse(out.startswith("소재지"))
        self.assertIn("경기도", out)

    def test_fragment_dong_merges(self):
        above = ["경기도수원시팔달구우만", "구분지상권 138", "388m2", "", "", "", "", "", ""]
        below = ["동", "1387", "대 x35.713", "540,860", "김병국(101-404)", "", "", "", ""]
        self.assertTrue(is_fragment_continuation_row(above, below, 9))

    def test_collapsed_first_row(self):
        row = [
            "소재지 경기도수원시팔달구 우만 동",
            "지번 구분지상권 138 1387",
            "수량 물건의 종류 면적 388m2",
            "단가/ 금액 436,995 540,860",
            "성명 김미영(101-1703) 김민정(상가-101)",
            "주소 충청남도천안시",
            "",
            "",
            "",
        ]
        headers = ["소재지", "지번", "구조", "금액", "성명", "주소", "a", "b", "c"]
        self.assertTrue(is_collapsed_table_row(row, headers, 9), "fused header+multi-owner row")

    def test_split_multi_owner_row(self):
        row = [
            "경기도수원시팔달구우만 동",
            "1387 구분지상권 138",
            "388m2 대",
            "540,860 436,995 757,460",
            "",
            "주소1 주소2",
            "김병선(102-2102) 김상중(103-602)",
            "",
            "",
        ]
        parts = split_multivalue_rows([row], 9)
        self.assertGreaterEqual(len(parts), 2)
        owners = sum(
            1
            for p in parts
            for c in p
            if "김병선" in c or "김상중" in c
        )
        self.assertGreaterEqual(owners, 1)

    def test_apply_fix_on_sample_fragment(self):
        headers = ["소재지", "지번", "구조", "금액", "비고", "주소", "성명", "권리", "주소2"]
        data = [
            ["경기도수원시팔달구우만", "구분지상권 138", "388m2", "436,995", "", "addr1", "", "", ""],
            ["동", "1387", "대 x35", "540,860", "", "addr1b", "김병국(101-404)", "", ""],
        ]
        table = {
            "cols": 9,
            "header_rows": 1,
            "headers": headers,
            "data": data,
            "all_rows": [headers] + data,
            "cells": [],
        }
        fixed, meta = apply_rule_structure_fix(table)
        self.assertEqual(len(fixed["data"]), 1, fixed["data"])
        merged_addr = fixed["data"][0][0].replace(" ", "")
        self.assertIn("동", merged_addr)
        self.assertIn("우만", merged_addr)
        self.assertGreater(meta["merged_wrap_rows"], 0)


    def test_reassemble_first_owner_record(self):
        """OCR-shattered rows → one 9-col compensation record."""
        from table_structure import reassemble_fragmented_records

        rows = [
            [
                "경기도수원시팔 달구 우만 동",
                "구분지상권 1 38 1387",
                "평균해수면(+ 5 100): 해수면(+100) 122.46m ~101.97m 388m2 대 x82.890/11 195. 지하심도:26.47,",
                "436,995 540,860",
                "김미영(10 1-1703",
                "충청남도천안시동남구동면 수남2길21 (등)경기도수원시팔달구중부대로 223번길 27,102동1401호(후만동,선경아파트) 경기도용인시",
                "주식회사 국민은행",
                "도통) [태전",
                "서울특별시영등포구 국제금융로8길26",
            ],
            [
                "경기도수원시 팔달구우만 동",
                "구분지상권 138 138",
                "평균해수면(+ 5 100): 해수면(+100)122.46m ~101.97m 388m' 대 x35.713/11195. 지하심도: 26.47,",
                "436,995 1,255,350",
                ") 김민정( 상가-101",
                "기흥구기흥역로63기흥역힐 스테이트202동1401호 (등)경기도용인 시기흥구기흥역로63,202 동1401호(구갈동,기흥역힐스테이트)",
                "(101 - 40",
                "동지점 ] 근",
                "(여의 (등)서울 특별시,영등포구",
            ],
        ]
        out, groups = reassemble_fragmented_records(rows, 9)
        self.assertGreaterEqual(groups, 1)
        first = out[0]
        self.assertIn("경기도", first[0])
        self.assertIn("우만", first[0])
        self.assertIn("구분지상권", first[1])
        self.assertIn("해수면", first[2])
        self.assertIn("436,995", first[3])
        self.assertIn("김미영", first[4])
        self.assertIn("101-1703", first[4].replace(" ", ""))
        self.assertIn("천안", first[5])
        self.assertIn("국민은행", first[6])


if __name__ == "__main__":
    unittest.main()
