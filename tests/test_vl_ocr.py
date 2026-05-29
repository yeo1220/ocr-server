"""Unit tests for VL OCR helpers (no GPU / vLLM required)."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from vl_ocr import _table_from_vl_json, image_to_data_url
from vllm_client import extract_json_object


class TestVlOcr(unittest.TestCase):
    def test_image_to_data_url_png(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "page.png"
            Image.new("RGB", (800, 600), color=(255, 255, 255)).save(path)
            url = image_to_data_url(str(path), max_side=400)
            self.assertTrue(url.startswith("data:image/png;base64,"))

    def test_table_from_vl_json_nine_cols(self):
        parsed = {
            "table": {
                "headers": ["소재지", "지번", "규격", "금액", "성명", "주소", "권리", "지점", "은행주소"],
                "data": [
                    ["부산", "101", "100㎡", "1,000,000", "김미영(101-1703)", "서울", "국민은행", "강남", "서울"],
                ],
            }
        }
        table = _table_from_vl_json(parsed, num_cols=9, header_rows=1)
        self.assertEqual(table["cols"], 9)
        self.assertEqual(len(table["data"]), 1)
        self.assertEqual(len(table["data"][0]), 9)

    def test_extract_json_from_fenced_vl_output(self):
        raw = '```json\n{"text": "hello"}\n```'
        obj = extract_json_object(raw)
        self.assertEqual(obj, {"text": "hello"})


if __name__ == "__main__":
    unittest.main()
