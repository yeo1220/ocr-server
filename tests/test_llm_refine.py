"""Tests for LLM JSON parsing helpers."""

import unittest

from llm_refine import _row_to_data_index, _validate_rows
from vllm_client import extract_json_object as _extract_json


class LlmRefineTests(unittest.TestCase):
    def test_extract_json_plain(self):
        self.assertEqual(_extract_json('{"rows": []}'), {"rows": []})

    def test_extract_json_fence(self):
        raw = 'Here:\n```json\n{"rows": [["a"]]}\n```'
        self.assertEqual(_extract_json(raw), {"rows": [["a"]]})

    def test_row_to_data_index(self):
        self.assertIsNone(_row_to_data_index(0, 1))
        self.assertEqual(_row_to_data_index(1, 1), 0)
        self.assertEqual(_row_to_data_index(2, 1), 1)

    def test_validate_rows(self):
        orig = [["a", "b"], ["c", "d"]]
        self.assertTrue(_validate_rows(orig, [["a", "b"], ["c", "d"]], 2))
        self.assertFalse(_validate_rows(orig, [["a"]], 2))


if __name__ == "__main__":
    unittest.main()
