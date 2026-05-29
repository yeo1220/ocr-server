"""Tests for vLLM client helpers (no HTTP)."""

import unittest

from vllm_client import _profile_from_model, extract_json_object, strip_thinking_content


class VllmClientTests(unittest.TestCase):
    def test_profile_thinking_model(self):
        p = _profile_from_model(
            "qwen-ocr",
            "/models/Qwen3-Next-80B-A3B-Thinking-FP8",
            base_url="http://127.0.0.1:8088/v1",
        )
        self.assertTrue(p.is_thinking)
        self.assertFalse(p.is_small_instruct)
        self.assertEqual(p.temperature, 0.0)
        self.assertIn(
            "enable_thinking",
            p.extra_body.get("extra_body", {}).get("chat_template_kwargs", {}),
        )

    def test_profile_small_refine_model(self):
        p = _profile_from_model(
            "qwen-refine",
            "/models/Qwen2.5-7B-Instruct",
            base_url="http://127.0.0.1:8002/v1",
        )
        self.assertTrue(p.is_small_instruct)
        self.assertFalse(p.is_thinking)
        self.assertEqual(p.max_output_tokens, 1024)

    def test_profile_14b_refine_model(self):
        from config import settings

        p = _profile_from_model(
            "qwen-refine",
            "/models/Qwen2.5-14B-Instruct",
            base_url="http://127.0.0.1:8002/v1",
        )
        self.assertTrue(p.is_small_instruct)
        self.assertEqual(p.max_output_tokens, settings.vllm_refine_max_tokens_medium)

    def test_strip_thinking(self):
        raw = "reasoning\n{\"rows\": []}"
        self.assertEqual(strip_thinking_content(raw), '{"rows": []}')

    def test_extract_json_after_thinking(self):
        raw = "x\n```json\n{\"rows\": [[\"a\"]]}\n```"
        self.assertEqual(extract_json_object(raw), {"rows": [["a"]]})


if __name__ == "__main__":
    unittest.main()
