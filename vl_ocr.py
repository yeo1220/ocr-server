"""Vision-Language OCR via vLLM (page image → text / table JSON)."""

from __future__ import annotations

import base64
import io
import logging
from pathlib import Path
from typing import Any

from PIL import Image

from config import settings
from table_builder import export_table_aliases
from table_structure import apply_rule_structure_fix, canonical_headers, normalize_table_headers
from vl_ocr_prompts import VL_OCR_SYSTEM, VL_OCR_TABLE_USER, VL_OCR_TEXT_USER
from vllm_client import chat_vision_json, extract_json_object, resolve_vl_model

logger = logging.getLogger(__name__)


def image_to_data_url(path: str, *, max_side: int | None = None) -> str:
    """Load image (optionally downscale) as a data URL for OpenAI-compatible VL APIs."""
    max_side = max_side or settings.vllm_vl_max_image_side
    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        if max(w, h) > max_side:
            scale = max_side / max(w, h)
            im = im.resize(
                (max(1, int(w * scale)), max(1, int(h * scale))),
                Image.Resampling.LANCZOS,
            )
        buf = io.BytesIO()
        im.save(buf, format="PNG", optimize=True)
        b64 = base64.standard_b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _blocks_from_table(table: dict[str, Any]) -> list[dict]:
    blocks: list[dict] = []
    for cell in table.get("cells") or []:
        blocks.append(
            {
                "text": cell.get("text", ""),
                "score": 1.0,
                "box": [],
            }
        )
    return blocks


def _table_from_vl_json(
    parsed: dict[str, Any],
    *,
    num_cols: int,
    header_rows: int,
) -> dict[str, Any]:
    raw_table = parsed.get("table") or {}
    headers = list(raw_table.get("headers") or [])
    data = [list(r) for r in raw_table.get("data") or []]

    if len(headers) < num_cols:
        headers.extend([""] * (num_cols - len(headers)))
    headers = headers[:num_cols]

    norm_data: list[list[str]] = []
    for row in data:
        row = list(row) + [""] * max(0, num_cols - len(row))
        norm_data.append(row[:num_cols])

    table: dict[str, Any] = {
        "cols": num_cols,
        "header_row": 0,
        "header_rows": max(1, header_rows),
        "row_count": len(norm_data),
        "headers": headers,
        "data": norm_data,
        "all_rows": [headers] + norm_data if header_rows <= 1 else [headers] + norm_data,
        "cells": [],
        "col_boundaries": [],
    }

    if header_rows > 1 and norm_data:
        table["all_rows"] = [headers] + norm_data

    table = normalize_table_headers(table)
    if not table.get("headers") or all(not h for h in table["headers"]):
        table["headers"] = canonical_headers(num_cols)

    table, _ = apply_rule_structure_fix(table)
    return table


async def run_vl_ocr_page(
    image_path: str,
    *,
    table_mode: bool = False,
    num_cols: int = 9,
    header_rows: int = 1,
) -> dict[str, Any]:
    """Run VL OCR on one page image file."""
    await resolve_vl_model()

    data_url = image_to_data_url(image_path)
    if table_mode:
        user_text = VL_OCR_TABLE_USER.format(
            num_cols=num_cols,
            header_rows=header_rows,
        )
    else:
        user_text = VL_OCR_TEXT_USER

    messages = [
        {"role": "system", "content": VL_OCR_SYSTEM},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": user_text},
            ],
        },
    ]

    content, meta = await chat_vision_json(
        messages,
        max_tokens=settings.vllm_vl_max_tokens,
    )
    parsed = extract_json_object(content) or {}

    if table_mode:
        table = _table_from_vl_json(
            parsed,
            num_cols=num_cols,
            header_rows=header_rows,
        )
        export_table_aliases(table)
        table["data_refined"] = [list(r) for r in table.get("data") or []]
        for cell in table.get("cells") or []:
            cell["text_refined"] = cell.get("text", "")
            cell["refined"] = False

        text = parsed.get("text") or ""
        if not text.strip():
            from table_builder import table_to_text

            text = table_to_text(table)

        return {
            "text": text,
            "blocks": _blocks_from_table(table),
            "raw_blocks": [],
            "avg_score": 1.0,
            "table": table,
            "vl_meta": meta,
            "ocr_backend": "vllm_vl",
        }

    text = str(parsed.get("text") or content or "").strip()
    return {
        "text": text,
        "blocks": [{"text": line, "score": 1.0, "box": []} for line in text.splitlines() if line.strip()],
        "raw_blocks": [],
        "avg_score": 1.0,
        "vl_meta": meta,
        "ocr_backend": "vllm_vl",
    }


def warmup_vl() -> None:
    """No-op placeholder; VL model warms on first vLLM request."""
    logger.info("VL OCR backend ready (lazy load on first request)")
