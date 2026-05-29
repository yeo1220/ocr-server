"""vLLM (OpenAI-compatible) table cell refinement."""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

import httpx

from config import settings
from llm_prompts import TABLE_REFINE_SYSTEM

logger = logging.getLogger(__name__)


def _extract_json(text: str) -> dict | None:
    text = (text or "").strip()
    if not text:
        return None
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                return None
    return None


def _row_to_data_index(row: int, header_rows: int) -> int | None:
    if row < header_rows:
        return None
    return row - header_rows


def _low_confidence_cells(
    table: dict[str, Any],
    threshold: float,
    header_rows: int,
) -> list[dict]:
    fix: list[dict] = []
    for cell in table.get("cells") or []:
        if int(cell.get("row", -1)) < header_rows:
            continue
        score = float(cell.get("score") or 0.0)
        text = str(cell.get("text") or "").strip()
        if text and score < threshold:
            fix.append(
                {
                    "row": int(cell["row"]),
                    "col": int(cell["col"]),
                    "text": text,
                    "score": score,
                }
            )
    return fix


def _build_user_prompt(
    headers: list[str],
    data: list[list[str]],
    fix_cells: list[dict],
) -> str:
    payload = {
        "headers": headers,
        "data_rows": data,
        "low_confidence_cells": fix_cells,
    }
    return (
        "Correct OCR errors in low_confidence_cells within data_rows.\n"
        'Return JSON: {"rows": [...]} with the full corrected data_rows '
        "(same row/column count).\n\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


def _validate_rows(
    original: list[list[str]],
    refined: list[list[str]],
    num_cols: int,
) -> bool:
    if len(refined) != len(original):
        return False
    for orig_row, ref_row in zip(original, refined):
        if len(ref_row) != num_cols or len(orig_row) != num_cols:
            return False
    return True


async def apply_refined_to_table(
    table: dict[str, Any],
    refined_data: list[list[str]],
) -> dict[str, Any]:
    header_rows = int(table.get("header_rows", 1))
    num_cols = int(table["cols"])
    original_data = table.get("data") or []

    if not _validate_rows(original_data, refined_data, num_cols):
        raise ValueError("refined rows shape mismatch")

    out = dict(table)
    out["data_refined"] = [list(row) for row in refined_data]

    updated_cells: list[dict] = []
    for cell in table.get("cells") or []:
        r = int(cell["row"])
        c = int(cell["col"])
        text = str(cell.get("text") or "")
        text_refined = text
        refined_flag = False

        d_idx = _row_to_data_index(r, header_rows)
        if d_idx is not None and 0 <= d_idx < len(refined_data) and c < num_cols:
            text_refined = refined_data[d_idx][c]
            refined_flag = text_refined != text

        updated_cells.append(
            {
                **cell,
                "text_refined": text_refined,
                "refined": refined_flag,
            }
        )

    out["cells"] = updated_cells
    return out


async def check_vllm_reachable() -> bool:
    if not settings.vllm_enabled:
        return False
    url = f"{settings.vllm_base_url.rstrip('/')}/models"
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(
                url,
                headers={"Authorization": f"Bearer {settings.vllm_api_key}"},
            )
            return r.status_code == 200
    except Exception:
        return False


async def refine_table(
    table: dict[str, Any],
    *,
    threshold: float | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Refine low-confidence data cells via vLLM. Returns (table, refinement_meta)."""
    threshold = threshold if threshold is not None else settings.vllm_refine_threshold
    header_rows = int(table.get("header_rows", 1))
    headers = list(table.get("headers") or [])
    data = [list(row) for row in table.get("data") or []]
    num_cols = int(table["cols"])

    meta: dict[str, Any] = {
        "enabled": True,
        "model": settings.vllm_model,
        "threshold": threshold,
        "cells_corrected": 0,
        "llm_latency_sec": 0.0,
        "skipped": False,
        "skip_reason": None,
        "error": None,
    }

    fix_cells = _low_confidence_cells(table, threshold, header_rows)
    if not fix_cells:
        meta["skipped"] = True
        meta["skip_reason"] = "no_low_confidence_cells"
        out = dict(table)
        out["data_refined"] = [list(row) for row in data]
        for cell in out.get("cells") or []:
            cell["text_refined"] = cell.get("text", "")
            cell["refined"] = False
        return out, meta

    if not settings.vllm_enabled:
        meta["skipped"] = True
        meta["skip_reason"] = "vllm_disabled"
        out = dict(table)
        out["data_refined"] = [list(row) for row in data]
        return out, meta

    payload = {
        "model": settings.vllm_model,
        "messages": [
            {"role": "system", "content": TABLE_REFINE_SYSTEM},
            {
                "role": "user",
                "content": _build_user_prompt(headers, data, fix_cells),
            },
        ],
        "temperature": 0.1,
        "max_tokens": 4096,
    }

    t0 = time.time()
    try:
        async with httpx.AsyncClient(timeout=settings.vllm_timeout) as client:
            r = await client.post(
                f"{settings.vllm_base_url.rstrip('/')}/chat/completions",
                headers={
                    "Authorization": f"Bearer {settings.vllm_api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            r.raise_for_status()
            body = r.json()
    except Exception as e:
        logger.warning("vLLM refine failed: %s", e)
        meta["skipped"] = True
        meta["skip_reason"] = "vllm_error"
        meta["error"] = str(e)
        meta["llm_latency_sec"] = round(time.time() - t0, 3)
        out = dict(table)
        out["data_refined"] = [list(row) for row in data]
        return out, meta

    meta["llm_latency_sec"] = round(time.time() - t0, 3)
    content = (
        body.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
    )
    parsed = _extract_json(content)
    if not parsed or "rows" not in parsed:
        meta["skipped"] = True
        meta["skip_reason"] = "invalid_llm_response"
        meta["error"] = "Could not parse JSON from LLM output"
        out = dict(table)
        out["data_refined"] = [list(row) for row in data]
        return out, meta

    refined_data = parsed["rows"]
    if not _validate_rows(data, refined_data, num_cols):
        meta["skipped"] = True
        meta["skip_reason"] = "shape_mismatch"
        meta["error"] = "LLM returned wrong row/column count"
        out = dict(table)
        out["data_refined"] = [list(row) for row in data]
        return out, meta

    out = await apply_refined_to_table(table, refined_data)
    meta["cells_corrected"] = sum(
        1 for c in out.get("cells") or [] if c.get("refined")
    )
    return out, meta
