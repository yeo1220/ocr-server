"""vLLM table cell refinement — rules first, batched calls, DGX-tuned parameters."""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from config import settings
from llm_prompts import (
    TABLE_REFINE_BATCH_SYSTEM,
    TABLE_REFINE_SYSTEM,
    TABLE_STRUCTURE_REFINE_SYSTEM,
)
from table_structure import (
    detect_structure_issues,
    rebuild_table_data,
)
from ocr_rules import apply_rules_to_table
from vllm_client import (
    chat_json,
    estimate_refine_max_tokens,
    extract_json_object,
    get_model_profile,
    get_resolved_model_id,
)

logger = logging.getLogger(__name__)

# Re-export for tests
_extract_json = extract_json_object


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
        "Correct low_confidence_cells in data_rows. "
        'Return JSON {"rows": [...]} only.\n\n'
        f"{json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}"
    )


def _build_batch_user_prompt(
    batch_items: list[dict[str, Any]],
) -> str:
    return (
        "Correct each table's low_confidence_cells. "
        'Return JSON {"results":[{"page":N,"rows":[...]},...]} only.\n\n'
        f"{json.dumps({'tables': batch_items}, ensure_ascii=False, separators=(',', ':'))}"
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


def _finish_without_llm(table: dict[str, Any], meta: dict[str, Any]) -> tuple[dict, dict]:
    data = [list(row) for row in table.get("data") or []]
    out = dict(table)
    out["data_refined"] = data
    for cell in out.get("cells") or []:
        cell["text_refined"] = cell.get("text", "")
        cell["refined"] = False
    meta["cells_corrected"] = 0
    return out, meta


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
    from vllm_client import check_reachable

    return await check_reachable()


def _prepare_table_for_refine(
    table: dict[str, Any],
    threshold: float,
) -> tuple[dict[str, Any], list[dict], dict[str, Any]]:
    """Rules → recount low-confidence cells. Returns (table, fix_cells, meta_base)."""
    header_rows = int(table.get("header_rows", 1))
    data = [list(row) for row in table.get("data") or []]
    num_cols = int(table["cols"])

    meta: dict[str, Any] = {
        "enabled": True,
        "model": get_resolved_model_id(),
        "threshold": threshold,
        "cells_corrected": 0,
        "rule_fixes": 0,
        "llm_latency_sec": 0.0,
        "skipped": False,
        "skip_reason": None,
        "error": None,
        "batched": False,
    }

    table, rule_fixes = apply_rules_to_table(table)
    meta["rule_fixes"] = rule_fixes
    data = [list(row) for row in table.get("data") or []]

    fix_cells = _low_confidence_cells(table, threshold, header_rows)
    return table, fix_cells, meta


async def _refine_single_via_llm(
    table: dict[str, Any],
    fix_cells: list[dict],
    meta: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    headers = list(table.get("headers") or [])
    data = [list(row) for row in table.get("data") or []]
    num_cols = int(table["cols"])

    messages = [
        {"role": "system", "content": TABLE_REFINE_SYSTEM},
        {
            "role": "user",
            "content": _build_user_prompt(headers, data, fix_cells),
        },
    ]
    max_tokens = estimate_refine_max_tokens(
        len(data), num_cols, len(fix_cells), profile=get_model_profile()
    )

    t0 = time.time()
    try:
        content, usage = await chat_json(messages, max_tokens=max_tokens)
    except Exception as e:
        logger.warning("vLLM refine failed: %s", e)
        meta["skipped"] = True
        meta["skip_reason"] = "vllm_error"
        meta["error"] = str(e)
        meta["llm_latency_sec"] = round(time.time() - t0, 3)
        return _finish_without_llm(table, meta)

    meta["llm_latency_sec"] = round(time.time() - t0, 3)
    meta["usage"] = usage
    parsed = extract_json_object(content)
    if not parsed or "rows" not in parsed:
        meta["skipped"] = True
        meta["skip_reason"] = "invalid_llm_response"
        meta["error"] = "Could not parse JSON from LLM output"
        return _finish_without_llm(table, meta)

    refined_data = parsed["rows"]
    if not _validate_rows(data, refined_data, num_cols):
        meta["skipped"] = True
        meta["skip_reason"] = "shape_mismatch"
        meta["error"] = "LLM returned wrong row/column count"
        return _finish_without_llm(table, meta)

    out = await apply_refined_to_table(table, refined_data)
    meta["cells_corrected"] = sum(1 for c in out.get("cells") or [] if c.get("refined"))
    return out, meta


def _structure_refine_enabled() -> bool:
    mode = settings.vllm_structure_refine
    return mode not in ("off", "false", "0", "no") and settings.vllm_enabled


def _validate_structure_rows(
    refined: list[list[str]],
    num_cols: int,
) -> bool:
    for row in refined:
        if len(row) != num_cols:
            return False
    return True


async def structure_refine_via_llm(
    table: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """One compact vLLM call: drop duplicate headers, merge wrapped rows."""
    headers = list(table.get("headers") or [])
    data = [list(r) for r in table.get("data") or []]
    num_cols = int(table.get("cols") or len(headers))
    header_rows = int(table.get("header_rows", 1))
    issues = detect_structure_issues(table)

    meta: dict[str, Any] = {
        "enabled": True,
        "skipped": True,
        "skip_reason": None,
        "issues_before": issues,
        "rows_before": len(data),
        "rows_after": len(data),
        "llm_latency_sec": 0.0,
        "error": None,
    }

    if not data:
        meta["skip_reason"] = "empty_table"
        return table, meta

    if not _structure_refine_enabled():
        meta["skip_reason"] = "structure_refine_disabled"
        return table, meta

    mode = settings.vllm_structure_refine
    if mode not in ("always", "1", "true", "yes") and not issues.get("needs_fix"):
        meta["skip_reason"] = "no_structure_issues"
        return table, meta

    payload = {
        "headers": headers,
        "data_rows": data,
        "issues": {
            "duplicate_header_rows": issues.get("duplicate_header_rows", 0),
            "merge_candidate_rows": issues.get("merge_candidate_rows", 0),
        },
    }
    messages = [
        {"role": "system", "content": TABLE_STRUCTURE_REFINE_SYSTEM},
        {
            "role": "user",
            "content": (
                "Fix structure in data_rows. Return JSON {\"rows\": [...]} only.\n\n"
                f"{json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}"
            ),
        },
    ]
    max_tokens = min(
        settings.vllm_structure_max_tokens,
        estimate_refine_max_tokens(len(data), num_cols, 0, profile=get_model_profile())
        + 512,
    )

    t0 = time.time()
    try:
        content, usage = await chat_json(messages, max_tokens=max_tokens)
    except Exception as e:
        logger.warning("vLLM structure refine failed: %s", e)
        meta["skip_reason"] = "vllm_error"
        meta["error"] = str(e)
        meta["llm_latency_sec"] = round(time.time() - t0, 3)
        return table, meta

    meta["llm_latency_sec"] = round(time.time() - t0, 3)
    meta["usage"] = usage
    parsed = extract_json_object(content)
    if not parsed or "rows" not in parsed:
        meta["skip_reason"] = "invalid_llm_response"
        meta["error"] = "Could not parse JSON from LLM output"
        return table, meta

    refined_data = parsed["rows"]
    if not _validate_structure_rows(refined_data, num_cols):
        meta["skip_reason"] = "shape_mismatch"
        meta["error"] = "LLM returned wrong column count"
        return table, meta

    if len(refined_data) > len(data):
        meta["skip_reason"] = "row_count_increased"
        meta["error"] = "LLM added rows"
        return table, meta

    meta["skipped"] = False
    meta["rows_after"] = len(refined_data)
    out = rebuild_table_data(table, refined_data, header_rows)
    return out, meta


async def refine_table(
    table: dict[str, Any],
    *,
    threshold: float | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Refine one table: structure (rules+vLLM), then cell OCR via vLLM if needed."""
    threshold = threshold if threshold is not None else settings.vllm_refine_threshold
    table, struct_meta = await structure_refine_via_llm(table)
    table, fix_cells, meta = _prepare_table_for_refine(table, threshold)
    meta["structure"] = struct_meta

    if not fix_cells:
        meta["skipped"] = True
        meta["skip_reason"] = "no_low_confidence_cells"
        return _finish_without_llm(table, meta)

    if not settings.vllm_enabled:
        meta["skipped"] = True
        meta["skip_reason"] = "vllm_disabled"
        return _finish_without_llm(table, meta)

    return await _refine_single_via_llm(table, fix_cells, meta)


async def refine_tables_batch(
    page_tables: list[tuple[int, dict[str, Any]]],
    *,
    threshold: float | None = None,
) -> dict[int, tuple[dict[str, Any], dict[str, Any]]]:
    """Refine multiple page tables; one vLLM call per chunk when batching is enabled."""
    threshold = threshold if threshold is not None else settings.vllm_refine_threshold
    results: dict[int, tuple[dict[str, Any], dict[str, Any]]] = {}

    pending: list[tuple[int, dict[str, Any], list[dict], dict[str, Any]]] = []
    for page_num, table in page_tables:
        table, struct_meta = await structure_refine_via_llm(table)
        table, fix_cells, meta = _prepare_table_for_refine(table, threshold)
        meta["structure"] = struct_meta
        if not fix_cells:
            meta["skipped"] = True
            meta["skip_reason"] = "no_low_confidence_cells"
            results[page_num] = _finish_without_llm(table, meta)
            continue
        if not settings.vllm_enabled:
            meta["skipped"] = True
            meta["skip_reason"] = "vllm_disabled"
            results[page_num] = _finish_without_llm(table, meta)
            continue
        pending.append((page_num, table, fix_cells, meta))

    if not pending:
        return results

    if not settings.vllm_refine_batch or len(pending) == 1:
        for page_num, table, fix_cells, meta in pending:
            out, m = await _refine_single_via_llm(table, fix_cells, meta)
            results[page_num] = (out, m)
        return results

    chunk_size = max(1, settings.vllm_max_tables_per_batch)
    for offset in range(0, len(pending), chunk_size):
        chunk = pending[offset : offset + chunk_size]
        await _refine_batch_chunk(chunk, results)

    return results


async def _refine_batch_chunk(
    chunk: list[tuple[int, dict[str, Any], list[dict], dict[str, Any]]],
    results: dict[int, tuple[dict[str, Any], dict[str, Any]]],
) -> None:
    batch_payload: list[dict[str, Any]] = []
    total_rows = 0
    total_cols = 0
    total_fix = 0

    for page_num, table, fix_cells, _meta in chunk:
        data = [list(row) for row in table.get("data") or []]
        num_cols = int(table["cols"])
        total_rows += len(data)
        total_cols = max(total_cols, num_cols)
        total_fix += len(fix_cells)
        batch_payload.append(
            {
                "page": page_num,
                "headers": list(table.get("headers") or []),
                "data_rows": data,
                "low_confidence_cells": fix_cells,
            }
        )

    messages = [
        {"role": "system", "content": TABLE_REFINE_BATCH_SYSTEM},
        {"role": "user", "content": _build_batch_user_prompt(batch_payload)},
    ]
    max_tokens = estimate_refine_max_tokens(
        total_rows,
        total_cols,
        total_fix,
        profile=get_model_profile(),
    )
    max_tokens = min(
        max_tokens * max(1, len(chunk) // 2 + 1),
        (get_model_profile().max_output_tokens if get_model_profile() else 2048)
        * 2,
    )

    t0 = time.time()
    try:
        content, usage = await chat_json(messages, max_tokens=max_tokens)
        parsed = extract_json_object(content)
    except Exception as e:
        logger.warning("vLLM batch refine failed, falling back per-page: %s", e)
        for page_num, table, fix_cells, meta in chunk:
            meta["batched"] = False
            out, m = await _refine_single_via_llm(table, fix_cells, meta)
            results[page_num] = (out, m)
        return

    latency = round(time.time() - t0, 3)
    by_page: dict[int, list[list[str]]] = {}
    if parsed:
        for item in parsed.get("results") or []:
            if isinstance(item, dict) and "page" in item and "rows" in item:
                by_page[int(item["page"])] = item["rows"]

    for page_num, table, fix_cells, meta in chunk:
        meta = dict(meta)
        meta["batched"] = True
        meta["llm_latency_sec"] = latency
        meta["usage"] = usage
        data = [list(row) for row in table.get("data") or []]
        num_cols = int(table["cols"])
        refined_data = by_page.get(page_num)

        if refined_data and _validate_rows(data, refined_data, num_cols):
            out = await apply_refined_to_table(table, refined_data)
            meta["cells_corrected"] = sum(
                1 for c in out.get("cells") or [] if c.get("refined")
            )
            results[page_num] = (out, meta)
            continue

        logger.warning(
            "Batch refine missing/invalid page %s; single-table fallback",
            page_num,
        )
        meta["batched"] = False
        out, m = await _refine_single_via_llm(table, fix_cells, meta)
        results[page_num] = (out, m)
