"""PaddleOCR-VL backend: in-process layout detection + 0.9B VL recognition.

Runs the official PaddleOCR-VL pipeline (PP-DocLayout + PaddleOCR-VL-0.9B) on a
page image and converts its HTML table output into the server's table dict so
the existing decision-table flow (and Django consumer) works unchanged.

Chosen for DGX Spark (GB10, 273 GB/s): the VLM only recognizes small cropped
layout blocks, so output stays compact/fast while the layout detector provides
accurate row/column geometry — unlike whole-page VLM transcription which is
token-bound and collapses dense tables.
"""

import asyncio
import json
import logging
import os
import re

from bs4 import BeautifulSoup

from config import settings

logger = logging.getLogger(__name__)

# gemma-chat 단일 모델로 OCR 텍스트 교정(refine)을 수행할지 여부.
# 추출은 PaddleOCR-VL(정확)이 담당하고, gemma는 한글 오인식만 가드 하에 보정한다.
_REFINE_ENABLED = os.getenv("PADDLE_VL_REFINE", "false").lower() in ("1", "true", "yes")
_DIGITS_ONLY = re.compile(r"\D")

_PIPELINE = None
_PREDICT_LOCK = asyncio.Lock()


def get_pipeline():
    global _PIPELINE
    if _PIPELINE is None:
        from paddleocr import PaddleOCRVL

        version = getattr(settings, "paddle_vl_version", "v1.6")
        logger.info("Loading PaddleOCR-VL pipeline (%s)...", version)
        _PIPELINE = PaddleOCRVL(pipeline_version=version)
        logger.info("PaddleOCR-VL pipeline ready")
    return _PIPELINE


def warmup() -> None:
    try:
        get_pipeline()
    except Exception as e:  # pragma: no cover - non-fatal
        logger.warning("PaddleOCR-VL warmup failed (non-fatal): %s", e)


def _result_markdown(res) -> str:
    md = getattr(res, "markdown", None)
    if isinstance(md, dict):
        val = md.get("markdown_texts")
        if val is None:
            val = md.get("text")
        return str(val or "")
    return str(md or "")


def _strip_html(md: str) -> str:
    soup = BeautifulSoup(md or "", "html.parser")
    return soup.get_text("\n", strip=True)


def _expand_table(table_tag) -> list[list[str]]:
    """Expand an HTML table into a dense matrix, replicating row/col spans."""
    grid: list[dict] = []
    occupied: dict[tuple[int, int], str] = {}
    for r, tr in enumerate(table_tag.find_all("tr")):
        while len(grid) <= r:
            grid.append({})
        c = 0
        for cell in tr.find_all(["td", "th"]):
            while (r, c) in occupied:
                grid[r][c] = occupied[(r, c)]
                c += 1
            text = cell.get_text(" ", strip=True)
            try:
                cs = max(1, int(cell.get("colspan", 1)))
            except (TypeError, ValueError):
                cs = 1
            try:
                rs = max(1, int(cell.get("rowspan", 1)))
            except (TypeError, ValueError):
                rs = 1
            for dc in range(cs):
                grid[r][c + dc] = text
                for dr in range(1, rs):
                    # rowspan-covered cells stay empty (faithful table shape);
                    # replicating text here makes every sub-row's key columns
                    # identical and the downstream record-merge collapses rows.
                    occupied[(r + dr, c + dc)] = ""
            c += cs

    n_rows = len(grid)
    for (rr, cc), t in occupied.items():
        while len(grid) <= rr:
            grid.append({})
        grid[rr].setdefault(cc, t)
        n_rows = max(n_rows, rr + 1)

    max_c = 0
    for rd in grid:
        if rd:
            max_c = max(max_c, max(rd.keys()) + 1)
    return [[rd.get(i, "") for i in range(max_c)] for rd in grid]


def _largest_table(md: str):
    soup = BeautifulSoup(md or "", "html.parser")
    tables = soup.find_all("table")
    if not tables:
        return None
    return max(tables, key=lambda t: len(t.find_all(["td", "th"])))


def _html_to_headers_data(md: str, header_rows: int):
    table_tag = _largest_table(md)
    if table_tag is None:
        return [], []
    rows = _expand_table(table_tag)
    if not rows:
        return [], []
    n = max(1, header_rows)
    if len(rows) > n:
        header_block, data = rows[:n], rows[n:]
    else:
        header_block, data = rows[:1], rows[1:]

    ncol = max(len(r) for r in rows)
    headers = []
    for ci in range(ncol):
        # Prefer the most specific (last non-empty) sub-header label per column;
        # the merged top header (e.g. "구분지상권의 표시") spans many columns.
        label = ""
        for hr in header_block:
            v = hr[ci] if ci < len(hr) else ""
            if v:
                label = v
        headers.append(label)
    return headers, data


def _merge_record_rows(data, num_cols, key_col=0):
    """Merge a record's stacked physical rows into one logical row.

    재결서 보상금내역 tables stack two physical rows per compensation record:
    the top row carries 물건의 종류·수량·단가 (and 소재지/지번/소유자), the
    bottom row carries 구조 및 규격·면적·금액. The layout detector emits both
    physical rows, so we regroup them into one logical record (the system's
    contract: "one compensation record = one data row, multiline cells merged").

    A new record begins on any row whose key column (소재지) is non-empty;
    rowspan-covered continuation rows have an empty key column and fold into the
    current record. Each column's values are joined top→bottom with newlines so
    stacked columns keep order (e.g. 단가 then 금액).
    """
    groups: list[list[list[str]]] = []
    for row in data:
        row = list(row) + [""] * max(0, num_cols - len(row))
        row = row[:num_cols]
        key = (row[key_col] or "").strip() if key_col < num_cols else ""
        if key or not groups:
            groups.append([row])
        else:
            groups[-1].append(row)

    merged: list[list[str]] = []
    for g in groups:
        out = []
        for ci in range(num_cols):
            vals = []
            for r in g:
                v = (r[ci] or "").strip()
                if v and v not in vals:
                    vals.append(v)
            out.append("\n".join(vals))
        merged.append(out)
    return merged


def _build_table_dict(headers, data, num_cols, header_rows):
    """Build the server table dict from layout-derived rows.

    Unlike the whole-page-VLM JSON path, we skip ``apply_rule_structure_fix``:
    PaddleOCR-VL's layout detector already yields correct row/column geometry,
    and that record-merge step collapses legitimate per-owner rows that share
    one parcel (소재지/지번).
    """
    from table_structure import canonical_headers, normalize_table_headers

    headers = list(headers)
    if len(headers) < num_cols:
        headers += [""] * (num_cols - len(headers))
    headers = headers[:num_cols]

    norm = []
    for row in data:
        row = list(row) + [""] * max(0, num_cols - len(row))
        norm.append(row[:num_cols])

    table = {
        "cols": num_cols,
        "header_row": 0,
        "header_rows": max(1, header_rows),
        "row_count": len(norm),
        "headers": headers,
        "data": norm,
        "all_rows": [headers] + norm,
        "cells": [],
        "col_boundaries": [],
    }
    table = normalize_table_headers(table)
    if not table.get("headers") or all(not h for h in table["headers"]):
        table["headers"] = canonical_headers(num_cols)
    return table


_REFINE_SYSTEM = (
    "You correct Korean OCR misreadings in a land-compensation table (보상금내역). "
    "You output STRICT JSON only, with no commentary."
)

_REFINE_USER_TMPL = (
    "Below is a JSON table extracted from a scanned Korean document by OCR.\n"
    "Fix ONLY clear Korean text misrecognitions in descriptive cells "
    "(for example '지방성도' -> '지방상권', '평균매수엔' -> '평균매수액').\n"
    "STRICT RULES:\n"
    "- NEVER change any digit, amount, unit price, area, ratio, or parenthetical ID "
    "such as (101-1703). Copy all numbers exactly.\n"
    "- NEVER change personal names or addresses unless an obvious single-character "
    "OCR slip; when unsure, keep the original.\n"
    "- Keep EXACTLY the same number of rows, and the same number of columns per row.\n"
    "- Keep empty strings empty; never add or delete rows.\n"
    "- Preserve newline characters inside cells.\n"
    'Return JSON of the form {"data": [["cell", ...], ...]} with identical shape.\n\n'
)


async def _refine_table_with_gemma(headers, data, num_cols):
    """Guarded gemma-chat correction of OCR text garbles.

    Protects data integrity: any cell whose digit sequence changes is reverted to
    the original, and a shape/row-count mismatch rejects the whole refine. This
    lets the single gemma-chat model improve descriptive text without risking the
    legally/financially significant numbers, IDs, names that PaddleOCR-VL already
    captured accurately.
    """
    if not data:
        return [list(r) for r in data]

    import httpx

    from vllm_client import check_chat_reachable, extract_json_object

    fallback = [list(r) for r in data]
    if not await check_chat_reachable():
        logger.info("paddle_vl refine skipped: chat (gemma) unreachable")
        return fallback

    user = _REFINE_USER_TMPL + json.dumps(
        {"headers": headers, "data": data}, ensure_ascii=False
    )
    base = settings.vllm_base_url.rstrip("/")
    payload = {
        "model": settings.vllm_model,
        "messages": [
            {"role": "system", "content": _REFINE_SYSTEM},
            {"role": "user", "content": user},
        ],
        "temperature": 0.0,
        "max_tokens": 4096,
        "response_format": {"type": "json_object"},
    }
    timeout = float(getattr(settings, "vllm_vl_timeout", 240) or 240)
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.post(
                f"{base}/chat/completions",
                headers={
                    "Authorization": f"Bearer {settings.vllm_api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            r.raise_for_status()
            content = (
                r.json().get("choices", [{}])[0].get("message", {}).get("content", "")
            )
    except Exception as exc:  # network/model errors must not break OCR
        logger.warning("paddle_vl gemma refine call failed: %r", exc)
        return fallback

    obj = extract_json_object(content)
    new = obj.get("data") if isinstance(obj, dict) else None
    if not isinstance(new, list) or len(new) != len(data):
        logger.warning(
            "paddle_vl gemma refine rejected (row count %s != %s)",
            len(new) if isinstance(new, list) else None,
            len(data),
        )
        return fallback

    out = []
    changed = 0
    for orig_row, new_row in zip(data, new):
        if not isinstance(new_row, list) or len(new_row) != len(orig_row):
            logger.warning("paddle_vl gemma refine rejected (column shape mismatch)")
            return fallback
        fixed = []
        for o, n in zip(orig_row, new_row):
            o = "" if o is None else str(o)
            n = "" if n is None else str(n)
            # Keep the original cell when:
            #  - any digit changed (protects 금액·단가·면적·ID), or
            #  - the cell carries a parenthetical token like (101-1703)
            #    which marks owner names / unit IDs we must never rewrite.
            if _DIGITS_ONLY.sub("", o) != _DIGITS_ONLY.sub("", n) or (
                "(" in o or ")" in o
            ):
                fixed.append(o)
            else:
                if n != o:
                    changed += 1
                fixed.append(n)
        out.append(fixed)
    logger.info("paddle_vl gemma refine applied: %s cell(s) corrected", changed)
    return out


async def run_paddle_vl_page(image_path, *, table_mode, num_cols, header_rows):
    pipe = get_pipeline()

    def _predict():
        parts = []
        for res in pipe.predict(image_path):
            parts.append(_result_markdown(res))
        return "\n".join(parts)

    async with _PREDICT_LOCK:
        md = await asyncio.to_thread(_predict)

    meta = {"backend": "paddle_vl", "markdown_len": len(md)}

    if table_mode:
        from table_builder import export_table_aliases, table_to_text

        headers, data = _html_to_headers_data(md, header_rows)
        data = _merge_record_rows(data, num_cols)
        table = _build_table_dict(headers, data, num_cols, header_rows)
        export_table_aliases(table)
        if _REFINE_ENABLED:
            table["data_refined"] = await _refine_table_with_gemma(
                table["headers"], table["data"], num_cols
            )
        else:
            table["data_refined"] = [list(r) for r in table.get("data") or []]
        for cell in table.get("cells") or []:
            cell["text_refined"] = cell.get("text", "")
            cell["refined"] = False
        text = table_to_text(table) or _strip_html(md)
        return {
            "text": text,
            "blocks": [],
            "raw_blocks": [],
            "avg_score": 1.0,
            "table": table,
            "vl_meta": meta,
            "ocr_backend": "paddle_vl",
        }

    text = _strip_html(md)
    return {
        "text": text,
        "blocks": [
            {"text": line, "score": 1.0, "box": []}
            for line in text.splitlines()
            if line.strip()
        ],
        "raw_blocks": [],
        "avg_score": 1.0,
        "vl_meta": meta,
        "ocr_backend": "paddle_vl",
    }
