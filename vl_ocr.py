"""Vision-Language OCR via vLLM (page image → text / table JSON)."""

from __future__ import annotations

import base64
import io
import logging
import re
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


def _coerce_rows(value):
    """Coerce an arbitrary table-ish value into a list of string-cell rows."""
    rows = []
    if not isinstance(value, list):
        return rows
    for r in value:
        if isinstance(r, list):
            rows.append(["" if c is None else str(c) for c in r])
        elif isinstance(r, dict):
            rows.append(["" if c is None else str(c) for c in r.values()])
        elif r is not None:
            rows.append([str(r)])
    return rows


def _looks_like_header_row(row):
    joined = "".join(row)
    return any(k in joined for k in ("\uc18c\uc7ac\uc9c0", "\uc9c0\ubc88", "\ubb3c\uac74", "\uae08\uc561", "\uc131\uba85", "\ub2e8\uac00"))


def _normalize_vl_table_payload(parsed, *, num_cols, header_rows):
    """Robustly extract (headers, data) from varied VL JSON shapes.

    Handles ``table`` as a dict ({headers, data|rows}) or as a bare list of
    rows, plus top-level ``headers``/``data`` fallbacks. Smaller VL models
    (e.g. 1.7B) are less consistent than 14B about the exact schema, so we
    must not assume a dict here (previously caused AttributeError -> 500).
    """
    raw_table = parsed.get("table")
    headers = []
    data = []

    if isinstance(raw_table, dict):
        headers = list(raw_table.get("headers") or [])
        data = _coerce_rows(raw_table.get("data") or raw_table.get("rows") or [])
    elif isinstance(raw_table, list):
        rows = _coerce_rows(raw_table)
        if rows and not list(parsed.get("headers") or []) and _looks_like_header_row(rows[0]):
            headers = rows[0]
            data = rows[max(1, header_rows):]
        else:
            data = rows

    if not headers:
        headers = list(parsed.get("headers") or [])
    if not data:
        data = _coerce_rows(parsed.get("data") or [])

    headers = ["" if h is None else str(h) for h in headers]
    return headers, data


def _table_from_vl_json(
    parsed: dict[str, Any],
    *,
    num_cols: int,
    header_rows: int,
) -> dict[str, Any]:
    headers, data = _normalize_vl_table_payload(
        parsed, num_cols=num_cols, header_rows=header_rows
    )

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


_UNIT_RE = re.compile(
    r"(?P<text>.*?)\s*(?P<x1>\d*\.\d+|\d+\.\d*)\s*,\s*(?P<y1>\d*\.\d+|\d+\.\d*)"
    r"\s*,\s*(?P<x2>\d*\.\d+|\d+\.\d*)\s*,\s*(?P<y2>\d*\.\d+|\d+\.\d*)\s*$"
)


def _poly(x1, y1, x2, y2):
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def _parse_ocr_units(content, scale_w=1.0, scale_h=1.0):
    """Parse VARCO-2.0-1.7B-OCR output lines: ``<text>x1, y1, x2, y2``.

    Coordinates are normalized (0..1); text is already word/phrase level.
    Returns geometric blocks ``{text, score, box}`` with coords scaled to
    pixel-like magnitudes so build_table's pixel thresholds behave correctly.
    """
    blocks = []
    for raw_line in (content or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        m = _UNIT_RE.match(line)
        if not m:
            continue
        txt = (m.group("text") or "").strip()
        if not txt:
            continue
        try:
            x1 = float(m.group("x1"))
            y1 = float(m.group("y1"))
            x2 = float(m.group("x2"))
            y2 = float(m.group("y2"))
        except (TypeError, ValueError):
            continue
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        x1 *= scale_w
        x2 *= scale_w
        y1 *= scale_h
        y2 *= scale_h
        blocks.append({"text": txt, "score": 1.0, "box": _poly(x1, y1, x2, y2)})
    return blocks


def _char_units_to_word_blocks(units):
    """Merge char units into word-level blocks (line cluster + x-gap grouping)."""
    if not units:
        return []
    hs = sorted((u["y2"] - u["y1"]) for u in units if u["y2"] > u["y1"])
    med_h = hs[len(hs) // 2] if hs else 12.0
    y_tol = max(6.0, med_h * 0.6)

    ordered = sorted(units, key=lambda u: ((u["y1"] + u["y2"]) / 2.0, u["x1"]))
    lines = []
    for u in ordered:
        cy = (u["y1"] + u["y2"]) / 2.0
        if lines:
            prev = lines[-1]
            prev_cy = sum((x["y1"] + x["y2"]) / 2.0 for x in prev) / len(prev)
            if abs(cy - prev_cy) <= y_tol:
                prev.append(u)
                continue
        lines.append([u])

    blocks = []
    for line in lines:
        line = sorted(line, key=lambda u: u["x1"])
        widths = sorted((u["x2"] - u["x1"]) for u in line if u["x2"] > u["x1"])
        med_w = widths[len(widths) // 2] if widths else med_h
        gap_tol = max(med_w * 0.9, med_h * 0.6)
        groups = [[line[0]]]
        for u in line[1:]:
            prev = groups[-1][-1]
            if (u["x1"] - prev["x2"]) <= gap_tol:
                groups[-1].append(u)
            else:
                groups.append([u])
        for g in groups:
            txt = "".join(x["text"] for x in g).strip()
            if not txt:
                continue
            x1 = min(x["x1"] for x in g)
            y1 = min(x["y1"] for x in g)
            x2 = max(x["x2"] for x in g)
            y2 = max(x["y2"] for x in g)
            blocks.append({"text": txt, "score": 1.0, "box": _poly(x1, y1, x2, y2)})
    return blocks


def _blocks_to_text(blocks):
    if not blocks:
        return ""

    def cy(b):
        ys = [p[1] for p in b["box"]]
        return sum(ys) / len(ys)

    def x0(b):
        return min(p[0] for p in b["box"])

    def h(b):
        ys = [p[1] for p in b["box"]]
        return max(ys) - min(ys)

    hs = sorted(h(b) for b in blocks if h(b) > 0)
    med_h = hs[len(hs) // 2] if hs else 12.0
    y_tol = max(6.0, med_h * 0.6)
    ordered = sorted(blocks, key=lambda b: (cy(b), x0(b)))
    lines = []
    for b in ordered:
        if lines:
            prev = lines[-1]
            prev_cy = sum(cy(x) for x in prev) / len(prev)
            if abs(cy(b) - prev_cy) <= y_tol:
                prev.append(b)
                continue
        lines.append([b])
    out = []
    for line in lines:
        line = sorted(line, key=x0)
        out.append(" ".join(b["text"] for b in line))
    return "\n".join(out)


async def _run_vl_ocr_char_bbox(image_path, *, table_mode, num_cols, header_rows):
    """OCR-specialized path: <ocr> char+bbox recognition -> geometric table builder."""
    from table_builder import build_table, table_to_text

    max_side = getattr(settings, "vllm_vl_ocr_image_side", None) or settings.vllm_vl_max_image_side
    with Image.open(image_path) as _im:
        _src_w, _src_h = _im.size
    data_url = image_to_data_url(image_path, max_side=max_side)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": "<ocr>"},
            ],
        }
    ]
    content, meta = await chat_vision_json(
        messages,
        max_tokens=settings.vllm_vl_max_tokens,
        json_mode=False,
    )
    blocks = _parse_ocr_units(content, scale_w=float(_src_w), scale_h=float(_src_h))
    meta = {**meta, "ocr_units": len(blocks)}

    if table_mode:
        table = build_table(
            blocks,
            num_cols=num_cols,
            header_row=0,
            header_rows=max(1, header_rows),
        )
        export_table_aliases(table)
        table["data_refined"] = [list(r) for r in table.get("data") or []]
        for cell in table.get("cells") or []:
            cell["text_refined"] = cell.get("text", "")
            cell["refined"] = False
        return {
            "text": table_to_text(table),
            "blocks": blocks,
            "raw_blocks": blocks,
            "avg_score": 1.0,
            "table": table,
            "vl_meta": meta,
            "ocr_backend": "vllm_vl_ocr",
        }

    return {
        "text": _blocks_to_text(blocks),
        "blocks": blocks,
        "raw_blocks": blocks,
        "avg_score": 1.0,
        "vl_meta": meta,
        "ocr_backend": "vllm_vl_ocr",
    }


async def run_vl_ocr_page(
    image_path: str,
    *,
    table_mode: bool = False,
    num_cols: int = 9,
    header_rows: int = 1,
) -> dict[str, Any]:
    """Run VL OCR on one page image file."""
    if settings.vl_output_mode == "paddle_vl":
        from paddle_vl_ocr import run_paddle_vl_page
        return await run_paddle_vl_page(
            image_path,
            table_mode=table_mode,
            num_cols=num_cols,
            header_rows=header_rows,
        )
    await resolve_vl_model()

    if settings.vl_output_mode == "char_bbox":
        return await _run_vl_ocr_char_bbox(
            image_path,
            table_mode=table_mode,
            num_cols=num_cols,
            header_rows=header_rows,
        )

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
    """Warm the active VL backend (PaddleOCR-VL loads weights; vLLM lazy)."""
    if settings.vl_output_mode == "paddle_vl":
        from paddle_vl_ocr import warmup as _paddle_vl_warmup
        _paddle_vl_warmup()
        return
    logger.info("VL OCR backend ready (lazy load on first request)")
