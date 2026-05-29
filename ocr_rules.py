"""Fast deterministic OCR fixes before vLLM (Hangul tables / land documents)."""

from __future__ import annotations

import re
from typing import Any

# Header / text typos (order: longer patterns first where relevant)
_TEXT_REPLACEMENTS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"소제지"), "소재지"),
    (re.compile(r"소재비"), "소재지"),
    (re.compile(r"단까"), "단가"),
    (re.compile(r"금엑"), "금액"),
)

_NUMERIC_HEADER_KW = frozenset(
    (
        "단가",
        "금액",
        "보상액",
        "수량",
        "면적",
        "수량/면적",
        "amount",
        "price",
    )
)


def _header_looks_numeric(header: str) -> bool:
    h = (header or "").lower()
    return any(kw in h for kw in _NUMERIC_HEADER_KW)


def _fix_numeric_ocr(text: str) -> str:
    """Common glyph confusions in amount / quantity cells."""
    s = (text or "").strip()
    if not s:
        return s
    # Preserve commas, units, spaces; fix letter↔digit in digit runs only
    out: list[str] = []
    for ch in s:
        if ch in "0123456789,. 원㎡m²/-":
            out.append(ch)
            continue
        repl = {
            "O": "0",
            "o": "0",
            "l": "1",
            "I": "1",
            "|": "1",
            "S": "5",
            "B": "8",
        }.get(ch)
        out.append(repl if repl is not None else ch)
    return "".join(out)


def fix_cell_text(text: str, header: str = "") -> tuple[str, bool]:
    """Return (corrected_text, changed)."""
    original = str(text or "")
    s = original.strip()
    if not s:
        return original, False

    if _header_looks_numeric(header):
        fixed = _fix_numeric_ocr(s)
    else:
        fixed = s
        for pat, repl in _TEXT_REPLACEMENTS:
            fixed = pat.sub(repl, fixed)

    if fixed != s or fixed != original:
        return fixed, True
    return original, False


def apply_rules_to_table(table: dict[str, Any]) -> tuple[dict[str, Any], int]:
    """Apply rules to data cells; sync ``cells`` text. Returns (table, fix_count)."""
    headers = list(table.get("headers") or [])
    data = [list(row) for row in table.get("data") or []]
    if not data:
        return table, 0

    n_cols = int(table.get("cols") or len(headers) or (len(data[0]) if data else 0))
    header_rows = int(table.get("header_rows", 1))
    fixes = 0

    for ri, row in enumerate(data):
        for ci in range(len(row)):
            hdr = headers[ci] if ci < len(headers) else ""
            new_val, changed = fix_cell_text(row[ci], hdr)
            if changed:
                row[ci] = new_val
                fixes += 1

    out = dict(table)
    out["data"] = data

    updated_cells: list[dict] = []
    for cell in table.get("cells") or []:
        r = int(cell.get("row", -1))
        c = int(cell.get("col", -1))
        d_idx = r - header_rows if r >= header_rows else None
        text = str(cell.get("text") or "")
        if (
            d_idx is not None
            and 0 <= d_idx < len(data)
            and c < len(data[d_idx])
        ):
            text = data[d_idx][c]
        updated_cells.append({**cell, "text": text})
    out["cells"] = updated_cells
    return out, fixes
