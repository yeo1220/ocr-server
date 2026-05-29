"""Build fixed-column table JSON from PaddleOCR raw blocks (box coordinates)."""

from __future__ import annotations

from typing import Any


def _box_center_x(box: list) -> float:
    xs = [float(p[0]) for p in box if isinstance(p, (list, tuple)) and len(p) >= 2]
    return (sum(xs) / len(xs)) if xs else 0.0


def _box_center_y(box: list) -> float:
    ys = [float(p[1]) for p in box if isinstance(p, (list, tuple)) and len(p) >= 2]
    return (sum(ys) / len(ys)) if ys else 0.0


def _box_xrange(box: list) -> tuple[float, float]:
    xs = [float(p[0]) for p in box if isinstance(p, (list, tuple)) and len(p) >= 2]
    if not xs:
        return 0.0, 0.0
    return min(xs), max(xs)


def _box_height(box: list) -> float:
    ys = [float(p[1]) for p in box if isinstance(p, (list, tuple)) and len(p) >= 2]
    if not ys:
        return 0.0
    return max(ys) - min(ys)


def _normalize_text(text: str) -> str:
    return " ".join(str(text or "").strip().split())


def _cluster_rows(blocks: list[dict]) -> list[list[dict]]:
    if not blocks:
        return []

    ordered = sorted(blocks, key=lambda b: _box_center_y(b.get("box") or []))
    heights = sorted(
        h for h in (_box_height(b.get("box") or []) for b in ordered) if h > 0
    )
    med_h = heights[len(heights) // 2] if heights else 12.0
    y_tol = max(6.0, med_h * 0.65)

    rows: list[list[dict]] = []
    for block in ordered:
        cy = _box_center_y(block.get("box") or [])
        if not rows:
            rows.append([block])
            continue
        prev_row = rows[-1]
        prev_cy = sum(_box_center_y(b.get("box") or []) for b in prev_row) / len(prev_row)
        if abs(cy - prev_cy) <= y_tol:
            prev_row.append(block)
        else:
            rows.append([block])
    return rows


def _merge_blocks_in_cell(blocks: list[dict]) -> dict:
    blocks = sorted(blocks, key=lambda b: _box_xrange(b.get("box") or [])[0])
    text = _normalize_text(" ".join(_normalize_text(b.get("text", "")) for b in blocks))
    score = sum(float(b.get("score") or 0.0) for b in blocks) / max(1, len(blocks))
    return {"text": text, "score": float(score), "box": blocks[0].get("box")}


def _derive_col_boundaries(
    header_blocks: list[dict],
    num_cols: int,
    all_blocks: list[dict],
    col_boundaries: list[float] | None = None,
) -> list[float]:
    if col_boundaries and len(col_boundaries) == num_cols + 1:
        return col_boundaries

    xs = [_box_xrange(b.get("box") or []) for b in all_blocks if b.get("box")]
    if xs:
        x_min = min(x[0] for x in xs)
        x_max = max(x[1] for x in xs)
    else:
        x_min, x_max = 0.0, float(num_cols * 100)

    if not header_blocks:
        width = max(x_max - x_min, 1.0)
        step = width / num_cols
        return [x_min + step * i for i in range(num_cols + 1)]

    header_blocks = sorted(header_blocks, key=lambda b: _box_center_x(b.get("box") or []))
    med_h = _box_height(header_blocks[0].get("box") or []) or 12.0
    gap_tol = max(10.0, med_h * 1.5)

    groups: list[list[dict]] = [[header_blocks[0]]]
    for block in header_blocks[1:]:
        prev_x1 = _box_xrange(groups[-1][-1].get("box") or [])[1]
        cur_x0 = _box_xrange(block.get("box") or [])[0]
        if (cur_x0 - prev_x1) <= gap_tol:
            groups[-1].append(block)
        else:
            groups.append([block])

    if len(groups) == num_cols:
        lefts = [_box_xrange(_merge_blocks_in_cell(g)["box"] or [])[0] for g in groups]
        rights = [_box_xrange(_merge_blocks_in_cell(g)["box"] or [])[1] for g in groups]
        boundaries = [x_min]
        for i in range(num_cols - 1):
            boundaries.append((rights[i] + lefts[i + 1]) / 2.0)
        boundaries.append(x_max)
        return boundaries

    width = max(x_max - x_min, 1.0)
    step = width / num_cols
    return [x_min + step * i for i in range(num_cols + 1)]


def _col_index(cx: float, boundaries: list[float]) -> int:
    for i in range(len(boundaries) - 1):
        if cx < boundaries[i + 1]:
            return i
    return len(boundaries) - 2


def build_table(
    raw_blocks: list[dict],
    *,
    num_cols: int,
    header_row: int = 0,
    col_boundaries: list[float] | None = None,
) -> dict[str, Any]:
    """Map OCR blocks onto a fixed-column grid using box coordinates."""
    if num_cols < 1:
        raise ValueError("num_cols must be >= 1")

    rows = _cluster_rows(raw_blocks)
    if not rows:
        return {
            "cols": num_cols,
            "header_row": header_row,
            "row_count": 0,
            "headers": [""] * num_cols,
            "data": [],
            "cells": [],
            "col_boundaries": _derive_col_boundaries([], num_cols, [], col_boundaries),
        }

    boundaries = _derive_col_boundaries(
        rows[header_row] if header_row < len(rows) else [],
        num_cols,
        raw_blocks,
        col_boundaries,
    )

    grid: list[list[list[dict]]] = [
        [[] for _ in range(num_cols)] for _ in range(len(rows))
    ]
    for r_idx, row_blocks in enumerate(rows):
        for block in row_blocks:
            cx = _box_center_x(block.get("box") or [])
            c_idx = _col_index(cx, boundaries)
            c_idx = max(0, min(num_cols - 1, c_idx))
            grid[r_idx][c_idx].append(block)

    table_rows: list[list[str]] = []
    table_scores: list[list[float]] = []
    cells: list[dict] = []

    for r_idx, row_cells in enumerate(grid):
        row_texts: list[str] = []
        row_scores: list[float] = []
        for c_idx, cell_blocks in enumerate(row_cells):
            if cell_blocks:
                merged = _merge_blocks_in_cell(cell_blocks)
                text = merged["text"]
                score = merged["score"]
            else:
                text = ""
                score = 0.0
            row_texts.append(text)
            row_scores.append(score)
            cells.append(
                {
                    "row": r_idx,
                    "col": c_idx,
                    "text": text,
                    "score": round(score, 4),
                }
            )
        table_rows.append(row_texts)
        table_scores.append(row_scores)

    headers = table_rows[header_row] if header_row < len(table_rows) else [""] * num_cols
    data = [row for i, row in enumerate(table_rows) if i != header_row]

    return {
        "cols": num_cols,
        "header_row": header_row,
        "row_count": len(data),
        "headers": headers,
        "data": data,
        "cells": cells,
        "col_boundaries": [round(x, 2) for x in boundaries],
    }


def table_to_text(table: dict[str, Any]) -> str:
    """Render table as tab-separated text for plain-text consumers."""
    lines = ["\t".join(table.get("headers") or [])]
    for row in table.get("data") or []:
        lines.append("\t".join(row))
    return "\n".join(lines)
