"""Build fixed-column table JSON from PaddleOCR raw blocks (box coordinates)."""

from __future__ import annotations

import re
from typing import Any

# Rightmost columns often hold amount / owner — empty on wrapped continuation lines.
_ANCHOR_COL_COUNT = 2
_BLANK_CELL = frozenset({"", "-", "―", "－"})


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


def _box_yrange(box: list) -> tuple[float, float]:
    ys = [float(p[1]) for p in box if isinstance(p, (list, tuple)) and len(p) >= 2]
    if not ys:
        return 0.0, 0.0
    return min(ys), max(ys)


def _blocks_y_extent(blocks: list[dict]) -> tuple[float, float]:
    y0s: list[float] = []
    y1s: list[float] = []
    for b in blocks:
        lo, hi = _box_yrange(b.get("box") or [])
        y0s.append(lo)
        y1s.append(hi)
    if not y0s:
        return 0.0, 0.0
    return min(y0s), max(y1s)


def _filled_cols(row: list[str]) -> set[int]:
    out: set[int] = set()
    for i, val in enumerate(row):
        t = (val or "").strip()
        if t and t not in _BLANK_CELL:
            out.add(i)
    return out


def _has_anchor_column_values(row: list[str], num_cols: int) -> bool:
    """True when right-side columns look like a complete row (amount and/or owner)."""
    start = max(0, num_cols - _ANCHOR_COL_COUNT)
    for c in range(start, num_cols):
        t = (row[c] if c < len(row) else "").strip()
        if not t or t in _BLANK_CELL:
            continue
        if re.search(r"\d", t):
            return True
        if re.search(r"[가-힣]{2,}", t):
            return True
    return False


def _median_row_height(row_groups: list[list[dict]]) -> float:
    heights: list[float] = []
    for group in row_groups:
        y0, y1 = _blocks_y_extent(group)
        if y1 > y0:
            heights.append(y1 - y0)
    if not heights:
        return 14.0
    heights.sort()
    return heights[len(heights) // 2]


def should_merge_continuation_rows(
    row_above: list[str],
    row_below: list[str],
    num_cols: int,
    *,
    allow_vertical_gap: bool = True,
    vertical_gap: float | None = None,
    median_h: float = 14.0,
) -> bool:
    """True when *row_below* is a wrapped continuation of the same logical table row."""
    filled_below = _filled_cols(row_below)
    if not filled_below:
        return False

    if not allow_vertical_gap and vertical_gap is not None:
        if median_h > 0 and vertical_gap > median_h * 2.25:
            return False
    elif vertical_gap is not None and median_h > 0:
        if vertical_gap > median_h * 3.0:
            return False

    if _has_anchor_column_values(row_below, num_cols):
        return False

    filled_above = _filled_cols(row_above)
    if not filled_above:
        return False

    anchor_start = max(0, num_cols - _ANCHOR_COL_COUNT)
    below_body = {c for c in filled_below if c < anchor_start}
    if not below_body:
        return False

    # Same filled columns = two complete records, not a wrap continuation.
    if filled_below == filled_above:
        return False

    if filled_below < filled_above:
        return True

    if max(filled_below) <= max(filled_above) + 2 and below_body:
        return True

    # Single continuation column (long 구조 및 규격 wrap).
    if len(filled_below) == 1 and next(iter(filled_below)) in filled_above:
        return True

    return False


def _should_merge_wrap_continuation(
    row_above: list[str],
    row_below: list[str],
    blocks_above: list[dict],
    blocks_below: list[dict],
    num_cols: int,
    median_h: float,
) -> bool:
    from table_structure import is_fragment_continuation_row

    if is_fragment_continuation_row(row_above, row_below, num_cols):
        return True
    gap = _blocks_y_extent(blocks_below)[0] - _blocks_y_extent(blocks_above)[1]
    return should_merge_continuation_rows(
        row_above,
        row_below,
        num_cols,
        allow_vertical_gap=False,
        vertical_gap=gap,
        median_h=median_h,
    )


def _merge_row_pair(
    row_above: list[str],
    row_below: list[str],
    grid_above: list[list[dict]],
    grid_below: list[list[dict]],
    num_cols: int,
) -> tuple[list[str], list[list[dict]]]:
    merged_row: list[str] = []
    merged_grid: list[list[dict]] = []
    for c in range(num_cols):
        a = row_above[c] if c < len(row_above) else ""
        b = row_below[c] if c < len(row_below) else ""
        if a.strip() and b.strip():
            merged_row.append(_normalize_text(f"{a} {b}"))
        else:
            merged_row.append(a.strip() or b.strip())
        merged_grid.append((grid_above[c] if c < len(grid_above) else []) + (
            grid_below[c] if c < len(grid_below) else []
        ))
    return merged_row, merged_grid


def _merge_wrapped_table_rows(
    table_rows: list[list[str]],
    grid: list[list[list[dict]]],
    row_block_groups: list[list[dict]],
    num_cols: int,
) -> tuple[list[list[str]], list[list[list[dict]]], list[list[dict]]]:
    """Merge OCR rows that are vertical wraps within one printed table row."""
    if len(table_rows) < 2:
        return table_rows, grid, row_block_groups

    median_h = _median_row_height(row_block_groups)
    merged_rows: list[list[str]] = [list(table_rows[0])]
    merged_grid: list[list[list[dict]]] = [
        [list(cell) for cell in grid[0]],
    ]
    merged_groups: list[list[dict]] = [list(row_block_groups[0])]

    for i in range(1, len(table_rows)):
        if _should_merge_wrap_continuation(
            merged_rows[-1],
            table_rows[i],
            merged_groups[-1],
            row_block_groups[i],
            num_cols,
            median_h,
        ):
            merged_rows[-1], merged_grid[-1] = _merge_row_pair(
                merged_rows[-1],
                table_rows[i],
                merged_grid[-1],
                grid[i],
                num_cols,
            )
            merged_groups[-1] = merged_groups[-1] + row_block_groups[i]
        else:
            merged_rows.append(list(table_rows[i]))
            merged_grid.append([list(cell) for cell in grid[i]])
            merged_groups.append(list(row_block_groups[i]))

    return merged_rows, merged_grid, merged_groups


def _preview_texts_for_row(row_blocks: list[dict]) -> list[str]:
    ordered = sorted(
        row_blocks,
        key=lambda b: _box_center_x(b.get("box") or []),
    )
    return [_normalize_text(b.get("text", "")) for b in ordered if b.get("text")]


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
    # Multiline cell: top-to-bottom, then left-to-right within a line.
    blocks = sorted(
        blocks,
        key=lambda b: (
            _box_center_y(b.get("box") or []),
            _box_xrange(b.get("box") or [])[0],
        ),
    )
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


def _merge_header_cells(table_rows: list[list[str]], col: int, n_header: int) -> str:
    """Merge the first *n_header* grid rows into one header label per column."""
    parts: list[str] = []
    for r in range(min(n_header, len(table_rows))):
        if col >= len(table_rows[r]):
            continue
        t = _normalize_text(table_rows[r][col])
        if t and t not in parts:
            parts.append(t)
    return " ".join(parts)


def export_table_aliases(table: dict[str, Any]) -> dict[str, Any]:
    """Add ``rows`` / ``rows_refined`` keys for Django decision_tasks consumers."""
    all_rows = table.get("all_rows")
    header_rows = int(table.get("header_rows", 1))

    if all_rows:
        table["rows"] = [list(r) for r in all_rows]
    else:
        headers = list(table.get("headers") or [])
        data = table.get("data") or []
        table["rows"] = [headers] + [list(r) for r in data]

    data_refined = table.get("data_refined")
    if data_refined is not None:
        if all_rows:
            table["rows_refined"] = [list(r) for r in all_rows[:header_rows]] + [
                list(r) for r in data_refined
            ]
        else:
            headers = list(table.get("headers") or [])
            table["rows_refined"] = [headers] + [list(r) for r in data_refined]
    return table


def build_table(
    raw_blocks: list[dict],
    *,
    num_cols: int,
    header_row: int = 0,
    header_rows: int = 1,
    col_boundaries: list[float] | None = None,
) -> dict[str, Any]:
    """Map OCR blocks onto a fixed-column grid using box coordinates."""
    if num_cols < 1:
        raise ValueError("num_cols must be >= 1")

    n_header = max(1, header_rows)
    rows = _cluster_rows(raw_blocks)
    if rows:
        from table_structure import find_table_region_start

        preview_rows = [_preview_texts_for_row(g) for g in rows]
        start = find_table_region_start(preview_rows, n_header)
        if start > 0:
            rows = rows[start:]
    if not rows:
        return {
            "cols": num_cols,
            "header_row": header_row,
            "header_rows": n_header,
            "row_count": 0,
            "headers": [""] * num_cols,
            "data": [],
            "all_rows": [],
            "cells": [],
            "col_boundaries": _derive_col_boundaries([], num_cols, [], col_boundaries),
        }

    hdr_blocks: list[dict] = []
    for r in range(min(n_header, len(rows))):
        hdr_blocks.extend(rows[r])
    boundaries = _derive_col_boundaries(
        hdr_blocks,
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
    for row_cells in grid:
        row_texts: list[str] = []
        for cell_blocks in row_cells:
            if cell_blocks:
                row_texts.append(_merge_blocks_in_cell(cell_blocks)["text"])
            else:
                row_texts.append("")
        table_rows.append(row_texts)

    if n_header < len(table_rows):
        head_rows = table_rows[:n_header]
        body_rows = table_rows[n_header:]
        body_grid = grid[n_header:]
        body_groups = rows[n_header:]
        body_rows, body_grid, body_groups = _merge_wrapped_table_rows(
            body_rows, body_grid, body_groups, num_cols
        )
        table_rows = head_rows + body_rows
        grid = grid[:n_header] + body_grid
        rows = rows[:n_header] + body_groups
    else:
        table_rows, grid, rows = _merge_wrapped_table_rows(
            table_rows, grid, rows, num_cols
        )

    cells: list[dict] = []
    for r_idx, row_cells in enumerate(grid):
        row_texts: list[str] = []
        for c_idx, cell_blocks in enumerate(row_cells):
            if cell_blocks:
                merged = _merge_blocks_in_cell(cell_blocks)
                text = merged["text"]
                score = merged["score"]
            else:
                text = ""
                score = 0.0
            row_texts.append(text)
            cells.append(
                {
                    "row": r_idx,
                    "col": c_idx,
                    "text": text,
                    "score": round(score, 4),
                }
            )
        table_rows[r_idx] = row_texts

    if n_header > 1:
        headers = [
            _merge_header_cells(table_rows, c, n_header) for c in range(num_cols)
        ]
        data = table_rows[n_header:] if len(table_rows) > n_header else []
    else:
        headers = (
            table_rows[header_row] if header_row < len(table_rows) else [""] * num_cols
        )
        data = [row for i, row in enumerate(table_rows) if i != header_row]

    result = {
        "cols": num_cols,
        "header_row": header_row,
        "header_rows": n_header,
        "row_count": len(data),
        "headers": headers,
        "data": data,
        "all_rows": table_rows,
        "cells": cells,
        "col_boundaries": [round(x, 2) for x in boundaries],
    }
    from table_structure import apply_rule_structure_fix

    result, _ = apply_rule_structure_fix(result)
    return result


def table_to_text(table: dict[str, Any]) -> str:
    """Render table as tab-separated text for plain-text consumers."""
    lines = ["\t".join(table.get("headers") or [])]
    for row in table.get("data") or []:
        lines.append("\t".join(row))
    return "\n".join(lines)
