"""Table structure cleanup: duplicate headers, wrapped rows, optional vLLM fix."""

from __future__ import annotations

import re
from typing import Any

from table_builder import (
    _BLANK_CELL,
    _filled_cols,
    _has_anchor_column_values,
    _merge_row_pair,
    _normalize_text,
    should_merge_continuation_rows,
)

_HEADER_KW = frozenset(
    {
        "소재지",
        "구조",
        "규격",
        "금액",
        "단가",
        "보상액",
        "수량",
        "면적",
        "소유자",
        "지번",
        "구분",
        "지상권",
        "비고",
        "순번",
        "번호",
        "성명",
        "주소",
        "권리",
    }
)

# Embedded column labels OCR often prepends inside body cells (재결서 보상금내역)
_CELL_LABEL_PREFIXES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"^소재지\s*"), ""),
    (re.compile(r"^지번\s*"), ""),
    (re.compile(r"^수량\s*"), ""),
    (re.compile(r"^물건의\s*종류\s*"), ""),
    (re.compile(r"^면적\s*\(m²\)\s*"), ""),
    (re.compile(r"^면적\s*"), ""),
    (re.compile(r"^구조\s*및\s*규격\s*"), ""),
    (re.compile(r"^구조및규격\s*"), ""),
    (re.compile(r"^단가\s*/\s*"), ""),
    (re.compile(r"^단가\s*"), ""),
    (re.compile(r"^금액\s*"), ""),
    (re.compile(r"^성명\s*"), ""),
    (re.compile(r"^소유자\s*"), ""),
    (re.compile(r"^주소\s*"), ""),
    (re.compile(r"^권리의\s*종류\s*"), ""),
    (re.compile(r"^권리\s*"), ""),
)

_OWNER_MARKER = re.compile(
    r"(?:주식회사)?[가-힣][가-힣A-Za-z0-9]*\([^)]{2,40}\)"
)
_AMOUNT_RE = re.compile(r"\d{1,3}(?:,\d{3})+")
_REGION_SPLIT = re.compile(
    r"(?=경기도|서울특별시|충청|전라|강원|제주|인천|부산|대구|대전|광주|울산|세종)"
)
_FRAGMENT_COL0 = frozenset({"동", "리", "읍", "면", "가", "로"})

_PAGE_CHROME_RE = re.compile(
    r"(?:Page\s*\d|별지\s*제?\s*\d|^\[별지)|\d+\s*/\s*\d+\s*$",
    re.IGNORECASE,
)

# 재결서 보상금내역 9열 (Django table_cols=9)
_CANONICAL_HEADERS_9 = [
    "소재지",
    "지번",
    "구조 및 규격",
    "수량/면적",
    "단가",
    "금액",
    "성명",
    "권리의 종류",
    "주소",
]

_HEADER_CELL_PATTERNS: tuple[tuple[re.Pattern[str], int], ...] = (
    (re.compile(r"^소재지"), 2),
    (re.compile(r"^지번"), 2),
    (re.compile(r"구분지상권"), 1),
    (re.compile(r"구조|규격"), 1),
    (re.compile(r"수량|면적"), 1),
    (re.compile(r"단가"), 1),
    (re.compile(r"^금액|보상액"), 1),
    (re.compile(r"^성명|소유자"), 1),
    (re.compile(r"권리"), 1),
    (re.compile(r"^주소"), 1),
)


def canonical_headers(num_cols: int) -> list[str]:
    if num_cols == len(_CANONICAL_HEADERS_9):
        return list(_CANONICAL_HEADERS_9)
    return [f"열{i + 1}" for i in range(num_cols)]


def score_header_candidate_row(row: list[str]) -> int:
    """Higher = more likely a real table header line (not page title)."""
    score = 0
    for cell in row:
        compact = _normalize_text(cell).replace(" ", "")
        if not compact or len(compact) > 32:
            continue
        if _PAGE_CHROME_RE.search(cell):
            continue
        for pat, pts in _HEADER_CELL_PATTERNS:
            if pat.search(compact):
                score += pts
                break
    return score


def find_table_region_start(table_rows: list[list[str]], n_header: int) -> int:
    """Skip page title / 별지 lines above the real column header row(s)."""
    if len(table_rows) <= n_header:
        return 0

    best_start = 0
    best_score = -1
    for i in range(0, len(table_rows) - n_header + 1):
        window = sum(score_header_candidate_row(table_rows[i + j]) for j in range(n_header))
        if window > best_score:
            best_score = window
            best_start = i

    if best_score < n_header + 1:
        return 0
    return best_start


def is_page_chrome_text(text: str) -> bool:
    t = _normalize_text(text)
    if not t:
        return False
    if _PAGE_CHROME_RE.search(t):
        return True
    if "별지" in t and len(t) > 15:
        return True
    if re.search(r"건설사업|수용\d+|차\(\d+", t) and "소재지" not in t[:20]:
        return True
    return False


def is_invalid_headers(headers: list[str]) -> bool:
    """Headers contaminated with page chrome instead of column labels."""
    non_empty = [h for h in headers if _normalize_text(h)]
    if not non_empty:
        return True

    joined = " ".join(headers)
    if _PAGE_CHROME_RE.search(joined):
        return True
    if any(is_page_chrome_text(h) for h in non_empty):
        return True
    if any(len(_normalize_text(h)) > 45 for h in non_empty):
        return True

    short_kw = sum(1 for h in non_empty if score_header_candidate_row([h]) > 0)
    if short_kw >= 2:
        return False
    # Short placeholders (tests, simple tables) are acceptable
    if all(len(_normalize_text(h)) <= 10 for h in non_empty) and len(non_empty) >= 2:
        return False
    return short_kw < 1


def normalize_table_headers(table: dict[str, Any]) -> dict[str, Any]:
    headers = list(table.get("headers") or [])
    num_cols = int(table.get("cols") or len(headers))
    if not is_invalid_headers(headers):
        return table
    fixed = canonical_headers(num_cols)
    header_rows = int(table.get("header_rows", 1))
    data = [list(r) for r in table.get("data") or []]
    out = dict(table)
    out["headers"] = fixed
    if header_rows <= 1:
        out["header_rows"] = 1
        out["all_rows"] = [list(fixed)] + data
    else:
        hdr_part = [list(fixed)]
        while len(hdr_part) < header_rows:
            hdr_part.append([""] * num_cols)
        out["all_rows"] = hdr_part + data
    return out


def row_has_embedded_header_labels(row: list[str]) -> bool:
    label_cols = 0
    for cell in row:
        c = _normalize_text(cell)
        if not c:
            continue
        if any(
            c.startswith(kw)
            for kw in (
                "소재지 ",
                "소재지",
                "지번 ",
                "수량 ",
                "구조",
                "단가",
                "금액 ",
                "성명 ",
                "주소 ",
                "권리",
            )
        ) and any(kw in c[:12] for kw in ("소재지", "지번", "수량", "단가", "성명")):
            label_cols += 1
    return label_cols >= 2


def _header_tokens(headers: list[str]) -> set[str]:
    tokens: set[str] = set()
    for h in headers:
        h = _normalize_text(h)
        if len(h) >= 2:
            tokens.add(h)
        for part in re.split(r"[\s/·]+", h):
            part = part.strip()
            if len(part) >= 2:
                tokens.add(part)
    return tokens


def strip_cell_header_pollution(cell: str, col: int, headers: list[str]) -> str:
    """Remove OCR column-title text accidentally fused into a body cell."""
    s = _normalize_text(cell)
    if not s:
        return s

    hdr = _normalize_text(headers[col] if col < len(headers) else "")
    if hdr and len(hdr) >= 2 and s.startswith(hdr):
        s = s[len(hdr) :].strip()

    for pat, repl in _CELL_LABEL_PREFIXES:
        s = pat.sub(repl, s).strip()

    for tok in _header_tokens(headers):
        if len(tok) >= 4 and s.startswith(tok + " "):
            s = s[len(tok) :].strip()

    return _normalize_text(s)


def strip_row_header_pollution(row: list[str], headers: list[str]) -> list[str]:
    return [
        strip_cell_header_pollution(row[c] if c < len(row) else "", c, headers)
        for c in range(max(len(row), len(headers)))
    ]


def _cell_matches_header(cell: str, headers: list[str], col: int) -> bool:
    t = _normalize_text(cell)
    if not t or t in _BLANK_CELL:
        return False
    hdr = _normalize_text(headers[col] if col < len(headers) else "")
    if hdr and (t == hdr or t in hdr or hdr in t):
        return True
    if t in _HEADER_KW:
        return True
    for tok in _header_tokens(headers):
        if len(tok) >= 3 and (t == tok or tok in t or t in tok):
            return True
    return False


def is_duplicate_header_row(row: list[str], headers: list[str]) -> bool:
    """True when a body row mostly repeats column header labels."""
    filled = [(i, (row[i] if i < len(row) else "").strip()) for i in range(len(headers))]
    filled = [(i, t) for i, t in filled if t and t not in _BLANK_CELL]
    if len(filled) < 2:
        return False

    header_like = sum(1 for i, t in filled if _cell_matches_header(t, headers, i))
    if header_like >= max(2, int(len(filled) * 0.55)):
        return True

    row_joined = " ".join(t for _, t in filled)
    hdr_joined = " ".join(_normalize_text(h) for h in headers if h.strip())
    if hdr_joined and row_joined == hdr_joined:
        return True
    return False


def count_owner_markers(text: str) -> int:
    return len(_OWNER_MARKER.findall(text or ""))


def count_amount_markers(text: str) -> int:
    return len(_AMOUNT_RE.findall(text or ""))


def is_mega_row(row: list[str], num_cols: int) -> bool:
    owners = sum(count_owner_markers(row[c] if c < len(row) else "") for c in range(num_cols))
    amounts = sum(count_amount_markers(row[c] if c < len(row) else "") for c in range(num_cols))
    max_len = max(len((row[c] if c < len(row) else "") or "") for c in range(num_cols))
    return owners >= 2 or amounts >= 4 or max_len > 350


def is_collapsed_table_row(row: list[str], headers: list[str], num_cols: int) -> bool:
    """Whole table OCR-fused into one row (header labels + many owners)."""
    label_cols = 0
    for c in range(num_cols):
        cell = row[c] if c < len(row) else ""
        if any(kw in cell for kw in ("소재지", "지번", "수량", "구조", "단가", "성명", "주소", "권리")):
            label_cols += 1
    owners = sum(count_owner_markers(row[c] if c < len(row) else "") for c in range(num_cols))
    amounts = sum(count_amount_markers(row[c] if c < len(row) else "") for c in range(num_cols))
    if label_cols >= 3 and (owners >= 2 or amounts >= 2):
        return True
    return label_cols >= 2 and is_mega_row(row, num_cols)


def _owner_column(row: list[str], num_cols: int) -> int:
    best_col, best = -1, 0
    for c in range(num_cols):
        n = count_owner_markers(row[c] if c < len(row) else "")
        if n > best:
            best, best_col = n, c
    return best_col if best >= 2 else -1


def _split_by_n_markers(text: str, n: int, pattern: re.Pattern[str]) -> list[str]:
    matches = list(pattern.finditer(text or ""))
    if len(matches) < n:
        return []
    parts: list[str] = []
    for i in range(n):
        start = matches[i].start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        parts.append(_normalize_text(text[start:end]))
    return parts


def _split_cell_into_n(
    text: str,
    n: int,
    col: int,
    *,
    owner_col: int,
    amount_col: int,
) -> list[str]:
    text = text or ""
    if n <= 1:
        return [_normalize_text(text)]

    if col == owner_col:
        parts = _split_by_n_markers(text, n, _OWNER_MARKER)
        if parts:
            return parts

    if col == amount_col:
        amounts = _AMOUNT_RE.findall(text)
        if len(amounts) >= n:
            return amounts[:n]

    if col in (5, 6, 7, 8) or "등)" in text or "(등)" in text:
        parts = re.split(r"(?=\(등\))", text)
        parts = [_normalize_text(p) for p in parts if p.strip()]
        if len(parts) >= n:
            return parts[:n]

    region_parts = [p for p in _REGION_SPLIT.split(text) if p.strip()]
    if len(region_parts) >= n:
        return [_normalize_text(p) for p in region_parts[:n]]

    if col == 0 and "동" in text:
        parts = re.split(r"(?=동\s)", text)
        parts = [_normalize_text(p) for p in parts if p.strip()]
        if len(parts) >= n:
            return parts[:n]

    if col == 1:
        parts = re.split(r"(?=구분지상권\s)", text)
        parts = [_normalize_text(p) for p in parts if p.strip()]
        if len(parts) >= n:
            return parts[:n]

    chunk = max(1, len(text) // n)
    return [_normalize_text(text[i : i + chunk]) for i in range(0, len(text), chunk)][:n]


def split_multivalue_rows(
    data: list[list[str]],
    num_cols: int,
    *,
    amount_col: int = 3,
) -> list[list[str]]:
    """Split rows where multiple owners/amounts were merged into one OCR line."""
    out: list[list[str]] = []
    for row in data:
        row = list(row) + [""] * max(0, num_cols - len(row))
        if not should_split_row(row, num_cols, amount_col=amount_col):
            out.append(row[:num_cols])
            continue
        owner_col = _owner_column(row, num_cols)
        n = count_owner_markers(row[owner_col] if owner_col >= 0 else "")
        if n < 2:
            n = count_amount_markers(row[amount_col] if amount_col < len(row) else "")
        if n < 2:
            out.append(row[:num_cols])
            continue

        split_rows = [[] for _ in range(n)]
        for c in range(num_cols):
            parts = _split_cell_into_n(
                row[c],
                n,
                c,
                owner_col=owner_col,
                amount_col=amount_col,
            )
            while len(parts) < n:
                parts.append("")
            for i in range(n):
                split_rows[i].append(parts[i])

        out.extend(split_rows)

    return out


def is_fragment_continuation_row(
    row_above: list[str],
    row_below: list[str],
    num_cols: int,
) -> bool:
    """Short tail lines: '동', '1387', partial spec — belong to the row above."""
    c0 = (row_below[0] if row_below else "").strip()
    above0 = (row_above[0] if row_above else "").strip()

    if c0 in _FRAGMENT_COL0 and above0 and not above0.rstrip().endswith(c0):
        return True

    c1 = (row_below[1] if len(row_below) > 1 else "").strip()
    above1 = (row_above[1] if len(row_above) > 1 else "").strip()
    below0 = (row_below[0] if row_below else "").strip()
    if (
        c1.isdigit()
        and len(c1) >= 3
        and above1
        and c1 not in above1
        and ("지상권" in above1 or "지번" in above1)
        and (below0 in _FRAGMENT_COL0 or not below0 or len(below0) <= 4)
    ):
        return True

    spec_col = 2
    below_spec = (row_below[spec_col] if spec_col < len(row_below) else "").strip()
    above_spec = (row_above[spec_col] if spec_col < len(row_above) else "").strip()
    if below_spec and (
        below_spec.startswith("대")
        or below_spec.startswith("x")
        or "지하심도" in below_spec
        or "해수면" in below_spec
    ):
        if not above_spec or below_spec not in above_spec:
            return True

    below_amt = count_amount_markers(
        row_below[3] if len(row_below) > 3 else ""
    )
    above_amt = count_amount_markers(
        row_above[3] if len(row_above) > 3 else ""
    )
    filled = _filled_cols(row_below)
    if below_amt and not above_amt and len(filled) <= 4:
        return True

    return False


def should_split_row(row: list[str], num_cols: int, *, amount_col: int = 3) -> bool:
    """Split only when one row clearly bundles multiple compensation records."""
    owner_col = _owner_column(row, num_cols)
    if owner_col < 0:
        return False
    n_owners = count_owner_markers(row[owner_col])
    if n_owners < 2:
        return False
    n_amounts = count_amount_markers(row[amount_col] if amount_col < len(row) else "")
    return n_amounts >= n_owners


def drop_empty_rows(data: list[list[str]]) -> tuple[list[list[str]], int]:
    dropped = 0
    kept: list[list[str]] = []
    for row in data:
        if not any((c or "").strip() and (c or "").strip() not in _BLANK_CELL for c in row):
            dropped += 1
            continue
        kept.append(list(row))
    return kept, dropped


def drop_repeated_header_rows(
    data: list[list[str]],
    headers: list[str],
) -> tuple[list[list[str]], int]:
    """Remove mid-table header repetitions (common on page breaks)."""
    if not data or not headers:
        return data, 0
    kept: list[list[str]] = []
    dropped = 0
    for row in data:
        if is_duplicate_header_row(row, headers):
            dropped += 1
            continue
        kept.append(list(row))
    return kept, dropped


# 9-column 재결서 보상금내역 layout (table_cols=9)
_COL_LOC = 0
_COL_JIBUN = 1
_COL_SPEC = 2
_COL_AMT = 3
_COL_OWNER = 4
_COL_ADDR = 5
_COL_RIGHTS = 6
_COL_BRANCH = 7
_COL_BANK_ADDR = 8

_BROKEN_OWNER_START = re.compile(r"^[\)\]\-,\d\s]+")
_OWNER_IN_TEXT = re.compile(
    r"(?:주식회사)?[가-힣]{2,8}[가-힣A-Za-z0-9]*\s*\([^)]+\)"
)
_OWNER_SKIP_NAME = frozenset(
    {"주식회사", "농협은행", "국민은행", "한국스탄다드", "여의도통", "서울특별시"}
)


def _pad_row(row: list[str], num_cols: int) -> list[str]:
    row = list(row)
    if len(row) < num_cols:
        row.extend([""] * (num_cols - len(row)))
    return row[:num_cols]


def _address_key(text: str) -> str:
    return re.sub(r"\s+", "", (text or ""))[:40]


def _owner_token_complete(text: str) -> bool:
    compact = re.sub(r"\s+", "", text or "")
    return bool(
        re.search(r"\(\d{2,}-\d{2,}\)", compact)
        or re.search(r"\(상가-\d+\)", compact)
    )


def record_complete(row: list[str], num_cols: int) -> bool:
    row = _pad_row(row, num_cols)
    amt = count_amount_markers(row[_COL_AMT])
    owners = _extract_owners(row[_COL_OWNER])
    if not owners or not any(_owner_token_complete(o) for o in owners):
        return False
    addr = len(row[_COL_ADDR].strip())
    loc = len(row[_COL_LOC].strip())
    return amt >= 1 and (addr >= 12 or loc >= 8)


def _merge_group_columns(rows: list[list[str]], num_cols: int) -> list[str]:
    merged = [""] * num_cols
    for row in rows:
        row = _pad_row(row, num_cols)
        for c in range(num_cols):
            a = merged[c].strip()
            b = row[c].strip()
            if not b:
                continue
            if not a:
                merged[c] = b
            elif b in a:
                continue
            elif a in b:
                merged[c] = b
            else:
                merged[c] = _normalize_text(f"{a} {b}")
    return merged


def _fix_owner_token(text: str) -> str:
    s = _normalize_text(text)
    s = _BROKEN_OWNER_START.sub("", s).strip()
    s = re.sub(
        r"\(\s*(\d+)\s+(\d+)\s*-\s*(\d+)\s*\)",
        r"(\1\2-\3)",
        s,
    )
    s = re.sub(r"\(\s*(\d+)\s*-\s*(\d+)\s*\)", r"(\1-\2)", s)
    s = re.sub(r"\(\s*상가\s*-\s*(\d+)\s*\)", r"(상가-\1)", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _is_plausible_owner_name(name: str) -> bool:
    n = name.replace(" ", "")
    if not n or n in _OWNER_SKIP_NAME:
        return False
    if "은행" in n or "지점" in n or "도통" in n:
        return False
    if len(n) > 12:
        return False
    return bool(re.match(r"^[가-힣]", n))


def _extract_owners(text: str) -> list[str]:
    owners: list[str] = []
    for m in _OWNER_IN_TEXT.finditer(text or ""):
        raw = m.group(0)
        name = re.sub(r"\([^)]*\)", "", raw).strip()
        if not _is_plausible_owner_name(name):
            continue
        t = _fix_owner_token(raw)
        if t and t not in owners:
            owners.append(t)
    broken = re.search(r"([가-힣]{2,8}[가-힣A-Za-z0-9]*)\s*\(([^)]+)$", text or "")
    if broken and _is_plausible_owner_name(broken.group(1)):
        inner = re.sub(r"\s+", "", broken.group(2))
        closed = f"{broken.group(1)}({inner})"
        t = _fix_owner_token(closed)
        if t and t not in owners:
            owners.append(t)
    return owners


def _owners_from_group(group: list[list[str]], num_cols: int) -> list[str]:
    owners: list[str] = []
    for row in group:
        row = _pad_row(row, num_cols)
        for o in _extract_owners(row[_COL_OWNER]):
            if o not in owners:
                owners.append(o)
    return owners


def _fix_location_cell(text: str) -> str:
    s = re.sub(r"\s+", " ", (text or "").strip())
    if not s:
        return s
    s = re.sub(r"수원시\s*팔달", "수원시 팔달", s)
    s = re.sub(r"팔달\s*구", "팔달구", s)
    s = re.sub(r"팔달구\s*우만", "팔달구 우만", s)
    s = re.sub(r"우만\s*동", "우만동", s)
    if "우만" in s and "우만동" not in s and "동" in s:
        s = s.replace("우만 동", "우만동")
    return _normalize_text(s)


def _fix_jibun_cell(text: str) -> str:
    s = _normalize_text(text)
    if not s:
        return s
    if "구분지상권" in s.replace(" ", ""):
        nums = re.findall(r"\d+", s)
        if nums:
            return _normalize_text("구분지상권 " + " ".join(nums[:3]))
        return "구분지상권"
    return s


def _fix_amount_cell(text: str) -> str:
    amounts = _AMOUNT_RE.findall(text or "")
    if not amounts:
        return _normalize_text(text)
    return " ".join(amounts[:2])


def _fix_address_cell(text: str) -> str:
    s = re.sub(r"\s+", " ", (text or "").strip())
    s = re.sub(r"\(\s*등\s*\)", "(등)", s)
    s = re.sub(r"\(\s*등\)", "(등)", s)
    s = re.sub(r",\s*", ", ", s)
    return _normalize_text(s)


def normalize_compensation_record(row: list[str], num_cols: int = 9) -> list[str]:
    """Normalize one merged compensation row (재결서 9열)."""
    row = _pad_row(row, num_cols)
    owners = _extract_owners(row[_COL_OWNER])
    owner_text = owners[0] if owners else _fix_owner_token(row[_COL_OWNER])

    rights = row[_COL_RIGHTS].strip()
    if not rights and "은행" in row[_COL_OWNER] and not owners:
        owner_text = ""
        rights = row[_COL_OWNER]

    out = list(row)
    out[_COL_LOC] = _fix_location_cell(row[_COL_LOC])
    out[_COL_JIBUN] = _fix_jibun_cell(row[_COL_JIBUN])
    out[_COL_SPEC] = _normalize_text(row[_COL_SPEC])
    out[_COL_AMT] = _fix_amount_cell(row[_COL_AMT])
    out[_COL_OWNER] = owner_text
    out[_COL_ADDR] = _fix_address_cell(row[_COL_ADDR])
    out[_COL_RIGHTS] = _normalize_text(rights or row[_COL_RIGHTS])
    branch = _normalize_text(row[_COL_BRANCH] + " " + row[_COL_RIGHTS])
    branch = re.sub(r"근저당권\s*", "", branch).strip()
    if "지점" in branch and len(branch) < 25:
        out[_COL_BRANCH] = ""
        if "근저당권" not in out[_COL_RIGHTS]:
            out[_COL_RIGHTS] = _normalize_text(
                (out[_COL_RIGHTS] + " 근저당권 " + branch).strip()
            )
    else:
        out[_COL_BRANCH] = ""
    out[_COL_BANK_ADDR] = _fix_address_cell(row[_COL_BANK_ADDR])
    return out


def _is_compensation_continuation(
    row: list[str],
    merged: list[str],
    num_cols: int,
) -> bool:
    row = _pad_row(row, num_cols)
    merged = _pad_row(merged, num_cols)

    if is_fragment_continuation_row(merged, row, num_cols):
        return True

    c0 = row[_COL_LOC].strip()
    c4 = row[_COL_OWNER].strip()
    if c4.startswith(")") or c4.startswith("(") and not _extract_owners(c4):
        return True
    if _BROKEN_OWNER_START.match(c4) and not record_complete(merged, num_cols):
        return True

    # Same parcel, record not complete yet
    if not record_complete(merged, num_cols):
        if c0 and not c0.startswith("경기도") and (
            "우만" in c0 or "팔달" in c0 or c0 in _FRAGMENT_COL0
        ):
            return True
        if any(row[c].strip() for c in range(num_cols)):
            return True

    if _address_key(c0) and _address_key(c0) == _address_key(merged[_COL_LOC]):
        if not record_complete(merged, num_cols):
            return True

    return False


def _starts_new_compensation_record(
    row: list[str],
    merged: list[str],
    num_cols: int,
) -> bool:
    if not merged or not record_complete(merged, num_cols):
        return False
    if _is_compensation_continuation(row, merged, num_cols):
        return False
    row = _pad_row(row, num_cols)
    if row[_COL_LOC].startswith("경기도") or row[_COL_LOC].startswith("서울"):
        return True
    return count_amount_markers(row[_COL_AMT]) > 0 and bool(
        _extract_owners(row[_COL_OWNER]) or row[_COL_OWNER].strip()
    )


def reassemble_fragmented_records(
    data: list[list[str]],
    num_cols: int,
) -> tuple[list[list[str]], int]:
    """
    Merge OCR-shattered rows into one row per compensation record (9열 재결서).
    Returns (new_data, group_count).
    """
    if num_cols != 9 or len(data) < 2:
        return data, 0

    groups: list[list[list[str]]] = []
    current: list[list[str]] = []
    merged: list[str] | None = None

    for raw in data:
        row = _pad_row(raw, num_cols)
        if not any(c.strip() for c in row):
            continue

        if current and _starts_new_compensation_record(row, merged or [], num_cols):
            groups.append(current)
            current = []
            merged = None

        if current and merged and not _is_compensation_continuation(row, merged, num_cols):
            groups.append(current)
            current = []
            merged = None

        current.append(row)
        merged = _merge_group_columns(current, num_cols)

    if current:
        groups.append(current)

    out: list[list[str]] = []
    for group in groups:
        merged_row = _merge_group_columns(group, num_cols)
        owners = _owners_from_group(group, num_cols)
        if len(owners) > 1 and len(group) >= 2:
            shared = normalize_compensation_record(
                _merge_group_columns(group, num_cols), num_cols
            )
            for i, owner in enumerate(owners):
                src = _pad_row(group[min(i, len(group) - 1)], num_cols)
                row = list(shared)
                row[_COL_OWNER] = _fix_owner_token(owner)
                for c in (_COL_ADDR, _COL_RIGHTS, _COL_BRANCH, _COL_BANK_ADDR):
                    if src[c].strip():
                        row[c] = src[c]
                out.append(normalize_compensation_record(row, num_cols))
        elif len(owners) > 1:
            base = normalize_compensation_record(merged_row, num_cols)
            for owner in owners:
                row = list(base)
                row[_COL_OWNER] = _fix_owner_token(owner)
                out.append(row)
        else:
            out.append(normalize_compensation_record(merged_row, num_cols))

    return out, len(groups)


def drop_collapsed_rows(
    data: list[list[str]],
    headers: list[str],
    num_cols: int,
) -> tuple[list[list[str]], int]:
    """Drop rows where OCR fused headers + many records into one line."""
    kept: list[list[str]] = []
    dropped = 0
    for row in data:
        if is_collapsed_table_row(row, headers, num_cols):
            dropped += 1
            continue
        kept.append(list(row))
    return kept, dropped


def merge_wrapped_data_rows(
    data: list[list[str]],
    num_cols: int,
) -> list[list[str]]:
    """Second-pass row merge using text layout only (after header strip)."""
    if len(data) < 2:
        return data

    merged: list[list[str]] = [list(data[0])]
    for row in data[1:]:
        if is_fragment_continuation_row(
            merged[-1], row, num_cols
        ) or should_merge_continuation_rows(
            merged[-1], row, num_cols, allow_vertical_gap=True
        ):
            merged[-1], _ = _merge_row_pair(merged[-1], row, [], [], num_cols)
        else:
            merged.append(list(row))
    return merged


def detect_structure_issues(table: dict[str, Any]) -> dict[str, Any]:
    headers = list(table.get("headers") or [])
    data = [list(r) for r in table.get("data") or []]
    num_cols = int(table.get("cols") or len(headers) or 0)

    invalid_headers = is_invalid_headers(headers)
    dup_headers = sum(1 for row in data if is_duplicate_header_row(row, headers))
    orphan_continuations = 0
    for i in range(1, len(data)):
        if should_merge_continuation_rows(
            data[i - 1], data[i], num_cols, allow_vertical_gap=True
        ) or is_fragment_continuation_row(data[i - 1], data[i], num_cols):
            orphan_continuations += 1

    mega_rows = sum(1 for row in data if is_mega_row(row, num_cols))
    collapsed = sum(
        1 for row in data if is_collapsed_table_row(row, headers, num_cols)
    )
    polluted = sum(1 for row in data if row_has_embedded_header_labels(row))

    needs_fix = (
        invalid_headers
        or dup_headers > 0
        or orphan_continuations > 0
        or mega_rows > 0
        or collapsed > 0
        or polluted > 0
    )
    return {
        "invalid_headers": invalid_headers,
        "duplicate_header_rows": dup_headers,
        "merge_candidate_rows": orphan_continuations,
        "mega_rows": mega_rows,
        "collapsed_rows": collapsed,
        "polluted_label_rows": polluted,
        "needs_fix": needs_fix,
    }


def apply_rule_structure_fix(table: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Deterministic structure cleanup on ``data`` / ``all_rows`` / ``cells``."""
    headers = list(table.get("headers") or [])
    data = [list(r) for r in table.get("data") or []]
    num_cols = int(table.get("cols") or len(headers) or 0)
    header_rows = int(table.get("header_rows", 1))

    meta: dict[str, Any] = {
        "dropped_header_rows": 0,
        "dropped_collapsed_rows": 0,
        "merged_wrap_rows": 0,
        "split_multivalue_rows": 0,
        "stripped_cell_labels": True,
    }
    before = len(data)

    was_invalid_headers = is_invalid_headers(headers)
    table = normalize_table_headers(table)
    headers = list(table.get("headers") or [])
    meta["headers_normalized"] = was_invalid_headers

    # Drop fused rows before strip (strip removes labels used for detection)
    data, collapsed = drop_collapsed_rows(data, headers, num_cols)
    meta["dropped_collapsed_rows"] = collapsed

    data, dropped = drop_repeated_header_rows(data, headers)
    meta["dropped_header_rows"] = dropped

    data, empty_dropped = drop_empty_rows(data)
    meta["dropped_empty_rows"] = empty_dropped

    if num_cols == 9:
        data, group_count = reassemble_fragmented_records(data, num_cols)
        meta["reassembled_record_groups"] = group_count
        meta["merged_wrap_rows"] = max(0, before - len(data))
    else:
        after_drop = len(data)
        data = merge_wrapped_data_rows(data, num_cols)
        merged_count = max(0, after_drop - len(data))
        before_split = len(data)
        data = split_multivalue_rows(data, num_cols)
        meta["split_multivalue_rows"] = max(0, len(data) - before_split)
        data = merge_wrapped_data_rows(data, num_cols)
        meta["merged_wrap_rows"] = merged_count + max(0, before_split - len(data))

    data = [strip_row_header_pollution(r, headers) for r in data]
    table = normalize_table_headers(table)
    headers = list(table.get("headers") or [])

    return rebuild_table_data(table, data, header_rows), meta


def rebuild_table_data(
    table: dict[str, Any],
    data: list[list[str]],
    header_rows: int,
) -> dict[str, Any]:
    """Rebuild cells / all_rows after ``data`` row count changed."""
    headers = list(table.get("headers") or [])
    num_cols = int(table.get("cols") or len(headers))
    all_rows = table.get("all_rows") or []
    if all_rows and len(all_rows) >= header_rows:
        header_part = [list(r) for r in all_rows[:header_rows]]
    else:
        header_part = [list(headers)]
        while len(header_part) < header_rows:
            header_part.append([""] * num_cols)

    old_by_rc: dict[tuple[int, int], dict] = {}
    for cell in table.get("cells") or []:
        old_by_rc[(int(cell["row"]), int(cell["col"]))] = cell

    new_cells: list[dict] = []

    def _append_row_cells(r_idx: int, row: list[str]) -> None:
        for c_idx in range(num_cols):
            text = row[c_idx] if c_idx < len(row) else ""
            old = old_by_rc.get((r_idx, c_idx))
            score = float(old.get("score") or 0.0) if old else 0.0
            new_cells.append(
                {
                    "row": r_idx,
                    "col": c_idx,
                    "text": text,
                    "score": round(score, 4),
                }
            )

    for r_idx, row in enumerate(header_part):
        _append_row_cells(r_idx, row)
    for d_idx, row in enumerate(data):
        _append_row_cells(header_rows + d_idx, row)

    out = dict(table)
    out["data"] = [list(r) for r in data]
    out["row_count"] = len(data)
    out["all_rows"] = header_part + [list(r) for r in data]
    out["cells"] = new_cells
    return out
