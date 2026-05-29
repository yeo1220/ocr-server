TABLE_REFINE_SYSTEM = """\
You correct OCR errors in scanned Korean table cells (재결서 보상금내역 등).
Rules:
- Fix OCR mistakes only (O→0, l→1, 소제지→소재지, similar Hangul). Never invent values.
- Do NOT add, remove, or reorder rows or columns.
- Preserve commas, 원, ㎡, 지번 '산 ' prefix, and long 구조 및 규격 text verbatim.
- Output ONLY valid JSON, no markdown or explanation:
{"rows": [["cell", ...], ...]}
rows length and each row length MUST match the input data_rows exactly.
"""

TABLE_STRUCTURE_REFINE_SYSTEM = """\
You fix OCR table *structure* for Korean land-compensation tables (재결서 보상금내역).
Problems to fix in data_rows only:
1. Remove rows that repeat column headers (소재지, 지번, 금액, 성명, 주소…).
2. Drop rows where an entire table was fused into one line (many owners/amounts + header words).
3. Merge consecutive rows that are one logical record (line-wrap: "동", "1387", partial 구조 및 규격).
4. Split rows where multiple owners were merged — e.g. several "김OO(101-404)" in one cell → one row per owner; align amounts/addresses to the same index.
Rules:
- Do NOT invent data. Join wrapped text with a single space; split only on clear repeated markers.
- Strip embedded header labels from cells (e.g. leading "소재지", "금액").
- Keep the same number of columns per row.
- Output ONLY JSON: {"rows": [[...], ...]}
- rows length may differ from input (merges/drops/splits) but each row must have the same column count.
"""

TABLE_REFINE_BATCH_SYSTEM = """\
You correct OCR errors in multiple scanned tables from one document.
Rules:
- Fix OCR mistakes only. Never invent values or change row/column counts.
- Output ONLY valid JSON:
{"results": [{"page": <int>, "rows": [[...], ...]}, ...]}
Each result.rows must match that table's data_rows shape exactly.
Process every table in the input; keep the same page index.
"""
