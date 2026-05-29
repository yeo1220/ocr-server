import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager

import paddle
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from config import settings
from llm_refine import check_vllm_reachable, refine_table
from ocr_engine import get_active_device, get_ocr, run_ocr_image, warmup
from pdf_utils import pdf_to_images
from preprocess import preprocess_for_ocr
from table_builder import build_table, table_to_text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.makedirs(settings.upload_dir, exist_ok=True)
os.makedirs(settings.page_dir, exist_ok=True)
os.makedirs(settings.result_dir, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_ocr()
    try:
        warmup()
    except Exception as e:
        logger.warning("Warmup failed (non-fatal): %s", e)
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "device": get_active_device(),
        "paddle_cuda": paddle.is_compiled_with_cuda(),
        "ocr_dpi_default": settings.ocr_dpi,
        "ocr_workers": settings.ocr_workers,
        "vllm_enabled": settings.vllm_enabled,
        "vllm_reachable": await check_vllm_reachable(),
        "vllm_model": settings.vllm_model,
    }


def _parse_col_boundaries(raw: str | None) -> list[float] | None:
    if not raw or not raw.strip():
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=422,
            detail=f"table_col_boundaries must be JSON array: {e}",
        ) from e
    if not isinstance(parsed, list):
        raise HTTPException(
            status_code=422,
            detail="table_col_boundaries must be a JSON array of numbers",
        )
    return [float(x) for x in parsed]


@app.post("/ocr")
async def ocr_endpoint(
    file: UploadFile = File(...),
    mode: str = Form("scan"),
    dpi: int = Form(None),
    format: str = Form("text"),
    table_cols: int = Form(None),
    table_header_row: int = Form(0),
    table_col_boundaries: str = Form(None),
    refine: str = Form("none"),
    refine_threshold: float = Form(None),
):
    start = time.time()
    job_id = str(uuid.uuid4())
    filename = file.filename or f"{job_id}.bin"
    ext = os.path.splitext(filename)[1].lower()
    render_dpi = dpi if dpi is not None else settings.ocr_dpi
    preprocessed = mode == "scan"
    table_mode = format.lower() == "table"
    col_boundaries = _parse_col_boundaries(table_col_boundaries)

    if table_mode and (table_cols is None or table_cols < 1):
        raise HTTPException(
            status_code=422,
            detail="format=table requires table_cols (positive integer)",
        )

    if refine.lower() not in ("none", "vllm"):
        raise HTTPException(
            status_code=422,
            detail="refine must be 'none' or 'vllm'",
        )
    use_vllm = refine.lower() == "vllm"
    threshold = (
        refine_threshold
        if refine_threshold is not None
        else settings.vllm_refine_threshold
    )

    upload_path = os.path.join(settings.upload_dir, f"{job_id}{ext}")
    with open(upload_path, "wb") as f:
        f.write(await file.read())

    page_results = []
    full_text_parts = []
    page_image_paths: list[str] = []

    if ext == ".pdf":
        image_paths = pdf_to_images(upload_path, job_id, dpi=render_dpi)
    else:
        image_paths = [upload_path]

    page_image_paths = list(image_paths)

    for idx, image_path in enumerate(image_paths, start=1):
        ocr_input = image_path
        if preprocessed:
            preprocessed_path = os.path.join(
                settings.page_dir, f"{job_id}_{idx}_pre.png"
            )
            preprocess_for_ocr(image_path, preprocessed_path)
            ocr_input = preprocessed_path
            page_image_paths.append(preprocessed_path)

        result = run_ocr_image(ocr_input)
        page_payload: dict = {
            "page": idx,
            "text": result["text"],
            "blocks": result["blocks"],
            "avg_score": result["avg_score"],
        }

        if table_mode:
            table = build_table(
                result["raw_blocks"],
                num_cols=table_cols,
                header_row=table_header_row,
                col_boundaries=col_boundaries,
            )
            refinement_meta = None
            if use_vllm:
                table, refinement_meta = await refine_table(
                    table, threshold=threshold
                )
            else:
                table["data_refined"] = [list(row) for row in table.get("data") or []]
                for cell in table.get("cells") or []:
                    cell["text_refined"] = cell.get("text", "")
                    cell["refined"] = False

            page_payload["table"] = table
            page_payload["refinement"] = refinement_meta
            page_payload["text"] = table_to_text(table)
            if table.get("data_refined"):
                page_payload["text_refined"] = table_to_text(
                    {**table, "data": table["data_refined"]}
                )

        full_text_parts.append(page_payload["text"])
        page_results.append(page_payload)

    elapsed = round(time.time() - start, 3)
    response = {
        "job_id": job_id,
        "filename": filename,
        "mode": mode,
        "format": format,
        "dpi": render_dpi,
        "device": get_active_device(),
        "preprocessed": preprocessed,
        "processing_time_sec": elapsed,
        "page_count": len(page_results),
        "text": "\n\n".join(full_text_parts),
        "pages": page_results,
        "upload_path": upload_path,
        "result_path": os.path.join(settings.result_dir, f"{job_id}.json"),
        "page_image_paths": page_image_paths,
    }
    if table_mode:
        response["table_cols"] = table_cols
        response["table_header_row"] = table_header_row
        response["refine"] = refine

    with open(response["result_path"], "w", encoding="utf-8") as f:
        json.dump(response, f, ensure_ascii=False, indent=2)

    return response
