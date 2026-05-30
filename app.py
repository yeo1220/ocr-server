import json
import httpx
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from config import settings
from llm_refine import refine_tables_batch
from vllm_client import (
    check_chat_reachable,
    check_refine_reachable,
    check_vl_reachable,
    close_client,
    get_model_profile,
    get_refine_base_url,
    get_resolved_model_id,
    get_vl_base_url,
    get_vl_resolved_model_id,
    resolve_refine_model,
    resolve_vl_model,
)
from pdf_utils import pdf_to_images
from table_builder import build_table, export_table_aliases, table_to_text
from vl_ocr import run_vl_ocr_page, warmup_vl

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.makedirs(settings.upload_dir, exist_ok=True)
os.makedirs(settings.page_dir, exist_ok=True)
os.makedirs(settings.result_dir, exist_ok=True)

_USE_PADDLE = settings.ocr_backend == "paddle"
_USE_VL = settings.ocr_backend == "vllm_vl"


def _paddle_cuda() -> bool:
    if not _USE_PADDLE:
        return False
    try:
        import paddle

        return paddle.is_compiled_with_cuda()
    except Exception:
        return False


def _active_device() -> str:
    if _USE_VL:
        return "vllm_vl"
    if _USE_PADDLE:
        from ocr_engine import get_active_device

        return get_active_device()
    return settings.ocr_device


@asynccontextmanager
async def lifespan(app: FastAPI):
    if _USE_PADDLE:
        from ocr_engine import get_ocr, warmup

        get_ocr()
        try:
            warmup()
        except Exception as e:
            logger.warning("Paddle warmup failed (non-fatal): %s", e)
    elif _USE_VL:
        warmup_vl()

    if settings.vllm_enabled:
        try:
            if _USE_VL:
                await resolve_vl_model()
            elif settings.ocr_refine_default == "vllm":
                await resolve_refine_model()
        except Exception as e:
            logger.warning("vLLM model resolve failed (non-fatal): %s", e)
    yield
    await close_client()


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health():
    payload = {
        "status": "ok",
        "ocr_backend": settings.ocr_backend,
        "device": _active_device(),
        "paddle_cuda": _paddle_cuda(),
        "ocr_dpi_default": settings.ocr_dpi,
        "ocr_workers": settings.ocr_workers,
        "ocr_det_limit_side_len": settings.text_det_limit_side_len,
        "ocr_preprocess_mode": settings.preprocess_mode,
        "ocr_refine_default": settings.ocr_refine_default,
        "ocr_max_page_pixels": settings.max_page_pixels,
        "vllm_enabled": settings.vllm_enabled,
        "vllm_chat_reachable": await check_chat_reachable(),
        "vllm_chat_model": settings.vllm_model,
        "vllm_vl_base_url": get_vl_base_url(),
        "vllm_vl_model": get_vl_resolved_model_id(),
        "vllm_vl_reachable": await check_vl_reachable(),
        "vllm_vl_max_image_side": settings.vllm_vl_max_image_side,
    }
    if _USE_PADDLE:
        payload["ocr_rec_model"] = settings.ocr_rec_model
        payload["vllm_refine_reachable"] = await check_refine_reachable()
        payload["vllm_refine_model"] = get_resolved_model_id()
        payload["vllm_refine_base_url"] = get_refine_base_url()
        prof = get_model_profile()
        payload["vllm_profile"] = (
            {
                "is_thinking": prof.is_thinking,
                "is_small_instruct": prof.is_small_instruct,
                "max_output_tokens": prof.max_output_tokens,
                "base_url": prof.base_url,
            }
            if prof
            else None
        )
    return payload


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
    preprocess: str = Form(None),
    dpi: int = Form(None),
    format: str = Form("text"),
    table_cols: int = Form(None),
    table_header_row: int = Form(0),
    table_header_rows: int = Form(1),
    table_col_boundaries: str = Form(None),
    refine: str = Form(None),
    refine_threshold: float = Form(None),
):
    start = time.time()
    job_id = str(uuid.uuid4())
    filename = file.filename or f"{job_id}.bin"
    ext = os.path.splitext(filename)[1].lower()
    render_dpi = dpi if dpi is not None else settings.ocr_dpi
    preprocessed = mode == "scan" and _USE_PADDLE
    preprocess_mode = (
        preprocess.strip().lower()
        if preprocess and preprocess.strip()
        else settings.preprocess_mode
    )
    if preprocess_mode not in ("enhance", "binary", "none"):
        raise HTTPException(
            status_code=422,
            detail="preprocess must be 'enhance', 'binary', or 'none'",
        )
    table_mode = format.lower() == "table"
    col_boundaries = _parse_col_boundaries(table_col_boundaries)

    if table_mode and (table_cols is None or table_cols < 1):
        raise HTTPException(
            status_code=422,
            detail="format=table requires table_cols (positive integer)",
        )
    if table_header_rows < 1:
        raise HTTPException(
            status_code=422,
            detail="table_header_rows must be >= 1",
        )

    refine_mode = (
        refine.strip().lower()
        if refine and refine.strip()
        else settings.ocr_refine_default
    )
    if refine_mode not in ("none", "vllm"):
        raise HTTPException(
            status_code=422,
            detail="refine must be 'none' or 'vllm'",
        )
    if _USE_VL and refine_mode == "vllm":
        logger.info("refine=vllm ignored: VL backend already reads page images end-to-end")
        refine_mode = "none"
    use_vllm_refine = refine_mode == "vllm" and _USE_PADDLE
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

    tables_for_refine: list[tuple[int, dict]] = []

    for idx, image_path in enumerate(image_paths, start=1):
        if _USE_VL:
            if not await check_vl_reachable():
                raise HTTPException(
                    status_code=503,
                    detail=(
                        "Vision OCR (vLLM VL) is not reachable at "
                        f"{settings.vllm_vl_base_url}. "
                        "Start vllm-ocr-vl: cd vllm && ./download-vl-model.sh && ./start.sh"
                    ),
                )
            try:
                result = await run_vl_ocr_page(
                    image_path,
                    table_mode=table_mode,
                    num_cols=table_cols or 9,
                    header_rows=table_header_rows,
                )
            except httpx.TimeoutException as exc:
                logger.error(
                    "VL OCR timeout on page %s after %.0fs: %s",
                    idx, settings.vllm_vl_timeout, exc,
                )
                raise HTTPException(
                    status_code=504,
                    detail=(
                        f"VL OCR timed out on page {idx} "
                        f"(>{int(settings.vllm_vl_timeout)}s). "
                        "Page may be too dense; retry or lower VLLM_VL_MAX_TOKENS."
                    ),
                ) from exc
        else:
            from ocr_engine import run_ocr_image
            from preprocess import preprocess_for_ocr

            ocr_input = image_path
            if preprocessed:
                preprocessed_path = os.path.join(
                    settings.page_dir, f"{job_id}_{idx}_pre.png"
                )
                preprocess_for_ocr(
                    image_path, preprocessed_path, mode=preprocess_mode
                )
                ocr_input = preprocessed_path
                page_image_paths.append(preprocessed_path)
            result = run_ocr_image(ocr_input)

        page_payload: dict = {
            "page": idx,
            "text": result["text"],
            "blocks": result["blocks"],
            "raw_blocks": result.get("raw_blocks", []),
            "avg_score": result["avg_score"],
            "ocr_backend": result.get("ocr_backend", settings.ocr_backend),
        }
        if result.get("vl_meta"):
            page_payload["vl_meta"] = result["vl_meta"]

        if table_mode:
            if _USE_VL:
                table = result.get("table")
                if not table:
                    raise HTTPException(
                        status_code=502,
                        detail="VL model did not return a table for this page",
                    )
                export_table_aliases(table)
                page_payload["table"] = table
                page_payload["text"] = table_to_text(table)
                if table.get("data_refined"):
                    page_payload["text_refined"] = table_to_text(
                        {**table, "data": table["data_refined"]}
                    )
            else:
                table = build_table(
                    result["raw_blocks"],
                    num_cols=table_cols,
                    header_row=table_header_row,
                    header_rows=table_header_rows,
                    col_boundaries=col_boundaries,
                )
                if use_vllm_refine:
                    tables_for_refine.append((idx, table))
                    page_payload["_table_pending_refine"] = True
                else:
                    table["data_refined"] = [list(row) for row in table.get("data") or []]
                    for cell in table.get("cells") or []:
                        cell["text_refined"] = cell.get("text", "")
                        cell["refined"] = False
                    export_table_aliases(table)
                    page_payload["table"] = table
                    page_payload["text"] = table_to_text(table)
                    if table.get("data_refined"):
                        page_payload["text_refined"] = table_to_text(
                            {**table, "data": table["data_refined"]}
                        )

        if not table_mode or (not use_vllm_refine and not _USE_VL):
            full_text_parts.append(page_payload["text"])
        elif _USE_VL and table_mode:
            full_text_parts.append(page_payload["text"])
        page_results.append(page_payload)

    if table_mode and use_vllm_refine and tables_for_refine:
        refined_map = await refine_tables_batch(tables_for_refine, threshold=threshold)
        for page_payload in sorted(page_results, key=lambda p: p["page"]):
            idx = page_payload["page"]
            if idx not in refined_map:
                continue
            table, refinement_meta = refined_map[idx]
            export_table_aliases(table)
            page_payload.pop("_table_pending_refine", None)
            page_payload["table"] = table
            page_payload["refinement"] = refinement_meta
            page_payload["text"] = table_to_text(table)
            if table.get("data_refined"):
                page_payload["text_refined"] = table_to_text(
                    {**table, "data": table["data_refined"]}
                )
            full_text_parts.append(page_payload["text"])
    elif table_mode and not _USE_VL:
        for page_payload in page_results:
            if page_payload.get("_table_pending_refine"):
                page_payload.pop("_table_pending_refine", None)
                full_text_parts.append(page_payload.get("text") or "")

    elapsed = round(time.time() - start, 3)
    response = {
        "job_id": job_id,
        "filename": filename,
        "mode": mode,
        "format": format,
        "dpi": render_dpi,
        "ocr_backend": settings.ocr_backend,
        "device": _active_device(),
        "preprocessed": preprocessed,
        "preprocess_mode": preprocess_mode if preprocessed else None,
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
        response["table_header_rows"] = table_header_rows
        response["refine"] = refine_mode

    with open(response["result_path"], "w", encoding="utf-8") as f:
        json.dump(response, f, ensure_ascii=False, indent=2)

    return response
