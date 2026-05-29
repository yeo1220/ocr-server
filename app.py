import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager

import paddle
from fastapi import FastAPI, File, Form, UploadFile

from config import settings
from ocr_engine import get_active_device, get_ocr, run_ocr_image, warmup
from pdf_utils import pdf_to_images
from preprocess import preprocess_for_ocr

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
def health():
    return {
        "status": "ok",
        "device": get_active_device(),
        "paddle_cuda": paddle.is_compiled_with_cuda(),
        "ocr_dpi_default": settings.ocr_dpi,
        "ocr_workers": settings.ocr_workers,
    }


@app.post("/ocr")
async def ocr_endpoint(
    file: UploadFile = File(...),
    mode: str = Form("scan"),
    dpi: int = Form(None),
):
    start = time.time()
    job_id = str(uuid.uuid4())
    filename = file.filename or f"{job_id}.bin"
    ext = os.path.splitext(filename)[1].lower()
    render_dpi = dpi if dpi is not None else settings.ocr_dpi
    preprocessed = mode == "scan"

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
        full_text_parts.append(result["text"])
        page_results.append(
            {
                "page": idx,
                "text": result["text"],
                "blocks": result["blocks"],
                "avg_score": result["avg_score"],
            }
        )

    elapsed = round(time.time() - start, 3)
    response = {
        "job_id": job_id,
        "filename": filename,
        "mode": mode,
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

    with open(response["result_path"], "w", encoding="utf-8") as f:
        json.dump(response, f, ensure_ascii=False, indent=2)

    return response
