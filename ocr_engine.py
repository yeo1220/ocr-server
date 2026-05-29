import ctypes
import ctypes.util
import logging
import os
import tempfile
from collections.abc import Mapping
from typing import Any

import paddle
from paddleocr import PaddleOCR

from config import settings

logger = logging.getLogger(__name__)

_ocr: PaddleOCR | None = None
_active_device: str = "cpu"

_CUDNN_PATHS = [
    "/home/wslaw/local/cudnn/usr/lib/aarch64-linux-gnu/libcudnn.so.9",
    "/usr/lib/aarch64-linux-gnu/libcudnn.so.9",
]


def _result_to_dict(page_result: Any) -> dict:
    """Normalize PaddleOCR page result objects/dicts to a plain dict."""
    if isinstance(page_result, Mapping):
        return dict(page_result)
    if hasattr(page_result, "json"):
        value = getattr(page_result, "json")
        if isinstance(value, Mapping):
            return dict(value)
    if hasattr(page_result, "to_dict"):
        try:
            value = page_result.to_dict()
            if isinstance(value, Mapping):
                return dict(value)
        except Exception:
            pass
    if hasattr(page_result, "model_dump"):
        try:
            value = page_result.model_dump()
            if isinstance(value, Mapping):
                return dict(value)
        except Exception:
            pass
    return {}


def _is_cudnn_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "dnn handle" in msg or "cudnn" in msg


def _cudnn_loadable() -> bool:
    for path in _CUDNN_PATHS:
        if os.path.isfile(path):
            try:
                ctypes.CDLL(path)
                return True
            except OSError:
                continue
    name = ctypes.util.find_library("cudnn")
    if name:
        try:
            ctypes.CDLL(name)
            return True
        except OSError:
            pass
    return False


def _probe_gpu(device: str) -> bool:
    if not paddle.is_compiled_with_cuda():
        return False
    if not _cudnn_loadable():
        logger.warning("cuDNN library not found; GPU inference will not work")
        return False
    try:
        paddle.device.set_device(device.split(":")[0] if ":" in device else device)
        x = paddle.randn([1, 3, 8, 8])
        conv = paddle.nn.Conv2D(3, 8, 3)
        y = conv(x)
        paddle.device.synchronize()
        return y.shape == [1, 8, 6, 6]
    except Exception as e:
        logger.warning("GPU probe failed (%s): %s", device, e)
        return False


def _resolve_device() -> str:
    requested = settings.ocr_device
    if requested.startswith("gpu") and _probe_gpu(requested):
        return requested
    if requested.startswith("gpu"):
        logger.warning(
            "GPU requested (%s) but unavailable; falling back to CPU",
            requested,
        )
    return "cpu"


def _build_ocr(device: str) -> PaddleOCR:
    return PaddleOCR(
        lang="korean",
        ocr_version="PP-OCRv5",
        device=device,
        cpu_threads=settings.cpu_threads,
        use_doc_orientation_classify=True,
        use_doc_unwarping=True,
        use_textline_orientation=True,
        text_det_limit_side_len=settings.text_det_limit_side_len,
    )


def _init_ocr(device: str) -> PaddleOCR:
    global _ocr, _active_device
    if _ocr is not None:
        try:
            _ocr.close()
        except Exception:
            pass
    _active_device = device
    logger.info("Initializing PaddleOCR on device=%s", _active_device)
    _ocr = _build_ocr(device)
    return _ocr


def _fallback_to_cpu(reason: str) -> None:
    if _active_device == "cpu":
        return
    logger.warning("Falling back to CPU: %s", reason)
    _init_ocr("cpu")


def get_active_device() -> str:
    return _active_device


def get_ocr() -> PaddleOCR:
    global _ocr
    if _ocr is None:
        _init_ocr(_resolve_device())
    return _ocr


def warmup() -> None:
    """Run a minimal predict to load models at startup."""
    import numpy as np
    from PIL import Image

    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img.fill(255)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        Image.fromarray(img).save(f.name)
        path = f.name
    try:
        try:
            get_ocr().predict(path)
        except Exception as e:
            if _active_device.startswith("gpu") and _is_cudnn_error(e):
                _fallback_to_cpu(str(e))
                get_ocr().predict(path)
            else:
                raise
        logger.info("OCR warmup complete on device=%s", _active_device)
    finally:
        os.unlink(path)


def run_ocr_image(image_path: str) -> dict:
    try:
        result = get_ocr().predict(image_path)
    except Exception as e:
        if _active_device.startswith("gpu") and _is_cudnn_error(e):
            _fallback_to_cpu(str(e))
            result = get_ocr().predict(image_path)
        else:
            raise

    lines: list[str] = []
    blocks: list[dict] = []
    scores: list[float] = []

    if not result:
        return {"text": "", "blocks": [], "avg_score": 0.0}

    for page_result in result:
        if not page_result:
            continue
        page = _result_to_dict(page_result)

        texts = page.get("rec_texts", [])
        page_scores = page.get("rec_scores", [])
        boxes = page.get("rec_polys", page.get("dt_polys", []))

        for text, score, box in zip(texts, page_scores, boxes):
            box = box.tolist() if hasattr(box, "tolist") else box
            lines.append(text)
            scores.append(float(score))
            blocks.append({"text": text, "score": float(score), "box": box})

    avg_score = sum(scores) / len(scores) if scores else 0.0
    return {
        "text": "\n".join(lines),
        "blocks": blocks,
        "avg_score": round(avg_score, 4),
    }
