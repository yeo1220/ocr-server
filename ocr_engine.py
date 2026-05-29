import ctypes
import ctypes.util
import logging
import os
import tempfile
from collections.abc import Mapping
from typing import Any

import cv2
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


def _normalize_ocr_text(text: str) -> str:
    t = str(text or "").strip()
    return " ".join(t.split())


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


def _merge_line_blocks(blocks: list[dict]) -> tuple[list[dict], list[str]]:
    """Merge adjacent OCR fragments on same line to reduce split-cell noise."""
    if not blocks:
        return [], []

    ordered = sorted(
        blocks,
        key=lambda b: (_box_center_y(b.get("box") or []), _box_xrange(b.get("box") or [])[0]),
    )
    heights = sorted(
        h for h in (_box_height(b.get("box") or []) for b in ordered) if h > 0
    )
    med_h = heights[len(heights) // 2] if heights else 12.0
    y_tol = max(6.0, med_h * 0.65)
    gap_tol = max(10.0, med_h * 1.5)

    lines: list[list[dict]] = []
    for b in ordered:
        cy = _box_center_y(b.get("box") or [])
        if not lines:
            lines.append([b])
            continue
        prev_line = lines[-1]
        prev_cy = sum(_box_center_y(x.get("box") or []) for x in prev_line) / len(prev_line)
        if abs(cy - prev_cy) <= y_tol:
            prev_line.append(b)
        else:
            lines.append([b])

    merged_blocks: list[dict] = []
    merged_lines: list[str] = []
    for line in lines:
        line = sorted(line, key=lambda b: _box_xrange(b.get("box") or [])[0])
        groups: list[list[dict]] = [[line[0]]]
        for b in line[1:]:
            prev = groups[-1][-1]
            prev_x1 = _box_xrange(prev.get("box") or [])[1]
            cur_x0 = _box_xrange(b.get("box") or [])[0]
            if (cur_x0 - prev_x1) <= gap_tol:
                groups[-1].append(b)
            else:
                groups.append([b])

        line_text_parts: list[str] = []
        for g in groups:
            txt = _normalize_ocr_text(" ".join(_normalize_ocr_text(x.get("text", "")) for x in g))
            if not txt:
                continue
            score = sum(float(x.get("score") or 0.0) for x in g) / max(1, len(g))
            box = g[0].get("box")
            merged_blocks.append({"text": txt, "score": float(score), "box": box})
            line_text_parts.append(txt)

        if line_text_parts:
            merged_lines.append(" ".join(line_text_parts).strip())

    return merged_blocks, merged_lines


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
        text_detection_model_name="PP-OCRv5_server_det",
        text_recognition_model_name="korean_PP-OCRv5_mobile_rec",
        device=device,
        cpu_threads=settings.cpu_threads,
        use_doc_orientation_classify=True,
        use_doc_unwarping=True,
        use_textline_orientation=True,
        text_det_limit_side_len=settings.text_det_limit_side_len,
        text_det_limit_type=settings.text_det_limit_type,
        text_det_thresh=settings.text_det_thresh,
        text_det_box_thresh=settings.text_det_box_thresh,
        text_det_unclip_ratio=settings.text_det_unclip_ratio,
        text_recognition_batch_size=settings.text_recognition_batch_size,
        text_rec_score_thresh=settings.text_rec_score_thresh,
    )


def _prepare_image_for_ocr(image_path: str) -> tuple[str, bool]:
    """Upscale low-resolution pages so small Hangul remains detectable."""
    image = cv2.imread(image_path)
    if image is None:
        return image_path, False

    h, w = image.shape[:2]
    min_side = min(h, w)
    target = settings.ocr_min_side_for_det
    if min_side >= target:
        return image_path, False

    scale = min(settings.ocr_max_upscale, target / min_side)
    if scale <= 1.01:
        return image_path, False

    new_w = int(w * scale)
    new_h = int(h * scale)
    upscaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    fd, tmp_path = tempfile.mkstemp(suffix=".png", prefix="ocr_upscale_")
    os.close(fd)
    cv2.imwrite(tmp_path, upscaled)
    logger.debug(
        "Upscaled OCR input %dx%d -> %dx%d (scale=%.2f)",
        w,
        h,
        new_w,
        new_h,
        scale,
    )
    return tmp_path, True


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
    ocr_input, is_temp = _prepare_image_for_ocr(image_path)
    try:
        try:
            result = get_ocr().predict(ocr_input)
        except Exception as e:
            if _active_device.startswith("gpu") and _is_cudnn_error(e):
                _fallback_to_cpu(str(e))
                result = get_ocr().predict(ocr_input)
            else:
                raise
    finally:
        if is_temp:
            try:
                os.unlink(ocr_input)
            except OSError:
                pass

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
            text = _normalize_ocr_text(text)
            if not text:
                continue
            box = box.tolist() if hasattr(box, "tolist") else box
            lines.append(text)
            scores.append(float(score))
            blocks.append({"text": text, "score": float(score), "box": box})

    merged_blocks, merged_lines = _merge_line_blocks(blocks)
    final_blocks = merged_blocks or blocks
    final_lines = merged_lines or lines
    avg_score = sum(scores) / len(scores) if scores else 0.0
    return {
        "text": "\n".join(final_lines),
        "blocks": final_blocks,
        "raw_blocks": blocks,
        "avg_score": round(avg_score, 4),
    }
