import os
from dataclasses import dataclass


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


@dataclass(frozen=True)
class Settings:
    upload_dir: str = os.getenv("OCR_UPLOAD_DIR", "/tmp/ocr_uploads")
    page_dir: str = os.getenv("OCR_PAGE_DIR", "/tmp/ocr_pages")
    result_dir: str = os.getenv("OCR_RESULT_DIR", "/tmp/ocr_results")
    ocr_device: str = os.getenv("OCR_DEVICE", "gpu:0")
    # DGX Spark: higher DPI + pixel budget for sharper Hangul on scans
    ocr_dpi: int = _env_int("OCR_DPI", 400)
    ocr_workers: int = _env_int("OCR_WORKERS", 8)
    cpu_threads: int = _env_int("OCR_CPU_THREADS", 20)
    # Detection resize: 960 downscales 300+ DPI scans heavily; 2560 keeps detail on GB10
    text_det_limit_side_len: int = _env_int("OCR_DET_LIMIT_SIDE_LEN", 2560)
    text_det_limit_type: str = os.getenv("OCR_DET_LIMIT_TYPE", "max")
    text_det_thresh: float = _env_float("OCR_TEXT_DET_THRESH", 0.25)
    text_det_box_thresh: float = _env_float("OCR_TEXT_DET_BOX_THRESH", 0.5)
    text_det_unclip_ratio: float = _env_float("OCR_TEXT_DET_UNCLIP_RATIO", 1.8)
    text_recognition_batch_size: int = _env_int("OCR_REC_BATCH_SIZE", 16)
    text_rec_score_thresh: float = _env_float("OCR_REC_SCORE_THRESH", 0.0)
    max_page_pixels: int = _env_int("OCR_MAX_PAGE_PIXELS", 60_000_000)
    # Upscale short side before OCR when render is still too small for small glyphs
    ocr_min_side_for_det: int = _env_int("OCR_MIN_SIDE_FOR_DET", 2000)
    ocr_max_upscale: float = _env_float("OCR_MAX_UPSCALE", 2.0)
    # enhance: color+CLAHE (best for Paddle); binary: legacy binarize for very poor scans
    preprocess_mode: str = os.getenv("OCR_PREPROCESS_MODE", "enhance").lower()
    ocr_refine_default: str = os.getenv("OCR_REFINE_DEFAULT", "vllm").lower()

    vllm_base_url: str = os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8088/v1")
    vllm_model: str = os.getenv("VLLM_MODEL", "qwen-ocr")
    vllm_api_key: str = os.getenv("VLLM_API_KEY", "local")
    vllm_timeout: float = _env_float("VLLM_TIMEOUT", 120)
    vllm_enabled: bool = os.getenv("VLLM_ENABLED", "true").lower() in ("1", "true", "yes")
    vllm_refine_threshold: float = _env_float("VLLM_REFINE_THRESHOLD", 0.85)


settings = Settings()
