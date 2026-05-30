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
    # paddle = PP-OCRv5 + optional refine; vllm_vl = page image → Qwen2.5-VL (recommended)
    ocr_backend: str = os.getenv("OCR_BACKEND", "vllm_vl").lower()
    ocr_refine_default: str = os.getenv("OCR_REFINE_DEFAULT", "none").lower()
    # mobile: PP-OCRv5 mobile det; server: server det (Korean rec is mobile-only in PP-OCRv5)
    ocr_rec_model: str = os.getenv("OCR_REC_MODEL", "server").lower()

    # Vision OCR (Qwen2.5-VL-32B-AWQ on :8003)
    vllm_vl_base_url: str = os.getenv("VLLM_VL_BASE_URL", "http://127.0.0.1:8003/v1")
    vllm_vl_model: str = os.getenv("VLLM_VL_MODEL", "qwen-vl-ocr")
    vllm_vl_max_tokens: int = _env_int("VLLM_VL_MAX_TOKENS", 8192)
    vllm_vl_timeout: float = _env_float("VLLM_VL_TIMEOUT", 180.0)
    vllm_vl_max_image_side: int = _env_int("VLLM_VL_MAX_IMAGE_SIDE", 2048)
    # char_bbox: OCR-specialized model (<ocr> -> <char>/<bbox>) + geometric table builder.
    # json_table: instruction-following VLM returning {"table": {...}} JSON.
    vl_output_mode: str = os.getenv("VL_OCR_OUTPUT_MODE", "paddle_vl").lower()
    vllm_vl_ocr_image_side: int = _env_int("VLLM_VL_OCR_IMAGE_SIDE", 2304)
    paddle_vl_version: str = os.getenv("PADDLE_VL_VERSION", "v1.6")

    # Chat / ai-chat nginx (Gemma 4)
    vllm_base_url: str = os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8088/v1")
    vllm_model: str = os.getenv("VLLM_MODEL", "gemma-chat")
    # Table refine dedicated stack (Qwen2.5 32B instruct on :8002)
    vllm_refine_base_url: str = os.getenv(
        "VLLM_REFINE_BASE_URL", "http://127.0.0.1:8002/v1"
    )
    vllm_refine_model: str = os.getenv("VLLM_REFINE_MODEL", "qwen-refine")
    vllm_refine_fallback_to_chat: bool = os.getenv(
        "VLLM_REFINE_FALLBACK_TO_CHAT", "false"
    ).lower() in ("1", "true", "yes")
    vllm_api_key: str = os.getenv("VLLM_API_KEY", "local")
    vllm_timeout: float = _env_float("VLLM_TIMEOUT", 60)
    vllm_enabled: bool = os.getenv("VLLM_ENABLED", "true").lower() in ("1", "true", "yes")
    vllm_auto_model: bool = os.getenv("VLLM_AUTO_MODEL", "true").lower() in (
        "1",
        "true",
        "yes",
    )
    vllm_refine_threshold: float = _env_float("VLLM_REFINE_THRESHOLD", 0.85)
    # auto: vLLM structure pass only when duplicate headers / wrap rows detected
    vllm_structure_refine: str = os.getenv("VLLM_STRUCTURE_REFINE", "auto").lower()
    vllm_structure_max_tokens: int = _env_int("VLLM_STRUCTURE_MAX_TOKENS", 2048)
    vllm_refine_batch: bool = os.getenv("VLLM_REFINE_BATCH", "true").lower() in (
        "1",
        "true",
        "yes",
    )
    vllm_max_tables_per_batch: int = _env_int("VLLM_MAX_TABLES_PER_BATCH", 12)
    vllm_refine_max_tokens_default: int = _env_int("VLLM_REFINE_MAX_TOKENS", 1536)
    vllm_refine_max_tokens_large: int = _env_int("VLLM_REFINE_MAX_TOKENS_LARGE", 2048)
    vllm_refine_max_tokens_thinking: int = _env_int(
        "VLLM_REFINE_MAX_TOKENS_THINKING", 2048
    )
    vllm_refine_max_tokens_small: int = _env_int("VLLM_REFINE_MAX_TOKENS_SMALL", 1024)
    vllm_refine_max_tokens_medium: int = _env_int(
        "VLLM_REFINE_MAX_TOKENS_MEDIUM", 1280
    )


settings = Settings()
