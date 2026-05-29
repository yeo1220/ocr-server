import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    upload_dir: str = os.getenv("OCR_UPLOAD_DIR", "/tmp/ocr_uploads")
    page_dir: str = os.getenv("OCR_PAGE_DIR", "/tmp/ocr_pages")
    result_dir: str = os.getenv("OCR_RESULT_DIR", "/tmp/ocr_results")
    ocr_device: str = os.getenv("OCR_DEVICE", "gpu:0")
    ocr_dpi: int = int(os.getenv("OCR_DPI", "300"))
    ocr_workers: int = int(os.getenv("OCR_WORKERS", "8"))
    cpu_threads: int = int(os.getenv("OCR_CPU_THREADS", "20"))
    text_det_limit_side_len: int = int(os.getenv("OCR_DET_LIMIT_SIDE_LEN", "960"))
    max_page_pixels: int = int(os.getenv("OCR_MAX_PAGE_PIXELS", "25000000"))

    vllm_base_url: str = os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8088/v1")
    vllm_model: str = os.getenv("VLLM_MODEL", "qwen-ocr")
    vllm_api_key: str = os.getenv("VLLM_API_KEY", "local")
    vllm_timeout: float = float(os.getenv("VLLM_TIMEOUT", "120"))
    vllm_enabled: bool = os.getenv("VLLM_ENABLED", "true").lower() in ("1", "true", "yes")
    vllm_refine_threshold: float = float(os.getenv("VLLM_REFINE_THRESHOLD", "0.85"))


settings = Settings()
