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


settings = Settings()
