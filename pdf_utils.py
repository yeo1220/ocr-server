import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import fitz

from config import settings


def _effective_dpi(page: fitz.Page, requested_dpi: int) -> int:
    """Reduce DPI for very large pages to stay within memory limits."""
    rect = page.rect
    zoom = requested_dpi / 72
    pixels = rect.width * zoom * rect.height * zoom
    if pixels <= settings.max_page_pixels:
        return requested_dpi

    scale = (settings.max_page_pixels / pixels) ** 0.5
    return max(72, int(requested_dpi * scale))


def _render_page(
    pdf_path: str,
    page_index: int,
    job_id: str,
    dpi: int,
) -> tuple[int, str]:
    doc = fitz.open(pdf_path)
    try:
        page = doc[page_index]
        effective = _effective_dpi(page, dpi)
        zoom = effective / 72
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        image_path = os.path.join(
            settings.page_dir, f"{job_id}_{page_index + 1}.png"
        )
        pix.save(image_path)
        return page_index, image_path
    finally:
        doc.close()


def pdf_to_images(pdf_path: str, job_id: str, dpi: int | None = None) -> list[str]:
    """Render PDF pages to PNG files in parallel."""
    dpi = dpi or settings.ocr_dpi
    os.makedirs(settings.page_dir, exist_ok=True)

    doc = fitz.open(pdf_path)
    page_count = len(doc)
    doc.close()

    if page_count == 0:
        return []

    workers = min(settings.ocr_workers, page_count)
    results: dict[int, str] = {}

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_render_page, pdf_path, i, job_id, dpi): i
            for i in range(page_count)
        }
        for future in as_completed(futures):
            page_index, image_path = future.result()
            results[page_index] = image_path

    return [results[i] for i in range(page_count)]


def cleanup_job_files(job_id: str) -> None:
    """Remove temporary page images for a job."""
    prefix = os.path.join(settings.page_dir, f"{job_id}_")
    if not os.path.isdir(settings.page_dir):
        return
    for name in os.listdir(settings.page_dir):
        if name.startswith(f"{job_id}_"):
            try:
                os.remove(os.path.join(settings.page_dir, name))
            except OSError:
                pass
