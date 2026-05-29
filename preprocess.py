import cv2
import numpy as np


def _estimate_skew_angle(gray: np.ndarray) -> float:
    coords = np.column_stack(np.where(gray < 250))
    if coords.size == 0:
        return 0.0
    rect = cv2.minAreaRect(coords.astype(np.float32))
    angle = rect[-1]
    if angle < -45:
        angle = 90 + angle
    if abs(angle) < 0.3:
        return 0.0
    return angle


def _rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    if abs(angle) < 0.3:
        return image
    h, w = image.shape[:2]
    m = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(
        image,
        m,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _enhance_color(image: np.ndarray) -> np.ndarray:
    """Contrast enhancement without binarization — keeps PaddleOCR-friendly RGB."""
    denoised = cv2.bilateralFilter(image, d=7, sigmaColor=75, sigmaSpace=75)
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    angle = _estimate_skew_angle(gray)
    return _rotate_image(enhanced, angle)


def _binarize_scan(image: np.ndarray) -> np.ndarray:
    """Legacy path for extremely faded scans; can hurt neural OCR on normal documents."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=12)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    deskewed = _rotate_image(enhanced, _estimate_skew_angle(enhanced))
    binary = cv2.adaptiveThreshold(
        deskewed,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        35,
        11,
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    cleaned = cv2.medianBlur(cleaned, 3)
    return cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)


def preprocess_for_ocr(
    input_path: str,
    output_path: str,
    mode: str = "enhance",
) -> str:
    """Preprocess scanned documents for Korean/table OCR.

    mode:
      enhance — denoise + CLAHE + deskew (color, recommended for PaddleOCR)
      binary  — adaptive threshold (very poor scans only)
      none    — copy original
    """
    image = cv2.imread(input_path)
    if image is None:
        raise ValueError(f"Cannot read image: {input_path}")

    mode = (mode or "enhance").lower()
    if mode == "none":
        out = image
    elif mode == "binary":
        out = _binarize_scan(image)
    else:
        out = _enhance_color(image)

    cv2.imwrite(output_path, out)
    return output_path
