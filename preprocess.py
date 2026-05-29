import cv2
import numpy as np


def _deskew(gray: np.ndarray) -> np.ndarray:
    """Estimate text angle and rotate to reduce skew for table lines."""
    coords = np.column_stack(np.where(gray < 250))
    if coords.size == 0:
        return gray
    rect = cv2.minAreaRect(coords.astype(np.float32))
    angle = rect[-1]
    if angle < -45:
        angle = 90 + angle
    if abs(angle) < 0.3:
        return gray
    h, w = gray.shape[:2]
    m = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(
        gray,
        m,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def preprocess_for_ocr(input_path: str, output_path: str) -> str:
    """Preprocess scanned documents for better Korean/table OCR."""
    image = cv2.imread(input_path)
    if image is None:
        raise ValueError(f"Cannot read image: {input_path}")

    # 1) gray + denoise
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=12)

    # 2) local contrast enhancement (helps faint Hangul glyphs)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)

    # 3) deskew
    deskewed = _deskew(enhanced)

    # 4) adaptive binarization
    binary = cv2.adaptiveThreshold(
        deskewed,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        35,
        11,
    )

    # 5) morphological cleanup: connect broken strokes while reducing speckle
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    cleaned = cv2.medianBlur(cleaned, 3)

    cv2.imwrite(output_path, cleaned)
    return output_path
