#!/usr/bin/env python3
"""Verify PaddlePaddle GPU and PaddleOCR on DGX Spark."""

import sys
import tempfile
from pathlib import Path

import numpy as np


def check_paddle_gpu() -> bool:
    import paddle

    print(f"PaddlePaddle version: {paddle.__version__}")
    cuda = paddle.is_compiled_with_cuda()
    print(f"Compiled with CUDA: {cuda}")

    if not cuda:
        print("FAIL: PaddlePaddle is CPU-only")
        return False

    paddle.device.set_device("gpu")
    x = paddle.to_tensor([1.0, 2.0, 3.0])
    y = x * 2
    paddle.device.synchronize()
    print(f"GPU tensor test: {y.numpy().tolist()}")
    print(f"Device: {paddle.device.get_device()}")
    return True


def check_ocr_gpu() -> bool:
    from paddleocr import PaddleOCR

    img = np.zeros((100, 300, 3), dtype=np.uint8)
    img.fill(255)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        from PIL import Image

        Image.fromarray(img).save(f.name)
        img_path = f.name

    try:
        ocr = PaddleOCR(
            lang="korean",
            device="gpu:0",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )
        result = ocr.predict(img_path)
        print(f"OCR predict OK, result type: {type(result)}")
        return True
    finally:
        Path(img_path).unlink(missing_ok=True)


def main() -> int:
    ok = True
    try:
        if not check_paddle_gpu():
            ok = False
    except Exception as e:
        print(f"FAIL paddle GPU check: {e}")
        ok = False

    if ok:
        try:
            if not check_ocr_gpu():
                ok = False
        except Exception as e:
            print(f"FAIL OCR GPU check: {e}")
            ok = False

    if ok:
        print("All GPU checks passed.")
        return 0
    print("Some checks failed.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
