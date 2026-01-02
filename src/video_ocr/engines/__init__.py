"""OCR engine implementations."""

from video_ocr.engines.base import BaseOCREngine, OCRResult, TextRegion
from video_ocr.engines.easyocr_engine import EasyOCREngine

__all__ = [
    "BaseOCREngine",
    "OCRResult",
    "TextRegion",
    "EasyOCREngine",
]

# Optional engines - import only if available
try:
    from video_ocr.engines.paddleocr_engine import PaddleOCREngine

    __all__.append("PaddleOCREngine")
except ImportError:
    pass

try:
    from video_ocr.engines.tesseract_engine import TesseractEngine

    __all__.append("TesseractEngine")
except ImportError:
    pass
