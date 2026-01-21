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
    from video_ocr.engines.tesseract_engine import TesseractEngine

    __all__.append("TesseractEngine")
except ImportError:
    pass

try:
    from video_ocr.engines.surya_engine import SuryaEngine

    __all__.append("SuryaEngine")
except ImportError:
    pass

try:
    from video_ocr.engines.doctr_engine import DocTREngine

    __all__.append("DocTREngine")
except ImportError:
    pass

try:
    from video_ocr.engines.trocr_engine import TrOCREngine

    __all__.append("TrOCREngine")
except ImportError:
    pass
