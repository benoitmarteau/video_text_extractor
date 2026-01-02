"""PaddleOCR engine implementation."""

from typing import List, Optional

import numpy as np

from video_ocr.engines.base import BaseOCREngine, TextRegion
from video_ocr.utils.logging_config import get_logger


class PaddleOCREngine(BaseOCREngine):
    """OCR engine using PaddleOCR."""

    def __init__(
        self,
        languages: List[str] = None,
        confidence_threshold: float = 0.5,
        gpu: bool = True,
    ):
        """
        Initialize PaddleOCR engine.

        Args:
            languages: List of language codes (uses first one)
            confidence_threshold: Minimum confidence threshold
            gpu: Use GPU if available
        """
        super().__init__(languages, confidence_threshold, gpu)
        self._ocr = None
        self.logger = get_logger()

    def _initialize(self) -> None:
        """Initialize the PaddleOCR reader."""
        from paddleocr import PaddleOCR

        # Map common language codes to PaddleOCR format
        lang_map = {
            "en": "en",
            "ch": "ch",
            "chinese": "ch",
            "zh": "ch",
            "fr": "french",
            "de": "german",
            "ja": "japan",
            "ko": "korean",
        }

        lang = self.languages[0] if self.languages else "en"
        paddle_lang = lang_map.get(lang, lang)

        self.logger.info(
            f"Initializing PaddleOCR with language: {paddle_lang}, GPU: {self.gpu}"
        )

        self._ocr = PaddleOCR(
            use_angle_cls=True,
            lang=paddle_lang,
            use_gpu=self.gpu,
            show_log=False,
        )

        self.logger.info("PaddleOCR initialized successfully")

    def _process_frame(self, frame: np.ndarray) -> List[TextRegion]:
        """
        Process a frame using PaddleOCR.

        Args:
            frame: BGR numpy array

        Returns:
            List of TextRegion objects
        """
        if self._ocr is None:
            raise RuntimeError("PaddleOCR not initialized")

        # PaddleOCR accepts BGR format
        results = self._ocr.ocr(frame, cls=True)

        regions = []

        # Handle case where no text is detected
        if results is None or len(results) == 0:
            return regions

        # PaddleOCR returns: [[[[x1,y1],[x2,y2],[x3,y3],[x4,y4]], (text, confidence)], ...]
        for result in results:
            if result is None:
                continue

            for line in result:
                if line is None or len(line) < 2:
                    continue

                polygon, (text, confidence) = line

                # Convert polygon to bounding box
                xs = [p[0] for p in polygon]
                ys = [p[1] for p in polygon]

                x = int(min(xs))
                y = int(min(ys))
                width = int(max(xs) - min(xs))
                height = int(max(ys) - min(ys))

                # Convert polygon to list of tuples
                polygon_tuples = [(int(p[0]), int(p[1])) for p in polygon]

                regions.append(
                    TextRegion(
                        text=text,
                        confidence=confidence,
                        bbox=(x, y, width, height),
                        polygon=polygon_tuples,
                    )
                )

        return regions

    @property
    def name(self) -> str:
        return "paddleocr"
