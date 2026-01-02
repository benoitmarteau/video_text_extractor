"""EasyOCR engine implementation."""

from typing import List, Optional

import numpy as np

from video_ocr.engines.base import BaseOCREngine, TextRegion
from video_ocr.utils.logging_config import get_logger


class EasyOCREngine(BaseOCREngine):
    """OCR engine using EasyOCR."""

    def __init__(
        self,
        languages: List[str] = None,
        confidence_threshold: float = 0.5,
        gpu: bool = True,
    ):
        """
        Initialize EasyOCR engine.

        Args:
            languages: List of language codes
            confidence_threshold: Minimum confidence threshold
            gpu: Use GPU if available
        """
        super().__init__(languages, confidence_threshold, gpu)
        self._reader: Optional["easyocr.Reader"] = None
        self.logger = get_logger()

    def _initialize(self) -> None:
        """Initialize the EasyOCR reader."""
        import easyocr

        self.logger.info(
            f"Initializing EasyOCR with languages: {self.languages}, GPU: {self.gpu}"
        )

        self._reader = easyocr.Reader(
            self.languages,
            gpu=self.gpu,
            verbose=False,
        )

        self.logger.info("EasyOCR initialized successfully")

    def _process_frame(self, frame: np.ndarray) -> List[TextRegion]:
        """
        Process a frame using EasyOCR.

        Args:
            frame: BGR numpy array

        Returns:
            List of TextRegion objects
        """
        if self._reader is None:
            raise RuntimeError("EasyOCR not initialized")

        # EasyOCR expects RGB format
        from video_ocr.utils.image_utils import frame_to_rgb

        rgb_frame = frame_to_rgb(frame)

        # Run OCR
        results = self._reader.readtext(rgb_frame)

        regions = []
        for detection in results:
            # EasyOCR returns: [polygon, text, confidence]
            polygon, text, confidence = detection

            # Convert polygon to bounding box
            # polygon is [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
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
        return "easyocr"
