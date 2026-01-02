"""Tesseract OCR engine implementation."""

from typing import List

import numpy as np

from video_ocr.engines.base import BaseOCREngine, TextRegion
from video_ocr.utils.logging_config import get_logger


class TesseractEngine(BaseOCREngine):
    """OCR engine using Tesseract via pytesseract."""

    def __init__(
        self,
        languages: List[str] = None,
        confidence_threshold: float = 0.5,
        gpu: bool = False,  # Tesseract doesn't use GPU
    ):
        """
        Initialize Tesseract engine.

        Args:
            languages: List of language codes
            confidence_threshold: Minimum confidence threshold
            gpu: Ignored (Tesseract doesn't use GPU)
        """
        super().__init__(languages, confidence_threshold, gpu=False)
        self.logger = get_logger()
        self._tesseract_lang: str = ""

    def _initialize(self) -> None:
        """Initialize Tesseract."""
        import pytesseract

        # Test that tesseract is available
        try:
            pytesseract.get_tesseract_version()
        except Exception as e:
            raise RuntimeError(
                f"Tesseract is not installed or not in PATH: {e}"
            )

        # Build language string for Tesseract
        self._tesseract_lang = "+".join(self.languages)

        self.logger.info(
            f"Initialized Tesseract with languages: {self._tesseract_lang}"
        )

    def _process_frame(self, frame: np.ndarray) -> List[TextRegion]:
        """
        Process a frame using Tesseract.

        Args:
            frame: BGR numpy array

        Returns:
            List of TextRegion objects
        """
        import cv2
        import pytesseract
        from PIL import Image

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert to PIL Image
        pil_image = Image.fromarray(rgb_frame)

        # Get detailed OCR data
        data = pytesseract.image_to_data(
            pil_image,
            lang=self._tesseract_lang,
            output_type=pytesseract.Output.DICT,
        )

        regions = []

        n_boxes = len(data["text"])
        for i in range(n_boxes):
            text = data["text"][i].strip()

            if not text:
                continue

            # Tesseract confidence is 0-100
            confidence = float(data["conf"][i]) / 100.0

            if confidence < 0:  # Tesseract uses -1 for invalid
                continue

            x = data["left"][i]
            y = data["top"][i]
            width = data["width"][i]
            height = data["height"][i]

            regions.append(
                TextRegion(
                    text=text,
                    confidence=confidence,
                    bbox=(x, y, width, height),
                )
            )

        return regions

    @property
    def name(self) -> str:
        return "tesseract"
