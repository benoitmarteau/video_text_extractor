"""Surya OCR engine implementation.

Surya is a fast, multilingual OCR engine supporting 90+ languages.
Requires GPU for optimal performance.

References:
- https://github.com/datalab-to/surya
- https://pypi.org/project/surya-ocr/
"""

from typing import List, Optional

import numpy as np

from video_ocr.engines.base import BaseOCREngine, TextRegion
from video_ocr.utils.logging_config import get_logger


class SuryaEngine(BaseOCREngine):
    """OCR engine using Surya (fast multilingual OCR)."""

    # Surya requires GPU for practical use
    REQUIRES_GPU = True

    def __init__(
        self,
        languages: List[str] = None,
        confidence_threshold: float = 0.5,
        gpu: bool = True,
    ):
        """
        Initialize Surya OCR engine.

        Args:
            languages: List of language codes (Surya auto-detects, but can hint)
            confidence_threshold: Minimum confidence threshold
            gpu: Use GPU (strongly recommended for Surya)
        """
        super().__init__(languages, confidence_threshold, gpu)
        self._predictor = None
        self._det_predictor = None
        self.logger = get_logger()

    def _initialize(self) -> None:
        """Initialize Surya models."""
        # Surya 0.17+ API with RecognitionPredictor + DetectionPredictor
        try:
            from surya.recognition import RecognitionPredictor
            from surya.detection import DetectionPredictor

            device = "cuda" if self.gpu else "cpu"
            self.logger.info(f"Initializing Surya OCR (v0.17+ API), device: {device}")

            # Try with FoundationPredictor first (some versions require it)
            try:
                from surya.foundation import FoundationPredictor
                foundation = FoundationPredictor()
                self._predictor = RecognitionPredictor(foundation)
                self.logger.debug("Using FoundationPredictor + RecognitionPredictor")
            except (ImportError, TypeError):
                # Fall back to direct RecognitionPredictor (simpler API)
                self._predictor = RecognitionPredictor()
                self.logger.debug("Using RecognitionPredictor directly")

            self._det_predictor = DetectionPredictor()

            self.logger.info("Surya OCR initialized successfully")
            return

        except ImportError as e:
            self.logger.debug(f"Surya 0.17+ API not available: {e}")

        raise ImportError(
            "Surya OCR is not installed or has incompatible version. "
            "Install with: pip install surya-ocr"
        )

    def _process_frame(self, frame: np.ndarray) -> List[TextRegion]:
        """
        Process a frame using Surya OCR.

        Args:
            frame: BGR numpy array

        Returns:
            List of TextRegion objects
        """
        if self._predictor is None:
            raise RuntimeError("Surya not initialized")

        from PIL import Image
        from video_ocr.utils.image_utils import frame_to_rgb

        # Convert BGR to RGB PIL Image
        rgb_frame = frame_to_rgb(frame)
        pil_image = Image.fromarray(rgb_frame)

        # Run OCR using the predictor with detection
        # The predictor returns a list of OCRResult objects
        # API: recognition_predictor(images, langs_list, det_predictor)
        # langs_list is a list of language lists, one per image
        try:
            # Try with language hints first
            predictions = self._predictor([pil_image], [self.languages], self._det_predictor)
        except TypeError:
            # Fall back to keyword argument style
            predictions = self._predictor([pil_image], det_predictor=self._det_predictor)

        regions = []
        if predictions and len(predictions) > 0:
            pred = predictions[0]

            # Get text lines from the prediction
            text_lines = getattr(pred, 'text_lines', None) or getattr(pred, 'lines', [])

            for line in text_lines:
                # Extract text
                text = getattr(line, 'text', '')
                if not text:
                    # Try to get text from chars if available
                    chars = getattr(line, 'chars', [])
                    if chars:
                        text = ''.join(getattr(c, 'char', '') for c in chars)

                if not text or not text.strip():
                    continue

                # Extract confidence
                confidence = getattr(line, 'confidence', 0.9)

                # Extract bounding box
                bbox = getattr(line, 'bbox', None)
                if bbox is None:
                    # Try polygon
                    polygon = getattr(line, 'polygon', None)
                    if polygon and len(polygon) >= 4:
                        xs = [p[0] if isinstance(p, (list, tuple)) else getattr(p, 'x', 0) for p in polygon]
                        ys = [p[1] if isinstance(p, (list, tuple)) else getattr(p, 'y', 0) for p in polygon]
                        bbox = [min(xs), min(ys), max(xs), max(ys)]

                # Get bounding box coordinates
                if bbox is not None and hasattr(bbox, '__iter__') and len(bbox) >= 4:
                    x, y = int(bbox[0]), int(bbox[1])
                    width = int(bbox[2] - bbox[0])
                    height = int(bbox[3] - bbox[1])
                else:
                    x, y, width, height = 0, 0, 100, 20

                regions.append(
                    TextRegion(
                        text=text.strip(),
                        confidence=float(confidence),
                        bbox=(x, y, max(1, width), max(1, height)),
                    )
                )

        return regions

    @property
    def name(self) -> str:
        return "surya"
