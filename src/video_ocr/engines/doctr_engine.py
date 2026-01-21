"""docTR OCR engine implementation.

docTR (Document Text Recognition) by Mindee provides state-of-the-art
text detection and recognition with a 2-stage pipeline.

References:
- https://github.com/mindee/doctr
- https://mindee.github.io/doctr/
"""

from typing import List, Optional

import numpy as np

from video_ocr.engines.base import BaseOCREngine, TextRegion
from video_ocr.utils.logging_config import get_logger


class DocTREngine(BaseOCREngine):
    """OCR engine using docTR (Mindee Document Text Recognition)."""

    # docTR works on both CPU and GPU
    REQUIRES_GPU = False

    def __init__(
        self,
        languages: List[str] = None,
        confidence_threshold: float = 0.5,
        gpu: bool = True,
        det_arch: str = "db_resnet50",
        reco_arch: str = "crnn_vgg16_bn",
    ):
        """
        Initialize docTR OCR engine.

        Args:
            languages: List of language codes (limited support)
            confidence_threshold: Minimum confidence threshold
            gpu: Use GPU if available
            det_arch: Detection architecture (db_resnet50, db_mobilenet_v3_large)
            reco_arch: Recognition architecture (crnn_vgg16_bn, master, vitstr_small)
        """
        super().__init__(languages, confidence_threshold, gpu)
        self._predictor = None
        self._det_arch = det_arch
        self._reco_arch = reco_arch
        self.logger = get_logger()

    def _initialize(self) -> None:
        """Initialize docTR predictor."""
        try:
            from doctr.models import ocr_predictor
        except ImportError:
            raise ImportError(
                "docTR is not installed. Install with: pip install python-doctr[torch]"
            )

        self.logger.info(
            f"Initializing docTR OCR (det={self._det_arch}, reco={self._reco_arch}), GPU: {self.gpu}"
        )

        # Set device preference before loading
        if self.gpu:
            try:
                import torch
                if torch.cuda.is_available():
                    # docTR will auto-detect GPU
                    pass
                else:
                    self.logger.warning("GPU requested but CUDA not available")
            except ImportError:
                pass

        # Create predictor
        self._predictor = ocr_predictor(
            det_arch=self._det_arch,
            reco_arch=self._reco_arch,
            pretrained=True,
            assume_straight_pages=True,  # Faster for screen recordings
        )

        # Move to GPU if available
        if self.gpu:
            try:
                import torch
                if torch.cuda.is_available():
                    self._predictor = self._predictor.cuda()
                    self.logger.info("docTR loaded on GPU")
            except Exception as e:
                self.logger.warning(f"Could not move to GPU: {e}")

        self.logger.info("docTR OCR initialized successfully")

    def _process_frame(self, frame: np.ndarray) -> List[TextRegion]:
        """
        Process a frame using docTR.

        Args:
            frame: BGR numpy array

        Returns:
            List of TextRegion objects
        """
        if self._predictor is None:
            raise RuntimeError("docTR not initialized")

        from doctr.io import DocumentFile

        # Convert BGR to RGB
        from video_ocr.utils.image_utils import frame_to_rgb
        rgb_frame = frame_to_rgb(frame)

        # docTR expects a document, create from numpy array
        # Convert to PIL and back to ensure correct format
        from PIL import Image
        import tempfile
        import os

        pil_image = Image.fromarray(rgb_frame)

        # docTR can process PIL images directly in newer versions
        try:
            # Try direct numpy processing
            doc = DocumentFile.from_images([rgb_frame])
        except Exception:
            # Fallback: save to temp file
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                tmp_path = tmp.name
                pil_image.save(tmp_path)

            try:
                doc = DocumentFile.from_images([tmp_path])
            finally:
                os.unlink(tmp_path)

        # Run OCR
        result = self._predictor(doc)

        regions = []
        export = result.export()

        # Parse the hierarchical result structure
        # Structure: pages -> blocks -> lines -> words
        if 'pages' in export:
            for page in export['pages']:
                page_height = page.get('dimensions', [1, 1])[0]
                page_width = page.get('dimensions', [1, 1])[1]

                for block in page.get('blocks', []):
                    for line in block.get('lines', []):
                        for word in line.get('words', []):
                            text = word.get('value', '')
                            confidence = word.get('confidence', 0.0)

                            if not text or not text.strip():
                                continue

                            # Geometry is normalized (0-1), convert to pixels
                            geo = word.get('geometry', [[0, 0], [1, 1]])
                            if len(geo) >= 2:
                                x1 = int(geo[0][0] * frame.shape[1])
                                y1 = int(geo[0][1] * frame.shape[0])
                                x2 = int(geo[1][0] * frame.shape[1])
                                y2 = int(geo[1][1] * frame.shape[0])

                                width = max(1, x2 - x1)
                                height = max(1, y2 - y1)

                                regions.append(
                                    TextRegion(
                                        text=text.strip(),
                                        confidence=float(confidence),
                                        bbox=(x1, y1, width, height),
                                    )
                                )

        return regions

    @property
    def name(self) -> str:
        return "doctr"
