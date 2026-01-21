"""TrOCR engine implementation.

TrOCR (Transformer-based OCR) by Microsoft uses Vision Transformers
for high-quality text recognition. Best for printed/handwritten text.

Note: TrOCR is a recognition-only model. This implementation uses
a simple text detector (from OpenCV or EasyOCR) for detection.

References:
- https://huggingface.co/docs/transformers/en/model_doc/trocr
- https://github.com/microsoft/unilm/tree/master/trocr
"""

from typing import List, Optional, Tuple

import numpy as np

from video_ocr.engines.base import BaseOCREngine, TextRegion
from video_ocr.utils.logging_config import get_logger


class TrOCREngine(BaseOCREngine):
    """OCR engine using Microsoft TrOCR (Transformer-based OCR)."""

    # TrOCR benefits significantly from GPU
    REQUIRES_GPU = True

    def __init__(
        self,
        languages: List[str] = None,
        confidence_threshold: float = 0.5,
        gpu: bool = True,
        model_type: str = "printed",  # "printed" or "handwritten"
        model_size: str = "base",  # "base" or "large"
    ):
        """
        Initialize TrOCR engine.

        Args:
            languages: List of language codes (TrOCR is primarily English)
            confidence_threshold: Minimum confidence threshold
            gpu: Use GPU (strongly recommended)
            model_type: "printed" for typed text, "handwritten" for handwriting
            model_size: "base" or "large" (large is more accurate but slower)
        """
        super().__init__(languages, confidence_threshold, gpu)
        self._processor = None
        self._model = None
        self._text_detector = None
        self._model_type = model_type
        self._model_size = model_size
        self.logger = get_logger()

    def _get_model_name(self) -> str:
        """Get the HuggingFace model name based on settings."""
        return f"microsoft/trocr-{self._model_size}-{self._model_type}"

    def _initialize(self) -> None:
        """Initialize TrOCR model and text detector."""
        try:
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        except ImportError:
            raise ImportError(
                "Transformers is not installed. Install with: pip install transformers"
            )

        model_name = self._get_model_name()
        self.logger.info(f"Initializing TrOCR ({model_name}), GPU: {self.gpu}")

        # Load processor and model
        self._processor = TrOCRProcessor.from_pretrained(model_name)
        self._model = VisionEncoderDecoderModel.from_pretrained(model_name)

        # Move to GPU if available
        if self.gpu:
            try:
                import torch
                if torch.cuda.is_available():
                    self._model = self._model.to("cuda")
                    self.logger.info("TrOCR loaded on GPU")
                else:
                    self.logger.warning("GPU requested but CUDA not available")
            except Exception as e:
                self.logger.warning(f"Could not move to GPU: {e}")

        # Initialize simple text detector using OpenCV EAST or MSER
        self._init_text_detector()

        self.logger.info("TrOCR initialized successfully")

    def _init_text_detector(self) -> None:
        """Initialize a simple text detector for TrOCR."""
        # We'll use OpenCV's MSER for text region detection
        # It's fast and doesn't require additional models
        import cv2
        self._mser = cv2.MSER_create()
        self._mser.setMinArea(100)
        self._mser.setMaxArea(50000)

    def _detect_text_regions(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect text regions in the frame using MSER.

        Args:
            frame: BGR numpy array

        Returns:
            List of bounding boxes (x, y, width, height)
        """
        import cv2

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect regions
        regions, _ = self._mser.detectRegions(gray)

        # Convert regions to bounding boxes and merge overlapping ones
        bboxes = []
        for region in regions:
            x, y, w, h = cv2.boundingRect(region)
            if w > 10 and h > 5:  # Filter tiny regions
                bboxes.append((x, y, w, h))

        # Merge overlapping/close bounding boxes
        merged = self._merge_bboxes(bboxes, frame.shape)

        return merged

    def _merge_bboxes(
        self,
        bboxes: List[Tuple[int, int, int, int]],
        frame_shape: Tuple[int, int, int],
    ) -> List[Tuple[int, int, int, int]]:
        """Merge overlapping bounding boxes into text lines."""
        if not bboxes:
            return []

        import cv2

        # Sort by y, then x
        bboxes = sorted(bboxes, key=lambda b: (b[1], b[0]))

        # Group into lines (boxes with similar y coordinates)
        lines = []
        current_line = [bboxes[0]]
        current_y = bboxes[0][1]

        for bbox in bboxes[1:]:
            # If y is close enough, same line
            if abs(bbox[1] - current_y) < 20:
                current_line.append(bbox)
            else:
                lines.append(current_line)
                current_line = [bbox]
                current_y = bbox[1]

        lines.append(current_line)

        # Merge boxes within each line
        merged = []
        for line in lines:
            if not line:
                continue

            # Sort line by x
            line = sorted(line, key=lambda b: b[0])

            # Merge into line bounding box
            min_x = min(b[0] for b in line)
            min_y = min(b[1] for b in line)
            max_x = max(b[0] + b[2] for b in line)
            max_y = max(b[1] + b[3] for b in line)

            # Add padding
            padding = 5
            min_x = max(0, min_x - padding)
            min_y = max(0, min_y - padding)
            max_x = min(frame_shape[1], max_x + padding)
            max_y = min(frame_shape[0], max_y + padding)

            width = max_x - min_x
            height = max_y - min_y

            if width > 20 and height > 10:
                merged.append((min_x, min_y, width, height))

        return merged

    def _recognize_text(self, image_crop: np.ndarray) -> Tuple[str, float]:
        """
        Recognize text in a cropped image region using TrOCR.

        Args:
            image_crop: RGB numpy array of text region

        Returns:
            Tuple of (text, confidence)
        """
        from PIL import Image
        import torch

        # Convert to PIL Image
        pil_image = Image.fromarray(image_crop)

        # Process image
        pixel_values = self._processor(pil_image, return_tensors="pt").pixel_values

        # Move to GPU if needed
        if self.gpu and torch.cuda.is_available():
            pixel_values = pixel_values.to("cuda")

        # Generate text
        with torch.no_grad():
            generated_ids = self._model.generate(
                pixel_values,
                max_length=64,
                num_beams=4,
                return_dict_in_generate=True,
                output_scores=True,
            )

        # Decode text
        text = self._processor.batch_decode(
            generated_ids.sequences, skip_special_tokens=True
        )[0]

        # Estimate confidence from generation scores
        # This is approximate since TrOCR doesn't provide direct confidence
        if hasattr(generated_ids, 'sequences_scores') and generated_ids.sequences_scores is not None:
            confidence = torch.sigmoid(generated_ids.sequences_scores[0]).item()
        else:
            confidence = 0.85  # Default confidence

        return text.strip(), confidence

    def _process_frame(self, frame: np.ndarray) -> List[TextRegion]:
        """
        Process a frame using TrOCR.

        Args:
            frame: BGR numpy array

        Returns:
            List of TextRegion objects
        """
        if self._model is None or self._processor is None:
            raise RuntimeError("TrOCR not initialized")

        # Convert BGR to RGB
        from video_ocr.utils.image_utils import frame_to_rgb
        rgb_frame = frame_to_rgb(frame)

        # Detect text regions
        bboxes = self._detect_text_regions(frame)

        regions = []
        for x, y, w, h in bboxes:
            # Crop the region
            crop = rgb_frame[y:y+h, x:x+w]

            if crop.size == 0:
                continue

            # Recognize text
            try:
                text, confidence = self._recognize_text(crop)

                if text and text.strip():
                    regions.append(
                        TextRegion(
                            text=text,
                            confidence=confidence,
                            bbox=(x, y, w, h),
                        )
                    )
            except Exception as e:
                self.logger.debug(f"TrOCR recognition failed for region: {e}")
                continue

        return regions

    @property
    def name(self) -> str:
        return "trocr"
