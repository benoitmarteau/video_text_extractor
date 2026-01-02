"""OCR Engine wrapper with support for multiple backends."""

from typing import Dict, List, Optional, Type

import numpy as np

from video_ocr.engines.base import BaseOCREngine, OCRResult
from video_ocr.engines.easyocr_engine import EasyOCREngine
from video_ocr.utils.logging_config import get_logger


class OCREngineError(Exception):
    """Error during OCR processing."""

    pass


class OCREngine:
    """
    High-level OCR engine wrapper.

    Provides a unified interface for multiple OCR backends.
    """

    # Registry of available engines
    _engines: Dict[str, Type[BaseOCREngine]] = {
        "easyocr": EasyOCREngine,
    }

    # Try to register optional engines
    try:
        from video_ocr.engines.paddleocr_engine import PaddleOCREngine

        _engines["paddleocr"] = PaddleOCREngine
    except ImportError:
        pass

    try:
        from video_ocr.engines.tesseract_engine import TesseractEngine

        _engines["tesseract"] = TesseractEngine
    except ImportError:
        pass

    def __init__(
        self,
        engine: str = "easyocr",
        languages: List[str] = None,
        confidence_threshold: float = 0.5,
        gpu: bool = True,
    ):
        """
        Initialize OCR engine.

        Args:
            engine: Engine name ('easyocr', 'paddleocr', 'tesseract')
            languages: List of language codes
            confidence_threshold: Minimum confidence to accept
            gpu: Use GPU acceleration if available
        """
        self.logger = get_logger()
        self.engine_name = engine.lower()
        self.languages = languages or ["en"]
        self.confidence_threshold = confidence_threshold
        self.gpu = gpu

        self._engine: Optional[BaseOCREngine] = None

    def _create_engine(self) -> BaseOCREngine:
        """Create the OCR engine instance."""
        if self.engine_name not in self._engines:
            available = ", ".join(self._engines.keys())
            raise OCREngineError(
                f"Unknown engine: {self.engine_name}. Available: {available}"
            )

        engine_class = self._engines[self.engine_name]

        self.logger.info(f"Creating OCR engine: {self.engine_name}")

        return engine_class(
            languages=self.languages,
            confidence_threshold=self.confidence_threshold,
            gpu=self.gpu,
        )

    @property
    def engine(self) -> BaseOCREngine:
        """Get the OCR engine (lazy initialization)."""
        if self._engine is None:
            self._engine = self._create_engine()
        return self._engine

    def process_frame(
        self,
        frame: np.ndarray,
        frame_number: int = 0,
        timestamp_seconds: float = 0.0,
    ) -> OCRResult:
        """
        Process a single frame.

        Args:
            frame: BGR numpy array
            frame_number: Frame number in video
            timestamp_seconds: Timestamp in seconds

        Returns:
            OCRResult with detected text
        """
        try:
            return self.engine.process_frame(
                frame,
                frame_number=frame_number,
                timestamp_seconds=timestamp_seconds,
            )
        except Exception as e:
            self.logger.error(f"OCR error on frame {frame_number}: {e}")
            raise OCREngineError(f"OCR processing failed: {e}") from e

    def process_frames(
        self,
        frames: List[np.ndarray],
        frame_numbers: List[int] = None,
        timestamps: List[float] = None,
    ) -> List[OCRResult]:
        """
        Process multiple frames.

        Args:
            frames: List of BGR numpy arrays
            frame_numbers: Optional list of frame numbers
            timestamps: Optional list of timestamps

        Returns:
            List of OCRResult objects
        """
        if frame_numbers is None:
            frame_numbers = list(range(len(frames)))
        if timestamps is None:
            timestamps = [0.0] * len(frames)

        results = []
        for i, frame in enumerate(frames):
            result = self.process_frame(
                frame,
                frame_number=frame_numbers[i],
                timestamp_seconds=timestamps[i],
            )
            results.append(result)

        return results

    @classmethod
    def available_engines(cls) -> List[str]:
        """Get list of available engine names."""
        return list(cls._engines.keys())

    @classmethod
    def is_engine_available(cls, engine_name: str) -> bool:
        """Check if an engine is available."""
        return engine_name.lower() in cls._engines
