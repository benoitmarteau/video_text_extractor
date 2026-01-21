"""OCR Engine wrapper with support for multiple backends."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Type

import numpy as np

from video_ocr.engines.base import BaseOCREngine, OCRResult
from video_ocr.engines.easyocr_engine import EasyOCREngine
from video_ocr.utils.logging_config import get_logger


class OCREngineError(Exception):
    """Error during OCR processing."""

    pass


@dataclass
class EngineInfo:
    """Information about an OCR engine."""

    name: str
    display_name: str
    requires_gpu: bool
    description: str
    installed: bool = False


class OCREngine:
    """
    High-level OCR engine wrapper.

    Provides a unified interface for multiple OCR backends.
    """

    # Registry of available engines
    _engines: Dict[str, Type[BaseOCREngine]] = {
        "easyocr": EasyOCREngine,
    }

    # Engine metadata (name -> info)
    _engine_info: Dict[str, EngineInfo] = {
        "easyocr": EngineInfo(
            name="easyocr",
            display_name="EasyOCR",
            requires_gpu=False,
            description="Good balance of speed and accuracy, 70+ languages",
        ),
        "tesseract": EngineInfo(
            name="tesseract",
            display_name="Tesseract",
            requires_gpu=False,
            description="Classic OCR, CPU-only, 100+ languages",
        ),
        "surya": EngineInfo(
            name="surya",
            display_name="Surya",
            requires_gpu=True,
            description="Very fast multilingual OCR, 90+ languages (GPU required)",
        ),
        "doctr": EngineInfo(
            name="doctr",
            display_name="docTR",
            requires_gpu=False,
            description="Document OCR by Mindee, high accuracy",
        ),
        "trocr": EngineInfo(
            name="trocr",
            display_name="TrOCR",
            requires_gpu=True,
            description="Microsoft Transformer OCR, best for printed/handwritten (GPU required)",
        ),
    }

    # Try to register optional engines
    try:
        from video_ocr.engines.tesseract_engine import TesseractEngine

        _engines["tesseract"] = TesseractEngine
    except ImportError:
        pass

    try:
        from video_ocr.engines.surya_engine import SuryaEngine

        _engines["surya"] = SuryaEngine
    except ImportError:
        pass

    try:
        from video_ocr.engines.doctr_engine import DocTREngine

        _engines["doctr"] = DocTREngine
    except ImportError:
        pass

    try:
        from video_ocr.engines.trocr_engine import TrOCREngine

        _engines["trocr"] = TrOCREngine
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
            engine: Engine name ('easyocr', 'tesseract', 'surya', 'doctr', 'trocr')
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

    @classmethod
    def get_engine_info(cls, engine_name: str) -> Optional[EngineInfo]:
        """Get information about a specific engine."""
        info = cls._engine_info.get(engine_name.lower())
        if info:
            # Update installed status
            info.installed = engine_name.lower() in cls._engines
        return info

    @classmethod
    def get_all_engines_info(cls) -> List[EngineInfo]:
        """Get information about all known engines (installed or not)."""
        result = []
        for name, info in cls._engine_info.items():
            info.installed = name in cls._engines
            result.append(info)
        return result

    @classmethod
    def get_available_engines_info(cls) -> List[EngineInfo]:
        """Get information about installed/available engines only."""
        result = []
        for name in cls._engines.keys():
            info = cls._engine_info.get(name)
            if info:
                info.installed = True
                result.append(info)
        return result

    @classmethod
    def engine_requires_gpu(cls, engine_name: str) -> bool:
        """Check if an engine requires GPU."""
        info = cls._engine_info.get(engine_name.lower())
        return info.requires_gpu if info else False
