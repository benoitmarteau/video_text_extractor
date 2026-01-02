"""Base class for OCR engines."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class TextRegion:
    """A detected text region with bounding box."""

    text: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x, y, width, height)

    # Original polygon points (if available)
    polygon: Optional[List[Tuple[int, int]]] = None

    @property
    def x(self) -> int:
        return self.bbox[0]

    @property
    def y(self) -> int:
        return self.bbox[1]

    @property
    def width(self) -> int:
        return self.bbox[2]

    @property
    def height(self) -> int:
        return self.bbox[3]

    @property
    def center(self) -> Tuple[int, int]:
        return (self.x + self.width // 2, self.y + self.height // 2)

    @property
    def area(self) -> int:
        return self.width * self.height


@dataclass
class OCRResult:
    """Result from OCR processing of a single frame."""

    regions: List[TextRegion] = field(default_factory=list)
    full_text: str = ""
    frame_number: int = 0
    timestamp_seconds: float = 0.0
    processing_time_ms: float = 0.0

    @property
    def has_text(self) -> bool:
        return len(self.regions) > 0

    @property
    def average_confidence(self) -> float:
        if not self.regions:
            return 0.0
        return sum(r.confidence for r in self.regions) / len(self.regions)

    def get_text_sorted_by_position(self) -> str:
        """Get text sorted by vertical then horizontal position."""
        if not self.regions:
            return ""

        # Sort by y coordinate, then x coordinate
        sorted_regions = sorted(self.regions, key=lambda r: (r.y, r.x))

        # Group by approximate lines (within 20 pixels)
        lines = []
        current_line = []
        current_y = -100

        for region in sorted_regions:
            if abs(region.y - current_y) > 20:
                if current_line:
                    # Sort current line by x
                    current_line.sort(key=lambda r: r.x)
                    lines.append(" ".join(r.text for r in current_line))
                current_line = [region]
                current_y = region.y
            else:
                current_line.append(region)

        if current_line:
            current_line.sort(key=lambda r: r.x)
            lines.append(" ".join(r.text for r in current_line))

        return "\n".join(lines)


class BaseOCREngine(ABC):
    """Abstract base class for OCR engines."""

    def __init__(
        self,
        languages: List[str] = None,
        confidence_threshold: float = 0.5,
        gpu: bool = True,
    ):
        """
        Initialize OCR engine.

        Args:
            languages: List of language codes to detect
            confidence_threshold: Minimum confidence to accept text
            gpu: Use GPU acceleration if available
        """
        self.languages = languages or ["en"]
        self.confidence_threshold = confidence_threshold
        self.gpu = gpu
        self._initialized = False

    @abstractmethod
    def _initialize(self) -> None:
        """Initialize the OCR engine. Called lazily on first use."""
        pass

    def ensure_initialized(self) -> None:
        """Ensure the engine is initialized."""
        if not self._initialized:
            self._initialize()
            self._initialized = True

    @abstractmethod
    def _process_frame(self, frame: np.ndarray) -> List[TextRegion]:
        """
        Process a single frame and return detected text regions.

        Args:
            frame: BGR numpy array

        Returns:
            List of TextRegion objects
        """
        pass

    def process_frame(
        self,
        frame: np.ndarray,
        frame_number: int = 0,
        timestamp_seconds: float = 0.0,
    ) -> OCRResult:
        """
        Process a frame and return OCR results.

        Args:
            frame: BGR numpy array
            frame_number: Frame number in video
            timestamp_seconds: Timestamp in seconds

        Returns:
            OCRResult with detected text
        """
        import time

        self.ensure_initialized()

        start_time = time.time()

        regions = self._process_frame(frame)

        # Filter by confidence threshold
        regions = [r for r in regions if r.confidence >= self.confidence_threshold]

        processing_time = (time.time() - start_time) * 1000

        # Build full text from regions
        result = OCRResult(
            regions=regions,
            frame_number=frame_number,
            timestamp_seconds=timestamp_seconds,
            processing_time_ms=processing_time,
        )

        result.full_text = result.get_text_sorted_by_position()

        return result

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this OCR engine."""
        pass

    @property
    def is_available(self) -> bool:
        """Check if this engine is available (dependencies installed)."""
        try:
            self.ensure_initialized()
            return True
        except Exception:
            return False
