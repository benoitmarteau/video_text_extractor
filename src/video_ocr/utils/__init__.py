"""Utility modules for Video OCR."""

from video_ocr.utils.logging_config import setup_logging, get_logger
from video_ocr.utils.similarity import (
    levenshtein_ratio,
    find_new_content,
    normalize_text,
)
from video_ocr.utils.image_utils import (
    calculate_frame_similarity,
    preprocess_frame,
)

__all__ = [
    "setup_logging",
    "get_logger",
    "levenshtein_ratio",
    "find_new_content",
    "normalize_text",
    "calculate_frame_similarity",
    "preprocess_frame",
]
