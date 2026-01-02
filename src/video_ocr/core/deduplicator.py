"""Text deduplication for scrolling video content."""

from dataclasses import dataclass, field
from typing import List, Optional

from video_ocr.engines.base import OCRResult
from video_ocr.utils.logging_config import get_logger
from video_ocr.utils.similarity import (
    find_new_content,
    levenshtein_ratio,
    normalize_text,
)


@dataclass
class DeduplicatedText:
    """Result of text deduplication."""

    text: str
    source_frame: int
    timestamp_seconds: float
    is_new: bool = True
    confidence: float = 0.0


@dataclass
class DeduplicationResult:
    """Complete result of deduplication process."""

    texts: List[DeduplicatedText] = field(default_factory=list)
    total_frames_processed: int = 0
    frames_with_new_content: int = 0
    duplicate_frames_skipped: int = 0

    @property
    def full_text(self) -> str:
        """Get all deduplicated text joined together."""
        return "\n".join(t.text for t in self.texts if t.text.strip())

    @property
    def deduplication_ratio(self) -> float:
        """Ratio of frames that had new content."""
        if self.total_frames_processed == 0:
            return 0.0
        return self.frames_with_new_content / self.total_frames_processed


class DeduplicationError(Exception):
    """Error during text deduplication."""

    pass


class TextDeduplicator:
    """
    Deduplicate text from scrolling video content.

    Uses fuzzy string matching to detect and remove duplicate text
    that appears across multiple frames as the content scrolls.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.85,
        min_new_chars: int = 10,
        window_size: int = 3,
    ):
        """
        Initialize text deduplicator.

        Args:
            similarity_threshold: Levenshtein ratio threshold for duplicates (0-1)
            min_new_chars: Minimum new characters to consider as new content
            window_size: Number of previous frames to compare against
        """
        self.similarity_threshold = similarity_threshold
        self.min_new_chars = min_new_chars
        self.window_size = window_size
        self.logger = get_logger()

    def deduplicate(
        self,
        ocr_results: List[OCRResult],
    ) -> DeduplicationResult:
        """
        Deduplicate text from a sequence of OCR results.

        Args:
            ocr_results: List of OCRResult objects from consecutive frames

        Returns:
            DeduplicationResult with unique text content
        """
        if not ocr_results:
            return DeduplicationResult()

        result = DeduplicationResult(total_frames_processed=len(ocr_results))

        # Track accumulated text for comparison
        accumulated_text = ""
        recent_texts: List[str] = []

        for ocr_result in ocr_results:
            current_text = ocr_result.full_text.strip()

            if not current_text:
                result.duplicate_frames_skipped += 1
                continue

            # Check if this frame's text is significantly different
            is_new, new_content = self._find_new_content(
                current_text,
                accumulated_text,
                recent_texts,
            )

            if is_new and new_content and len(new_content.strip()) >= self.min_new_chars:
                result.texts.append(
                    DeduplicatedText(
                        text=new_content,
                        source_frame=ocr_result.frame_number,
                        timestamp_seconds=ocr_result.timestamp_seconds,
                        is_new=True,
                        confidence=ocr_result.average_confidence,
                    )
                )
                result.frames_with_new_content += 1

                # Update accumulated text
                accumulated_text += " " + new_content
            else:
                result.duplicate_frames_skipped += 1

            # Maintain sliding window of recent texts
            recent_texts.append(current_text)
            if len(recent_texts) > self.window_size:
                recent_texts.pop(0)

        self.logger.info(
            f"Deduplication complete: {result.frames_with_new_content}/{result.total_frames_processed} "
            f"frames had new content (ratio: {result.deduplication_ratio:.2f})"
        )

        return result

    def _find_new_content(
        self,
        current_text: str,
        accumulated_text: str,
        recent_texts: List[str],
    ) -> tuple[bool, Optional[str]]:
        """
        Find new content in current text that doesn't appear in previous text.

        Args:
            current_text: Text from current frame
            accumulated_text: All accumulated text so far
            recent_texts: Recent frame texts for comparison

        Returns:
            Tuple of (is_new, new_content)
        """
        # Quick check: if current text is very similar to accumulated, skip
        if accumulated_text:
            overall_sim = levenshtein_ratio(current_text, accumulated_text[-len(current_text) * 2 :])
            if overall_sim > self.similarity_threshold:
                return False, None

        # Check against recent frames
        for recent in recent_texts:
            sim = levenshtein_ratio(current_text, recent)
            if sim > self.similarity_threshold:
                # Very similar to a recent frame
                return False, None

        # Try to find genuinely new content
        new_content = find_new_content(
            accumulated_text,
            current_text,
            threshold=self.similarity_threshold,
        )

        if new_content and len(new_content.strip()) >= self.min_new_chars:
            return True, new_content

        # If no overlap detection worked, check line by line
        new_lines = self._find_new_lines(current_text, accumulated_text, recent_texts)

        if new_lines:
            return True, "\n".join(new_lines)

        return False, None

    def _find_new_lines(
        self,
        current_text: str,
        accumulated_text: str,
        recent_texts: List[str],
    ) -> List[str]:
        """
        Find lines in current text that are new.

        Args:
            current_text: Text from current frame
            accumulated_text: All accumulated text
            recent_texts: Recent frame texts

        Returns:
            List of new lines
        """
        current_lines = current_text.split("\n")
        all_previous_lines = accumulated_text.split("\n")

        for recent in recent_texts:
            all_previous_lines.extend(recent.split("\n"))

        # Normalize previous lines for comparison
        normalized_previous = {normalize_text(line) for line in all_previous_lines if line.strip()}

        new_lines = []
        for line in current_lines:
            line_stripped = line.strip()
            if not line_stripped:
                continue

            normalized_current = normalize_text(line_stripped)

            # Check for exact or near matches
            is_duplicate = False

            for prev_normalized in normalized_previous:
                if not prev_normalized:
                    continue

                sim = levenshtein_ratio(normalized_current, prev_normalized, normalize=False)
                if sim > self.similarity_threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                new_lines.append(line_stripped)

        return new_lines

    def deduplicate_simple(
        self,
        texts: List[str],
    ) -> List[str]:
        """
        Simple deduplication of a list of text strings.

        Args:
            texts: List of text strings

        Returns:
            List of unique text strings
        """
        if not texts:
            return []

        result = [texts[0]]
        accumulated = texts[0]

        for text in texts[1:]:
            new_content = find_new_content(
                accumulated,
                text,
                threshold=self.similarity_threshold,
            )

            if new_content and len(new_content.strip()) >= self.min_new_chars:
                result.append(new_content)
                accumulated += " " + new_content

        return result
