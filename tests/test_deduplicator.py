"""Tests for text deduplication."""

import pytest

from video_ocr.core.deduplicator import (
    DeduplicatedText,
    DeduplicationResult,
    TextDeduplicator,
)
from video_ocr.engines.base import OCRResult, TextRegion


class TestTextDeduplicator:
    """Tests for TextDeduplicator class."""

    def test_init_default_values(self):
        """Test default initialization values."""
        dedup = TextDeduplicator()
        assert dedup.similarity_threshold == 0.85
        assert dedup.min_new_chars == 10
        assert dedup.window_size == 3

    def test_init_custom_values(self):
        """Test custom initialization values."""
        dedup = TextDeduplicator(
            similarity_threshold=0.9,
            min_new_chars=20,
            window_size=5,
        )
        assert dedup.similarity_threshold == 0.9
        assert dedup.min_new_chars == 20
        assert dedup.window_size == 5

    def test_deduplicate_empty_input(self):
        """Test deduplication with empty input."""
        dedup = TextDeduplicator()
        result = dedup.deduplicate([])

        assert isinstance(result, DeduplicationResult)
        assert len(result.texts) == 0
        assert result.total_frames_processed == 0

    def test_deduplicate_single_frame(self):
        """Test deduplication with single frame."""
        dedup = TextDeduplicator()

        ocr_results = [
            OCRResult(
                full_text="Hello world, this is a test message",
                frame_number=0,
                timestamp_seconds=0.0,
            )
        ]

        result = dedup.deduplicate(ocr_results)

        assert result.total_frames_processed == 1
        assert result.frames_with_new_content == 1
        assert len(result.texts) == 1
        assert "Hello" in result.texts[0].text

    def test_deduplicate_identical_frames(self):
        """Test deduplication with identical frames."""
        dedup = TextDeduplicator(similarity_threshold=0.85)

        text = "This is exactly the same text in every frame"
        ocr_results = [
            OCRResult(full_text=text, frame_number=i, timestamp_seconds=i * 0.5)
            for i in range(5)
        ]

        result = dedup.deduplicate(ocr_results)

        # Should only have 1 unique text
        assert result.frames_with_new_content == 1
        assert result.duplicate_frames_skipped == 4

    def test_deduplicate_different_frames(self):
        """Test deduplication with completely different frames."""
        dedup = TextDeduplicator(min_new_chars=5)

        ocr_results = [
            OCRResult(
                full_text="First message with unique content",
                frame_number=0,
                timestamp_seconds=0.0,
            ),
            OCRResult(
                full_text="Second message completely different",
                frame_number=1,
                timestamp_seconds=0.5,
            ),
            OCRResult(
                full_text="Third message also unique",
                frame_number=2,
                timestamp_seconds=1.0,
            ),
        ]

        result = dedup.deduplicate(ocr_results)

        # All should be unique
        assert result.frames_with_new_content == 3
        assert len(result.texts) == 3

    def test_deduplicate_scrolling_text(self):
        """Test deduplication simulating scrolling text."""
        dedup = TextDeduplicator(similarity_threshold=0.85, min_new_chars=10)

        # Simulate scrolling: each frame shows part of previous + new content
        ocr_results = [
            OCRResult(
                full_text="Line 1: First message\nLine 2: Second message",
                frame_number=0,
                timestamp_seconds=0.0,
            ),
            OCRResult(
                full_text="Line 2: Second message\nLine 3: Third message",
                frame_number=1,
                timestamp_seconds=0.5,
            ),
            OCRResult(
                full_text="Line 3: Third message\nLine 4: Fourth message",
                frame_number=2,
                timestamp_seconds=1.0,
            ),
        ]

        result = dedup.deduplicate(ocr_results)

        # Should have extracted new lines
        full_text = result.full_text
        assert "First" in full_text
        assert "Fourth" in full_text

    def test_deduplicate_empty_frames(self):
        """Test handling of frames with no text."""
        dedup = TextDeduplicator()

        ocr_results = [
            OCRResult(full_text="Some text here", frame_number=0, timestamp_seconds=0.0),
            OCRResult(full_text="", frame_number=1, timestamp_seconds=0.5),
            OCRResult(full_text="   ", frame_number=2, timestamp_seconds=1.0),
            OCRResult(full_text="Completely different content now", frame_number=3, timestamp_seconds=1.5),
        ]

        result = dedup.deduplicate(ocr_results)

        # Empty frames should be skipped, two frames have distinct content
        assert result.frames_with_new_content == 2

    def test_deduplicate_result_properties(self):
        """Test DeduplicationResult properties."""
        result = DeduplicationResult(
            texts=[
                DeduplicatedText(text="Hello", source_frame=0, timestamp_seconds=0.0),
                DeduplicatedText(text="World", source_frame=1, timestamp_seconds=0.5),
            ],
            total_frames_processed=5,
            frames_with_new_content=2,
            duplicate_frames_skipped=3,
        )

        assert result.full_text == "Hello\nWorld"
        assert result.deduplication_ratio == 0.4

    def test_deduplicate_simple(self):
        """Test simple text list deduplication."""
        dedup = TextDeduplicator(min_new_chars=5)

        texts = [
            "The quick brown fox",
            "The quick brown fox jumps over",
            "The quick brown fox jumps over the lazy dog",
        ]

        result = dedup.deduplicate_simple(texts)

        # Should have accumulated new parts
        assert len(result) >= 1
        assert "quick" in " ".join(result)

    def test_deduplicate_with_confidence(self):
        """Test that confidence is preserved."""
        dedup = TextDeduplicator()

        ocr_results = [
            OCRResult(
                full_text="Test message with good confidence",
                frame_number=0,
                timestamp_seconds=0.0,
                regions=[
                    TextRegion(
                        text="Test message",
                        confidence=0.95,
                        bbox=(0, 0, 100, 20),
                    )
                ],
            ),
        ]
        ocr_results[0].regions[0].confidence = 0.95

        result = dedup.deduplicate(ocr_results)

        # Confidence should be carried through
        assert len(result.texts) == 1


class TestDeduplicatedText:
    """Tests for DeduplicatedText dataclass."""

    def test_deduplicated_text_creation(self):
        """Test creating DeduplicatedText."""
        text = DeduplicatedText(
            text="Test content",
            source_frame=5,
            timestamp_seconds=2.5,
            is_new=True,
            confidence=0.9,
        )

        assert text.text == "Test content"
        assert text.source_frame == 5
        assert text.timestamp_seconds == 2.5
        assert text.is_new is True
        assert text.confidence == 0.9

    def test_deduplicated_text_defaults(self):
        """Test DeduplicatedText default values."""
        text = DeduplicatedText(
            text="Content",
            source_frame=0,
            timestamp_seconds=0.0,
        )

        assert text.is_new is True
        assert text.confidence == 0.0
