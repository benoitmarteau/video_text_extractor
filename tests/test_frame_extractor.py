"""Tests for frame extraction."""

from pathlib import Path

import numpy as np
import pytest

from video_ocr.core.frame_extractor import (
    FrameExtractor,
    FrameExtractionError,
    FrameInfo,
)


class TestFrameExtractor:
    """Tests for FrameExtractor class."""

    def test_init_default_values(self):
        """Test default initialization values."""
        extractor = FrameExtractor()
        assert extractor.fps == 2.0
        assert extractor.skip_similar_threshold == 0.95
        assert extractor.max_frames is None

    def test_init_custom_values(self):
        """Test custom initialization values."""
        extractor = FrameExtractor(fps=5.0, skip_similar_threshold=0.9, max_frames=100)
        assert extractor.fps == 5.0
        assert extractor.skip_similar_threshold == 0.9
        assert extractor.max_frames == 100

    def test_validate_video_not_found(self):
        """Test validation with non-existent file."""
        extractor = FrameExtractor()
        is_valid, error = extractor.validate_video(Path("nonexistent.mp4"))
        assert not is_valid
        assert "not found" in error.lower()

    def test_validate_video_unsupported_format(self, temp_dir):
        """Test validation with unsupported format."""
        bad_file = temp_dir / "test.xyz"
        bad_file.write_text("not a video")

        extractor = FrameExtractor()
        is_valid, error = extractor.validate_video(bad_file)
        assert not is_valid
        assert "unsupported" in error.lower()

    def test_validate_video_success(self, sample_video):
        """Test successful video validation."""
        extractor = FrameExtractor()
        is_valid, error = extractor.validate_video(sample_video)
        assert is_valid
        assert error == ""

    def test_get_video_info(self, sample_video):
        """Test getting video information."""
        extractor = FrameExtractor()
        info = extractor.get_video_info(sample_video)

        assert info["width"] == 640
        assert info["height"] == 480
        assert info["fps"] == 30.0
        assert info["frame_count"] == 60  # 2 seconds * 30 fps
        assert abs(info["duration_seconds"] - 2.0) < 0.1

    def test_extract_frames_generator(self, sample_video):
        """Test frame extraction as generator."""
        extractor = FrameExtractor(fps=2.0, skip_similar_threshold=1.0)  # Don't skip
        frames = list(extractor.extract_frames(sample_video, skip_similar=False))

        # At 2 FPS for 2 seconds, expect ~4 frames
        assert len(frames) >= 3
        assert len(frames) <= 6

        # Check frame info
        for frame in frames:
            assert isinstance(frame, FrameInfo)
            assert frame.frame is not None
            assert frame.frame.shape == (480, 640, 3)
            assert frame.timestamp_seconds >= 0

    def test_extract_frames_with_skip_similar(self, sample_video):
        """Test frame extraction with similar frame skipping."""
        extractor = FrameExtractor(fps=10.0, skip_similar_threshold=0.99)
        frames_skip = list(extractor.extract_frames(sample_video, skip_similar=True))

        extractor2 = FrameExtractor(fps=10.0)
        frames_no_skip = list(extractor2.extract_frames(sample_video, skip_similar=False))

        # With skipping, should have fewer or equal frames
        assert len(frames_skip) <= len(frames_no_skip)

    def test_extract_frames_max_limit(self, sample_video):
        """Test frame extraction with max frames limit."""
        extractor = FrameExtractor(fps=10.0, max_frames=3)
        frames = list(extractor.extract_frames(sample_video, skip_similar=False))

        assert len(frames) == 3

    def test_extract_frames_to_list(self, sample_video):
        """Test extracting frames to list."""
        extractor = FrameExtractor(fps=2.0)
        frames = extractor.extract_frames_to_list(sample_video)

        assert isinstance(frames, list)
        assert len(frames) > 0
        assert all(isinstance(f, FrameInfo) for f in frames)

    def test_frame_info_timestamp_str(self):
        """Test FrameInfo timestamp string conversion."""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        info = FrameInfo(
            frame_number=0,
            timestamp_seconds=3661.5,  # 1:01:01.5
            frame=frame,
        )

        assert info.timestamp_str == "01:01:01"

    def test_extract_single_frame(self, sample_video):
        """Test extracting a single specific frame."""
        extractor = FrameExtractor()
        frame_info = extractor.extract_single_frame(sample_video, frame_number=15)

        assert frame_info is not None
        assert frame_info.frame_number == 15
        assert frame_info.frame.shape == (480, 640, 3)

    def test_extract_single_frame_invalid(self, sample_video):
        """Test extracting frame beyond video length."""
        extractor = FrameExtractor()
        frame_info = extractor.extract_single_frame(sample_video, frame_number=10000)

        assert frame_info is None

    def test_extract_frames_invalid_video(self, temp_dir):
        """Test extraction from invalid video."""
        invalid_file = temp_dir / "invalid.mp4"
        invalid_file.write_bytes(b"not a video")

        extractor = FrameExtractor()

        with pytest.raises(FrameExtractionError):
            list(extractor.extract_frames(invalid_file))
