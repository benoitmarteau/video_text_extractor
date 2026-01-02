"""Pytest configuration and fixtures."""

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_frame():
    """Create a sample frame with text for testing."""
    # Create a white image
    frame = np.ones((480, 640, 3), dtype=np.uint8) * 255

    # Add some black text
    cv2.putText(
        frame,
        "Hello World",
        (50, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        (0, 0, 0),
        2,
    )
    cv2.putText(
        frame,
        "11:30:00",
        (50, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 0),
        1,
    )
    cv2.putText(
        frame,
        "Speaker Name",
        (50, 200),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 0),
        1,
    )
    cv2.putText(
        frame,
        "This is a test message",
        (50, 250),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        1,
    )

    return frame


@pytest.fixture
def sample_video(temp_dir):
    """Create a sample video file for testing."""
    video_path = temp_dir / "test_video.mp4"

    # Video parameters
    width, height = 640, 480
    fps = 30
    duration_seconds = 2
    total_frames = fps * duration_seconds

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))

    # Generate frames with scrolling text
    for i in range(total_frames):
        frame = np.ones((height, width, 3), dtype=np.uint8) * 255

        # Simulate scrolling by changing y position
        y_offset = int((i / total_frames) * 200)

        cv2.putText(
            frame,
            f"Frame {i}",
            (50, 100 - y_offset % 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 0),
            2,
        )

        cv2.putText(
            frame,
            "Test Speaker",
            (50, 200 - y_offset % 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            1,
        )

        cv2.putText(
            frame,
            "11:30:00",
            (50, 240 - y_offset % 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            1,
        )

        cv2.putText(
            frame,
            "This is message content that scrolls",
            (50, 280 - y_offset % 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
        )

        writer.write(frame)

    writer.release()

    return video_path


@pytest.fixture
def sample_ocr_text():
    """Sample OCR text for transcript parsing tests."""
    return """John Smith
11:30:00
Hello everyone, welcome to the meeting.

Jane Doe
11:30:15
Thanks for joining us today.

John Smith
11:30:30
Let's begin with the agenda.
We have three topics to discuss today.

Jane Doe
11:31:00
First topic is the project update.
"""


@pytest.fixture
def sample_transcript_entries():
    """Sample parsed transcript entries."""
    from video_ocr.core.transcript_parser import TranscriptEntry

    return [
        TranscriptEntry(
            speaker="John Smith",
            timestamp="11:30:00",
            timestamp_seconds=41400,
            text="Hello everyone, welcome to the meeting.",
            confidence=0.95,
        ),
        TranscriptEntry(
            speaker="Jane Doe",
            timestamp="11:30:15",
            timestamp_seconds=41415,
            text="Thanks for joining us today.",
            confidence=0.92,
        ),
        TranscriptEntry(
            speaker="John Smith",
            timestamp="11:30:30",
            timestamp_seconds=41430,
            text="Let's begin with the agenda. We have three topics to discuss today.",
            confidence=0.90,
        ),
    ]
