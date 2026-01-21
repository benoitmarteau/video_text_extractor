"""Frame extraction from video files."""

from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Optional, Tuple

import cv2
import numpy as np

from video_ocr.utils.image_utils import calculate_frame_similarity
from video_ocr.utils.logging_config import get_logger


@dataclass
class FrameInfo:
    """Information about an extracted frame."""

    frame_number: int
    timestamp_seconds: float
    frame: np.ndarray

    @property
    def timestamp_str(self) -> str:
        """Get timestamp as HH:MM:SS format."""
        hours = int(self.timestamp_seconds // 3600)
        minutes = int((self.timestamp_seconds % 3600) // 60)
        seconds = int(self.timestamp_seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class VideoOCRError(Exception):
    """Base exception for Video OCR errors."""

    pass


class FrameExtractionError(VideoOCRError):
    """Error during frame extraction."""

    pass


class FrameExtractor:
    """
    Extract frames from video files with smart sampling.

    Supports configurable FPS extraction and frame differencing
    to skip near-identical frames.
    """

    SUPPORTED_FORMATS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}

    def __init__(
        self,
        fps: float = 2.0,
        skip_similar_threshold: float = 0.95,
        max_frames: Optional[int] = None,
    ):
        """
        Initialize frame extractor.

        Args:
            fps: Target frames per second to extract
            skip_similar_threshold: Skip frames with similarity above this (0-1)
            max_frames: Maximum number of frames to extract
        """
        # Ensure fps is a float (defensive against string values from forms)
        self.fps = float(fps) if fps is not None else 2.0
        self.skip_similar_threshold = float(skip_similar_threshold)
        self.max_frames = max_frames
        self.logger = get_logger()
        self.logger.debug(f"FrameExtractor initialized with fps={self.fps} (input was {fps!r}), skip_similar_threshold={self.skip_similar_threshold}, max_frames={max_frames}")

    def validate_video(self, video_path: Path) -> Tuple[bool, str]:
        """
        Validate that a video file can be processed.

        Args:
            video_path: Path to video file

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not video_path.exists():
            return False, f"Video file not found: {video_path}"

        if video_path.suffix.lower() not in self.SUPPORTED_FORMATS:
            return False, f"Unsupported format: {video_path.suffix}"

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            cap.release()
            return False, f"Could not open video: {video_path}"

        # Check video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS)

        cap.release()

        if width < 640 or height < 480:
            return False, f"Resolution too low: {width}x{height} (minimum 640x480)"

        if frame_count == 0:
            return False, "Video has no frames"

        if video_fps <= 0:
            return False, "Could not determine video FPS"

        return True, ""

    def get_video_info(self, video_path: Path) -> dict:
        """
        Get information about a video file.

        Args:
            video_path: Path to video file

        Returns:
            Dictionary with video properties
        """
        cap = cv2.VideoCapture(str(video_path))

        info = {
            "path": str(video_path),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "duration_seconds": 0.0,
        }

        if info["fps"] > 0:
            info["duration_seconds"] = info["frame_count"] / info["fps"]

        cap.release()
        return info

    def extract_frames(
        self,
        video_path: Path,
        skip_similar: bool = True,
    ) -> Generator[FrameInfo, None, None]:
        """
        Extract frames from a video file.

        This is a generator that yields frames one at a time for memory efficiency.

        Args:
            video_path: Path to video file
            skip_similar: Skip frames that are too similar to the previous

        Yields:
            FrameInfo objects containing frame data and metadata
        """
        video_path = Path(video_path)

        is_valid, error = self.validate_video(video_path)
        if not is_valid:
            raise FrameExtractionError(error)

        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise FrameExtractionError(f"Could not open video file: {video_path}")

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if video_fps <= 0 or total_frames <= 0:
            cap.release()
            raise FrameExtractionError(
                f"Invalid video properties: FPS={video_fps}, frames={total_frames}"
            )

        # Calculate frame interval based on target FPS
        # Use round() instead of int() to handle non-standard frame rates better
        # For example: video at 29.97 FPS, target 15 FPS -> interval should be 2, not 1
        if self.fps >= video_fps:
            # Target FPS is higher than or equal to video FPS, extract every frame
            frame_interval = 1
        else:
            frame_interval = max(1, round(video_fps / self.fps))

        # Calculate expected frames
        expected_frames = total_frames // frame_interval

        self.logger.info(
            f"Extracting frames from {video_path.name}: "
            f"native_fps={video_fps:.2f}, target_fps={self.fps}, interval={frame_interval}, "
            f"total_frames={total_frames}, expected_extract=~{expected_frames}"
        )

        frame_number = 0
        extracted_count = 0
        previous_frame: Optional[np.ndarray] = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Check if we should extract this frame based on interval
            if frame_number % frame_interval != 0:
                frame_number += 1
                continue

            # Check similarity with previous frame if enabled
            if skip_similar and previous_frame is not None:
                similarity = calculate_frame_similarity(previous_frame, frame)
                if similarity > self.skip_similar_threshold:
                    self.logger.debug(
                        f"Skipping frame {frame_number} (similarity: {similarity:.3f})"
                    )
                    frame_number += 1
                    continue

            # Calculate timestamp
            timestamp = frame_number / video_fps

            yield FrameInfo(
                frame_number=frame_number,
                timestamp_seconds=timestamp,
                frame=frame,
            )

            previous_frame = frame.copy()
            extracted_count += 1

            # Check max frames limit
            if self.max_frames and extracted_count >= self.max_frames:
                self.logger.info(f"Reached max frames limit: {self.max_frames}")
                break

            frame_number += 1

        cap.release()

        if extracted_count == 0:
            self.logger.warning(
                f"WARNING: No frames extracted from {video_path.name}! "
                f"Total frames in video: {total_frames}, interval: {frame_interval}"
            )
        else:
            self.logger.info(
                f"Extracted {extracted_count} frames from {total_frames} total "
                f"(expected ~{expected_frames})"
            )

    def extract_frames_to_list(
        self,
        video_path: Path,
        skip_similar: bool = True,
    ) -> list[FrameInfo]:
        """
        Extract all frames to a list.

        Note: This loads all frames into memory. Use extract_frames() generator
        for large videos.

        Args:
            video_path: Path to video file
            skip_similar: Skip frames that are too similar

        Returns:
            List of FrameInfo objects
        """
        return list(self.extract_frames(video_path, skip_similar))

    def extract_single_frame(
        self,
        video_path: Path,
        frame_number: int,
    ) -> Optional[FrameInfo]:
        """
        Extract a single frame from a video.

        Args:
            video_path: Path to video file
            frame_number: Frame number to extract

        Returns:
            FrameInfo if successful, None otherwise
        """
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            return None

        video_fps = cap.get(cv2.CAP_PROP_FPS)

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()

        cap.release()

        if not ret:
            return None

        return FrameInfo(
            frame_number=frame_number,
            timestamp_seconds=frame_number / video_fps,
            frame=frame,
        )
