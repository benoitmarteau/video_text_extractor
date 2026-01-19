"""Parallel OCR processing for improved CPU performance."""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable, Iterator, List, Optional, Tuple

import numpy as np

from video_ocr.engines.base import OCRResult
from video_ocr.utils.logging_config import get_logger


@dataclass
class FrameTask:
    """A frame to be processed by OCR."""

    frame: np.ndarray
    frame_number: int
    timestamp_seconds: float


@dataclass
class ParallelOCRConfig:
    """Configuration for parallel OCR processing."""

    num_workers: int = 0  # 0 = auto-detect based on CPU cores
    engine: str = "easyocr"
    languages: List[str] = None
    confidence_threshold: float = 0.5
    gpu: bool = False  # GPU mode doesn't benefit from threading

    def __post_init__(self):
        if self.languages is None:
            self.languages = ["en"]
        if self.num_workers == 0:
            # Use number of CPU cores, but cap at 4 to avoid memory issues
            self.num_workers = min(os.cpu_count() or 2, 4)


class ParallelOCRProcessor:
    """
    Process multiple frames in parallel using thread pool.

    Each worker thread has its own OCR engine instance to avoid
    serialization issues and enable true parallel processing.

    Note: This is most beneficial for CPU-only processing. When using GPU,
    the GPU is already efficient and threading may not help much.
    """

    def __init__(self, config: ParallelOCRConfig = None):
        """
        Initialize parallel OCR processor.

        Args:
            config: Parallel processing configuration
        """
        self.config = config or ParallelOCRConfig()
        self.logger = get_logger()
        self._engines = {}  # Thread ID -> OCR engine

    def _get_or_create_engine(self):
        """Get or create an OCR engine for the current thread."""
        import threading

        thread_id = threading.current_thread().ident

        if thread_id not in self._engines:
            # Create a new engine for this thread
            from video_ocr.core.ocr_engine import OCREngine

            engine = OCREngine(
                engine=self.config.engine,
                languages=self.config.languages,
                confidence_threshold=self.config.confidence_threshold,
                gpu=self.config.gpu,
            )
            self._engines[thread_id] = engine
            self.logger.debug(f"Created OCR engine for thread {thread_id}")

        return self._engines[thread_id]

    def _process_single_frame(self, task: FrameTask) -> Tuple[int, OCRResult]:
        """Process a single frame (called from worker thread)."""
        engine = self._get_or_create_engine()

        result = engine.process_frame(
            task.frame,
            frame_number=task.frame_number,
            timestamp_seconds=task.timestamp_seconds,
        )

        return task.frame_number, result

    def process_frames(
        self,
        frames: List[FrameTask],
        progress_callback: Optional[Callable[[int, int, int], None]] = None,
    ) -> List[OCRResult]:
        """
        Process multiple frames in parallel.

        Args:
            frames: List of FrameTask objects to process
            progress_callback: Optional callback(current, total, texts_found)

        Returns:
            List of OCRResult objects in frame order
        """
        if not frames:
            return []

        total_frames = len(frames)
        num_workers = min(self.config.num_workers, total_frames)

        self.logger.info(
            f"Starting parallel OCR with {num_workers} workers on {total_frames} frames"
        )

        # Results dict to maintain order
        results_dict = {}
        texts_found = 0
        completed = 0

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit all tasks
            future_to_frame = {
                executor.submit(self._process_single_frame, task): task.frame_number
                for task in frames
            }

            # Collect results as they complete
            for future in as_completed(future_to_frame):
                frame_num, result = future.result()
                results_dict[frame_num] = result

                if result.has_text:
                    texts_found += 1

                completed += 1

                if progress_callback:
                    progress_callback(completed, total_frames, texts_found)

        # Sort results by frame number
        sorted_results = [
            results_dict[frame_num]
            for frame_num in sorted(results_dict.keys())
        ]

        self.logger.info(
            f"Parallel OCR complete: {texts_found}/{total_frames} frames had text"
        )

        return sorted_results

    def process_frames_iterator(
        self,
        frame_iterator: Iterator[FrameTask],
        batch_size: int = 20,
        progress_callback: Optional[Callable[[int, int, int], None]] = None,
    ) -> Iterator[OCRResult]:
        """
        Process frames from an iterator in batches.

        This is memory-efficient for large videos as it doesn't load
        all frames at once.

        Args:
            frame_iterator: Iterator yielding FrameTask objects
            batch_size: Number of frames to process per batch
            progress_callback: Optional callback(current, total, texts_found)

        Yields:
            OCRResult objects in frame order
        """
        batch = []
        total_processed = 0
        total_texts = 0

        for task in frame_iterator:
            batch.append(task)

            if len(batch) >= batch_size:
                results = self.process_frames(batch)

                for result in results:
                    total_processed += 1
                    if result.has_text:
                        total_texts += 1

                    if progress_callback:
                        progress_callback(total_processed, -1, total_texts)

                    yield result

                batch = []

        # Process remaining frames
        if batch:
            results = self.process_frames(batch)

            for result in results:
                total_processed += 1
                if result.has_text:
                    total_texts += 1

                if progress_callback:
                    progress_callback(total_processed, -1, total_texts)

                yield result

    def cleanup(self):
        """Clean up resources (clear engine instances)."""
        self._engines.clear()


def create_parallel_processor(
    engine: str = "easyocr",
    languages: List[str] = None,
    confidence_threshold: float = 0.5,
    gpu: bool = False,
    num_workers: int = 0,
) -> ParallelOCRProcessor:
    """
    Create a parallel OCR processor.

    Args:
        engine: OCR engine name
        languages: Language codes
        confidence_threshold: Minimum confidence
        gpu: Use GPU (disables parallel benefit)
        num_workers: Number of worker threads (0 = auto)

    Returns:
        ParallelOCRProcessor instance
    """
    config = ParallelOCRConfig(
        num_workers=num_workers,
        engine=engine,
        languages=languages or ["en"],
        confidence_threshold=confidence_threshold,
        gpu=gpu,
    )

    return ParallelOCRProcessor(config)
