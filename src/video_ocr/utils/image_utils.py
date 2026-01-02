"""Image processing utilities for Video OCR."""

from typing import Optional, Tuple

import cv2
import numpy as np


def calculate_frame_similarity(
    frame1: np.ndarray,
    frame2: np.ndarray,
    method: str = "histogram",
) -> float:
    """
    Calculate similarity between two frames.

    Args:
        frame1: First frame (BGR numpy array)
        frame2: Second frame (BGR numpy array)
        method: Comparison method ('histogram', 'mse', 'ssim')

    Returns:
        Similarity score between 0.0 and 1.0
    """
    if frame1 is None or frame2 is None:
        return 0.0

    if frame1.shape != frame2.shape:
        return 0.0

    if method == "histogram":
        return _histogram_similarity(frame1, frame2)
    elif method == "mse":
        return _mse_similarity(frame1, frame2)
    else:
        # Default to histogram
        return _histogram_similarity(frame1, frame2)


def _histogram_similarity(frame1: np.ndarray, frame2: np.ndarray) -> float:
    """Calculate histogram-based similarity."""
    # Convert to grayscale
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Calculate histograms
    hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])

    # Normalize histograms
    cv2.normalize(hist1, hist1)
    cv2.normalize(hist2, hist2)

    # Compare histograms
    score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

    # Convert to 0-1 range (correlation can be -1 to 1)
    return max(0.0, (score + 1) / 2)


def _mse_similarity(frame1: np.ndarray, frame2: np.ndarray) -> float:
    """Calculate MSE-based similarity."""
    # Convert to grayscale
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Calculate MSE
    mse = np.mean((gray1.astype(float) - gray2.astype(float)) ** 2)

    # Convert to similarity (higher MSE = lower similarity)
    # Normalize assuming max MSE is 255^2
    max_mse = 255.0 ** 2
    similarity = 1.0 - (mse / max_mse)

    return max(0.0, similarity)


def preprocess_frame(
    frame: np.ndarray,
    target_size: Optional[Tuple[int, int]] = None,
    enhance_contrast: bool = True,
    denoise: bool = False,
) -> np.ndarray:
    """
    Preprocess a frame for better OCR results.

    Args:
        frame: Input frame (BGR numpy array)
        target_size: Optional target size (width, height)
        enhance_contrast: Apply contrast enhancement
        denoise: Apply denoising (slower but may help)

    Returns:
        Preprocessed frame
    """
    result = frame.copy()

    # Resize if needed
    if target_size:
        result = cv2.resize(result, target_size, interpolation=cv2.INTER_LANCZOS4)

    # Convert to grayscale for processing
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

    # Enhance contrast using CLAHE
    if enhance_contrast:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

    # Denoise if requested
    if denoise:
        gray = cv2.fastNlMeansDenoising(gray, h=10)

    # Convert back to BGR
    result = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    return result


def crop_frame(
    frame: np.ndarray,
    region: Optional[Tuple[int, int, int, int]] = None,
) -> np.ndarray:
    """
    Crop a region from a frame.

    Args:
        frame: Input frame
        region: (x, y, width, height) region to crop, or None for full frame

    Returns:
        Cropped frame
    """
    if region is None:
        return frame

    x, y, w, h = region
    return frame[y : y + h, x : x + w]


def detect_text_regions(
    frame: np.ndarray,
    min_area: int = 100,
) -> list[Tuple[int, int, int, int]]:
    """
    Detect potential text regions in a frame using MSER or contours.

    Args:
        frame: Input frame
        min_area: Minimum region area to consider

    Returns:
        List of (x, y, width, height) bounding boxes
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply threshold
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    regions = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_area:
            x, y, w, h = cv2.boundingRect(contour)
            regions.append((x, y, w, h))

    return regions


def frame_to_rgb(frame: np.ndarray) -> np.ndarray:
    """
    Convert a BGR frame to RGB.

    Args:
        frame: BGR numpy array

    Returns:
        RGB numpy array
    """
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def save_frame(frame: np.ndarray, path: str) -> bool:
    """
    Save a frame to disk.

    Args:
        frame: Frame to save
        path: Output path

    Returns:
        True if successful
    """
    try:
        cv2.imwrite(path, frame)
        return True
    except Exception:
        return False
