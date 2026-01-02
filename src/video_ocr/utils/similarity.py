"""String similarity and text comparison utilities."""

import re
from typing import Optional, Tuple

from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein


def normalize_text(text: str) -> str:
    """
    Normalize text for comparison.

    Args:
        text: Input text

    Returns:
        Normalized text with consistent whitespace and case
    """
    # Convert to lowercase
    text = text.lower()
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)
    # Strip leading/trailing whitespace
    text = text.strip()
    return text


def levenshtein_ratio(s1: str, s2: str, normalize: bool = True) -> float:
    """
    Calculate Levenshtein similarity ratio between two strings.

    Args:
        s1: First string
        s2: Second string
        normalize: Whether to normalize strings before comparison

    Returns:
        Similarity ratio between 0.0 and 1.0
    """
    if normalize:
        s1 = normalize_text(s1)
        s2 = normalize_text(s2)

    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0

    return fuzz.ratio(s1, s2) / 100.0


def find_overlap(text1: str, text2: str, min_overlap: int = 20) -> Tuple[int, int]:
    """
    Find the overlapping region between the end of text1 and start of text2.

    This is useful for detecting scrolling text where the bottom of one frame
    appears at the top of the next frame.

    Args:
        text1: First text (earlier in sequence)
        text2: Second text (later in sequence)
        min_overlap: Minimum overlap length to consider

    Returns:
        Tuple of (overlap_start_in_text1, overlap_length)
    """
    if len(text1) < min_overlap or len(text2) < min_overlap:
        return (-1, 0)

    # Try to find where text2 starts in the end of text1
    best_overlap = 0
    best_position = -1

    # Check different potential overlap lengths
    for overlap_len in range(min_overlap, min(len(text1), len(text2)) + 1):
        end_of_text1 = text1[-overlap_len:]
        start_of_text2 = text2[:overlap_len]

        similarity = fuzz.ratio(end_of_text1, start_of_text2) / 100.0

        if similarity > 0.85 and overlap_len > best_overlap:
            best_overlap = overlap_len
            best_position = len(text1) - overlap_len

    return (best_position, best_overlap)


def find_new_content(
    accumulated_text: str,
    new_text: str,
    threshold: float = 0.85,
) -> Optional[str]:
    """
    Find the portion of new_text that is genuinely new content.

    This handles scrolling text where part of new_text overlaps with
    accumulated_text from previous frames.

    Args:
        accumulated_text: Text accumulated from previous frames
        new_text: Text from the current frame
        threshold: Similarity threshold for considering text as duplicate

    Returns:
        The new content portion, or None if entirely duplicate
    """
    if not accumulated_text:
        return new_text

    if not new_text:
        return None

    # Check if the entire new_text is similar to accumulated
    overall_similarity = levenshtein_ratio(accumulated_text, new_text)
    if overall_similarity > threshold:
        return None

    # Try to find overlap between end of accumulated and start of new
    norm_accumulated = normalize_text(accumulated_text)
    norm_new = normalize_text(new_text)

    overlap_pos, overlap_len = find_overlap(norm_accumulated, norm_new)

    if overlap_pos >= 0 and overlap_len > 0:
        # Return the portion after the overlap
        # Map back to original text (approximate)
        words_in_overlap = len(norm_new[:overlap_len].split())
        new_words = new_text.split()

        if words_in_overlap < len(new_words):
            return " ".join(new_words[words_in_overlap:])

    # If no clear overlap found, check line by line
    new_lines = new_text.split("\n")
    accumulated_lines = accumulated_text.split("\n")

    new_content_lines = []
    for line in new_lines:
        line_stripped = line.strip()
        if not line_stripped:
            continue

        is_duplicate = False
        for acc_line in accumulated_lines[-20:]:  # Check last 20 lines
            if levenshtein_ratio(line_stripped, acc_line.strip()) > threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            new_content_lines.append(line)

    if new_content_lines:
        return "\n".join(new_content_lines)

    return None


def calculate_text_similarity_matrix(texts: list[str]) -> list[list[float]]:
    """
    Calculate pairwise similarity matrix for a list of texts.

    Args:
        texts: List of text strings

    Returns:
        2D matrix of similarity scores
    """
    n = len(texts)
    matrix = [[0.0] * n for _ in range(n)]

    for i in range(n):
        for j in range(i, n):
            if i == j:
                matrix[i][j] = 1.0
            else:
                sim = levenshtein_ratio(texts[i], texts[j])
                matrix[i][j] = sim
                matrix[j][i] = sim

    return matrix
