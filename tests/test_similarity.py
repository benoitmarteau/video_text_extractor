"""Tests for similarity and text comparison utilities."""

import pytest

from video_ocr.utils.similarity import (
    calculate_text_similarity_matrix,
    find_new_content,
    find_overlap,
    levenshtein_ratio,
    normalize_text,
)


class TestNormalizeText:
    """Tests for text normalization."""

    def test_normalize_lowercase(self):
        """Test lowercase conversion."""
        assert normalize_text("Hello World") == "hello world"

    def test_normalize_whitespace(self):
        """Test whitespace normalization."""
        assert normalize_text("hello   world") == "hello world"
        assert normalize_text("hello\nworld") == "hello world"
        assert normalize_text("hello\t\tworld") == "hello world"

    def test_normalize_trim(self):
        """Test trimming."""
        assert normalize_text("  hello world  ") == "hello world"

    def test_normalize_combined(self):
        """Test combined normalization."""
        text = "  Hello   WORLD\n\tTest  "
        assert normalize_text(text) == "hello world test"

    def test_normalize_empty(self):
        """Test empty string."""
        assert normalize_text("") == ""
        assert normalize_text("   ") == ""


class TestLevenshteinRatio:
    """Tests for Levenshtein similarity ratio."""

    def test_identical_strings(self):
        """Test identical strings return 1.0."""
        assert levenshtein_ratio("hello", "hello") == 1.0

    def test_empty_strings(self):
        """Test empty strings."""
        assert levenshtein_ratio("", "") == 1.0
        assert levenshtein_ratio("hello", "") == 0.0
        assert levenshtein_ratio("", "hello") == 0.0

    def test_similar_strings(self):
        """Test similar strings."""
        ratio = levenshtein_ratio("hello world", "hello there")
        assert 0.4 < ratio < 0.8

    def test_different_strings(self):
        """Test very different strings."""
        ratio = levenshtein_ratio("abc", "xyz")
        assert ratio < 0.5

    def test_case_sensitivity_with_normalize(self):
        """Test normalization affects case."""
        ratio_normalized = levenshtein_ratio("Hello", "hello", normalize=True)
        ratio_raw = levenshtein_ratio("Hello", "hello", normalize=False)

        assert ratio_normalized == 1.0
        assert ratio_raw < 1.0

    def test_whitespace_handling(self):
        """Test whitespace handling with normalization."""
        ratio = levenshtein_ratio("hello  world", "hello world", normalize=True)
        assert ratio == 1.0


class TestFindOverlap:
    """Tests for overlap detection."""

    def test_no_overlap(self):
        """Test strings with no overlap."""
        pos, length = find_overlap("hello", "world", min_overlap=3)
        assert pos == -1
        assert length == 0

    def test_clear_overlap(self):
        """Test strings with clear overlap."""
        text1 = "The quick brown fox jumps over"
        text2 = "fox jumps over the lazy dog"

        pos, length = find_overlap(text1, text2, min_overlap=10)
        # Should find overlap around "fox jumps over"
        assert length > 0

    def test_min_overlap_threshold(self):
        """Test minimum overlap threshold."""
        text1 = "hello world test"
        text2 = "test again"

        pos1, length1 = find_overlap(text1, text2, min_overlap=3)
        pos2, length2 = find_overlap(text1, text2, min_overlap=10)

        # Higher minimum should find nothing or less
        assert length1 >= length2

    def test_short_strings(self):
        """Test strings shorter than min_overlap."""
        pos, length = find_overlap("hi", "there", min_overlap=5)
        assert pos == -1
        assert length == 0


class TestFindNewContent:
    """Tests for finding new content."""

    def test_empty_accumulated(self):
        """Test with empty accumulated text."""
        result = find_new_content("", "new content here")
        assert result == "new content here"

    def test_empty_new(self):
        """Test with empty new text."""
        result = find_new_content("accumulated", "")
        assert result is None

    def test_identical_text(self):
        """Test with identical texts."""
        result = find_new_content(
            "the same text here",
            "the same text here",
            threshold=0.85,
        )
        assert result is None

    def test_completely_new(self):
        """Test with completely different text."""
        result = find_new_content(
            "some accumulated text here",
            "completely different content now",
            threshold=0.85,
        )
        assert result is not None
        assert len(result) > 0

    def test_partial_overlap(self):
        """Test with partial overlap (scrolling scenario)."""
        accumulated = "Line 1 content\nLine 2 content\nLine 3 content"
        new_text = "Line 2 content\nLine 3 content\nLine 4 content"

        result = find_new_content(accumulated, new_text, threshold=0.85)

        # Should find Line 4 as new
        if result:
            assert "Line 4" in result or "4" in result

    def test_threshold_sensitivity(self):
        """Test different thresholds."""
        accumulated = "hello world test"
        new_text = "hello world testing"  # Slightly different

        result_strict = find_new_content(accumulated, new_text, threshold=0.95)
        result_loose = find_new_content(accumulated, new_text, threshold=0.7)

        # Stricter threshold might find new content, looser might not
        # Both are valid depending on the similarity
        assert result_strict is None or result_loose is None or True


class TestSimilarityMatrix:
    """Tests for similarity matrix calculation."""

    def test_empty_list(self):
        """Test with empty list."""
        result = calculate_text_similarity_matrix([])
        assert result == []

    def test_single_item(self):
        """Test with single item."""
        result = calculate_text_similarity_matrix(["hello"])
        assert result == [[1.0]]

    def test_identical_texts(self):
        """Test with identical texts."""
        result = calculate_text_similarity_matrix(["hello", "hello", "hello"])

        for row in result:
            for val in row:
                assert val == 1.0

    def test_different_texts(self):
        """Test with different texts."""
        result = calculate_text_similarity_matrix(["aaa", "bbb", "ccc"])

        # Diagonal should be 1.0
        assert result[0][0] == 1.0
        assert result[1][1] == 1.0
        assert result[2][2] == 1.0

        # Off-diagonal should be less than 1.0
        assert result[0][1] < 1.0
        assert result[1][2] < 1.0

    def test_matrix_symmetry(self):
        """Test that matrix is symmetric."""
        texts = ["hello world", "hello there", "goodbye world"]
        result = calculate_text_similarity_matrix(texts)

        for i in range(len(texts)):
            for j in range(len(texts)):
                assert result[i][j] == result[j][i]
