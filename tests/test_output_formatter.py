"""Tests for output formatting."""

import json
from pathlib import Path

import pytest

from video_ocr.core.output_formatter import (
    ExtractionMetadata,
    ExtractionStatistics,
    OutputFormatter,
    create_metadata,
)
from video_ocr.core.transcript_parser import ParsedTranscript, TranscriptEntry


@pytest.fixture
def sample_metadata():
    """Create sample extraction metadata."""
    return ExtractionMetadata(
        source_file="test_video.mp4",
        duration_seconds=120.5,
        frames_processed=241,
        frames_with_text=180,
        extraction_timestamp="2025-01-15T10:30:00Z",
        ocr_engine="easyocr",
        confidence_threshold=0.5,
    )


@pytest.fixture
def sample_parsed_transcript():
    """Create sample parsed transcript."""
    entries = [
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
            text="Let's begin with the agenda.",
            confidence=0.90,
        ),
    ]

    return ParsedTranscript(
        entries=entries,
        raw_text="Hello everyone...",
        speakers=["John Smith", "Jane Doe"],
    )


class TestOutputFormatter:
    """Tests for OutputFormatter class."""

    def test_init_default_values(self):
        """Test default initialization."""
        formatter = OutputFormatter()
        assert formatter.include_metadata is True
        assert formatter.include_statistics is True
        assert formatter.include_raw_text is True
        assert formatter.pretty_print is True

    def test_init_custom_values(self):
        """Test custom initialization."""
        formatter = OutputFormatter(
            include_metadata=False,
            include_statistics=False,
            include_raw_text=False,
            pretty_print=False,
        )
        assert formatter.include_metadata is False
        assert formatter.include_statistics is False
        assert formatter.include_raw_text is False
        assert formatter.pretty_print is False

    def test_format_json(self, sample_parsed_transcript, sample_metadata):
        """Test JSON output formatting."""
        formatter = OutputFormatter()
        result = formatter.format_json(
            sample_parsed_transcript,
            sample_metadata,
            "raw text here",
        )

        # Should be valid JSON
        data = json.loads(result)

        assert "metadata" in data
        assert "transcript" in data
        assert "statistics" in data
        assert "raw_text" in data

        assert len(data["transcript"]) == 3
        assert data["metadata"]["source_file"] == "test_video.mp4"

    def test_format_json_without_metadata(self, sample_parsed_transcript, sample_metadata):
        """Test JSON output without metadata."""
        formatter = OutputFormatter(include_metadata=False)
        result = formatter.format_json(
            sample_parsed_transcript,
            sample_metadata,
        )

        data = json.loads(result)
        assert "metadata" not in data
        assert "transcript" in data

    def test_format_json_without_statistics(self, sample_parsed_transcript, sample_metadata):
        """Test JSON output without statistics."""
        formatter = OutputFormatter(include_statistics=False)
        result = formatter.format_json(
            sample_parsed_transcript,
            sample_metadata,
        )

        data = json.loads(result)
        assert "statistics" not in data

    def test_format_json_compact(self, sample_parsed_transcript, sample_metadata):
        """Test compact JSON output."""
        formatter = OutputFormatter(pretty_print=False)
        result = formatter.format_json(
            sample_parsed_transcript,
            sample_metadata,
        )

        # Compact JSON should not have newlines between elements
        assert result.count("\n") < 5

    def test_format_markdown(self, sample_parsed_transcript, sample_metadata):
        """Test Markdown output formatting."""
        formatter = OutputFormatter()
        result = formatter.format_markdown(sample_parsed_transcript, sample_metadata)

        # Check structure
        assert "# Meeting Transcript" in result
        assert "test_video.mp4" in result
        assert "John Smith" in result
        assert "Jane Doe" in result
        assert "11:30:00" in result
        assert "Hello everyone" in result

    def test_format_markdown_without_metadata(self, sample_parsed_transcript, sample_metadata):
        """Test Markdown output without metadata."""
        formatter = OutputFormatter(include_metadata=False)
        result = formatter.format_markdown(sample_parsed_transcript, sample_metadata)

        # Should still have transcript
        assert "# Meeting Transcript" in result
        assert "John Smith" in result
        # But no source info
        assert "**Source**:" not in result

    def test_format_text(self, sample_parsed_transcript, sample_metadata):
        """Test plain text output formatting."""
        formatter = OutputFormatter()
        result = formatter.format_text(sample_parsed_transcript, sample_metadata)

        assert "John Smith" in result
        assert "Jane Doe" in result
        assert "11:30:00" in result
        assert "Hello everyone" in result

    def test_format_text_without_metadata(self, sample_parsed_transcript):
        """Test plain text output without metadata."""
        formatter = OutputFormatter(include_metadata=False)
        result = formatter.format_text(sample_parsed_transcript)

        assert "John Smith" in result
        assert "Source:" not in result

    def test_format_srt(self, sample_parsed_transcript):
        """Test SRT subtitle output formatting."""
        formatter = OutputFormatter()
        result = formatter.format_srt(sample_parsed_transcript)

        # Check SRT format
        lines = result.split("\n")

        # First entry should be numbered
        assert "1" in lines[0]

        # Should have timing line
        assert "-->" in result

        # Should have speaker and text
        assert "John Smith" in result

    def test_save_all_formats(self, sample_parsed_transcript, sample_metadata, temp_dir):
        """Test saving all output formats."""
        formatter = OutputFormatter()
        saved = formatter.save_all_formats(
            sample_parsed_transcript,
            sample_metadata,
            "raw text",
            temp_dir,
            "test_transcript",
        )

        assert "json" in saved
        assert "markdown" in saved
        assert "text" in saved
        assert "srt" in saved

        # Check files exist
        assert saved["json"].exists()
        assert saved["markdown"].exists()
        assert saved["text"].exists()
        assert saved["srt"].exists()

        # Check content
        json_content = saved["json"].read_text(encoding="utf-8")
        assert "John Smith" in json_content

        md_content = saved["markdown"].read_text(encoding="utf-8")
        assert "# Meeting Transcript" in md_content


class TestExtractionMetadata:
    """Tests for ExtractionMetadata dataclass."""

    def test_metadata_to_dict(self, sample_metadata):
        """Test converting metadata to dictionary."""
        result = sample_metadata.to_dict()

        assert result["source_file"] == "test_video.mp4"
        assert result["duration_seconds"] == 120.5
        assert result["frames_processed"] == 241
        assert result["ocr_engine"] == "easyocr"


class TestExtractionStatistics:
    """Tests for ExtractionStatistics dataclass."""

    def test_statistics_to_dict(self):
        """Test converting statistics to dictionary."""
        stats = ExtractionStatistics(
            total_speakers=5,
            total_messages=100,
            word_count=5000,
            unique_words=800,
            character_count=25000,
            average_confidence=0.923,
        )

        result = stats.to_dict()

        assert result["total_speakers"] == 5
        assert result["total_messages"] == 100
        assert result["word_count"] == 5000
        assert result["average_confidence"] == 0.923


class TestCreateMetadata:
    """Tests for create_metadata helper function."""

    def test_create_metadata(self):
        """Test creating metadata."""
        metadata = create_metadata(
            source_file="video.mp4",
            duration_seconds=60.0,
            frames_processed=120,
            frames_with_text=100,
            ocr_engine="easyocr",
            confidence_threshold=0.5,
        )

        assert isinstance(metadata, ExtractionMetadata)
        assert metadata.source_file == "video.mp4"
        assert metadata.duration_seconds == 60.0
        assert "T" in metadata.extraction_timestamp  # ISO format
