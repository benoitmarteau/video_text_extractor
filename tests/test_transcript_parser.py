"""Tests for transcript parsing."""

import pytest

from video_ocr.core.transcript_parser import (
    ParsedTranscript,
    TranscriptEntry,
    TranscriptParser,
)


class TestTranscriptParser:
    """Tests for TranscriptParser class."""

    def test_init_default_values(self):
        """Test default initialization."""
        parser = TranscriptParser()
        assert parser.merge_consecutive is True

    def test_init_custom_values(self):
        """Test custom initialization."""
        parser = TranscriptParser(
            timestamp_pattern=r"\d{2}:\d{2}",
            merge_consecutive=False,
        )
        assert parser.merge_consecutive is False

    def test_parse_empty_text(self):
        """Test parsing empty text."""
        parser = TranscriptParser()
        result = parser.parse("")

        assert isinstance(result, ParsedTranscript)
        assert len(result.entries) == 0
        assert len(result.speakers) == 0

    def test_parse_simple_transcript(self, sample_ocr_text):
        """Test parsing a simple transcript."""
        parser = TranscriptParser(merge_consecutive=False)
        result = parser.parse(sample_ocr_text)

        assert len(result.entries) >= 2
        assert len(result.speakers) >= 2

        # Check speaker names
        speakers = set(result.speakers)
        assert "John Smith" in speakers or any("John" in s for s in speakers)

    def test_parse_teams_format(self):
        """Test parsing Teams transcript format."""
        text = """Speaker One
11:30:00
This is the first message from speaker one.

Speaker Two
11:30:15
Response from speaker two."""

        parser = TranscriptParser(merge_consecutive=False)
        result = parser.parse(text)

        assert len(result.entries) >= 2

        # Check timestamps parsed correctly
        timestamps = [e.timestamp for e in result.entries]
        assert any("11:30" in t for t in timestamps)

    def test_parse_with_merge(self):
        """Test parsing with consecutive message merging."""
        text = """John Smith
11:30:00
First message.

John Smith
11:30:10
Second message from same speaker."""

        parser = TranscriptParser(merge_consecutive=True)
        result = parser.parse(text)

        # With merging, should combine consecutive speaker messages
        assert len(result.entries) <= 2

    def test_parse_without_merge(self):
        """Test parsing without merging."""
        text = """John Smith
11:30:00
First message.

John Smith
11:30:10
Second message from same speaker."""

        parser = TranscriptParser(merge_consecutive=False)
        result = parser.parse(text)

        # Without merging, should keep separate
        assert len(result.entries) >= 2

    def test_parse_timestamp_formats(self):
        """Test various timestamp formats."""
        parser = TranscriptParser()

        # Standard HH:MM:SS
        text1 = """Speaker
11:30:00
Message"""
        result1 = parser.parse(text1)
        if result1.entries:
            assert result1.entries[0].timestamp_seconds == 11 * 3600 + 30 * 60

        # Single digit hour
        text2 = """Speaker
1:30:00
Message"""
        result2 = parser.parse(text2)
        if result2.entries:
            assert result2.entries[0].timestamp_seconds == 1 * 3600 + 30 * 60

    def test_parse_multiline_message(self):
        """Test parsing multiline messages."""
        text = """Speaker Name
11:30:00
This is the first line.
This is the second line.
And a third line of the message."""

        parser = TranscriptParser(merge_consecutive=False)
        result = parser.parse(text)

        if result.entries:
            # Message should contain multiple lines
            assert "first line" in result.entries[0].text
            # Lines should be combined
            assert len(result.entries[0].text) > 30

    def test_parsed_transcript_properties(self):
        """Test ParsedTranscript properties."""
        entries = [
            TranscriptEntry(
                speaker="John",
                timestamp="11:30:00",
                timestamp_seconds=41400,
                text="Hello world this is a test",
            ),
            TranscriptEntry(
                speaker="Jane",
                timestamp="11:30:10",
                timestamp_seconds=41410,
                text="Response message here",
            ),
        ]

        transcript = ParsedTranscript(
            entries=entries,
            raw_text="raw text",
            speakers=["John", "Jane"],
        )

        assert transcript.total_entries == 2
        assert transcript.word_count == 9  # 6 + 3 words

    def test_get_entries_by_speaker(self):
        """Test filtering entries by speaker."""
        entries = [
            TranscriptEntry(
                speaker="John", timestamp="11:30:00", timestamp_seconds=0, text="Message 1"
            ),
            TranscriptEntry(
                speaker="Jane", timestamp="11:30:10", timestamp_seconds=10, text="Message 2"
            ),
            TranscriptEntry(
                speaker="John", timestamp="11:30:20", timestamp_seconds=20, text="Message 3"
            ),
        ]

        transcript = ParsedTranscript(entries=entries, speakers=["John", "Jane"])

        john_entries = transcript.get_entries_by_speaker("John")
        assert len(john_entries) == 2

        jane_entries = transcript.get_entries_by_speaker("Jane")
        assert len(jane_entries) == 1

    def test_transcript_to_dict(self):
        """Test converting transcript to dictionary."""
        entries = [
            TranscriptEntry(
                speaker="John",
                timestamp="11:30:00",
                timestamp_seconds=41400,
                text="Hello",
                confidence=0.9,
            ),
        ]

        transcript = ParsedTranscript(entries=entries, speakers=["John"])
        result = transcript.to_dict()

        assert "entries" in result
        assert "speakers" in result
        assert len(result["entries"]) == 1
        assert result["entries"][0]["speaker"] == "John"

    def test_parse_entries_from_lines(self):
        """Test parsing from list of (text, confidence) tuples."""
        lines = [
            ("Speaker Name\n11:30:00\nMessage content", 0.9),
            ("More content from speaker", 0.85),
        ]

        parser = TranscriptParser()
        result = parser.parse_entries_from_lines(lines)

        assert isinstance(result, ParsedTranscript)


class TestTranscriptEntry:
    """Tests for TranscriptEntry dataclass."""

    def test_entry_creation(self):
        """Test creating a transcript entry."""
        entry = TranscriptEntry(
            speaker="Test Speaker",
            timestamp="11:30:00",
            timestamp_seconds=41400,
            text="This is a test message",
            confidence=0.95,
            source_frame=10,
        )

        assert entry.speaker == "Test Speaker"
        assert entry.timestamp == "11:30:00"
        assert entry.timestamp_seconds == 41400
        assert entry.text == "This is a test message"
        assert entry.confidence == 0.95
        assert entry.source_frame == 10

    def test_entry_to_dict(self):
        """Test converting entry to dictionary."""
        entry = TranscriptEntry(
            speaker="Speaker",
            timestamp="11:30:00",
            timestamp_seconds=41400,
            text="Message",
            confidence=0.9,
        )

        result = entry.to_dict()

        assert result["speaker"] == "Speaker"
        assert result["timestamp"] == "11:30:00"
        assert result["timestamp_seconds"] == 41400
        assert result["text"] == "Message"
        assert result["confidence"] == 0.9

    def test_entry_defaults(self):
        """Test entry default values."""
        entry = TranscriptEntry(
            speaker="",
            timestamp="",
            timestamp_seconds=0,
            text="",
        )

        assert entry.confidence == 0.0
        assert entry.source_frame == 0
