"""Transcript parsing for structured content extraction."""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from video_ocr.utils.logging_config import get_logger


@dataclass
class TranscriptEntry:
    """A single entry in a transcript."""

    speaker: str
    timestamp: str
    timestamp_seconds: int
    text: str
    confidence: float = 0.0
    source_frame: int = 0

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "speaker": self.speaker,
            "timestamp": self.timestamp,
            "timestamp_seconds": self.timestamp_seconds,
            "text": self.text,
            "confidence": self.confidence,
        }


@dataclass
class ParsedTranscript:
    """Complete parsed transcript."""

    entries: List[TranscriptEntry] = field(default_factory=list)
    raw_text: str = ""
    speakers: List[str] = field(default_factory=list)

    @property
    def total_entries(self) -> int:
        return len(self.entries)

    @property
    def word_count(self) -> int:
        return sum(len(entry.text.split()) for entry in self.entries)

    def get_entries_by_speaker(self, speaker: str) -> List[TranscriptEntry]:
        """Get all entries from a specific speaker."""
        return [e for e in self.entries if e.speaker == speaker]

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "entries": [e.to_dict() for e in self.entries],
            "speakers": self.speakers,
            "total_entries": self.total_entries,
            "word_count": self.word_count,
        }


class TranscriptParser:
    """
    Parse structured transcript content from OCR text.

    Designed primarily for Microsoft Teams transcript format:
    - Speaker name
    - Timestamp (HH:MM:SS)
    - Message text
    """

    # Patterns for Teams transcript format
    DEFAULT_TIMESTAMP_PATTERN = r"(\d{1,2}:\d{2}:\d{2})"
    DEFAULT_SPEAKER_PATTERN = r"^([A-Z][A-Za-z\s\.\-@&]+?)(?:\s*\n|\s+\d)"

    def __init__(
        self,
        timestamp_pattern: str = None,
        merge_consecutive: bool = True,
    ):
        """
        Initialize transcript parser.

        Args:
            timestamp_pattern: Regex pattern for timestamps
            merge_consecutive: Merge consecutive messages from same speaker
        """
        self.timestamp_pattern = timestamp_pattern or self.DEFAULT_TIMESTAMP_PATTERN
        self.merge_consecutive = merge_consecutive
        self.logger = get_logger()

        # Compile patterns
        self._timestamp_re = re.compile(self.timestamp_pattern)

    def parse(
        self,
        text: str,
        default_confidence: float = 0.0,
    ) -> ParsedTranscript:
        """
        Parse transcript from raw text.

        Args:
            text: Raw OCR text
            default_confidence: Default confidence score for entries

        Returns:
            ParsedTranscript with structured entries
        """
        result = ParsedTranscript(raw_text=text)

        if not text.strip():
            return result

        # Split into potential entries
        entries = self._extract_entries(text, default_confidence)

        # Merge consecutive entries from same speaker if enabled
        if self.merge_consecutive and entries:
            entries = self._merge_consecutive(entries)

        result.entries = entries

        # Extract unique speakers
        result.speakers = list(dict.fromkeys(e.speaker for e in entries if e.speaker))

        self.logger.info(
            f"Parsed {len(entries)} transcript entries from {len(result.speakers)} speakers"
        )

        return result

    def _extract_entries(
        self,
        text: str,
        default_confidence: float,
    ) -> List[TranscriptEntry]:
        """Extract transcript entries from text."""
        entries = []

        # Split text into lines
        lines = text.split("\n")

        current_speaker = ""
        current_timestamp = ""
        current_timestamp_seconds = 0
        current_text_lines: List[str] = []

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            if not line:
                i += 1
                continue

            # Try to detect timestamp
            timestamp_match = self._timestamp_re.search(line)

            if timestamp_match:
                # Save previous entry if exists
                if current_text_lines:
                    entries.append(
                        TranscriptEntry(
                            speaker=current_speaker,
                            timestamp=current_timestamp,
                            timestamp_seconds=current_timestamp_seconds,
                            text=" ".join(current_text_lines).strip(),
                            confidence=default_confidence,
                        )
                    )
                    current_text_lines = []

                # Check if there's a speaker name before or on the same line
                timestamp_str = timestamp_match.group(1)
                before_timestamp = line[: timestamp_match.start()].strip()

                # Look at previous line for speaker
                if not before_timestamp and i > 0:
                    prev_line = lines[i - 1].strip()
                    if prev_line and not self._timestamp_re.search(prev_line):
                        # Check if it looks like a speaker name
                        if self._looks_like_speaker(prev_line):
                            current_speaker = self._clean_speaker_name(prev_line)
                elif before_timestamp:
                    if self._looks_like_speaker(before_timestamp):
                        current_speaker = self._clean_speaker_name(before_timestamp)

                current_timestamp = timestamp_str
                current_timestamp_seconds = self._parse_timestamp(timestamp_str)

                # Check for text after timestamp on same line
                after_timestamp = line[timestamp_match.end() :].strip()
                if after_timestamp:
                    current_text_lines.append(after_timestamp)

            elif current_timestamp:
                # We have a timestamp, this is message text
                # Check if this line looks like a new speaker (without timestamp)
                if self._looks_like_speaker(line) and len(line) < 50:
                    # Could be a new speaker, save current entry
                    if current_text_lines:
                        entries.append(
                            TranscriptEntry(
                                speaker=current_speaker,
                                timestamp=current_timestamp,
                                timestamp_seconds=current_timestamp_seconds,
                                text=" ".join(current_text_lines).strip(),
                                confidence=default_confidence,
                            )
                        )
                        current_text_lines = []
                    current_speaker = self._clean_speaker_name(line)
                else:
                    current_text_lines.append(line)

            i += 1

        # Save final entry
        if current_text_lines:
            entries.append(
                TranscriptEntry(
                    speaker=current_speaker,
                    timestamp=current_timestamp,
                    timestamp_seconds=current_timestamp_seconds,
                    text=" ".join(current_text_lines).strip(),
                    confidence=default_confidence,
                )
            )

        return entries

    def _looks_like_speaker(self, text: str) -> bool:
        """Check if text looks like a speaker name."""
        if not text:
            return False

        # Remove common artifacts
        text = text.strip()

        # Speaker names typically:
        # - Start with capital letter
        # - Are relatively short (under 50 chars)
        # - Don't contain timestamps
        # - May contain @ for email-style names

        if len(text) > 50:
            return False

        if self._timestamp_re.search(text):
            return False

        # Check if starts with capital or common name patterns
        if text[0].isupper():
            return True

        if "@" in text:
            return True

        return False

    def _clean_speaker_name(self, name: str) -> str:
        """Clean up speaker name."""
        # Remove common artifacts
        name = name.strip()

        # Remove leading/trailing punctuation
        name = name.strip(".,;:-")

        # Remove avatar indicators
        name = re.sub(r"^\[.*?\]\s*", "", name)

        return name.strip()

    def _parse_timestamp(self, timestamp: str) -> int:
        """Parse timestamp string to seconds."""
        parts = timestamp.split(":")

        if len(parts) == 3:
            hours, minutes, seconds = map(int, parts)
            return hours * 3600 + minutes * 60 + seconds
        elif len(parts) == 2:
            minutes, seconds = map(int, parts)
            return minutes * 60 + seconds

        return 0

    def _merge_consecutive(
        self,
        entries: List[TranscriptEntry],
    ) -> List[TranscriptEntry]:
        """Merge consecutive entries from the same speaker."""
        if not entries:
            return entries

        merged = [entries[0]]

        for entry in entries[1:]:
            last = merged[-1]

            # Check if same speaker and close in time (within 60 seconds)
            if (
                entry.speaker == last.speaker
                and abs(entry.timestamp_seconds - last.timestamp_seconds) < 60
            ):
                # Merge text
                last.text = last.text + " " + entry.text
                # Keep earlier timestamp
            else:
                merged.append(entry)

        return merged

    def parse_entries_from_lines(
        self,
        lines: List[Tuple[str, float]],
    ) -> ParsedTranscript:
        """
        Parse transcript from list of (text, confidence) tuples.

        Args:
            lines: List of (text, confidence) tuples

        Returns:
            ParsedTranscript
        """
        # Combine into single text with confidences tracked separately
        full_text = "\n".join(line[0] for line in lines)
        avg_confidence = sum(line[1] for line in lines) / len(lines) if lines else 0.0

        return self.parse(full_text, default_confidence=avg_confidence)
