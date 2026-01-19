"""Output formatting for extraction results."""

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from video_ocr.core.transcript_parser import ParsedTranscript


@dataclass
class ExtractionMetadata:
    """Metadata about the extraction process."""

    source_file: str
    duration_seconds: float
    frames_processed: int
    frames_with_text: int
    extraction_timestamp: str
    ocr_engine: str
    confidence_threshold: float

    def to_dict(self) -> dict:
        return {
            "source_file": self.source_file,
            "duration_seconds": self.duration_seconds,
            "frames_processed": self.frames_processed,
            "frames_with_text": self.frames_with_text,
            "extraction_timestamp": self.extraction_timestamp,
            "ocr_engine": self.ocr_engine,
            "confidence_threshold": self.confidence_threshold,
        }


@dataclass
class ExtractionStatistics:
    """Statistics about the extraction."""

    total_speakers: int
    total_messages: int
    word_count: int
    unique_words: int
    character_count: int
    average_confidence: float

    def to_dict(self) -> dict:
        return {
            "total_speakers": self.total_speakers,
            "total_messages": self.total_messages,
            "word_count": self.word_count,
            "unique_words": self.unique_words,
            "character_count": self.character_count,
            "average_confidence": round(self.average_confidence, 3),
        }


class OutputFormatter:
    """
    Format extraction results into various output formats.

    Supports JSON, Markdown, and plain text output.
    """

    def __init__(
        self,
        include_metadata: bool = True,
        include_statistics: bool = True,
        include_raw_text: bool = True,
        pretty_print: bool = True,
    ):
        """
        Initialize output formatter.

        Args:
            include_metadata: Include extraction metadata
            include_statistics: Include statistics
            include_raw_text: Include raw text in output
            pretty_print: Pretty print JSON output
        """
        self.include_metadata = include_metadata
        self.include_statistics = include_statistics
        self.include_raw_text = include_raw_text
        self.pretty_print = pretty_print

    def format_json(
        self,
        transcript: ParsedTranscript,
        metadata: ExtractionMetadata,
        raw_text: str = "",
    ) -> str:
        """
        Format results as JSON.

        Args:
            transcript: Parsed transcript
            metadata: Extraction metadata
            raw_text: Raw extracted text

        Returns:
            JSON string
        """
        output: Dict[str, Any] = {}

        if self.include_metadata:
            output["metadata"] = metadata.to_dict()

        # Transcript entries
        output["transcript"] = [entry.to_dict() for entry in transcript.entries]

        if self.include_raw_text:
            output["raw_text"] = raw_text or transcript.raw_text

        if self.include_statistics:
            stats = self._calculate_statistics(transcript, raw_text)
            output["statistics"] = stats.to_dict()

        if self.pretty_print:
            return json.dumps(output, indent=2, ensure_ascii=False)
        else:
            return json.dumps(output, ensure_ascii=False)

    def format_markdown(
        self,
        transcript: ParsedTranscript,
        metadata: ExtractionMetadata,
    ) -> str:
        """
        Format results as Markdown.

        Args:
            transcript: Parsed transcript
            metadata: Extraction metadata

        Returns:
            Markdown string
        """
        lines = []

        # Header
        lines.append("# Meeting Transcript")
        lines.append("")

        # Metadata
        if self.include_metadata:
            lines.append(f"**Source**: {metadata.source_file}")
            lines.append(f"**Duration**: {self._format_duration(metadata.duration_seconds)}")
            lines.append(f"**Date Extracted**: {metadata.extraction_timestamp[:10]}")
            lines.append(f"**OCR Engine**: {metadata.ocr_engine}")
            lines.append("")

        # Statistics
        if self.include_statistics:
            stats = self._calculate_statistics(transcript, "")
            lines.append("## Summary")
            lines.append("")
            lines.append(f"- **Speakers**: {stats.total_speakers}")
            lines.append(f"- **Messages**: {stats.total_messages}")
            lines.append(f"- **Word Count**: {stats.word_count}")
            lines.append("")

        # Separator
        lines.append("---")
        lines.append("")

        # Transcript
        lines.append("## Transcript")
        lines.append("")

        for entry in transcript.entries:
            if entry.speaker:
                lines.append(f"**{entry.speaker}** ({entry.timestamp})")
            else:
                lines.append(f"*({entry.timestamp})*")

            lines.append(entry.text)
            lines.append("")

        # Footer
        lines.append("---")
        lines.append("")
        lines.append("*Extracted using Video OCR Transcript Extractor v1.0*")

        return "\n".join(lines)

    def format_text(
        self,
        transcript: ParsedTranscript,
        metadata: Optional[ExtractionMetadata] = None,
    ) -> str:
        """
        Format results as plain text.

        Args:
            transcript: Parsed transcript
            metadata: Optional extraction metadata

        Returns:
            Plain text string
        """
        lines = []

        # Header
        if metadata and self.include_metadata:
            lines.append(f"Source: {metadata.source_file}")
            lines.append(f"Duration: {self._format_duration(metadata.duration_seconds)}")
            lines.append("")
            lines.append("-" * 50)
            lines.append("")

        # Transcript entries
        for entry in transcript.entries:
            if entry.speaker:
                lines.append(f"[{entry.timestamp}] {entry.speaker}:")
            else:
                lines.append(f"[{entry.timestamp}]")

            lines.append(f"  {entry.text}")
            lines.append("")

        return "\n".join(lines)

    def format_srt(
        self,
        transcript: ParsedTranscript,
    ) -> str:
        """
        Format results as SRT subtitles.

        Args:
            transcript: Parsed transcript

        Returns:
            SRT format string
        """
        lines = []

        for i, entry in enumerate(transcript.entries, 1):
            lines.append(str(i))

            # Calculate end time (use next entry's start or add 5 seconds)
            start_time = self._seconds_to_srt_time(entry.timestamp_seconds)

            if i < len(transcript.entries):
                end_seconds = transcript.entries[i].timestamp_seconds
            else:
                end_seconds = entry.timestamp_seconds + 5

            end_time = self._seconds_to_srt_time(end_seconds)

            lines.append(f"{start_time} --> {end_time}")

            # Add speaker prefix if available
            if entry.speaker:
                lines.append(f"[{entry.speaker}] {entry.text}")
            else:
                lines.append(entry.text)

            lines.append("")

        return "\n".join(lines)

    def save_all_formats(
        self,
        transcript: ParsedTranscript,
        metadata: ExtractionMetadata,
        raw_text: str,
        output_dir: Path,
        base_name: str = "transcript",
    ) -> Dict[str, Path]:
        """
        Save results in all formats.

        Args:
            transcript: Parsed transcript
            metadata: Extraction metadata
            raw_text: Raw extracted text
            output_dir: Output directory
            base_name: Base filename (without extension)

        Returns:
            Dictionary mapping format to file path
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        saved_files = {}

        # JSON
        json_path = output_dir / f"{base_name}.json"
        json_content = self.format_json(transcript, metadata, raw_text)
        json_path.write_text(json_content, encoding="utf-8")
        saved_files["json"] = json_path

        # Markdown
        md_path = output_dir / f"{base_name}.md"
        md_content = self.format_markdown(transcript, metadata)
        md_path.write_text(md_content, encoding="utf-8")
        saved_files["markdown"] = md_path

        # Plain text
        txt_path = output_dir / f"{base_name}.txt"
        txt_content = self.format_text(transcript, metadata)
        txt_path.write_text(txt_content, encoding="utf-8")
        saved_files["text"] = txt_path

        # SRT
        srt_path = output_dir / f"{base_name}.srt"
        srt_content = self.format_srt(transcript)
        srt_path.write_text(srt_content, encoding="utf-8")
        saved_files["srt"] = srt_path

        return saved_files

    def _calculate_statistics(
        self,
        transcript: ParsedTranscript,
        raw_text: str,
    ) -> ExtractionStatistics:
        """Calculate extraction statistics."""
        # Try to get text from transcript entries first
        all_text = " ".join(e.text for e in transcript.entries)

        # If no transcript entries, use raw_text instead
        # This fixes the "0 words" issue when transcript format isn't detected
        if not all_text.strip() and raw_text:
            all_text = raw_text

        words = all_text.split()

        confidences = [e.confidence for e in transcript.entries if e.confidence > 0]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        return ExtractionStatistics(
            total_speakers=len(transcript.speakers),
            total_messages=len(transcript.entries),
            word_count=len(words),
            unique_words=len(set(w.lower() for w in words)),
            character_count=len(all_text),
            average_confidence=avg_confidence,
        )

    def _format_duration(self, seconds: float) -> str:
        """Format duration in seconds to human-readable string."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)

        if hours > 0:
            return f"{hours}h {minutes}m {secs}s"
        elif minutes > 0:
            return f"{minutes}m {secs}s"
        else:
            return f"{secs}s"

    def _seconds_to_srt_time(self, seconds: float) -> str:
        """Convert seconds to SRT timestamp format (HH:MM:SS,mmm)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)

        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def create_metadata(
    source_file: str,
    duration_seconds: float,
    frames_processed: int,
    frames_with_text: int,
    ocr_engine: str,
    confidence_threshold: float,
) -> ExtractionMetadata:
    """Create extraction metadata."""
    return ExtractionMetadata(
        source_file=source_file,
        duration_seconds=duration_seconds,
        frames_processed=frames_processed,
        frames_with_text=frames_with_text,
        extraction_timestamp=datetime.utcnow().isoformat() + "Z",
        ocr_engine=ocr_engine,
        confidence_threshold=confidence_threshold,
    )
