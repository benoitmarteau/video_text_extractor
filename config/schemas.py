"""Configuration schemas and dataclasses for Video OCR."""

from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path
import yaml


@dataclass
class FrameExtractionConfig:
    """Configuration for frame extraction."""
    fps: float = 2.0
    skip_similar_threshold: float = 0.95
    max_frames: Optional[int] = None


@dataclass
class OCRConfig:
    """Configuration for OCR processing."""
    engine: str = "easyocr"
    languages: List[str] = field(default_factory=lambda: ["en"])
    confidence_threshold: float = 0.5
    gpu: bool = True
    batch_size: int = 1


@dataclass
class DeduplicationConfig:
    """Configuration for text deduplication."""
    similarity_threshold: float = 0.85
    min_new_chars: int = 10
    window_size: int = 3


@dataclass
class TranscriptParsingConfig:
    """Configuration for transcript parsing."""
    enabled: bool = True
    timestamp_pattern: str = r"\d{1,2}:\d{2}:\d{2}"
    speaker_detection: bool = True
    merge_consecutive: bool = True


@dataclass
class OutputConfig:
    """Configuration for output generation."""
    format: str = "all"
    include_metadata: bool = True
    include_statistics: bool = True
    include_raw_text: bool = True
    pretty_print: bool = True


@dataclass
class LoggingConfig:
    """Configuration for logging."""
    level: str = "INFO"
    file: Optional[str] = None
    rich_formatting: bool = True


@dataclass
class ProcessingConfig:
    """Configuration for processing behavior."""
    temp_dir: Optional[str] = None
    cleanup_temp: bool = True
    save_debug_frames: bool = False


@dataclass
class VideoOCRConfig:
    """Main configuration container for Video OCR."""
    frame_extraction: FrameExtractionConfig = field(default_factory=FrameExtractionConfig)
    ocr: OCRConfig = field(default_factory=OCRConfig)
    deduplication: DeduplicationConfig = field(default_factory=DeduplicationConfig)
    transcript_parsing: TranscriptParsingConfig = field(default_factory=TranscriptParsingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)

    @classmethod
    def from_yaml(cls, path: Path) -> "VideoOCRConfig":
        """Load configuration from a YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> "VideoOCRConfig":
        """Create configuration from a dictionary."""
        return cls(
            frame_extraction=FrameExtractionConfig(
                **data.get("frame_extraction", {})
            ),
            ocr=OCRConfig(**data.get("ocr", {})),
            deduplication=DeduplicationConfig(**data.get("deduplication", {})),
            transcript_parsing=TranscriptParsingConfig(
                **data.get("transcript_parsing", {})
            ),
            output=OutputConfig(**data.get("output", {})),
            logging=LoggingConfig(**data.get("logging", {})),
            processing=ProcessingConfig(**data.get("processing", {})),
        )

    def to_dict(self) -> dict:
        """Convert configuration to a dictionary."""
        from dataclasses import asdict
        return asdict(self)

    def merge_with(self, overrides: dict) -> "VideoOCRConfig":
        """Create a new config with overrides applied."""
        base = self.to_dict()

        def deep_merge(base_dict: dict, override_dict: dict) -> dict:
            result = base_dict.copy()
            for key, value in override_dict.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = deep_merge(result[key], value)
                elif value is not None:
                    result[key] = value
            return result

        merged = deep_merge(base, overrides)
        return VideoOCRConfig.from_dict(merged)


def get_default_config() -> VideoOCRConfig:
    """Get the default configuration."""
    return VideoOCRConfig()


def load_config(config_path: Optional[Path] = None) -> VideoOCRConfig:
    """Load configuration from file or return defaults."""
    if config_path and config_path.exists():
        return VideoOCRConfig.from_yaml(config_path)

    # Try to load from default locations
    default_locations = [
        Path("config/default.yaml"),
        Path("~/.video-ocr/config.yaml").expanduser(),
        Path("/etc/video-ocr/config.yaml"),
    ]

    for location in default_locations:
        if location.exists():
            return VideoOCRConfig.from_yaml(location)

    return get_default_config()
