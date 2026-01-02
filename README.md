# Video OCR Transcript Extractor

Extract and deduplicate scrolling text from screen recordings using OCR. Designed primarily for extracting meeting transcripts from Microsoft Teams recordings.

## Features

- **Smart Frame Extraction**: Extracts frames at configurable FPS with automatic skipping of similar frames
- **Multiple OCR Engines**: Support for EasyOCR (default), PaddleOCR, and Tesseract
- **Intelligent Deduplication**: Uses fuzzy string matching to remove duplicate text from scrolling content
- **Transcript Parsing**: Automatically detects speakers, timestamps, and message content from Teams format
- **Multiple Output Formats**: JSON, Markdown, plain text, and SRT subtitles
- **CLI and Web Interface**: Both command-line and browser-based interfaces

## Installation

### Basic Installation

```bash
pip install -e .
```

### With Optional Dependencies

```bash
# With PaddleOCR support
pip install -e ".[paddleocr]"

# With Tesseract support
pip install -e ".[tesseract]"

# With web interface
pip install -e ".[web]"

# All optional dependencies
pip install -e ".[all]"
```

### System Requirements

- Python 3.9+
- FFmpeg (optional, for some video formats)
- For Tesseract: Install Tesseract OCR engine separately

## Quick Start

### Command Line

```bash
# Basic usage
video-ocr extract meeting_recording.mp4

# With options
video-ocr extract meeting_recording.mp4 \
    --engine easyocr \
    --fps 2 \
    --output-format all \
    -o ./output/

# View video info
video-ocr info video.mp4

# List available OCR engines
video-ocr engines
```

### Python API

```python
from pathlib import Path
from video_ocr.core.frame_extractor import FrameExtractor
from video_ocr.core.ocr_engine import OCREngine
from video_ocr.core.deduplicator import TextDeduplicator
from video_ocr.core.transcript_parser import TranscriptParser
from video_ocr.core.output_formatter import OutputFormatter, create_metadata

# Initialize components
frame_extractor = FrameExtractor(fps=2.0)
ocr_engine = OCREngine(engine="easyocr", languages=["en"])
deduplicator = TextDeduplicator(similarity_threshold=0.85)
transcript_parser = TranscriptParser()
formatter = OutputFormatter()

# Process video
video_path = Path("meeting.mp4")
frames = list(frame_extractor.extract_frames(video_path))

# Run OCR on frames
ocr_results = []
for frame_info in frames:
    result = ocr_engine.process_frame(
        frame_info.frame,
        frame_number=frame_info.frame_number,
        timestamp_seconds=frame_info.timestamp_seconds,
    )
    ocr_results.append(result)

# Deduplicate text
dedup_result = deduplicator.deduplicate(ocr_results)

# Parse transcript
text_lines = [(t.text, t.confidence) for t in dedup_result.texts]
parsed = transcript_parser.parse_entries_from_lines(text_lines)

# Output as JSON
video_info = frame_extractor.get_video_info(video_path)
metadata = create_metadata(
    source_file=video_path.name,
    duration_seconds=video_info["duration_seconds"],
    frames_processed=len(frames),
    frames_with_text=dedup_result.frames_with_new_content,
    ocr_engine="easyocr",
    confidence_threshold=0.5,
)

json_output = formatter.format_json(parsed, metadata, dedup_result.full_text)
print(json_output)
```

### Web Interface

```bash
# Start web server
python -m video_ocr.web.app

# Or with custom options
python -c "from video_ocr.web.app import run_server; run_server(port=8080)"
```

Then open http://localhost:8080 in your browser.

## Output Formats

### JSON

```json
{
  "metadata": {
    "source_file": "meeting.mp4",
    "duration_seconds": 120.5,
    "frames_processed": 241,
    "ocr_engine": "easyocr"
  },
  "transcript": [
    {
      "speaker": "John Smith",
      "timestamp": "11:30:00",
      "timestamp_seconds": 41400,
      "text": "Hello everyone, welcome to the meeting.",
      "confidence": 0.95
    }
  ],
  "statistics": {
    "total_speakers": 3,
    "total_messages": 45,
    "word_count": 1250
  }
}
```

### Markdown

```markdown
# Meeting Transcript

**Source**: meeting.mp4
**Duration**: 2m 0s

---

## Transcript

**John Smith** (11:30:00)
Hello everyone, welcome to the meeting.

**Jane Doe** (11:30:15)
Thanks for joining us.
```

## Configuration

Configuration can be set via YAML file or command-line options:

```yaml
# config/default.yaml
frame_extraction:
  fps: 2.0
  skip_similar_threshold: 0.95

ocr:
  engine: "easyocr"
  languages: ["en"]
  confidence_threshold: 0.5
  gpu: true

deduplication:
  similarity_threshold: 0.85
  min_new_chars: 10
```

## OCR Engine Comparison

| Engine | Accuracy | Speed | GPU Support | Languages |
|--------|----------|-------|-------------|-----------|
| EasyOCR | High | Medium | Yes | 70+ |
| PaddleOCR | Highest | Fast | Yes | 80+ |
| Tesseract | Medium | Medium | No | 100+ |

## Development

### Setup

```bash
# Clone and install dev dependencies
git clone <repo-url>
cd video-ocr-transcript-extractor
pip install -e ".[dev]"
```

### Running Tests

```bash
pytest
pytest --cov=video_ocr --cov-report=html
```

### Code Quality

```bash
black src tests
ruff check src tests
mypy src
```

## Project Structure

```
video-ocr-transcript-extractor/
├── src/video_ocr/
│   ├── core/               # Main processing components
│   │   ├── frame_extractor.py
│   │   ├── ocr_engine.py
│   │   ├── deduplicator.py
│   │   ├── transcript_parser.py
│   │   └── output_formatter.py
│   ├── engines/            # OCR engine implementations
│   │   ├── base.py
│   │   ├── easyocr_engine.py
│   │   ├── paddleocr_engine.py
│   │   └── tesseract_engine.py
│   ├── utils/              # Utility functions
│   ├── web/                # Web interface
│   └── cli.py              # CLI interface
├── tests/                  # Unit tests
├── config/                 # Configuration files
└── pyproject.toml
```

## License

MIT License

## Contributing

Contributions are welcome! Please read the contributing guidelines and submit pull requests.
