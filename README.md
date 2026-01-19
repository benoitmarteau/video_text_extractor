# Video OCR Transcript Extractor

A Python application that extracts, deduplicates, and parses scrolling text from screen recordings using Optical Character Recognition (OCR). Designed for extracting meeting transcripts from **Zoom** and **Microsoft Teams** recordings, but works with any video containing scrolling text content.

## Why This Project?

When you record a Zoom or Teams meeting, the transcript/caption panel shows scrolling text with speaker names, timestamps, and messages. This tool:
1. **Extracts frames** from the video at regular intervals
2. **Runs OCR** on each frame to detect text
3. **Deduplicates** the text (since scrolling content repeats across frames)
4. **Parses** the structured transcript format (speaker, timestamp, message)
5. **Outputs** the result in multiple formats

## Features

- **Smart Frame Extraction**: Extracts frames at configurable FPS (default: 2) using OpenCV
- **Multiple OCR Engines**: Support for EasyOCR (default), PaddleOCR (fastest/most accurate), and Tesseract
- **Intelligent Deduplication**: Uses Levenshtein-based fuzzy string matching to handle scrolling content overlap
- **Transcript Parsing**: Automatically detects Zoom/Teams format with speakers, timestamps (`HH:MM:SS`), and messages
- **Multiple Output Formats**: JSON, Markdown, plain text, and SRT subtitles
- **GPU Acceleration**: Optional CUDA support for faster OCR processing
- **Parallel Processing**: Multi-threaded CPU processing for faster extraction without GPU
- **Dual Interface**: Both command-line (CLI) and web browser interfaces
- **Real-time Progress**: Web interface shows processing progress with frame-by-frame updates

## Installation

### Prerequisites

- Python 3.9 or higher
- For Tesseract: Install [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) system-wide

### Basic Installation

```bash
# Clone the repository
git clone https://github.com/benoitmarteau/video_text_extractor.git
cd video_text_extractor

# Install in development mode
pip install -e .
```

### With Optional Dependencies

```bash
# With PaddleOCR support (recommended for production)
pip install -e ".[paddleocr]"

# With Tesseract support
pip install -e ".[tesseract]"

# With web interface
pip install -e ".[web]"

# All optional dependencies
pip install -e ".[all]"
```

### GPU Support

For GPU acceleration with EasyOCR or PaddleOCR, install PyTorch with CUDA:

```bash
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Then set `gpu: true` in configuration or use `--gpu` flag in CLI.

## Quick Start

### Command Line Interface

```bash
# Basic usage - extract transcript from video
video-ocr extract meeting_recording.mp4

# With all options
video-ocr extract meeting_recording.mp4 \
    --engine easyocr \
    --fps 2 \
    --similarity-threshold 0.85 \
    --confidence-threshold 0.5 \
    --language en \
    --gpu \
    --output-format all \
    -o ./output/

# Use parallel processing with 4 workers (CPU mode)
video-ocr extract meeting_recording.mp4 --no-gpu --parallel --workers 4

# Disable parallel processing
video-ocr extract meeting_recording.mp4 --no-parallel

# View video information
video-ocr info video.mp4

# List available OCR engines
video-ocr engines
```

### Web Interface

```bash
# Start the web server
python -m video_ocr.web.app

# Opens at http://localhost:5000
```

The web interface provides:
- Drag-and-drop video upload
- Real-time processing progress (frame counter, percentage)
- Preview of extracted transcript
- Download in JSON, Markdown, or plain text formats

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
ocr_engine = OCREngine(engine="easyocr", languages=["en"], gpu=True)
deduplicator = TextDeduplicator(similarity_threshold=0.85)
transcript_parser = TranscriptParser()
formatter = OutputFormatter()

# Process video
video_path = Path("meeting.mp4")

# Extract all frames (skip_similar=False for scrolling content)
frames = list(frame_extractor.extract_frames(video_path, skip_similar=False))

# Run OCR on each frame
ocr_results = []
for frame_info in frames:
    result = ocr_engine.process_frame(
        frame_info.frame,
        frame_number=frame_info.frame_number,
        timestamp_seconds=frame_info.timestamp_seconds,
    )
    ocr_results.append(result)

# Deduplicate text from scrolling content
dedup_result = deduplicator.deduplicate(ocr_results)

# Parse transcript structure
text_lines = [(t.text, t.confidence) for t in dedup_result.texts]
parsed = transcript_parser.parse_entries_from_lines(text_lines)

# Generate output
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
```

## Processing Pipeline

```
Video File
    │
    ▼
┌─────────────────────────────────────┐
│ 1. Frame Extraction                 │
│    - Extract at 2 FPS               │
│    - OpenCV video processing        │
│    - Memory-efficient generators    │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ 2. OCR Processing                   │
│    - EasyOCR / PaddleOCR / Tesseract│
│    - Detect text regions            │
│    - Filter by confidence (≥0.5)    │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ 3. Text Deduplication               │
│    - Levenshtein fuzzy matching     │
│    - Sliding window comparison      │
│    - Handle scrolling overlap       │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ 4. Transcript Parsing               │
│    - Detect timestamps (HH:MM:SS)   │
│    - Identify speaker names         │
│    - Merge consecutive messages     │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ 5. Output Formatting                │
│    - JSON (structured data)         │
│    - Markdown (human readable)      │
│    - Plain text (simple)            │
│    - SRT (subtitles)                │
└─────────────────────────────────────┘
```

## Output Formats

### JSON

```json
{
  "metadata": {
    "source_file": "meeting.mp4",
    "duration_seconds": 3600,
    "frames_processed": 7200,
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
  },
  "raw_text": "Full deduplicated OCR text..."
}
```

### Markdown

```markdown
# Meeting Transcript

**Source**: meeting.mp4
**Duration**: 1h 0m 0s

---

## Transcript

**John Smith** (11:30:00)
Hello everyone, welcome to the meeting.

**Jane Doe** (11:30:15)
Thanks for joining us today.
```

### SRT (Subtitles)

```
1
00:00:10,000 --> 00:00:15,000
[John Smith] Hello everyone, welcome to the meeting.

2
00:00:15,000 --> 00:00:20,000
[Jane Doe] Thanks for joining us today.
```

## OCR Engine Comparison

| Engine | Accuracy | Speed | GPU Support | Languages | Best For |
|--------|----------|-------|-------------|-----------|----------|
| EasyOCR | High | Medium | Yes | 70+ | Default, good balance |
| PaddleOCR | Highest | Fast | Yes | 80+ | Production, accuracy |
| Tesseract | Medium | Medium | No | 100+ | Offline, CPU-only |

## Configuration

### Default Configuration (`config/default.yaml`)

```yaml
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
  window_size: 3

transcript_parsing:
  merge_consecutive: true
  merge_threshold_seconds: 60

output:
  formats: ["json", "markdown", "text"]
  include_metadata: true
  include_statistics: true

logging:
  level: "INFO"
  rich_formatting: true
```

## Project Structure

```
video_text_extractor/
├── src/video_ocr/
│   ├── core/                   # Main processing components
│   │   ├── frame_extractor.py  # Video frame extraction
│   │   ├── ocr_engine.py       # OCR wrapper/interface
│   │   ├── deduplicator.py     # Text deduplication
│   │   ├── transcript_parser.py # Teams format parsing
│   │   └── output_formatter.py # Output generation
│   ├── engines/                # OCR implementations
│   │   ├── base.py             # Abstract base class
│   │   ├── easyocr_engine.py   # EasyOCR backend
│   │   ├── paddleocr_engine.py # PaddleOCR backend
│   │   └── tesseract_engine.py # Tesseract backend
│   ├── utils/                  # Utility functions
│   │   ├── logging_config.py   # Rich console logging
│   │   ├── similarity.py       # Fuzzy string matching
│   │   └── image_utils.py      # Frame preprocessing
│   ├── web/                    # Web interface
│   │   ├── app.py              # Flask application
│   │   └── templates/          # HTML templates
│   └── cli.py                  # Click CLI
├── config/
│   ├── default.yaml            # Default settings
│   └── schemas.py              # Configuration dataclasses
├── tests/                      # Unit tests
├── pyproject.toml              # Project metadata
└── README.md
```

## Development

### Setup

```bash
git clone https://github.com/benoitmarteau/video_text_extractor.git
cd video_text_extractor
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

## Performance Notes

- **Frame extraction**: ~100 frames/second
- **EasyOCR**: ~0.5-1 second per frame (with GPU)
- **PaddleOCR**: ~0.2-0.5 second per frame (with GPU)
- **Tesseract**: ~0.5-1 second per frame (CPU only)
- **Parallel CPU processing**: ~2-4x speedup with multi-threading
- **1-hour video**: ~10-30 minutes total processing time

### Parallel Processing

When running without a GPU, the tool automatically uses multi-threaded processing to speed up OCR:

- By default, uses up to 4 CPU cores (or fewer based on available cores)
- Each worker thread has its own OCR engine instance
- Particularly effective for CPU-only environments
- Can be disabled with `--no-parallel` flag

## Troubleshooting

### "No transcript entries found"

This usually means the OCR detected text but couldn't parse it as a structured transcript. The raw OCR text is still available in the output. Try:
- Lowering the confidence threshold (`--confidence-threshold 0.3`)
- Using a different OCR engine (`--engine paddleocr`)
- Checking if the video actually contains Teams transcript format

### Out of memory with large videos

For videos longer than 1 hour:
- Use a lower FPS (`--fps 1`)
- Process in chunks
- Ensure GPU mode is enabled for faster processing

### GPU not detected

Make sure PyTorch is installed with CUDA support:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

## License

MIT License

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.
