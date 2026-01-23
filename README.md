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
- **Multiple OCR Engines**: Support for EasyOCR (default), Tesseract, Surya, docTR, and TrOCR
- **Intelligent Deduplication**: Uses Levenshtein-based fuzzy string matching to handle scrolling content overlap
- **Transcript Parsing**: Automatically detects Zoom/Teams format with speakers, timestamps (`HH:MM:SS`), and messages
- **Multiple Output Formats**: JSON, Markdown, plain text, and SRT subtitles
- **GPU Acceleration**: Optional CUDA support for faster OCR processing
- **Batch Processing**: Process multiple videos sequentially with automatic saving
- **Dual Interface**: Both command-line (CLI) and web browser interfaces
- **Queue System**: Web interface supports queueing multiple videos for batch processing
- **Real-time Progress**: Web interface shows processing progress with stable ETA display

## Installation

### Prerequisites

- Python 3.9 or higher
- For Tesseract: Install [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) system-wide

### Basic Installation (CPU only)

```bash
# Clone the repository
git clone https://github.com/benoitmarteau/video_text_extractor.git
cd video_text_extractor

# Install in development mode
pip install -e .
```

### Installation with GPU Support (Recommended)

**Option A: Use the install script (easiest)**
```bash
git clone https://github.com/benoitmarteau/video_text_extractor.git
cd video_text_extractor

# Run the GPU installation script (installs PyTorch with CUDA + all OCR engines)
python install_gpu.py

# Or specify CUDA version: --cuda 118, --cuda 126 (default), --cuda 128
python install_gpu.py --cuda 126
```

**Option B: Manual installation**
```bash
git clone https://github.com/benoitmarteau/video_text_extractor.git
cd video_text_extractor

# Step 1: Install PyTorch with CUDA FIRST
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# Step 2: Install package with all OCR engines + web interface
pip install -e ".[all]"
```

### Installation Options

| Option | Command | Description |
|--------|---------|-------------|
| `[all]` | `pip install -e ".[all]"` | All OCR engines + web + dev tools |
| `[all-ocr-web]` | `pip install -e ".[all-ocr-web]"` | All OCR engines + web (no dev tools) |
| `[all-ocr]` | `pip install -e ".[all-ocr]"` | All OCR engines only |
| `[surya]` | `pip install -e ".[surya]"` | Surya OCR (GPU, Python 3.10+) |
| `[doctr]` | `pip install -e ".[doctr]"` | docTR (Mindee) |
| `[trocr]` | `pip install -e ".[trocr]"` | TrOCR (Microsoft, GPU) |
| `[tesseract]` | `pip install -e ".[tesseract]"` | Tesseract |
| `[web]` | `pip install -e ".[web]"` | Web interface (Flask) |
| `[dev]` | `pip install -e ".[dev]"` | Dev tools (pytest, black, etc.) |

### GPU Support

For GPU acceleration with EasyOCR, Surya, docTR, or TrOCR, install PyTorch with CUDA **before** installing other dependencies:

```bash
# Recommended: CUDA 12.6 (compatible with PyTorch 2.7+ required by Surya)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# Alternative: CUDA 11.8 (for older systems)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Alternative: CUDA 12.8 (for newest GPUs)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

**Important Notes:**
- **Surya OCR** requires PyTorch 2.7+ and Python 3.10+. Use `cu126` or `cu128` for full compatibility.
- **TrOCR** also requires GPU and PyTorch with CUDA.
- Install PyTorch with CUDA **first**, then install the package dependencies.

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

# Batch process multiple videos (files processed sequentially, results auto-saved)
video-ocr batch video1.mp4 video2.mp4 video3.mp4 -o ./transcripts/

# Batch process all videos in a directory
video-ocr batch *.mp4 -o ./transcripts/ --engine surya --gpu

# View video information
video-ocr info video.mp4

# List available OCR engines
video-ocr engines
```

### Web Interface

```bash
# Start the web server (default: http://localhost:8080)
python -m video_ocr.web.app

# Or from Python:
from video_ocr.web.app import run_server
run_server(host="0.0.0.0", port=8080, debug=False)
```

**Note:** Ensure Flask is installed (`pip install -e ".[web]"` or `pip install flask flask-cors`).

The web interface provides:
- **Single Video Mode**:
  - Drag-and-drop video upload
  - Real-time processing progress with stable ETA display
  - Preview of extracted transcript
  - Download in JSON, Markdown, or plain text formats
- **Queue Mode**:
  - Add multiple videos to a processing queue (supports multi-file selection)
  - Process all videos sequentially with per-video progress tracking
  - Results are automatically saved
  - Download individual results or **Download All** combined into single file
  - **Cross-video deduplication**: Remove duplicate content across overlapping recordings

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
│    - EasyOCR / Tesseract / Surya    │
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

| Engine | Accuracy | Speed | GPU Required | Languages | Best For |
|--------|----------|-------|--------------|-----------|----------|
| EasyOCR | High | Medium | No (optional) | 70+ | Default, good balance |
| Tesseract | Medium | Medium | No | 100+ | Offline, CPU-only |
| Surya | High | Very Fast | Yes* | 90+ | Fast multilingual OCR |
| docTR | High | Fast | No (optional) | Limited | Document-focused OCR |
| TrOCR | Highest | Slow | Yes | Limited | Printed/handwritten text |

*Surya requires Python 3.10+ and PyTorch 2.7+

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
│   │   ├── tesseract_engine.py # Tesseract backend
│   │   ├── surya_engine.py     # Surya OCR backend (GPU)
│   │   ├── doctr_engine.py     # docTR backend
│   │   └── trocr_engine.py     # TrOCR backend (GPU)
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
- **Tesseract**: ~0.5-1 second per frame (CPU only)
- **Surya**: ~0.1-0.3 second per frame (GPU required)
- **docTR**: ~0.3-0.5 second per frame (with GPU)
- **TrOCR**: ~1-2 seconds per frame (GPU required)
- **1-hour video**: ~10-30 minutes total processing time

### GPU Acceleration

For optimal performance, use GPU acceleration:
- Significant speedup with CUDA-enabled GPUs
- EasyOCR, Surya, docTR, and TrOCR support GPU processing
- Surya and TrOCR require GPU (will not work on CPU-only systems)
- Ensure PyTorch is installed with CUDA support

## Troubleshooting

### "No transcript entries found"

This usually means the OCR detected text but couldn't parse it as a structured transcript. The raw OCR text is still available in the output. Try:
- Lowering the confidence threshold (`--confidence-threshold 0.3`)
- Using a different OCR engine (`--engine doctr`)
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

### Web server won't start

If the web server fails to start:

1. **Flask not installed**: Install web dependencies:
   ```bash
   pip install -e ".[web]"
   # Or manually:
   pip install flask flask-cors
   ```

2. **Port already in use**: The default port is 8080. If it's in use, modify the call:
   ```python
   from video_ocr.web.app import run_server
   run_server(port=8081)  # Use a different port
   ```

3. **Import errors**: Ensure the package is installed in development mode:
   ```bash
   pip install -e .
   ```

## License

MIT License

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.
