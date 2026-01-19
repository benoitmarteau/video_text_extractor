# Project Context Document

This document tracks the development history, architecture decisions, and technical details for the Video OCR Transcript Extractor project.

## Project Overview

**Video OCR Transcript Extractor** extracts scrolling text from screen recordings using OCR and intelligently deduplicates the content. The primary use case is extracting meeting transcripts from **Zoom** and **Microsoft Teams** recordings where the transcript/caption panel shows scrolling text with speaker names, timestamps, and messages.

## Current Status

- **Phase**: Complete - Production ready
- **Last Updated**: 2025-01-19
- **Version**: 1.0.0
- **Repository**: https://github.com/benoitmarteau/video_text_extractor

## Development History

### Initial Implementation
- Built complete processing pipeline: frame extraction, OCR, deduplication, parsing, output formatting
- Implemented three OCR engine backends: EasyOCR (default), PaddleOCR, Tesseract
- Created CLI interface with Click and web interface with Flask

### Bug Fixes
- **Frame similarity issue**: Initial implementation used aggressive frame skipping (skip_similar=True) which caused only 1 frame to be extracted from 997 total frames. Fixed by disabling skip_similar for scrolling content since the OCR deduplication handles duplicates better.
- **"No transcript entries found"**: Added raw_text fallback in results and updated web UI to show raw OCR text when no structured transcript is detected.

### Enhancements
- **Progress tracking**: Added detailed logging with job ID prefix, frame-by-frame progress, and status phases
- **Web UI improvements**: Real-time percentage display, frame counter, progress phases, 500ms status polling
- **Parallel processing**: Added multi-threaded OCR processing for CPU-only environments (2-4x speedup)
- **Word count fix**: Statistics now show word count from raw OCR text when no transcript format is detected

## Architecture

### Processing Pipeline

```
Video → Frame Extraction → OCR → Deduplication → Transcript Parsing → Output
```

1. **Frame Extraction** (`core/frame_extractor.py`)
   - Extracts frames at configurable FPS (default: 2)
   - Uses OpenCV for video processing
   - Generator-based for memory efficiency
   - Optional histogram-based similarity detection

2. **OCR Engine** (`core/ocr_engine.py` + `engines/`)
   - Unified interface with lazy initialization
   - Registry pattern for engine selection
   - GPU acceleration support
   - Confidence filtering

3. **Deduplication** (`core/deduplicator.py`)
   - Levenshtein-based fuzzy matching (rapidfuzz)
   - Sliding window comparison (default: 3 frames)
   - Line-by-line new content detection
   - Handles scrolling text overlap

4. **Transcript Parsing** (`core/transcript_parser.py`)
   - Regex-based timestamp detection (`HH:MM:SS`)
   - Speaker name heuristics
   - Consecutive message merging
   - Teams format optimized

5. **Output Formatting** (`core/output_formatter.py`)
   - JSON with metadata and statistics
   - Markdown for human readability
   - Plain text with timestamps
   - SRT subtitles

### Design Patterns Used

| Pattern | Where | Why |
|---------|-------|-----|
| Generator | Frame extraction | Memory efficiency for large videos |
| Lazy Init | OCR engines | Avoid loading models until needed |
| Abstract Base | BaseOCREngine | Common interface for all OCR backends |
| Registry | OCR engine selection | Dynamic engine loading |
| Dataclass | All data structures | Type-safe, clean data handling |

## Components

### Core Components

| Component | File | Description |
|-----------|------|-------------|
| Frame Extractor | `core/frame_extractor.py` | Extract frames from video with smart sampling |
| OCR Engine | `core/ocr_engine.py` | Unified interface for multiple OCR backends |
| Parallel OCR | `core/parallel_ocr.py` | Multi-threaded OCR processing for CPU mode |
| Deduplicator | `core/deduplicator.py` | Remove duplicate text from scrolling content |
| Transcript Parser | `core/transcript_parser.py` | Parse Teams transcript format |
| Output Formatter | `core/output_formatter.py` | Generate JSON, Markdown, Text, SRT output |

### OCR Engines

| Engine | File | GPU | Notes |
|--------|------|-----|-------|
| EasyOCR | `engines/easyocr_engine.py` | Yes | Default, good balance of speed/accuracy |
| PaddleOCR | `engines/paddleocr_engine.py` | Yes | Best accuracy, requires extra install |
| Tesseract | `engines/tesseract_engine.py` | No | Requires system Tesseract installation |

### Utilities

| Utility | File | Description |
|---------|------|-------------|
| Logging | `utils/logging_config.py` | Rich console logging with fallback |
| Similarity | `utils/similarity.py` | Levenshtein ratio, text normalization |
| Image Utils | `utils/image_utils.py` | Frame preprocessing, similarity calc |

### Interfaces

| Interface | File | Framework | Description |
|-----------|------|-----------|-------------|
| CLI | `cli.py` | Click + Rich | Commands: extract, info, engines |
| Web | `web/app.py` | Flask | Upload, process, download with progress |

## Key Technical Decisions

1. **OCR Engine Selection**
   - EasyOCR as default for balance of accuracy and ease of installation
   - PaddleOCR recommended for production (faster, more accurate)
   - Tesseract for offline/CPU-only environments

2. **Deduplication Algorithm**
   - Levenshtein ratio with 0.85 threshold
   - Sliding window of 3 frames for comparison
   - Line-by-line detection for granular deduplication

3. **Frame Sampling**
   - Default 2 FPS (sufficient for scrolling text)
   - `skip_similar=False` for scrolling content (OCR dedup handles it better)
   - Histogram-based similarity available for static content

4. **Transcript Format**
   - Optimized for Microsoft Teams format
   - Pattern: Speaker Name → Timestamp → Message
   - Consecutive messages from same speaker merged within 60 seconds

5. **Web Processing**
   - Job-based async processing
   - In-memory job storage
   - Progress phases: Initialize (0-5%), Analyze (5%), Extract (5-15%), OCR (15-85%), Dedupe (85-90%), Parse (90-95%), Output (95-100%)

## Configuration Defaults

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
```

## Performance Characteristics

| Operation | Speed | Notes |
|-----------|-------|-------|
| Frame extraction | ~100 fps | Generator-based, memory efficient |
| EasyOCR | 0.5-1 sec/frame | With GPU |
| PaddleOCR | 0.2-0.5 sec/frame | Fastest with GPU |
| Tesseract | 0.5-1 sec/frame | CPU only |
| Parallel CPU | 2-4x speedup | Uses up to 4 workers |
| Deduplication | Near-instant | In-memory |
| 1-hour video | 10-30 min | Depending on engine |

## Dependencies

### Required
- opencv-python-headless >= 4.8.0
- easyocr >= 1.7.0
- rapidfuzz >= 3.0.0
- python-Levenshtein >= 0.21.0
- click >= 8.1.0
- rich >= 13.0.0
- pyyaml >= 6.0
- Pillow >= 10.0.0
- numpy >= 1.24.0

### Optional
- paddleocr >= 2.7.0 (PaddleOCR engine)
- pytesseract >= 0.3.10 (Tesseract engine)
- flask >= 3.0.0, flask-cors >= 4.0.0 (Web interface)

## Known Limitations

- PaddleOCR may have version compatibility issues on some systems
- Tesseract requires separate system installation
- Very long videos (>1 hour) may need increased memory
- GPU support requires CUDA-enabled PyTorch installation

## Testing

- Unit tests in `tests/` directory
- Test files: frame_extractor, deduplicator, transcript_parser, output_formatter, similarity
- Run with: `pytest`
- Coverage target: 80%

## Future Enhancement Ideas

1. Batch processing for multiple videos
2. Region-of-interest selection for targeted OCR
3. Support for additional transcript formats (Zoom, Google Meet)
4. Async processing improvements for web interface
5. GPU memory management for large batches
6. FFmpeg integration for broader format support
