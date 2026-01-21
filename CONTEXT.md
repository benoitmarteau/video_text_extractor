# Project Context Document

This document tracks the development history, architecture decisions, and technical details for the Video OCR Transcript Extractor project.

## Project Overview

**Video OCR Transcript Extractor** extracts scrolling text from screen recordings using OCR and intelligently deduplicates the content. The primary use case is extracting meeting transcripts from **Zoom** and **Microsoft Teams** recordings where the transcript/caption panel shows scrolling text with speaker names, timestamps, and messages.

## Current Status

- **Phase**: Complete - Production ready
- **Last Updated**: 2026-01-20
- **Version**: 1.0.0
- **Repository**: https://github.com/benoitmarteau/video_text_extractor

## Development History

### Initial Implementation
- Built complete processing pipeline: frame extraction, OCR, deduplication, parsing, output formatting
- Implemented multiple OCR engine backends: EasyOCR (default), Tesseract, Surya, docTR, TrOCR
- Created CLI interface with Click and web interface with Flask

### Bug Fixes
- **Frame similarity issue**: Initial implementation used aggressive frame skipping (skip_similar=True) which caused only 1 frame to be extracted from 997 total frames. Fixed by disabling skip_similar for scrolling content since the OCR deduplication handles duplicates better.
- **"No transcript entries found"**: Added raw_text fallback in results and updated web UI to show raw OCR text when no structured transcript is detected.
- **Queue file re-selection**: Fixed issue where users couldn't add videos after clearing the queue. The file input now resets after adding files, allowing the same files to be selected again.

### Enhancements
- **Progress tracking**: Added detailed logging with job ID prefix, frame-by-frame progress, and status phases
- **Web UI improvements**: Real-time percentage display, frame counter, progress phases, 500ms status polling
- **Word count fix**: Statistics now show word count from raw OCR text when no transcript format is detected
- **Stable ETA display**: Implemented exponential smoothing for consistent remaining time estimates
- **Batch processing**: Added CLI `batch` command for processing multiple videos sequentially with auto-save
- **Queue system**: Web interface now supports queueing multiple videos for batch processing
- **Simplified processing**: Removed parallel processing to reduce complexity and improve reliability
- **Zoom format support**: Added avatar initials detection (e.g., "OO", "MN") and improved speaker name parsing for Zoom transcript format
- **Download All feature**: Queue mode now supports downloading all results combined into a single file
- **Cross-video deduplication**: Option to remove duplicate content when combining transcripts from overlapping recordings
- **Raw text fallback**: Output formatters now include raw OCR text when no structured transcript entries are found
- **FPS debugging**: Added comprehensive logging to trace FPS values through the queue processing pipeline
- **GPU memory cleanup**: Added explicit CUDA cache clearing between queue items to prevent memory accumulation

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
| Deduplicator | `core/deduplicator.py` | Remove duplicate text from scrolling content |
| Transcript Parser | `core/transcript_parser.py` | Parse Teams transcript format |
| Output Formatter | `core/output_formatter.py` | Generate JSON, Markdown, Text, SRT output |

### OCR Engines

| Engine | File | GPU Required | Notes |
|--------|------|--------------|-------|
| EasyOCR | `engines/easyocr_engine.py` | No (optional) | Default, good balance of speed/accuracy |
| Tesseract | `engines/tesseract_engine.py` | No | Requires system Tesseract installation |
| Surya | `engines/surya_engine.py` | Yes* | Very fast multilingual OCR, 90+ languages |
| docTR | `engines/doctr_engine.py` | No (optional) | Mindee document OCR, high accuracy |
| TrOCR | `engines/trocr_engine.py` | Yes | Microsoft Transformer OCR, best for printed/handwritten |

*Surya (v0.17+) requires Python 3.10+ and PyTorch 2.7+. Use cu126 or cu128 CUDA index.

### Utilities

| Utility | File | Description |
|---------|------|-------------|
| Logging | `utils/logging_config.py` | Rich console logging with fallback |
| Similarity | `utils/similarity.py` | Levenshtein ratio, text normalization |
| Image Utils | `utils/image_utils.py` | Frame preprocessing, similarity calc |

### Interfaces

| Interface | File | Framework | Description |
|-----------|------|-----------|-------------|
| CLI | `cli.py` | Click + Rich | Commands: extract, batch, info, engines |
| Web | `web/app.py` | Flask | Single video + queue mode with progress |

## Key Technical Decisions

1. **OCR Engine Selection**
   - EasyOCR as default for balance of accuracy and ease of installation
   - Surya recommended for production with GPU (very fast)
   - Tesseract for offline/CPU-only environments
   - TrOCR for highest accuracy on printed/handwritten text (GPU required)

2. **Deduplication Algorithm**
   - Levenshtein ratio with 0.85 threshold
   - Sliding window of 3 frames for comparison
   - Line-by-line detection for granular deduplication

3. **Frame Sampling**
   - Default 2 FPS (sufficient for scrolling text)
   - `skip_similar=False` for scrolling content (OCR dedup handles it better)
   - Histogram-based similarity available for static content

4. **Transcript Format**
   - Optimized for Microsoft Teams and Zoom formats
   - Pattern: [Avatar Initials] → Speaker Name → Timestamp → Message
   - Avatar initials detection (e.g., "OO", "MN", "BT")
   - Supports Zoom-style names with @ (e.g., "May D Wang@GT&EmoryU")
   - Consecutive messages from same speaker merged within 60 seconds
   - Falls back to raw OCR text when no structured format detected

5. **Web Processing**
   - Job-based processing with progress tracking
   - In-memory job storage
   - Queue system for batch processing with multi-file upload
   - Progress phases: Initialize (0-5%), Analyze (5%), Extract (5-15%), OCR (15-85%), Dedupe (85-90%), Parse (90-95%), Output (95-100%)
   - ETA display with exponential smoothing for stable estimates
   - **Download All**: Combine all queue results into single file
   - **Cross-video deduplication**: Remove duplicates when recordings overlap
   - GPU memory cleanup between queue items

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
| Tesseract | 0.5-1 sec/frame | CPU only |
| Surya | 0.1-0.3 sec/frame | GPU required, fastest |
| docTR | 0.3-0.5 sec/frame | With GPU |
| TrOCR | 1-2 sec/frame | GPU required, highest accuracy |
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
- pytesseract >= 0.3.10 (Tesseract engine)
- surya-ocr >= 0.4.0 (Surya engine, GPU required, Python 3.10+ for v0.17+)
- python-doctr[torch] >= 0.7.0 (docTR engine)
- transformers >= 4.35.0, torch >= 2.0.0 (TrOCR engine, GPU required)
- flask >= 3.0.0, flask-cors >= 4.0.0 (Web interface)

## GPU Installation

For GPU support, install PyTorch with CUDA **before** installing other dependencies:

```bash
# Recommended: CUDA 12.6 (compatible with PyTorch 2.7+ required by Surya)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# Alternative: CUDA 11.8 (for older systems) or cu128 (for newest GPUs)
```

**PyTorch CUDA Compatibility:**
- PyTorch 2.7+ (required for Surya v0.17+): cu118, cu126, cu128
- PyTorch 2.5-2.6: cu118, cu121, cu124

## Known Limitations

- Tesseract requires separate system installation
- Very long videos (>1 hour) may need increased memory
- GPU support requires CUDA-enabled PyTorch installation (install torch with CUDA first)
- Surya v0.17+ requires Python 3.10+ and PyTorch 2.7+ (use cu126 or cu128)

## Testing

- Unit tests in `tests/` directory
- Test files: frame_extractor, deduplicator, transcript_parser, output_formatter, similarity
- Run with: `pytest`
- Coverage target: 80%

## Future Enhancement Ideas

1. Region-of-interest selection for targeted OCR
2. Support for additional transcript formats (Zoom, Google Meet)
3. Async processing improvements for web interface (background workers)
4. GPU memory management for large batches
5. FFmpeg integration for broader format support
6. VLM-based OCR engine (olmOCR) for improved accuracy on complex layouts
