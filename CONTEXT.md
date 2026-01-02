# Project Context Document

This document tracks the development status and decisions for the Video OCR Transcript Extractor project.

## Current Status

- **Phase**: Complete - All core components implemented
- **Last Updated**: 2025-01-15
- **Version**: 1.0.0

## Components Implemented

### Core Components

| Component | File | Status | Description |
|-----------|------|--------|-------------|
| Frame Extractor | `core/frame_extractor.py` | Complete | Extracts frames from video with smart sampling |
| OCR Engine | `core/ocr_engine.py` | Complete | Unified interface for multiple OCR backends |
| Deduplicator | `core/deduplicator.py` | Complete | Removes duplicate text from scrolling content |
| Transcript Parser | `core/transcript_parser.py` | Complete | Parses Teams transcript format |
| Output Formatter | `core/output_formatter.py` | Complete | Generates JSON, Markdown, Text, SRT output |

### OCR Engines

| Engine | File | Status | Notes |
|--------|------|--------|-------|
| EasyOCR | `engines/easyocr_engine.py` | Complete | Default engine, good accuracy |
| PaddleOCR | `engines/paddleocr_engine.py` | Complete | Best accuracy, requires extra install |
| Tesseract | `engines/tesseract_engine.py` | Complete | Requires system Tesseract |

### Utilities

| Utility | File | Status | Description |
|---------|------|--------|-------------|
| Logging | `utils/logging_config.py` | Complete | Rich console logging |
| Similarity | `utils/similarity.py` | Complete | Fuzzy string matching |
| Image Utils | `utils/image_utils.py` | Complete | Frame preprocessing |

### Interfaces

| Interface | File | Status | Description |
|-----------|------|--------|-------------|
| CLI | `cli.py` | Complete | Click-based CLI with rich output |
| Web | `web/app.py` | Complete | Flask-based web interface |

## Decisions Made

1. **OCR Engine Selection**: EasyOCR as default for balance of accuracy and ease of use. PaddleOCR available for production with best accuracy.

2. **Deduplication Algorithm**: Levenshtein ratio with sliding window comparison. Threshold of 0.85 provides good balance.

3. **Frame Sampling**: Default 2 FPS with histogram-based similarity detection to skip identical frames.

4. **Transcript Format**: Primarily designed for Microsoft Teams format with speaker names, timestamps, and messages.

5. **Output Formats**: JSON as primary structured format, Markdown for human readability, SRT for subtitle compatibility.

## Architecture Decisions

1. **Generator Pattern**: Frame extraction uses generators for memory efficiency with large videos.

2. **Lazy Initialization**: OCR engines initialize lazily on first use to avoid loading models unnecessarily.

3. **Abstract Base Class**: BaseOCREngine provides common interface for all OCR implementations.

4. **Dataclasses**: Used throughout for clean data structures with type hints.

## Known Issues

- [ ] PaddleOCR may have version compatibility issues on some systems
- [ ] Tesseract requires separate system installation
- [ ] Very long videos (>1 hour) may need increased memory

## Performance Notes

- Frame extraction: ~100 frames/second
- EasyOCR: ~0.5-1 second per frame (with GPU)
- PaddleOCR: ~0.2-0.5 second per frame (with GPU)
- Tesseract: ~0.5-1 second per frame (CPU only)

## Dependencies

### Required
- opencv-python-headless >= 4.8.0
- easyocr >= 1.7.0
- rapidfuzz >= 3.0.0
- click >= 8.1.0
- rich >= 13.0.0
- pyyaml >= 6.0

### Optional
- paddleocr >= 2.7.0 (for PaddleOCR engine)
- pytesseract >= 0.3.10 (for Tesseract engine)
- flask >= 3.0.0 (for web interface)

## Testing

- Unit tests in `tests/` directory
- Run with: `pytest`
- Coverage target: 80%

## Next Steps (Future Enhancements)

1. Add support for more video formats via FFmpeg
2. Implement batch processing for multiple videos
3. Add confidence-based filtering options
4. Support custom transcript formats beyond Teams
5. Add GPU memory management for large batches
6. Implement async processing for web interface
7. Add support for region-of-interest selection
