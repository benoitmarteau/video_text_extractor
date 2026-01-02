"""Flask web application for Video OCR."""

import tempfile
import uuid
from pathlib import Path
from typing import Dict, Optional

try:
    from flask import Flask, jsonify, render_template, request, send_file
    from flask_cors import CORS

    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False

from video_ocr.core.deduplicator import TextDeduplicator
from video_ocr.core.frame_extractor import FrameExtractor
from video_ocr.core.ocr_engine import OCREngine
from video_ocr.core.output_formatter import OutputFormatter, create_metadata
from video_ocr.core.transcript_parser import TranscriptParser
from video_ocr.utils.logging_config import setup_logging, get_logger

# Processing jobs storage
_jobs: Dict[str, dict] = {}

# Logger
logger = None


def create_app(config: Optional[dict] = None) -> "Flask":
    """Create and configure the Flask application."""
    if not FLASK_AVAILABLE:
        raise ImportError("Flask is not installed. Install with: pip install flask flask-cors")

    app = Flask(
        __name__,
        template_folder="templates",
        static_folder="static",
    )

    CORS(app)

    # Configuration
    app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500MB max
    app.config["UPLOAD_FOLDER"] = tempfile.mkdtemp()

    if config:
        app.config.update(config)

    global logger
    setup_logging()
    logger = get_logger()

    @app.route("/")
    def index():
        """Render the main page."""
        return render_template("index.html")

    @app.route("/api/upload", methods=["POST"])
    def upload_video():
        """Handle video upload and start processing."""
        if "video" not in request.files:
            return jsonify({"error": "No video file provided"}), 400

        video_file = request.files["video"]

        if video_file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        # Save uploaded file
        job_id = str(uuid.uuid4())
        filename = f"{job_id}_{video_file.filename}"
        filepath = Path(app.config["UPLOAD_FOLDER"]) / filename
        video_file.save(filepath)

        # Get processing options
        engine = request.form.get("engine", "easyocr")
        fps = float(request.form.get("fps", 2.0))
        language = request.form.get("language", "en")

        # Initialize job
        _jobs[job_id] = {
            "status": "pending",
            "progress": 0,
            "progress_message": "Waiting to start...",
            "filepath": str(filepath),
            "engine": engine,
            "fps": fps,
            "language": language,
            "result": None,
            "error": None,
            "total_frames": 0,
            "current_frame": 0,
        }

        logger.info(f"[Job {job_id[:8]}] Video uploaded: {video_file.filename} (engine={engine}, fps={fps}, lang={language})")

        return jsonify({"job_id": job_id, "status": "pending"})

    @app.route("/api/process/<job_id>", methods=["POST"])
    def process_video(job_id: str):
        """Process a video job."""
        if job_id not in _jobs:
            return jsonify({"error": "Job not found"}), 404

        job = _jobs[job_id]

        if job["status"] not in ["pending", "error"]:
            return jsonify({"error": "Job already processing or complete"}), 400

        job["status"] = "processing"

        try:
            result = _process_video_job(job_id, job)
            job["result"] = result
            job["status"] = "complete"
            job["progress"] = 100

            return jsonify({
                "status": "complete",
                "result": result,
            })

        except Exception as e:
            job["status"] = "error"
            job["error"] = str(e)
            return jsonify({"error": str(e)}), 500

    @app.route("/api/status/<job_id>")
    def get_status(job_id: str):
        """Get job status."""
        if job_id not in _jobs:
            return jsonify({"error": "Job not found"}), 404

        job = _jobs[job_id]

        return jsonify({
            "job_id": job_id,
            "status": job["status"],
            "progress": job["progress"],
            "progress_message": job.get("progress_message", ""),
            "total_frames": job.get("total_frames", 0),
            "current_frame": job.get("current_frame", 0),
            "error": job.get("error"),
        })

    @app.route("/api/result/<job_id>")
    def get_result(job_id: str):
        """Get job result."""
        if job_id not in _jobs:
            return jsonify({"error": "Job not found"}), 404

        job = _jobs[job_id]

        if job["status"] != "complete":
            return jsonify({"error": "Job not complete"}), 400

        return jsonify(job["result"])

    @app.route("/api/download/<job_id>/<format>")
    def download_result(job_id: str, format: str):
        """Download result in specified format."""
        if job_id not in _jobs:
            return jsonify({"error": "Job not found"}), 404

        job = _jobs[job_id]

        if job["status"] != "complete":
            return jsonify({"error": "Job not complete"}), 400

        # Generate file
        result = job["result"]

        output_dir = Path(app.config["UPLOAD_FOLDER"])
        filename = f"{job_id}_transcript"

        if format == "json":
            filepath = output_dir / f"{filename}.json"
            filepath.write_text(result.get("json", "{}"), encoding="utf-8")
            mimetype = "application/json"
        elif format == "markdown":
            filepath = output_dir / f"{filename}.md"
            filepath.write_text(result.get("markdown", ""), encoding="utf-8")
            mimetype = "text/markdown"
        elif format == "text":
            filepath = output_dir / f"{filename}.txt"
            filepath.write_text(result.get("text", ""), encoding="utf-8")
            mimetype = "text/plain"
        else:
            return jsonify({"error": "Invalid format"}), 400

        return send_file(
            filepath,
            mimetype=mimetype,
            as_attachment=True,
            download_name=filepath.name,
        )

    @app.route("/api/engines")
    def list_engines():
        """List available OCR engines."""
        return jsonify({
            "engines": OCREngine.available_engines(),
        })

    return app


def _process_video_job(job_id: str, job: dict) -> dict:
    """Process a video extraction job."""
    job_short = job_id[:8]
    filepath = Path(job["filepath"])

    logger.info(f"[Job {job_short}] Starting processing...")

    # Phase 1: Initialize components (0-5%)
    job["progress"] = 2
    job["progress_message"] = "Initializing OCR engine..."
    logger.info(f"[Job {job_short}] Initializing {job['engine']} OCR engine...")

    frame_extractor = FrameExtractor(fps=job["fps"])
    ocr_engine = OCREngine(
        engine=job["engine"],
        languages=[job["language"]],
        gpu=True,
    )
    deduplicator = TextDeduplicator()
    transcript_parser = TranscriptParser()
    output_formatter = OutputFormatter()

    # Phase 2: Get video info (5%)
    job["progress"] = 5
    job["progress_message"] = "Analyzing video..."
    video_info = frame_extractor.get_video_info(filepath)
    logger.info(f"[Job {job_short}] Video: {video_info['duration_seconds']:.1f}s, {video_info['width']}x{video_info['height']}, {video_info['fps']:.1f} FPS")

    # Phase 3: Extract frames (5-15%)
    job["progress"] = 8
    job["progress_message"] = "Extracting frames from video..."
    logger.info(f"[Job {job_short}] Extracting frames at {job['fps']} FPS...")

    # Disable skip_similar for scrolling content - OCR deduplication handles duplicates better
    frames = list(frame_extractor.extract_frames(filepath, skip_similar=False))
    total_frames = len(frames)
    job["total_frames"] = total_frames
    job["progress"] = 15
    job["progress_message"] = f"Extracted {total_frames} frames"
    logger.info(f"[Job {job_short}] Extracted {total_frames} frames from video")

    # Phase 4: OCR processing (15-85%) - This is the longest phase
    job["progress_message"] = f"Running OCR on {total_frames} frames..."
    logger.info(f"[Job {job_short}] Starting OCR processing on {total_frames} frames...")

    ocr_results = []
    texts_found = 0
    for i, frame_info in enumerate(frames):
        frame_num = i + 1
        job["current_frame"] = frame_num

        result = ocr_engine.process_frame(
            frame_info.frame,
            frame_number=frame_info.frame_number,
            timestamp_seconds=frame_info.timestamp_seconds,
        )
        ocr_results.append(result)

        if result.has_text:
            texts_found += 1

        # Update progress (15-85% range = 70% of progress bar)
        progress_pct = 15 + int((frame_num / total_frames) * 70)
        job["progress"] = progress_pct
        job["progress_message"] = f"OCR: Frame {frame_num}/{total_frames} ({texts_found} with text)"

        # Log every 10 frames or on last frame
        if frame_num % 10 == 0 or frame_num == total_frames:
            logger.info(f"[Job {job_short}] OCR progress: {frame_num}/{total_frames} frames ({texts_found} with text)")

    logger.info(f"[Job {job_short}] OCR complete: {texts_found}/{total_frames} frames contained text")

    # Phase 5: Deduplication (85-90%)
    job["progress"] = 87
    job["progress_message"] = "Removing duplicate text..."
    logger.info(f"[Job {job_short}] Deduplicating extracted text...")

    dedup_result = deduplicator.deduplicate(ocr_results)
    logger.info(f"[Job {job_short}] Deduplication: {dedup_result.frames_with_new_content} unique content blocks (ratio: {dedup_result.deduplication_ratio:.1%})")

    # Phase 6: Parse transcript (90-95%)
    job["progress"] = 92
    job["progress_message"] = "Parsing transcript structure..."
    logger.info(f"[Job {job_short}] Parsing transcript structure...")

    text_lines = [(t.text, t.confidence) for t in dedup_result.texts]
    parsed = transcript_parser.parse_entries_from_lines(text_lines)
    logger.info(f"[Job {job_short}] Found {len(parsed.entries)} transcript entries from {len(parsed.speakers)} speakers")

    # Phase 7: Generate outputs (95-100%)
    job["progress"] = 96
    job["progress_message"] = "Generating output formats..."
    logger.info(f"[Job {job_short}] Generating output formats...")

    metadata = create_metadata(
        source_file=filepath.name,
        duration_seconds=video_info["duration_seconds"],
        frames_processed=len(frames),
        frames_with_text=dedup_result.frames_with_new_content,
        ocr_engine=job["engine"],
        confidence_threshold=0.5,
    )

    job["progress"] = 100
    job["progress_message"] = "Complete!"
    logger.info(f"[Job {job_short}] Processing complete! Words: {parsed.word_count}, Speakers: {len(parsed.speakers)}")

    # Generate outputs
    return {
        "json": output_formatter.format_json(parsed, metadata, dedup_result.full_text),
        "markdown": output_formatter.format_markdown(parsed, metadata),
        "text": output_formatter.format_text(parsed, metadata),
        "transcript": [e.to_dict() for e in parsed.entries],
        "speakers": parsed.speakers,
        "raw_text": dedup_result.full_text,  # Include raw OCR text as fallback
        "statistics": {
            "total_speakers": len(parsed.speakers),
            "total_messages": len(parsed.entries),
            "word_count": parsed.word_count,
            "frames_processed": len(frames),
            "deduplication_ratio": dedup_result.deduplication_ratio,
        },
    }


def run_server(host: str = "0.0.0.0", port: int = 8080, debug: bool = False):
    """Run the web server."""
    app = create_app()
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    run_server(debug=True)
