"""Flask web application for Video OCR."""

import os
import tempfile
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional

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

# Queue storage - ordered dict to maintain queue order
_queue: OrderedDict[str, dict] = OrderedDict()
_queue_processing: bool = False

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
            "processing_mode": job.get("processing_mode"),
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
        """List available OCR engines with detailed info."""
        engines_info = []
        for info in OCREngine.get_all_engines_info():
            engines_info.append({
                "name": info.name,
                "display_name": info.display_name,
                "requires_gpu": info.requires_gpu,
                "description": info.description,
                "installed": info.installed,
            })

        return jsonify({
            "engines": OCREngine.available_engines(),
            "engines_info": engines_info,
        })

    @app.route("/api/system-info")
    def system_info():
        """Get system information (GPU availability, CPU cores)."""
        try:
            import torch
            gpu_available = torch.cuda.is_available()
            gpu_name = torch.cuda.get_device_name(0) if gpu_available else None
        except ImportError:
            gpu_available = False
            gpu_name = None

        # Calculate number of workers
        import os
        cpu_count = os.cpu_count() or 2
        num_workers = min(cpu_count, 4)

        return jsonify({
            "gpu_available": gpu_available,
            "gpu_name": gpu_name,
            "cpu_count": cpu_count,
            "num_workers": num_workers,
        })

    # ==================== Queue Management Endpoints ====================

    @app.route("/api/queue/add", methods=["POST"])
    def add_to_queue():
        """Add a video to the processing queue."""
        if "video" not in request.files:
            return jsonify({"error": "No video file provided"}), 400

        video_file = request.files["video"]

        if video_file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        # Save uploaded file
        queue_id = str(uuid.uuid4())
        filename = f"{queue_id}_{video_file.filename}"
        filepath = Path(app.config["UPLOAD_FOLDER"]) / filename
        video_file.save(filepath)

        # Get processing options - log raw values for debugging
        raw_fps = request.form.get("fps")
        engine = request.form.get("engine", "easyocr")
        fps = float(raw_fps) if raw_fps else 2.0
        language = request.form.get("language", "en")

        logger.debug(f"[Queue] Raw form values: fps={raw_fps}, engine={engine}, language={language}")

        # Add to queue
        _queue[queue_id] = {
            "status": "queued",
            "position": len(_queue) + 1,
            "filename": video_file.filename,
            "filepath": str(filepath),
            "engine": engine,
            "fps": fps,
            "language": language,
            "result": None,
            "error": None,
        }

        logger.info(f"[Queue] Added {video_file.filename} to queue (position {len(_queue)}, fps={fps}, engine={engine})")

        return jsonify({
            "queue_id": queue_id,
            "position": len(_queue),
            "status": "queued",
            "filename": video_file.filename,
        })

    @app.route("/api/queue/status")
    def queue_status():
        """Get the status of the entire queue."""
        queue_items = []
        for i, (queue_id, item) in enumerate(_queue.items()):
            queue_items.append({
                "queue_id": queue_id,
                "position": i + 1,
                "filename": item["filename"],
                "status": item["status"],
                "error": item.get("error"),
            })

        return jsonify({
            "processing": _queue_processing,
            "total_items": len(_queue),
            "items": queue_items,
        })

    @app.route("/api/queue/item/<queue_id>")
    def queue_item_status(queue_id: str):
        """Get the status of a specific queue item."""
        if queue_id not in _queue:
            return jsonify({"error": "Queue item not found"}), 404

        item = _queue[queue_id]
        position = list(_queue.keys()).index(queue_id) + 1

        response = {
            "queue_id": queue_id,
            "position": position,
            "filename": item["filename"],
            "status": item["status"],
            "error": item.get("error"),
        }

        # Include job progress if processing
        if item["status"] == "processing" and item.get("job_id"):
            job = _jobs.get(item["job_id"], {})
            response["progress"] = job.get("progress", 0)
            response["progress_message"] = job.get("progress_message", "")
            response["current_frame"] = job.get("current_frame", 0)
            response["total_frames"] = job.get("total_frames", 0)

        return jsonify(response)

    @app.route("/api/queue/remove/<queue_id>", methods=["DELETE"])
    def remove_from_queue(queue_id: str):
        """Remove an item from the queue (only if not processing)."""
        if queue_id not in _queue:
            return jsonify({"error": "Queue item not found"}), 404

        item = _queue[queue_id]
        if item["status"] == "processing":
            return jsonify({"error": "Cannot remove item that is currently processing"}), 400

        # Clean up file
        try:
            filepath = Path(item["filepath"])
            if filepath.exists():
                filepath.unlink()
        except Exception as e:
            logger.warning(f"[Queue] Failed to delete file for {queue_id}: {e}")

        del _queue[queue_id]
        logger.info(f"[Queue] Removed {item['filename']} from queue")

        return jsonify({"status": "removed", "queue_id": queue_id})

    @app.route("/api/queue/clear", methods=["DELETE"])
    def clear_queue():
        """Clear all queued (not processing) items from the queue."""
        global _queue
        removed = 0
        to_remove = []

        for queue_id, item in _queue.items():
            if item["status"] not in ["processing"]:
                to_remove.append(queue_id)

        for queue_id in to_remove:
            item = _queue[queue_id]
            try:
                filepath = Path(item["filepath"])
                if filepath.exists():
                    filepath.unlink()
            except Exception:
                pass
            del _queue[queue_id]
            removed += 1

        logger.info(f"[Queue] Cleared {removed} items from queue")

        return jsonify({"status": "cleared", "removed_count": removed})

    @app.route("/api/queue/process", methods=["POST"])
    def process_queue():
        """Start processing the queue sequentially."""
        global _queue_processing

        if _queue_processing:
            return jsonify({"error": "Queue is already being processed"}), 400

        if not _queue:
            return jsonify({"error": "Queue is empty"}), 400

        _queue_processing = True
        processed = 0
        errors = 0

        try:
            for queue_id, item in list(_queue.items()):
                if item["status"] != "queued":
                    continue

                item["status"] = "processing"
                logger.info(f"[Queue] Processing {item['filename']}...")
                logger.info(f"[Queue] Settings: engine={item['engine']}, fps={item['fps']}, language={item['language']}")

                # Create a job for this queue item
                job_id = str(uuid.uuid4())
                item["job_id"] = job_id

                _jobs[job_id] = {
                    "status": "processing",
                    "progress": 0,
                    "progress_message": "Starting...",
                    "filepath": item["filepath"],
                    "engine": item["engine"],
                    "fps": item["fps"],
                    "language": item["language"],
                    "result": None,
                    "error": None,
                    "total_frames": 0,
                    "current_frame": 0,
                }
                logger.debug(f"[Queue] Job {job_id[:8]} created with fps={_jobs[job_id]['fps']}")

                try:
                    result = _process_video_job(job_id, _jobs[job_id])
                    _jobs[job_id]["result"] = result
                    _jobs[job_id]["status"] = "complete"
                    item["status"] = "complete"
                    item["result"] = result

                    # Auto-save result
                    _auto_save_result(item, result)
                    processed += 1

                    logger.info(f"[Queue] Completed {item['filename']}")

                except Exception as e:
                    _jobs[job_id]["status"] = "error"
                    _jobs[job_id]["error"] = str(e)
                    item["status"] = "error"
                    item["error"] = str(e)
                    errors += 1

                    logger.error(f"[Queue] Error processing {item['filename']}: {e}")

                finally:
                    # Cleanup between queue items to prevent memory issues
                    _cleanup_gpu_memory()

        finally:
            _queue_processing = False

        return jsonify({
            "status": "complete",
            "processed": processed,
            "errors": errors,
        })

    @app.route("/api/queue/results")
    def queue_results():
        """Get all completed results from the queue."""
        results = []
        for queue_id, item in _queue.items():
            if item["status"] == "complete" and item.get("result"):
                results.append({
                    "queue_id": queue_id,
                    "filename": item["filename"],
                    "result": item["result"],
                })

        return jsonify({"results": results})

    @app.route("/api/queue/download/<queue_id>/<format>")
    def download_queue_result(queue_id: str, format: str):
        """Download a queue item result in specified format."""
        if queue_id not in _queue:
            return jsonify({"error": "Queue item not found"}), 404

        item = _queue[queue_id]

        if item["status"] != "complete":
            return jsonify({"error": "Item not complete"}), 400

        result = item.get("result")
        if not result:
            return jsonify({"error": "No result available"}), 400

        output_dir = Path(app.config["UPLOAD_FOLDER"])
        base_filename = Path(item["filename"]).stem

        if format == "json":
            filepath = output_dir / f"{base_filename}_transcript.json"
            filepath.write_text(result.get("json", "{}"), encoding="utf-8")
            mimetype = "application/json"
        elif format == "markdown":
            filepath = output_dir / f"{base_filename}_transcript.md"
            filepath.write_text(result.get("markdown", ""), encoding="utf-8")
            mimetype = "text/markdown"
        elif format == "text":
            filepath = output_dir / f"{base_filename}_transcript.txt"
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

    @app.route("/api/queue/download-all/<format>")
    def download_all_queue_results(format: str):
        """Download all completed queue results combined into a single file."""
        # Get deduplication option from query parameter
        deduplicate = request.args.get("deduplicate", "false").lower() == "true"

        # Collect all completed results
        completed_items = []
        for queue_id, item in _queue.items():
            if item["status"] == "complete" and item.get("result"):
                completed_items.append({
                    "filename": item["filename"],
                    "result": item["result"],
                })

        if not completed_items:
            return jsonify({"error": "No completed results to download"}), 400

        # Combine results
        combined_content = _combine_queue_results(completed_items, format, deduplicate)

        if combined_content is None:
            return jsonify({"error": "Invalid format"}), 400

        # Determine file extension and mimetype
        ext_map = {"json": ".json", "markdown": ".md", "text": ".txt"}
        mime_map = {"json": "application/json", "markdown": "text/markdown", "text": "text/plain"}

        if format not in ext_map:
            return jsonify({"error": "Invalid format"}), 400

        # Save combined file
        output_dir = Path(app.config["UPLOAD_FOLDER"])
        suffix = "_deduplicated" if deduplicate else ""
        filepath = output_dir / f"combined_transcripts{suffix}{ext_map[format]}"
        filepath.write_text(combined_content, encoding="utf-8")

        return send_file(
            filepath,
            mimetype=mime_map[format],
            as_attachment=True,
            download_name=filepath.name,
        )

    return app


def _combine_queue_results(items: List[dict], format: str, deduplicate: bool = False) -> Optional[str]:
    """
    Combine multiple queue results into a single output.

    Args:
        items: List of dicts with 'filename' and 'result' keys
        format: Output format ('json', 'markdown', 'text')
        deduplicate: Whether to remove duplicate content across videos

    Returns:
        Combined content string, or None if invalid format
    """
    if not items:
        return ""

    # Collect all raw text for deduplication
    all_raw_texts = []
    for item in items:
        raw_text = item["result"].get("raw_text", "")
        if raw_text:
            all_raw_texts.append(raw_text)

    # Deduplicate if requested
    if deduplicate and all_raw_texts:
        combined_raw = _deduplicate_across_videos(all_raw_texts)
    else:
        combined_raw = "\n\n".join(all_raw_texts)

    if format == "json":
        import json
        combined = {
            "metadata": {
                "combined_from": [item["filename"] for item in items],
                "total_videos": len(items),
                "deduplicated": deduplicate,
            },
            "videos": [],
            "combined_raw_text": combined_raw,
            "statistics": {
                "total_words": len(combined_raw.split()) if combined_raw else 0,
            },
        }

        for item in items:
            result = item["result"]
            combined["videos"].append({
                "filename": item["filename"],
                "transcript": result.get("transcript", []),
                "speakers": result.get("speakers", []),
                "statistics": result.get("statistics", {}),
            })

        return json.dumps(combined, indent=2, ensure_ascii=False)

    elif format == "markdown":
        lines = []
        lines.append("# Combined Meeting Transcripts")
        lines.append("")
        lines.append(f"**Videos Combined**: {len(items)}")
        lines.append(f"**Deduplication**: {'Enabled' if deduplicate else 'Disabled'}")
        lines.append("")

        # List source files
        lines.append("## Source Files")
        lines.append("")
        for i, item in enumerate(items, 1):
            lines.append(f"{i}. {item['filename']}")
        lines.append("")
        lines.append("---")
        lines.append("")

        if deduplicate:
            # Show combined deduplicated text
            lines.append("## Combined Transcript (Deduplicated)")
            lines.append("")
            if combined_raw.strip():
                lines.append(combined_raw.strip())
            else:
                lines.append("*No text content found.*")
            lines.append("")
        else:
            # Show each video's content separately
            for item in items:
                result = item["result"]
                lines.append(f"## {item['filename']}")
                lines.append("")

                transcript = result.get("transcript", [])
                raw_text = result.get("raw_text", "")

                if transcript:
                    for entry in transcript:
                        if entry.get("speaker"):
                            lines.append(f"**{entry['speaker']}** ({entry.get('timestamp', '')})")
                        else:
                            lines.append(f"*({entry.get('timestamp', '')})*")
                        lines.append(entry.get("text", ""))
                        lines.append("")
                elif raw_text:
                    lines.append("*No structured transcript format. Raw OCR text:*")
                    lines.append("")
                    lines.append(raw_text.strip())
                    lines.append("")
                else:
                    lines.append("*No text content found.*")
                    lines.append("")

                lines.append("---")
                lines.append("")

        lines.append("*Generated by Video OCR Transcript Extractor*")
        return "\n".join(lines)

    elif format == "text":
        lines = []
        lines.append("=" * 60)
        lines.append("COMBINED MEETING TRANSCRIPTS")
        lines.append("=" * 60)
        lines.append(f"Videos: {len(items)}")
        lines.append(f"Deduplication: {'Enabled' if deduplicate else 'Disabled'}")
        lines.append("")

        if deduplicate:
            lines.append("-" * 60)
            lines.append("COMBINED TRANSCRIPT (DEDUPLICATED)")
            lines.append("-" * 60)
            lines.append("")
            if combined_raw.strip():
                lines.append(combined_raw.strip())
            else:
                lines.append("(No text content found)")
            lines.append("")
        else:
            for item in items:
                result = item["result"]
                lines.append("-" * 60)
                lines.append(f"FILE: {item['filename']}")
                lines.append("-" * 60)
                lines.append("")

                transcript = result.get("transcript", [])
                raw_text = result.get("raw_text", "")

                if transcript:
                    for entry in transcript:
                        if entry.get("speaker"):
                            lines.append(f"[{entry.get('timestamp', '')}] {entry['speaker']}:")
                        else:
                            lines.append(f"[{entry.get('timestamp', '')}]")
                        lines.append(f"  {entry.get('text', '')}")
                        lines.append("")
                elif raw_text:
                    lines.append("(No structured transcript format. Raw OCR text below)")
                    lines.append("")
                    lines.append(raw_text.strip())
                    lines.append("")
                else:
                    lines.append("(No text content found)")
                    lines.append("")

        return "\n".join(lines)

    return None


def _deduplicate_across_videos(texts: List[str]) -> str:
    """
    Deduplicate text content across multiple videos.

    This handles the case where multiple video recordings have overlapping content
    (e.g., recording the same meeting from different time windows).

    Args:
        texts: List of raw text from each video

    Returns:
        Deduplicated combined text
    """
    from video_ocr.utils.similarity import levenshtein_ratio, normalize_text

    if not texts:
        return ""

    if len(texts) == 1:
        return texts[0]

    # Split each text into lines
    all_lines = []
    for text in texts:
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        all_lines.extend(lines)

    if not all_lines:
        return ""

    # Deduplicate lines using similarity matching
    unique_lines = []
    seen_normalized = set()
    similarity_threshold = 0.85

    for line in all_lines:
        normalized = normalize_text(line)

        # Skip if too short
        if len(normalized) < 5:
            continue

        # Check if we've seen a similar line
        is_duplicate = False
        for seen in seen_normalized:
            if levenshtein_ratio(normalized, seen, normalize=False) > similarity_threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            unique_lines.append(line)
            seen_normalized.add(normalized)

    return "\n".join(unique_lines)


def _cleanup_gpu_memory() -> None:
    """Clean up GPU memory between queue items."""
    import gc
    gc.collect()

    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.debug("[Queue] GPU memory cache cleared")
    except ImportError:
        pass
    except Exception as e:
        logger.debug(f"[Queue] GPU cleanup warning: {e}")


def _auto_save_result(item: dict, result: dict) -> None:
    """Auto-save queue item result to the upload folder."""
    try:
        output_dir = Path(item["filepath"]).parent
        base_filename = Path(item["filename"]).stem

        # Save all formats
        json_path = output_dir / f"{base_filename}_transcript.json"
        json_path.write_text(result.get("json", "{}"), encoding="utf-8")

        md_path = output_dir / f"{base_filename}_transcript.md"
        md_path.write_text(result.get("markdown", ""), encoding="utf-8")

        txt_path = output_dir / f"{base_filename}_transcript.txt"
        txt_path.write_text(result.get("text", ""), encoding="utf-8")

        logger.info(f"[Queue] Auto-saved results for {item['filename']}")

    except Exception as e:
        logger.warning(f"[Queue] Failed to auto-save results for {item['filename']}: {e}")


def _process_video_job(job_id: str, job: dict) -> dict:
    """Process a video extraction job."""
    job_short = job_id[:8]
    filepath = Path(job["filepath"])

    # Log all job parameters for debugging
    logger.info(f"[Job {job_short}] Starting processing...")
    logger.info(f"[Job {job_short}] Job parameters: fps={job.get('fps')}, engine={job.get('engine')}, language={job.get('language')}")

    # Phase 1: Initialize components (0-5%)
    job["progress"] = 2
    job["progress_message"] = "Initializing OCR engine..."

    frame_extractor = FrameExtractor(fps=job["fps"])
    deduplicator = TextDeduplicator()
    transcript_parser = TranscriptParser()
    output_formatter = OutputFormatter()

    # Check if GPU is available
    try:
        import torch
        gpu_available = torch.cuda.is_available()
    except ImportError:
        gpu_available = False

    # Initialize single OCR engine (simpler, more reliable)
    ocr_engine = OCREngine(
        engine=job["engine"],
        languages=[job["language"]],
        gpu=gpu_available,
    )

    if gpu_available:
        logger.info(f"[Job {job_short}] Initializing {job['engine']} OCR engine (GPU accelerated)")
        job["processing_mode"] = {
            "gpu": True,
            "parallel": False,
            "workers": 1,
        }
    else:
        logger.info(f"[Job {job_short}] Initializing {job['engine']} OCR engine (CPU mode)")
        job["processing_mode"] = {
            "gpu": False,
            "parallel": False,
            "workers": 1,
        }

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

    # Sequential processing (GPU or CPU)
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

    # Calculate word count from either transcript entries or raw text
    # This fixes the "0 words" issue when transcript format isn't detected
    if parsed.word_count > 0:
        word_count = parsed.word_count
    else:
        # Fallback: count words in raw OCR text
        word_count = len(dedup_result.full_text.split()) if dedup_result.full_text else 0

    logger.info(f"[Job {job_short}] Processing complete! Words: {word_count}, Speakers: {len(parsed.speakers)}, Entries: {len(parsed.entries)}")

    # Generate outputs - pass raw_text to all formatters for fallback
    return {
        "json": output_formatter.format_json(parsed, metadata, dedup_result.full_text),
        "markdown": output_formatter.format_markdown(parsed, metadata, dedup_result.full_text),
        "text": output_formatter.format_text(parsed, metadata, dedup_result.full_text),
        "transcript": [e.to_dict() for e in parsed.entries],
        "speakers": parsed.speakers,
        "raw_text": dedup_result.full_text,  # Include raw OCR text as fallback
        "statistics": {
            "total_speakers": len(parsed.speakers),
            "total_messages": len(parsed.entries),
            "word_count": word_count,  # Use calculated word count
            "raw_word_count": len(dedup_result.full_text.split()) if dedup_result.full_text else 0,
            "frames_processed": len(frames),
            "frames_with_text": texts_found,
            "deduplication_ratio": dedup_result.deduplication_ratio,
        },
    }


def run_server(host: str = "0.0.0.0", port: int = 8080, debug: bool = False):
    """Run the web server."""
    app = create_app()
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    run_server(debug=True)
