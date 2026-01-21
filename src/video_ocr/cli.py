"""Command-line interface for Video OCR Transcript Extractor."""

import sys
from pathlib import Path
from typing import List, Optional

import click
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from video_ocr import __version__
from video_ocr.core.deduplicator import TextDeduplicator
from video_ocr.core.frame_extractor import FrameExtractor
from video_ocr.core.ocr_engine import OCREngine
from video_ocr.core.output_formatter import OutputFormatter, create_metadata
from video_ocr.core.transcript_parser import TranscriptParser
from video_ocr.utils.logging_config import setup_logging

console = Console()


def print_banner():
    """Print application banner."""
    console.print(
        f"[bold blue]Video OCR Transcript Extractor[/bold blue] v{__version__}",
        highlight=False,
    )
    console.print()


@click.group()
@click.version_option(version=__version__)
def main():
    """Video OCR Transcript Extractor - Extract text from screen recordings."""
    pass


@main.command()
@click.argument("video_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "-o",
    "--output",
    type=click.Path(path_type=Path),
    help="Output file or directory path",
)
@click.option(
    "--engine",
    type=click.Choice(["easyocr", "tesseract", "surya", "doctr", "trocr"]),
    default="easyocr",
    help="OCR engine to use (surya/trocr require GPU)",
)
@click.option(
    "--fps",
    type=float,
    default=2.0,
    help="Frames per second to extract (default: 2.0)",
)
@click.option(
    "--similarity-threshold",
    type=float,
    default=0.85,
    help="Similarity threshold for deduplication (0-1, default: 0.85)",
)
@click.option(
    "--confidence-threshold",
    type=float,
    default=0.5,
    help="Minimum OCR confidence threshold (0-1, default: 0.5)",
)
@click.option(
    "--output-format",
    type=click.Choice(["json", "markdown", "text", "all"]),
    default="all",
    help="Output format (default: all)",
)
@click.option(
    "--language",
    "-l",
    multiple=True,
    default=["en"],
    help="Languages to detect (can be specified multiple times)",
)
@click.option(
    "--gpu/--no-gpu",
    default=True,
    help="Use GPU acceleration if available",
)
@click.option(
    "--skip-similar/--no-skip-similar",
    default=False,
    help="Skip visually similar frames (default: disabled for scrolling content)",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose output",
)
def extract(
    video_path: Path,
    output: Optional[Path],
    engine: str,
    fps: float,
    similarity_threshold: float,
    confidence_threshold: float,
    output_format: str,
    language: tuple,
    gpu: bool,
    skip_similar: bool,
    verbose: bool,
):
    """Extract transcript from a video file.

    VIDEO_PATH is the path to the video file to process.
    """
    print_banner()

    # Setup logging
    log_level = "DEBUG" if verbose else "INFO"
    setup_logging(level=log_level)

    # Validate video
    frame_extractor = FrameExtractor(fps=fps, skip_similar_threshold=0.95)
    is_valid, error = frame_extractor.validate_video(video_path)

    if not is_valid:
        console.print(f"[red]Error:[/red] {error}")
        sys.exit(1)

    # Get video info
    video_info = frame_extractor.get_video_info(video_path)

    # Display video info
    info_table = Table(title="Video Information", show_header=False)
    info_table.add_column("Property", style="cyan")
    info_table.add_column("Value", style="green")
    info_table.add_row("File", video_path.name)
    info_table.add_row("Resolution", f"{video_info['width']}x{video_info['height']}")
    info_table.add_row("Duration", f"{video_info['duration_seconds']:.1f}s")
    info_table.add_row("FPS", f"{video_info['fps']:.1f}")
    info_table.add_row("Total Frames", str(video_info["frame_count"]))
    console.print(info_table)
    console.print()

    # Check if GPU is available
    try:
        import torch
        gpu_available = torch.cuda.is_available()
        if gpu_available:
            gpu_name = torch.cuda.get_device_name(0)
        else:
            gpu_name = None
    except ImportError:
        gpu_available = False
        gpu_name = None

    # Initialize components
    deduplicator = TextDeduplicator(
        similarity_threshold=similarity_threshold,
        min_new_chars=10,
    )

    transcript_parser = TranscriptParser(merge_consecutive=True)

    output_formatter = OutputFormatter(
        include_metadata=True,
        include_statistics=True,
        include_raw_text=True,
    )

    # Initialize OCR engine
    ocr_engine = OCREngine(
        engine=engine,
        languages=list(language),
        confidence_threshold=confidence_threshold,
        gpu=gpu and gpu_available,
    )

    if gpu and gpu_available:
        console.print(f"[cyan]Using GPU-accelerated OCR ({gpu_name})[/cyan]")
    else:
        console.print("[cyan]Using CPU OCR[/cyan]")

    # Process video with progress bar
    ocr_results = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        # Extract frames
        frames_task = progress.add_task("[cyan]Extracting frames...", total=None)

        frames = list(frame_extractor.extract_frames(video_path, skip_similar=skip_similar))

        progress.update(frames_task, completed=len(frames), total=len(frames))
        console.print(f"[green]Extracted {len(frames)} frames[/green]")

        # OCR processing
        ocr_task = progress.add_task("[cyan]Running OCR...", total=len(frames))

        for frame_info in frames:
            result = ocr_engine.process_frame(
                frame_info.frame,
                frame_number=frame_info.frame_number,
                timestamp_seconds=frame_info.timestamp_seconds,
            )
            ocr_results.append(result)
            progress.advance(ocr_task)

        texts_found = sum(1 for r in ocr_results if r.has_text)
        console.print(f"[green]OCR complete: {len(ocr_results)} frames processed ({texts_found} with text)[/green]")

        # Deduplication
        dedup_task = progress.add_task("[cyan]Deduplicating text...", total=1)

        dedup_result = deduplicator.deduplicate(ocr_results)

        progress.update(dedup_task, completed=1)
        console.print(
            f"[green]Deduplication complete: {dedup_result.frames_with_new_content} "
            f"frames with unique content[/green]"
        )

        # Parse transcript
        parse_task = progress.add_task("[cyan]Parsing transcript...", total=1)

        # Build text with confidence info
        text_lines = [(t.text, t.confidence) for t in dedup_result.texts]
        parsed = transcript_parser.parse_entries_from_lines(text_lines)

        progress.update(parse_task, completed=1)
        console.print(
            f"[green]Parsed {len(parsed.entries)} transcript entries[/green]"
        )

    # Create metadata
    metadata = create_metadata(
        source_file=video_path.name,
        duration_seconds=video_info["duration_seconds"],
        frames_processed=len(frames),
        frames_with_text=dedup_result.frames_with_new_content,
        ocr_engine=engine,
        confidence_threshold=confidence_threshold,
    )

    # Generate output
    console.print()

    if output is None:
        output = video_path.parent / video_path.stem
    else:
        output = Path(output)

    # Determine if output is a directory or file
    if output_format == "all":
        # Save all formats to directory
        output_dir = output if output.suffix == "" else output.parent
        saved_files = output_formatter.save_all_formats(
            parsed,
            metadata,
            dedup_result.full_text,
            output_dir,
            base_name=video_path.stem,
        )

        console.print("[bold]Output files:[/bold]")
        for fmt, path in saved_files.items():
            console.print(f"  [cyan]{fmt}[/cyan]: {path}")
    else:
        # Save single format
        if output.suffix == "":
            ext_map = {"json": ".json", "markdown": ".md", "text": ".txt"}
            output = output.with_suffix(ext_map.get(output_format, ".txt"))

        output.parent.mkdir(parents=True, exist_ok=True)

        if output_format == "json":
            content = output_formatter.format_json(parsed, metadata, dedup_result.full_text)
        elif output_format == "markdown":
            content = output_formatter.format_markdown(parsed, metadata, dedup_result.full_text)
        else:
            content = output_formatter.format_text(parsed, metadata, dedup_result.full_text)

        output.write_text(content, encoding="utf-8")
        console.print(f"[bold]Output saved to:[/bold] {output}")

    # Print summary
    console.print()

    # Calculate word count - use raw text if no transcript entries found
    if parsed.word_count > 0:
        word_count = parsed.word_count
    else:
        word_count = len(dedup_result.full_text.split()) if dedup_result.full_text else 0

    summary_table = Table(title="Extraction Summary", show_header=False)
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="green")
    summary_table.add_row("Total Speakers", str(len(parsed.speakers)))
    summary_table.add_row("Total Messages", str(len(parsed.entries)))
    summary_table.add_row("Word Count", str(word_count))
    if parsed.word_count == 0 and word_count > 0:
        summary_table.add_row("", "[yellow](from raw OCR text)[/yellow]")
    summary_table.add_row(
        "Deduplication Ratio",
        f"{dedup_result.deduplication_ratio:.1%}",
    )
    console.print(summary_table)


@main.command()
def engines():
    """List available OCR engines."""
    print_banner()

    available = OCREngine.available_engines()

    console.print("[bold]Available OCR Engines:[/bold]")
    console.print()

    for engine_name in available:
        console.print(f"  [green]✓[/green] {engine_name}")

    # Check for optional engines
    all_engines = ["easyocr", "tesseract", "surya", "doctr", "trocr"]
    for engine_name in all_engines:
        if engine_name not in available:
            # Get engine info if available
            info = OCREngine.get_engine_info(engine_name)
            gpu_note = " (GPU required)" if info and info.requires_gpu else ""
            console.print(f"  [yellow]○[/yellow] {engine_name} (not installed){gpu_note}")


def process_single_video(
    video_path: Path,
    output_dir: Path,
    engine: str,
    fps: float,
    similarity_threshold: float,
    confidence_threshold: float,
    output_format: str,
    languages: List[str],
    gpu: bool,
    skip_similar: bool,
    ocr_engine: OCREngine,
) -> dict:
    """
    Process a single video file and save results.

    Returns dict with status and statistics.
    """
    frame_extractor = FrameExtractor(fps=fps, skip_similar_threshold=0.95)
    deduplicator = TextDeduplicator(similarity_threshold=similarity_threshold, min_new_chars=10)
    transcript_parser = TranscriptParser(merge_consecutive=True)
    output_formatter = OutputFormatter(include_metadata=True, include_statistics=True, include_raw_text=True)

    # Validate video
    is_valid, error = frame_extractor.validate_video(video_path)
    if not is_valid:
        return {"status": "error", "error": error, "video": video_path.name}

    # Get video info
    video_info = frame_extractor.get_video_info(video_path)

    # Extract frames
    frames = list(frame_extractor.extract_frames(video_path, skip_similar=skip_similar))

    # Run OCR
    ocr_results = []
    for frame_info in frames:
        result = ocr_engine.process_frame(
            frame_info.frame,
            frame_number=frame_info.frame_number,
            timestamp_seconds=frame_info.timestamp_seconds,
        )
        ocr_results.append(result)

    # Deduplicate
    dedup_result = deduplicator.deduplicate(ocr_results)

    # Parse transcript
    text_lines = [(t.text, t.confidence) for t in dedup_result.texts]
    parsed = transcript_parser.parse_entries_from_lines(text_lines)

    # Create metadata
    metadata = create_metadata(
        source_file=video_path.name,
        duration_seconds=video_info["duration_seconds"],
        frames_processed=len(frames),
        frames_with_text=dedup_result.frames_with_new_content,
        ocr_engine=engine,
        confidence_threshold=confidence_threshold,
    )

    # Save outputs
    output_dir.mkdir(parents=True, exist_ok=True)

    if output_format == "all":
        output_formatter.save_all_formats(
            parsed, metadata, dedup_result.full_text, output_dir, base_name=video_path.stem
        )
    else:
        ext_map = {"json": ".json", "markdown": ".md", "text": ".txt"}
        output_file = output_dir / f"{video_path.stem}{ext_map.get(output_format, '.txt')}"

        if output_format == "json":
            content = output_formatter.format_json(parsed, metadata, dedup_result.full_text)
        elif output_format == "markdown":
            content = output_formatter.format_markdown(parsed, metadata, dedup_result.full_text)
        else:
            content = output_formatter.format_text(parsed, metadata, dedup_result.full_text)

        output_file.write_text(content, encoding="utf-8")

    # Calculate word count
    word_count = parsed.word_count if parsed.word_count > 0 else len(dedup_result.full_text.split())

    return {
        "status": "success",
        "video": video_path.name,
        "frames": len(frames),
        "speakers": len(parsed.speakers),
        "messages": len(parsed.entries),
        "words": word_count,
        "dedup_ratio": dedup_result.deduplication_ratio,
    }


@main.command()
@click.argument("video_paths", nargs=-1, type=click.Path(exists=True, path_type=Path), required=True)
@click.option(
    "-o", "--output-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Output directory for all transcripts (default: same as each video)",
)
@click.option(
    "--engine",
    type=click.Choice(["easyocr", "tesseract", "surya", "doctr", "trocr"]),
    default="easyocr",
    help="OCR engine to use (surya/trocr require GPU)",
)
@click.option("--fps", type=float, default=2.0, help="Frames per second to extract")
@click.option("--similarity-threshold", type=float, default=0.85, help="Deduplication threshold")
@click.option("--confidence-threshold", type=float, default=0.5, help="OCR confidence threshold")
@click.option(
    "--output-format",
    type=click.Choice(["json", "markdown", "text", "all"]),
    default="all",
    help="Output format",
)
@click.option("-l", "--language", multiple=True, default=["en"], help="Languages to detect")
@click.option("--gpu/--no-gpu", default=True, help="Use GPU acceleration if available")
@click.option("--skip-similar/--no-skip-similar", default=False, help="Skip similar frames")
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose output")
def batch(
    video_paths: tuple,
    output_dir: Optional[Path],
    engine: str,
    fps: float,
    similarity_threshold: float,
    confidence_threshold: float,
    output_format: str,
    language: tuple,
    gpu: bool,
    skip_similar: bool,
    verbose: bool,
):
    """
    Process multiple videos in a queue (batch mode).

    VIDEO_PATHS can be multiple video files or glob patterns.

    Examples:
        video-ocr batch video1.mp4 video2.mp4 video3.mp4
        video-ocr batch *.mp4 -o ./transcripts/
        video-ocr batch meeting_*.mp4 --engine surya --gpu
    """
    print_banner()

    # Setup logging
    log_level = "DEBUG" if verbose else "INFO"
    setup_logging(level=log_level)

    videos = list(video_paths)
    total_videos = len(videos)

    if total_videos == 0:
        console.print("[red]Error:[/red] No video files specified")
        sys.exit(1)

    console.print(f"[bold]Batch Processing Queue:[/bold] {total_videos} videos")
    console.print()

    # List videos in queue
    queue_table = Table(title="Video Queue", show_header=True)
    queue_table.add_column("#", style="dim", width=4)
    queue_table.add_column("Video File", style="cyan")
    queue_table.add_column("Status", style="yellow")

    for i, vp in enumerate(videos, 1):
        queue_table.add_row(str(i), vp.name, "Pending")

    console.print(queue_table)
    console.print()

    # Check GPU availability
    try:
        import torch
        gpu_available = torch.cuda.is_available()
        if gpu_available:
            gpu_name = torch.cuda.get_device_name(0)
            console.print(f"[green]GPU Available:[/green] {gpu_name}")
        else:
            console.print("[yellow]GPU:[/yellow] Not available (using CPU)")
    except ImportError:
        gpu_available = False
        console.print("[yellow]GPU:[/yellow] PyTorch not installed (using CPU)")

    console.print()

    # Initialize OCR engine once (reuse for all videos)
    console.print("[cyan]Initializing OCR engine...[/cyan]")
    ocr_engine = OCREngine(
        engine=engine,
        languages=list(language),
        confidence_threshold=confidence_threshold,
        gpu=gpu and gpu_available,
    )

    # Force initialization
    _ = ocr_engine.engine
    console.print("[green]OCR engine ready[/green]")
    console.print()

    # Process videos sequentially
    results = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        overall_task = progress.add_task(
            f"[bold cyan]Processing {total_videos} videos...",
            total=total_videos,
        )

        for i, video_path in enumerate(videos, 1):
            progress.update(overall_task, description=f"[bold cyan][{i}/{total_videos}] {video_path.name}")

            # Determine output directory
            if output_dir:
                vid_output_dir = output_dir
            else:
                vid_output_dir = video_path.parent

            try:
                result = process_single_video(
                    video_path=video_path,
                    output_dir=vid_output_dir,
                    engine=engine,
                    fps=fps,
                    similarity_threshold=similarity_threshold,
                    confidence_threshold=confidence_threshold,
                    output_format=output_format,
                    languages=list(language),
                    gpu=gpu and gpu_available,
                    skip_similar=skip_similar,
                    ocr_engine=ocr_engine,
                )
                results.append(result)

                if result["status"] == "success":
                    console.print(f"  [green]✓[/green] {video_path.name} - {result['words']} words, {result['speakers']} speakers")
                else:
                    console.print(f"  [red]✗[/red] {video_path.name} - {result.get('error', 'Unknown error')}")

            except Exception as e:
                results.append({"status": "error", "video": video_path.name, "error": str(e)})
                console.print(f"  [red]✗[/red] {video_path.name} - Error: {e}")

            progress.advance(overall_task)

    # Print summary
    console.print()
    console.print("[bold]Batch Processing Complete[/bold]")
    console.print()

    success_count = sum(1 for r in results if r["status"] == "success")
    error_count = sum(1 for r in results if r["status"] == "error")

    summary_table = Table(title="Batch Summary", show_header=True)
    summary_table.add_column("Video", style="cyan")
    summary_table.add_column("Status", style="green")
    summary_table.add_column("Words", justify="right")
    summary_table.add_column("Speakers", justify="right")
    summary_table.add_column("Messages", justify="right")

    for r in results:
        if r["status"] == "success":
            summary_table.add_row(
                r["video"],
                "[green]Success[/green]",
                str(r["words"]),
                str(r["speakers"]),
                str(r["messages"]),
            )
        else:
            summary_table.add_row(
                r["video"],
                f"[red]Error: {r.get('error', 'Unknown')[:30]}[/red]",
                "-",
                "-",
                "-",
            )

    console.print(summary_table)
    console.print()
    console.print(f"[green]Successful:[/green] {success_count}/{total_videos}")

    if error_count > 0:
        console.print(f"[red]Failed:[/red] {error_count}/{total_videos}")

    if output_dir:
        console.print(f"[bold]Output directory:[/bold] {output_dir}")


@main.command()
@click.argument("video_path", type=click.Path(exists=True, path_type=Path))
def info(video_path: Path):
    """Display information about a video file."""
    print_banner()

    frame_extractor = FrameExtractor()
    is_valid, error = frame_extractor.validate_video(video_path)

    if not is_valid:
        console.print(f"[red]Error:[/red] {error}")
        sys.exit(1)

    video_info = frame_extractor.get_video_info(video_path)

    info_table = Table(title="Video Information", show_header=False)
    info_table.add_column("Property", style="cyan")
    info_table.add_column("Value", style="green")
    info_table.add_row("Path", str(video_path))
    info_table.add_row("Resolution", f"{video_info['width']}x{video_info['height']}")
    info_table.add_row("Duration", f"{video_info['duration_seconds']:.2f}s")
    info_table.add_row("Frame Rate", f"{video_info['fps']:.2f} fps")
    info_table.add_row("Total Frames", str(video_info["frame_count"]))

    # Calculate estimated frames at different extraction rates
    console.print(info_table)
    console.print()
    console.print("[bold]Estimated frames to process:[/bold]")

    for target_fps in [1.0, 2.0, 3.0, 5.0]:
        estimated = int(video_info["duration_seconds"] * target_fps)
        console.print(f"  At {target_fps} FPS: ~{estimated} frames")


if __name__ == "__main__":
    main()
