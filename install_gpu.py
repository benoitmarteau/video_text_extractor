#!/usr/bin/env python3
"""
GPU Installation Helper Script for Video OCR Transcript Extractor.

This script installs PyTorch with CUDA support first, then installs
all OCR engines and dependencies.

Usage:
    python install_gpu.py              # Install with CUDA 12.6 (recommended)
    python install_gpu.py --cuda 118   # Install with CUDA 11.8
    python install_gpu.py --cuda 128   # Install with CUDA 12.8
    python install_gpu.py --cpu        # Install CPU-only version
"""

import argparse
import subprocess
import sys


def run_command(cmd: list[str], description: str) -> bool:
    """Run a command and return success status."""
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"{'='*60}")
    print(f"Running: {' '.join(cmd)}\n")

    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"\n❌ Failed: {description}")
        return False

    print(f"\n✓ Success: {description}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Install Video OCR Transcript Extractor with GPU support"
    )
    parser.add_argument(
        "--cuda",
        choices=["118", "126", "128"],
        default="126",
        help="CUDA version (default: 126 for CUDA 12.6)"
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Install CPU-only version (no CUDA)"
    )
    parser.add_argument(
        "--no-dev",
        action="store_true",
        help="Skip dev dependencies (pytest, black, etc.)"
    )

    args = parser.parse_args()

    print("\n" + "="*60)
    print("  Video OCR Transcript Extractor - GPU Installation")
    print("="*60)

    # Step 1: Install PyTorch
    if args.cpu:
        print("\n📦 Installing PyTorch (CPU version)...")
        torch_cmd = [sys.executable, "-m", "pip", "install", "torch", "torchvision"]
    else:
        cuda_index = f"https://download.pytorch.org/whl/cu{args.cuda}"
        print(f"\n🚀 Installing PyTorch with CUDA {args.cuda}...")
        torch_cmd = [
            sys.executable, "-m", "pip", "install",
            "torch", "torchvision",
            "--index-url", cuda_index
        ]

    if not run_command(torch_cmd, "Install PyTorch"):
        print("\n❌ PyTorch installation failed. Please check your Python version and try again.")
        sys.exit(1)

    # Step 2: Install the package with all dependencies
    if args.no_dev:
        extras = "all-ocr-web"
    else:
        extras = "all"

    print(f"\n📦 Installing video-ocr-transcript-extractor[{extras}]...")
    package_cmd = [sys.executable, "-m", "pip", "install", "-e", f".[{extras}]"]

    if not run_command(package_cmd, f"Install package with [{extras}]"):
        print("\n❌ Package installation failed.")
        sys.exit(1)

    # Step 3: Verify installation
    print("\n" + "="*60)
    print("  Verifying Installation")
    print("="*60)

    verify_cmd = [
        sys.executable, "-c",
        "import torch; "
        "print(f'PyTorch version: {torch.__version__}'); "
        "print(f'CUDA available: {torch.cuda.is_available()}'); "
        "print(f'CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}'); "
        "from video_ocr.core.ocr_engine import OCREngine; "
        "print(f'Available OCR engines: {OCREngine.available_engines()}')"
    ]

    subprocess.run(verify_cmd)

    print("\n" + "="*60)
    print("  ✅ Installation Complete!")
    print("="*60)
    print("\nTo start the web server:")
    print("  python -m video_ocr.web.app")
    print("\nOr use the CLI:")
    print("  video-ocr extract <video.mp4>")
    print()


if __name__ == "__main__":
    main()
