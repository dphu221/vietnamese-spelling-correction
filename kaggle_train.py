#!/usr/bin/env python3
"""Kaggle training wrapper script for Vietnamese Spelling Correction.

This script simplifies running the streaming training pipeline inside Kaggle Notebooks or Kaggle Scripts:
- Automatically detects GPU acceleration & optimal precision (FP16 on T4/P100, BF16 on L4/A100).
- Automatically resolves dataset files from Kaggle input (/kaggle/input/...) or downloads from HuggingFace.
- Directs outputs to /kaggle/working/checkpoints.

Usage in Kaggle cell:
  !python kaggle_train.py --epochs 3 --batch-size 64

Usage in Python code:
  import kaggle_train
  kaggle_train.run(epochs=3, batch_size=64)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from train_stream import main as train_stream_main

ROOT = Path(__file__).resolve().parent


def run(
    epochs: int = 3,
    batch_size: int = 64,
    bucket_size: int = 4096,
    warmup_epochs: int = 2,
    amp_dtype: str = "auto",
    output_dir: str | Path | None = None,
    max_train_batches: int = 0,
    max_validation_batches: int = 0,
    auto_download: bool = True,
    dataset_repo: str = "Sanng1112/vietnamese-spelling-synthetic-1gb",
    hf_token: str | None = None,
    prefetch_factor: int = 16,
    resume: bool = True,
    resume_checkpoint: str | Path | None = None,
) -> None:
    """Programmatic entry point for Kaggle Notebooks."""
    import os
    if output_dir is None:
        if Path("/kaggle/working").exists():
            output_dir = Path("/kaggle/working/checkpoints/full_1gb_bf16_3_epochs")
        else:
            output_dir = ROOT / "checkpoints/full_1gb_bf16_3_epochs"

    token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    if amp_dtype == "auto":
        import torch
        if torch.cuda.is_available() and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
            resolved_amp = "bf16"
        else:
            resolved_amp = "fp16"
    else:
        resolved_amp = amp_dtype

    cmd_args = [
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--bucket-size", str(bucket_size),
        "--warmup-epochs", str(warmup_epochs),
        "--amp-dtype", resolved_amp,
        "--output-dir", str(output_dir),
        "--dataset-repo", dataset_repo,
        "--prefetch-factor", str(prefetch_factor),
    ]

    if token:
        cmd_args.extend(["--hf-token", token])
    if max_train_batches > 0:
        cmd_args.extend(["--max-train-batches", str(max_train_batches)])
    if max_validation_batches > 0:
        cmd_args.extend(["--max-validation-batches", str(max_validation_batches)])
    if auto_download:
        os.environ["AUTO_DOWNLOAD"] = "1"
        try:
            cmd_args.append("--auto-download")
        except Exception:
            pass

    if resume:
        try:
            cmd_args.append("--resume")
        except Exception:
            pass
    if resume_checkpoint:
        cmd_args.extend(["--resume-checkpoint", str(resume_checkpoint)])

    sys_argv_backup = sys.argv
    try:
        sys.argv = [sys.argv[0]] + cmd_args
        print(f"Launching Kaggle training with args: {' '.join(cmd_args)}", flush=True)
        train_stream_main()
    finally:
        sys.argv = sys_argv_backup


def main() -> None:
    parser = argparse.ArgumentParser(description="Kaggle Training Entry Point")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--bucket-size", type=int, default=4096, help="Length bucket size for memory efficiency")
    parser.add_argument("--warmup-epochs", type=int, default=2, help="Linear LR warmup epochs")
    parser.add_argument("--amp-dtype", choices=("bf16", "fp16", "auto"), default="auto", help="Precision (auto/bf16/fp16)")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory for checkpoints")
    parser.add_argument("--data-dir", type=Path, default=None, help="Dataset directory")
    parser.add_argument("--dataset-repo", type=str, default="Sanng1112/vietnamese-spelling-synthetic-1gb", help="HuggingFace dataset repository")
    parser.add_argument("--hf-token", type=str, default=None, help="HuggingFace access token")
    parser.add_argument("--prefetch-factor", type=int, default=16, help="Asynchronous GPU prefetch factor")
    parser.add_argument("--max-train-batches", type=int, default=0, help="For bounded run (0=all)")
    parser.add_argument("--max-validation-batches", type=int, default=0, help="For bounded run (0=all)")
    parser.add_argument("--no-auto-download", action="store_true", help="Disable auto download from HuggingFace")
    parser.add_argument("--no-resume", action="store_true", help="Disable auto-resuming from existing checkpoint")
    parser.add_argument("--resume-checkpoint", type=Path, default=None, help="Path to checkpoint file (.pt) to resume from")

    args = parser.parse_args()

    run(
        epochs=args.epochs,
        batch_size=args.batch_size,
        bucket_size=args.bucket_size,
        warmup_epochs=args.warmup_epochs,
        amp_dtype=args.amp_dtype,
        output_dir=args.output_dir,
        max_train_batches=args.max_train_batches,
        max_validation_batches=args.max_validation_batches,
        auto_download=not args.no_auto_download,
        dataset_repo=args.dataset_repo,
        hf_token=args.hf_token,
        prefetch_factor=args.prefetch_factor,
        resume=not args.no_resume,
        resume_checkpoint=args.resume_checkpoint,
    )




if __name__ == "__main__":
    main()

