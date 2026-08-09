#!/usr/bin/env python3
"""Inspect model parameters, epoch history, and validation metrics from a checkpoint (.pt) or history.json file.

Usage:
  python inspect_checkpoint.py best.pt
  python inspect_checkpoint.py history.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from models import HierarchicalSpellingCorrector, SpellingCorrectionConfig, compact_1m_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect checkpoint metrics and model parameters")
    parser.add_argument("path", type=Path, help="Path to checkpoint (.pt) or history.json")
    args = parser.parse_args()

    target = args.path
    if not target.exists():
        print(f"Error: File not found: {target}")
        return

    print("=" * 60)
    print(f"INSPECTING FILE: {target.resolve()}")
    print("=" * 60)

    if target.suffix == ".json":
        data = json.loads(target.read_text(encoding="utf-8"))
        if isinstance(data, list):
            print(f"\nFound {len(data)} training epoch records in {target.name}:")
            for row in data:
                print(f"\n--- Epoch {row.get('epoch', '?')} ---")
                print(f"  Train Loss                   : {row.get('train_loss', 0):.4f}")
                print(f"  Val Loss                     : {row.get('loss', 0):.4f}")
                print(f"  Detector F1 Score            : {row.get('detector_f1', 0):.4f} (Precision: {row.get('detector_precision', 0):.4f}, Recall: {row.get('detector_recall', 0):.4f})")
                print(f"  Corrector Detected Accuracy  : {row.get('corrector_detected_accuracy', 0):.4f}")
                print(f"  End-to-End Correction Recall : {row.get('end_to_end_correction_recall', 0):.4f}")
                print(f"  Learning Rate                : {row.get('learning_rate', 0):.2e}")
                print(f"  Epoch Duration               : {row.get('epoch_seconds', 0):.1f}s")
        else:
            print(json.dumps(data, indent=2))

    elif target.suffix in (".pt", ".pth"):
        ckpt = torch.load(target, map_location="cpu")
        print("\n--- Checkpoint Metadata ---")
        print(f"  Epoch       : {ckpt.get('epoch', 'N/A')}")
        print(f"  Global Step : {ckpt.get('global_step', 'N/A')}")

        config_dict = ckpt.get("config")
        if config_dict:
            config = SpellingCorrectionConfig(**config_dict)
            print("\n--- Model Configuration ---")
            for k, v in config_dict.items():
                print(f"  {k}: {v}")
        else:
            config = compact_1m_config()

        model = HierarchicalSpellingCorrector(config)
        model.load_state_dict(ckpt["model_state_dict"])
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n  Total Trainable Parameters   : {total_params:,} (~{total_params / 1e6:.2f}M)")

        val = ckpt.get("validation")
        if val:
            print("\n--- Saved Validation Metrics ---")
            print(f"  Validation Loss              : {val.get('loss', 0):.4f}")
            print(f"  Detector F1 Score            : {val.get('detector_f1', 0):.4f}")
            print(f"  Detector Precision           : {val.get('detector_precision', 0):.4f}")
            print(f"  Detector Recall              : {val.get('detector_recall', 0):.4f}")
            print(f"  Corrector Detected Accuracy  : {val.get('corrector_detected_accuracy', 0):.4f}")
            print(f"  End-to-End Correction Recall : {val.get('end_to_end_correction_recall', 0):.4f}")
        else:
            print("\n  No validation dictionary saved in this checkpoint.")
    else:
        print(f"Unsupported file format: {target.suffix}. Provide a .pt checkpoint or .json history file.")


if __name__ == "__main__":
    main()
