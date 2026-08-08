#!/usr/bin/env python3
"""Inference pipeline for Vietnamese Spelling Correction model.

Given a trained model checkpoint and vocabulary files, this module tokenizes input text,
runs the Hierarchical Spelling Corrector, and returns corrected sentences.

Example CLI:
  python infer.py --checkpoint checkpoints/full_1gb_bf16_3_epochs/best.pt \
                  --text "Tôi là sinh viên trường dại học bách khoa"

Example Python API:
  from infer import VietnameseSpellingCorrectorPipeline
  pipeline = VietnameseSpellingCorrectorPipeline.from_checkpoint("checkpoints/.../best.pt")
  result = pipeline.correct_text("Tôi là sinh viên trường dại học bách khoa")
  print(result["corrected_text"])
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from models import HierarchicalSpellingCorrector, SpellingCorrectionConfig, compact_1m_config

ROOT = Path(__file__).resolve().parent


class VietnameseSpellingCorrectorPipeline:
    """End-to-end inference pipeline for Vietnamese spelling correction."""

    def __init__(
        self,
        model: HierarchicalSpellingCorrector,
        word_vocab: dict[str, int],
        char_vocab: dict[str, int],
        device: torch.device | str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.model.eval()

        self.word_vocab = word_vocab
        self.char_vocab = char_vocab
        self.inv_word_vocab = {idx: word for word, idx in word_vocab.items()}
        self.inv_char_vocab = {idx: char for char, idx in char_vocab.items()}

        self.pad_word = word_vocab.get("<pad>", 0)
        self.unk_word = word_vocab.get("<unk>", 1)
        self.pad_char = char_vocab.get("<pad>", 0)
        self.unk_char = char_vocab.get("<unk>", 1)
        self.max_chars = model.config.max_chars_per_token

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Path | str,
        word_vocab_path: Path | str | None = None,
        char_vocab_path: Path | str | None = None,
        device: torch.device | str | None = None,
    ) -> VietnameseSpellingCorrectorPipeline:
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        ckpt = torch.load(checkpoint_path, map_location="cpu")
        config_dict = ckpt.get("config")
        if config_dict is None:
            config = compact_1m_config()
        else:
            config = SpellingCorrectionConfig(**config_dict)

        model = HierarchicalSpellingCorrector(config)
        model.load_state_dict(ckpt["model_state_dict"])

        # Auto-locate vocabularies if not provided
        if word_vocab_path is None or char_vocab_path is None:
            candidate_dirs = [
                checkpoint_path.parent,
                ROOT / "datasets/processed",
                ROOT / "data",
                Path("data"),
                Path("/kaggle/input/vietnamese-spelling-synthetic-1gb"),
                Path("/kaggle/input/vietnamese-spelling-synthetic-1gb/data"),
            ]
            w_path, c_path = None, None
            for d in candidate_dirs:
                if (d / "word_vocab_full.json").exists() and (d / "char_vocab_full.json").exists():
                    w_path = d / "word_vocab_full.json"
                    c_path = d / "char_vocab_full.json"
                    break
                elif (d / "word_vocab_seed.json").exists() and (d / "char_vocab_seed.json").exists():
                    w_path = d / "word_vocab_seed.json"
                    c_path = d / "char_vocab_seed.json"
                    break

            if word_vocab_path is None:
                word_vocab_path = w_path
            if char_vocab_path is None:
                char_vocab_path = c_path

        if word_vocab_path is None or char_vocab_path is None:
            raise FileNotFoundError(
                "Could not auto-locate word_vocab and char_vocab. Please pass word_vocab_path and char_vocab_path explicitly."
            )

        word_vocab = json.loads(Path(word_vocab_path).read_text(encoding="utf-8"))
        char_vocab = json.loads(Path(char_vocab_path).read_text(encoding="utf-8"))

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        return cls(model=model, word_vocab=word_vocab, char_vocab=char_vocab, device=device)

    def encode_tokens(self, tokens: list[str]) -> tuple[dict[str, torch.Tensor], list[str]]:
        word_ids = [self.word_vocab.get(token, self.unk_word) for token in tokens]
        char_rows: list[list[int]] = []
        for token in tokens:
            row = [self.char_vocab.get(char, self.unk_char) for char in token[: self.max_chars]]
            row += [self.pad_char] * (self.max_chars - len(row))
            char_rows.append(row)

        tensors = {
            "word_ids": torch.tensor([word_ids], dtype=torch.long, device=self.device),
            "char_ids": torch.tensor([char_rows], dtype=torch.long, device=self.device),
            "attention_mask": torch.ones((1, len(tokens)), dtype=torch.bool, device=self.device),
            "char_attention_mask": torch.tensor(
                [[[(val != self.pad_char) for val in row] for row in char_rows]],
                dtype=torch.bool,
                device=self.device,
            ),
        }
        return tensors, tokens

    @torch.no_grad()
    def correct_text(self, text: str) -> dict[str, Any]:
        """Correct typos in a raw text string."""
        tokens = text.split()
        if not tokens:
            return {"original_text": text, "corrected_text": text, "corrections": []}

        tensors, orig_tokens = self.encode_tokens(tokens)
        corrected_ids, error_flags, suggestions = self.model.correct(
            word_ids=tensors["word_ids"],
            char_ids=tensors["char_ids"],
            attention_mask=tensors["attention_mask"],
            char_attention_mask=tensors["char_attention_mask"],
        )

        flags = error_flags[0].cpu().tolist()
        sug_ids = suggestions[0].cpu().tolist()

        corrected_tokens = []
        corrections = []

        for i, orig in enumerate(orig_tokens):
            if flags[i]:
                suggested_word = self.inv_word_vocab.get(sug_ids[i], orig)
                if suggested_word in ("<unk>", "<pad>"):
                    suggested_word = orig
                corrected_tokens.append(suggested_word)
                corrections.append({"index": i, "original": orig, "corrected": suggested_word})
            else:
                corrected_tokens.append(orig)

        corrected_text = " ".join(corrected_tokens)
        return {
            "original_text": text,
            "corrected_text": corrected_text,
            "tokens": orig_tokens,
            "corrected_tokens": corrected_tokens,
            "corrections": corrections,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Vietnamese Spelling Corrector Inference CLI")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--word-vocab", type=Path, default=None, help="Path to word vocabulary JSON")
    parser.add_argument("--char-vocab", type=Path, default=None, help="Path to character vocabulary JSON")
    parser.add_argument("--text", type=str, default="Tôi là sinh viên trường dại học bách khoa", help="Text to correct")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda or cpu)")
    args = parser.parse_args()

    pipeline = VietnameseSpellingCorrectorPipeline.from_checkpoint(
        checkpoint_path=args.checkpoint,
        word_vocab_path=args.word_vocab,
        char_vocab_path=args.char_vocab,
        device=args.device,
    )

    result = pipeline.correct_text(args.text)
    print("\n--- Vietnamese Spelling Correction Result ---")
    print(f"Original : {result['original_text']}")
    print(f"Corrected: {result['corrected_text']}")
    if result["corrections"]:
        print("\nDetected Corrections:")
        for corr in result["corrections"]:
            print(f"  Word #{corr['index']}: '{corr['original']}' -> '{corr['corrected']}'")
    else:
        print("\nNo typos detected.")


if __name__ == "__main__":
    main()
