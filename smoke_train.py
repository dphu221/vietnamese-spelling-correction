#!/usr/bin/env python3
"""Run one real AdamW training step on one generated dataset example.

Run with:
  conda run -n deeplearning_env python smoke_train.py
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from models import HierarchicalSpellingCorrector, compact_1m_config


ROOT = Path(__file__).resolve().parent


def read_first_jsonl(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                return json.loads(line)
    raise ValueError(f"No JSONL record found in {path}")


def encode_example(
    example: dict,
    word_vocab: dict[str, int],
    char_vocab: dict[str, int],
    max_chars: int,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], int]:
    """Turn one aligned JSONL record into model tensors with batch size one."""
    noisy_tokens = example["noisy_tokens"]
    clean_tokens = example["clean_tokens"]
    labels = example["detection_labels"]
    if not (len(noisy_tokens) == len(clean_tokens) == len(labels)):
        raise ValueError("Dataset record is not token aligned.")

    pad_word, unk_word = word_vocab["<pad>"], word_vocab["<unk>"]
    pad_char, unk_char = char_vocab["<pad>"], char_vocab["<unk>"]
    word_ids = [word_vocab.get(token, unk_word) for token in noisy_tokens]
    correction_ids = [word_vocab.get(token, unk_word) for token in clean_tokens]

    char_rows: list[list[int]] = []
    for token in noisy_tokens:
        row = [char_vocab.get(character, unk_char) for character in token[:max_chars]]
        char_rows.append(row + [pad_char] * (max_chars - len(row)))

    tensors = {
        "word_ids": torch.tensor([word_ids], dtype=torch.long, device=device),
        "char_ids": torch.tensor([char_rows], dtype=torch.long, device=device),
        "attention_mask": torch.ones((1, len(noisy_tokens)), dtype=torch.bool, device=device),
        "char_attention_mask": torch.tensor([[[(value != pad_char) for value in row] for row in char_rows]], dtype=torch.bool, device=device),
        "detection_labels": torch.tensor([labels], dtype=torch.long, device=device),
        "correction_labels": torch.tensor([correction_ids], dtype=torch.long, device=device),
    }
    target_unk_count = sum(token == unk_word for token in correction_ids)
    return tensors, target_unk_count


def loss_for(model: HierarchicalSpellingCorrector, batch: dict[str, torch.Tensor]) -> float:
    model.eval()
    with torch.no_grad():
        output = model(**batch)
    assert output.loss is not None
    return output.loss.item()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=ROOT / "datasets/processed/train_seed.jsonl")
    parser.add_argument("--word-vocab", type=Path, default=ROOT / "datasets/processed/word_vocab_seed.json")
    parser.add_argument("--char-vocab", type=Path, default=ROOT / "datasets/processed/char_vocab_seed.json")
    parser.add_argument("--seed", type=int, default=20260808)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    word_vocab = json.loads(args.word_vocab.read_text(encoding="utf-8"))
    char_vocab = json.loads(args.char_vocab.read_text(encoding="utf-8"))
    config = compact_1m_config()
    if len(word_vocab) != config.word_vocab_size or len(char_vocab) != config.char_vocab_size:
        raise ValueError("Vocabulary sizes must match compact_1m_config().")

    example = read_first_jsonl(args.dataset)
    batch, target_unk_count = encode_example(
        example, word_vocab, char_vocab, config.max_chars_per_token, device
    )
    model = HierarchicalSpellingCorrector(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        betas=(0.9, 0.95),
        weight_decay=0.01,
    )

    before_loss = loss_for(model, batch)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    output = model(**batch)
    assert output.loss is not None
    output.loss.backward()
    optimizer.step()
    after_loss = loss_for(model, batch)

    errors = int(batch["detection_labels"].sum().item())
    print(f"device={device}")
    print(f"example_id={example['id']}; tokens={len(example['noisy_tokens'])}; true_errors={errors}")
    print(f"AdamW(lr=1e-4, betas=(0.9, 0.95), weight_decay=0.01)")
    print(f"target_tokens_mapped_to_unk={target_unk_count}")
    print(f"loss_before={before_loss:.6f}")
    print(f"train_loss={output.loss.item():.6f}")
    print(f"detector_loss={output.detection_loss.item():.6f}")
    print(f"corrector_loss={output.correction_loss.item():.6f}")
    print(f"loss_after={after_loss:.6f}")


if __name__ == "__main__":
    main()
