#!/usr/bin/env python3
"""Train Compact-1M on the generated seed data and report convergence.

Example:
  conda run -n deeplearning_env python train_seed.py --epochs 100
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from models import HierarchicalSpellingCorrector, compact_1m_config


ROOT = Path(__file__).resolve().parent


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


class SpellingDataset(Dataset[dict[str, Any]]):
    def __init__(self, records: list[dict[str, Any]], word_vocab: dict[str, int], char_vocab: dict[str, int], max_chars: int) -> None:
        self.records = records
        self.word_vocab = word_vocab
        self.char_vocab = char_vocab
        self.max_chars = max_chars

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        noisy, clean, labels = record["noisy_tokens"], record["clean_tokens"], record["detection_labels"]
        if not (len(noisy) == len(clean) == len(labels)):
            raise ValueError(f"Unaligned record: {record.get('id')}")
        return record

    @property
    def lengths(self) -> list[int]:
        return [len(record["noisy_tokens"]) for record in self.records]

    def collate(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        pad_word, unk_word = self.word_vocab["<pad>"], self.word_vocab["<unk>"]
        pad_char, unk_char = self.char_vocab["<pad>"], self.char_vocab["<unk>"]
        max_tokens = max(len(record["noisy_tokens"]) for record in batch)
        word_ids, char_ids, masks, char_masks, detector, corrector = [], [], [], [], [], []
        for record in batch:
            noisy, clean, labels = record["noisy_tokens"], record["clean_tokens"], record["detection_labels"]
            count = len(noisy)
            row_words = [self.word_vocab.get(token, unk_word) for token in noisy]
            row_correct = [self.word_vocab.get(token, unk_word) for token in clean]
            row_chars, row_char_masks = [], []
            for token in noisy:
                encoded = [self.char_vocab.get(character, unk_char) for character in token[:self.max_chars]]
                row_chars.append(encoded + [pad_char] * (self.max_chars - len(encoded)))
                row_char_masks.append([True] * len(encoded) + [False] * (self.max_chars - len(encoded)))
            padding = max_tokens - count
            row_words += [pad_word] * padding
            row_correct += [-100] * padding
            labels = labels + [-100] * padding
            row_chars += [[pad_char] * self.max_chars for _ in range(padding)]
            row_char_masks += [[False] * self.max_chars for _ in range(padding)]
            word_ids.append(row_words); char_ids.append(row_chars)
            masks.append([True] * count + [False] * padding); char_masks.append(row_char_masks)
            detector.append(labels); corrector.append(row_correct)
        return {
            "word_ids": torch.tensor(word_ids, dtype=torch.long),
            "char_ids": torch.tensor(char_ids, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.bool),
            "char_attention_mask": torch.tensor(char_masks, dtype=torch.bool),
            "detection_labels": torch.tensor(detector, dtype=torch.long),
            "correction_labels": torch.tensor(corrector, dtype=torch.long),
        }


class LengthBucketBatchSampler:
    """Shuffle batches while keeping similarly sized sentences together.

    Padding cost for attention scales quadratically with sequence length. The
    seed corpus has a long tail up to 192 tokens, so ordinary random batching
    makes almost every batch as expensive as its longest sentence.
    """

    def __init__(self, lengths: list[int], batch_size: int, seed: int) -> None:
        self.sorted_indices = sorted(range(len(lengths)), key=lengths.__getitem__)
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        batches = [self.sorted_indices[start:start + self.batch_size] for start in range(0, len(self.sorted_indices), self.batch_size)]
        random.Random(self.seed + self.epoch).shuffle(batches)
        self.epoch += 1
        yield from batches

    def __len__(self) -> int:
        return (len(self.sorted_indices) + self.batch_size - 1) // self.batch_size


def move(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {name: value.to(device, non_blocking=True) for name, value in batch.items()}


def evaluate(model: HierarchicalSpellingCorrector, loader: DataLoader, device: torch.device, amp_enabled: bool) -> dict[str, float]:
    model.eval()
    loss_sum = batches = tp = fp = fn = correct_detected = detected_true = correct_total = true_total = 0
    with torch.no_grad():
        for cpu_batch in loader:
            batch = move(cpu_batch, device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                output = model(**batch)
            assert output.loss is not None
            loss_sum += output.loss.item(); batches += 1
            valid = batch["attention_mask"]
            truth = batch["detection_labels"].eq(1) & valid
            predicted = output.detection_logits.argmax(dim=-1).eq(1) & valid
            correction = output.correction_logits.argmax(dim=-1)
            tp += int((predicted & truth).sum()); fp += int((predicted & ~truth & valid).sum()); fn += int((~predicted & truth).sum())
            detected_true += int((predicted & truth).sum())
            correct_detected += int((correction.eq(batch["correction_labels"]) & predicted & truth).sum())
            correct_total += int((correction.eq(batch["correction_labels"]) & truth).sum())
            true_total += int(truth.sum())
    precision = tp / max(1, tp + fp); recall = tp / max(1, tp + fn)
    return {
        "loss": loss_sum / max(1, batches),
        "detector_f1": 2 * precision * recall / max(1e-12, precision + recall),
        "detector_precision": precision,
        "detector_recall": recall,
        "corrector_detected_accuracy": correct_detected / max(1, detected_true),
        "end_to_end_correction_recall": correct_total / max(1, true_total),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=ROOT / "datasets/processed/train_seed.jsonl")
    parser.add_argument("--validation", type=Path, default=ROOT / "datasets/processed/validation_seed.jsonl")
    parser.add_argument("--word-vocab", type=Path, default=ROOT / "datasets/processed/word_vocab_seed.json")
    parser.add_argument("--char-vocab", type=Path, default=ROOT / "datasets/processed/char_vocab_seed.json")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--warmup-epochs", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "checkpoints/seed_100_epochs")
    args = parser.parse_args()

    random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = device.type == "cuda"
    if amp_enabled:
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
    word_vocab = json.loads(args.word_vocab.read_text(encoding="utf-8"))
    char_vocab = json.loads(args.char_vocab.read_text(encoding="utf-8"))
    config = compact_1m_config()
    if len(word_vocab) != config.word_vocab_size or len(char_vocab) != config.char_vocab_size:
        raise ValueError("Vocabulary sizes do not match compact_1m_config().")

    train_data = SpellingDataset(load_jsonl(args.train), word_vocab, char_vocab, config.max_chars_per_token)
    validation_data = SpellingDataset(load_jsonl(args.validation), word_vocab, char_vocab, config.max_chars_per_token)
    train_sampler = LengthBucketBatchSampler(train_data.lengths, args.batch_size, args.seed)
    train_loader = DataLoader(train_data, batch_sampler=train_sampler, collate_fn=train_data.collate, pin_memory=amp_enabled)
    validation_loader = DataLoader(validation_data, batch_size=args.batch_size, shuffle=False, collate_fn=validation_data.collate, pin_memory=amp_enabled)

    model = HierarchicalSpellingCorrector(config).to(device)
    base_lr = 1e-4
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, betas=(0.9, 0.95), weight_decay=0.01)
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"device={device}; train={len(train_data)}; validation={len(validation_data)}; batch_size={args.batch_size}")
    warmup_steps = args.warmup_epochs * len(train_loader)
    print("AdamW(lr=1e-4, betas=(0.9, 0.95), weight_decay=0.01); epochs=" + str(args.epochs))
    print(f"warmup_epochs={args.warmup_epochs}; warmup_steps={warmup_steps}")

    history: list[dict[str, float]] = []
    best_loss = float("inf")
    history_path = args.output_dir / "history.jsonl"
    with history_path.open("w", encoding="utf-8") as history_handle:
        global_step = 0
        for epoch in range(1, args.epochs + 1):
            epoch_started = time.perf_counter()
            model.train(); train_loss_sum = steps = 0
            for cpu_batch in train_loader:
                batch = move(cpu_batch, device)
                # Linear warmup starts at 10% of the requested LR and reaches
                # the requested 1e-4 exactly at the end of warmup epoch 2.
                if warmup_steps:
                    scale = 0.1 + 0.9 * min(1.0, (global_step + 1) / warmup_steps)
                    for group in optimizer.param_groups:
                        group["lr"] = base_lr * scale
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                    output = model(**batch)
                    assert output.loss is not None
                scaler.scale(output.loss).backward()
                scaler.step(optimizer); scaler.update()
                train_loss_sum += output.loss.item(); steps += 1; global_step += 1
            metrics = evaluate(model, validation_loader, device, amp_enabled)
            row = {
                "epoch": epoch,
                "global_step": global_step,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "epoch_seconds": time.perf_counter() - epoch_started,
                "train_loss": train_loss_sum / max(1, steps),
                **metrics,
            }
            history.append(row)
            history_handle.write(json.dumps(row) + "\n"); history_handle.flush()
            if metrics["loss"] < best_loss:
                best_loss = metrics["loss"]
                torch.save({"model_state_dict": model.state_dict(), "config": asdict(config), "epoch": epoch, "validation": metrics}, args.output_dir / "best.pt")
            if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
                print("epoch={:03d} lr={:.2e} train_loss={:.4f} val_loss={:.4f} f1={:.4f} correction_recall={:.4f} seconds={:.1f}".format(epoch, row["learning_rate"], row["train_loss"], row["loss"], row["detector_f1"], row["end_to_end_correction_recall"], row["epoch_seconds"]))
    (args.output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"best_val_loss={best_loss:.6f}; saved={args.output_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
