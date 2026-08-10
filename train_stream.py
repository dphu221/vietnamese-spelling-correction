#!/usr/bin/env python3
"""Memory-bounded training for the full synthetic Vietnamese corpus.

Unlike ``train_seed.py``, this program never materializes a JSONL split in
RAM.  It reads a bounded window, groups examples of similar token length, and
then sends each already-collated batch to the GPU.  This keeps the 4.56M-sample
training set practical on the 30 GB host while avoiding excessive padding.

Example:
  conda run -n deeplearning_env python train_stream.py --epochs 10 \
    --warmup-epochs 2 --output-dir checkpoints/full_1gb_10_epochs
"""

from __future__ import annotations

import argparse
import json
import random
import time
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterator

import torch
from torch.utils.data import DataLoader, IterableDataset

from models import HierarchicalSpellingCorrector, compact_1m_config


ROOT = Path(__file__).resolve().parent


def download_hf_dataset(repo_id: str, target_dir: Path, token: str | None = None) -> Path:
    """Download public, gated, or private dataset from HuggingFace Hub to target_dir."""
    target_dir.mkdir(parents=True, exist_ok=True)
    hf_token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    print(f"Downloading dataset repository '{repo_id}' to '{target_dir}'...", flush=True)
    if hf_token:
        print("Using HuggingFace authentication token for gated/private dataset access.", flush=True)
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir=str(target_dir), token=hf_token)
        return target_dir
    except ImportError:
        import subprocess
        print("huggingface_hub Python library not found. Installing...", flush=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"], check=True)
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir=str(target_dir), token=hf_token)
        return target_dir


def resolve_dataset_paths(
    train_path: Path,
    val_path: Path,
    word_vocab_path: Path,
    char_vocab_path: Path,
    data_dir: Path | None = None,
    auto_download: bool = False,
    dataset_repo: str = "Sanng1112/vietnamese-spelling-synthetic-1gb",
    hf_token: str | None = None,
) -> tuple[Path, Path, Path, Path]:
    """Resolve paths to dataset splits and vocabularies, handling local, Kaggle, and HF downloads."""
    required = {
        "train": "train_full.jsonl",
        "val": "validation_full.jsonl",
        "word_vocab": "word_vocab_full.json",
        "char_vocab": "char_vocab_full.json",
    }

    # 1. Direct check of given paths
    if train_path.exists() and val_path.exists() and word_vocab_path.exists() and char_vocab_path.exists():
        return train_path, val_path, word_vocab_path, char_vocab_path

    # 2. Check candidate directories (e.g. --data-dir, ROOT/data, Kaggle inputs)
    candidates: list[Path] = []
    if data_dir:
        candidates.append(data_dir)

    candidates.extend([
        ROOT / "datasets/processed",
        ROOT / "data",
        Path("data"),
        Path("/kaggle/input/vietnamese-spelling-synthetic-1gb"),
        Path("/kaggle/input/vietnamese-spelling-synthetic-1gb/data"),
        Path("/kaggle/input/vietnamese-spelling-synthetic-1gb/processed"),
    ])

    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        for sub in kaggle_input.iterdir():
            if sub.is_dir():
                candidates.append(sub)
                candidates.append(sub / "data")
                candidates.append(sub / "processed")

    for dir_path in candidates:
        if not dir_path.exists():
            continue
        t = dir_path / required["train"]
        v = dir_path / required["val"]
        w = dir_path / required["word_vocab"]
        c = dir_path / required["char_vocab"]
        if t.exists() and v.exists() and w.exists() and c.exists():
            print(f"Auto-resolved dataset location: {dir_path}", flush=True)
            return t, v, w, c

    # 3. Auto-download from HuggingFace if requested or AUTO_DOWNLOAD=1 env set
    should_dl = auto_download or os.environ.get("AUTO_DOWNLOAD", "0") == "1"
    if should_dl:
        dl_dir = data_dir if data_dir else (ROOT / "data")
        download_hf_dataset(dataset_repo, dl_dir, token=hf_token)
        t = dl_dir / required["train"]
        v = dl_dir / required["val"]
        w = dl_dir / required["word_vocab"]
        c = dl_dir / required["char_vocab"]
        if t.exists() and v.exists() and w.exists() and c.exists():
            print(f"Dataset files ready in: {dl_dir}", flush=True)
            return t, v, w, c
        raise FileNotFoundError(f"Downloaded repo '{dataset_repo}' to '{dl_dir}', but required dataset files were not found.")


    raise FileNotFoundError(
        "Dataset files not found. Provide valid paths with --train/--validation/--word-vocab/--char-vocab, "
        "or pass --data-dir, or use --auto-download (or AUTO_DOWNLOAD=1 environment variable)."
    )


def resolve_checkpoint_path(
    resume_checkpoint: Path | None,
    output_dir: Path,
    resume: bool = False,
) -> Path | None:
    """Find the latest checkpoint (.pt) to resume training from."""
    if resume_checkpoint and resume_checkpoint.exists():
        return resume_checkpoint

    should_resume = resume or os.environ.get("RESUME", "0") == "1"
    if not should_resume:
        return None

    candidates: list[Path] = [
        output_dir / "last.pt",
        output_dir / "best.pt",
        Path("/kaggle/working/checkpoints") / output_dir.name / "last.pt",
        Path("/kaggle/working/checkpoints") / output_dir.name / "best.pt",
    ]

    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        for root, _, files in os.walk(kaggle_input):
            for file in files:
                if file in ("last.pt", "best.pt"):
                    candidates.append(Path(root) / file)

    for cand in candidates:
        if cand and cand.exists():
            return cand

    return None




import numpy as np
import queue
import threading


class RecordCollator:
    """Convert a pre-grouped list of JSON records to the model's six tensors using ultra-fast NumPy array slicing."""

    def __init__(self, word_vocab: dict[str, int], char_vocab: dict[str, int], max_chars: int) -> None:
        self.word_vocab, self.char_vocab, self.max_chars = word_vocab, char_vocab, max_chars
        self.pad_word = word_vocab["<pad>"]
        self.unk_word = word_vocab["<unk>"]
        self.pad_char = char_vocab["<pad>"]
        self.unk_char = char_vocab["<unk>"]

    def __call__(self, records: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        batch_size = len(records)
        max_tokens = max(len(record["noisy_tokens"]) for record in records)
        max_c = self.max_chars

        word_ids_np = np.full((batch_size, max_tokens), self.pad_word, dtype=np.int64)
        char_ids_np = np.full((batch_size, max_tokens, max_c), self.pad_char, dtype=np.int64)
        att_mask_np = np.zeros((batch_size, max_tokens), dtype=bool)
        char_att_mask_np = np.zeros((batch_size, max_tokens, max_c), dtype=bool)
        det_labels_np = np.full((batch_size, max_tokens), -100, dtype=np.int64)
        corr_labels_np = np.full((batch_size, max_tokens), -100, dtype=np.int64)

        w_get = self.word_vocab.get
        c_get = self.char_vocab.get
        unk_w = self.unk_word
        unk_c = self.unk_char

        for i, record in enumerate(records):
            noisy = record["noisy_tokens"]
            clean = record["clean_tokens"]
            labels = record["detection_labels"]
            count = len(noisy)

            word_ids_np[i, :count] = [w_get(t, unk_w) for t in noisy]
            corr_labels_np[i, :count] = [w_get(t, unk_w) for t in clean]
            det_labels_np[i, :count] = labels
            att_mask_np[i, :count] = True

            for j, token in enumerate(noisy):
                chars = token[:max_c]
                c_count = len(chars)
                char_ids_np[i, j, :c_count] = [c_get(ch, unk_c) for ch in chars]
                char_att_mask_np[i, j, :c_count] = True

        return {
            "word_ids": torch.from_numpy(word_ids_np),
            "char_ids": torch.from_numpy(char_ids_np),
            "attention_mask": torch.from_numpy(att_mask_np),
            "char_attention_mask": torch.from_numpy(char_att_mask_np),
            "detection_labels": torch.from_numpy(det_labels_np),
            "correction_labels": torch.from_numpy(corr_labels_np),
        }



class BackgroundPrefetcher:
    """Asynchronously prefetch collated batches into GPU memory using a background thread."""

    def __init__(self, loader: DataLoader, device: torch.device, prefetch_factor: int = 16) -> None:
        self.loader = loader
        self.device = device
        self.queue: queue.Queue = queue.Queue(maxsize=prefetch_factor)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _worker(self) -> None:
        for cpu_batch in self.loader:
            if self.stop_event.is_set():
                break
            if self.device.type == "cuda":
                gpu_batch = {
                    k: v.to(self.device, non_blocking=True)
                    for k, v in cpu_batch.items()
                }
            else:
                gpu_batch = cpu_batch
            self.queue.put(gpu_batch)
        self.queue.put(None)

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        while not self.stop_event.is_set():
            batch = self.queue.get()
            if batch is None:
                break
            yield batch

    def close(self) -> None:
        self.stop_event.set()
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except Exception:
                break


try:
    import orjson
    def fast_json_loads(line: str | bytes) -> dict[str, Any]:
        return orjson.loads(line)
except ImportError:
    try:
        import ujson
        def fast_json_loads(line: str | bytes) -> dict[str, Any]:
            return ujson.loads(line)
    except ImportError:
        def fast_json_loads(line: str | bytes) -> dict[str, Any]:
            return json.loads(line)


class JsonlBucketBatches(IterableDataset[list[dict[str, Any]]]):
    """Stream JSONL in bounded length buckets.

    Attention's padding cost grows roughly with the square of sequence length.
    Sorting only a small ``bucket_size`` window is a useful compromise: it
    achieves near-length-bucket efficiency, while RAM stays bounded and the
    batch order is shuffled independently every epoch.
    """

    def __init__(self, path: Path, batch_size: int, bucket_size: int, seed: int, shuffle: bool) -> None:
        super().__init__()
        if bucket_size < batch_size:
            raise ValueError("bucket_size must be at least batch_size")
        self.path, self.batch_size, self.bucket_size = path, batch_size, bucket_size
        self.seed, self.shuffle, self.epoch = seed, shuffle, 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _batches(self, records: list[dict[str, Any]], rng: random.Random) -> Iterator[list[dict[str, Any]]]:
        records.sort(key=lambda record: len(record["noisy_tokens"]))
        batches = [records[offset:offset + self.batch_size] for offset in range(0, len(records), self.batch_size)]
        if self.shuffle:
            rng.shuffle(batches)
        yield from batches

    def __iter__(self) -> Iterator[list[dict[str, Any]]]:
        worker = torch.utils.data.get_worker_info()
        if worker is not None:
            raise RuntimeError("JsonlBucketBatches requires num_workers=0 to preserve a single ordered stream.")
        rng = random.Random(self.seed + self.epoch)
        bucket: list[dict[str, Any]] = []
        with self.path.open("rb") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = fast_json_loads(line)
                bucket.append(record)
                if len(bucket) == self.bucket_size:
                    yield from self._batches(bucket, rng)
                    bucket = []
        if bucket:
            yield from self._batches(bucket, rng)



def move(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {name: value.to(device, non_blocking=True) for name, value in batch.items()}



def evaluate(
    model: HierarchicalSpellingCorrector,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> dict[str, float]:
    model.eval()
    loss_sum = batches = tp = fp = fn = correct_detected = detected_true = correct_total = true_total = 0
    with torch.no_grad():
        for cpu_batch in loader:
            batch = move(cpu_batch, device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
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
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
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
    parser.add_argument("--train", type=Path, default=ROOT / "datasets/processed/train_full.jsonl")
    parser.add_argument("--validation", type=Path, default=ROOT / "datasets/processed/validation_full.jsonl")
    parser.add_argument("--word-vocab", type=Path, default=ROOT / "datasets/processed/word_vocab_full.json")
    parser.add_argument("--char-vocab", type=Path, default=ROOT / "datasets/processed/char_vocab_full.json")
    parser.add_argument("--data-dir", type=Path, default=None, help="Directory containing dataset files.")
    parser.add_argument("--auto-download", action="store_true", help="Automatically download dataset from HuggingFace if dataset files are missing.")
    parser.add_argument("--dataset-repo", type=str, default="Sanng1112/vietnamese-spelling-synthetic-1gb", help="HuggingFace dataset repository.")
    parser.add_argument("--hf-token", type=str, default=None, help="HuggingFace access token for gated/private datasets.")
    parser.add_argument("--train-examples", type=int, default=4_564_618)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--bucket-size", type=int, default=4_096)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument(
        "--amp-dtype",
        choices=("bf16", "fp16", "auto"),
        default="auto",
        help="CUDA autocast dtype. 'auto' selects bf16 if supported by hardware (Ampere+), otherwise fp16.",
    )
    parser.add_argument("--log-every", type=int, default=1_000, help="Print every N optimization steps.")
    parser.add_argument("--max-train-batches", type=int, default=0, help="For a bounded smoke run; 0 means all batches.")
    parser.add_argument("--max-validation-batches", type=int, default=0, help="For a bounded smoke run; 0 means all batches.")
    parser.add_argument("--prefetch-factor", type=int, default=16, help="Number of batches to prefetch asynchronously into GPU memory.")
    parser.add_argument("--compile", action="store_true", help="Compile model using PyTorch 2.x torch.compile for Triton kernel fusion.")
    parser.add_argument("--resume", action="store_true", help="Automatically resume training from latest checkpoint if available.")
    parser.add_argument("--resume-checkpoint", type=Path, default=None, help="Explicit path to checkpoint file (.pt) to resume from.")
    parser.add_argument("--seed", type=int, default=20_260_808)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "checkpoints/full_1gb_10_epochs")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch-size must be positive")

    random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = device.type == "cuda"

    if args.amp_dtype == "auto":
        if amp_enabled:
            if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
                amp_dtype_str = "bf16"
                amp_dtype = torch.bfloat16
            else:
                amp_dtype_str = "fp16"
                amp_dtype = torch.float16
        else:
            amp_dtype_str = "fp32"
            amp_dtype = torch.float32
    else:
        amp_dtype_str = args.amp_dtype
        amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16

    if amp_enabled:
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    train_path, val_path, word_vocab_path, char_vocab_path = resolve_dataset_paths(
        train_path=args.train,
        val_path=args.validation,
        word_vocab_path=args.word_vocab,
        char_vocab_path=args.char_vocab,
        data_dir=args.data_dir,
        auto_download=args.auto_download,
        dataset_repo=args.dataset_repo,
        hf_token=args.hf_token,
    )

    word_vocab = json.loads(word_vocab_path.read_text(encoding="utf-8"))
    char_vocab = json.loads(char_vocab_path.read_text(encoding="utf-8"))
    config = compact_1m_config()
    if len(word_vocab) != config.word_vocab_size or len(char_vocab) != config.char_vocab_size:
        raise ValueError("Vocabulary sizes do not match compact_1m_config().")

    collator = RecordCollator(word_vocab, char_vocab, config.max_chars_per_token)
    train_data = JsonlBucketBatches(train_path, args.batch_size, args.bucket_size, args.seed, shuffle=True)
    validation_data = JsonlBucketBatches(val_path, args.batch_size, args.bucket_size, args.seed, shuffle=False)
    train_loader = DataLoader(train_data, batch_size=None, collate_fn=collator, pin_memory=amp_enabled, num_workers=0)
    validation_loader = DataLoader(validation_data, batch_size=None, collate_fn=collator, pin_memory=amp_enabled, num_workers=0)

    model = HierarchicalSpellingCorrector(config).to(device)
    if args.compile and hasattr(torch, "compile"):
        print("Compiling model with PyTorch 2.x torch.compile for fused Triton kernels...", flush=True)
        try:
            model = torch.compile(model)  # type: ignore[assignment]
        except Exception as e:
            print(f"Warning: torch.compile failed ({e}), falling back to standard execution.", flush=True)

    base_lr = 1e-4

    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, betas=(0.9, 0.95), weight_decay=0.01)
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled and amp_dtype == torch.float16)
    steps_per_epoch = (args.train_examples + args.batch_size - 1) // args.batch_size
    warmup_steps = args.warmup_epochs * steps_per_epoch

    output_dir = args.output_dir
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        output_dir = Path("/kaggle/working") / "checkpoints" / args.output_dir.name
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Fallback output directory to writeable path: {output_dir}", flush=True)

    args.output_dir = output_dir

    start_epoch = 0
    global_step = 0
    best_loss = float("inf")
    history: list[dict[str, float]] = []

    ckpt_path = resolve_checkpoint_path(args.resume_checkpoint, args.output_dir, args.resume)
    if ckpt_path:
        print(f"Loading checkpoint state from: {ckpt_path}", flush=True)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            except Exception as e:
                print(f"Warning: Could not restore optimizer state ({e}). Optimizer re-initialized.", flush=True)
        start_epoch = ckpt.get("epoch", 0)
        global_step = ckpt.get("global_step", 0)
        if "validation" in ckpt and "loss" in ckpt["validation"]:
            best_loss = ckpt["validation"]["loss"]

        hist_json = args.output_dir / "history.json"
        if not hist_json.exists() and ckpt_path.parent != args.output_dir:
            hist_json = ckpt_path.parent / "history.json"
        if hist_json.exists():
            try:
                history = json.loads(hist_json.read_text(encoding="utf-8"))
                if history:
                    best_loss = min(best_loss, min(h["loss"] for h in history if "loss" in h))
                print(f"Restored history log with {len(history)} previous epoch records.", flush=True)
            except Exception:
                pass
        print(f"Resumed at start_epoch={start_epoch}, global_step={global_step}, best_loss={best_loss:.6f}", flush=True)

    print(f"device={device}; train={args.train_examples}; batch_size={args.batch_size}; bucket_size={args.bucket_size}", flush=True)
    print(f"AdamW(lr=1e-4, betas=(0.9, 0.95), weight_decay=0.01); epochs={args.epochs}", flush=True)
    print(f"warmup_epochs={args.warmup_epochs}; warmup_steps={warmup_steps}; streaming=true; amp={amp_dtype_str}; prefetch={args.prefetch_factor}", flush=True)
    print(f"train_data={train_path}; val_data={val_path}", flush=True)

    history_path = args.output_dir / "history.jsonl"
    file_mode = "a" if start_epoch > 0 else "w"
    with history_path.open(file_mode, encoding="utf-8") as history_handle:
        for epoch in range(start_epoch + 1, start_epoch + args.epochs + 1):
            started = time.perf_counter()
            model.train(); train_loss_sum = steps = 0
            train_data.set_epoch(epoch)
            prefetcher = BackgroundPrefetcher(train_loader, device, prefetch_factor=args.prefetch_factor)
            try:
                for batch in prefetcher:
                    if args.max_train_batches and steps >= args.max_train_batches:
                        break
                    if warmup_steps:
                        scale = 0.1 + 0.9 * min(1.0, (global_step + 1) / warmup_steps)
                        for group in optimizer.param_groups:
                            group["lr"] = base_lr * scale
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                        output = model(**batch)
                        assert output.loss is not None
                    scaler.scale(output.loss).backward()
                    scaler.step(optimizer); scaler.update()
                    train_loss_sum += output.loss.item(); steps += 1; global_step += 1
                    if steps % args.log_every == 0:
                        print(f"epoch={epoch:03d} step={steps} global_step={global_step} train_loss={train_loss_sum / steps:.4f} lr={optimizer.param_groups[0]['lr']:.2e}", flush=True)
            finally:
                prefetcher.close()

            validation_data.set_epoch(epoch)
            if args.max_validation_batches:
                validation_iter = (batch for index, batch in enumerate(validation_loader) if index < args.max_validation_batches)
                metrics = evaluate(model, validation_iter, device, amp_enabled, amp_dtype)  # type: ignore[arg-type]
            else:
                metrics = evaluate(model, validation_loader, device, amp_enabled, amp_dtype)
            row = {
                "epoch": epoch,
                "global_step": global_step,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "epoch_seconds": time.perf_counter() - started,
                "train_loss": train_loss_sum / max(1, steps),
                "train_batches": steps,
                **metrics,
            }
            history.append(row)
            history_handle.write(json.dumps(row) + "\n"); history_handle.flush()
            torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "config": asdict(config), "epoch": epoch, "global_step": global_step, "validation": metrics}, args.output_dir / "last.pt")
            if metrics["loss"] < best_loss:
                best_loss = metrics["loss"]
                torch.save({"model_state_dict": model.state_dict(), "config": asdict(config), "epoch": epoch, "global_step": global_step, "validation": metrics}, args.output_dir / "best.pt")
            print("epoch={:03d} lr={:.2e} train_loss={:.4f} val_loss={:.4f} f1={:.4f} correction_recall={:.4f} seconds={:.1f}".format(epoch, row["learning_rate"], row["train_loss"], row["loss"], row["detector_f1"], row["end_to_end_correction_recall"], row["epoch_seconds"]), flush=True)
    (args.output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"best_val_loss={best_loss:.6f}; saved={args.output_dir / 'best.pt'}", flush=True)


if __name__ == "__main__":
    main()
