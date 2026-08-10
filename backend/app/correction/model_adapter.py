"""Adapter for loading the custom model from local files or Hugging Face Hub."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import snapshot_download

from models import HierarchicalSpellingCorrector, SpellingCorrectionConfig

from ..settings import Settings
from .types import AdapterStatus, Alternative, TokenPrediction


MANIFEST_NAME = "model_manifest.json"
REQUIRED_ARTIFACTS = {MANIFEST_NAME, "best.pt", "word_vocab_full.json", "char_vocab_full.json"}


@dataclass(frozen=True)
class ModelManifest:
    architecture: str
    checkpoint: str
    word_vocab: str
    char_vocab: str
    max_tokens: int
    max_chars_per_token: int
    word_vocab_size: int
    char_vocab_size: int


def read_manifest(directory: Path) -> ModelManifest:
    path = directory / MANIFEST_NAME
    if not path.is_file():
        raise ValueError(f"Thiếu tệp {MANIFEST_NAME} trong {directory}.")
    try:
        raw: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        manifest = ModelManifest(**raw)
    except (json.JSONDecodeError, TypeError) as error:
        raise ValueError(f"{MANIFEST_NAME} không hợp lệ: {error}") from error
    if manifest.architecture != "hierarchical-spelling-corrector":
        raise ValueError("Kiến trúc trong model_manifest.json không được hỗ trợ.")
    for name in (manifest.checkpoint, manifest.word_vocab, manifest.char_vocab):
        if Path(name).name != name or not (directory / name).is_file():
            raise ValueError(f"Thiếu hoặc sai đường dẫn tệp mô hình: {name}")
    return manifest


def resolve_model_directory(settings: Settings) -> tuple[Path, str]:
    if settings.model_source == "local":
        return settings.model_local_dir, str(settings.model_local_dir)
    if settings.model_source != "huggingface":
        raise ValueError("Nguồn mô hình thực phải là local hoặc huggingface.")
    if not settings.hf_model_repo:
        raise ValueError("Cần đặt HF_MODEL_REPO khi MODEL_SOURCE=huggingface.")
    downloaded = snapshot_download(
        repo_id=settings.hf_model_repo,
        revision=settings.hf_model_revision,
        token=settings.hf_token,
        allow_patterns=sorted(REQUIRED_ARTIFACTS),
    )
    return Path(downloaded), settings.hf_model_repo


def _read_vocab(path: Path, expected_size: int) -> dict[str, int]:
    try:
        vocab = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Vocabulary không phải JSON hợp lệ: {path.name}") from error
    if not isinstance(vocab, dict) or not all(isinstance(key, str) and isinstance(value, int) for key, value in vocab.items()):
        raise ValueError(f"Vocabulary sai định dạng: {path.name}")
    if len(vocab) != expected_size or set(vocab.values()) != set(range(expected_size)):
        raise ValueError(f"Vocabulary {path.name} phải có {expected_size} ID liên tiếp.")
    if "<pad>" not in vocab or "<unk>" not in vocab:
        raise ValueError(f"Vocabulary {path.name} thiếu <pad> hoặc <unk>.")
    return vocab


class HierarchicalModelAdapter:
    def __init__(self, settings: Settings) -> None:
        directory, source_label = resolve_model_directory(settings)
        manifest = read_manifest(directory)
        checkpoint = torch.load(directory / manifest.checkpoint, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint or "config" not in checkpoint:
            raise ValueError("Checkpoint phải chứa model_state_dict và config.")
        config = SpellingCorrectionConfig(**checkpoint["config"])
        expected = (
            config.max_tokens,
            config.max_chars_per_token,
            config.word_vocab_size,
            config.char_vocab_size,
        )
        declared = (
            manifest.max_tokens,
            manifest.max_chars_per_token,
            manifest.word_vocab_size,
            manifest.char_vocab_size,
        )
        if expected != declared:
            raise ValueError("Cấu hình checkpoint không khớp model_manifest.json.")
        self.word_vocab = _read_vocab(directory / manifest.word_vocab, config.word_vocab_size)
        self.char_vocab = _read_vocab(directory / manifest.char_vocab, config.char_vocab_size)
        self.id_to_word = [""] * len(self.word_vocab)
        for token, index in self.word_vocab.items():
            self.id_to_word[index] = token
        self.config = config
        self.device = self._select_device(settings.model_device)
        self.model = HierarchicalSpellingCorrector(config).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        self.model.eval()
        self.source_label = source_label

    @staticmethod
    def _select_device(requested: str) -> torch.device:
        if requested == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if requested not in {"cpu", "cuda"}:
            raise ValueError("MODEL_DEVICE phải là auto, cpu hoặc cuda.")
        if requested == "cuda" and not torch.cuda.is_available():
            raise ValueError("MODEL_DEVICE=cuda nhưng CUDA không khả dụng.")
        return torch.device(requested)

    def _encode(self, tokens: list[str]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(tokens) > self.config.max_tokens:
            raise ValueError(f"Một đoạn không được vượt quá {self.config.max_tokens} token.")
        pad_word = self.word_vocab["<pad>"]
        unk_word = self.word_vocab["<unk>"]
        pad_char = self.char_vocab["<pad>"]
        unk_char = self.char_vocab["<unk>"]
        word_ids = [self.word_vocab.get(token, unk_word) for token in tokens]
        char_rows: list[list[int]] = []
        char_masks: list[list[bool]] = []
        for token in tokens:
            encoded = [self.char_vocab.get(char, unk_char) for char in token[:self.config.max_chars_per_token]]
            padding = self.config.max_chars_per_token - len(encoded)
            char_rows.append(encoded + [pad_char] * padding)
            char_masks.append([True] * len(encoded) + [False] * padding)
        return (
            torch.tensor([word_ids or [pad_word]], dtype=torch.long, device=self.device),
            torch.tensor([char_rows], dtype=torch.long, device=self.device),
            torch.ones((1, len(tokens)), dtype=torch.bool, device=self.device),
            torch.tensor([char_masks], dtype=torch.bool, device=self.device),
        )

    @torch.inference_mode()
    def predict_tokens(
        self,
        tokens: list[str],
        threshold: float,
        top_k: int = 3,
    ) -> list[TokenPrediction | None]:
        if not tokens:
            return []
        word_ids, char_ids, attention_mask, char_mask = self._encode(tokens)
        output = self.model(word_ids, char_ids, attention_mask, char_mask)
        detection_probabilities = output.detection_logits.softmax(dim=-1)[0, :, 1]
        correction_probabilities = output.correction_logits.softmax(dim=-1)[0]
        predictions: list[TokenPrediction | None] = []
        candidate_count = min(len(self.id_to_word), max(top_k + 12, top_k))
        for index, token in enumerate(tokens):
            detection_confidence = float(detection_probabilities[index].item())
            if detection_confidence < threshold:
                predictions.append(None)
                continue
            values, ids = correction_probabilities[index].topk(candidate_count)
            top_candidate = self.id_to_word[ids[0].item()]
            if top_candidate in {"<pad>", "<unk>"} or top_candidate.startswith("<unused_"):
                predictions.append(None)
                continue
            alternatives: list[Alternative] = []
            for probability, token_id in zip(values.tolist(), ids.tolist()):
                candidate = self.id_to_word[token_id]
                if candidate in {"<pad>", "<unk>"} or candidate.startswith("<unused_"):
                    continue
                alternatives.append(Alternative(candidate, float(probability)))
                if len(alternatives) == top_k:
                    break
            if not alternatives or alternatives[0].token == token:
                predictions.append(None)
                continue
            predictions.append(TokenPrediction(
                replacement=alternatives[0].token,
                detection_confidence=detection_confidence,
                correction_confidence=alternatives[0].confidence,
                alternatives=alternatives,
            ))
        return predictions

    def status(self) -> AdapterStatus:
        return AdapterStatus(
            adapter="hierarchical-transformer",
            source=self.source_label,
            model_loaded=True,
            device=str(self.device),
        )
