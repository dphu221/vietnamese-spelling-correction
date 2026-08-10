import json
from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch

from backend.app.correction.model_adapter import HierarchicalModelAdapter, read_manifest
from backend.app.settings import Settings
from models import HierarchicalSpellingCorrector, SpellingCorrectionConfig


def test_manifest_requires_supported_architecture_and_files(tmp_path) -> None:
    (tmp_path / "model_manifest.json").write_text(json.dumps({
        "architecture": "wrong",
        "checkpoint": "best.pt",
        "word_vocab": "word_vocab_full.json",
        "char_vocab": "char_vocab_full.json",
        "max_tokens": 192,
        "max_chars_per_token": 32,
        "word_vocab_size": 9500,
        "char_vocab_size": 400,
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="Kiến trúc"):
        read_manifest(tmp_path)


def test_manifest_requires_artifacts(tmp_path) -> None:
    (tmp_path / "model_manifest.json").write_text(json.dumps({
        "architecture": "hierarchical-spelling-corrector",
        "checkpoint": "best.pt",
        "word_vocab": "word_vocab_full.json",
        "char_vocab": "char_vocab_full.json",
        "max_tokens": 192,
        "max_chars_per_token": 32,
        "word_vocab_size": 9500,
        "char_vocab_size": 400,
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="Thiếu hoặc sai"):
        read_manifest(tmp_path)


def test_local_adapter_loads_contract_and_runs_inference(tmp_path) -> None:
    config = SpellingCorrectionConfig(
        word_vocab_size=8,
        char_vocab_size=10,
        max_tokens=8,
        max_chars_per_token=4,
        char_hidden_size=8,
        char_num_layers=1,
        char_num_heads=2,
        word_embedding_size=8,
        word_hidden_size=8,
        word_num_layers=1,
        word_num_heads=2,
        detector_hidden_size=4,
        dropout=0,
    )
    model = HierarchicalSpellingCorrector(config)
    torch.save({"model_state_dict": model.state_dict(), "config": asdict(config)}, tmp_path / "best.pt")
    word_vocab = {"<pad>": 0, "<unk>": 1, "sai": 2, "đúng": 3, "từ": 4, ".": 5, "a": 6, "b": 7}
    char_vocab = {"<pad>": 0, "<unk>": 1, "s": 2, "a": 3, "i": 4, "đ": 5, "ú": 6, "n": 7, "g": 8, ".": 9}
    (tmp_path / "word_vocab_full.json").write_text(json.dumps(word_vocab, ensure_ascii=False), encoding="utf-8")
    (tmp_path / "char_vocab_full.json").write_text(json.dumps(char_vocab, ensure_ascii=False), encoding="utf-8")
    (tmp_path / "model_manifest.json").write_text(json.dumps({
        "architecture": "hierarchical-spelling-corrector",
        "checkpoint": "best.pt",
        "word_vocab": "word_vocab_full.json",
        "char_vocab": "char_vocab_full.json",
        "max_tokens": 8,
        "max_chars_per_token": 4,
        "word_vocab_size": 8,
        "char_vocab_size": 10,
    }), encoding="utf-8")
    adapter = HierarchicalModelAdapter(Settings(
        model_source="local",
        model_local_dir=tmp_path,
        model_device="cpu",
    ))
    predictions = adapter.predict_tokens(["sai", "."], threshold=0, top_k=3)
    assert len(predictions) == 2
    assert adapter.status().model_loaded is True
    assert adapter.status().device == "cpu"


def test_adapter_does_not_fall_through_from_unknown_top_prediction() -> None:
    adapter = object.__new__(HierarchicalModelAdapter)
    adapter.id_to_word = ["<pad>", "<unk>", "Joseph", "đẹp,"]
    dummy = torch.zeros(1)
    adapter._encode = lambda tokens: (dummy, dummy, dummy, dummy)
    adapter.model = lambda *args: SimpleNamespace(
        detection_logits=torch.tensor([[[0.0, 5.0]]]),
        correction_logits=torch.tensor([[[0.0, 5.0, 4.0, 1.0]]]),
    )

    predictions = adapter.predict_tokens(["dep,"], threshold=0.5, top_k=3)

    assert predictions == [None]
