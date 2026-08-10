"""Environment-driven settings for local inference."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    """Runtime settings with a safe demo-mode default."""

    model_source: str = "demo"
    model_local_dir: Path = PROJECT_ROOT / "model_artifacts"
    hf_model_repo: str | None = None
    hf_model_revision: str | None = None
    hf_token: str | None = None
    model_device: str = "auto"

    @classmethod
    def from_env(cls) -> "Settings":
        source = os.getenv("MODEL_SOURCE", "demo").strip().lower()
        if source not in {"demo", "local", "huggingface"}:
            raise ValueError("MODEL_SOURCE phải là demo, local hoặc huggingface.")
        local_value = os.getenv("MODEL_LOCAL_DIR")
        local_dir = Path(local_value).expanduser() if local_value else PROJECT_ROOT / "model_artifacts"
        if not local_dir.is_absolute():
            local_dir = PROJECT_ROOT / local_dir
        return cls(
            model_source=source,
            model_local_dir=local_dir.resolve(),
            hf_model_repo=os.getenv("HF_MODEL_REPO") or None,
            hf_model_revision=os.getenv("HF_MODEL_REVISION") or None,
            hf_token=os.getenv("HF_TOKEN") or None,
            model_device=os.getenv("MODEL_DEVICE", "auto").strip().lower(),
        )
