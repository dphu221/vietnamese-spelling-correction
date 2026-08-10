"""Internal types shared by correction adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Alternative:
    token: str
    confidence: float


@dataclass(frozen=True)
class TokenPrediction:
    replacement: str
    detection_confidence: float
    correction_confidence: float
    alternatives: list[Alternative]


@dataclass(frozen=True)
class AdapterStatus:
    adapter: str
    source: str
    model_loaded: bool
    device: str | None = None
    detail: str | None = None


class CorrectionAdapter(Protocol):
    def predict_tokens(
        self,
        tokens: list[str],
        threshold: float,
        top_k: int = 3,
    ) -> list[TokenPrediction | None]: ...

    def status(self) -> AdapterStatus: ...
