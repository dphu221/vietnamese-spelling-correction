"""Public API schemas."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, field_validator


MAX_TEXT_LENGTH = 5_000


class CorrectionMode(str, Enum):
    conservative = "conservative"
    balanced = "balanced"
    aggressive = "aggressive"


class CorrectionRequest(BaseModel):
    text: str = Field(max_length=MAX_TEXT_LENGTH)
    mode: CorrectionMode = CorrectionMode.balanced

    @field_validator("text")
    @classmethod
    def text_must_have_content(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Vui lòng nhập văn bản cần kiểm tra.")
        return value


class AlternativeResponse(BaseModel):
    token: str
    confidence: float = Field(ge=0, le=1)


class CorrectionItemResponse(BaseModel):
    original: str
    replacement: str
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    detection_confidence: float = Field(ge=0, le=1)
    correction_confidence: float = Field(ge=0, le=1)
    alternatives: list[AlternativeResponse]
    error_type: str
    error_type_label: str
    explanation_is_inferred: bool = True


class CorrectionResponse(BaseModel):
    original_text: str
    corrected_text: str
    corrections: list[CorrectionItemResponse]
    processing_ms: float = Field(ge=0)
    mode: CorrectionMode
    threshold: float = Field(ge=0, le=1)
    correction_threshold: float = Field(ge=0, le=1)
    adapter: str
    model_loaded: bool


class HealthResponse(BaseModel):
    status: str
    adapter: str
    source: str
    model_loaded: bool
    device: str | None = None
    detail: str | None = None
