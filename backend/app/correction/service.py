"""Offset-safe paragraph processing shared by every model adapter."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass

from ..schemas import CorrectionMode
from .explanations import infer_error_type
from .types import CorrectionAdapter


MODE_THRESHOLDS = {
    CorrectionMode.conservative: 0.80,
    CorrectionMode.balanced: 0.50,
    CorrectionMode.aggressive: 0.30,
}
CORRECTION_THRESHOLDS = {
    CorrectionMode.conservative: 0.80,
    CorrectionMode.balanced: 0.50,
    CorrectionMode.aggressive: 0.30,
}
TOKEN = re.compile(r"\S+")
SENTENCE_END = re.compile(r"[.!?…][\"'”’)}\]]*$")


@dataclass(frozen=True)
class TokenSpan:
    text: str
    start: int
    end: int


def token_spans(text: str) -> list[TokenSpan]:
    return [TokenSpan(match.group(), match.start(), match.end()) for match in TOKEN.finditer(text)]


def chunk_token_spans(spans: list[TokenSpan], max_tokens: int = 192) -> list[list[TokenSpan]]:
    if max_tokens < 1:
        raise ValueError("max_tokens phải dương.")
    chunks: list[list[TokenSpan]] = []
    first = 0
    while first < len(spans):
        hard_end = min(first + max_tokens, len(spans))
        end = hard_end
        if hard_end < len(spans):
            search_floor = first + max_tokens // 2
            for index in range(hard_end - 1, search_floor - 1, -1):
                if SENTENCE_END.search(spans[index].text):
                    end = index + 1
                    break
        chunks.append(spans[first:end])
        first = end
    return chunks


class CorrectionService:
    def __init__(self, adapter: CorrectionAdapter, max_tokens: int = 192) -> None:
        self.adapter = adapter
        self.max_tokens = max_tokens

    def correct(self, text: str, mode: CorrectionMode) -> dict[str, object]:
        started = time.perf_counter()
        threshold = MODE_THRESHOLDS[mode]
        correction_threshold = CORRECTION_THRESHOLDS[mode]
        changes: list[dict[str, object]] = []
        for chunk in chunk_token_spans(token_spans(text), self.max_tokens):
            tokens = [span.text for span in chunk]
            predictions = self.adapter.predict_tokens(tokens, threshold=threshold, top_k=3)
            if len(predictions) != len(chunk):
                raise RuntimeError("Adapter trả về số lượng dự đoán không khớp số token.")
            for span, prediction in zip(chunk, predictions):
                if (
                    prediction is None
                    or prediction.replacement == span.text
                    or prediction.correction_confidence < correction_threshold
                ):
                    continue
                error_type, error_label = infer_error_type(span.text, prediction.replacement)
                changes.append({
                    "original": span.text,
                    "replacement": prediction.replacement,
                    "start": span.start,
                    "end": span.end,
                    "detection_confidence": round(prediction.detection_confidence, 6),
                    "correction_confidence": round(prediction.correction_confidence, 6),
                    "alternatives": [
                        {"token": alternative.token, "confidence": round(alternative.confidence, 6)}
                        for alternative in prediction.alternatives
                    ],
                    "error_type": error_type,
                    "error_type_label": error_label,
                    "explanation_is_inferred": True,
                })
        cursor = 0
        rebuilt: list[str] = []
        for change in changes:
            start, end = int(change["start"]), int(change["end"])
            rebuilt.extend((text[cursor:start], str(change["replacement"])))
            cursor = end
        rebuilt.append(text[cursor:])
        status = self.adapter.status()
        return {
            "original_text": text,
            "corrected_text": "".join(rebuilt),
            "corrections": changes,
            "processing_ms": round((time.perf_counter() - started) * 1000, 2),
            "mode": mode,
            "threshold": threshold,
            "correction_threshold": correction_threshold,
            "adapter": status.adapter,
            "model_loaded": status.model_loaded,
        }
