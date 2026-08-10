"""Demo and failure-state correction adapters."""

from __future__ import annotations

from dataclasses import dataclass

from .explanations import CORE
from .types import AdapterStatus, Alternative, TokenPrediction


@dataclass(frozen=True)
class DemoEntry:
    replacement: str
    detection_confidence: float
    correction_confidence: float
    alternatives: tuple[str, ...]


DEMO_ENTRIES: dict[str, DemoEntry] = {
    "hom": DemoEntry("hôm", 0.93, 0.91, ("hôm", "họp", "hòm")),
    "troi": DemoEntry("trời", 0.94, 0.90, ("trời", "trôi", "tôi")),
    "dep": DemoEntry("đẹp", 0.91, 0.88, ("đẹp", "dẹp", "dép")),
    "tui": DemoEntry("tôi", 0.96, 0.93, ("tôi", "túi", "tụi")),
    "mik": DemoEntry("mình", 0.97, 0.94, ("mình", "mịch", "minh")),
    "mk": DemoEntry("mình", 0.90, 0.84, ("mình", "mẹ", "mà")),
    "ko": DemoEntry("không", 0.99, 0.97, ("không", "có", "khó")),
    "k": DemoEntry("không", 0.76, 0.74, ("không", "khi", "kì")),
    "hok": DemoEntry("không", 0.96, 0.92, ("không", "học", "họ")),
    "khong": DemoEntry("không", 0.98, 0.96, ("không", "khổng", "khòng")),
    "dc": DemoEntry("được", 0.95, 0.91, ("được", "đọc", "đức")),
    "đc": DemoEntry("được", 0.97, 0.94, ("được", "đọc", "đức")),
    "chs": DemoEntry("chơi", 0.96, 0.92, ("chơi", "chờ", "chỉ")),
    "nx": DemoEntry("nữa", 0.89, 0.82, ("nữa", "này", "nơi")),
}


def _match_case(source: str, target: str) -> str:
    if source.isupper():
        return target.upper()
    if source[:1].isupper():
        return target[:1].upper() + target[1:]
    return target


class DemoCorrectionAdapter:
    """Small deterministic adapter that exercises the complete UI contract."""

    def predict_tokens(
        self,
        tokens: list[str],
        threshold: float,
        top_k: int = 3,
    ) -> list[TokenPrediction | None]:
        predictions: list[TokenPrediction | None] = []
        for token in tokens:
            match = CORE.match(token)
            if match is None:
                predictions.append(None)
                continue
            prefix, core, suffix = match.groups()
            entry = DEMO_ENTRIES.get(core.casefold())
            if entry is None or entry.detection_confidence < threshold:
                predictions.append(None)
                continue
            replacement_core = _match_case(core, entry.replacement)
            replacement = prefix + replacement_core + suffix
            alternatives = [
                Alternative(prefix + _match_case(core, value) + suffix, max(0.01, entry.correction_confidence - index * 0.17))
                for index, value in enumerate(entry.alternatives[:top_k])
            ]
            predictions.append(TokenPrediction(
                replacement=replacement,
                detection_confidence=entry.detection_confidence,
                correction_confidence=entry.correction_confidence,
                alternatives=alternatives,
            ))
        return predictions

    def status(self) -> AdapterStatus:
        return AdapterStatus(
            adapter="demo",
            source="Bộ sửa mẫu xác định",
            model_loaded=False,
            detail="Chế độ minh họa — chưa sử dụng mô hình đã huấn luyện.",
        )


class UnavailableCorrectionAdapter:
    def __init__(self, source: str, detail: str) -> None:
        self.source = source
        self.detail = detail

    def predict_tokens(
        self,
        tokens: list[str],
        threshold: float,
        top_k: int = 3,
    ) -> list[TokenPrediction | None]:
        raise RuntimeError(self.detail)

    def status(self) -> AdapterStatus:
        return AdapterStatus(
            adapter="unavailable",
            source=self.source,
            model_loaded=False,
            detail=self.detail,
        )
