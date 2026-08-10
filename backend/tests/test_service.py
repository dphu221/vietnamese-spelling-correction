from backend.app.correction.adapters import DemoCorrectionAdapter
from backend.app.correction.service import CorrectionService, chunk_token_spans, token_spans
from backend.app.correction.types import AdapterStatus, Alternative, TokenPrediction
from backend.app.schemas import CorrectionMode


class FixedConfidenceAdapter:
    def __init__(self, correction_confidence: float) -> None:
        self.correction_confidence = correction_confidence

    def predict_tokens(self, tokens: list[str], threshold: float, top_k: int = 3):
        return [TokenPrediction(
            replacement="đúng",
            detection_confidence=0.99,
            correction_confidence=self.correction_confidence,
            alternatives=[Alternative("đúng", self.correction_confidence)],
        ) for _ in tokens]

    def status(self) -> AdapterStatus:
        return AdapterStatus(adapter="test", source="test", model_loaded=True)


def test_preserves_whitespace_punctuation_and_newlines() -> None:
    text = "Hom   nay,\ntui ko đi học."
    result = CorrectionService(DemoCorrectionAdapter()).correct(text, CorrectionMode.balanced)
    assert result["corrected_text"] == "Hôm   nay,\ntôi không đi học."
    assert [(item["start"], item["end"]) for item in result["corrections"]] == [(0, 3), (11, 14), (15, 17)]


def test_modes_apply_different_thresholds() -> None:
    service = CorrectionService(DemoCorrectionAdapter())
    conservative = service.correct("k đi nx", CorrectionMode.conservative)
    aggressive = service.correct("k đi nx", CorrectionMode.aggressive)
    assert conservative["corrected_text"] == "k đi nữa"
    assert aggressive["corrected_text"] == "không đi nữa"


def test_modes_also_gate_low_confidence_corrections() -> None:
    service = CorrectionService(FixedConfidenceAdapter(correction_confidence=0.60))
    conservative = service.correct("sai", CorrectionMode.conservative)
    balanced = service.correct("sai", CorrectionMode.balanced)
    assert conservative["corrected_text"] == "sai"
    assert balanced["corrected_text"] == "đúng"
    assert conservative["correction_threshold"] == 0.80
    assert balanced["correction_threshold"] == 0.50


def test_chunking_never_exceeds_model_limit_and_prefers_sentence_boundary() -> None:
    text = " ".join(["từ"] * 120 + ["xong."] + ["từ"] * 120)
    chunks = chunk_token_spans(token_spans(text), max_tokens=192)
    assert [len(chunk) for chunk in chunks] == [121, 120]
    assert all(len(chunk) <= 192 for chunk in chunks)


def test_no_change_round_trips_exactly() -> None:
    text = "Tiếng Việt đúng sẵn.\n  Dòng thứ hai."
    result = CorrectionService(DemoCorrectionAdapter()).correct(text, CorrectionMode.balanced)
    assert result["corrected_text"] == text
    assert result["corrections"] == []
