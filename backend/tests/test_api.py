from fastapi.testclient import TestClient

from backend.app.correction.adapters import DemoCorrectionAdapter, UnavailableCorrectionAdapter
from backend.app.main import create_app


def test_health_and_correction_contract() -> None:
    with TestClient(create_app(adapter=DemoCorrectionAdapter())) as client:
        health = client.get("/api/health")
        assert health.status_code == 200
        assert health.json()["adapter"] == "demo"
        response = client.post("/api/correct", json={"text": "Hom nay tui ko đi học.", "mode": "balanced"})
        assert response.status_code == 200
        payload = response.json()
        assert payload["corrected_text"] == "Hôm nay tôi không đi học."
        assert len(payload["corrections"]) == 3
        assert payload["threshold"] == 0.5
        assert payload["correction_threshold"] == 0.5
        assert payload["corrections"][0]["explanation_is_inferred"] is True


def test_invalid_input_and_unavailable_model() -> None:
    with TestClient(create_app(adapter=DemoCorrectionAdapter())) as client:
        assert client.post("/api/correct", json={"text": "   "}).status_code == 422
        assert client.post("/api/correct", json={"text": "x" * 5001}).status_code == 422
    with TestClient(create_app(adapter=UnavailableCorrectionAdapter("test", "missing model"))) as client:
        assert client.get("/api/health").json()["status"] == "unavailable"
        response = client.post("/api/correct", json={"text": "văn bản"})
        assert response.status_code == 503
        assert response.json()["detail"] == "missing model"
