from pathlib import Path

from fastapi.testclient import TestClient

import api


class DummyModel:
    classes_ = ["A", "B", "C"]

    def predict_proba(self, _):
        return [[0.7, 0.2, 0.1]]


def _configure_test_state(monkeypatch):
    monkeypatch.setattr(api, "SYMPTOM_COLUMNS", ["cough", "fever", "fatigue"])
    monkeypatch.setattr(api, "FREQ_MAP", {"A": ["cough", "fever", "fatigue"]})
    monkeypatch.setattr(api, "model", DummyModel())


def test_root_endpoint(monkeypatch):
    _configure_test_state(monkeypatch)
    client = TestClient(api.app)

    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["status"] == "BetterHealth AI API is running"


def test_analyze_returns_need_more_info_with_few_symptoms(monkeypatch):
    _configure_test_state(monkeypatch)
    monkeypatch.setattr(
        api, "extract_symptoms_with_gemini", lambda *_: ["cough", "fever"]
    )
    client = TestClient(api.app)

    response = client.post(
        "/analyze",
        json={"text": "patient has cough and fever", "session_id": "s1"},
    )

    body = response.json()
    assert response.status_code == 200
    assert body["status"] == "need_more_info"
    assert body["session_id"] == "s1"


def test_followup_merges_new_symptoms(monkeypatch):
    _configure_test_state(monkeypatch)
    monkeypatch.setattr(
        api, "parse_followup_answers_with_gemini", lambda *_: {"fatigue": 1}
    )
    client = TestClient(api.app)

    response = client.post(
        "/followup",
        json={
            "session_id": "s2",
            "existing_symptoms": {"cough": 1, "fever": 1},
            "follow_up_answers": "yes to fatigue",
            "questions_asked": ["Is there fatigue?"],
        },
    )

    body = response.json()
    assert response.status_code == 200
    assert body["status"] in {"prediction_ready", "prediction_with_followup"}
    assert body["extracted_symptoms"]["fatigue"] == 1


def test_save_case_writes_jsonl(monkeypatch, tmp_path):
    _configure_test_state(monkeypatch)
    target_file = tmp_path / "new_cases.jsonl"
    monkeypatch.setattr(api, "CASES_DIR", tmp_path)
    monkeypatch.setattr(api, "CASES_FILE", target_file)
    client = TestClient(api.app)

    payload = {
        "session_id": "save-1",
        "symptoms": {"cough": 1},
        "confirmed_diagnosis": "A",
        "notes": "ok",
        "medication": "none",
        "test_results": "pending",
    }
    response = client.post("/save-case", json=payload)

    assert response.status_code == 200
    assert target_file.exists()
    assert "save-1" in target_file.read_text(encoding="utf-8")
