from fastapi.testclient import TestClient

from fin_agent.main import app


def test_empty_question_rejected():
    with TestClient(app) as client:
        assert client.post("/ask", json={"question": "   "}).status_code == 422


def test_too_long_question_rejected():
    with TestClient(app) as client:
        assert client.post("/ask", json={"question": "x" * 10_000}).status_code == 422
