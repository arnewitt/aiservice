import pytest
from fastapi import FastAPI, UploadFile
from fastapi.testclient import TestClient
from app.models import WhisperTranscriber, OllamaChatModel
from app.routes import router

app = FastAPI()
app.include_router(router)

# Mock model for dependency injection
class MockOllamaChatModel:
    def chat(self, text: str) -> dict:
        return {"response": f"Mocked response for: {text}"}

# Dependency override
app.dependency_overrides[OllamaChatModel] = MockOllamaChatModel

client = TestClient(app)

# Mock objects and functions
class MockWhisperTranscriber:
    async def transcribe(self, file):
        return "Mock transcription"

async def mock_handle_transcription(file: UploadFile, transcriber: WhisperTranscriber):
    return {"transcript": "Mock transcription"}

@pytest.fixture
def mock_transcriber(monkeypatch):
    monkeypatch.setattr("app.models.WhisperTranscriber", MockWhisperTranscriber)
    monkeypatch.setattr("app.utils.handle_transcription", mock_handle_transcription)

def test_transcribe_audio(mock_transcriber):
    test_file_path = "tests/sample_data/test.m4a"
    with open(test_file_path, "rb") as audio_file:
        response = client.post(
            "/transcribe",
            files={"file": ("test,m4a", audio_file, "audio/mpeg")}
        )
    assert response.status_code == 200

def test_get_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "OK"}

def test_chat_model_response():
    response = client.post("/chat_response", json={"text": "Hello"})
    assert response.status_code == 200
    assert response.json() == {"response": "Mocked response for: Hello"}

def test_chat_model_response_empty_text():
    response = client.post("/chat_response", json={"text": ""})
    assert response.status_code == 200
    assert response.json() == {"response": "Mocked response for: "}

def test_chat_model_response_special_characters():
    special_text = "!@#$%^&*()_+-=[]{}|;':,.<>/?"
    response = client.post("/chat_response", json={"text": special_text})
    assert response.status_code == 200
    assert response.json() == {"response": f"Mocked response for: {special_text}"}
