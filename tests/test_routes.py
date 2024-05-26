import pytest
from fastapi import FastAPI, UploadFile
from fastapi.testclient import TestClient
from app.models import WhisperTranscriber
from app.routes import router

app = FastAPI()
app.include_router(router)

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
