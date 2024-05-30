import pytest
from fastapi.testclient import TestClient
from app.main import create_app
from app.models import WhisperTranscriber, OllamaChatModel
from app.database import ChromaDBHandler
from configparser import ConfigParser

config = ConfigParser()
config.read("config.ini")
app = create_app(config['test'])

client = TestClient(app)

# Mock dependencies
class MockWhisperTranscriber:
    async def transcribe(self, file):
        return "transcribed text"

class MockOllamaChatModel:
    def chat(self, text):
        return f"response to {text}"

class MockChromaDBHandler:
    def similarity_search(self, text, k):
        class MockDocument:
            def __init__(self, content):
                self.page_content = content
        return [MockDocument(f'document {i}') for i in range(k)]

@pytest.fixture
def mock_audio_file(tmp_path):
    audio_file = tmp_path / "test_audio.wav"
    audio_file.write_bytes(b"fake audio data")
    return audio_file

# Override the dependencies
app.dependency_overrides[WhisperTranscriber] = MockWhisperTranscriber
app.dependency_overrides[OllamaChatModel] = MockOllamaChatModel
app.dependency_overrides[ChromaDBHandler] = MockChromaDBHandler

def test_healthcheck():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "OK"}

@pytest.fixture
def mock_audio_file(tmp_path):
    audio_file = tmp_path / "test_audio.wav"
    audio_file.write_bytes(b"fake audio data")
    return audio_file

def test_chat_model_response():
    response = client.post("/chat_response", json={"text": "Hello"})
    assert response.status_code == 200
    assert response.json() == {"response": "response to Hello"}

def test_similarity_search():
    response = client.post("/similarity", json={"text": "sample", "k": 3})
    assert response.status_code == 200
    expected_response = {
        "documents": [
            {"content": "document 0", "top_k": 0},
            {"content": "document 1", "top_k": 1},
            {"content": "document 2", "top_k": 2},
        ]
    }
    assert response.json() == expected_response