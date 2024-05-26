import pytest
from fastapi.testclient import TestClient
from app.main import create_app

# Initialize the FastAPI app
app = create_app(model_size="small")
client = TestClient(app)

def test_transcribe_audio():
    """Test case for the /transcribe endpoint with a valid audio file."""
    audio_file_path = "tests/sample_data/test.m4a"
    
    with open(audio_file_path, "rb") as audio_file:
        response = client.post("/transcribe", files={"file": audio_file})
    
    assert response.status_code == 200
    assert "transcript" in response.json()
    assert isinstance(response.json()["transcript"], str)

def test_transcribe_invalid_file():
    """Test case for the /transcribe endpoint with an invalid file."""
    invalid_file_content = b"This is not an audio file"
    
    response = client.post("/transcribe", files={"file": ("tests/sample_data/invalid.txt", invalid_file_content)})
    
    assert response.status_code == 500

def test_transcribe_no_file():
    """Test case for the /transcribe endpoint without a file."""
    response = client.post("/transcribe")
    
    assert response.status_code == 422

if __name__ == "__main__":
    pytest.main()
