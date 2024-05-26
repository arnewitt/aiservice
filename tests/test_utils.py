import pytest
import json
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi import UploadFile, HTTPException
from app.models import WhisperTranscriber
from app.utils import handle_transcription, transcribe_async  

class AsyncContextManagerMock:
    def __init__(self, obj):
        self.obj = obj

    async def __aenter__(self):
        return self.obj

    async def __aexit__(self, exc_type, exc, tb):
        pass

@pytest.mark.asyncio
async def test_handle_transcription_success():
    mock_file = MagicMock(spec=UploadFile)
    mock_file.read = AsyncMock(return_value=b"fake audio data")
    mock_file.filename = "test.wav"
    
    mock_transcriber = MagicMock(spec=WhisperTranscriber)
    mock_transcriber.transcribe = MagicMock(return_value="fake transcript")

    mock_tempfile = MagicMock()
    mock_tempfile.write = AsyncMock()
    mock_tempfile.name = "/fake/temp/path/test.wav"
    
    with patch('aiofiles.tempfile.NamedTemporaryFile', return_value=AsyncContextManagerMock(mock_tempfile)):
        with patch('os.remove') as mock_remove:
            response = await handle_transcription(mock_file, mock_transcriber)
    
            mock_file.read.assert_awaited_once()
            mock_tempfile.write.assert_awaited_once_with(b"fake audio data")
            mock_remove.assert_called_once_with("/fake/temp/path/test.wav")
            assert response.status_code == 200
            assert json.loads(response.body) == {"transcript": "fake transcript"}

@pytest.mark.asyncio
async def test_handle_transcription_failure():
    mock_file = MagicMock(spec=UploadFile)
    mock_file.read = AsyncMock(side_effect=Exception("Read error"))
    mock_file.filename = "test.wav"
    
    mock_transcriber = MagicMock(spec=WhisperTranscriber)

    with patch('aiofiles.tempfile.NamedTemporaryFile', new_callable=AsyncMock):
        with pytest.raises(HTTPException) as exc_info:
            await handle_transcription(mock_file, mock_transcriber)
            
            assert exc_info.value.status_code == 500
            assert exc_info.value.detail == "Read error"

@pytest.mark.asyncio
async def test_transcribe_async():
    mock_transcriber = MagicMock(spec=WhisperTranscriber)
    mock_transcriber.transcribe = MagicMock(return_value="fake transcript")
    file_path = "/fake/path/test.wav"

    transcript = await transcribe_async(mock_transcriber, file_path)
    
    mock_transcriber.transcribe.assert_called_once_with(file_path)
    assert transcript == "fake transcript"
