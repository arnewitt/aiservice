from fastapi import APIRouter, UploadFile, File, Depends
from app.models import WhisperTranscriber
from app.utils import handle_transcription

router = APIRouter()

@router.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...), transcriber: WhisperTranscriber = Depends()):
    """
    Endpoint to transcribe an uploaded audio file to text.

    Parameters:
    - file (UploadFile): The uploaded audio file.
    - transcriber (WhisperTranscriber): An instance of WhisperTranscriber to perform the transcription (injected by FastAPI).

    Returns:
    - JSONResponse: A JSON response containing the transcript text.
    """
    return await handle_transcription(file, transcriber)
