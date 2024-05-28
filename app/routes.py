from fastapi import APIRouter, UploadFile, File, Depends, status
from pydantic import BaseModel
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

class HealthCheck(BaseModel):
    """Response model to validate and return when performing a health check."""
    status: str = "OK"

@router.get(
    "/health",
    tags=["healthcheck"],
    summary="Perform a Health Check",
    response_description="Return HTTP Status Code 200 (OK)",
    status_code=status.HTTP_200_OK,
    response_model=HealthCheck,
)
def get_health() -> HealthCheck:
    """
    ## Perform a Health Check
    Endpoint to perform a healthcheck on. This endpoint can primarily be used Docker
    to ensure a robust container orchestration and management is in place. Other
    services which rely on proper functioning of the API service will not deploy if this
    endpoint returns any other HTTP status code except 200 (OK).
    Returns:
        HealthCheck: Returns a JSON response with the health status
    """
    return HealthCheck(status="OK")