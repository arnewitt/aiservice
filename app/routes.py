from fastapi import APIRouter, UploadFile, File, Depends, status
from pydantic import BaseModel
from app.models import WhisperTranscriber, OllamaChatModel
from app.utils import handle_transcription
from app.database import ChromaDBHandler

router = APIRouter()

class ChatItem(BaseModel):
    """Response model for a chat item."""
    text: str

class SimilaritySearchItem(BaseModel):
    """Response model for a similarity search item."""
    text: str
    k: int

class HealthCheck(BaseModel):
    """Response model to validate and return when performing a health check."""
    status: str = "OK"

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

@router.post(
    "/chat_response",
    tags=["chat"],
    summary="Generate a response from the OllamaChatModel",
    response_description="Return a response from the OllamaChatModel",
    status_code=status.HTTP_200_OK,
    response_model=dict,
)
async def chat_model_response(prompt: ChatItem, model: OllamaChatModel = Depends()):
    """
    Endpoint to generate a response from the OllamaChatModel.

    Parameters:
    - prompt (ChatItem): The prompt to generate a response for.
    - model (OllamaChatModel): The OllamaChatModel to use for generating the response.

    Returns:
    - JSONResponse: A JSON response containing the generated response.
    """
    return model.chat(prompt.text)

@router.post(
    "/similarity",
    tags=["similarity"],
    summary="Retrieve the top k most similar documents from the database",
    response_description="Return the top k most similar documents from the database",
    status_code=status.HTTP_200_OK,
    response_model=dict,
    )
async def similarity(prompt: SimilaritySearchItem, db: ChromaDBHandler = Depends()):
    """
    Endpoint to retrieve the top k similar documents from the database.

    Parameters:
    - prompt (SimilaritySearchItem): The prompt to search for k most similar documents.

    Returns:
    - JSONResponse: A JSON response containing the top k similar documents. 
    """
    k_most_similar = db.similarity_search(prompt.text, k=prompt.k)
    response = [{'content': doc.page_content, 'top_k': index} for index, doc in enumerate(k_most_similar)]
    return {"documents": response}

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
    Endpoint to perform a healthcheck on. 
    
    Returns:
        HealthCheck: Returns a JSON response with the health status
    """
    return HealthCheck(status="OK")