import aiofiles
import os
import logging
import asyncio
from fastapi.responses import JSONResponse
from fastapi import UploadFile, HTTPException
from app.models import WhisperTranscriber

# Setup logging
logger = logging.getLogger(__name__)

async def handle_transcription(file: UploadFile, transcriber: WhisperTranscriber) -> JSONResponse:
    """
    Handle the transcription of an uploaded audio file.

    Parameters:
    - file (UploadFile): The uploaded audio file.
    - transcriber (WhisperTranscriber): An instance of WhisperTranscriber to perform the transcription.

    Returns:
    - JSONResponse: A JSON response containing the transcript text.

    Raises:
    - HTTPException: If an error occurs during transcription.
    """
    try:
        audio_bytes = await file.read()
        suffix = os.path.splitext(file.filename)[-1]  # Get the file extension from the original file

        async with aiofiles.tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_audio_file:
            await temp_audio_file.write(audio_bytes)
            temp_audio_file_path = temp_audio_file.name

        transcript_text = await transcribe_async(transcriber, temp_audio_file_path)
        
        # Clean up the temporary file
        os.remove(temp_audio_file_path)
        
        return JSONResponse(content={"transcript": transcript_text})
    
    except Exception as e:
        logger.error(f"Error during transcription: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def transcribe_async(transcriber: WhisperTranscriber, file_path: str) -> str:
    """
    Perform transcription asynchronously to avoid blocking the event loop.

    Parameters:
    - transcriber (WhisperTranscriber): An instance of WhisperTranscriber to perform the transcription.
    - file_path (str): The path to the temporary audio file.

    Returns:
    - str: The transcribed text.
    """
    # Run the transcription in a separate thread to avoid blocking the event loop
    loop = asyncio.get_running_loop()
    transcript_text = await loop.run_in_executor(None, transcriber.transcribe, file_path)
    return transcript_text
