from fastapi import FastAPI
from app.routes import router
from app.models import WhisperTranscriber

def create_app(model_size: str = "large") -> FastAPI:
    """
    Create and configure an instance of the FastAPI application.

    Parameters:
    - model_size (str): The size of the Whisper model to load (e.g., 'tiny', 'base', 'small', 'medium', 'large').

    Returns:
    - FastAPI: The configured FastAPI application instance.
    """
    app = FastAPI()
    app.dependency_overrides[WhisperTranscriber] = lambda: WhisperTranscriber(model_size)
    app.include_router(router)
    return app

def main():
    """
    The main entry point for running the FastAPI application.
    """
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="Run a transcription server using Whisper model.")
    parser.add_argument('--model-size', type=str, default="large", help="Size of the Whisper model to load (e.g., tiny, base, small, medium, large).")
    parser.add_argument('--host', type=str, default="0.0.0.0", help="Host to run the server on.")
    parser.add_argument('--port', type=int, default=8000, help="Port to run the server on.")
    
    args = parser.parse_args()
    
    app = create_app(model_size=args.model_size)
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()
