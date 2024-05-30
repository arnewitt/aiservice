from configparser import ConfigParser
from fastapi import FastAPI
from app.models import WhisperTranscriber, OllamaChatModel
from app.database import ChromaDBHandler
from app.routes import router

def create_app(config: ConfigParser) -> FastAPI:
    """
    Create and configure an instance of the FastAPI application.

    Parameters:
    - config (ConfigParser): ConfigParser instance containing the configuration for the service.

    Returns:
    - FastAPI: The configured FastAPI application instance.
    """
    app = FastAPI()
    # Include transcription service
    whisper_size = config['WhisperSize']
    app.dependency_overrides[WhisperTranscriber] = lambda: WhisperTranscriber(whisper_size)
    
    # Include ollama chat model
    ollama_model = config['OllamaModel']
    app.dependency_overrides[OllamaChatModel] = lambda: OllamaChatModel(ollama_model)

    # Include database handler
    db = ChromaDBHandler(
        model_name=config['EmbeddingModelName'],
        persist_directory=config['ChromaDBPersistDir'],
        chunk_size=config['DocumentChunkSize'],
        chunk_overlap=config['DocumentChunkOverlap']
    )
    db.load_from_disk()
    app.dependency_overrides[ChromaDBHandler] = lambda: db

    # Add routes to service
    app.include_router(router)
    return app

def main():
    """
    The main entry point for running the FastAPI application.
    """
    import argparse
    import uvicorn

    config = ConfigParser()
    config.read('config.ini')

    parser = argparse.ArgumentParser(description="Run your own AI services.")
    parser.add_argument('--host', type=str, default="0.0.0.0", help="Host to run the server on.")
    parser.add_argument('--port', type=int, default=8000, help="Port to run the server on.")
    parser.add_argument('--config', type=str, default="default", help="Name of configuration file during use.")
    
    args = parser.parse_args()
    
    app = create_app(config=config[args.config])
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()
