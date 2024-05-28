import whisper
import torch
import logging
import ssl
from langchain_community.llms import Ollama

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WhisperTranscriber:
    """
    A class to handle transcription using the OpenAI Whisper model.
    
    Attributes:
    - model (whisper.Whisper): The Whisper model instance.
    - device (str): The device (CPU or GPU) to run the model on.
    """

    def __init__(self, model_size: str = "small"):
        """
        Initialize the WhisperTranscriber with the specified model size.

        Parameters:
        - model_size (str): The size of the Whisper model to load (e.g., 'tiny', 'base', 'small', 'medium', 'large').
        """
        # Disable SSL verification - WARNING: NOT SAFE
        ssl._create_default_https_context = ssl._create_unverified_context

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Loading Whisper model: {model_size} on {self.device}")

        self.model = whisper.load_model(model_size, device=self.device)

    def transcribe(self, audio_file_path: str) -> str:
        """
        Transcribe the given audio file to text.
        ffmpeg is used to load audio. Examples are: m4a, mp3, webm, mp4, mpga, wav and mpeg.

        Parameters:
        - audio_file_path (str): The path to the audio file to be transcribed.

        Returns:
        - str: The transcribed text.

        Raises:
        - RuntimeError: If transcription fails.
        """
        try:
            logger.info(f"Transcribing audio file: {audio_file_path}")
            result = self.model.transcribe(audio_file_path)
            return result["text"]
        except Exception as e:
            logger.error(f"Transcription failed: {str(e)}")
            raise RuntimeError(f"Transcription failed: {str(e)}")


class OllamaChatModel:
    """
    A class to handle OllamaChat model.

    Attributes:
    - model_name (str): The name of the model. Must be listed on https://ollama.com/library 
    """

    def __init__(self, model_name: str):
        """
        Initialize the OllamaChatModel with the specified model path.

        Parameters:
        - model_name (str): The name of the model.
        """
        self.model_name = model_name

    def chat(self, prompt: str) -> str:
        """
        Generate a response to the given prompt.

        Parameters:
        - prompt (str): The prompt to generate a response for.

        Returns:
        - str: The generated response.
        """
        llm = Ollama(model=self.model_name, base_url="http://ollama:11434")
        return llm.invoke(prompt)