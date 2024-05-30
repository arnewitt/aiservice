# AI Services API

## Overview

The AI Services API is a fast and flexible module designed to provide various AI services through a RESTful API. Thanks to @ChatGPT for writing my test cases 🤗

## Features

- **Demo AI-service:**
  - Speech-to-Text: Convert audio files to text using the [OpenAI Whisper model](https://github.com/openai/whisper).
  - Chat: Chat with a local LLM, e.g., [gemma:2b](https://huggingface.co/google/gemma-2b).
  - Semantic Similarity Search: Search for knowledge in a database.
- **Modular Architecture**: Easily add new AI services.
- **Scalable**: Deployable with Docker for scalable and consistent environments.
- **FastAPI Framework**: Built with FastAPI for high performance and automatic interactive API documentation.

## Getting Started

### Prerequisites

- Python 3.11+ with dependencies
- FFmpeg (required for audio processing)
- Docker (optional, but preferred for containerized deployment)

### Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/arnewitt/aiservice
    cd ai-services-api
    ```

2. **Install dependencies:**

    Create and activate a virtual environment (optional but recommended):

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate` instead
    ```

3. **Install the required packages:**

    ```bash
    pip install -r requirements.txt
    ```

4. **Running the Server:**

    Using Python directly:

    ```bash
    python server.py --host 0.0.0.0 --port 8000 --model-size large
    ```

5. **Using Docker:**

    Build the Docker image:

    ```bash
    docker build -t whisper-transcription-server .
    ```

    Run the Docker container:

    ```bash
    docker run -p 8000:8000 whisper-transcription-server
    ```

### API Usage

#### 1. Demo Speech-to-Text

- **Endpoint:** POST /transcribe
- **Description:** Transcribe an audio file to text.
  - As per [OpenAI Whisper Documentation](https://help.openai.com/en/articles/7031512-whisper-audio-api-faq), supported file formats are: m4a, mp3, webm, mp4, mpga, wav, mpeg
- **Request:**
  - **File:** An audio file to be transcribed (e.g., .wav, .mp3, .m4a).
- **Response:**
  - **200 OK:** A JSON object containing the transcript.
  - **500 Internal Server Error:** An error message if the transcription fails.
  - **422 Unprocessable Entity:** If no file is provided or the file format is invalid.
- **Example:**

    ```bash
    curl -X POST "http://localhost:8000/transcribe" -F "file=@path/to/your/audiofile.m4a"
    ```

- **Response:**

    ```json
    {
      "transcript": "Transcribed text from the audio file."
    }
    ```

- **Warning:** We disable SSL verification in `app/models.py` during the initial model downloading process, creating a security risk. Consider properly installing SSL certificates.

#### 2. Chat with an LLM

- **Endpoint:** POST /chat_response
- **Description:** Get a response from an LLM to a prompt.
- **Request:**
  - **JSON Object:** `{"text": "<prompt>"}`
- **Response:**
  - **200 OK:** A JSON object with the LLM's response.
  - **500 Internal Server Error:** An error message if no message can be generated.
- **Example:**

    ```bash
    curl --header "Content-Type: application/json" \
    --request POST \
    --data '{"text": "hello, world"}' \
    http://localhost:8000/chat_response
    ```

- **Response:**

    ```json
    {
      "response": "Hello! 👋 It's great to hear from you. What can I do for you today? 😊"
    }
    ```

- **Tip:** Make sure you download a model on the Ollama instance:

    ```bash
    docker exec -it ollama ollama run gemma:2b
    ```

#### 3. Get K Nearest Documents from Database Based on Semantic Similarity Search

- **Endpoint:** POST /similarity
- **Description:** Retrieve the k most similar documents to a sentence from the database behind this service.
- **Request:** `{"text": "<prompt>", "k": 3}`
- **Response:**
  - **200 OK:** A JSON object with the top k most similar documents.
  - **500 Internal Server Error:** An error message if no response can be generated.
- **Example:**

    ```bash
    curl --header "Content-Type: application/json" \
    --request POST \
    --data '{"text": "What is a group of long-legged, pink birds called?", "k": 1}' \
    http://localhost:8000/similarity
    ```

- **Response:**

    ```json
    {
      "documents": [
          {
              "content": "A group of flamingos is called a \"flamboyance.\"",
              "top_k": 0
          }
      ]
    }
    ```

### Testing

To run the test suite, use pytest:

```bash
pytest tests/test_server.py
```

If you run into troubles here, make sure your .venv is activated and ensure that the PYTHONPATH variable is set to the root directory of the project.

```bash
source .venv/bin/activate  # On Windows, use: .venv\Scripts\activate
export PYTHONPATH=$(pwd) # On Windows, use: set PYTHONPATH=%cd%
```

### Extending the API
To add new AI services, follow these steps:

1. Define the service model: Create a new file in the app/models directory for your AI model class.

2. Implement the service logic: Add the service logic in the app/utils.py or create a new utility module.

3. Create API routes: Define new endpoints in the app/routes.py file or create a new route module.

4. Register the routes: Include the new routes in the app/main.py.