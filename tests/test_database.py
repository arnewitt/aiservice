import pytest
from unittest.mock import patch, MagicMock
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from app.database import ChromaDBHandler

@pytest.fixture
def handler():
    return ChromaDBHandler()

@patch("os.path.exists", return_value=True)
@patch.object(ChromaDBHandler, "load_from_disk")
def test_initialize_db_load_existing(mock_load_from_disk, mock_exists, handler):
    handler._initialize_db()
    mock_load_from_disk.assert_called_once()

@patch("os.path.exists", return_value=False)
def test_initialize_db_no_existing(mock_exists, handler):
    handler._initialize_db()
    assert handler.db is None
    assert handler.docs is None

@patch.object(TextLoader, "load", side_effect=Exception("File not found"))
def test_load_documents_exception(mock_load, handler):
    handler.load_documents("non_existent_file.txt")
    assert handler.docs is None

def test_create_db_no_documents(handler):
    with patch("builtins.print") as mocked_print:
        handler.create_db()
        mocked_print.assert_called_with("No documents loaded. Please load documents first.")

@patch.object(Chroma, "from_documents")
def test_create_db_with_documents(mock_from_documents, handler):
    handler.docs = ["This is a test document."]
    handler.create_db()
    mock_from_documents.assert_called_once_with(handler.docs, handler.embedding_function)
    
def test_similarity_search_no_db(handler):
    handler.db = None
    with patch("builtins.print") as mocked_print:
        results = handler.similarity_search("test query", k=3)
        assert results == []
        mocked_print.assert_called_with("Database not initialized. Please create or load the database first.")

@patch.object(Chroma, "from_documents")
def test_save_to_disk_with_documents(mock_from_documents, handler):
    handler.docs = ["This is a test document."]
    handler.save_to_disk()
    mock_from_documents.assert_called_once_with(handler.docs, handler.embedding_function, persist_directory=handler.persist_directory)

def test_save_to_disk_no_documents(handler):
    with patch("builtins.print") as mocked_print:
        handler.save_to_disk()
        mocked_print.assert_called_with("No documents to save. Please load documents first.")

@patch.object(Chroma, "__init__", return_value=None)
def test_load_from_disk(mock_chroma_init, handler):
    handler.load_from_disk()
    mock_chroma_init.assert_called_once_with(persist_directory=handler.persist_directory, embedding_function=handler.embedding_function)
