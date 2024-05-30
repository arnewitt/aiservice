import os
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_text_splitters import CharacterTextSplitter

class ChromaDBHandler:
    """
    A class to handle interactions with a Chroma database.

    Attributes:
    - model_name (str): The name of the sentence transformer model to use.
    - chunk_size (int): The maximum number of characters in a chunk.
    - chunk_overlap (int): The number of characters to overlap between chunks.
    - persist_directory (str): The directory to persist the database to.
    """
    def __init__(self, model_name="all-MiniLM-L6-v2", chunk_size=60, chunk_overlap=0, persist_directory="./chroma_db"):
        self.persist_directory = persist_directory
        self.embedding_function = SentenceTransformerEmbeddings(model_name=model_name)
        self.text_splitter = CharacterTextSplitter(separator='\n', chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.db = None
        self.docs = None
        self._initialize_db()

    def _initialize_db(self):
        """
        Initialize the database, loading from disk if it exists.
        """
        if os.path.exists(self.persist_directory):
            self.load_from_disk()
        else:
            self.db = None
            self.docs = None

    def load_documents(self, file_path):
        """
        Loads a document into memory.
        
        Parameters:
        - file_path (str): The path to the document file.
        """
        try:
            loader = TextLoader(file_path)
            documents = loader.load()
            if self.docs:
                self.docs.extend(self.text_splitter.split_documents(documents))
            else:
                self.docs = self.text_splitter.split_documents(documents)
        except Exception as e:
            print(f"Error loading documents: {e}")

    def create_db(self):
        """
        Creates a Chroma database from the loaded documents.
        """
        if self.docs is None:
            print("No documents loaded. Please load documents first.")
            return
        self.db = Chroma.from_documents(self.docs, self.embedding_function)

    def similarity_search(self, query, k=3):
        """
        Performs a similarity search on the database.

        Parameters:
        - query (str): The query to search for.
        - k (int): The number of results to return.

        Returns:
        - List[Document]: The list of k documents that match the query.
        """
        if self.db is None:
            print("Database not initialized. Please create or load the database first.")
            return []
        return self.db.similarity_search(query, k=k)

    def save_to_disk(self):
        """
        Store Chroma DB to disk.
        """
        if self.docs is None:
            print("No documents to save. Please load documents first.")
            return
        self.db = Chroma.from_documents(self.docs, self.embedding_function, persist_directory=self.persist_directory)

    def load_from_disk(self):
        """
        Load Chroma DB from disk.
        """
        self.db = Chroma(persist_directory=self.persist_directory, embedding_function=self.embedding_function)

# Example usage
if __name__ == "__main__":
    handler = ChromaDBHandler()
    query = "What is a group of flamingos called?"
    # Load documents and create the database
    handler.load_documents("./tests/sample_data /test.txt")
    handler.create_db()

    # Save to disk
    handler.save_to_disk()
    
    # Load from disk and perform another similarity search
    handler.load_from_disk()
    docs = handler.similarity_search(query, k=3)
    for index, doc in enumerate(docs):
        print(f"{index} - {doc.page_content}")
