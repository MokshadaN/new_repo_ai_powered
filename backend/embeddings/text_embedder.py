# BGEM3 Embeddings
"""
Text embedding generation using BGE-M3 (served via Ollama)
"""
from typing import List, Any
# import torch  # old local model path
# from sentence_transformers import SentenceTransformer  # old local model path
import requests
from backend.config.settings import settings
from backend.config.model_config import ModelRegistry
from backend.utils.logger import app_logger as logger
# from transformers import AutoModel, AutoTokenizer  # old local model path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ollama import Client

client = Client()





class TextEmbedder:
    """Generate text embeddings using BGE-M3 (via Ollama embeddings API)"""

    # Previous local SentenceTransformer initialization left for reference:
    # def __init__(self):
    #     logger.info("Loading BGE-M3 model...")
    #     self.model_config = ModelRegistry.TEXT_EMBEDDING
    #     self.device = "cuda" if torch.cuda.is_available() and settings.enable_gpu else "cpu"
    #     self.model = SentenceTransformer(self.model_config.path, device=self.device)
    #     logger.info(f"BGE-M3 loaded on {self.device}")

    def __init__(self):
        self.model_config = ModelRegistry.TEXT_EMBEDDING
        # Name must match the Ollama model you pulled (e.g., "bge-m3")
        self.model_name = getattr(self.model_config, "ollama_name", "bge-m3")
        self.ollama_url = getattr(settings, "ollama_base_url", "http://127.0.0.1:11434")
        self.timeout = getattr(settings, "ollama_timeout_seconds", 180)
        logger.info(f"Using Ollama embeddings model '{self.model_name}' at {self.ollama_url}")
    def generate_embedding(self, text, file_name=None):
        """
        Generate embedding(s) for text input.
        - If text is a single long string → chunk it automatically.
        - If text is a list[str] (already chunked) → embed directly.
        """

        # --- Chunk text if it's a string ---
        if isinstance(text, str):
            print("Input is a long string → chunking it...")

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,         # You can tune this
                chunk_overlap=100,      # Recommended overlap
                separators=["\n\n", "\n", " ", ""]
            )

            texts = splitter.split_text(text)

        else:
            # Already a list of strings
            texts = text

        print(f"Number of chunks: {len(texts)}")

        # Log embedding operation
        if file_name:
            logger.info(f"Generating text embedding for: {file_name}")
        else:
            logger.info(f"Generating text embedding for {len(texts)} text chunk(s)")

        try:
            # Call Ollama embeddings endpoint for all chunks
            print("Calling ollama with " , len(texts))
            embeddings_list = []
            for text in texts:
                embedding  = self._embed_with_ollama(text)
                embeddings_list.append(embedding)
            print("Got the embeddings with length " , len(embeddings_list))
            print("Embedding dimension " , len(embeddings_list[0]))
            return {"embeddings": embeddings_list, "texts": texts}

        except Exception as e:
            logger.error(f"Error generating embedding for {file_name or 'text'}: {e}")
            # On error return a dict with zeroed embeddings and the original texts
            return {"embeddings": [[0.0] * self.model_config.dimension for _ in texts], "texts": texts}

    def _embed_with_ollama(self, texts: List[str]) -> List[List[float]]:
        """Call Ollama embeddings endpoint for a list of texts."""
        print("Creating embeddings for ", texts[:10])
        response = client.embeddings(model='bge-m3', prompt=texts)
        # print(response['embedding'])
        return response['embedding']
        # payload = {"model": self.model_name, "input": texts}
        # response = requests.post(
        #     f"{self.ollama_url}/api/embeddings",
        #     json=payload,
        #     timeout=self.timeout,
        # )
        # response.raise_for_status()
        # data = response.json()

        # # Ollama returns either {"embeddings": [...]} for batch or {"embedding": [...]} for single
        # if "embeddings" in data:
        #     embeddings = data["embeddings"]
        # elif "embedding" in data:
        #     embeddings = [data["embedding"]]
        # else:
        #     raise ValueError("Unexpected embeddings response from Ollama")

        # return embeddings
