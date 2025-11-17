# BGEM3 Embeddings
"""
Text embedding generation using BGE-M3
"""
from typing import List, Any
import torch
from sentence_transformers import SentenceTransformer
from backend.config.settings import settings
from backend.config.model_config import ModelRegistry
from backend.utils.logger import app_logger as logger
from transformers import AutoModel, AutoTokenizer
from langchain_text_splitters import RecursiveCharacterTextSplitter



class TextEmbedder:
    """Generate text embeddings using BGE-M3"""
    
    # def __init__(self):
    #     logger.info("Loading BGE-M3 model...")
        
    #     self.model_config = ModelRegistry.TEXT_EMBEDDING
    #     self.device = "cuda" if torch.cuda.is_available() and settings.enable_gpu else "cpu"
        
    #     try:
    #         self.model = SentenceTransformer(
    #             self.model_config.path,
    #             device=self.device
    #         )
    #         logger.info(f"BGE-M3 loaded on {self.device}")
    #     except Exception as e:
    #         logger.error(f"Failed to load BGE-M3: {e}")
    #         raise
    def __init__(self):
        logger.info("Loading BGE-M3 model...")
        
        self.model_config = ModelRegistry.TEXT_EMBEDDING
        self.device = "cuda" if torch.cuda.is_available() and settings.enable_gpu else "cpu"
        
        self.model = SentenceTransformer(
            self.model_config.path,
            device=self.device
        )
        logger.info(f"BGE-M3 loaded on {self.device}")
        
        # except Exception as e:
        #     if "meta tensor" in str(e).lower() or "Cannot copy out of meta tensor" in str(e):
        #         logger.warning("Meta tensor detected, forcing re-download...")
                
        #         # Clear cache and force download
        #         import os
        #         os.environ['TRANSFORMERS_OFFLINE'] = '0'
                
        #         # Use transformers directly with force download
                
        #         model = AutoModel.from_pretrained(
        #             self.model_config.path,
        #             torch_dtype=torch.float16,
        #             trust_remote_code=True,
        #             force_download=True,
        #             local_files_only=False
        #         )
        #         tokenizer = AutoTokenizer.from_pretrained(
        #             self.model_config.path,
        #             trust_remote_code=True
        #         )
                
        #         # Convert to SentenceTransformer format
        #         self.model = SentenceTransformer(
        #             model_name_or_path=self.model_config.path,
        #             device=self.device
        #         )
                
        #     else:
        #         logger.error(f"Failed to load BGE-M3: {e}")
        #         raise
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
            is_batch = True  # because now we have multiple chunks

        else:
            # Already a list of strings
            is_batch = True
            texts = text

        print(f"Number of chunks: {len(texts)}")

        # Log embedding operation
        if file_name:
            logger.info(f"Generating text embedding for: {file_name}")
        else:
            logger.info(f"Generating text embedding for {len(texts)} text chunk(s)")

        try:
            # Encode all chunks
            embeddings = self.model.encode(
                texts,
                batch_size=settings.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=len(texts) > 10
            )

            embeddings_list = embeddings.tolist()
            print(len(embeddings_list))
            # Return a dict with embeddings and the chunk texts for clarity
            return {"embeddings": embeddings_list, "texts": texts}

        except Exception as e:
            logger.error(f"Error generating embedding for {file_name or 'text'}: {e}")
            # On error return a dict with zeroed embeddings and the original texts
            return {"embeddings": [[0.0] * self.model_config.dimension for _ in texts], "texts": texts}
