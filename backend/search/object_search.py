"""
Object search functionality.
"""
from typing import List, Dict, Any
import numpy as np
from backend.vector_store.store_manager import VectorStoreManager
from backend.embeddings.embedding_manager import EmbeddingManager
from backend.utils.logger import app_logger as logger


class ObjectSearch:
    """Search for similar images/regions using object embeddings."""

    def __init__(self):
        logger.info("Initializing ObjectSearch")
        self.store_manager = VectorStoreManager()
        self.embedding_manager = EmbeddingManager()

    def search_by_image(self, image_path: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Search objects using an image reference.
        """
        try:
            logger.info("Object search for image: %s", image_path)

            # Generate embedding from image path
            embedding = self.embedding_manager.generate_image_embedding(image_path)
            if embedding is None or (isinstance(embedding, list) and not embedding):
                logger.warning("No embedding generated for image: %s", image_path)
                return []

            # Normalize embedding
            vec = np.asarray(embedding, dtype=np.float32)
            norm = np.linalg.norm(vec)
            if norm == 0:
                logger.warning("Zero-norm embedding for image: %s", image_path)
                return []
            embedding = (vec / norm).tolist()

            logger.info("Query embedding norm for object search: %.4f", norm)

            results = self.store_manager.search_objects(embedding, top_k)
            logger.info("Object search returned %d results for image %s", len(results), image_path)
            return results
        except Exception as exc:
            logger.error("Object search failed for %s: %s", image_path, exc)
            return []

    def search_by_embedding(self, embedding: List[float], top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Search objects using a precomputed embedding.
        """
        try:
            vec = np.asarray(embedding, dtype=np.float32)
            norm = np.linalg.norm(vec)
            if norm == 0:
                logger.warning("Zero-norm object embedding supplied")
                return []
            normalized = (vec / norm).tolist()

            logger.info("Object search by embedding norm: %.4f", norm)
            results = self.store_manager.search_objects(normalized, top_k)
            logger.info("Object search returned %d results for embedding query", len(results))
            return results
        except Exception as exc:
            logger.error("Object search failed for embedding: %s", exc)
            return []

    def search_by_label(self, label_query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Search objects by label string (e.g., 'apple').
        """
        try:
            logger.info("Object label search for '%s'", label_query)
            results = self.store_manager.search_objects(label_query, top_k)
            logger.info("Object label search returned %d results for '%s'", len(results), label_query)
            return results
        except Exception as exc:
            logger.error("Object label search failed for '%s': %s", label_query, exc)
            return []
