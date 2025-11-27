"""
Semantic search implementation
"""
from typing import List, Dict, Any, Optional
from backend.vector_store.store_manager import VectorStoreManager
from backend.embeddings.embedding_manager import EmbeddingManager
from backend.utils.logger import app_logger as logger
import numpy as np


class SemanticSearch:
    """Semantic search across all content types"""
    
    def __init__(self):
        self.store_manager = VectorStoreManager()
        self.embedding_manager = EmbeddingManager()
    
    def search_all(self, query: str, top_k: int = 20) -> Dict[str, List[Dict]]:
        """Search across all vector stores"""
        try:
            # Use BGE for text and SigLIP text tower for image cross-modal
            text_query_embedding = self._extract_query_vector(query, target="text")
            image_query_embedding = self._extract_query_vector(query, target="image")
            
            # Search all stores
            results = {
                'text': self.store_manager.search_text(text_query_embedding, top_k),
                'images': self.store_manager.search_images(image_query_embedding, top_k),
                'faces': self.store_manager.search_faces(image_query_embedding, top_k) if hasattr(self.store_manager, "search_faces") else [],
                'objects': self.store_manager.search_objects(image_query_embedding, top_k) if hasattr(self.store_manager, "search_objects") else [],
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            return {'text': [], 'images': [], 'faces': [], 'objects': []}
    
    def search_by_type(self, query: str, search_type: str, top_k: int = 20) -> List[Dict]:
        """Search specific content type"""
        try:
            query_embedding = self._extract_query_vector(query, target=search_type)
            
            if search_type == 'text':
                return self.store_manager.search_text(query_embedding, top_k)
            elif search_type == 'image':
                return self.store_manager.search_images(query_embedding, top_k)
            elif search_type == 'face':
                return self.store_manager.search_faces(query_embedding, top_k) if hasattr(self.store_manager, "search_faces") else []
            elif search_type == 'object':
                return self.store_manager.search_objects(query_embedding, top_k) if hasattr(self.store_manager, "search_objects") else []
            else:
                return []
                
        except Exception as e:
            logger.error(f"Error searching by type: {e}")
            return []

    def search_similar_images(self, image: Any, top_k: int = 20) -> List[Dict]:
        """
        Image-to-image search using stored image embeddings.
        """
        try:
            embedding = self.embedding_manager.generate_image_embedding(image)
            if not embedding:
                return []
            return self.store_manager.search_images(embedding, top_k)
        except Exception as e:
            logger.error(f"Error in image-to-image search: {e}")
            return []

    def _extract_query_vector(self, query: str, target: str = "text") -> List[float]:
        """
        Normalize embedder output to a single vector:
        - If the embedder returns a dict with 'embeddings', take the first vector.
        - If multiple chunks are returned, average them.
        - For image target, use SigLIP text tower to align with image embeddings.
        """
        if target == "image":
            raw = self.embedding_manager.image_embedder.generate_text_embedding(query)
            target_dim = self.store_manager.faiss_images.dimension
        else:
            raw = self.embedding_manager.generate_text_embedding(query)
            target_dim = self.store_manager.faiss_text.dimension

        # Handle dict format: {"embeddings": [...], "texts": [...]}
        if isinstance(raw, dict) and "embeddings" in raw:
            vectors = raw.get("embeddings") or []
            if not vectors:
                return [0.0] * target_dim
            # If multiple chunk embeddings, average them for a single query vector
            try:
                arr = np.asarray(vectors, dtype=np.float32)
                if arr.ndim == 1:
                    return arr.tolist()
                return arr.mean(axis=0).tolist()
            except Exception:
                return list(vectors[0])

        # If embedder already returns a vector
        if isinstance(raw, list):
            try:
                arr = np.asarray(raw, dtype=np.float32)
                if arr.ndim > 1:
                    arr = arr.flatten()
                if len(arr) != target_dim:
                    arr = self._resize_vector(arr, target_dim)
                return arr.tolist()
            except Exception:
                return [0.0] * target_dim

        # Fallback zero vector
        return [0.0] * target_dim

    def _resize_vector(self, vec: np.ndarray, target_dim: int) -> np.ndarray:
        """Pad or truncate a vector to target_dim."""
        current = len(vec)
        if current == target_dim:
            return vec
        if current > target_dim:
            return vec[:target_dim]
        padded = np.zeros(target_dim, dtype=np.float32)
        padded[:current] = vec
        return padded
