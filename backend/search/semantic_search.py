"""
Semantic search implementation
"""
from typing import List, Dict, Any, Optional
from backend.vector_store.store_manager import VectorStoreManager
from backend.embeddings.embedding_manager import EmbeddingManager
from backend.utils.logger import app_logger as logger


class SemanticSearch:
    """Semantic search across all content types"""
    
    def __init__(self):
        self.store_manager = VectorStoreManager()
        self.embedding_manager = EmbeddingManager()
    
    def search_all(self, query: str, top_k: int = 20) -> Dict[str, List[Dict]]:
        """Search across all vector stores"""
        try:
            # Generate query embedding
            query_embedding = self.embedding_manager.generate_text_embedding(query)
            
            # Search all stores
            results = {
                'text': self.store_manager.search_text(query_embedding, top_k),
                'images': self.store_manager.search_images_by_text(query_embedding, top_k),
                'faces': self.store_manager.search_faces(query_embedding, top_k) if hasattr(self.store_manager, "search_faces") else [],
                'objects': self.store_manager.search_objects(query_embedding, top_k) if hasattr(self.store_manager, "search_objects") else [],
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            return {'text': [], 'images': [], 'faces': [], 'objects': []}
    
    def search_by_type(self, query: str, search_type: str, top_k: int = 20) -> List[Dict]:
        """Search specific content type"""
        try:
            query_embedding = self.embedding_manager.generate_text_embedding(query)
            
            if search_type == 'text':
                return self.store_manager.search_text(query_embedding, top_k)
            elif search_type == 'image':
                return self.store_manager.search_images_by_text(query_embedding, top_k)
            elif search_type == 'face':
                return self.store_manager.search_faces(query_embedding, top_k) if hasattr(self.store_manager, "search_faces") else []
            elif search_type == 'object':
                return self.store_manager.search_objects(query_embedding, top_k) if hasattr(self.store_manager, "search_objects") else []
            else:
                return []
                
        except Exception as e:
            logger.error(f"Error searching by type: {e}")
            return []
