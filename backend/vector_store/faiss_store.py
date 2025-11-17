# """
# FAISS vector store implementation
# """
# from typing import List, Dict, Any, Optional
# import numpy as np
# import faiss
# import pickle
# from pathlib import Path
# from backend.config.settings import settings
# from backend.utils.logger import app_logger as logger


# class FAISSStore:
#     """FAISS-based vector store"""
    
#     def __init__(self, store_name: str, dimension: int):
#         self.store_name = store_name
#         self.dimension = dimension
#         self.index = None
#         self.metadata_store = []
        
#         # Paths
#         self.index_path = settings.faiss_path / f"{store_name}.index"
#         self.metadata_path = settings.faiss_path / f"{store_name}_metadata.pkl"
        
#         # Initialize or load
#         self._initialize_store()
        
#         logger.info(f"FAISS store '{store_name}' ready with {self.get_count()} vectors")
    
#     def _initialize_store(self):
#         """Initialize or load FAISS index"""
#         if self.index_path.exists():
#             # Load existing index
#             try:
#                 self.index = faiss.read_index(str(self.index_path))
                
#                 if self.metadata_path.exists():
#                     with open(self.metadata_path, 'rb') as f:
#                         self.metadata_store = pickle.load(f)
                
#                 logger.info(f"Loaded existing FAISS index: {self.store_name}")
#             except Exception as e:
#                 logger.error(f"Error loading FAISS index: {e}")
#                 self._create_new_index()
#         else:
#             self._create_new_index()
    
#     def _create_new_index(self):
#         """Create new FAISS index"""
#         # Use IndexFlatIP for inner product (cosine similarity with normalized vectors)
#         self.index = faiss.IndexFlatIP(self.dimension)
#         self.metadata_store = []
#         logger.info(f"Created new FAISS index: {self.store_name}")
    
#     def store(self, embeddings: List[List[float]], metadata: List[Dict]) -> bool:
#         """Store embeddings with metadata"""
#         try:
#             # Convert to numpy array
#             vectors = np.array(embeddings, dtype=np.float32)
            
#             # Normalize vectors for cosine similarity
#             faiss.normalize_L2(vectors)
            
#             # Add to index
#             self.index.add(vectors)
            
#             # Store metadata
#             self.metadata_store.extend(metadata)
            
#             # Save to disk
#             self._save()
            
#             logger.debug(f"Stored {len(embeddings)} vectors in {self.store_name}")
#             return True
            
#         except Exception as e:
#             logger.error(f"Error storing in FAISS: {e}")
#             return False
    
#     def search(self, query_embedding: List[float], top_k: int = 10) -> List[Dict]:
#         """Search for similar vectors"""
#         try:
#             if self.index.ntotal == 0:
#                 return []
            
#             # Convert query to numpy array
#             query_vector = np.array([query_embedding], dtype=np.float32)
            
#             # Normalize
#             faiss.normalize_L2(query_vector)
            
#             # Search
#             k = min(top_k, self.index.ntotal)
#             distances, indices = self.index.search(query_vector, k)
            
#             # Format results
#             results = []
#             for distance, idx in zip(distances[0], indices[0]):
#                 if idx < len(self.metadata_store):
#                     result = self.metadata_store[idx].copy()
#                     result['similarity'] = float(distance)
#                     result['distance'] = float(1 - distance)
#                     results.append(result)
            
#             return results
            
#         except Exception as e:
#             logger.error(f"Error searching FAISS: {e}")
#             return []
    
#     def _save(self):
#         """Save index and metadata to disk"""
#         try:
#             # Save index
#             faiss.write_index(self.index, str(self.index_path))
            
#             # Save metadata
#             with open(self.metadata_path, 'wb') as f:
#                 pickle.dump(self.metadata_store, f)
                
#         except Exception as e:
#             logger.error(f"Error saving FAISS index: {e}")
    
#     def get_count(self) -> int:
#         """Get number of vectors"""
#         return self.index.ntotal if self.index else 0
    
#     def clear(self):
#         """Clear the index"""
#         self._create_new_index()
#         self._save()
"""
FAISS vector store implementation
"""
from typing import List, Dict
import numpy as np
import faiss
import pickle
from backend.config.settings import settings
from backend.utils.logger import app_logger as logger


class FAISSStore:
    """FAISS-based vector store with dimension validation"""

    def __init__(self, store_name: str, dimension: int):
        self.store_name = store_name
        self.dimension = dimension
        self.index = None
        self.metadata_store: List[Dict] = []

        # Paths
        self.index_path = settings.faiss_path / f"{store_name}.index"
        self.metadata_path = settings.faiss_path / f"{store_name}_metadata.pkl"

        # Initialize or load
        self._initialize_store()

        logger.info(f"FAISS store '{store_name}' ready with {self.get_count()} vectors")

    def _initialize_store(self):
        """Initialize or load FAISS index"""
        if self.index_path.exists():
            try:
                self.index = faiss.read_index(str(self.index_path))
                if self.metadata_path.exists():
                    with open(self.metadata_path, 'rb') as f:
                        self.metadata_store = pickle.load(f)
                logger.info(f"Loaded existing FAISS index: {self.store_name}")
            except Exception as e:
                logger.error(f"Error loading FAISS index: {e}")
                self._create_new_index()
        else:
            self._create_new_index()

    def _create_new_index(self):
        """Create new FAISS index"""
        # IndexFlatIP with normalized vectors for cosine similarity
        self.index = faiss.IndexFlatIP(self.dimension)
        self.metadata_store = []
        logger.info(f"Created new FAISS index: {self.store_name}")

    def _validate_embeddings(self, embeddings: List[List[float]]) -> bool:
        """Ensure all embeddings have the correct dimension"""
        for emb in embeddings:
            if len(emb) != self.dimension:
                logger.error(
                    f"Embedding dimension mismatch: expected {self.dimension}, got {len(emb)}"
                )
                return False
        return True

    def store(self, embeddings: List[List[float]], metadata: List[Dict]) -> bool:
        """Store embeddings with metadata"""
        if not self._validate_embeddings(embeddings):
            return False

        try:
            # Log file names being stored
            print("Metdata Type", type(metadata))
            # print(metadata[0])
            print("In FAISS store")
            for meta in metadata:
                file_path = meta.get('file_path', 'unknown')
                print(f"Storing in FAISS ({self.store_name}): {file_path}")
                logger.info(f"Storing in FAISS ({self.store_name}): {file_path}")
            
            vectors = np.array(embeddings, dtype=np.float32)
            faiss.normalize_L2(vectors)
            self.index.add(vectors)

            # Store metadata
            self.metadata_store.extend(metadata)

            # Save index and metadata
            self._save()

            logger.debug(f"Stored {len(embeddings)} vectors in {self.store_name}")
            return True

        except Exception as e:
            logger.error(f"Error storing in FAISS: {e}")
            return False

    def search(self, query_embedding: List[float], top_k: int = 10) -> List[Dict]:
        """Search for similar vectors"""
        if len(query_embedding) != self.dimension:
            logger.error(
                f"Query embedding dimension mismatch: expected {self.dimension}, got {len(query_embedding)}"
            )
            return []

        if self.index.ntotal == 0:
            return []

        try:
            query_vector = np.array([query_embedding], dtype=np.float32)
            faiss.normalize_L2(query_vector)

            k = min(top_k, self.index.ntotal)
            distances, indices = self.index.search(query_vector, k)

            results = []
            for distance, idx in zip(distances[0], indices[0]):
                if idx < len(self.metadata_store):
                    res = self.metadata_store[idx].copy()
                    res['similarity'] = float(distance)
                    res['distance'] = float(1 - distance)
                    results.append(res)
            print("Faiss result")
            return results

        except Exception as e:
            logger.error(f"Error searching FAISS: {e}")
            return []

    def _save(self):
        """Save index and metadata"""
        try:
            faiss.write_index(self.index, str(self.index_path))
            with open(self.metadata_path, 'wb') as f:
                pickle.dump(self.metadata_store, f)
        except Exception as e:
            logger.error(f"Error saving FAISS index: {e}")

    def get_count(self) -> int:
        """Get number of vectors"""
        return self.index.ntotal if self.index else 0

    def clear(self):
        """Clear the index"""
        self._create_new_index()
        self._save()
