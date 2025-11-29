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
    """FAISS-based vector store with ID tracking and delete support."""

    def __init__(self, store_name: str, dimension: int):
        self.store_name = store_name
        self.dimension = dimension
        self.index = None  # IDMap over FlatIP
        self.metadata_map: Dict[int, Dict] = {}  # id -> metadata
        self.path_id_map: Dict[str, List[int]] = {}  # path -> [ids]
        self.next_id: int = 0

        # Paths
        self.index_path = settings.faiss_path / f"{store_name}.index"
        self.metadata_path = settings.faiss_path / f"{store_name}_metadata.pkl"

        # Initialize or load
        self._initialize_store()

        logger.info(
            "FAISS store '%s' ready with %d vectors (next_id=%d)",
            store_name,
            self.get_count(),
            self.next_id,
        )

    def _initialize_store(self):
        """Initialize or load FAISS index + metadata/ID mapping."""
        if self.index_path.exists():
            try:
                self.index = faiss.read_index(str(self.index_path))
                # Ensure we have an IDMap wrapper
                if not isinstance(self.index, faiss.IndexIDMap):
                    self.index = faiss.IndexIDMap(self.index)
                if self.metadata_path.exists():
                    with open(self.metadata_path, "rb") as f:
                        data = pickle.load(f)
                        if isinstance(data, dict) and "metadata_map" in data:
                            self.metadata_map = data.get("metadata_map", {})
                            self.path_id_map = data.get("path_id_map", {})
                            self.next_id = data.get("next_id", 0)
                        else:
                            # Backward compatibility: old metadata list
                            self.metadata_map = {idx: meta for idx, meta in enumerate(data)}
                            self.path_id_map = {}
                            for idx, meta in enumerate(data):
                                p = meta.get("file_path") or meta.get("image") or meta.get("path")
                                if not p:
                                    continue
                                self.path_id_map.setdefault(str(p), []).append(idx)
                            self.next_id = len(self.metadata_map)
                logger.info("Loaded existing FAISS index: %s", self.store_name)
            except Exception as e:
                logger.error("Error loading FAISS index: %s", e)
                self._create_new_index()
        else:
            self._create_new_index()

    def _create_new_index(self):
        """Create new FAISS index with ID mapping."""
        base = faiss.IndexFlatIP(self.dimension)
        self.index = faiss.IndexIDMap(base)
        self.metadata_map = {}
        self.path_id_map = {}
        self.next_id = 0
        logger.info("Created new FAISS index: %s", self.store_name)

    def _validate_embeddings(self, embeddings: List[List[float]]) -> bool:
        """Ensure all embeddings have the correct dimension."""
        for emb in embeddings:
            if len(emb) != self.dimension:
                logger.error(
                    "Embedding dimension mismatch: expected %d, got %d",
                    self.dimension,
                    len(emb),
                )
                return False
        return True

    def store(self, embeddings: List[List[float]], metadata: List[Dict]) -> bool:
        """Store embeddings with metadata and track IDs per path."""
        if not self._validate_embeddings(embeddings):
            return False
        if len(embeddings) != len(metadata):
            logger.error("Embeddings and metadata length mismatch")
            return False

        try:
            vectors = np.array(embeddings, dtype=np.float32)
            faiss.normalize_L2(vectors)

            ids = np.arange(self.next_id, self.next_id + len(embeddings), dtype=np.int64)
            self.index.add_with_ids(vectors, ids)

            for idx, meta in zip(ids, metadata):
                path = meta.get("file_path") or meta.get("image") or meta.get("path") or ""
                path = str(path)
                self.metadata_map[int(idx)] = meta
                if path:
                    self.path_id_map.setdefault(path, []).append(int(idx))
                logger.info(f"Storing in FAISS ({self.store_name}): {path} (id={idx})")

            self.next_id += len(embeddings)
            self._save()
            logger.debug("Stored %d vectors in %s", len(embeddings), self.store_name)
            return True
        except Exception as e:
            logger.error("Error storing in FAISS: %s", e)
            return False

    def delete(self, path: str) -> bool:
        """Delete all embeddings/metadata associated with a path."""
        key = str(path)
        ids = self.path_id_map.get(key, [])
        if not ids:
            logger.info("No embeddings to delete for path: %s", key)
            return False
        try:
            id_array = np.array(ids, dtype=np.int64)
            self.index.remove_ids(id_array)
            for i in ids:
                self.metadata_map.pop(i, None)
            self.path_id_map.pop(key, None)
            self._save()
            logger.info(f"Deleted {len(ids)} embeddings for path {key} from {self.store_name}")
            return True
        except Exception as e:
            logger.error("Error deleting from FAISS (%s): %s", self.store_name, e)
            return False

    def search(self, query_embedding: List[float], top_k: int = 10) -> List[Dict]:
        """Search for similar vectors."""
        if len(query_embedding) != self.dimension:
            logger.error(
                "Query embedding dimension mismatch: expected %d, got %d",
                self.dimension,
                len(query_embedding),
            )
            return []
        if self.index is None or self.index.ntotal == 0:
            return []
        try:
            query_vector = np.array([query_embedding], dtype=np.float32)
            faiss.normalize_L2(query_vector)

            k = min(top_k, self.index.ntotal)
            distances, indices = self.index.search(query_vector, k)

            results = []
            for distance, idx in zip(distances[0], indices[0]):
                meta = self.metadata_map.get(int(idx))
                if meta is None:
                    continue
                res = meta.copy()
                res["similarity"] = float(distance)
                res["distance"] = float(1 - distance)
                results.append(res)
            logger.info(f"FAISS search returned {len(results)} results (requested {top_k})")
            return results
        except Exception as e:
            logger.error("Error searching FAISS: %s", e)
            return []

    def _save(self):
        """Save index and metadata/id maps."""
        try:
            faiss.write_index(self.index, str(self.index_path))
            data = {
                "metadata_map": self.metadata_map,
                "path_id_map": self.path_id_map,
                "next_id": self.next_id,
            }
            with open(self.metadata_path, "wb") as f:
                pickle.dump(data, f)
        except Exception as e:
            logger.error("Error saving FAISS index: %s", e)

    def get_count(self) -> int:
        return self.index.ntotal if self.index else 0

    def clear(self):
        self._create_new_index()
        self._save()
