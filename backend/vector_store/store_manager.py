"""
Vector store manager - FAISS only
"""
from typing import Dict, List, Union

import numpy as np

from backend.vector_store.faiss_store import FAISSStore
from backend.config.settings import settings
from backend.utils.logger import app_logger as logger
from backend.embeddings.face_embedder import FaceEmbedder



class VectorStoreManager:
    """Manage FAISS vector stores for text, images, faces, and objects."""

    def __init__(self):
        logger.info("Initializing FAISS vector stores")

        self.faiss_text = FAISSStore("text_faiss", settings.embedding_dimension)  # 1024
        self.faiss_images = FAISSStore("image_faiss", settings.image_embedding_dimension)  # 768
        self.faiss_faces = FAISSStore("face_faiss", settings.face_embedding_dimension)  # 512
        self.faiss_objects = FAISSStore("object_faiss", settings.object_embedding_dimension)  # 768
        self.face_embedder=FaceEmbedder()
        logger.info(
            "Vector stores ready (Text: %dD, Image: %dD, Face: %dD, Object: %dD)",
            settings.embedding_dimension,
            settings.image_embedding_dimension,
            settings.face_embedding_dimension,
            settings.object_embedding_dimension,
        )

    # Helpers
    def ensure_correct_dimension(
        self, embedding: Union[List[float], List[List[float]]], expected_dim: int
    ) -> Union[List[float], List[List[float]]]:
        """
        Ensure embedding(s) match expected dimension by truncating or padding.
        Accepts a single embedding or a list of embeddings.
        """
        is_single = isinstance(embedding[0], (float, int))
        embeddings = [embedding] if is_single else embedding

        fixed: List[List[float]] = []
        for emb in embeddings:
            arr = np.array(emb).flatten()
            current_dim = len(arr)
            if current_dim == expected_dim:
                fixed.append(arr.tolist())
            elif current_dim > expected_dim:
                logger.warning("Embedding truncated from %d to %d", current_dim, expected_dim)
                fixed.append(arr[:expected_dim].tolist())
            else:
                logger.warning("Embedding padded from %d to %d", current_dim, expected_dim)
                padded = np.zeros(expected_dim)
                padded[:current_dim] = arr
                fixed.append(padded.tolist())

        return fixed[0] if is_single else fixed

    def store_text(self, embedding: List[float], chunks: List[str], file_path: str) -> bool:
        """
        Store text embeddings (per chunk) with chunk-specific metadata.
        `embedding` can be a single vector or a list of vectors aligned to `chunks`.
        """
        logger.info(f"Storing text embedding(s) for: {file_path}")

        is_single = isinstance(embedding[0], (float, int))
        embeddings = [embedding] if is_single else embedding
        chunk_texts = chunks or []

        fixed_embeddings = [
            self.ensure_correct_dimension(e, settings.embedding_dimension)
            for e in embeddings
        ]

        total_chunks = len(fixed_embeddings)
        metadatas: List[Dict] = []
        for idx, emb in enumerate(fixed_embeddings):
            text = chunk_texts[idx] if idx < len(chunk_texts) else f"Chunk {idx + 1}"
            metadatas.append({
                "file_path": file_path,
                "type": "text",
                "chunk_id": idx,
                "total_chunks": total_chunks,
                "chunk_content": text,
                "content_preview": text,
                "embedding_dim": len(emb),
            })
        print("Metadata of faiss text store of length", len(chunk_texts))
        print(metadatas[0])
        success = True

        success = success and self.faiss_text.store(fixed_embeddings, metadatas)
        logger.info(f"Stored {len(fixed_embeddings)} chunks in FAISS")

        return success

    # def search_text(self, query_embedding: List[float], top_k: int | None = None) -> List[Dict]:
    #     query_embedding = self.ensure_correct_dimension(query_embedding, settings.embedding_dimension)
    #     if isinstance(query_embedding, list) and isinstance(query_embedding[0], list):
    #         query_embedding = query_embedding[0]

    #     top_k = top_k or settings.top_k_results
    #     return self.faiss_text.search(query_embedding, top_k)

    # # Image operations
    def store_images(self, embeddings: List[List[float]], metadata: List[Dict]) -> bool:
        for meta in metadata:
            logger.info("Storing image embedding for: %s", meta.get("file_path", "unknown"))

        processed = [
            self.ensure_correct_dimension(e, settings.image_embedding_dimension) for e in embeddings
        ]
        return self.faiss_images.store(processed, metadata)

    # def search_images(self, query_embedding: List[float], top_k: int | None = None) -> List[Dict]:
    #     query_embedding = self.ensure_correct_dimension(
    #         query_embedding, settings.image_embedding_dimension
    #     )
    #     top_k = top_k or settings.top_k_results
    #     return self.faiss_images.search(query_embedding, top_k)

    def search_images_by_text(
        self, text_embedding: List[float], top_k: int | None = None
    ) -> List[Dict]:
        """
        Cross-modal search (text -> image).
        If a 1024D text embedding is provided, project to 768D via truncation.
        """
        text_embedding_array = np.array(text_embedding).flatten()
        if len(text_embedding_array) == settings.embedding_dimension:
            projected = text_embedding_array[: settings.image_embedding_dimension]
            logger.info("Projected text embedding from 1024D to 768D for cross-modal search")
            return self.search_images(projected.tolist(), top_k)
        return self.search_images(text_embedding, top_k)

    # # Face operations
    # def store_faces(self, embeddings: List[List[float]], metadata: List[Dict]) -> bool:
    #     for meta in metadata:
    #         logger.info("Storing face embedding for: %s", meta.get("image", "unknown"))

    #     processed = [
    #         self.ensure_correct_dimension(e, settings.face_embedding_dimension) for e in embeddings
    #     ]
    #     return self.faiss_faces.store(processed, metadata)

    def store_faces(self, detections: List[Dict]) -> bool:
        if not detections:
            logger.warning("No face detections supplied for FAISS storage")
            return False

        embeddings: List[List[float]] = []
        metadatas: List[Dict] = []

        for detection in detections:
            logger.info("Storing face embedding for: %s", detection.get("image", "unknown"))

            embedding = detection.get("embedding")
            if embedding is None:
                embedding = self.face_embedder.generate_embedding(detection)

            if not embedding:
                logger.warning("Skipping detection with no embedding: %s", detection.get("image"))
                continue

            fixed = self.ensure_correct_dimension(
                embedding,
                settings.face_embedding_dimension,
            )
            embeddings.append(fixed)
            metadatas.append(
                {
                    "image": detection.get("image"),
                    "bbox": detection.get("bbox"),
                    "confidence": detection.get("confidence"),
                    "landmarks": detection.get("landmarks"),
                }
            )

        if not embeddings:
            logger.warning("No valid face embeddings available to store")
            return False

        return self.faiss_faces.store(embeddings, metadatas)

    def search_faces(self, query_embedding: List[float], top_k: int | None = None) -> List[Dict]:
        """
        Search the face FAISS index with normalization and a basic similarity filter.
        """
        # Ensure correct shape and pick first vector if list-of-lists
        query_embedding = self.ensure_correct_dimension(
            query_embedding, settings.face_embedding_dimension
        )
        if isinstance(query_embedding, list) and isinstance(query_embedding[0], list):
            query_embedding = query_embedding[0]

        # Normalize; guard against zero vector
        vec = np.asarray(query_embedding, dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm == 0:
            logger.warning("Face search skipped: zero-norm query embedding")
            return []
        query_embedding = (vec / norm).tolist()

        # Limit to a small, relevant set
        top_k = top_k or min(settings.top_k_results or 5, 5)
        results = self.faiss_faces.search(query_embedding, top_k)

        # Filter very low similarity hits (FAISS returns cosine similarity in [0,1])
        MIN_SIMILARITY = 0.35
        filtered = [r for r in results if r.get("similarity", 0) >= MIN_SIMILARITY]

        if filtered:
            similarities = [r.get("similarity", 0) for r in filtered]
            logger.info(
                f"Face search similarity range (filtered): min={min(similarities)}, max={max(similarities)}",
            )
        else:
            logger.info(
                f"Face search returned no results above similarity threshold {MIN_SIMILARITY}",
            )
        return filtered

    # Object operations
    def store_objects(self, embeddings: List[List[float]], metadata: List[Dict]) -> bool:
        for meta in metadata:
            logger.info("Storing object embeddings for: %s", meta.get("image", "unknown"))

        processed = [
            self.ensure_correct_dimension(e, settings.object_embedding_dimension) for e in embeddings
        ]
        return self.faiss_objects.store(processed, metadata)

    # def search_objects(self, query_embedding: List[float], top_k: int | None = None) -> List[Dict]:
    #     query_embedding = self.ensure_correct_dimension(
    #         query_embedding, settings.object_embedding_dimension
    #     )
    #     top_k = top_k or settings.top_k_results
    #     return self.faiss_objects.search(query_embedding, top_k)

    # # Stats & helpers
    # def get_statistics(self) -> Dict:
    #     return {
    #         "faiss_text_count": self.faiss_text.get_count(),
    #         "faiss_image_count": self.faiss_images.get_count(),
    #         "faiss_face_count": self.faiss_faces.get_count(),
    #         "faiss_object_count": self.faiss_objects.get_count(),
    #     }

    # def get_dimension_info(self) -> Dict:
    #     return {
    #         "text_embedding_dim": settings.embedding_dimension,
    #         "image_embedding_dim": settings.image_embedding_dimension,
    #         "face_embedding_dim": settings.face_embedding_dimension,
    #         "object_embedding_dim": settings.object_embedding_dimension,
    #     }

    def check_file_exists(self, file_path: str) -> bool:
        """Check FAISS metadata stores for a given file path."""
        def contains_path(metadata_list: List[Dict]) -> bool:
            return any(meta.get("file_path") == file_path for meta in metadata_list)

        return contains_path(self.faiss_text.metadata_store) or contains_path(
            self.faiss_images.metadata_store
        )

    def search_text(self, query_embedding: list, top_k: int | None = None) -> list:
        """
        Search the text FAISS index. Accepts a single vector or a list (first vector used).
        Returns a list of metadata dicts with similarity scores.
        """
        # Ensure correct dimension and single-vector input
        query_embedding = self.ensure_correct_dimension(query_embedding, settings.embedding_dimension)
        if isinstance(query_embedding, list) and isinstance(query_embedding[0], list):
            query_embedding = query_embedding[0]

        top_k = top_k or settings.top_k_results
        return self.faiss_text.search(query_embedding, top_k)
