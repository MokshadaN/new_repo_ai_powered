"""
Face-based search functionality
"""
from typing import List, Dict, Any, Optional
import numpy as np
from backend.vector_store.store_manager import VectorStoreManager
from backend.embeddings.embedding_manager import EmbeddingManager
from backend.detection.face_detector import FaceDetector
from backend.utils.logger import app_logger as logger


class FaceSearch:
    """Search for images by face"""

    def __init__(self):
        logger.info("Initializing FaceSearch")
        self.store_manager = VectorStoreManager()
        self.embedding_manager = EmbeddingManager()
        self.face_detector = FaceDetector()
        logger.info("FaceSearch ready")

    def search_by_face_image(self, face_image_path: str, top_k: int = 20) -> List[Dict]:
        """Search for similar faces using a reference face image path."""
        try:
            logger.info("Searching faces for: %s", face_image_path)

            embedding = self._get_face_embedding(face_image_path)
            if embedding is None:
                logger.warning("No face embedding could be generated")
                return []

            # Ensure unit-norm embedding
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = (np.asarray(embedding, dtype=np.float32) / norm).tolist()

            results = self.store_manager.search_faces(embedding, top_k)
            return self._process_search_results(results)
        except Exception as e:
            logger.error("Error in face search: %s", e)
            return []

    def _get_face_embedding(self, image_path: str) -> Optional[List[float]]:
        """
        Generate a face embedding.
        Tries direct embedding first; falls back to detection-assisted embedding.
        """
        # Try direct
        try:
            direct = self.embedding_manager.generate_face_embedding(image_path)
            if direct and not all(x == 0 for x in direct):
                logger.info("Direct face embedding succeeded")
                return direct
        except Exception as e:
            logger.warning("Direct face embedding failed: %s", e)

        # Fallback: detect face, then embed the first detection
        try:
            detections = self.face_detector.detect([image_path])
            if not detections or not detections[0]:
                logger.warning("No faces detected in image for embedding")
                return None

            face_data = detections[0]
            detected = self.embedding_manager.generate_face_embedding(face_data)
            if detected and not all(x == 0 for x in detected):
                logger.info("Detection-based face embedding succeeded")
                return detected
        except Exception as e:
            logger.warning("Detection-based face embedding failed: %s", e)

        return None

    def _process_search_results(self, results: List[Dict]) -> List[Dict]:
        """Normalize and sort search results by confidence."""
        processed: List[Dict[str, Any]] = []
        for res in results or []:
            path = res.get("image") or res.get("file_path")
            if not path:
                continue

            similarity = res.get("similarity", 0)
            confidence = res.get("confidence", similarity)
            # Clamp confidence to [0,1]
            confidence = max(0.0, min(1.0, confidence))

            processed.append(
                {
                    "image": path,
                    "confidence": confidence,
                    "similarity": similarity,
                    "bbox": res.get("bbox", []),
                    "metadata": res.get("metadata", {}),
                }
            )

        processed.sort(key=lambda r: r.get("confidence", 0), reverse=True)
        return processed

    def search_by_person_name(self, person_name: str, top_k: int = 20) -> List[Dict]:
        """
        Placeholder for searching faces by a stored person name.
        Currently returns empty; extend when name metadata is available.
        """
        logger.info("Searching for person by name is not implemented: %s", person_name)
        return []

    def validate_face_store(self) -> Dict[str, Any]:
        """Basic stats on the face vector store, if available."""
        info: Dict[str, Any] = {"store_available": False, "total_faces": 0}
        try:
            face_store = getattr(self.store_manager, "faiss_faces", None)
            if face_store:
                info["store_available"] = True
                info["total_faces"] = getattr(face_store, "index", None).ntotal if hasattr(face_store, "index") else 0
        except Exception as e:
            logger.error("Error validating face store: %s", e)
            info["error"] = str(e)
        return info
