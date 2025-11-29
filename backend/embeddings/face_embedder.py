"""
Face embedding reuse/fallback utilities.
"""
from typing import Any, List, Optional

import cv2
import numpy as np
from backend.config.model_config import ModelRegistry
from backend.utils.logger import app_logger as logger

try:
    from insightface.app import FaceAnalysis

    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    logger.warning("InsightFace not available; embeddings will fall back to zeros.")


class FaceEmbedder:
    """
    Provides a consistent interface for fetching face embeddings.

    The detector already attaches InsightFace embeddings to each detection.
    This class first reuses that cached vector and only falls back to running
    FaceAnalysis again if the embedding is missing.
    """

    def __init__(self) -> None:
        self.model_config = ModelRegistry.FACE_EMBEDDING
        self.app: Optional[FaceAnalysis] = None

        if not INSIGHTFACE_AVAILABLE:
            return

        try:
            self.app = FaceAnalysis(
                name="buffalo_l",
                providers=["CPUExecutionProvider"],
                allowed_modules=["recognition", "detection"],
            )
            self.app.prepare(ctx_id=-1)  # CPU only
            logger.info("✅ InsightFace embedding backend ready (fallback mode).")
        except Exception as exc:  # pragma: no cover
            self.app = None
            logger.warning("Failed to initialize InsightFace fallback: %s", exc)

    def generate_embedding(self, face_data: Any) -> List[float]:
        """
        Return a normalized embedding for the supplied face data.

        Preference order:
            1. Cached embedding inside the detection dict.
            2. Recompute using InsightFace if available.
            3. Zero vector fallback.
        """
        try:
            cached = self._extract_cached_embedding(face_data)
            if cached is not None:
                return cached

            if self.app is None:
                logger.warning("InsightFace fallback unavailable; returning zero embedding.")
                return self._get_zero_embedding()

            face_img = self._preprocess_face_data(face_data)
            if face_img is None:
                return self._get_zero_embedding()

            faces = self.app.get(face_img)
            if not faces:
                logger.warning("No face visible while recomputing embedding.")
                return self._get_zero_embedding()

            embedding = getattr(faces[0], "normed_embedding", None)
            if embedding is None:
                logger.warning("InsightFace produced no embedding; returning zeros.")
                return self._get_zero_embedding()

            return np.asarray(embedding, dtype=np.float32).tolist()

        except Exception as exc:  # pragma: no cover
            logger.error("Error generating face embedding: %s", exc)
            return self._get_zero_embedding()

    def _extract_cached_embedding(self, face_data: Any) -> Optional[List[float]]:
        """Return the embedding already computed by the detector, if present."""
        if isinstance(face_data, dict):
            embedding = face_data.get("embedding")
            if embedding is None:
                return None

            vector = np.asarray(embedding, dtype=np.float32)
            if vector.size == 0:
                return None

            norm = np.linalg.norm(vector)
            if norm == 0:
                return self._get_zero_embedding()

            return (vector / norm).tolist()

        return None

    def _preprocess_face_data(self, face_data: Any) -> Optional[np.ndarray]:
        """
        Convert various input formats (dict with bbox, path, ndarray) into a BGR crop.
        """
        try:
            if isinstance(face_data, dict):
                image_path = face_data.get("image")
                bbox = face_data.get("bbox")

                if isinstance(image_path, str):
                    image = cv2.imread(image_path)
                    if image is None:
                        logger.error("Failed to load image: %s", image_path)
                        return None

                    if bbox:
                        x1, y1, x2, y2 = map(int, bbox)
                        h, w = image.shape[:2]
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w, x2), min(h, y2)
                        if x2 > x1 and y2 > y1:
                            return image[y1:y2, x1:x2]
                    return image

            if isinstance(face_data, str):
                image = cv2.imread(face_data)
                if image is None:
                    logger.error("Failed to load face image from path: %s", face_data)
                    return None
                return image

            if isinstance(face_data, np.ndarray):
                return face_data

            logger.error("Unsupported face data type: %s", type(face_data))
            return None

        except Exception as exc:  # pragma: no cover
            logger.error("Error preprocessing face data: %s", exc)
            return None

    def _get_zero_embedding(self) -> List[float]:
        """Return a zero vector with the configured embedding dimension."""
        return [0.0] * self.model_config.dimension
