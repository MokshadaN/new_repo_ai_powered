# coordnate embedding generation
"""
Centralized embedding generation manager
"""
from typing import List, Any, Union
import numpy as np
import cv2
from backend.embeddings.text_embedder import TextEmbedder
from backend.embeddings.image_embedder import ImageEmbedder
from backend.embeddings.face_embedder import FaceEmbedder
from backend.utils.logger import app_logger as logger


class EmbeddingManager:
    """Manage all embedding generation"""
    
    def __init__(self):
        logger.info("Initializing Embedding Manager")
        
        self.text_embedder = TextEmbedder()
        self.image_embedder = ImageEmbedder()
        self.face_embedder = FaceEmbedder()
        
        logger.info("All embedders initialized")
    
    def generate_text_embedding(self, text: str) -> List[float]:
        """Generate text embedding"""
        return self.text_embedder.generate_embedding(text)

    def generate_query_embedding(self, text: str) -> dict:
        """Generate an embedding specifically for short query text.

        Returns a dict compatible with existing pipeline handling:
        {"embeddings": [...], "texts": [...]}
        """
        return self.text_embedder.generate_query_embedding(text)
    
    # def generate_text_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
    #     """Generate batch of text embeddings"""
    #     return self.text_embedder.generate_batch_embeddings(texts)
    
    def generate_image_embedding(self, image_data: Any) -> List[float]:
        """Generate image embedding"""
        return self.image_embedder.generate_embedding(image_data)
    
    def generate_image_embeddings(self, images: List[Any]) -> List[List[float]]:
        """Generate batch of image embeddings"""
        return self.image_embedder.generate_batch_embeddings(images)
    
    def generate_image_embedding_from_bytes(self, image_bytes: bytes) -> List[float]:
        """Generate embedding from image bytes"""
        return self.image_embedder.generate_embedding_from_bytes(image_bytes)
    
    def generate_face_embedding(self, face_data: Any) -> List[float]:
        """Generate face embedding"""
        return self.face_embedder.generate_embedding(face_data)

    def generate_face_embedding_from_bytes(self, image_bytes: bytes) -> List[float]:
        """Generate face embedding from raw image bytes."""
        try:
            array = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(array, cv2.IMREAD_COLOR)
            if image is None:
                logger.warning("Failed to decode image bytes for face embedding")
                return self.face_embedder.generate_embedding(None)
            return self.face_embedder.generate_embedding(image)
        except Exception as exc:
            logger.error("Error generating face embedding from bytes: %s", exc)
            return self.face_embedder.generate_embedding(None)
    
    def generate_face_embeddings(self, faces: List[Any]) -> List[List[float]]:
        """Generate batch of face embeddings"""
        return [self.face_embedder.generate_embedding(face) for face in faces]
    
    def generate_object_embeddings(self, objects: List[Any]) -> List[List[float]]:
        """Generate object embeddings (using image embedder)"""
        return [self.image_embedder.generate_embedding(obj) for obj in objects]
