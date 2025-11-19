"""
Face detection using InsightFace (superior accuracy)
"""
from typing import Optional, List, Dict, Any
import uuid
import cv2
import numpy as np
from backend.utils.logger import app_logger as logger

try:
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    logger.warning("InsightFace not available")


class FaceDetector:
    """Detect faces using InsightFace"""
    
    def __init__(self, confidence_threshold: float = 0.7):
        logger.info("Initializing Face Detector (InsightFace)")
        self.confidence_threshold = confidence_threshold
        self.app: Optional[FaceAnalysis] = None
        self.embedding_store: Dict[str, np.ndarray]={}
        
        if INSIGHTFACE_AVAILABLE:
            try:
                self.app = FaceAnalysis(
                    name='buffalo_l',
                    providers=['CPUExecutionProvider']
                )
                self.app.prepare(ctx_id=-1)  # CPU mode
                logger.info("✅ InsightFace loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load InsightFace: {e}")
            
    def detect(self, images: List[Any]) -> List[Dict]:
        """Detect faces across multiple images."""
        if not images:
            logger.warning("Face detection skipped: no images supplied")
            return []

        if not self.app:
            logger.warning("Face detection skipped: InsightFace not initialized")
            return []

        all_faces: List[Dict] = []
        for image in images:
            if image in (None, ""):
                logger.warning("Skipping blank image entry in batch: %s", image)
                continue

            faces = self._detect_in_image(image)
            if faces:
                all_faces.extend(faces)

        logger.info(
            "Detected %d faces across %d images",
            len(all_faces),
            len(images),
        )
        # for face in all_faces:
        #     embedding=np.array(face['embedding'])
        #     face_id=str(uuid.uuid4())
        #     face['face_id']=face_id
        #     self.embedding_store[face_id]=embedding
        return all_faces

    def _detect_in_image(self, image: Any) -> List[Dict]:
            """Detect faces in a single image via InsightFace."""
            try:
                image_path = image.get('path') if isinstance(image, dict) else image
                if not isinstance(image_path, str) or not image_path:
                    logger.warning("Face detection skipped: invalid image reference %s", image)
                    return []

                if not self.app:
                    logger.warning("Face detection skipped: InsightFace not initialized")
                    return []

                img = cv2.imread(image_path)
                if img is None:
                    logger.warning("Could not read image: %s", image_path)
                    return []

                try:
                    faces = self.app.get(img)
                except Exception as inference_error:
                    logger.error("InsightFace inference failed for %s: %s", image_path, inference_error)
                    return []

                detections: List[Dict] = []
                i=0
                for face in faces:
                    confidence = float(getattr(face, "det_score", 0.0))
                    if confidence < self.confidence_threshold:
                        continue
                    i+=1
                    bbox = face.bbox.astype(int).tolist() if hasattr(face, "bbox") else []
                    landmarks = face.kps.tolist() if hasattr(face, "kps") else []
                    embedding = face.normed_embedding.tolist() if hasattr(face, "normed_embedding") else None
                    detections.append(
                        {
                            "image": image_path,
                            "bbox": bbox,
                            "confidence": confidence,
                            "embedding": embedding,
                            "landmarks": landmarks,
                            "metadata":{
                                "width":img.shape[1],
                                "height":img.shape[0]
                            }
                        }
                    )
                    print(f"Embedding for face {i}",len(embedding))
                print("Faces detected: ",len(detections))
                return detections
            except Exception as e:
                logger.error("Error detecting faces for %s: %s", image, e)
                return []
if __name__=="__main__":
    detector=FaceDetector()
    detections=detector.detect(["D:\\BTP\\new_ai_powered\\new_repo_ai_powered\\IMG-20250322-WA0044.jpg","D:\\BTP\\new_ai_powered\\new_repo_ai_powered\\IMG-20240921-WA0022.jpg","D:\\BTP\\new_ai_powered\\new_repo_ai_powered\\IMG-20240811-WA0030.jpg"])
    print("Detections:",len(detections))