
# """
# Face detection using OpenCV fallback (no InsightFace/RetinaFace required)
# """
# from typing import List, Dict, Any
# import cv2
# import numpy as np
# from pathlib import Path
# from backend.config.settings import settings
# from backend.utils.logger import app_logger as logger

# # ❌ Disable RetinaFace completely
# RETINAFACE_AVAILABLE = False  

# # Load OpenCV HaarCascade model (ships with OpenCV)
# CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
# face_cascade = cv2.CascadeClassifier(CASCADE_PATH)


# class FaceDetector:
#     """Detect faces in images using OpenCV HaarCascade"""

#     def __init__(self, confidence_threshold: float = 0.6):
#         logger.info("Initializing Face Detector (OpenCV fallback)")
#         self.confidence_threshold = confidence_threshold
#         self.available = RETINAFACE_AVAILABLE  # will always be False now

#     def detect(self, images: List[Any]) -> List[Dict]:
#         """Detect faces in multiple images"""
#         all_faces = []
#         print("Face 1")
#         for image in images:
#             faces = self._detect_in_image(image)
#             all_faces.extend(faces)

#         print("Face 2")
#         logger.info(f"Detected {len(all_faces)} faces across {len(images)} images")
#         return all_faces

#     def _detect_in_image(self, image: Any) -> List[Dict]:
#         """Detect faces in a single image"""

#         try:
#             # Normalize input
#             if isinstance(image, dict):
#                 image_path = image.get('path')
#             elif isinstance(image, str):
#                 image_path = image
#             else:
#                 return []
#             print("Image face 1")
#             # Always use OpenCV fallback
#             print("Image face 2")
#             return self._opencv_fallback(image_path)

#         except Exception as e:
#             logger.error(f"Error detecting faces: {e}")
#             return []

#     def _opencv_fallback(self, image_path: str) -> List[Dict]:
#         """OpenCV HaarCascade fallback detection"""

#         img = cv2.imread(image_path)
#         if img is None:
#             return []

#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#         # Detect faces
#         faces = face_cascade.detectMultiScale(
#             gray,
#             scaleFactor=1.1,
#             minNeighbors=5,
#             minSize=(40, 40)
#         )
#         print("Faces found:", len(faces))
#         detections = []
#         for (x, y, w, h) in faces:
#             detections.append({
#                 'image': image_path,
#                 'bbox': [x, y, x + w, y + h],
#                 'confidence': 0.95,   # HaarCascade has no score → fixed
#                 'landmarks': {}
#             })

#         return detections

#     def extract_face_images(self, detections: List[Dict]) -> List[np.ndarray]:
#         """Extract face regions from detections"""
#         face_images = []

#         for detection in detections:
#             try:
#                 image_path = detection['image']
#                 bbox = detection['bbox']

#                 img = cv2.imread(image_path)
#                 x1, y1, x2, y2 = bbox

#                 face_img = img[y1:y2, x1:x2]
#                 face_images.append(face_img)

#             except Exception as e:
#                 logger.error(f"Error extracting face: {e}")
#         print("Faces extracted",len(face_images))
#         return face_images
"""
Face detection using InsightFace (superior accuracy)
"""
from typing import Optional, List, Dict, Any
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
    
    def __init__(self, confidence_threshold: float = 0.6):
        logger.info("Initializing Face Detector (InsightFace)")
        self.confidence_threshold = confidence_threshold
        self.app: Optional[FaceAnalysis] = None
        
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
            f"Detected {len(all_faces)} faces across {len(images)} images",
        )
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
            for face in faces:
                confidence = float(getattr(face, "det_score", 0.0))
                if confidence < self.confidence_threshold:
                    continue

                bbox = face.bbox.astype(int).tolist() if hasattr(face, "bbox") else []
                landmarks = face.kps.tolist() if hasattr(face, "kps") else []
                embedding = face.normed_embedding.tolist() if hasattr(face, "normed_embedding") else None
                detections.append(
                    {
                        "image": image_path,
                        "bbox": bbox,
                        "confidence": confidence,
                        "landmarks": landmarks,
                        "embedding":embedding
                    }
                )
                print("Faces detected: ",len(detections))
            return detections
        except Exception as e:
            logger.error("Error detecting faces for %s: %s", image, e)
            return []

    def extract_face_images(self, detections: List[Dict]) -> List[np.ndarray]:
        """Extract face regions from detections"""
        face_images = []
        for detection in detections:
            try:
                img = cv2.imread(detection['image'])
                bbox = detection['bbox']
                x1, y1, x2, y2 = map(int, bbox)
                
                # Bounds checking
                h, w = img.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                if x2 > x1 and y2 > y1:
                    face_images.append(img[y1:y2, x1:x2])
            except Exception as e:
                logger.error(f"Error extracting face: {e}")
        
        return face_images