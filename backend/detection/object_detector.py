"""
Object detection using YOLO or similar
"""
from typing import List, Dict, Any, Any
import cv2
import numpy as np
from backend.utils.logger import app_logger as logger

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    logger.warning("YOLO not available")
    YOLO_AVAILABLE = False


class ObjectDetector:
    """Detect and classify objects in images"""
    
    def __init__(self, confidence_threshold: float = 0.5, model_size: str = 'yolov8n'):
        logger.info("Initializing Object Detector")
        self.confidence_threshold = confidence_threshold
        self.model = None
        
        if YOLO_AVAILABLE:
            try:
                self.model = YOLO(f'{model_size}.pt')
                logger.info(f"YOLO model {model_size} loaded")
            except Exception as e:
                logger.error(f"Failed to load YOLO: {e}")
    
    def detect(self, images: List[Any]) -> List[Dict]:
        """Detect objects in multiple images"""
        all_objects = []
        
        for image in images:
            objects = self._detect_in_image(image)
            all_objects.extend(objects)
        
        logger.info(f"Detected {len(all_objects)} objects across {len(images)} images")
        return all_objects
    
    def _detect_in_image(self, image: Any) -> List[Dict]:
        """Detect objects in a single image"""
        try:
            # Handle different input types
            if isinstance(image, dict):
                image_path = image.get('path')
            elif isinstance(image, str):
                image_path = image
            else:
                return []
            
            if not self.model:
                return self._mock_detection(image_path)
            
            # Perform detection
            results = self.model(image_path, verbose=False)
            
            detected_objects = []
            for result in results:
                boxes = result.boxes
                
                for box in boxes:
                    confidence = float(box.conf[0])
                    
                    if confidence >= self.confidence_threshold:
                        cls = int(box.cls[0])
                        label = result.names[cls]
                        bbox = box.xyxy[0].cpu().numpy().tolist()
                        
                        detected_objects.append({
                            'image': image_path,
                            'label': label,
                            'confidence': confidence,
                            'bbox': bbox,
                            'class_id': cls
                        })
            
            return detected_objects
            
        except Exception as e:
            logger.error(f"Error detecting objects: {e}")
            return []
    
    def _mock_detection(self, image_path: str) -> List[Dict]:
        """Mock object detection"""
        return [{
            'image': image_path,
            'label': 'unknown_object',
            'confidence': 0.85,
            'bbox': [0, 0, 100, 100],
            'class_id': 0
        }]
    
    def get_object_crops(self, detections: List[Dict]) -> List[np.ndarray]:
        """Extract cropped object images"""
        crops = []
        
        for detection in detections:
            try:
                image_path = detection['image']
                bbox = detection['bbox']
                
                img = cv2.imread(image_path)
                x1, y1, x2, y2 = map(int, bbox)
                
                crop = img[y1:y2, x1:x2]
                crops.append(crop)
                
            except Exception as e:
                logger.error(f"Error cropping object: {e}")
        
        return crops