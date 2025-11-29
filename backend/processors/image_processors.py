# Process Images
"""
Image processing and preparation
"""
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, Any
from backend.utils.logger import app_logger as logger


@dataclass
class ProcessingResult:
    """Result container"""
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None


class ImageProcessor:
    """Process and prepare images for embedding"""
    
    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        self.target_size = target_size
    
    def process(self, image_path: str) -> ProcessingResult:
        """Process image file"""
        try:
            # Load image
            image = self._load_image(image_path)
            
            # Preprocess
            processed = self._preprocess_image(image)
            
            return ProcessingResult(
                True,
                data={
                    'path': image_path,
                    'array': processed,
                    'original': image
                }
            )
            
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            return ProcessingResult(False, error=str(e))
    
    def _load_image(self, path: str) -> np.ndarray:
        """Load image using OpenCV"""
        image = cv2.imread(path)
        if image is None:
            raise ValueError(f"Failed to load image: {path}")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for embedding"""
        # Resize
        resized = cv2.resize(image, self.target_size, interpolation=cv2.INTER_LANCZOS4)
        
        # Normalize to [0, 1]
        normalized = resized.astype(np.float32) / 255.0
        
        return normalized
    
    def process_pil(self, image: Image.Image) -> np.ndarray:
        """Process PIL Image"""
        # Convert to numpy array
        image_array = np.array(image)
        
        # Convert to RGB if needed
        if len(image_array.shape) == 2:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
        elif image_array.shape[2] == 4:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)
        
        return self._preprocess_image(image_array)
    
    def extract_face_region(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """Extract face region from image"""
        x1, y1, x2, y2 = bbox
        face = image[y1:y2, x1:x2]
        
        # Resize to standard size
        face_resized = cv2.resize(face, (112, 112))
        
        return face_resized
    
    def augment_image(self, image: np.ndarray) -> np.ndarray:
        """Apply data augmentation"""
        # Random horizontal flip
        if np.random.random() > 0.5:
            image = cv2.flip(image, 1)
        
        # Random brightness adjustment
        brightness_factor = np.random.uniform(0.8, 1.2)
        image = np.clip(image * brightness_factor, 0, 1)
        
        return image