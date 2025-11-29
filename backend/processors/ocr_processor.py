# OCR extraction
"""
OCR text extraction from images
"""
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, List
from backend.utils.logger import app_logger as logger

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    logger.warning("pytesseract not available")
    TESSERACT_AVAILABLE = False

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    logger.warning("easyocr not available")
    EASYOCR_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    logger.warning("Pillow not available for Tesseract OCR")
    PIL_AVAILABLE = False


class OCRProcessor:
    """Extract text from images using OCR"""
    
    def __init__(self, languages: List[str] = ['en'], use_gpu: bool = False):
        self.languages = languages
        self.use_gpu = use_gpu
        
        # Initialize EasyOCR if available
        self.reader = None
        if EASYOCR_AVAILABLE:
            try:
                self.reader = easyocr.Reader(languages, gpu=use_gpu)
                logger.info("EasyOCR initialized")
            except Exception as e:
                logger.error(f"Failed to initialize EasyOCR: {e}")
    
    def extract_from_file(self, file_path: str) -> Optional[str]:
        """Extract text from image file"""
        try:
            # Check if file is an image
            path = Path(file_path)
            if path.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                return None
            
            # Try EasyOCR first (generally better accuracy)
            if self.reader is not None:
                text = self._extract_with_easyocr(file_path)
                if text:
                    return text
            
            # Fallback to Tesseract
            if TESSERACT_AVAILABLE:
                text = self._extract_with_tesseract(file_path)
                if text:
                    return text
            
            return None
            
        except Exception as e:
            logger.error(f"OCR extraction failed for {file_path}: {e}")
            return None
    
    def extract_from_image(self, image: np.ndarray) -> Optional[str]:
        """Extract text from image array"""
        try:
            if self.reader is not None:
                results = self.reader.readtext(image)
                text_parts = [result[1] for result in results if result[2] > 0.5]
                return " ".join(text_parts) if text_parts else None
            
            if TESSERACT_AVAILABLE and PIL_AVAILABLE:
                # Convert to PIL Image format
                pil_image = Image.fromarray(image)
                text = pytesseract.image_to_string(pil_image)
                return text.strip() if text.strip() else None
            
            return None
            
        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            return None
    
    def _extract_with_easyocr(self, file_path: str) -> Optional[str]:
        """Extract using EasyOCR"""
        results = self.reader.readtext(file_path)
        
        # Filter by confidence
        text_parts = [result[1] for result in results if result[2] > 0.5]
        
        return " ".join(text_parts) if text_parts else None
    
    def _extract_with_tesseract(self, file_path: str) -> Optional[str]:
        """Extract using Tesseract"""
        # Load and preprocess image
        image = cv2.imread(file_path)
        if image is None:
            logger.error(f"Unable to read image for Tesseract OCR: {file_path}")
            return None
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply thresholding
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Extract text
        text = pytesseract.image_to_string(binary)
        
        return text.strip() if text.strip() else None
    
    def preprocess_for_ocr(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for better OCR"""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(gray)
        
        # Increase contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        
        # Binarize
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return binary
