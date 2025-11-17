# # ArcFace Embeddings
# """
# Face embedding generation using ArcFace
# """
# from typing import List, Any
# import numpy as np
# import cv2
# from backend.config.model_config import ModelRegistry
# from backend.utils.logger import app_logger as logger

# try:
#     from insightface.app import FaceAnalysis
#     INSIGHTFACE_AVAILABLE = True
# except ImportError:
#     logger.warning("insightface not available")
#     INSIGHTFACE_AVAILABLE = False


# class FaceEmbedder:
#     """Generate face embeddings using ArcFace"""
    
#     def __init__(self):
#         logger.info("Initializing Face Embedder")
        
#         self.model_config = ModelRegistry.FACE_EMBEDDING
#         self.app = None
        
#         if INSIGHTFACE_AVAILABLE:
#             try:
#                 self.app = FaceAnalysis(allowed_modules=['recognition'])
#                 self.app.prepare(ctx_id=0 if cv2.cuda.getCudaEnabledDeviceCount() > 0 else -1)
#                 logger.info("ArcFace model loaded")
#             except Exception as e:
#                 logger.error(f"Failed to load ArcFace: {e}")
    
#     def generate_embedding(self, face_data: Any) -> List[float]:
#         """Generate embedding for a face"""
#         try:
#             if not self.app:
#                 return [0.0] * self.model_config.dimension
            
#             # Handle different input formats
#             if isinstance(face_data, dict):
#                 image_path = face_data.get('image')
#                 bbox = face_data.get('bbox')
                
#                 # Load image
#                 image = cv2.imread(image_path)
                
#                 # Extract face region if bbox provided
#                 if bbox:
#                     x1, y1, x2, y2 = bbox
#                     face_img = image[y1:y2, x1:x2]
#                 else:
#                     face_img = image
#             else:
#                 face_img = face_data
            
#             # Get face embedding
#             faces = self.app.get(face_img)
            
#             if faces:
#                 embedding = faces[0].embedding
#                 # Normalize
#                 embedding = embedding / np.linalg.norm(embedding)
#                 return embedding.tolist()
#             else:
#                 logger.warning("No face detected in image")
#                 return [0.0] * self.model_config.dimension
                
#         except Exception as e:
#             logger.error(f"Error generating face embedding: {e}")
#             return [0.0] * self.model_config.dimension


# ArcFace Embeddings
"""
Face embedding generation using ArcFace
"""
from typing import List, Any, Optional
import numpy as np
import cv2
from backend.config.model_config import ModelRegistry
from backend.utils.logger import app_logger as logger

try:
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    logger.warning("insightface not available")
    INSIGHTFACE_AVAILABLE = False


class FaceEmbedder:
    """Generate face embeddings using ArcFace"""
    
    def __init__(self):
        logger.info("Initializing Face Embedder")
        
        self.model_config = ModelRegistry.FACE_EMBEDDING
        self.app: Optional[FaceAnalysis] = None
        
        if not INSIGHTFACE_AVAILABLE:
            logger.error("InsightFace is not available. Please install it with: pip install insightface")
            return
            
        try:
            # Initialize with proper configuration
            self.app = FaceAnalysis(
                name='buffalo_l',  # Specify the model name
                providers=['CPUExecutionProvider'],  # Force CPU provider
                allowed_modules=['recognition', 'detection']  # Allow detection for face cropping
            )
            
            # Prepare with CPU context
            self.app.prepare(ctx_id=-1)  # -1 for CPU, 0 for GPU
            
            logger.info("✅ ArcFace model loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to load ArcFace: {e}")
            logger.info("Attempting alternative initialization...")
            self._initialize_alternative()
    
    def _initialize_alternative(self):
        """Alternative initialization method"""
        try:
            # Try with minimal configuration
            self.app = FaceAnalysis(providers=['CPUExecutionProvider'])
            self.app.prepare(ctx_id=-1)
            logger.info("✅ ArcFace loaded with alternative method")
        except Exception as e:
            logger.error(f"❌ Alternative initialization also failed: {e}")
            self.app = None
    
    def generate_embedding(self, face_data: Any) -> List[float]:
        """Generate embedding for a face"""
        try:
            if self.app is None:
                logger.warning("ArcFace model not available, returning zero embedding")
                return self._get_zero_embedding()
            
            # Extract file name for logging if available
            file_name = self._extract_file_name(face_data)
            if file_name:
                logger.info(f"Generating face embedding for: {file_name}")
            
            # Handle different input formats
            face_img = self._preprocess_face_data(face_data)
            if face_img is None:
                return self._get_zero_embedding()
            
            # Ensure image is in correct format
            if isinstance(face_img, np.ndarray):
                if len(face_img.shape) == 2:  # Grayscale
                    face_img = cv2.cvtColor(face_img, cv2.COLOR_GRAY2BGR)
                elif face_img.shape[2] == 4:  # RGBA
                    face_img = cv2.cvtColor(face_img, cv2.COLOR_RGBA2BGR)
                elif face_img.shape[2] == 3:  # Already BGR
                    pass
                else:
                    logger.error(f"Unexpected image format: {face_img.shape}")
                    return self._get_zero_embedding()
            
            # Get face embedding
            faces = self.app.get(face_img)
            
            if not faces:
                logger.warning("No face detected in image")
                return self._get_zero_embedding()
            
            # Use the first face found
            embedding = faces[0].embedding
            
            # Normalize the embedding
            embedding = embedding / np.linalg.norm(embedding)
            
            # Ensure correct dimension
            if len(embedding) != self.model_config.dimension:
                logger.warning(f"Embedding dimension mismatch: {len(embedding)} vs {self.model_config.dimension}")
                embedding = self._adjust_embedding_dimension(embedding)
            
            return embedding.tolist()
            
        except Exception as e:
            logger.error(f"Error generating face embedding: {e}")
            return self._get_zero_embedding()
    
    def _preprocess_face_data(self, face_data: Any) -> Optional[np.ndarray]:
        """Preprocess face data to numpy array"""
        try:
            if isinstance(face_data, dict):
                if 'image' in face_data:
                    image_path = face_data['image']
                    bbox = face_data.get('bbox')
                    
                    if isinstance(image_path, str):
                        image = cv2.imread(image_path)
                        if image is None:
                            logger.error(f"Failed to load image: {image_path}")
                            return None
                        
                        # Extract face region if bbox provided
                        if bbox:
                            x1, y1, x2, y2 = map(int, bbox)
                            # Ensure coordinates are within image bounds
                            h, w = image.shape[:2]
                            x1, y1 = max(0, x1), max(0, y1)
                            x2, y2 = min(w, x2), min(h, y2)
                            
                            if x2 > x1 and y2 > y1:
                                face_img = image[y1:y2, x1:x2]
                            else:
                                logger.warning("Invalid bounding box, using full image")
                                face_img = image
                        else:
                            face_img = image
                        
                        return face_img
                    else:
                        logger.error("No image path found in face_data")
                        return None
                        
            elif isinstance(face_data, str):
                # File path
                image = cv2.imread(face_data)
                if image is None:
                    logger.error(f"Failed to load image from path: {face_data}")
                    return None
                return image
                
            elif isinstance(face_data, np.ndarray):
                # Already numpy array
                return face_data
                
            else:
                logger.error(f"Unsupported face data type: {type(face_data)}")
                return None
                
        except Exception as e:
            logger.error(f"Error preprocessing face data: {e}")
            return None
    
    def _extract_file_name(self, face_data: Any) -> str:
        """Extract file name from face data for logging"""
        try:
            if isinstance(face_data, dict):
                if 'image' in face_data:
                    val = face_data['image']
                    if isinstance(val, str):
                        from pathlib import Path
                        return str(Path(val).name)
            elif isinstance(face_data, str):
                from pathlib import Path
                return str(Path(face_data).name)
        except Exception:
            pass
        return None
    
    def _get_zero_embedding(self) -> List[float]:
        """Return zero vector as fallback embedding"""
        return [0.0] * self.model_config.dimension
    
    def _adjust_embedding_dimension(self, embedding: np.ndarray) -> np.ndarray:
        """Adjust embedding dimension if needed"""
        current_dim = len(embedding)
        target_dim = self.model_config.dimension
        
        if current_dim < target_dim:
            # Pad with zeros
            padded = np.zeros(target_dim)
            padded[:current_dim] = embedding
            return padded
        else:
            # Truncate
            return embedding[:target_dim]
    
    def _preprocess_face_data_old(self, face_data: Any) -> Optional[np.ndarray]:
        """Preprocess face data to numpy array"""
        try:
            if isinstance(face_data, dict):
                # Handle dictionary input
                image_path = face_data.get('image_path') or face_data.get('image')
                bbox = face_data.get('bbox')
                
                if image_path:
                    image = cv2.imread(image_path)
                    if image is None:
                        logger.error(f"Failed to load image: {image_path}")
                        return None
                    
                    # Extract face region if bbox provided
                    if bbox:
                        x1, y1, x2, y2 = map(int, bbox)
                        # Ensure coordinates are within image bounds
                        h, w = image.shape[:2]
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w, x2), min(h, y2)
                        
                        if x2 > x1 and y2 > y1:
                            face_img = image[y1:y2, x1:x2]
                        else:
                            logger.warning("Invalid bounding box, using full image")
                            face_img = image
                    else:
                        face_img = image
                    
                    return face_img
                else:
                    logger.error("No image path found in face_data")
                    return None
                    
            elif isinstance(face_data, str):
                # File path
                image = cv2.imread(face_data)
                if image is None:
                    logger.error(f"Failed to load image from path: {face_data}")
                    return None
                return image
                
            elif isinstance(face_data, np.ndarray):
                # Already numpy array
                return face_data
                
            else:
                logger.error(f"Unsupported face data type: {type(face_data)}")
                return None
                
        except Exception as e:
            logger.error(f"Error preprocessing face data: {e}")
            return None
    
    def _get_zero_embedding(self) -> List[float]:
        """Return zero embedding of correct dimension"""
        return [0.0] * self.model_config.dimension
    
    def _adjust_embedding_dimension(self, embedding: np.ndarray) -> np.ndarray:
        """Adjust embedding to correct dimension"""
        current_dim = len(embedding)
        target_dim = self.model_config.dimension
        
        if current_dim < target_dim:
            # Pad with zeros
            padded = np.zeros(target_dim)
            padded[:current_dim] = embedding
            return padded
        else:
            # Truncate
            return embedding[:target_dim]
    
    def generate_face_embeddings(self, faces: List[Any]) -> List[List[float]]:
        """Generate batch of face embeddings"""
        return [self.generate_embedding(face) for face in faces]
    
    def is_available(self) -> bool:
        """Check if face embedder is available"""
        return self.app is not None