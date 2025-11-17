# """
# Image embedding generation using SigLIP
# new fixed code
# """
# from typing import List, Any
# import torch
# import numpy as np
# from PIL import Image
# from transformers import AutoProcessor, AutoModel
# from backend.config.settings import settings
# from backend.config.model_config import ModelRegistry
# from backend.utils.logger import app_logger as logger
# from backend.config.model_config import ModelRegistry
# import io

# class ImageEmbedder:
#     """Generate image embeddings using SigLIP"""

#     # def __init__(self):
#     #     logger.info("Loading SigLIP model...")

#     #     self.model_config = ModelRegistry.IMAGE_EMBEDDING
#     #     self.device = "cuda" if torch.cuda.is_available() and settings.enable_gpu else "cpu"

#     #     try:
#     #         self.processor = AutoProcessor.from_pretrained(self.model_config.path)
#     #         self.model = AutoModel.from_pretrained(self.model_config.path).to(self.device)
#     #         self.model.eval()

#     #         logger.info(f"SigLIP loaded on {self.device}")
#     #     except Exception as e:
#     #         logger.error(f"Failed to load SigLIP: {e}")
#     #         raise

#     def __init__(self):
#             self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#             self.model_config = ModelRegistry.IMAGE_EMBEDDING
#             try:
#                 logger.info("Loading SigLIP model...")
#                 self.processor = AutoProcessor.from_pretrained(self.model_config.path)
                
#                 # FIX: Load model without immediate .to(device) call
#                 self.model = AutoModel.from_pretrained(self.model_config.path)
                
#                 # Move model to device after loading
#                 self.model = self.model.to(self.device)
                
#                 self.model.eval()
#                 logger.info(f"SigLIP loaded on {self.device}")
                
#             except Exception as e:
#                 logger.error(f"Failed to load SigLIP: {e}")
#                 raise

#     def generate_embedding(self, image_data: Any) -> List[float]:
#         """Generate embedding for single image"""
#         try:
#             # Handle different input types safely
#             if isinstance(image_data, dict):
#                 if "array" in image_data:
#                     image = image_data["array"]
#                 elif "original" in image_data:
#                     image = image_data["original"]
#                 else:
#                     # Support other common keys produced by detectors/processors
#                     path_key = None
#                     for k in ("image", "path", "file_path", "filepath"):
#                         if k in image_data:
#                             path_key = k
#                             break

#                     if path_key:
#                         try:
#                             val = image_data[path_key]
#                             # If the value is a numpy array or PIL image, use it
#                             if isinstance(val, np.ndarray):
#                                 image = val
#                             else:
#                                 # Assume it's a path-like string; try to open with PIL
#                                 image = Image.open(val).convert("RGB")
#                         except Exception as e:
#                             logger.error(f"Failed to load image from '{path_key}': {e}")
#                             return [0.0] * self.model_config.dimension
#                     else:
#                         logger.error("Image dict does not contain 'array', 'original', or a path key (image/path/file_path)")
#                         return [0.0] * self.model_config.dimension

#             elif isinstance(image_data, np.ndarray):
#                 image = image_data

#             elif isinstance(image_data, str):
#                 image = Image.open(image_data).convert("RGB")

#             else:
#                 image = image_data

#             # Convert numpy array to PIL
#             if isinstance(image, np.ndarray):
#                 if image.dtype != np.uint8:
#                     image = (image * 255).astype(np.uint8)
#                 image = Image.fromarray(image)

#             # Process image
#             inputs = self.processor(images=image, return_tensors="pt").to(self.device)

#             # Generate embedding
#             with torch.no_grad():
#                 outputs = self.model.get_image_features(**inputs)
#                 embedding = outputs.squeeze().cpu().numpy()

#             # Normalize vector
#             norm = np.linalg.norm(embedding)
#             if norm == 0:
#                 return [0.0] * self.model_config.dimension

#             embedding = embedding / norm

#             return embedding.tolist()

#         except Exception as e:
#             logger.error(f"Error generating image embedding: {e}")
#             return [0.0] * self.model_config.dimension

#     def generate_batch_embeddings(self, images: List[Any]) -> List[List[float]]:
#         """Generate embeddings for multiple images"""
#         return [self.generate_embedding(img) for img in images]

#     def generate_embedding_from_bytes(self, image_bytes: bytes) -> List[float]:
#         """Generate embedding from image bytes"""
#         try:
#             image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
#             return self.generate_embedding(image)
#         except Exception as e:
#             logger.error(f"Error generating embedding from bytes: {e}")
#             return [0.0] * self.model_config.dimension
"""
Image embedding generation using SigLIP - Fixed Version
"""
from typing import List, Any
import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModel
from backend.config.settings import settings
from backend.config.model_config import ModelRegistry
from backend.utils.logger import app_logger as logger
import io


class ImageEmbedder:
    """Generate image embeddings using SigLIP"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_config = ModelRegistry.IMAGE_EMBEDDING
        
        logger.info(f"Initializing ImageEmbedder on device: {self.device}")
        
        try:
            # Load processor first
            self.processor = AutoProcessor.from_pretrained(
                self.model_config.path,
                trust_remote_code=True
            )
            
            # Load model with device mapping to avoid meta tensor issues
            self.model = AutoModel.from_pretrained(
                self.model_config.path,
                trust_remote_code=True,
                torch_dtype=torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            # If not using device_map, move manually
            if not torch.cuda.is_available():
                self.model = self.model.to(self.device)
            
            self.model.eval()
            
            logger.info(f"✅ SigLIP successfully loaded on {self.device}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load SigLIP: {e}")
            if "meta tensor" in str(e).lower():
                logger.error("Meta tensor issue detected - trying alternative loading method...")
                self._load_with_alternative_method()
            else:
                raise
    
    def _load_with_alternative_method(self):
        """Proper alternative loading method for meta tensor issues"""
        try:
            logger.info("Trying alternative loading method...")
            
            # Method 1: Try with low_cpu_mem_usage and device_map
            try:
                self.model = AutoModel.from_pretrained(
                    self.model_config.path,
                    trust_remote_code=True,
                    torch_dtype=torch.float32,
                    device_map="auto",
                    low_cpu_mem_usage=True
                )
                logger.info("✅ SigLIP loaded with device_map method")
                
            except Exception as e1:
                logger.warning(f"Device map method failed: {e1}, trying manual loading...")
                
                # Method 2: Manual loading without device movement first
                self.model = AutoModel.from_pretrained(
                    self.model_config.path,
                    trust_remote_code=True,
                    torch_dtype=torch.float32
                )
                
                # Check for meta tensors and handle properly
                has_meta_tensors = any(param.is_meta for param in self.model.parameters())
                
                if has_meta_tensors:
                    logger.warning("Model contains meta tensors - initializing on device...")
                    # Create model on target device with empty tensors
                    self.model = self.model.apply(lambda module: module.to_empty(device=self.device, recurse=False))
                    # This requires re-loading weights, so we need to re-initialize
                    logger.error("Meta tensors detected - model needs re-downloading")
                    raise RuntimeError("Model has meta tensors. Please delete cache and re-download.")
                else:
                    # Regular device movement
                    self.model = self.model.to(self.device)
                    logger.info("✅ SigLIP loaded with manual method")
            
            self.model.eval()
            
        except Exception as e:
            logger.error(f"❌ All alternative loading methods failed: {e}")
            logger.info("Attempting to load with force_download...")
            self._load_with_force_download()
    
    def _load_with_force_download(self):
        """Force re-download the model to fix corrupted cache"""
        try:
            import shutil
            from pathlib import Path
            from transformers.utils import HUGGINGFACE_HUB_CACHE
            
            logger.warning("Attempting to clear cache and re-download model...")
            
            # Clear the specific model cache
            model_path = self.model_config.path
            if '/' in model_path:
                cache_name = model_path.replace('/', '--')
                cache_path = Path(HUGGINGFACE_HUB_CACHE) / cache_name
                if cache_path.exists():
                    shutil.rmtree(cache_path)
                    logger.info(f"Cleared cache: {cache_path}")
            
            # Re-download with force
            self.processor = AutoProcessor.from_pretrained(
                model_path,
                trust_remote_code=True,
                force_download=True
            )
            
            self.model = AutoModel.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch.float32,
                device_map="auto",
                force_download=True
            )
            
            self.model.eval()
            logger.info("✅ SigLIP successfully re-downloaded and loaded")
            
        except Exception as e:
            logger.error(f"❌ Force download also failed: {e}")
            raise RuntimeError(f"Failed to load SigLIP model after multiple attempts: {e}")

    def generate_embedding(self, image_data: Any) -> List[float]:
        """Generate embedding for single image"""
        try:
            # Extract file name for logging if available
            file_name = self._extract_file_name(image_data)
            if file_name:
                logger.info(f"Generating image embedding for: {file_name}")
            
            # Handle different input types
            image = self._load_image(image_data)
            if image is None:
                return self._get_zero_embedding()
            
            # Process image
            inputs = self.processor(
                images=image, 
                return_tensors="pt"
            )
            
            # Move inputs to same device as model
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Generate embedding
            with torch.no_grad():
                outputs = self.model.get_image_features(**inputs)
                embedding = outputs.squeeze().cpu().numpy()
            
            # Handle different output shapes
            if embedding.ndim == 0:
                embedding = np.array([embedding])
            elif embedding.ndim > 1:
                embedding = embedding.flatten()
            
            # Normalize
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            else:
                logger.warning("Zero norm embedding detected")
                return self._get_zero_embedding()
            
            # Ensure correct dimension
            if len(embedding) != self.model_config.dimension:
                logger.warning(f"Embedding dimension mismatch: {len(embedding)} vs {self.model_config.dimension}")
                embedding = self._adjust_embedding_dimension(embedding)
            
            return embedding.tolist()
            
        except Exception as e:
            logger.error(f"Error generating image embedding: {e}")
            return self._get_zero_embedding()
    
    def _extract_file_name(self, image_data: Any) -> str:
        """Extract file name from image data for logging"""
        try:
            if isinstance(image_data, dict):
                for key in ['path', 'file_path', 'image']:
                    if key in image_data:
                        val = image_data[key]
                        if isinstance(val, str):
                            from pathlib import Path
                            return str(Path(val).name)
            elif isinstance(image_data, str):
                from pathlib import Path
                return str(Path(image_data).name)
        except Exception:
            pass
        return None
    
    def _load_image(self, image_data: Any) -> Any:
        """Load and validate image from various input types"""
        try:
            if isinstance(image_data, dict):
                for key in ['array', 'original', 'image', 'path', 'file_path']:
                    if key in image_data and image_data[key] is not None:
                        return self._load_single_image(image_data[key])
                return None
            elif isinstance(image_data, str):
                return Image.open(image_data).convert('RGB')
            elif isinstance(image_data, np.ndarray):
                return self._convert_array_to_image(image_data)
            elif isinstance(image_data, Image.Image):
                return image_data.convert('RGB')
            else:
                logger.error(f"Unsupported image data type: {type(image_data)}")
                return None
        except Exception as e:
            logger.error(f"Error loading image: {e}")
            return None
    
    def _load_single_image(self, data: Any) -> Any:
        """Load single image from various data types"""
        if isinstance(data, str):
            return Image.open(data).convert('RGB')
        elif isinstance(data, np.ndarray):
            return self._convert_array_to_image(data)
        elif isinstance(data, Image.Image):
            return data.convert('RGB')
        else:
            return data
    
    def _convert_array_to_image(self, array: np.ndarray) -> Image.Image:
        """Convert numpy array to PIL Image"""
        if array.dtype != np.uint8:
            if array.max() <= 1.0:
                array = (array * 255).astype(np.uint8)
            else:
                array = array.astype(np.uint8)
        return Image.fromarray(array)
    
    def _get_zero_embedding(self) -> List[float]:
        """Return zero embedding of correct dimension"""
        return [0.0] * self.model_config.dimension
    
    def _adjust_embedding_dimension(self, embedding: np.ndarray) -> np.ndarray:
        """Adjust embedding to correct dimension"""
        current_dim = len(embedding)
        target_dim = self.model_config.dimension
        
        if current_dim < target_dim:
            padded = np.zeros(target_dim)
            padded[:current_dim] = embedding
            return padded
        else:
            return embedding[:target_dim]
    
    def generate_batch_embeddings(self, images: List[Any]) -> List[List[float]]:
        """Generate embeddings for multiple images"""
        return [self.generate_embedding(img) for img in images]
    
    def generate_embedding_from_bytes(self, image_bytes: bytes) -> List[float]:
        """Generate embedding from image bytes"""
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            return self.generate_embedding(image)
        except Exception as e:
            logger.error(f"Error generating embedding from bytes: {e}")
            return self._get_zero_embedding()