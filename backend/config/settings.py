"""
Configuration management for AI Disk Analyzer
"""
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path
from typing import Optional, Any

class Settings(BaseSettings):
    """Application settings"""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )
    
    # Paths
    base_dir: Path = Path(__file__).parent.parent.parent
    data_dir: Path = base_dir / "data"
    models_dir: Path = data_dir / "models"
    vector_stores_dir: Path = data_dir / "vector_stores"
    cache_dir: Path = data_dir / "cache"
    logs_dir: Path = data_dir / "logs"
    
    # Model paths
    bge_model_path: str = "BAAI/bge-m3"
    siglip_model_path: str = "google/siglip-base-patch16-224"
    retinaface_model_path: Optional[str] = None
    arcface_model_path: Optional[str] = None
    llm_model_path: str = "mistralai/Mistral-7B-Instruct-v0.2"
    # Face detection / recognition models
    USE_RETINAFACE: bool = False 


    OLLAMA_API_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "bge-m3:latest" 
    # Storage
    chroma_path: Path = vector_stores_dir / "chroma"
    faiss_path: Path = vector_stores_dir / "faiss"
    
    # Processing settings
    max_file_size_mb: int = 100
    batch_size: int = 32
    num_workers: int = 4
    embedding_dimension: int = 1024
    image_embedding_dimension: int = 768
    face_embedding_dimension: int = 512
    object_embedding_dimension: int=768
    
    # Search settings
    top_k_results: int = 20
    similarity_threshold: float = 0.7
    
    # File monitoring
    watch_directories: list[str] = []
    excluded_extensions: list[str] = ['.tmp', '.log', '.cache']
    
    # OCR settings
    ocr_languages: list[str] = ['en']
    ocr_confidence_threshold: float = 0.6
    
    # Clustering
    min_cluster_size: int = 3
    clustering_metric: str = 'cosine'
    
    # Performance
    enable_gpu: bool = True
    gpu_memory_fraction: float = 0.8
    
    # Logging
    enable_logging: bool = False
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._create_directories()
    
    def _create_directories(self):
        """Create necessary directories"""
        for dir_path in [
            self.data_dir,
            self.models_dir,
            self.vector_stores_dir,
            self.cache_dir,
            self.logs_dir,
            self.chroma_path,
            self.faiss_path
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)

settings = Settings()


# """
# Configuration management for AI Disk Analyzer
# """
# from pydantic_settings import BaseSettings, SettingsConfigDict
# from pathlib import Path
# from typing import Optional, Any
# import torch


# class Settings(BaseSettings):
#     """Application settings"""

#     model_config = SettingsConfigDict(
#         env_file=".env",
#         env_file_encoding="utf-8",
#         case_sensitive=False
#     )

#     # Paths
#     base_dir: Path = Path(__file__).parent.parent.parent
#     data_dir: Path = base_dir / "data"
#     models_dir: Path = data_dir / "models"
#     vector_stores_dir: Path = data_dir / "vector_stores"
#     cache_dir: Path = data_dir / "cache"
#     logs_dir: Path = data_dir / "logs"

#     # Embedding / Model paths
#     bge_model_path: str = "BAAI/bge-m3"
#     siglip_model_path: str = "google/siglip-base-patch16-224"

#     # Face detection / recognition models
#     retinaface_model_path: Optional[str] = None
#     arcface_model_path: Optional[str] = None
#     USE_RETINAFACE: bool = False  # Disabled to avoid KerasTensor errors

#     # LLM (Ollama)
#     OLLAMA_API_URL: str = "http://localhost:11434"
#     OLLAMA_MODEL: str = "bge-m3:latest"  # Or bge-m3:latest, llama3, etc.

#     # Vector stores
#     chroma_path: Path = vector_stores_dir / "chroma"
#     faiss_path: Path = vector_stores_dir / "faiss"

#     # Processing settings
#     max_file_size_mb: int = 100
#     batch_size: int = 32
#     num_workers: int = 4

#     # Dimensions (auto-read from model config)
#     embedding_dimension: int = 1024   # BGE-M3
#     image_embedding_dimension: int = 768  # SigLIP
#     face_embedding_dimension: int = 512   # ArcFace (if used)

#     # Search settings
#     top_k_results: int = 20
#     similarity_threshold: float = 0.7

#     # File monitoring
#     watch_directories: list[str] = []
#     excluded_extensions: list[str] = ['.tmp', '.log', '.cache']

#     # OCR settings
#     ocr_languages: list[str] = ['en']
#     ocr_confidence_threshold: float = 0.6

#     # Clustering
#     min_cluster_size: int = 3
#     clustering_metric: str = 'cosine'

#     # GPU settings
#     enable_gpu: bool = True      # User preference
#     gpu_memory_fraction: float = 0.8

#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)

#         # Create directories
#         self._create_directories()

#         # Auto-disable GPU if not available (Intel ARC can't run PyTorch CUDA)
#         if not torch.cuda.is_available():
#             self.enable_gpu = False

#     def _create_directories(self):
#         """Create necessary directories"""
#         for dir_path in [
#             self.data_dir,
#             self.models_dir,
#             self.vector_stores_dir,
#             self.cache_dir,
#             self.logs_dir,
#             self.chroma_path,
#             self.faiss_path
#         ]:
#             dir_path.mkdir(parents=True, exist_ok=True)


# settings = Settings()
