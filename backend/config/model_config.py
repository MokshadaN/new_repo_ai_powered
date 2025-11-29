"""
Model configuration and loading
"""
from dataclasses import dataclass
from typing import Optional, Dict, Any, Any
from pathlib import Path

@dataclass
class ModelConfig:
    """Configuration for a single model"""
    name: str
    path: str
    dimension: int
    device: str = "cuda"
    batch_size: int = 32
    max_length: Optional[int] = None
    additional_params: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.additional_params is None:
            self.additional_params = {}

class ModelRegistry:
    """Registry of all models used in the application"""
    
    TEXT_EMBEDDING = ModelConfig(
        name="BGE-M3",
        path="BAAI/bge-m3",
        dimension=1024,
        max_length=8192
    )
    
    IMAGE_EMBEDDING = ModelConfig(
        name="SigLIP",
        path="google/siglip-base-patch16-224",
        dimension=768
    )
    
    FACE_DETECTION = ModelConfig(
        name="RetinaFace",
        path="buffalo_l",
        dimension=512
    )
    
    FACE_EMBEDDING = ModelConfig(
        name="ArcFace",
        path="buffalo_l",
        dimension=512
    )
    
    LOCAL_LLM = ModelConfig(
        name="Mistral-7B",
        path="mistralai/Mistral-7B-Instruct-v0.2",
        additional_params={
            "temperature": 0.7,
            "max_new_tokens": 2048,
            "top_p": 0.95
        },
        dimension=4096
    )
    OLLAMA_API_URL: str = "http://localhost:11434/api/chat"
    OLLAMA_LLM_NAME: str = "bge-m3:latest" 
    # OLLAMA_LLM_ANSWER: str = "deepseek-r1:latest" 
    # OLLAMA_LLM_ANSWER: str = "qwen3:4b" 
    # OLLAMA_LLM_ANSWER: str = "phi3:mini" 
    OLLAMA_LLM_ANSWER: str = "llama3.2:1b" 