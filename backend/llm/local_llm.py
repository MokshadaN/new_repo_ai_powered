# """
# Local LLM integration for context generation and Q&A
# """
# from typing import List, Optional, Any
# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
# from backend.config.settings import settings
# from backend.config.model_config import ModelRegistry
# from backend.utils.logger import app_logger as logger
# from PIL import Image
# import io


# class LocalLLM:
#     """Local LLM wrapper"""
    
#     def __init__(self):
#         logger.info("Initializing Local LLM...")
        
#         self.model_config = ModelRegistry.LOCAL_LLM
#         self.device = "cuda" if torch.cuda.is_available() and settings.enable_gpu else "cpu"
        
#         try:
#             self.tokenizer = AutoTokenizer.from_pretrained(self.model_config.path)
#             self.model = AutoModelForCausalLM.from_pretrained(
#                 self.model_config.path,
#                 torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
#                 device_map="auto" if self.device == "cuda" else None,
#                 low_cpu_mem_usage=True
#             )
            
#             if self.device == "cpu":
#                 self.model = self.model.to(self.device)
            
#             self.model.eval()
            
#             logger.info(f"Local LLM loaded on {self.device}")
            
#         except Exception as e:
#             logger.error(f"Failed to load LLM: {e}")
#             self.model = None
#             self.tokenizer = None
    
#     def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> str:
#         """Generate text from prompt"""
#         if not self.model:
#             return "LLM not available"
        
#         try:
#             inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
#             with torch.no_grad():
#                 outputs = self.model.generate(
#                     **inputs,
#                     max_new_tokens=max_tokens,
#                     temperature=temperature,
#                     top_p=self.model_config.additional_params.get('top_p', 0.95),
#                     do_sample=True,
#                     pad_token_id=self.tokenizer.eos_token_id
#                 )
            
#             response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
#             # Remove the prompt from response
#             if response.startswith(prompt):
#                 response = response[len(prompt):].strip()
            
#             return response
            
#         except Exception as e:
#             logger.error(f"Error generating text: {e}")
#             return "Error generating response"


# class VisionLLM:
#     """Vision-enabled LLM for image understanding"""
    
#     def __init__(self):
#         logger.info("Initializing Vision LLM")
#         self.llm = OllamaLLM()
    
#     def generate_context(self, images: List[Any], query: str = "") -> str:
#         """Generate textual context from images"""
#         try:
#             # For production, use actual vision model like BLIP, LLaVA, etc.
#             # This is a simplified version
            
#             contexts = []
#             for img in images:
#                 if isinstance(img, dict):
#                     path = img.get('path', 'unknown')
#                 elif isinstance(img, str):
#                     path = img
#                 else:
#                     path = 'image_data'
                
#                 # Simple context generation
#                 context = f"Image from {Path(path).name}: Contains visual content"
#                 contexts.append(context)
            
#             combined = " | ".join(contexts)
            
#             # Use LLM to enhance context if available
#             if query:
#                 prompt = f"Describe what might be in these images based on the query '{query}': {combined}"
#                 enhanced = self.llm.generate(prompt, max_tokens=200)
#                 return enhanced
            
#             return combined
            
#         except Exception as e:
#             logger.error(f"Error generating image context: {e}")
#             return "Unable to generate image context"
    
#     def generate_context_from_bytes(self, image_bytes: bytes, query: str = "") -> str:
#         """Generate context from image bytes"""
#         try:
#             # For production, implement actual vision model
#             return f"Image analysis: Relevant to query '{query}'"
#         except Exception as e:
#             logger.error(f"Error generating context from bytes: {e}")
#             return "Unable to analyze image"


import base64
import json
import requests
from pathlib import Path
from typing import List, Optional, Any
from backend.config.settings import settings
from backend.config.model_config import ModelRegistry
from backend.utils.logger import app_logger as logger
class OllamaLLM:
    """Ollama LLM wrapper"""
    
    def __init__(self):
        logger.info("Initializing Ollama LLM...")
        
        self.api_url = settings.OLLAMA_API_URL+"/api/generate"  # Ollama API base URL
        self.model_name = ModelRegistry.OLLAMA_LLM_ANSWER  # The specific model you want to use
        
    # def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> str:
    #     """Generate text from prompt using Ollama API"""
    #     try:
    #         # Prepare the payload to send to Ollama
    #         payload = {
    #             "model": self.model_name,
    #             "prompt": prompt,
    #             "max_tokens": max_tokens,
    #             "temperature": temperature
    #         }
            
    #         headers = {
    #             "Content-Type": "application/json"
    #         }
            
    #         # Make the API request to Ollama
    #         response = requests.post(self.api_url, headers=headers, data=json.dumps(payload))
            
    #         if response.status_code == 200:
    #             data = response.json()
    #             return data.get('text', 'No response from Ollama')
    #         else:
    #             logger.error(f"Ollama API error: {response.status_code} - {response.text}")
    #             return "Error generating response"
            
    #     except Exception as e:
    #         logger.error(f"Error generating text from Ollama: {e}")
    #         return "Error generating response"
    def generate(self, prompt: str, max_tokens: int = 2048) -> str:
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "num_predict": 5000,
                "stream": True
            }

            # print("Ollama Payload:", payload)

            response = requests.post(self.api_url, json=payload, stream=True)

            if response.status_code != 200:
                raise Exception(f"Ollama Error {response.status_code}")

            final_text = ""

            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if "response" in obj:
                        final_text += obj["response"]
                except json.JSONDecodeError:
                    logger.error(f"Bad JSON line from Ollama: {line}")
                    continue
            print("Ollama response:",final_text)
            return final_text.strip()

        except Exception as e:
            logger.error(f"Error generating text from Ollama: {e}")
            return "LLM Error"

from ollama import Client
class VisionLLM:
    """Vision-enabled LLM for image understanding with Ollama"""
    
    def __init__(self):
        logger.info("Initializing Vision LLM with Ollama")
        self.llm = OllamaLLM()
        # Allow overriding the vision model via settings; default to gemma3:latest as requested
        self.vision_model = getattr(settings, "OLLAMA_VISION_MODEL", "gemma3:latest")
        self.OLLAMA_HOST = getattr(settings, "OLLAMA_API_URL", "http://127.0.0.1:11434")
        self.client = Client(host=self.OLLAMA_HOST)
        self.MODEL_NAME = self.vision_model

    def _encode_image(self, img: Any) -> Optional[str]:
        """Convert supported image inputs to base64 for Ollama."""
        try:
            if isinstance(img, bytes):
                data = img
            elif isinstance(img, str):
                data = Path(img).read_bytes()
            elif isinstance(img, dict) and "path" in img:
                data = Path(img["path"]).read_bytes()
            else:
                return None
            return base64.b64encode(data).decode("utf-8")
        except Exception as e:
            logger.error(f"Error encoding image for LLM: {e}")
            return None
    
    def generate_context(self, images: List[Any], query: str = "") -> str:
        """Generate textual context from images using a vision-capable Ollama model."""
        try:
            b64_images = [enc for enc in (self._encode_image(i) for i in images) if enc]
            if not b64_images:
                logger.warning("No encodable images provided to VisionLLM.generate_context")
                return ""

            base_prompt = (
                "Describe these images in a concise, search-friendly way: main objects, colors, "
                "materials, any readable text, scene/setting, actions, viewpoint, lighting, and counts."
            )
            final_prompt = f"{base_prompt} Query/intent: {query}" if query else base_prompt
            
            res = self.client.generate(
                model=self.MODEL_NAME,
                prompt=final_prompt,
                images=b64_images,  # list of base64-encoded images
                stream=False,
            )
            return res.get("response", "").strip()

        except Exception as e:
            logger.error(f"Error generating image context: {e}")
            return "Unable to generate image context"
    
    def generate_context_from_bytes(self, image_bytes: bytes, query: str = "") -> str:
        """Generate context from image bytes"""
        try:
            # For production, implement actual vision model
            return f"Image analysis: Relevant to query '{query}'"
        except Exception as e:
            logger.error(f"Error generating context from bytes: {e}")
            return "Unable to analyze image"
