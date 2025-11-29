# State definitions
"""
State models for LangGraph pipelines
"""
from typing import TypedDict, Any, List, Optional, Literal, Dict, Any, Any

class IngestionState(TypedDict):
    """State for file ingestion pipeline"""
    folder_path: str
    all_files: Optional[List[str]]
    text_files: Optional[List[str]]
    image_files: Optional[List[str]]
    current_file: Optional[str]
    current_file_type: Optional[Literal["text", "image"]]
    
    # Text processing
    extracted_text: Optional[str]
    text_embedding: Optional[List[float]]
    
    # Image processing
    processed_image: Optional[Any]
    image_embedding: Optional[List[float]]
    extracted_images_from_docs: Optional[List[str]]
    
    # OCR
    ocr_text: Optional[str]
    
    # Vision context
    image_context: Optional[str]
    image_context_embedding: Optional[List[float]]
    
    # Face detection
    detected_faces: Optional[List[Dict]]
    face_embeddings: Optional[List[List[float]]]
    
    # Object detection
    detected_objects: Optional[List[Dict]]
    object_embeddings: Optional[List[List[float]]]
    
    # Progress tracking
    files_processed: int
    total_files: int
    errors: List[str]

class QueryState(TypedDict):
    """State for search/query pipeline"""
    query_text: str
    query_image: Optional[bytes]
    query_type: Literal["text_only", "text_image", "face_search", "object_search"]
    
    # Multimodal processing
    image_context: Optional[str]
    combined_query: Optional[str]
    
    # Embeddings
    text_query_embedding: Optional[List[float]]
    image_query_embedding: Optional[List[float]]
    
    # Search results
    results_from_text_faiss: List[Dict]
    results_from_image_faiss_text: List[Dict]
    results_from_image_faiss_image: List[Dict]
    results_from_face_faiss: List[Dict]
    results_from_object_faiss: List[Dict]
    skip_llm: bool | None
    
    # Aggregation
    aggregated_context: str
    reranked_results: List[Dict]
    
    # Final output
    final_response: str
    insights: Optional[Dict]
