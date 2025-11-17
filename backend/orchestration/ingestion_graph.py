from typing import Dict, Any, List, Literal, Any
from langgraph.graph import StateGraph, END, START
# from backend.orchestration.state_models import IngestionState
from backend.ingestion.file_scanner import FileScanner
# from backend.processors.text_processor import TextProcessor
# from backend.processors.image_processor import ImageProcessor
# from backend.processors.ocr_processor import OCRProcessor
from backend.embeddings.embedding_manager import EmbeddingManager
# from backend.detection.face_detector import FaceDetector
# from backend.detection.object_detector import ObjectDetector
# from backend.llm.local_llm import VisionLLM
from backend.vector_store.store_manager import VectorStoreManager
from backend.utils.logger import app_logger as logger
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


class IngestionPipeline:
    """LangGraph-based file ingestion pipeline"""
    
    def __init__(self):
        logger.info("Initializing Ingestion Pipeline")
        
        # Initialize components
        # print("I_1")
        self.file_scanner = FileScanner()
        # print("I_2")
        # self.text_processor = TextProcessor()
        # print("I_3")
        # self.image_processor = ImageProcessor()
        # print("I_4")
        # self.ocr_processor = OCRProcessor()
        # print("I_5")
        # self.embedding_manager = EmbeddingManager()
        # print("I_6")
        # self.face_detector = FaceDetector()
        # print("I_7")
        # self.object_detector = ObjectDetector()
        # print("I_8")
        # self.vision_llm = VisionLLM()
        # print("I_9")
        self.store_manager = VectorStoreManager()
        # print("I_10")
        
        # Build graph
        self.graph = self._build_graph()
        self.app = self.graph.compile()
        
        logger.info("Ingestion Pipeline initialized successfully")
    
    def _build_graph(self) -> StateGraph:
        """Build the ingestion graph following Image 1 architecture"""
        workflow = StateGraph(IngestionState)
        
        # Add nodes
        workflow.add_node("scan_and_segregate", self.scan_and_segregate)
        workflow.add_node("get_next_file", self.get_next_file)
        workflow.add_node("file_type_splitter", self.file_type_splitter)
        
        # # Text path nodes
        # workflow.add_node("process_text", self.process_text)
        # workflow.add_node("generate_text_embedding", self.generate_text_embedding)
        # workflow.add_node("store_text_faiss", self.store_text_faiss)
        # workflow.add_node("extract_images_from_text", self.extract_images_from_text)
        
        # # Image path nodes
        # workflow.add_node("prepare_images", self.prepare_images)
        # workflow.add_node("generate_image_embeddings", self.generate_image_embeddings)
        # workflow.add_node("store_images_faiss", self.store_images_faiss)
        # workflow.add_node("generate_image_context_llm", self.generate_image_context_llm)
        # workflow.add_node("store_context_text_faiss", self.store_context_text_faiss)
        # workflow.add_node("detect_faces", self.detect_faces)
        # workflow.add_node("store_faces_faiss", self.store_faces_faiss)
        # workflow.add_node("detect_objects", self.detect_objects)
        # workflow.add_node("store_objects_faiss", self.store_objects_faiss)
        
        # # Progress tracking
        # workflow.add_node("increment_progress", self.increment_progress)
        
        # Define edges
        workflow.set_entry_point("scan_and_segregate")
        workflow.add_edge("scan_and_segregate", "get_next_file")
        # workflow.add_edge("scan_and_segregate", END)
        
        # # Conditional routing from get_next_file
        workflow.add_conditional_edges(
            "get_next_file",
            self.route_next_file,
            {
                "process_file": "file_type_splitter",
                "end": END
            }
        )
        
        # # File type splitter
        workflow.add_conditional_edges(
            "file_type_splitter",
            lambda state: state["current_file_type"],
            {
                "text": "process_text",
                "image": "prepare_images"
            }
        )
        
        # # Text processing path
        workflow.add_edge("process_text", END)
        # workflow.add_edge("process_text", "generate_text_embedding")
        # workflow.add_edge("generate_text_embedding", "store_text_faiss")
        # workflow.add_edge("store_text_faiss", "extract_images_from_text")
        
        # # Extract images decision
        # workflow.add_conditional_edges(
        #     "extract_images_from_text",
        #     self.route_after_text_extraction,
        #     {
        #         "process_images": "prepare_images",
        #         "next_file": "increment_progress"
        #     }
        # )
        
        # # Image processing path (parallel branches)
        workflow.add_edge("prepare_images", END)
        # workflow.add_edge("prepare_images", "generate_image_embeddings")
        # workflow.add_edge("generate_image_embeddings", "store_images_faiss")
        # workflow.add_edge("store_images_faiss", "generate_image_context_llm")
        # workflow.add_edge("generate_image_context_llm", "store_context_text_faiss")
        # workflow.add_edge("store_context_text_faiss", "detect_faces")
        # workflow.add_edge("detect_faces", "store_faces_faiss")
        # workflow.add_edge("store_faces_faiss", "detect_objects")
        # workflow.add_edge("detect_objects", "store_objects_faiss")
        # workflow.add_edge("store_objects_faiss", "increment_progress")
        
        # # Loop back
        # workflow.add_edge("increment_progress", "get_next_file")
        
        return workflow
    
    def scan_and_segregate(self, state: IngestionState) -> IngestionState:
        """Scan folder, filter existing, and categorize files"""
        logger.info(f"Scanning folder: {state['folder_path']}")
        
        all_files, text_files, image_files = self.file_scanner.scan_folder(
            state["folder_path"]
        )
        
        # Filter out existing files
        new_all_files, new_text_files, new_image_files = self.check_existing_files(
            all_files, text_files, image_files
        )

        logger.info(f"Found {len(new_text_files)} new text files, {len(new_image_files)} new image files")
        print({
            "all_files": new_all_files,
            "text_files": new_text_files,
            "image_files": new_image_files,
            "total_files": len(new_all_files),
            "files_processed": 0,
            "errors": []
        })
        return {
            "all_files": new_all_files,
            "text_files": new_text_files,
            "image_files": new_image_files,
            "total_files": len(new_all_files),
            "files_processed": 0,
            "errors": []
        }
    
    def check_existing_files(self, all_files: List[str], text_files: List[str], image_files: List[str]) -> tuple[List[str], List[str], List[str]]:
        """Check for existing files in the vector store"""
        new_all_files = [f for f in all_files if not self.store_manager.check_file_exists(f)]
        new_text_files = [f for f in text_files if f in new_all_files]
        new_image_files = [f for f in image_files if f in new_all_files]

        return new_all_files, new_text_files, new_image_files

    def get_next_file(self, state: IngestionState) -> IngestionState:
        """Get next file to process"""
        text_files = state.get("text_files", [])
        image_files = state.get("image_files", [])
        
        if text_files:
            logger.info(f"Next file: {text_files[0]} (text)")
            return {
                "current_file": text_files[0],
                "current_file_type": "text",
                "text_files": text_files[1:]
            }
        elif image_files:
            logger.info(f"Next file: {image_files[0]} (image)")
            return {
                "current_file": image_files[0],
                "current_file_type": "image",
                "image_files": image_files[1:]
            }
        else:
            logger.info("No more files to process")
            return {
                "current_file": None,
                "current_file_type": None
            }
    
    def file_type_splitter(self, state: IngestionState) -> IngestionState:
        """Split processing based on file type"""
        return {}  # Just a routing node
    
    def route_next_file(self, state: IngestionState) -> Literal["process_file", "end"]:
        """Route based on whether there are more files"""
        if state.get("current_file"):
            return "process_file"
        return "end"
   
    


    def run(self, folder_path: str) -> Dict:
        """Execute the ingestion pipeline"""
        logger.info(f"Starting ingestion pipeline for: {folder_path}")
        
        initial_state = {
            "folder_path": folder_path,
            "files_processed": 0,
            "errors": []
        }
        
        result = self.app.invoke(initial_state, {"recursion_limit": 1000})
        
        logger.info("Ingestion pipeline completed")
        return result