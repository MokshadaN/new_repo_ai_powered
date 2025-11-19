from typing import Dict, Any, List, Literal, Any
from langgraph.graph import StateGraph, END, START
# from backend.orchestration.state_models import IngestionState
from backend.ingestion.file_scanner import FileScanner
from backend.processors.text_processor import TextProcessor
from backend.processors.image_processors import ImageProcessor
from backend.processors.ocr_processor import OCRProcessor
from backend.embeddings.embedding_manager import EmbeddingManager
from backend.detection.face_detector import FaceDetector
# from backend.detection.object_detector import ObjectDetector
from backend.llm.local_llm import VisionLLM
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

    texts: Optional[List[str]]
    text_chunks : Optional[List[str]]


class IngestionPipeline:
    """LangGraph-based file ingestion pipeline"""
    
    def __init__(self):
        logger.info("Initializing Ingestion Pipeline")
        
        # Initialize components
        # print("I_1")
        self.file_scanner = FileScanner()
        # print("I_2")
        self.text_processor = TextProcessor()
        # print("I_3")
        self.image_processor = ImageProcessor()
        # print("I_4")
        self.ocr_processor = OCRProcessor()
        # print("I_5")
        self.embedding_manager = EmbeddingManager()
        # print("I_6")
        self.face_detector = FaceDetector()
        # print("I_7")
        # self.object_detector = ObjectDetector()
        # print("I_8")
        self.vision_llm = VisionLLM()
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
        workflow.add_node("process_text", self.process_text)
        workflow.add_node("generate_text_embedding", self.generate_text_embedding)
        workflow.add_node("store_text_faiss", self.store_text_faiss)
        workflow.add_node("extract_images_from_text", self.extract_images_from_text)
        
        # # Image path nodes
        workflow.add_node("prepare_images", self.prepare_images)
        workflow.add_node("generate_image_embeddings", self.generate_image_embeddings)
        workflow.add_node("store_images_faiss", self.store_images_faiss)
        workflow.add_node("generate_image_context_llm", self.generate_image_context_llm)
        workflow.add_node("store_context_text_faiss", self.store_context_text_faiss)
        workflow.add_node("detect_faces", self.detect_faces)
        workflow.add_node("store_faces_faiss", self.store_faces_faiss)
        # workflow.add_node("detect_objects", self.detect_objects)
        # workflow.add_node("store_objects_faiss", self.store_objects_faiss)
        
        # # Progress tracking
        workflow.add_node("increment_progress", self.increment_progress)
        
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
        # workflow.add_edge("process_text", END)
        workflow.add_edge("process_text", "generate_text_embedding")
        # workflow.add_edge("generate_text_embedding", END)
        workflow.add_edge("generate_text_embedding", "store_text_faiss")
        # workflow.add_edge("store_text_faiss", END)
        workflow.add_edge("store_text_faiss", "extract_images_from_text")
        
        # # Extract images decision
        # workflow.add_conditional_edges(
        #     "extract_images_from_text",
        #     self.route_after_text_extraction,
        #     {
        #         "process_images": END,
        #         "next_file": END
        #     }
        # )
        workflow.add_conditional_edges(
            "extract_images_from_text",
            self.route_after_text_extraction,
            {
                "process_images": "prepare_images",
                "next_file": "increment_progress"
            }
        )
        
        # # Image processing path (parallel branches)
        # workflow.add_edge("prepare_images", END)
        workflow.add_edge("prepare_images", "generate_image_embeddings")
        # workflow.add_edge("generate_image_embeddings", END)
        workflow.add_edge("generate_image_embeddings", "store_images_faiss")
        workflow.add_edge("store_images_faiss", "generate_image_context_llm")
        # workflow.add_edge("store_images_faiss", END)
        # workflow.add_edge("generate_image_context_llm", END)
        workflow.add_edge("generate_image_context_llm", "store_context_text_faiss")
        #workflow.add_edge("store_context_text_faiss", END)
        workflow.add_edge("store_context_text_faiss", "detect_faces")
        workflow.add_edge("detect_faces",END)
        #workflow.add_edge("detect_faces", "store_faces_faiss")
        # workflow.add_edge("store_faces_faiss", "detect_objects")
        # workflow.add_edge("detect_objects", "store_objects_faiss")
        # workflow.add_edge("store_objects_faiss", "increment_progress")
        
        # # Loop back
        workflow.add_edge("increment_progress", "get_next_file")
        
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
    
    def generate_text_embedding(self, state: IngestionState) -> IngestionState:
        """Generate embedding for text"""
        logger.info("Generating text embedding")
        try:
            extracted = state.get("extracted_text")
            if not extracted:
                logger.warning("No extracted_text available for embedding")
                return {}

            result = self.embedding_manager.generate_text_embedding(extracted)

            if isinstance(result, dict):
                embedding = result.get("embeddings")
                texts = result.get("texts")
            elif isinstance(result, tuple) and len(result) >= 2:
                embedding, texts = result[0], result[1]
            else:
                embedding = result
                texts = extracted

            if isinstance(texts, str):
                texts = [texts]

            # for i, t in enumerate((texts or [])[:10], start=1):
            #     preview = t[:50].replace("\n", " ")
            #     print(f"Chunk {i} preview: {preview}")
            #     logger.info(f"Chunk {i} preview: {preview}")
            logger.info(f"Embeddings generated")
            print("Embedding Generated" , len(embedding))
            print("Texts length",len(texts))
            return {"text_embedding": embedding, "text_chunks": texts}
        except Exception as e:
            logger.error(f"Error generating text embedding: {e}")
            return {}
    def store_text_faiss(self, state: IngestionState) -> IngestionState:
        """Store text embedding in FAISS"""
        logger.info("Storing text embedding")
        try:
            chunks = state.get("text_chunks") or state.get("texts", [])
            print(len(chunks))
            print(len(state["text_embedding"]))
            self.store_manager.store_text(
                embedding=state["text_embedding"],
                chunks = chunks,
                file_path = state["current_file"]
            )
            return {}
        except Exception as e:
            logger.error(f"Error storing text: {e}")
            return {}
    def extract_images_from_text(self, state: IngestionState) -> IngestionState:
        """Extract embedded images from documents"""
        logger.info("Extracting embedded images")
        
        try:
            images = self.text_processor.extract_embedded_images(
                state["current_file"]
            )
            return {"extracted_images_from_docs": images}
        except Exception as e:
            logger.error(f"Error extracting images: {e}")
            return {"extracted_images_from_docs": []}
    
    def route_after_text_extraction(self, state: IngestionState) -> Literal["process_images", "next_file"]:
        """Route based on whether embedded images were found"""
        images = state.get("extracted_images_from_docs", [])
        if images:
            return "process_images"
        return "next_file"
    def file_type_splitter(self, state: IngestionState) -> IngestionState:
        """Split processing based on file type"""
        return {}  # Just a routing node
    
    def route_next_file(self, state: IngestionState) -> Literal["process_file", "end"]:
        """Route based on whether there are more files"""
        if state.get("current_file"):
            return "process_file"
        return "end"
    def process_text(self, state: IngestionState) -> IngestionState:
        """Extract text from document"""
        logger.info(f"Processing text file: {state['current_file']}")
        
        try:
            result = self.text_processor.process(state["current_file"])
            # print("Process_text",result)            
            if result.success:
                # Also try OCR if it's an image-based document
                ocr_text = self.ocr_processor.extract_from_file(state["current_file"])
                
                combined_text = result.data
                if ocr_text:
                    combined_text = f"{result.data}\n\nOCR Extracted:\n{ocr_text}"
                
                return {
                    "extracted_text": combined_text,
                    "ocr_text": ocr_text
                }
            else:
                logger.error(f"Failed to process text: {result.error}")
                errors = state.get("errors", [])
                errors.append(f"{state['current_file']}: {result.error}")
                return {"errors": errors}
                
        except Exception as e:
            logger.error(f"Error processing text: {e}")
            errors = state.get("errors", [])
            errors.append(f"{state['current_file']}: {str(e)}")
            return {"errors": errors}
        
    def prepare_images(self, state: IngestionState) -> IngestionState:
        """Prepare images for processing"""
        logger.info("Preparing images")
        
        # Get images from either direct image file or extracted from docs
        if state.get("current_file_type") == "image":
            images = [state["current_file"]]
        else:
            images = state.get("extracted_images_from_docs", [])
        
        if not images:
            return {}
        
        # Process each image
        processed = []
        for img_path in images:
            try:
                result = self.image_processor.process(img_path)
                if result.success:
                    processed.append(result.data)
            except Exception as e:
                logger.error(f"Error processing image {img_path}: {e}")
        # print({"processed_image": processed})
        return {"processed_image": processed}
    def increment_progress(self, state: IngestionState) -> IngestionState:
        """Update progress counter"""
        processed = state.get("files_processed", 0) + 1
        total = state.get("total_files", 0)
        
        logger.info(f"Progress: {processed}/{total} files processed")
        
        return {"files_processed": processed}
    
    def generate_image_embeddings(self, state: IngestionState) -> IngestionState:
        """Generate embeddings for images"""
        logger.info("Generating image embeddings")
        
        try:
            images = state.get("processed_image", [])
            if not images:
                return {}
            
            embeddings = self.embedding_manager.generate_image_embeddings(images)
            print(embeddings[0][:5])
            return {"image_embedding": embeddings}
        except Exception as e:
            logger.error(f"Error generating image embeddings: {e}")
            return {}
    
    def store_images_faiss(self, state: IngestionState) -> IngestionState:
        """Store image embeddings"""
        logger.info("Storing image embeddings")
        
        try:
            embeddings = state.get("image_embedding", [])
            if not embeddings:
                return {}
            
            # Create metadata for each image
            metadata = [{
                "file_path": state["current_file"],
                "type": "image",
                "index": i
            } for i in range(len(embeddings))]
            
            self.store_manager.store_images(embeddings, metadata)
            return {}
        except Exception as e:
            logger.error(f"Error storing images: {e}")
            return {}
    
    
    def generate_image_context_llm(self, state: IngestionState) -> IngestionState:
        """Generate textual context from images using Vision LLM"""
        logger.info("Generating image context with Vision LLM")
        
        try:
            images = state.get("processed_image", [])
            if not images:
                return {}
            # Rich, search-friendly description prompt (objects, colors, text, setting, actions).
            query = (
                "Describe each image in vivid, search-friendly detail so it is easy to retrieve by query. "
                "Mention key objects, their colors, size, material, and position; any readable text; "
                "the scene/setting; actions/events; camera/view (e.g., close-up, top-down); lighting; "
                "and counts of notable objects. Example: 'A bright red rubber ball on green grass, single ball, "
                "daylight, close-up shot'."
            )

            context = self.vision_llm.generate_context(images, query=query)
            print(context)

            # Embed the generated context; capture chunks and embeddings separately.
            context_result = self.embedding_manager.generate_text_embedding(context)

            if isinstance(context_result, dict):
                context_embedding = context_result.get("embeddings")
                context_chunks = context_result.get("texts") or [context]
            else:
                context_embedding = context_result
                context_chunks = [context]

            return {
                "image_context": context,
                "image_context_chunks": context_chunks,
                "image_context_embedding": context_embedding,
            }
        except Exception as e:
            logger.error(f"Error generating image context: {e}")
            return {}
        
    def store_context_text_faiss(self, state: IngestionState) -> IngestionState:
        """Store image context as text"""
        logger.info("Storing image context")
        
        try:
            embeddings = state.get("image_context_embedding")
            chunks = state.get("image_context_chunks") or []
            if embeddings:
                self.store_manager.store_text(
                    embedding=embeddings,
                    chunks=chunks,
                    file_path=state.get("current_file", "unknown"),
                )
            return {}
        except Exception as e:
            logger.error(f"Error storing context: {e}")
            return {}
    
    def embed_and_store_faces(image_paths):
        detector = FaceDetector()
        vector_store = VectorStoreManager()

        detections = detector.detect(image_paths)
        if not detections:
            print("No faces detected.")
            return False

        stored = vector_store.store_faces(detections)
        print(f"Stored {len(detections)} face embeddings: {stored}")
        return stored

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
