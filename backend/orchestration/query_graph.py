#Langgraph Query Pipeline
"""
LangGraph-based query pipeline
Follows the architecture from Image 2
"""
from typing import Literal, Dict, Any
import numpy as np
from langgraph.graph import StateGraph, END
from backend.orchestration.state_models import QueryState
from backend.embeddings.embedding_manager import EmbeddingManager
from backend.config.model_config import ModelRegistry
from backend.config.settings import settings
from backend.llm.local_llm import VisionLLM
from backend.vector_store.store_manager import VectorStoreManager
from backend.llm.qa_engine import QAEngine
from backend.search.hybrid_search import HybridSearch
from backend.utils.logger import app_logger as logger


class QueryPipeline:
    """LangGraph-based query/search pipeline"""
    
    def __init__(self):
        logger.info("Initializing Query Pipeline")
        
        # Initialize components
        self.embedding_manager = EmbeddingManager()
        self.vision_llm = VisionLLM()
        self.store_manager = VectorStoreManager()
        self.qa_engine = QAEngine()
        self.hybrid_search = HybridSearch(self.store_manager)
        
        # Build graph
        self.graph = self._build_graph()
        self.app = self.graph.compile()
        
        logger.info("Query Pipeline initialized successfully")
    
    def _build_graph(self) -> StateGraph:
        """Build the query graph following Image 2 architecture"""
        workflow = StateGraph(QueryState)
        
        # Add nodes
        workflow.add_node("detect_query_type", self.detect_query_type)
        workflow.add_node("generate_text_embedding", self.generate_text_embedding)
        # workflow.add_node("generate_context_and_combine", self.generate_context_and_combine)
        # workflow.add_node("generate_image_embedding", self.generate_image_embedding)
        
    #     # Search nodes
        workflow.add_node("search_text_faiss", self.search_text_faiss)
    #     workflow.add_node("search_image_faiss_from_text", self.search_image_faiss_from_text)
    #     workflow.add_node("search_image_faiss_from_image", self.search_image_faiss_from_image)
    #     workflow.add_node("search_face_faiss", self.search_face_faiss)
    #     workflow.add_node("search_object_faiss", self.search_object_faiss)
        
    #     # Aggregation nodes
        workflow.add_node("aggregate_text_only", self.aggregate_text_only)
    #     workflow.add_node("aggregate_multimodal", self.aggregate_multimodal)
        workflow.add_node("invoke_llm", self.invoke_llm)
        
    #     # Define edges
        workflow.set_entry_point("detect_query_type")

        # Main conditional split
        workflow.add_conditional_edges(
            "detect_query_type",
            lambda state: state["query_type"],
            {
                "text_only": "generate_text_embedding",
                # "text_image": "generate_context_and_combine",
                # "face_search": "generate_image_embedding",
                # "object_search": "generate_image_embedding"
            }
        )

    #     # Text+Image path (parallel branches)
    #     workflow.add_edge("generate_context_and_combine", "generate_text_embedding")
    #     workflow.add_edge("generate_context_and_combine", "generate_image_embedding")
        
    #     # Text-based searches (parallel)
        workflow.add_edge("generate_text_embedding", "search_text_faiss")
    #     workflow.add_edge("generate_text_embedding", "search_image_faiss_from_text")
        
    #     # Image-based searches (parallel)
    #     workflow.add_edge("generate_image_embedding", "search_image_faiss_from_image")
    #     workflow.add_edge("generate_image_embedding", "search_face_faiss")
    #     workflow.add_edge("generate_image_embedding", "search_object_faiss")
        
    #     # Route to appropriate aggregator
    #     workflow.add_conditional_edges(
    #         "search_text_faiss",
    #         self.route_to_aggregator,
    #         {
    #             "aggregate_text_only": "aggregate_text_only",
    #             "aggregate_multimodal": "aggregate_multimodal"
    #         }
    #     )
        
        workflow.add_conditional_edges(
            "search_text_faiss",
            self.route_to_aggregator,
            {
                "aggregate_text_only": "aggregate_text_only",
                # "aggregate_multimodal": "aggregate_multimodal"
            }
        )
        
    #     workflow.add_conditional_edges(
    #         "search_image_faiss_from_text",
    #         self.route_to_aggregator,
    #         {
    #             "aggregate_text_only": "aggregate_text_only",
    #             "aggregate_multimodal": "aggregate_multimodal"
    #         }
    #     )
        
    #     # Image searches always go to multimodal aggregator
    #     workflow.add_edge("search_image_faiss_from_image", "aggregate_multimodal")
    #     workflow.add_edge("search_face_faiss", "aggregate_multimodal")
    #     workflow.add_edge("search_object_faiss", "aggregate_multimodal")
        
    #     # Final steps
        workflow.add_edge("aggregate_text_only", "invoke_llm")
    #     workflow.add_edge("aggregate_multimodal", "invoke_llm")
        workflow.add_edge("invoke_llm", END)

        # Return the constructed workflow
        return workflow
    # # Node implementations
    
    def detect_query_type(self, state: QueryState) -> QueryState:
        """Detect query type"""
        logger.info("Detecting query type")
        
        query_text = state["query_text"].lower()
        has_image = state.get("query_image") is not None
        
        # Determine query type
        if "face" in query_text or "person" in query_text or "who" in query_text:
            if has_image:
                query_type = "face_search"
            else:
                query_type = "text_only"
        
        elif has_image:
            query_type = "text_image"
        else:
            query_type = "text_only"
        
        logger.info(f"Query type: {query_type}")
        return {"query_type": query_type}
    
    def generate_text_embedding(self, state: QueryState) -> QueryState:
        """Generate text query embedding"""
        logger.info("Generating text embedding")
        
        # Use combined query if multimodal, otherwise use original
        text = state.get("combined_query") or state["query_text"]
        
        try:
            # For short queries we use a dedicated query embedding call
            result = self.embedding_manager.generate_query_embedding(text)

            # Accept multiple return formats from the embedder
            if isinstance(result, dict):
                embeddings = result.get("embeddings", [])
                texts = result.get("texts", [])
            elif isinstance(result, (list, tuple)) and len(result) >= 2 and not isinstance(result[0], (int, float)):
                # Could be (embeddings, texts)
                embeddings, texts = result[0], result[1]
            else:
                embeddings = result
                texts = [text]

            # Normalize embeddings: ensure list-of-vectors and texts is list
            if embeddings is None:
                embeddings = []

            # If a single vector was returned, wrap it into a list
            if embeddings and not isinstance(embeddings[0], (list, tuple, np.ndarray)):
                embeddings = [embeddings]

            if texts is None:
                texts = [text]
            elif isinstance(texts, str):
                texts = [texts]

            # Ensure vectors are simple Python lists
            clean_embeddings = []
            for vec in embeddings:
                if hasattr(vec, 'tolist'):
                    try:
                        clean_embeddings.append(list(np.asarray(vec, dtype=np.float32).tolist()))
                    except Exception:
                        clean_embeddings.append(list(vec))
                else:
                    clean_embeddings.append(list(vec))
            logger.info(f"Text embeddings generated (chunks={len(clean_embeddings)}, dim={len(clean_embeddings[0]) if clean_embeddings else 0})")

            # Ensure each chunk vector has the configured embedding dimension
            target_dim = getattr(ModelRegistry.TEXT_EMBEDDING, "dimension", None) or 1024
            normalized_embeddings = []
            for vec in clean_embeddings:
                v = list(vec)
                if len(v) < target_dim:
                    v = v + [0.0] * (target_dim - len(v))
                elif len(v) > target_dim:
                    v = v[:target_dim]
                normalized_embeddings.append(v)

            # Compute an aggregate embedding (mean pooling) with fixed dimension
            if normalized_embeddings:
                agg = np.mean(np.asarray(normalized_embeddings, dtype=np.float32), axis=0)
                agg = list(agg.tolist())
            else:
                agg = [0.0] * target_dim

            # Return all chunk embeddings and texts (no loss of chunk data), plus an aggregate 1024-d vector
            return {
                "text_query_embeddings": normalized_embeddings,
                "text_query_embedding": agg,
                "text_chunks": texts,
            }
        except Exception as e:
            logger.error(f"Error generating text embedding: {e}")
            return {}
    
    # def generate_context_and_combine(self, state: QueryState) -> QueryState:
    #     """Generate image context and combine with text"""
    #     logger.info("Generating image context")
        
    #     try:
    #         context = self.vision_llm.generate_context_from_bytes(
    #             state["query_image"],
    #             state["query_text"]
    #         )
            
    #         combined = f"{state['query_text']} (Image context: {context})"
            
    #         return {
    #             "image_context": context,
    #             "combined_query": combined
    #         }
    #     except Exception as e:
    #         logger.error(f"Error generating context: {e}")
    #         return {"combined_query": state["query_text"]}
    
    # def generate_image_embedding(self, state: QueryState) -> QueryState:
    #     """Generate image query embedding"""
    #     logger.info("Generating image embedding")
        
    #     try:
    #         embedding = self.embedding_manager.generate_image_embedding_from_bytes(
    #             state["query_image"]
    #         )
    #         return {"image_query_embedding": embedding}
    #     except Exception as e:
    #         logger.error(f"Error generating image embedding: {e}")
    #         return {}
    
    # # Search nodes
    
    def search_text_faiss(self, state: QueryState) -> QueryState:
        """Search text FAISS index"""
        logger.info("Searching text FAISS")
        
        try:
            # Prefer per-chunk embeddings if available
            top_k = settings.top_k_results or 20
            if "text_query_embeddings" in state and state.get("text_query_embeddings"):
                per_doc = {}
                for emb in state["text_query_embeddings"]:
                    try:
                        res_list = self.store_manager.search_text(emb, top_k)
                    except Exception as e:
                        logger.error("Error searching chunk: %s", e)
                        continue

                    for r in (res_list or []):
                        file_path = r.get("file_path")
                        if not file_path:
                            continue
                        prev = per_doc.get(file_path)
                        # Keep the best-scoring chunk for the document
                        if not prev or r.get("similarity", 0) > prev.get("similarity", 0):
                            per_doc[file_path] = r

                # Return top-K documents by best chunk similarity
                results = sorted(per_doc.values(), key=lambda x: x.get("similarity", 0), reverse=True)[:top_k]
                # Log and surface the top result's text context if available
                top_text_context = None
                if results:
                    top = results[0]
                    top_text_context = (top.get("chunk_content") or top.get("content_preview") or top.get("content") or None)
                    logger.info(f"Top text result - file: {top.get("file_path")}, chunk: {top.get("chunk_id")}, similarity:{float(top.get("similarity", 0.0))}"),
                    if top_text_context:
                        logger.info(f"Top text context: {top_text_context[:300]}")

                return {"results_from_text_faiss": results}

            # Fallback to single aggregated embedding
            query_emb = state.get("text_query_embedding")
            if query_emb is None:
                logger.warning("No text query embedding available for search")
                return {"results_from_text_faiss": []}

            results = self.store_manager.search_text(query_emb, top_k)

            # Surface top result text context for the aggregated-search fallback
            top_text_context = None
            if results:
                    top = results[0]
                    top_text_context = (top.get("chunk_content") or top.get("content_preview") or top.get("content") or None)
                    logger.info(f"Top text result - file: {top.get("file_path")}, chunk: {top.get("chunk_id")}, similarity:{float(top.get("similarity", 0.0))}"),
                    if top_text_context:
                        logger.info(f"Top text context: {top_text_context[:300]}")
            return {"results_from_text_faiss": results}
        except Exception as e:
            logger.error(f"Error searching text FAISS: {e}")
            return {"results_from_text_faiss": []}
    
    # def search_image_faiss_from_text(self, state: QueryState) -> QueryState:
    #     """Search image FAISS with text embedding"""
    #     logger.info("Searching image FAISS (text query)")
        
    #     try:
    #         results = self.store_manager.search_images_by_text(
    #             state["text_query_embedding"]
    #         )
    #         return {"results_from_image_faiss_text": results}
    #     except Exception as e:
    #         logger.error(f"Error searching images: {e}")
    #         return {"results_from_image_faiss_text": []}
    
    # def search_image_faiss_from_image(self, state: QueryState) -> QueryState:
    #     """Search image FAISS with image embedding"""
    #     logger.info("Searching image FAISS (image query)")
        
    #     try:
    #         results = self.store_manager.search_images(
    #             state["image_query_embedding"]
    #         )
    #         return {"results_from_image_faiss_image": results}
    #     except Exception as e:
    #         logger.error(f"Error searching images: {e}")
    #         return {"results_from_image_faiss_image": []}
    
    # def search_face_faiss(self, state: QueryState) -> QueryState:
    #     """Search face FAISS"""
    #     logger.info("Searching face FAISS")
        
    #     try:
    #         results = self.store_manager.search_faces(
    #             state["image_query_embedding"]
    #         )
    #         return {"results_from_face_faiss": results}
    #     except Exception as e:
    #         logger.error(f"Error searching faces: {e}")
    #         return {"results_from_face_faiss": []}
    
    # def search_object_faiss(self, state: QueryState) -> QueryState:
    #     """Search object FAISS"""
    #     logger.info("Searching object FAISS")
        
    #     try:
    #         results = self.store_manager.search_objects(
    #             state["image_query_embedding"]
    #         )
    #         return {"results_from_object_faiss": results}
    #     except Exception as e:
    #         logger.error(f"Error searching objects: {e}")
    #         return {"results_from_object_faiss": []}
    
    # # Aggregation nodes
    
    def aggregate_text_only(self, state: QueryState) -> QueryState:
        """Aggregate text-only search results"""
        logger.info("Aggregating text-only results")
        
        text_results = state.get("results_from_text_faiss", [])
        image_text_results = state.get("results_from_image_faiss_text", [])
        
        # Combine and format context
        context_parts = []
        
        if text_results:
            context_parts.append("=== Text Documents ===")
            for i, result in enumerate(text_results[:5], 1):
                preview = (
                    result.get("content_preview")
                    or result.get("chunk_content")
                    or result.get("content")
                    or "N/A"
                )
                file_path = result.get("file_path", "unknown file")
                context_parts.append(f"{i}. {file_path}: {preview}")
        
        if image_text_results:
            context_parts.append("\n=== Images (via text search) ===")
            for i, result in enumerate(image_text_results[:5], 1):
                context_parts.append(f"{i}. {result.get('context', 'N/A')}")
        
        context = "\n".join(context_parts)
        # print(context)
        
        # Rerank results
        all_results = text_results + image_text_results
        reranked = self.hybrid_search.rerank(all_results, state["query_text"])
        
        return {
            "aggregated_context": context,
            "reranked_results": reranked
        }
    
    # def aggregate_multimodal(self, state: QueryState) -> QueryState:
    #     """Aggregate all multimodal search results"""
    #     logger.info("Aggregating multimodal results")
        
    #     text_results = state.get("results_from_text_faiss", [])
    #     image_text_results = state.get("results_from_image_faiss_text", [])
    #     image_results = state.get("results_from_image_faiss_image", [])
    #     face_results = state.get("results_from_face_faiss", [])
    #     object_results = state.get("results_from_object_faiss", [])
        
    #     # Combine all context
    #     context_parts = []
        
    #     if text_results:
    #         context_parts.append("=== Text Documents ===")
    #         for i, result in enumerate(text_results[:3], 1):
    #             context_parts.append(f"{i}. {result.get('content_preview', 'N/A')}")
        
    #     if image_text_results:
    #         context_parts.append("\n=== Images (text-based search) ===")
    #         for i, result in enumerate(image_text_results[:3], 1):
    #             context_parts.append(f"{i}. {result.get('context', 'N/A')}")
        
    #     if image_results:
    #         context_parts.append("\n=== Images (visual similarity) ===")
    #         for i, result in enumerate(image_results[:3], 1):
    #             context_parts.append(f"{i}. Image: {result.get('file_path', 'N/A')}")
        
    #     if face_results:
    #         context_parts.append("\n=== Detected Faces ===")
    #         for i, result in enumerate(face_results[:3], 1):
    #             context_parts.append(f"{i}. Face in: {result.get('image', 'N/A')}")
        
    #     if object_results:
    #         context_parts.append("\n=== Detected Objects ===")
    #         for i, result in enumerate(object_results[:3], 1):
    #             context_parts.append(f"{i}. {result.get('label', 'N/A')} in {result.get('image', 'N/A')}")
        
    #     context = "\n".join(context_parts)
        
    #     # Rerank all results
    #     all_results = (text_results + image_text_results + image_results + 
    #                   face_results + object_results)
    #     reranked = self.hybrid_search.rerank(all_results, state.get("combined_query") or state["query_text"])
        
    #     return {
    #         "aggregated_context": context,
    #         "reranked_results": reranked
    #     }
    
    def invoke_llm(self, state: QueryState) -> QueryState:
        """Generate final response using LLM"""
        logger.info("Invoking LLM for final response")
        
        try:
            print("Combined query")
            query = state.get("combined_query") or state["query_text"]
            print("Aggregated Context")
            context = state.get("aggregated_context", "")
            
            print("QA 1")
            response = self.qa_engine.answer(query, context)
            print("QA 2")
            
            # Generate insights
            insights = self.qa_engine.generate_insights(state["reranked_results"])
            
            return {
                "final_response": response,
                "insights": insights
            }
        except Exception as e:
            logger.error(f"Error invoking LLM: {e}")
            return {
                "final_response": "I apologize, but I encountered an error generating the response.",
                "insights": {}
            }
    
    # # Router
    
    def route_to_aggregator(self, state: QueryState) -> Literal["aggregate_text_only"]:
        """
        Route to the text-only aggregator.
        Multimodal path is disabled; always return aggregate_text_only to avoid missing node errors.
        """
        return "aggregate_text_only"
    
    def run(self, query_text: str, query_image: bytes = None) -> Dict:
        """Execute the query pipeline"""
        logger.info(f"Starting query pipeline: {query_text[:50]}...")
        print("Query Pipeline 1")
        initial_state = {
            "query_text": query_text,
            "query_image": query_image,
            "results_from_text_faiss": [],
            "results_from_image_faiss_text": [],
            "results_from_image_faiss_image": [],
            "results_from_face_faiss": [],
            "results_from_object_faiss": []
        }
        print("Query Pipeline 2")
        print(initial_state)
        
        result = self.app.invoke(initial_state, {"recursion_limit": 100})
        print("Query Pipeline 3")
        
        logger.info("Query pipeline completed")
        print("Query Pipeline 4")
        return result
