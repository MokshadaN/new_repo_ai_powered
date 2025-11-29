#Combined Search
"""
Hybrid search with reranking
"""
from typing import List, Dict
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from backend.utils.logger import app_logger as logger


class HybridSearch:
    """Implement hybrid search and reranking"""
    
    def __init__(self, store_manager):
        self.store_manager = store_manager
    
    def rerank(self, results: List[Dict], query: str) -> List[Dict]:
        """Rerank results based on multiple factors"""
        try:
            if not results:
                return results
            
            # Calculate scores
            for result in results:
                score = 0.0
                
                # Similarity score
                similarity = result.get('similarity', 0.5)
                score += similarity * 0.6  # 60% weight on similarity
                
                # Recency score (if timestamp available)
                if 'timestamp' in result:
                    recency_score = self._calculate_recency_score(result['timestamp'])
                    score += recency_score * 0.2  # 20% weight on recency
                
                # Type preference score
                type_score = self._get_type_score(result.get('type', 'unknown'))
                score += type_score * 0.2  # 20% weight on type
                
                result['final_score'] = score
            
            # Sort by final score
            reranked = sorted(results, key=lambda x: x.get('final_score', 0), reverse=True)
            
            return reranked
            
        except Exception as e:
            logger.error(f"Error reranking results: {e}")
            return results
    
    def _calculate_recency_score(self, timestamp) -> float:
        """Calculate recency score (newer = higher)"""
        # Placeholder implementation
        return 0.5
    
    def _get_type_score(self, result_type: str) -> float:
        """Get preference score based on result type"""
        type_preferences = {
            'text': 1.0,
            'image': 0.9,
            'image_context': 0.95,
            'face': 0.8,
            'object': 0.7
        }
        return type_preferences.get(result_type, 0.5)
    
    def deduplicate_results(self, results: List[Dict], threshold: float = 0.95) -> List[Dict]:
        """Keep only the best-scoring chunk per file/image to avoid duplicate listings."""
        if not results:
            return results

        best_by_file = {}
        for result in results:
            file_path = result.get('file_path') or result.get('image') or result.get('path', '')
            if not file_path:
                continue

            current_best = best_by_file.get(file_path)
            # Prefer final_score when available (already includes similarity/type weighting); fall back to similarity.
            score = result.get('final_score', result.get('similarity', 0))
            best_score = current_best.get('final_score', current_best.get('similarity', 0)) if current_best else None
            if best_score is None or score > best_score:
                best_by_file[file_path] = result

        return list(best_by_file.values())
    
    def group_by_file_path(self, results: List[Dict]) -> Dict[str, Dict]:
        """Group results by file path and consolidate context for LLM
        
        Returns a dict mapping file_path -> consolidated context with all results for that file
        Each file entry contains:
        - file_path: path to the file
        - type: type of content (text, image, face, object)
        - full_context: combined text context from all results (for LLM consumption)
        - contexts: list of individual contexts with metadata
        - confidence: best similarity score (0-1)
        - relevance_count: number of matching contexts
        """
        if not results:
            return {}
        
        grouped = {}
        
        for result in results:
            # Get file path from result (check multiple possible keys)
            file_path = result.get('file_path') or result.get('image') or result.get('path', 'unknown')
            
            if file_path not in grouped:
                # Initialize entry with base metadata
                grouped[file_path] = {
                    'file_path': file_path,
                    'type': result.get('type', 'unknown'),
                    'contexts': [],  # Only the best context per file will be kept
                    'full_context': '',  # Combined context for LLM
                    'confidence': result.get('similarity', 0),
                    'relevance_count': 0
                }
            
            # Add this result to contexts, but keep only the best chunk per file (by similarity)
            content = result.get('content', '')
            similarity = result.get('similarity', 0)
            context_entry = {
                'similarity': similarity,
                'distance': result.get('distance', 1),
                'content': content,
                'source_type': result.get('type', 'unknown'),
                'final_score': result.get('final_score', 0)
            }

            grouped[file_path]['relevance_count'] += 1

            existing_contexts = grouped[file_path]['contexts']
            if not existing_contexts or similarity > existing_contexts[0].get('similarity', 0):
                # Replace with the higher-similarity chunk to avoid flooding context with all chunks
                grouped[file_path]['contexts'] = [context_entry]
                grouped[file_path]['confidence'] = similarity
        
        # Build full context for LLM by combining all content
        for file_path, file_data in grouped.items():
            contexts_text = []
            
            for ctx in file_data['contexts']:
                if ctx.get('content'):
                    # Include content with relevance indicator
                    confidence = ctx.get('similarity', 0)
                    contexts_text.append(f"[Relevance: {confidence:.2%}] {ctx['content']}")
            
            # Combine all contexts into a single string for LLM
            file_data['full_context'] = '\n\n'.join(contexts_text) if contexts_text else 'No text content available'
        
        logger.info(f"Grouped {len(results)} results into {len(grouped)} unique files for LLM context")
        return grouped
    
    def format_context_for_llm(self, grouped_results: Dict[str, Dict], top_files: int = 5) -> str:
        """Format grouped results into a single context string for LLM
        
        Args:
            grouped_results: Output from group_by_file_path()
            top_files: Limit to top N most relevant files
            
        Returns:
            Formatted context string ready to pass to LLM
        """
        if not grouped_results:
            return "No relevant context found."
        
        # Sort by confidence score
        sorted_files = sorted(
            grouped_results.items(),
            key=lambda x: x[1].get('confidence', 0),
            reverse=True
        )[:top_files]
        
        context_parts = []
        
        for file_path, file_data in sorted_files:
            # Create a formatted section for each file
            section = f"""
File: {file_path}
Type: {file_data.get('type', 'unknown')}
Confidence: {file_data.get('confidence', 0):.2%}
Relevance Matches: {file_data.get('relevance_count', 0)}
---
{file_data.get('full_context', 'No content')}
"""
            context_parts.append(section)
        
        return "\n".join(context_parts)
    
    def get_context_dict_for_llm(self, grouped_results: Dict[str, Dict]) -> Dict:
        """Return dict specifically formatted for LLM consumption
        
        Returns a dict with:
        - contexts: dict mapping file_path -> consolidated context string
        - metadata: file metadata (type, confidence, match count)
        - summary: overall statistics
        """
        contexts_dict = {}
        metadata_dict = {}
        
        for file_path, file_data in grouped_results.items():
            # Store the full context text
            contexts_dict[file_path] = file_data.get('full_context', '')
            
            # Store metadata separately
            metadata_dict[file_path] = {
                'type': file_data.get('type'),
                'confidence': file_data.get('confidence'),
                'match_count': file_data.get('relevance_count'),
                'num_contexts': len(file_data.get('contexts', []))
            }
        
        return {
            'contexts': contexts_dict,
            'metadata': metadata_dict,
            'summary': {
                'total_files': len(grouped_results),
                'total_contexts': sum(m['match_count'] for m in metadata_dict.values()),
                'avg_confidence': sum(m['confidence'] for m in metadata_dict.values()) / len(metadata_dict) if metadata_dict else 0
            }
        }
