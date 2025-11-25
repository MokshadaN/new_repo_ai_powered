"""
Question answering engine
"""
from typing import Dict, Any, List

from backend.llm.local_llm import OllamaLLM
from backend.utils.logger import app_logger as logger


class QAEngine:
    """Question answering using RAG with grouped file context"""
    
    def __init__(self):
        logger.info("QA Engine initializing")
        self.llm = OllamaLLM()
        logger.info("QA Engine ready")
    
    def answer(self, query: str, context: str) -> str:
        """Generate answer from query and context"""
        try:
            print("build prompt")
            prompt = self._build_prompt(query, context)
            print("generating answer")
            response = self.llm.generate(prompt, max_tokens=512)
            print("generated answer")
            return response
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return "I apologize, but I encountered an error while generating the answer."
    
    def answer_from_grouped_context(self, query: str, grouped_context: Dict[str, Dict]) -> str:
        """Generate answer from query and grouped file context
        
        Args:
            query: User's question
            grouped_context: Output from HybridSearch.group_by_file_path()
        
        Returns:
            Answer from LLM based on the context
        """
        try:
            # Format grouped context into readable string
            context_str = self._format_grouped_context(grouped_context)
            
            # Build and execute prompt
            prompt = self._build_prompt(query, context_str)
            response = self.llm.generate(prompt, max_tokens=512)
            
            logger.info(f"QA Answer generated for query: '{query}'")
            return response
            
        except Exception as e:
            logger.error(f"Error generating answer from grouped context: {e}")
            return "I apologize, but I encountered an error while generating the answer."
    
    def answer_from_llm_dict(self, query: str, llm_context_dict: Dict) -> str:
        """Generate answer from query and LLM-formatted context dict
        
        Args:
            query: User's question
            llm_context_dict: Output from HybridSearch.get_context_dict_for_llm()
        
        Returns:
            Answer from LLM
        """
        try:
            # Extract contexts and build context string
            contexts = llm_context_dict.get('contexts', {})
            context_str = self._format_llm_dict(contexts, llm_context_dict.get('metadata', {}))
            
            # Build and execute prompt
            prompt = self._build_prompt(query, context_str)
            response = self.llm.generate(prompt, max_tokens=512)
            
            logger.info(f"QA Answer generated from LLM dict for query: '{query}'")
            return response
            
        except Exception as e:
            logger.error(f"Error generating answer from LLM dict: {e}")
            return "I apologize, but I encountered an error while generating the answer."
    
    def _format_grouped_context(self, grouped_context: Dict[str, Dict]) -> str:
        """Convert grouped context dict to readable string for LLM"""
        if not grouped_context:
            return "No context available."
        
        context_parts = []
        
        # Sort by confidence score
        sorted_files = sorted(
            grouped_context.items(),
            key=lambda x: x[1].get('confidence', 0),
            reverse=True
        )
        
        for file_path, file_data in sorted_files:
            part = f"""File: {file_path}
Type: {file_data.get('type', 'unknown')}
Confidence: {file_data.get('confidence', 0):.2%}
Matches: {file_data.get('relevance_count', 0)}

{file_data.get('full_context', 'No content available')}
---"""
            context_parts.append(part)
        
        return "\n\n".join(context_parts)
    
    def _format_llm_dict(self, contexts: Dict[str, str], metadata: Dict[str, Dict]) -> str:
        """Convert LLM dict format to readable string"""
        if not contexts:
            return "No context available."
        
        context_parts = []
        
        # Sort by confidence from metadata
        sorted_files = sorted(
            contexts.items(),
            key=lambda x: metadata.get(x[0], {}).get('confidence', 0),
            reverse=True
        )
        
        for file_path, content in sorted_files:
            meta = metadata.get(file_path, {})
            part = f"""File: {file_path}
Type: {meta.get('type', 'unknown')}
Confidence: {meta.get('confidence', 0):.2%}
Matches: {meta.get('match_count', 0)}

{content}
---"""
            context_parts.append(part)
        
        return "\n\n".join(context_parts)
    
    def _build_prompt(self, query: str, context: str) -> str:
        """Build prompt for LLM with context"""
        prompt = f"""Based on the following context from multiple files, please answer the question thoroughly.

===== CONTEXT =====
{context}

===== QUESTION =====
{query}

===== ANSWER ====="""
        return prompt
    
    def generate_insights(self, results: List[Dict]) -> Dict:
        """Generate insights from search results"""
        try:
            insights = {
                'total_results': len(results),
                'top_result': results[0] if results else None,
                'result_types': self._count_result_types(results),
                'average_similarity': self._calculate_avg_similarity(results)
            }
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating insights: {e}")
            return {}
    
    def _count_result_types(self, results: List[Dict]) -> Dict[str, int]:
        """Count results by type"""
        type_counts = {}
        for result in results:
            result_type = result.get('type', 'unknown')
            type_counts[result_type] = type_counts.get(result_type, 0) + 1
        return type_counts
    
    def _calculate_avg_similarity(self, results: List[Dict]) -> float:
        """Calculate average similarity score"""
        if not results:
            return 0.0
        
        similarities = [r.get('similarity', 0) for r in results]
        return sum(similarities) / len(similarities)