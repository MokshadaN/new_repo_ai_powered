"""
Smart semantic search page
"""
import streamlit as st
import os
import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent.parent.parent / "backend"
sys.path.insert(0, str(backend_path))

from backend.orchestration.query_graph import QueryPipeline
from backend.utils.logger import app_logger as logger

st.set_page_config(page_title="Smart Search", page_icon="🔍", layout="wide")

st.title("🔍 Smart Semantic Search")
st.markdown("Search your files using natural language queries")

# Search interface
search_query = st.text_input(
    "What are you looking for?",
    placeholder="e.g., 'Find my tax documents from 2023' or 'Show me photos of beaches'"
)

# Advanced options
with st.expander("Advanced Options"):
    col1, col2 = st.columns(2)
    
    with col1:
        top_k = st.slider("Number of results", 1, 20, 5)
        search_type = st.selectbox(
            "Search type",
            ["All", "Text only", "Images only", "Faces", "Objects"]
        )
    
    with col2:
        enable_rerank = st.checkbox("Enable reranking", value=True)
        show_metadata = st.checkbox("Show metadata", value=False)

# Image upload for multimodal/object search
uploaded_image = st.file_uploader(
    "Upload an image for visual search or object search (optional)",
    type=['png', 'jpg', 'jpeg']
)


def _group_results_by_file(results):
    """Group search results by file_path and keep only the best chunk per file.
    If a face result has no text, fall back to its image path and confidence.
    """
    grouped = {}
    for res in results or []:
        file_path = res.get('file_path') or res.get('image') or res.get('path') or 'Unknown'
        chunk_text = (
            res.get('chunk_content')
            or res.get('content_preview')
            or res.get('content')
            or res.get('context')
            or res.get('text')
        )
        # Handle face results that only carry image paths/confidence
        image_path = res.get('image') or res.get('image_path')
        if not chunk_text and image_path:
            chunk_text = f"Face match: {image_path}"
        # Ensure we still carry a usable path for display
        if not image_path:
            image_path = file_path

        # Prefer final_score (already combines similarity + type weighting), fallback to similarity
        score = res.get('final_score', res.get('similarity', 0))
        similarity = res.get('similarity', res.get('confidence', 0))

        current_best = grouped.get(file_path)
        if current_best is None or score > current_best.get('score', 0):
            grouped[file_path] = {
                'text': chunk_text,
                'similarity': similarity,
                'score': score,
                'type': res.get('type', 'unknown'),
                'chunk_id': res.get('chunk_id'),
                'image_path': image_path,
            }

    # Convert to list of tuples sorted by score desc
    sorted_items = sorted(grouped.items(), key=lambda item: item[1].get('score', 0), reverse=True)
    return sorted_items

# Search buttons (text/face/object via label or image)
col_search, col_obj = st.columns([1, 1])
do_search = col_search.button("🔍 Search (by text/label)", type="primary")
do_object_image = col_obj.button("🖼️ Search Object by Image", type="secondary", help="Upload an image to search objects visually")

if do_search or do_object_image:
    # If object-by-image button pressed without an upload, warn early
    if do_object_image and not uploaded_image:
        st.warning("Please upload an image to search for objects.")
    elif search_query or uploaded_image:
        with st.spinner("Searching..."):
            try:
                query_pipeline = QueryPipeline()
                image_bytes = uploaded_image.read() if uploaded_image else None

                # For object-image button, force empty text so image path is used
                effective_query = "" if do_object_image else search_query

                result = query_pipeline.run(effective_query, image_bytes)

                st.markdown("---")
                st.markdown("## Search Results")
                
                # Show answer
                if result.get('final_response'):
                    st.markdown("### Answer")
                    st.info(result['final_response'])
                
                # Show insights
                if result.get('insights'):
                    with st.expander("📊 Insights"):
                        insights = result['insights']
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Total Results", insights.get('total_results', 0))
                        with col2:
                            st.metric("Avg Similarity", f"{insights.get('average_similarity', 0):.2%}")
                        with col3:
                            types = insights.get('result_types', {})
                            st.metric("Result Types", len(types))
                
                # Show detailed results
                st.markdown("### Detailed Results")
                
                reranked_results = result.get('reranked_results', []) or result.get('results_from_text_faiss', [])
                grouped_results = _group_results_by_file(reranked_results)

                if not grouped_results:
                    st.warning("No results found")
                else:
                    for idx, (file_path, chunk) in enumerate(grouped_results, 1):
                        with st.container():
                            st.markdown(f"**{idx}. {file_path}**")
                            chunk_id = chunk.get('chunk_id')
                            label = f"Chunk {chunk_id}" if chunk_id is not None else "Top Match"
                            st.markdown(f"**{label}:**")

                            if chunk.get('image_path'):
                                st.write(f"**Path:** {chunk['image_path']}")
                                st.metric("Confidence", f"{chunk.get('similarity', 0):.2%}")
                            else:
                                st.markdown(chunk.get('text', 'N/A'))
                                st.metric("Similarity", f"{chunk.get('similarity', 0):.2%}")

                            st.badge(chunk.get('type', 'unknown'))
                            st.divider()

                            if show_metadata:
                                with st.expander("Show metadata"):
                                    st.json(chunk)
                            
                            st.markdown("---")
                
            except Exception as e:
                st.error(f"Search error: {str(e)}")
                logger.error(f"Search error: {e}")
    else:
        st.warning("Please enter a search query or upload an image")

# Search history
with st.sidebar:
    st.markdown("### Recent Searches")
    if 'search_history' not in st.session_state:
        st.session_state.search_history = []
    
    for query in st.session_state.search_history[-5:]:
        st.text(f"• {query}")
