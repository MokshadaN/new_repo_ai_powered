"""
Smart semantic search page
"""
import streamlit as st
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

# Image upload for multimodal search
uploaded_image = st.file_uploader(
    "Upload an image for visual search (optional)",
    type=['png', 'jpg', 'jpeg']
)

# Search button
if st.button("🔍 Search", type="primary"):
    if search_query or uploaded_image:
        with st.spinner("Searching..."):
            try:
                # Initialize pipeline
                print("Search 1")
                query_pipeline = QueryPipeline()
                print("Search 2")
                
                # Prepare query
                image_bytes = uploaded_image.read() if uploaded_image else None
                
                print("Search 3")
                # Execute search
                result = query_pipeline.run(search_query, image_bytes)
                
                print("Search 4")
                # Display results
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
                
                reranked_results = result.get('results_from_text_faiss', [])
                
                if not reranked_results:
                    st.warning("No results found")
                else:
                    for idx, res in enumerate(reranked_results[:top_k], 1):
                        with st.container():
                            col1, col2 = st.columns([3, 1])
                            
                            with col1:
                                st.markdown(f"**{idx}. {res.get('file_path', 'Unknown')}**")
                                
                                # Show chunk text (associated with similarity score)
                                chunk_text = res.get('chunk_content', res.get('content_preview', res.get('context', 'N/A')))
                                if chunk_text and chunk_text != 'N/A':
                                    st.markdown("**Relevant Chunk:**")
                                    st.markdown(chunk_text)
                            
                            with col2:
                                similarity = res.get('similarity', 0)
                                st.metric("Similarity", f"{similarity:.2%}")
                                
                                result_type = res.get('type', 'unknown')
                                st.badge(result_type)
                            
                            if show_metadata:
                                with st.expander("Show metadata"):
                                    st.json(res)
                            
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