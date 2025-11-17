"""
Main Streamlit application
"""
import streamlit as st
from pathlib import Path
import sys

# Add backend to path
backend_path = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_path))

from backend.utils.logger import setup_logger

# Setup
st.set_page_config(
    page_title="AI Disk Analyzer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize logger
logger = setup_logger()

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stat-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🔍 AI Disk Analyzer</h1>', unsafe_allow_html=True)
st.markdown("### Local AI-Powered File Management System")

# Sidebar
with st.sidebar:
    st.image("https://via.placeholder.com/150x150.png?text=AI+Disk", width=True)
    st.markdown("---")
    st.markdown("## Navigation")
    st.markdown("Use the pages in the sidebar to:")
    st.markdown("- 🔍 Search your files")
    st.markdown("- 🖼️ Find images")
    st.markdown("- 📁 Organize content")
    st.markdown("- 💡 Get insights")
    st.markdown("---")
    
    # Quick stats
    st.markdown("### System Status")
    
    # try:
    #     from backend.vectorstore.store_manager import VectorStoreManager
    #     store_manager = VectorStoreManager()
    #     stats = store_manager.get_statistics()
        
    #     st.metric("Text Documents", stats.get('text_count', 0))
    #     st.metric("Images", stats.get('image_count', 0))
    #     st.metric("Faces", stats.get('face_count', 0))
    #     st.metric("Objects", stats.get('object_count', 0))
    # except Exception as e:
    #     st.warning("Unable to load statistics")
    #     logger.error(f"Error loading stats: {e}")

# Main content
st.markdown("## Welcome to AI Disk Analyzer")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<div class="stat-box">', unsafe_allow_html=True)
    st.markdown("### 🔍 Smart Search")
    st.markdown("Natural language search across all your documents and images")
    if st.button("Go to Search", key="search_btn"):
        # st.switch_page("D:/BTP/ai_powered_file_explorer/frontend/pages/Smart_Search.py")
        st.switch_page("./pages/Smart_Search.py")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="stat-box">', unsafe_allow_html=True)
    st.markdown("### 🖼️ Image Search")
    st.markdown("Find images by content, faces, or objects")
    if st.button("Go to Images", key="image_btn"):
        # st.switch_page("D:/BTP/ai_powered_file_explorer/frontend/pages/Image_Search.py")
        st.switch_page("./pages/Image_Search.py")
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="stat-box">', unsafe_allow_html=True)
    st.markdown("### 📁 Organization")
    st.markdown("Smart file clustering and duplicate detection")
    if st.button("Go to Organization", key="org_btn"):
        # st.switch_page("D:/BTP/ai_powered_file_explorer/frontend/pages/Smart_Search.py")
        st.switch_page("./pages/Smart_Search.py")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")

# Quick actions
st.markdown("## Quick Actions")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Index New Folder")
    folder_path = st.text_input("Enter folder path to index")
    
    if st.button("Start Indexing"):
        if folder_path:
            with st.spinner("Indexing files..."):
                try:
                    print("111111111111111111111111")
                    from backend.orchestration.ingestion_graph import IngestionPipeline
                    print("2222222222222222222222")
                    pipeline = IngestionPipeline()
                    print("3333333333333")
                    result = pipeline.run(folder_path)
                    print("44444444444444")
                    
                    st.success(f"✅ Indexed {result.get('files_processed', 0)} files!")
                    st.balloons()
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    logger.error(f"Indexing error: {e}")
        else:
            st.warning("Please enter a folder path")

with col2:
    st.markdown("### Recent Activity")
    st.info("No recent activity")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using LangGraph, Streamlit, and Local AI Models")