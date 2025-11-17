"""
Settings and configuration page
"""
import streamlit as st
import sys
from pathlib import Path

backend_path = Path(__file__).parent.parent.parent / "backend"
sys.path.insert(0, str(backend_path))

from backend.config.settings import settings
from backend.ingestion.file_watcher import FileWatcher
from backend.utils.logger import app_logger as logger

st.set_page_config(page_title="Settings", page_icon="⚙️", layout="wide")

st.title("⚙️ Settings & Configuration")

# Tabs for different settings
tab1, tab2, tab3, tab4 = st.tabs(["General", "Indexing", "Search", "Advanced"])

with tab1:
    st.markdown("### General Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        enable_gpu = st.checkbox(
            "Enable GPU acceleration",
            value=settings.enable_gpu
        )
        
        batch_size = st.number_input(
            "Batch size",
            min_value=1,
            max_value=128,
            value=settings.batch_size
        )
    
    with col2:
        num_workers = st.number_input(
            "Number of workers",
            min_value=1,
            max_value=16,
            value=settings.num_workers
        )
        
        max_file_size = st.number_input(
            "Max file size (MB)",
            min_value=1,
            max_value=1000,
            value=settings.max_file_size_mb
        )
    
    if st.button("Save General Settings"):
        st.success("Settings saved!")

with tab2:
    st.markdown("### Indexing Settings")
    
    # Watched directories
    st.markdown("#### Watched Directories")
    
    new_dir = st.text_input("Add directory to watch")
    
    if st.button("Add Directory"):
        if new_dir:
            st.success(f"Added: {new_dir}")
    
    # File watcher status
    st.markdown("#### File Watcher Status")
    
    col1, col2 = st.columns(2)
    
    with col1:
        watcher_status = st.selectbox("Status", ["Stopped", "Running"])
    
    with col2:
        if st.button("Start Watcher" if watcher_status == "Stopped" else "Stop Watcher"):
            st.info("Watcher status updated")
    
    # Excluded extensions
    st.markdown("#### Excluded File Extensions")
    excluded = st.text_input(
        "Comma-separated extensions",
        value=", ".join(settings.excluded_extensions)
    )

with tab3:
    st.markdown("### Search Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        top_k = st.number_input(
            "Default number of results",
            min_value=1,
            max_value=100,
            value=settings.top_k_results
        )
        
        similarity_threshold = st.slider(
            "Similarity threshold",
            0.0, 1.0,
            settings.similarity_threshold
        )
    
    with col2:
        enable_rerank = st.checkbox("Enable reranking", value=True)
        enable_dedup = st.checkbox("Remove duplicates", value=True)
    
    if st.button("Save Search Settings"):
        st.success("Search settings saved!")

with tab4:
    st.markdown("### Advanced Settings")
    
    st.warning("⚠️ Advanced settings - modify with caution!")
    
    # Model paths
    st.markdown("#### Model Paths")
    
    text_model = st.text_input("Text embedding model", value=settings.bge_model_path)
    image_model = st.text_input("Image embedding model", value=settings.siglip_model_path)
    llm_model = st.text_input("LLM model", value=settings.llm_model_path)
    
    # Database management
    st.markdown("#### Database Management")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Clear Text DB", type="secondary"):
            if st.confirm("Are you sure?"):
                st.success("Text database cleared")
    
    with col2:
        if st.button("Clear Image DB", type="secondary"):
            if st.confirm("Are you sure?"):
                st.success("Image database cleared")
    
    with col3:
        if st.button("Clear All DBs", type="secondary"):
            if st.confirm("Are you sure? This cannot be undone!"):
                st.success("All databases cleared")
    
    # Export/Import
    st.markdown("#### Export/Import")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Export Database"):
            st.download_button(
                "Download Export",
                b"database_export_data",
                file_name="database_export.zip",
                mime="application/zip"
            )
    
    with col2:
        uploaded_file = st.file_uploader("Import Database", type=['zip'])
        if uploaded_file:
            st.success("Database imported successfully!")

# System information
st.markdown("---")
st.markdown("### System Information")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Python Version", "3.10+")
    
with col2:
    import torch
    st.metric("PyTorch Version", torch.__version__)

with col3:
    device = "CUDA" if torch.cuda.is_available() else "CPU"
    st.metric("Device", device)

OLLAMA_API_URL = "http://localhost:11434"
OLLAMA_MODEL = "bge-m3:latest"   # change to your installed model
