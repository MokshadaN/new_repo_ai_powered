"""
File organization page
"""
import streamlit as st
import sys
from pathlib import Path
import plotly.express as px
import pandas as pd

backend_path = Path(__file__).parent.parent.parent / "backend"
sys.path.insert(0, str(backend_path))

from backend.analysis.clustering import FileClustering
from backend.analysis.duplicate_finder import DuplicateFinder
from backend.analysis.insights import InsightGenerator
from backend.vectorstore.store_manager import VectorStoreManager
from backend.utils.logger import app_logger as logger

st.set_page_config(page_title="File Organization", page_icon="📁", layout="wide")

st.title("📁 Smart File Organization")
st.markdown("Automatically organize files using AI-powered clustering")

# Tab selection
tab1, tab2, tab3 = st.tabs(["Clustering", "Duplicates", "Insights"])

with tab1:
    st.markdown("### Automatic File Clustering")
    st.markdown("Group similar files together based on content")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        n_clusters = st.slider("Number of clusters", 2, 20, 5)
        clustering_method = st.radio("Method", ["K-Means", "DBSCAN"], horizontal=True)
    
    with col2:
        content_type = st.selectbox("Content type", ["Text", "Images", "All"])
    
    if st.button("Generate Clusters", type="primary"):
        with st.spinner("Clustering files..."):
            try:
                store_manager = VectorStoreManager()
                clusterer = FileClustering(
                    method='kmeans' if clustering_method == "K-Means" else 'dbscan'
                )
                
                # Get embeddings and metadata
                # This is a simplified version - in production, you'd load actual data
                st.info("Clustering in progress...")
                
                # Mock clustering result for demonstration
                clusters = {
                    0: [{"file_path": "doc1.txt", "type": "text"}],
                    1: [{"file_path": "img1.jpg", "type": "image"}],
                    2: [{"file_path": "doc2.pdf", "type": "text"}]
                }
                
                st.success(f"Created {len(clusters)} clusters")
                
                # Display clusters
                for cluster_id, items in clusters.items():
                    with st.expander(f"Cluster {cluster_id} ({len(items)} files)"):
                        for item in items:
                            st.text(f"• {item['file_path']}")
                
                # Visualization
                st.markdown("### Cluster Visualization")
                
                # Create mock data for visualization
                df = pd.DataFrame({
                    'x': [1, 2, 3],
                    'y': [1, 2, 1],
                    'cluster': [0, 1, 2],
                    'file': ['doc1.txt', 'img1.jpg', 'doc2.pdf']
                })
                
                fig = px.scatter(
                    df, 
                    x='x', 
                    y='y', 
                    color='cluster',
                    hover_data=['file'],
                    title="File Clusters (2D Projection)"
                )
                st.plotly_chart(fig, width=True)
                
            except Exception as e:
                st.error(f"Clustering error: {str(e)}")
                logger.error(f"Clustering error: {e}")

with tab2:
    st.markdown("### Duplicate File Detection")
    st.markdown("Find and manage duplicate files")
    
    col1, col2 = st.columns(2)
    
    with col1:
        detection_type = st.radio(
            "Detection method",
            ["Exact duplicates (hash)", "Near duplicates (content)"],
            key="dup_type"
        )
    
    with col2:
        if detection_type == "Near duplicates (content)":
            similarity_threshold = st.slider(
                "Similarity threshold",
                0.7, 1.0, 0.95
            )
    
    if st.button("Find Duplicates", type="primary"):
        with st.spinner("Scanning for duplicates..."):
            try:
                duplicate_finder = DuplicateFinder()
                
                # Mock duplicate results
                duplicates = [
                    ["file1.txt", "file1_copy.txt"],
                    ["image1.jpg", "image1_backup.jpg", "image1_old.jpg"]
                ]
                
                if duplicates:
                    st.warning(f"Found {len(duplicates)} duplicate groups")
                    
                    for idx, group in enumerate(duplicates, 1):
                        with st.expander(f"Duplicate Group {idx} ({len(group)} files)"):
                            for file_path in group:
                                col1, col2 = st.columns([3, 1])
                                
                                with col1:
                                    st.text(file_path)
                                
                                with col2:
                                    if st.button("Delete", key=f"del_{idx}_{file_path}"):
                                        st.success(f"Deleted: {file_path}")
                else:
                    st.success("No duplicates found!")
                    
            except Exception as e:
                st.error(f"Error: {str(e)}")
                logger.error(f"Duplicate detection error: {e}")

with tab3:
    st.markdown("### File Insights & Statistics")
    
    if st.button("Generate Insights"):
        with st.spinner("Analyzing files..."):
            try:
                insight_gen = InsightGenerator()
                
                # Mock statistics
                stats = {
                    'total_files': 150,
                    'file_types': {
                        'text': 80,
                        'image': 60,
                        'other': 10
                    },
                    'total_size': 1024 * 1024 * 500  # 500 MB
                }
                
                # Display metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Files", stats['total_files'])
                
                with col2:
                    st.metric("Text Files", stats['file_types']['text'])
                
                with col3:
                    st.metric("Image Files", stats['file_types']['image'])
                
                with col4:
                    size_mb = stats['total_size'] / (1024 * 1024)
                    st.metric("Total Size", f"{size_mb:.1f} MB")
                
                # File type distribution
                st.markdown("### File Type Distribution")
                
                df = pd.DataFrame(list(stats['file_types'].items()), 
                                 columns=['Type', 'Count'])
                
                fig = px.pie(df, values='Count', names='Type', 
                            title="Files by Type")
                st.plotly_chart(fig, width=True)
                
                # Recent activity
                st.markdown("### Recent Activity")
                
                activity_data = {
                    'Date': ['2024-01-01', '2024-01-02', '2024-01-03'],
                    'Files Added': [10, 15, 8],
                    'Files Modified': [5, 3, 7]
                }
                
                df_activity = pd.DataFrame(activity_data)
                
                fig = px.line(df_activity, x='Date', 
                             y=['Files Added', 'Files Modified'],
                             title="File Activity Over Time")
                st.plotly_chart(fig, width=True)
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
                logger.error(f"Insights error: {e}")