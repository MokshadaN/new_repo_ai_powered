# """
# Image search page
# """
# import streamlit as st
# import sys
# from pathlib import Path
# from PIL import Image

# backend_path = Path(__file__).parent.parent.parent / "backend"
# sys.path.insert(0, str(backend_path))

# from backend.search.face_search import FaceSearch
# from backend.search.semantic_search import SemanticSearch
# from backend.utils.logger import app_logger as logger

# st.set_page_config(page_title="Image Search", page_icon="🖼️", layout="wide")

# st.title("🖼️ Image Search")
# st.markdown("Search images by content, faces, or objects")

# # Search mode selector
# search_mode = st.radio(
#     "Search mode:",
#     ["Text to Image", "Image to Image", "Face Search", "Object Search"],
#     horizontal=True
# )

# # Initialize search engines
# semantic_search = SemanticSearch()
# face_search = FaceSearch()

# if search_mode == "Text to Image":
#     st.markdown("### Search images using text descriptions")
    
#     query = st.text_input(
#         "Describe what you're looking for",
#         placeholder="e.g., 'sunset over mountains' or 'group photo at birthday party'"
#     )
    
#     if st.button("Search Images"):
#         if query:
#             with st.spinner("Searching images..."):
#                 try:
#                     results = semantic_search.search_by_type(query, 'image', top_k=20)
                    
#                     if results:
#                         st.success(f"Found {len(results)} images")
                        
#                         # Display in grid
#                         cols = st.columns(4)
#                         for idx, result in enumerate(results):
#                             with cols[idx % 4]:
#                                 img_path = result.get('file_path') or result.get('path')
                                
#                                 try:
#                                     img = Image.open(img_path)
#                                     st.image(img, width=True)
#                                     st.caption(f"Similarity: {result.get('similarity', 0):.2%}")
#                                 except:
#                                     st.error("Unable to load image")
#                     else:
#                         st.warning("No images found")
                        
#                 except Exception as e:
#                     st.error(f"Error: {str(e)}")
#                     logger.error(f"Image search error: {e}")
#         else:
#             st.warning("Please enter a search query")

# elif search_mode == "Image to Image":
#     st.markdown("### Find similar images")
    
#     uploaded_file = st.file_uploader(
#         "Upload a reference image",
#         type=['png', 'jpg', 'jpeg']
#     )
    
#     if uploaded_file:
#         col1, col2 = st.columns([1, 2])
        
#         with col1:
#             st.image(uploaded_file, caption="Reference image", width=True)
        
#         with col2:
#             if st.button("Find Similar Images"):
#                 with st.spinner("Searching..."):
#                     try:
#                         # Save temporarily
#                         temp_path = Path("/tmp") / uploaded_file.name
#                         with open(temp_path, "wb") as f:
#                             f.write(uploaded_file.getvalue())
                        
#                         # Search
#                         from backend.embeddings.embedding_manager import EmbeddingManager
#                         from backend.vectorstore.store_manager import VectorStoreManager
                        
#                         embedding_manager = EmbeddingManager()
#                         store_manager = VectorStoreManager()
                        
#                         # Generate embedding
#                         embedding = embedding_manager.generate_image_embedding(str(temp_path))
                        
#                         # Search
#                         results = store_manager.search_images(embedding, top_k=20)
                        
#                         if results:
#                             st.success(f"Found {len(results)} similar images")
                            
#                             # Display results
#                             cols = st.columns(4)
#                             for idx, result in enumerate(results):
#                                 with cols[idx % 4]:
#                                     img_path = result.get('file_path') or result.get('path')
                                    
#                                     try:
#                                         img = Image.open(img_path)
#                                         st.image(img, width=True)
#                                         st.caption(f"Similarity: {result.get('similarity', 0):.2%}")
#                                     except:
#                                         st.error("Unable to load image")
#                         else:
#                             st.warning("No similar images found")
                            
#                     except Exception as e:
#                         st.error(f"Error: {str(e)}")
#                         logger.error(f"Image similarity search error: {e}")

# elif search_mode == "Face Search":
#     st.markdown("### Search for people in your images")
    
#     uploaded_file = st.file_uploader(
#         "Upload a face image",
#         type=['png', 'jpg', 'jpeg']
#     )
    
#     if uploaded_file:
#         col1, col2 = st.columns([1, 2])
        
#         with col1:
#             st.image(uploaded_file, caption="Reference face", width='stretch')
        
#         with col2:
#             if st.button("Find This Person"):
#                 with st.spinner("Searching for faces..."):
#                     try:
#                         # Save temporarily
#                         temp_path = Path("/tmp") / uploaded_file.name
#                         with open(temp_path, "wb") as f:
#                             f.write(uploaded_file.getvalue())
                        
#                         # Search
#                         results = face_search.search_by_face_image(str(temp_path), top_k=20)
                        
#                         if results:
#                             st.success(f"Found {len(results)} matches")
                            
#                             # Display results
#                             cols = st.columns(4)
#                             for idx, result in enumerate(results):
#                                 with cols[idx % 4]:
#                                     img_path = result.get('image')
#                                     bbox = result.get('bbox', [])
                                    
#                                     try:
#                                         img = Image.open(img_path)
                                        
#                                         # Crop to face if bbox available
#                                         if bbox and len(bbox) == 4:
#                                             img = img.crop(bbox)
                                        
#                                         st.image(img, width=True)
#                                         st.caption(f"Confidence: {result.get('confidence', 0):.2%}")
#                                     except:
#                                         st.error("Unable to load image")
#                         else:
#                             st.warning("No matches found")
                            
#                     except Exception as e:
#                         st.error(f"Error: {str(e)}")
#                         logger.error(f"Face search error: {e}")

# else:  # Object Search
#     st.markdown("### Search for specific objects in images")
    
#     object_query = st.text_input(
#         "What object are you looking for?",
#         placeholder="e.g., 'car', 'laptop', 'cat'"
#     )
    
#     if st.button("Search for Object"):
#         if object_query:
#             with st.spinner("Searching..."):
#                 try:
#                     results = semantic_search.search_by_type(object_query, 'object', top_k=20)
                    
#                     if results:
#                         st.success(f"Found {len(results)} images with '{object_query}'")
                        
#                         # Display results
#                         cols = st.columns(4)
#                         for idx, result in enumerate(results):
#                             with cols[idx % 4]:
#                                 img_path = result.get('image')
                                
#                                 try:
#                                     img = Image.open(img_path)
#                                     st.image(img, width=True)
                                    
#                                     label = result.get('label', 'unknown')
#                                     confidence = result.get('confidence', 0)
#                                     st.caption(f"{label} ({confidence:.2%})")
#                                 except:
#                                     st.error("Unable to load image")
#                     else:
#                         st.warning(f"No images found with '{object_query}'")
                        
#                 except Exception as e:
#                     st.error(f"Error: {str(e)}")
#                     logger.error(f"Object search error: {e}")
#         else:
#             st.warning("Please enter an object to search for")

# # Sidebar filters
# with st.sidebar:
#     st.markdown("### Filters")
    
#     date_range = st.date_input("Date range", [])
#     file_types = st.multiselect("File types", ["JPG", "PNG", "GIF", "BMP"])
#     min_similarity = st.slider("Min similarity", 0.0, 1.0, 0.5)
"""
Image search page - Fixed Version
"""
import streamlit as st
import sys
import os
from pathlib import Path
from PIL import Image
import tempfile

backend_path = Path(__file__).parent.parent.parent / "backend"
sys.path.insert(0, str(backend_path))

from backend.search.face_search import FaceSearch
from backend.search.semantic_search import SemanticSearch
from backend.orchestration.query_graph import QueryPipeline
from backend.utils.logger import app_logger as logger

st.set_page_config(page_title="Image Search", page_icon="🖼️", layout="wide")

st.title("🖼️ Image Search")
st.markdown("Search images by content, faces, or objects")

# Default similarity threshold (used only for Image→Image flow)
if "image_search_min_sim" not in st.session_state:
    st.session_state["image_search_min_sim"] = 0.90

# Search mode selector
search_mode = st.radio(
    "Search mode:",
    ["Text to Image", "Image to Image", "Face Search", "Object Search"],
    horizontal=True
)

# Initialize search engines
def initialize_search_engines():
    """Initialize search engines with error handling"""
    try:
        semantic_search = SemanticSearch()
        face_search = FaceSearch()
        query_pipeline = QueryPipeline()
        return semantic_search, face_search, query_pipeline
    except Exception as e:
        st.error(f"Failed to initialize search engines: {e}")
        logger.error(f"Search engine initialization error: {e}")
        return None, None, None

# Initialize and check if successful
semantic_search, face_search, query_pipeline = initialize_search_engines()
if semantic_search is None or face_search is None or query_pipeline is None:
    st.stop()

def display_image_results(results, image_key='file_path', similarity_key='similarity', min_similarity: float | None = None):
    """Display search results with proper error handling"""
    threshold = float(min_similarity) if min_similarity is not None else 0.0

    normalized = []
    for r in (results or []):
        raw = r.get(similarity_key, 0) or 0
        # FAISS IP can be [-1, 1]; shift to [0,1] for filtering and display
        scaled = max(0.0, min(1.0, (raw + 1.0) / 2.0))
        r = dict(r)
        r["_scaled_similarity"] = scaled
        normalized.append(r)

    filtered = [r for r in normalized if r["_scaled_similarity"] >= threshold]

    if not filtered:
        st.warning("No images found")
        return
    
    filtered = sorted(filtered, key=lambda r: r.get(similarity_key, 0), reverse=True)
    st.success(f"Found {len(filtered)} images")
    
    # Display in grid
    cols = st.columns(4)
    
    for idx, result in enumerate(filtered):
        col_idx = idx % 4
        with cols[col_idx]:
            # Try multiple possible image path keys
            img_path = None
            for key in [image_key, 'path', 'file_path', 'image_path', 'image']:
                if key in result and result[key]:
                    img_path = result[key]
                    break
            
            if not img_path:
                st.error("No image path")
                show_similarity(result, similarity_key)
                continue
            
            # Convert to string and check if file exists
            img_path = str(img_path)
            if not os.path.exists(img_path):
                st.error(f"File not found")
                st.text(f"Path: {os.path.basename(img_path)}")
                show_similarity(result, similarity_key)
                continue
            
            try:
                # Load and display image
                img = Image.open(img_path)
                
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Resize for consistent display
                img.thumbnail((200, 200))
                
                # Display image
                st.image(img, width='stretch')
                
                # Show similarity/confidence
                show_similarity(result, similarity_key)
                
                # Show filename
                filename = os.path.basename(img_path)
                st.caption(f"{filename}")
                
            except Exception as e:
                st.error("Unable to load image")
                st.text(f"Error: {str(e)[:50]}...")
                show_similarity(result, similarity_key)

def show_similarity(result, similarity_key='similarity'):
    """Display similarity score with proper formatting"""
    raw = result.get(similarity_key, 0) or 0
    scaled = result.get("_scaled_similarity")

    if scaled is None:
        # Fallback: map [-1,1] to [0,1]
        scaled = max(0.0, min(1.0, (raw + 1.0) / 2.0))

    st.write(f"Similarity: {scaled * 100:.2f}%")

def save_uploaded_file(uploaded_file):
    """Save uploaded file to temporary location"""
    try:
        # Create temp directory if it doesn't exist
        temp_dir = Path(tempfile.gettempdir()) / "image_search"
        temp_dir.mkdir(exist_ok=True)
        
        temp_path = temp_dir / uploaded_file.name
        
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        
        return str(temp_path)
    except Exception as e:
        logger.error(f"Error saving uploaded file: {e}")
        return None

def filter_object_hits(results, needle: str, min_confidence: float = 0.25):
    """Keep only object hits that match the label text and meet a confidence floor."""
    if not results:
        return []

    query = (needle or "").strip().lower()
    filtered = []
    for res in results:
        res_type = str(res.get("type", "")).lower()
        label = str(res.get("label", "")).lower()
        confidence = res.get("confidence", res.get("similarity", 0)) or 0

        # Only keep object-type hits unless the label explicitly matches the query
        if res_type and res_type != "object" and (not query or query not in label):
            continue

        # Enforce label match when a query is provided
        if query and label and query not in label:
            continue

        if confidence < min_confidence:
            continue

        # Normalize fields so the UI can display consistently
        cleaned = dict(res)
        cleaned["confidence"] = confidence
        cleaned.setdefault("label", label or needle or "object")
        filtered.append(cleaned)

    return sorted(filtered, key=lambda r: r.get("confidence", 0), reverse=True)

# Main search logic
if search_mode == "Text to Image":
    st.markdown("### Search images using text descriptions")
    
    query = st.text_input(
        "Describe what you're looking for",
        value="group photo",  # Default value for testing
        placeholder="e.g., 'sunset over mountains' or 'group photo at birthday party'"
    )
    
    if st.button("Search Images"):
        if query:
            with st.spinner("Searching images..."):
                try:
                    # Debug: Show what we're searching for
                    st.write(f"Searching for: '{query}'")
                    
                    results = semantic_search.search_by_type(query, 'image', top_k=20)
                    
                    # Debug: Show raw results
                    if results:
                        st.write(f"Raw results sample: {results[0].keys()}")
                    
                    display_image_results(results, min_similarity=0.0)
                        
                except Exception as e:
                    st.error(f"Search error: {str(e)}")
                    logger.error(f"Image search error: {e}")
        else:
            st.warning("Please enter a search query")

elif search_mode == "Image to Image":
    st.markdown("### Find similar images")
    
    uploaded_file = st.file_uploader(
        "Upload a reference image",
        type=['png', 'jpg', 'jpeg']
    )
    
    if uploaded_file:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.image(uploaded_file, caption="Reference image", width='stretch')
        
        with col2:
            if st.button("Find Similar Images"):
                with st.spinner("Searching..."):
                    try:
                        # Save uploaded file
                        temp_path = save_uploaded_file(uploaded_file)
                        
                        if not temp_path:
                            st.error("Failed to save uploaded image")
                        else:
                            # Search using semantic search
                            results = semantic_search.search_similar_images(temp_path, top_k=20)
                            display_image_results(results, min_similarity=st.session_state.get("image_search_min_sim", 0.9))
                            
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        logger.error(f"Image similarity search error: {e}")

elif search_mode == "Face Search":
    st.markdown("### Search for people in your images")
    
    uploaded_file = st.file_uploader(
        "Upload a face image",
        type=['png', 'jpg', 'jpeg']
    )
    
    if uploaded_file:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.image(uploaded_file, caption="Reference face", width='stretch')
        
        with col2:
            if st.button("Find This Person"):
                with st.spinner("Searching for faces..."):
                    try:
                        # Save uploaded file
                        temp_path = save_uploaded_file(uploaded_file)
                        
                        if not temp_path:
                            st.error("Failed to save uploaded image")
                        else:
                            # Search for faces
                            results = face_search.search_by_face_image(temp_path, top_k=20)
                            
                            if results:
                                st.success(f"Found {len(results)} matches")
                                
                                # Display face-specific results
                                cols = st.columns(4)
                                for idx, result in enumerate(results):
                                    with cols[idx % 4]:
                                        img_path = result.get('image')
                                        bbox = result.get('bbox', [])
                                        
                                        if not img_path or not os.path.exists(str(img_path)):
                                            st.error("Image not found")
                                            st.write(f"Confidence: {result.get('confidence', 0):.2%}")
                                            continue
                                        
                                        try:
                                            img = Image.open(img_path)
                                            
                                            # Crop to face if bbox available
                                            if bbox and len(bbox) == 4:
                                                x1, y1, x2, y2 = map(int, bbox)
                                                # Ensure coordinates are within image bounds
                                                width, height = img.size
                                                x1 = max(0, min(x1, width-1))
                                                y1 = max(0, min(y1, height-1))
                                                x2 = max(0, min(x2, width))
                                                y2 = max(0, min(y2, height))
                                                
                                                if x2 > x1 and y2 > y1:  # Valid bbox
                                                    img = img.crop((x1, y1, x2, y2))
                                            
                                            img.thumbnail((200, 200))
                                            st.image(img, width='stretch')
                                            st.write(f"Confidence: {result.get('confidence', 0):.2%}")
                                            
                                        except Exception as img_error:
                                            st.error("Unable to load image")
                                            st.write(f"Confidence: {result.get('confidence', 0):.2%}")
                                            logger.error(f"Face image display error: {img_error}")
                            else:
                                st.warning("No matches found")
                            
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        logger.error(f"Face search error: {e}")

else:  # Object Search
    st.markdown("### Search for specific objects in images")
    
    object_query = st.text_input(
        "What object are you looking for?",
        value="car",  # Default for testing
        placeholder="e.g., 'car', 'laptop', 'cat'"
    )

    uploaded_obj = st.file_uploader(
        "Optional: upload an image to search for objects visually",
        type=['png', 'jpg', 'jpeg'],
        key="object_image_upload"
    )
    
    if st.button("Search for Object"):
        if object_query or uploaded_obj:
            with st.spinner("Searching..."):
                try:
                    # Prepare image bytes if provided
                    image_bytes = uploaded_obj.read() if uploaded_obj else None
                    result = query_pipeline.run(object_query, image_bytes)

                    raw_results = result.get('reranked_results', []) or result.get('results_from_object_faiss', [])
                    object_hits = filter_object_hits(raw_results, object_query)

                    if not object_hits:
                        st.warning(f"No images found with '{object_query or 'image'}'")
                    else:
                        st.success(f"Found {len(object_hits)} object hits")
                        cols = st.columns(4)
                        for idx, res in enumerate(object_hits):
                            with cols[idx % 4]:
                                img_path = res.get('image') or res.get('image_path') or res.get('file_path')
                                if not img_path or not os.path.exists(str(img_path)):
                                    st.error("Image not found")
                                    continue
                                try:
                                    img = Image.open(img_path)
                                    img.thumbnail((200, 200))
                                    st.image(img, width='stretch')
                                    label = res.get('label', 'object')
                                    confidence = res.get('confidence', res.get('similarity', 0))
                                    st.write(f"{label} ({confidence:.2%})")
                                except Exception as img_error:
                                    st.error("Unable to load image")
                                    logger.error(f"Object image display error: {img_error}")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    logger.error(f"Object search error: {e}")
        else:
            st.warning("Please enter an object or upload an image")

# Sidebar filters
with st.sidebar:
    st.markdown("### Filters")
    
    date_range = st.date_input("Date range", [])
    file_types = st.multiselect("File types", ["JPG", "PNG", "GIF", "BMP"])
    min_similarity = st.slider("Min similarity (Image → Image)", 0.0, 1.0, st.session_state.get("image_search_min_sim", 0.90))
    st.session_state["image_search_min_sim"] = min_similarity
    
    # Debug section
    st.markdown("---")
    st.markdown("### Debug Info")
    if st.checkbox("Show debug information"):
        st.write(f"Backend path: {backend_path}")
        st.write(f"Backend exists: {backend_path.exists()}")
        
        # Test image loading
        test_images = list(Path(".").glob("*.jpg")) + list(Path(".").glob("*.png"))
        if test_images:
            st.write(f"Found test images: {[img.name for img in test_images[:3]]}")
