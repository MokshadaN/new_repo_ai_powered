"""
Smart semantic search page with a dark glass UI similar to the reference mock.
"""
import streamlit as st
import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent.parent.parent / "backend"
sys.path.insert(0, str(backend_path))

from backend.orchestration.query_graph import QueryPipeline
from backend.utils.logger import app_logger as logger

st.set_page_config(page_title="Smart Search", page_icon="AI", layout="wide")

st.markdown(
    """
    <style>
    body {
        background: radial-gradient(circle at 20% 20%, rgba(66,105,255,0.15), transparent 40%),
                    radial-gradient(circle at 80% 0%, rgba(158,70,255,0.12), transparent 35%),
                    #0a0c12;
    }
    .block-container { padding: 1.5rem 2rem 3rem 2rem; }
    .glass {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 18px;
        box-shadow: 0 20px 60px rgba(0,0,0,0.45);
        backdrop-filter: blur(16px);
    }
    .soft-card {
        background: linear-gradient(180deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02));
        border: 1px solid rgba(255,255,255,0.05);
        border-radius: 14px;
        padding: 16px;
    }
    .header-bar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 12px 16px;
        border-bottom: 1px solid rgba(255,255,255,0.06);
    }
    .pill {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.08);
        color: #cbd5e1;
        padding: 6px 10px;
        border-radius: 999px;
        font-size: 12px;
    }
    .nav-btn { width: 100%; display: flex; align-items: center; gap: 10px; padding: 10px 12px; border-radius: 12px; color: #cbd5e1; border: 1px solid transparent; }
    .nav-btn:hover { background: rgba(62,123,255,0.12); border-color: rgba(62,123,255,0.4); }
    .nav-btn.active { background: linear-gradient(90deg, rgba(62,123,255,0.18), rgba(62,123,255,0.05)); border-color: rgba(62,123,255,0.5); box-shadow: 0 0 25px rgba(62,123,255,0.25); color: #fff; }
    .table-head, .table-row { display: grid; grid-template-columns: 2fr 1fr 1fr; padding: 10px 14px; align-items: center; }
    .table-head { color: #8b9bb5; font-size: 12px; border-bottom: 1px solid rgba(255,255,255,0.05); }
    .table-row { color: #e2e8f0; font-size: 13px; border-bottom: 1px solid rgba(255,255,255,0.05); }
    .table-row:hover { background: rgba(255,255,255,0.04); }
    .result-card { background: rgba(255,255,255,0.03); border: 1px solid rgba(168,85,247,0.35); border-radius: 14px; padding: 12px; box-shadow: 0 0 25px rgba(168,85,247,0.25); }
    .badge { background: rgba(168,85,247,0.2); border: 1px solid rgba(168,85,247,0.4); color: #e9d5ff; padding: 2px 10px; border-radius: 999px; font-size: 11px; font-weight: 600; }
    .tag { background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.08); padding: 4px 10px; border-radius: 999px; font-size: 11px; color: #cbd5e1; }
    </style>
    """,
    unsafe_allow_html=True,
)


def _group_results_by_file(results):
    """Group search results by file_path and keep only the best chunk per file."""
    grouped = {}
    for res in results or []:
        file_path = res.get("file_path") or res.get("image") or res.get("path") or "Unknown"
        chunk_text = (
            res.get("chunk_content")
            or res.get("content_preview")
            or res.get("content")
            or res.get("context")
            or res.get("text")
        )
        image_path = res.get("image") or res.get("image_path")
        if not chunk_text and image_path:
            chunk_text = f"Face match: {image_path}"
        if not image_path:
            image_path = file_path

        score = res.get("final_score", res.get("similarity", 0))
        similarity = res.get("similarity", res.get("confidence", 0))

        current_best = grouped.get(file_path)
        if current_best is None or score > current_best.get("score", 0):
            grouped[file_path] = {
                "file_path": file_path,
                "text": chunk_text or "No text available",
                "score": score,
                "similarity": similarity,
                "type": res.get("type", "unknown"),
                "chunk_id": res.get("chunk_id"),
                "image_path": image_path,
            }

    return sorted(grouped.items(), key=lambda x: x[1].get("score", 0), reverse=True)


with st.sidebar:
    st.markdown("### Explorer")
    st.markdown('<div class="nav-btn active">Home</div>', unsafe_allow_html=True)
    st.markdown('<div class="nav-btn">Files</div>', unsafe_allow_html=True)
    st.markdown("### Search")
    st.markdown('<div class="nav-btn active">Text Search</div>', unsafe_allow_html=True)
    st.markdown('<div class="nav-btn">Image Search</div>', unsafe_allow_html=True)
    st.markdown('<div class="nav-btn">Face Search</div>', unsafe_allow_html=True)
    st.markdown('<div class="nav-btn">Object Search</div>', unsafe_allow_html=True)
    st.markdown("### Other")
    st.markdown('<div class="nav-btn">Insights</div>', unsafe_allow_html=True)
    st.markdown('<div class="nav-btn">Settings</div>', unsafe_allow_html=True)

st.markdown(
    """
    <div class="glass" style="padding:18px 18px 12px 18px; margin-bottom: 18px;">
      <div class="header-bar">
        <div style="display:flex; align-items:center; gap:12px;">
          <div style="display:flex; gap:6px;">
            <span style="width:10px; height:10px; border-radius:50%; background:#ff5f56;"></span>
            <span style="width:10px; height:10px; border-radius:50%; background:#ffbd2e;"></span>
            <span style="width:10px; height:10px; border-radius:50%; background:#27c93f;"></span>
          </div>
          <div style="color:#e2e8f0; font-weight:700;">AI Explorer</div>
          <div class="pill">Home / Documents / AI Project</div>
        </div>
        <div style="display:flex; gap:10px; align-items:center;">
          <span class="pill">Mode: Auto</span>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("### Smart Semantic Search")

col_top_left, col_top_right = st.columns([2, 1])
with col_top_left:
    search_query = st.text_input(
        "Search documents...",
        placeholder="Find tax documents, AI reports, neural net diagrams...",
    )
with col_top_right:
    st.write("")
    st.write("")
    search_button = st.button("Search", use_container_width=True)

uploaded_image = st.file_uploader(
    "Upload image (optional for image/face/object search)",
    type=["png", "jpg", "jpeg"],
    label_visibility="collapsed",
)
top_k = st.slider("Top-K results", 1, 20, 5, key="topk_slider")
enable_rerank = st.checkbox("Enable reranking", value=True)

answer = None
grouped_results = []

if search_button:
    if not search_query and not uploaded_image:
        st.warning("Enter a query or upload an image to search.")
    else:
        with st.spinner("Running AI search..."):
            try:
                pipeline = QueryPipeline()
                image_bytes = uploaded_image.read() if uploaded_image else None
                result = pipeline.run(search_query, image_bytes)
                answer = result.get("final_response")
                reranked_results = result.get("reranked_results", []) or result.get("results_from_text_faiss", [])
                grouped_results = _group_results_by_file(reranked_results)[:top_k]
            except Exception as exc:
                st.error(f"Search error: {exc}")
                logger.error(f"Search error: {exc}")

left, right = st.columns([1.5, 1])
with left:
    st.markdown('<div class="soft-card" style="margin-bottom:14px;">'
                '<div style="display:flex; justify-content:space-between; align-items:center;">'
                '<div style="color:#e2e8f0; font-weight:700;">File Explorer</div>'
                '<div style="color:#94a3b8; font-size:12px;">70%</div>'
                '</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="glass" style="overflow:hidden;">'
                '<div class="table-head">'
                '<div>Name</div><div>Type</div><div>Size</div>'
                '</div>',
                unsafe_allow_html=True)

    rows = grouped_results or [
        ("Project_Proposal.pdf", {"type": "Document", "similarity": 0.88, "text": "", "size": "1.2MB"}),
        ("Data_Set_v2.csv", {"type": "CSV", "similarity": 0.82, "text": "", "size": "1.2MB"}),
        ("Neural_Net_Diagram.png", {"type": "PNG", "similarity": 0.76, "text": "", "size": "23.7KB"}),
    ]
    table_html = ""
    for name, meta in rows:
        size = meta.get("size") or "-"
        r_type = meta.get("type", "Document")
        table_html += f'<div class="table-row"><div>{name}</div><div>{r_type}</div><div>{size}</div></div>'
    st.markdown(table_html + "</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="soft-card" style="margin-bottom:12px;">'
                '<div style="color:#e2e8f0; font-weight:700; margin-bottom:6px;">Text Search</div>'
                '<div style="color:#94a3b8; font-size:12px;">Mode: Text</div>'
                '</div>', unsafe_allow_html=True)

    if grouped_results:
        top_name, top_meta = grouped_results[0]
        st.markdown(
            f"""
            <div class="result-card">
              <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                <div style="color:#e2e8f0; font-weight:700;">{top_name}</div>
                <span class="badge">Similarity {top_meta.get('similarity',0)*100:.0f}%</span>
              </div>
              <div style="color:#94a3b8; font-size:12px;">{top_meta.get('file_path','')}</div>
              <div style="color:#cbd5e1; margin:8px 0 6px 0; font-size:13px;">{top_meta.get('text','')}</div>
              <div style="display:flex; gap:6px; flex-wrap:wrap;">
                <span class="tag">AI</span><span class="tag">Ethics</span><span class="tag">Report</span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="result-card" style="border-color:rgba(255,255,255,0.05); box-shadow:none;">
              <div style="color:#cbd5e1;">No results yet.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

if answer:
    st.markdown("### LLM Answer")
    st.info(answer)
