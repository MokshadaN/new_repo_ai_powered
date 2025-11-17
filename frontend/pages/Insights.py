"""
Insights and analytics page
"""
import streamlit as st
import sys
from pathlib import Path

backend_path = Path(__file__).parent.parent.parent / "backend"
sys.path.insert(0, str(backend_path))

from backend.llm.summarizer import Summarizer
from backend.llm.qa_engine import QAEngine
from backend.utils.logger import app_logger as logger

st.set_page_config(page_title="Insights", page_icon="💡", layout="wide")

st.title("💡 AI Insights")
st.markdown("Generate insights, summaries, and analysis")

# Feature selector
feature = st.selectbox(
    "Select feature",
    ["Document Summarization", "Keyword Extraction", "Compare Documents", "Q&A"]
)

summarizer = Summarizer()
qa_engine = QAEngine()

if feature == "Document Summarization":
    st.markdown("### Summarize Documents")
    
    text_input = st.text_area(
        "Enter text to summarize",
        height=200,
        placeholder="Paste your document text here..."
    )
    
    max_length = st.slider("Summary length (words)", 50, 500, 200)
    
    if st.button("Generate Summary"):
        if text_input:
            with st.spinner("Generating summary..."):
                try:
                    summary = summarizer.summarize_text(text_input, max_length)
                    
                    st.markdown("### Summary")
                    st.info(summary)
                    
                    # Download button
                    st.download_button(
                        "Download Summary",
                        summary,
                        file_name="summary.txt",
                        mime="text/plain"
                    )
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    logger.error(f"Summarization error: {e}")
        else:
            st.warning("Please enter some text to summarize")

elif feature == "Keyword Extraction":
    st.markdown("### Extract Keywords")
    
    text_input = st.text_area(
        "Enter text",
        height=200,
        placeholder="Paste your text here..."
    )
    
    num_keywords = st.slider("Number of keywords", 5, 20, 10)
    
    if st.button("Extract Keywords"):
        if text_input:
            with st.spinner("Extracting keywords..."):
                try:
                    keywords = summarizer.extract_keywords(text_input, num_keywords)
                    
                    st.markdown("### Keywords")
                    
                    # Display as tags
                    cols = st.columns(5)
                    for idx, keyword in enumerate(keywords):
                        with cols[idx % 5]:
                            st.button(keyword, key=f"kw_{idx}", disabled=True)
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    logger.error(f"Keyword extraction error: {e}")
        else:
            st.warning("Please enter some text")

elif feature == "Compare Documents":
    st.markdown("### Compare Two Documents")
    
    col1, col2 = st.columns(2)
    
    with col1:
        doc1 = st.text_area("Document 1", height=200)
    
    with col2:
        doc2 = st.text_area("Document 2", height=200)
    
    if st.button("Compare Documents"):
        if doc1 and doc2:
            with st.spinner("Comparing documents..."):
                try:
                    comparison = summarizer.compare_documents(doc1, doc2)
                    
                    st.markdown("### Comparison")
                    st.info(comparison)
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    logger.error(f"Comparison error: {e}")
        else:
            st.warning("Please enter both documents")

else:  # Q&A
    st.markdown("### Ask Questions About Your Documents")
    
    context = st.text_area(
        "Context (document content)",
        height=200,
        placeholder="Paste the document content here..."
    )
    
    question = st.text_input(
        "Your question",
        placeholder="What do you want to know about this document?"
    )
    
    if st.button("Get Answer"):
        if context and question:
            with st.spinner("Generating answer..."):
                try:
                    answer = qa_engine.answer(question, context)
                    
                    st.markdown("### Answer")
                    st.success(answer)
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    logger.error(f"Q&A error: {e}")
        else:
            st.warning("Please provide both context and question")