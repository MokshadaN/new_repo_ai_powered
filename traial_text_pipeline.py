import os
import pickle
import numpy as np
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
from typing import List, Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')

class PDFRAGSystem:
    def __init__(self, model_name="all-MiniLM-L6-v2", chunk_size=512, chunk_overlap=50):
        """
        Initialize the RAG system with all-MiniLM-L6-v2 model
        
        Args:
            model_name: Name of the Sentence Transformer model
            chunk_size: Size of text chunks for processing
            chunk_overlap: Overlap between consecutive chunks
        """
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model = None
        self.index = None
        self.text_chunks = []
        self.metadata = []
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the all-MiniLM-L6-v2 model"""
        print("Loading all-MiniLM-L6-v2 model...")
        self.model = SentenceTransformer(self.model_name)
        print("Model loaded successfully!")
        print(f"Model max sequence length: {self.model.max_seq_length}")
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text from PDF file
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text as string
        """
        print(f"Extracting text from {pdf_path}...")
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PdfReader(file)
                total_pages = len(pdf_reader.pages)
                
                for page_num in range(total_pages):
                    page = pdf_reader.pages[page_num]
                    page_text = page.extract_text()
                    if page_text:
                        text += f"Page {page_num + 1}: {page_text}\n\n"
                    
                    # Progress indicator
                    if (page_num + 1) % 10 == 0:
                        print(f"Processed {page_num + 1}/{total_pages} pages")
                
                print(f"Successfully extracted text from {total_pages} pages")
                return text
        except Exception as e:
            print(f"Error extracting text from PDF: {e}")
            return ""
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks optimized for all-MiniLM-L6-v2
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of text chunks
        """
        print("Chunking text...")
        
        # Simple sentence-aware chunking
        sentences = text.split('. ')
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # If adding this sentence would exceed chunk size, save current chunk and start new one
            if len(current_chunk) + len(sentence) > self.chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                # Start new chunk with overlap from previous chunk
                overlap_sentences = current_chunk.split('. ')[-self.chunk_overlap//100:]
                current_chunk = '. '.join(overlap_sentences) + '. ' + sentence
            else:
                current_chunk += sentence + '. '
        
        # Add the last chunk if it's not empty
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        print(f"Created {len(chunks)} chunks")
        return chunks
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for text chunks using all-MiniLM-L6-v2
        
        Args:
            texts: List of text chunks
            
        Returns:
            Numpy array of embeddings
        """
        print("Generating embeddings with all-MiniLM-L6-v2...")
        
        # all-MiniLM-L6-v2 automatically normalizes embeddings
        embeddings = self.model.encode(
            texts,
            convert_to_tensor=False,
            show_progress_bar=True,
            normalize_embeddings=True
        )
        
        print(f"Generated embeddings with shape: {embeddings.shape}")
        print(f"Embedding dimension: {embeddings.shape[1]}")
        return embeddings
    
    def build_faiss_index(self, embeddings: np.ndarray):
        """
        Build FAISS index for efficient similarity search
        
        Args:
            embeddings: Numpy array of embeddings
        """
        print("Building FAISS index...")
        dimension = embeddings.shape[1]
        
        # Use Inner Product for cosine similarity (since embeddings are normalized)
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings.astype('float32'))
        
        print("FAISS index built successfully!")
        print(f"Index contains {self.index.ntotal} vectors")
    
    def process_pdf(self, pdf_path: str, save_path: str = None):
        """
        Process PDF: extract text, chunk, generate embeddings, and build index
        
        Args:
            pdf_path: Path to the PDF file
            save_path: Path to save the RAG system state
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        # Extract and chunk text
        text = self.extract_text_from_pdf(pdf_path)
        if not text:
            raise ValueError("No text extracted from PDF")
        
        print(f"Extracted text length: {len(text)} characters")
        
        self.text_chunks = self.chunk_text(text)
        self.metadata = [{
            "chunk_id": i, 
            "length": len(chunk),
            "preview": chunk[:100] + "..." if len(chunk) > 100 else chunk
        } for i, chunk in enumerate(self.text_chunks)]
        
        # Generate embeddings
        embeddings = self.generate_embeddings(self.text_chunks)
        
        # Build FAISS index
        self.build_faiss_index(embeddings)
        
        # Save if path provided
        if save_path:
            self.save(save_path)
    
    def search(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        """
        Search for similar text chunks
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of tuples (text_chunk, similarity_score)
        """
        if self.index is None or not self.text_chunks:
            raise ValueError("RAG system not initialized. Please process a PDF first.")
        
        # Generate query embedding
        query_embedding = self.model.encode(
            [query], 
            normalize_embeddings=True
        ).astype('float32')
        
        # Search in FAISS index
        similarities, indices = self.index.search(query_embedding, k)
        
        # Return results
        results = []
        for idx, score in zip(indices[0], similarities[0]):
            if idx < len(self.text_chunks) and idx >= 0:
                results.append((self.text_chunks[idx], float(score)))
        
        return results
    
    def save(self, file_path: str):
        """
        Save the RAG system state
        
        Args:
            file_path: Path to save the system state
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path) if os.path.dirname(file_path) else '.', exist_ok=True)
        
        state = {
            'text_chunks': self.text_chunks,
            'metadata': self.metadata,
            'model_name': self.model_name,
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap
        }
        
        # Save FAISS index
        faiss.write_index(self.index, f"{file_path}_index.faiss")
        
        # Save other data
        with open(f"{file_path}_data.pkl", 'wb') as f:
            pickle.dump(state, f)
        
        print(f"RAG system saved to {file_path}")
    
    def load(self, file_path: str):
        """
        Load the RAG system state
        
        Args:
            file_path: Path to load the system state from
        """
        if not os.path.exists(f"{file_path}_index.faiss") or not os.path.exists(f"{file_path}_data.pkl"):
            raise FileNotFoundError("Saved RAG system files not found")
        
        # Load FAISS index
        self.index = faiss.read_index(f"{file_path}_index.faiss")
        
        # Load other data
        with open(f"{file_path}_data.pkl", 'rb') as f:
            state = pickle.load(f)
        
        self.text_chunks = state['text_chunks']
        self.metadata = state['metadata']
        self.model_name = state['model_name']
        self.chunk_size = state['chunk_size']
        self.chunk_overlap = state['chunk_overlap']
        
        # Reinitialize model
        self._initialize_model()
        
        print(f"RAG system loaded from {file_path}")
        print(f"Loaded {len(self.text_chunks)} text chunks")

def interactive_search(rag_system):
    """Interactive search interface"""
    print("\n" + "="*60)
    print("Interactive Search Mode")
    print("Type 'quit' to exit, 'info' for system info")
    print("="*60)
    
    while True:
        query = input("\n🔍 Enter your search query: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            break
        
        if query.lower() == 'info':
            print(f"\nSystem Information:")
            print(f"- Model: {rag_system.model_name}")
            print(f"- Chunks: {len(rag_system.text_chunks)}")
            print(f"- Index size: {rag_system.index.ntotal if rag_system.index else 0}")
            continue
        
        if not query:
            continue
        
        try:
            results = rag_system.search(query, k=5)
            
            if not results:
                print("No results found.")
                continue
                
            print(f"\n📚 Found {len(results)} results:")
            for i, (chunk, score) in enumerate(results, 1):
                print(f"\n{i}. 📊 Score: {score:.4f}")
                # Display chunk with better formatting
                preview = chunk[:400] + "..." if len(chunk) > 400 else chunk
                print(f"   {preview}")
                print("-" * 80)
                
        except Exception as e:
            print(f"❌ Search error: {e}")

def main():
    """Main function to demonstrate the RAG system"""
    
    # Initialize RAG system with all-MiniLM-L6-v2
    rag_system = PDFRAGSystem(
        model_name="all-MiniLM-L6-v2",
        chunk_size=512,
        chunk_overlap=50
    )
    
    # Check if we should load existing system or process new PDF
    save_path = "rag_system_mini"
    
    if os.path.exists(f"{save_path}_index.faiss") and os.path.exists(f"{save_path}_data.pkl"):
        load_existing = input("Found existing RAG system. Load it? (y/n): ").lower()
        if load_existing == 'y':
            rag_system.load(save_path)
        else:
            # Process new PDF
            pdf_path = input("Enter PDF file path: ").strip()
            if os.path.exists(pdf_path):
                rag_system.process_pdf(pdf_path, save_path)
            else:
                print("PDF file not found!")
                return
    else:
        # Process new PDF
        pdf_path = input("Enter PDF file path: ").strip()
        if os.path.exists(pdf_path):
            rag_system.process_pdf(pdf_path, save_path)
        else:
            print("PDF file not found!")
            return
    
    # Start interactive search
    interactive_search(rag_system)

# Advanced version with better text processing
class AdvancedPDFRAGSystem(PDFRAGSystem):
    def __init__(self, model_name="all-MiniLM-L6-v2", chunk_size=512, chunk_overlap=50):
        super().__init__(model_name, chunk_size, chunk_overlap)
    
    def clean_text(self, text: str) -> str:
        """Clean extracted text"""
        import re
        
        # Remove excessive whitespace and newlines
        text = re.sub(r'\s+', ' ', text)
        
        # Remove common PDF artifacts
        text = re.sub(r'Page \d+ of \d+', '', text)
        text = re.sub(r'\n+', ' ', text)
        text = re.sub(r'\t+', ' ', text)
        
        return text.strip()
    
    def chunk_text(self, text: str) -> List[str]:
        """Improved text chunking with sentence boundaries"""
        print("Chunking text with sentence-aware splitting...")
        
        cleaned_text = self.clean_text(text)
        
        # Split into sentences (simple approach)
        sentences = re.split(r'[.!?]+', cleaned_text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # If adding this sentence would exceed chunk size
            if len(current_chunk) + len(sentence) > self.chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                # Keep some overlap by taking the last few sentences
                previous_sentences = current_chunk.split('. ')
                overlap_count = max(1, len(previous_sentences) // 3)
                current_chunk = '. '.join(previous_sentences[-overlap_count:]) + '. ' + sentence + '.'
            else:
                current_chunk += sentence + '.'
        
        # Add the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        # Filter out very short chunks
        chunks = [chunk for chunk in chunks if len(chunk) > 50]
        
        print(f"Created {len(chunks)} chunks after cleaning")
        return chunks
    
    def batch_search(self, queries: List[str], k: int = 3) -> Dict[str, List[Tuple[str, float]]]:
        """
        Search multiple queries at once
        
        Args:
            queries: List of search queries
            k: Number of results per query
            
        Returns:
            Dictionary mapping queries to their results
        """
        if self.index is None or not self.text_chunks:
            raise ValueError("RAG system not initialized.")
        
        # Generate embeddings for all queries
        query_embeddings = self.model.encode(
            queries, 
            normalize_embeddings=True
        ).astype('float32')
        
        # Batch search
        similarities, indices = self.index.search(query_embeddings, k)
        
        # Organize results
        results = {}
        for i, query in enumerate(queries):
            query_results = []
            for idx, score in zip(indices[i], similarities[i]):
                if idx < len(self.text_chunks) and idx >= 0:
                    query_results.append((self.text_chunks[idx], float(score)))
            results[query] = query_results
        
        return results

# Example usage
if __name__ == "__main__":
    print("PDF RAG System with all-MiniLM-L6-v2")
    print("=" * 50)
    
    # Choose between basic and advanced system
    system_type = input("Use advanced system? (y/n): ").lower()
    
    if system_type == 'y':
        rag_system = AdvancedPDFRAGSystem()
    else:
        rag_system = PDFRAGSystem()
    
    main()
