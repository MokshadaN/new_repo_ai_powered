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
    def __init__(self, model_name="BAAI/bge-m3", chunk_size=256, chunk_overlap=50):
        """
        Initialize the RAG system with BGE-M3 model
        
        Args:
            model_name: Name of the BGE model to use
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
        """Initialize the BGE-M3 model"""
        print("Loading BGE-M3 model...")
        self.model = SentenceTransformer(self.model_name)
        print("Model loaded successfully!")
    
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
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text += f"Page {page_num + 1}: {page_text}\n\n"
            print(f"Successfully extracted text from {len(pdf_reader.pages)} pages")
            return text
        except Exception as e:
            print(f"Error extracting text from PDF: {e}")
            return ""
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of text chunks
        """
        print("Chunking text...")
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk = ' '.join(words[i:i + self.chunk_size])
            chunks.append(chunk)
            
            if i + self.chunk_size >= len(words):
                break
        
        print(f"Created {len(chunks)} chunks")
        return chunks
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for text chunks using BGE-M3
        
        Args:
            texts: List of text chunks
            
        Returns:
            Numpy array of embeddings
        """
        print("Generating embeddings...")
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        print(f"Generated embeddings with shape: {embeddings.shape}")
        return embeddings
    
    def build_faiss_index(self, embeddings: np.ndarray):
        """
        Build FAISS index for efficient similarity search
        
        Args:
            embeddings: Numpy array of embeddings
        """
        print("Building FAISS index...")
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        self.index.add(embeddings.astype('float32'))
        print("FAISS index built successfully!")
    
    def process_pdf(self, pdf_path: str, save_path: str = None):
        """
        Process PDF: extract text, chunk, generate embeddings, and build index
        
        Args:
            pdf_path: Path to the PDF file
            save_path: Path to save the RAG system state
        """
        # Extract and chunk text
        text = self.extract_text_from_pdf(pdf_path)
        if not text:
            raise ValueError("No text extracted from PDF")
        
        self.text_chunks = self.chunk_text(text)
        self.metadata = [{"chunk_id": i, "length": len(chunk)} 
                        for i, chunk in enumerate(self.text_chunks)]
        
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
        query_embedding = self.model.encode([query], normalize_embeddings=True)
        query_embedding = query_embedding.astype('float32')
        
        # Search in FAISS index
        similarities, indices = self.index.search(query_embedding, k)
        
        # Return results
        results = []
        for idx, score in zip(indices[0], similarities[0]):
            if idx < len(self.text_chunks):
                results.append((self.text_chunks[idx], float(score)))
        
        return results
    
    def save(self, file_path: str):
        """
        Save the RAG system state
        
        Args:
            file_path: Path to save the system state
        """
        state = {
            'text_chunks': self.text_chunks,
            'metadata': self.metadata,
            'model_name': self.model_name,
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap
        }
        
        # Save FAISS index separately
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

def main():
    """Example usage of the PDF RAG system"""
    print("in main")
    # Initialize RAG system
    rag_system = PDFRAGSystem()
    print("rag system called")
    
    # Process PDF file
    pdf_path = "F:/test_folder/test_folder/Unit 2_ On the Language of Literature.pptx [Read-Only] (1).pdf"
    save_path = "rag_system"
    
    try:
        # Process the PDF
        rag_system.process_pdf(pdf_path, save_path)
        
        # Example searches
        queries = [
            "WHo did king agamemnon fight with"
        ]
        
        for query in queries:
            print(f"\n{'='*50}")
            print(f"Query: {query}")
            print(f"{'='*50}")
            
            results = rag_system.search(query, k=3)
            
            for i, (chunk, score) in enumerate(results, 1):
                print(f"\nResult {i} (Score: {score:.4f}):")
                print(f"{chunk}..." if len(chunk) > 200 else chunk)
                
    except Exception as e:
        print(f"Error: {e}")

def interactive_search(rag_system):
    """Interactive search interface"""
    print("\nInteractive Search Mode")
    print("Type 'quit' to exit")
    
    while True:
        query = input("\nEnter your search query: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            break
        
        if not query:
            continue
        
        try:
            results = rag_system.search(query, k=5)
            
            print(f"\nFound {len(results)} results:")
            for i, (chunk, score) in enumerate(results, 1):
                print(f"\n{i}. Score: {score:.4f}")
                print(f"Text: {chunk[:300]}..." if len(chunk) > 300 else chunk)
                print("-" * 80)
                
        except Exception as e:
            print(f"Search error: {e}")

if __name__ == "__main__":
    # You can either run the main example or use interactive mode
    
    # Option 1: Run with a specific PDF
    print("calling main")
    main()
    print("main done ")
    
    # Option 2: Interactive mode (uncomment below)
    rag_system = PDFRAGSystem()
    
    # Load existing system or process new PDF
    try:
        rag_system.load("rag_system")
        print("Loaded existing RAG system")
    except:
        print("No existing system found. Please process a PDF first.")
        pdf_path = input("Enter PDF file path: ")
        if os.path.exists(pdf_path):
            rag_system.process_pdf(pdf_path, "rag_system")
        else:
            print("PDF file not found!")
            exit()
    
    interactive_search(rag_system)
