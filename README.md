# new_repo_ai_powered
# AI-Powered Disk Analyzer

A production-ready, fully local AI system for intelligent file management, semantic search, and content organization.

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)

## 🌟 Features

### Core Capabilities
- **🔍 Semantic Search**: Natural language queries across all documents and images
- **🖼️ Visual Search**: Find images by content, similarity, or upload reference images
- **👤 Face Recognition**: Search for people across your photo library
- **🎯 Object Detection**: Find images containing specific objects
- **📊 Smart Organization**: Automatic file clustering based on content similarity
- **🔄 Duplicate Detection**: Find exact and near-duplicate files
- **💡 AI Insights**: Summarization, keyword extraction, and Q&A
- **⚡ Real-time Monitoring**: Automatic indexing of new/modified files

### Privacy First
- ✅ 100% local processing
- ✅ No cloud API calls
- ✅ All data stays on your device
- ✅ No telemetry or tracking

## 🏗️ Architecture

### Technology Stack
- **Backend**: Python, LangChain, LangGraph
- **Vector Stores**: ChromaDB, FAISS
- **Models**:
  - Text: BGE-M3 (1024-dim embeddings)
  - Images: SigLIP (768-dim embeddings)
  - Faces: RetinaFace + ArcFace
  - Objects: YOLOv8
  - LLM: Mistral-7B (local inference)
- **Frontend**: Streamlit (multipage app)
- **OCR**: Tesseract + EasyOCR

### Pipeline Architecture
The system uses two main LangGraph pipelines:

1. **Ingestion Pipeline**: Processes files, extracts content, generates embeddings
2. **Query Pipeline**: Handles searches with multimodal support and intelligent routing

## 📦 Installation

### Prerequisites
- Python 3.10 or higher
- 16GB RAM (minimum)
- 50GB free disk space (for models)
- GPU recommended but not required

### Quick Start
```bash
# Clone repository
git clone <repository-url>
cd ai-disk-analyzer

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your preferences

# Download models
python scripts/download_models.py

# Initialize databases
python scripts/setup_database.py
source venv/Scripts/activate
# Run application
streamlit run frontend/app.py
python -m streamlit run frontend/app.py
```

The application will open in your browser at `http://localhost:8501`

## 🚀 Usage

### Indexing Files

1. Open the app and go to the home page
2. Enter a folder path in "Index New Folder"
3. Click "Start Indexing"
4. Wait for processing to complete

### Searching

**Text Search:**
```
"Find my tax documents from 2023"
"Show me presentations about machine learning"
```

**Image Search:**
```
"Find photos of beaches"
"Show me images with cars"
```

**Face Search:**
- Upload a reference face image
- System finds all images with that person

### Organization

- View automatic file clusters
- Find and manage duplicates
- Get insights and statistics

## 📁 Project Structure
```
ai-disk-analyzer/
├── backend/
│   ├── config/              # Configuration management
│   ├── ingestion/           # File scanning and monitoring
│   ├── processors/          # Text, image, OCR processors
│   ├── embeddings/          # Embedding generation
│   ├── detection/           # Face and object detection
│   ├── vectorstore/         # Vector database operations
│   ├── orchestration/       # LangGraph pipelines
│   ├── search/              # Search implementations
│   ├── llm/                 # LLM integration
│   ├── analysis/            # Clustering and insights
│   └── utils/               # Utilities
├── frontend/
│   ├── app.py               # Main Streamlit app
│   └── pages/               # Multipage components
├── data/
│   ├── vector_stores/       # Vector databases
│   ├── models/              # Downloaded models
│   └── logs/                # Application logs
├── scripts/
│   ├── download_models.py   # Model downloader
│   ├── setup_database.py    # DB initialization
│   └── benchmark.py         # Performance testing
└── requirements.txt
```

## 🔧 Configuration

Edit `.env` to customize:
```bash
# Model settings
BGE_MODEL_PATH=BAAI/bge-m3
ENABLE_GPU=true

# Processing
BATCH_SIZE=32
MAX_FILE_SIZE_MB=100

# Search
TOP_K_RESULTS=5
SIMILARITY_THRESHOLD=0.6
```

## 📊 Performance

Benchmark your system:
```bash
python scripts/benchmark.py
```

**Expected Performance** (with GPU):
- Text embedding: ~50 texts/sec
- Image embedding: ~10 images/sec
- Vector search: ~100 searches/sec

## 🐛 Troubleshooting

### Common Issues

**Out of Memory:**
- Reduce `BATCH_SIZE` in `.env`
- Enable GPU if available
- Process files in smaller batches

**Slow Processing:**
- Ensure GPU is enabled
- Check `ENABLE_GPU=true` in `.env`
- Reduce model sizes if needed

**Models Not Found:**
- Run `python scripts/download_models.py` again
- Check internet connection
- Verify HuggingFace access

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- LangChain & LangGraph for orchestration
- HuggingFace for model hosting
- Streamlit for the frontend framework
- All open-source contributors

## 📞 Support

For issues and questions:
- GitHub Issues: [Create an issue]
- Documentation: [Read the docs]

---

**Made with ❤️ for privacy-conscious AI enthusiasts**