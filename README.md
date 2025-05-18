# PDF RAG System with Ollama, Multiple Vector Stores, and Flask

This project implements a Retrieval-Augmented Generation (RAG) system for PDF documents using:
- Locally downloaded Ollama models for LLM inference
- Multiple vector database options:
  - FAISS for memory-efficient storage
  - ChromaDB for faster retrieval performance
- LangChain for orchestrating the RAG pipeline
- Flask for the web interface with streaming responses

## Features

- Load and process PDF documents
- Extract and chunk text from PDFs
- Create vector embeddings using Ollama's embedding models
- Store and search embeddings using multiple vector stores:
  - FAISS: Memory-efficient but slower retrieval
  - ChromaDB: Faster retrieval but higher memory usage
- Generate answers to questions using RAG with Ollama LLMs
- Web interface for uploading PDFs and querying the system
- Streaming responses for real-time answer generation
- Support for multiple Ollama models
- Ability to switch between vector stores on-the-fly

## Requirements

- Python 3.8+
- Ollama installed and running locally
- Required Python packages (see `requirements.txt`)

## Installation

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Ensure Ollama is installed and running with the required models:
   - An LLM model (e.g., `llama2`, `mistral`, etc.)
   - An embedding model (e.g., `nomic-embed-text`)

## Usage

### Web Interface with Multiple Vector Stores

Start the Flask application with support for both FAISS and ChromaDB:

```
python web_app.py
```

Then open your browser and navigate to `http://localhost:5000` to access the web interface.

The web interface allows you to:
- Upload PDF documents
- View indexed PDFs
- Select different Ollama models
- Switch between FAISS and ChromaDB vector stores
- Ask questions about your documents
- See retrieved context documents
- Get streaming responses in real-time

### Original Web Interface (FAISS only)

The original Flask application with only FAISS support is still available:

```
python app.py
```

### Command-Line Interface (Legacy)

The original command-line interface is still available:

```
python main.py --help
```

## Project Structure

- `web_app.py`: Flask application with multiple vector store support
- `app.py`: Original Flask application (FAISS only)
- `main.py`: Command-line interface entry point
- `src/pdf_processor.py`: PDF loading and text processing
- `src/vector_store.py`: FAISS vector database management
- `src/chroma_store.py`: ChromaDB vector database management
- `src/ollama_client.py`: Ollama LLM integration with streaming support
- `src/rag_system.py`: RAG system orchestration with vector store switching
- `templates/`: HTML templates for the web interface
- `static/`: Static files (CSS, JavaScript) for the web interface
- `data/pdfs/`: Directory for PDF documents
- `data/index/`: Directory for FAISS index storage
- `data/chroma_db/`: Directory for ChromaDB storage

## Customization

You can customize the system by modifying the parameters in the `RAGSystem` class:

- `llm_model`: Ollama LLM model name
- `embedding_model`: Ollama embedding model name
- `chunk_size`: Size of text chunks
- `chunk_overlap`: Overlap between chunks
- `top_k`: Number of documents to retrieve for each query
- `vector_store_type`: Type of vector store to use (FAISS or ChromaDB)

## Performance Comparison

### FAISS
- **Pros**: Memory-efficient, works well with large document collections
- **Cons**: Slower retrieval, especially for large indexes

### ChromaDB
- **Pros**: Faster retrieval, simpler API, better for real-time applications
- **Cons**: Higher memory usage, may struggle with very large document collections

Choose the vector store that best fits your use case:
- Use ChromaDB for faster performance with moderate-sized document collections
- Use FAISS for larger document collections where memory efficiency is important

## License

MIT



Summary and Conclusion
I've successfully created a PDF RAG (Retrieval-Augmented Generation) system using locally downloaded Ollama models, FAISS vector database, and LangChain. Here's a summary of what we've accomplished:

Project Structure:
Created a modular architecture with separate components for PDF processing, vector database management, and LLM integration
Implemented a main application with a command-line interface
PDF Processing:
Implemented functionality to load and extract text from PDF files
Created text chunking for better retrieval
Vector Database:
Set up FAISS for efficient similarity search
Implemented embedding generation using Ollama's embedding models
Created functions to save and load the index
LLM Integration:
Integrated with locally running Ollama models
Implemented RAG pipeline for answering questions
User Interface:
Created a command-line interface with various commands
Implemented interactive mode for continuous querying
Testing:
Downloaded a sample PDF (the original RAG paper)
Successfully indexed the document
Tested querying functionality
The system is now fully functional and can be used to:

Add PDF documents to the knowledge base
Index documents for efficient retrieval
Query the system with natural language questions
Get answers based on the content of the indexed PDFs
Next Steps
If you want to enhance the system further, here are some potential improvements:

Update to the latest LangChain API: Fix the deprecation warnings by updating to the latest LangChain API
Add a web interface: Create a simple web UI for easier interaction
Implement document filtering: Allow filtering by document source or metadata
Add support for more document types: Extend beyond PDFs to other document formats
Improve answer quality: Fine-tune prompts and implement re-ranking of retrieved documents
Add citation support: Include citations to source documents in the answers
To use the system, simply:

Place PDF files in the data/pdfs directory
Run python main.py index to index the documents
Run python main.py query "Your question here" to ask questions
Or use python main.py interactive for an interactive session
The system leverages your locally downloaded Ollama models, so there's no need for external API calls, ensuring privacy and reducing costs.