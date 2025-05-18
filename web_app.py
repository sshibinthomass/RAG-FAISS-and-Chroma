"""
Flask application for PDF RAG System with multiple vector store options.
This script provides a web interface to the RAG system with streaming capabilities.
"""

import os
import time
from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from werkzeug.utils import secure_filename

from src.rag_system import RAGSystem, VectorStoreType
from src.ollama_utils import get_available_models
from src.web_rag_system import WebRAGSystem

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
app.config['UPLOAD_FOLDER'] = 'data/pdfs'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Initialize RAG systems
faiss_rag_system = RAGSystem(
    llm_model="llama2",
    embedding_model="nomic-embed-text",
    top_k=5,
    vector_store_type=VectorStoreType.FAISS
)

chroma_rag_system = RAGSystem(
    llm_model="llama2",
    embedding_model="nomic-embed-text",
    top_k=5,
    vector_store_type=VectorStoreType.CHROMA
)

# Add Web RAG system
web_rag_system = WebRAGSystem(
    llm_model="llama2",
    max_results=5
)

# Track active system (default to PDF)
active_system = "pdf"

# Current active RAG system (default to ChromaDB for speed)
active_rag_system = chroma_rag_system

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'pdf'}

def allowed_file(filename):
    """Check if the file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Render the main page."""
    # Get list of available models from Ollama
    current_system = active_rag_system if active_system == "pdf" else web_rag_system
    models = get_available_models(current_system.ollama_client.api_base)
    if not models:
        models = ["llama2"]
    # Get list of indexed PDFs
    pdfs = [f for f in os.listdir(app.config['UPLOAD_FOLDER']) if f.endswith('.pdf')]
    # Get current vector store type (only for PDF)
    vector_store_type = active_rag_system.vector_store_type.value if active_system == "pdf" else None
    return render_template('index.html', 
                          models=models, 
                          pdfs=pdfs, 
                          vector_store_type=vector_store_type,
                          active_system=active_system)

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Reindex after adding the file for both RAG systems
        try:
            faiss_rag_system.add_pdf(file_path)
            chroma_rag_system.add_pdf(file_path)
            return jsonify({'success': True, 'filename': filename}), 200
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'File type not allowed'}), 400

@app.route('/index', methods=['POST'])
def index_documents():
    """Force reindexing of all documents."""
    try:
        active_rag_system.index_documents(force_reindex=True)
        return jsonify({'success': True}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/switch-vector-store', methods=['POST'])
def switch_vector_store():
    """Switch between vector store types."""
    global active_rag_system
    
    data = request.json
    if not data or 'vector_store_type' not in data:
        return jsonify({'error': 'No vector store type provided'}), 400
    
    vector_store_type = data['vector_store_type'].lower()
    
    if vector_store_type == 'faiss':
        active_rag_system = faiss_rag_system
        print("Switched to FAISS vector store")
    elif vector_store_type == 'chroma':
        active_rag_system = chroma_rag_system
        print("Switched to ChromaDB vector store")
    else:
        return jsonify({'error': f'Invalid vector store type: {vector_store_type}'}), 400
    
    return jsonify({
        'success': True, 
        'vector_store_type': active_rag_system.vector_store_type.value
    }), 200

@app.route('/switch-system', methods=['POST'])
def switch_system():
    global active_system
    data = request.json
    if not data or 'system' not in data:
        return jsonify({'error': 'No system type provided'}), 400
    system_type = data['system'].lower()
    if system_type == 'pdf':
        active_system = "pdf"
        print("Switched to PDF RAG system")
    elif system_type == 'web':
        active_system = "web"
        print("Switched to Web RAG system")
    else:
        return jsonify({'error': f'Invalid system type: {system_type}'}), 400
    return jsonify({'success': True, 'active_system': active_system}), 200

@app.route('/query', methods=['POST'])
def query():
    data = request.json
    if not data or 'question' not in data:
        return jsonify({'error': 'No question provided'}), 400
    question = data['question']
    model = data.get('model', 'llama2')
    # Get the current RAG system based on active_system
    current_system = active_rag_system if active_system == "pdf" else web_rag_system
    # Update model if different from current
    if model != current_system.ollama_client.model_name:
        current_system.ollama_client.model_name = model
        current_system.ollama_client.llm = current_system.ollama_client.llm.__class__(
            model=model,
            base_url=current_system.ollama_client.api_base
        )
    try:
        # Get retrieved documents first
        retrieved_docs = current_system.get_retrieved_docs(question)
        # Format documents for display based on the active system
        formatted_docs = []
        if active_system == "pdf":
            for i, doc in enumerate(retrieved_docs):
                formatted_docs.append({
                    'index': i + 1,
                    'source': doc['metadata']['source'],
                    'content': doc['content'][:200] + '...' if len(doc['content']) > 200 else doc['content'],
                    'score': doc['score']
                })
        else:
            for doc in retrieved_docs:
                formatted_docs.append({
                    'index': doc['index'],
                    'title': doc['title'],
                    'content': doc['content'],
                    'url': doc['url']
                })
        response = {
            'documents': formatted_docs,
            'streaming': True,
            'active_system': active_system
        }
        if active_system == "pdf":
            response['vector_store_type'] = active_rag_system.vector_store_type.value
        return jsonify(response), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/stream', methods=['GET', 'POST'])
def stream():
    # Handle both GET and POST requests
    if request.method == 'GET':
        question = request.args.get('question')
        model = request.args.get('model', 'llama2')
    else:
        data = request.json
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        question = data.get('question')
        model = data.get('model', 'llama2')
    if not question:
        return jsonify({'error': 'No question provided'}), 400
    print(f"Stream request received - Question: {question}, Model: {model}")
    # Get the current RAG system based on active_system
    current_system = active_rag_system if active_system == "pdf" else web_rag_system
    # Update model if different from current
    if model != current_system.ollama_client.model_name:
        print(f"Changing model from {current_system.ollama_client.model_name} to {model}")
        current_system.ollama_client.model_name = model
        current_system.ollama_client.llm = current_system.ollama_client.llm.__class__(
            model=model,
            base_url=current_system.ollama_client.api_base
        )
    headers = {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'X-Accel-Buffering': 'no'
    }
    @stream_with_context
    def generate():
        try:
            print(f"Starting streaming response for question: {question} using model: {model}")
            if active_system == "pdf":
                print(f"Using vector store: {active_rag_system.vector_store_type.value}")
            yield "data: Connection established\n\n"
            for chunk in current_system.stream_query(question):
                yield f"data: {chunk}\n\n"
                time.sleep(0.01)
        except Exception as e:
            error_msg = str(e)
            print(f"Error in streaming: {error_msg}")
            yield f"data: Error: {error_msg}\n\n"
        finally:
            print("Streaming completed, sending DONE signal")
            yield "data: [DONE]\n\n"
    return Response(generate(), headers=headers)

@app.route('/models')
def get_models():
    # Get the current RAG system based on active_system
    current_system = active_rag_system if active_system == "pdf" else web_rag_system
    models = get_available_models(current_system.ollama_client.api_base)
    if not models:
        models = ["llama2"]
    return jsonify(models)

@app.route('/vector-stores')
def get_vector_stores():
    """Get list of available vector store types."""
    vector_stores = [vs.value for vs in VectorStoreType]
    return jsonify(vector_stores)

if __name__ == '__main__':
    # Ensure the indexes are loaded
    try:
        faiss_rag_system.index_documents()
        chroma_rag_system.index_documents()
    except Exception as e:
        print(f"Warning: Could not load indexes: {str(e)}")
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)
