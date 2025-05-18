"""
Flask application for Combined RAG System.
This script provides a web interface to both PDF and Web RAG systems with streaming capabilities.
"""

import os
import time
from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from werkzeug.utils import secure_filename

from src.rag_system import RAGSystem
from src.web_rag_system import WebRAGSystem
from src.ollama_utils import get_available_models

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
app.config['UPLOAD_FOLDER'] = 'data/pdfs'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Initialize RAG systems
pdf_rag_system = RAGSystem(
    llm_model="llama2",  # Default model, can be changed via UI
    embedding_model="nomic-embed-text",
    top_k=5
)

web_rag_system = WebRAGSystem(
    llm_model="llama2",  # Default model, can be changed via UI
    max_results=5
)

# Set default active system
active_system = "pdf"  # Can be "pdf" or "web"

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
    models = get_available_models(pdf_rag_system.ollama_client.api_base)

    # If no models are found, provide a default
    if not models:
        models = ["llama2"]

    # Get list of indexed PDFs
    pdfs = [f for f in os.listdir(app.config['UPLOAD_FOLDER']) if f.endswith('.pdf')]

    return render_template('index.html', models=models, pdfs=pdfs, active_system=active_system)

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload."""
    # Only allow file uploads when in PDF mode
    if active_system != "pdf":
        return jsonify({'error': 'File upload is only available in PDF mode'}), 400

    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Reindex after adding the file
        try:
            pdf_rag_system.add_pdf(file_path)
            return jsonify({'success': True, 'filename': filename}), 200
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    return jsonify({'error': 'File type not allowed'}), 400

@app.route('/index', methods=['POST'])
def index_documents():
    """Force reindexing of all documents."""
    # Only allow reindexing when in PDF mode
    if active_system != "pdf":
        return jsonify({'error': 'Reindexing is only available in PDF mode'}), 400

    try:
        pdf_rag_system.index_documents(force_reindex=True)
        return jsonify({'success': True}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/switch-system', methods=['POST'])
def switch_system():
    """Switch between PDF and Web RAG systems."""
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

    return jsonify({
        'success': True,
        'active_system': active_system
    }), 200

@app.route('/query', methods=['POST'])
def query():
    """Handle a query request."""
    data = request.json

    if not data or 'question' not in data:
        return jsonify({'error': 'No question provided'}), 400

    question = data['question']
    model = data.get('model', 'llama2')

    # Get the current RAG system based on active_system
    current_system = pdf_rag_system if active_system == "pdf" else web_rag_system

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
            # Format for PDF RAG system
            for i, doc in enumerate(retrieved_docs):
                formatted_docs.append({
                    'index': i + 1,
                    'source': doc['metadata']['source'],
                    'content': doc['content'][:200] + '...' if len(doc['content']) > 200 else doc['content'],
                    'score': doc['score']
                })
        else:
            # Format for Web RAG system
            for doc in retrieved_docs:
                formatted_docs.append({
                    'index': doc['index'],
                    'title': doc['title'],
                    'content': doc['content'],
                    'url': doc['url']
                })

        return jsonify({
            'documents': formatted_docs,
            'streaming': True,  # Indicate that answer will be streamed
            'active_system': active_system  # Include the active system in the response
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/stream', methods=['GET', 'POST'])
def stream():
    """Stream a response for a query."""
    # Handle both GET and POST requests
    if request.method == 'GET':
        # Get parameters from query string
        question = request.args.get('question')
        model = request.args.get('model', 'llama2')
    else:
        # Get parameters from JSON body
        data = request.json
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        question = data.get('question')
        model = data.get('model', 'llama2')

    # Validate question
    if not question:
        return jsonify({'error': 'No question provided'}), 400

    print(f"Stream request received - Question: {question}, Model: {model}")

    # Get the current RAG system based on active_system
    current_system = pdf_rag_system if active_system == "pdf" else web_rag_system

    # Update model if different from current
    if model != current_system.ollama_client.model_name:
        print(f"Changing model from {current_system.ollama_client.model_name} to {model}")
        current_system.ollama_client.model_name = model
        current_system.ollama_client.llm = current_system.ollama_client.llm.__class__(
            model=model,
            base_url=current_system.ollama_client.api_base
        )

    # Set response headers for SSE
    headers = {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'X-Accel-Buffering': 'no'
    }

    @stream_with_context
    def generate():
        try:
            print(f"Starting streaming response for question: {question} using model: {model}")
            # Send an initial message to establish the connection
            yield "data: Connection established\n\n"

            for chunk in current_system.stream_query(question):
                yield f"data: {chunk}\n\n"
                time.sleep(0.01)  # Small delay to prevent overwhelming the client
        except Exception as e:
            error_msg = str(e)
            print(f"Error in streaming: {error_msg}")
            yield f"data: Error: {error_msg}\n\n"
        finally:
            print("Streaming completed, sending DONE signal")
            yield "data: [DONE]\n\n"

    return Response(generate(), headers=headers)

@app.route('/models')
def get_models_endpoint():
    """Get list of available models from Ollama."""
    # Get the current RAG system based on active_system
    current_system = pdf_rag_system if active_system == "pdf" else web_rag_system

    models = get_available_models(current_system.ollama_client.api_base)

    # If no models are found, provide a default
    if not models:
        models = ["llama2"]

    return jsonify(models)

if __name__ == '__main__':
    # Ensure the PDF index is loaded
    try:
        pdf_rag_system.index_documents()
        print("PDF RAG system initialized successfully")
    except Exception as e:
        print(f"Warning: Could not load PDF index: {str(e)}")

    print("Web RAG system initialized successfully")
    print(f"Starting with active system: {active_system}")

    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)
