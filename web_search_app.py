"""
Flask application for Web Search RAG System.
This script provides a web interface to the web-based RAG system with streaming capabilities.
"""

import os
import time
from flask import Flask, render_template, request, jsonify, Response, stream_with_context

from src.web_rag_system import WebRAGSystem
from src.ollama_utils import get_available_models

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'

# Initialize Web RAG system
web_rag_system = WebRAGSystem(
    llm_model="llama2",  # Default model, can be changed via UI
    max_results=5
)

@app.route('/')
def index():
    """Render the main page."""
    models = get_available_models(web_rag_system.ollama_client.api_base)
    if not models:
        models = ["llama2"]
    return render_template('web_search.html', models=models)

@app.route('/query', methods=['POST'])
def query():
    """Handle a query request."""
    data = request.json

    if not data or 'question' not in data:
        return jsonify({'error': 'No question provided'}), 400

    question = data['question']
    model = data.get('model', 'llama2')

    # Update model if different from current
    if model != web_rag_system.ollama_client.model_name:
        web_rag_system.ollama_client.model_name = model
        web_rag_system.ollama_client.llm = web_rag_system.ollama_client.llm.__class__(
            model=model,
            base_url=web_rag_system.ollama_client.api_base
        )

    try:
        # Get retrieved documents first
        retrieved_docs = web_rag_system.get_retrieved_docs(question)

        # Format documents for display
        formatted_docs = []
        for doc in retrieved_docs:
            formatted_docs.append({
                'index': doc['index'],
                'title': doc['title'],
                'content': doc['content'],
                'url': doc['url']
            })

        return jsonify({
            'documents': formatted_docs,
            'streaming': True  # Indicate that answer will be streamed
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

    # Update model if different from current
    if model != web_rag_system.ollama_client.model_name:
        print(f"Changing model from {web_rag_system.ollama_client.model_name} to {model}")
        web_rag_system.ollama_client.model_name = model
        web_rag_system.ollama_client.llm = web_rag_system.ollama_client.llm.__class__(
            model=model,
            base_url=web_rag_system.ollama_client.api_base
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

            for chunk in web_rag_system.stream_query(question):
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
    models = get_available_models(web_rag_system.ollama_client.api_base)
    if not models:
        models = ["llama2"]
    return jsonify(models)

if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)
