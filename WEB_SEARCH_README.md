# Web Search RAG System with DuckDuckGo and Ollama

This project implements a web-based Retrieval-Augmented Generation (RAG) system using:
- DuckDuckGo Search API for retrieving information from the web
- Locally downloaded Ollama models for LLM inference
- Flask for the web interface with streaming responses

## Features

- Search the web for information using DuckDuckGo
- Generate answers based on retrieved web content
- Web interface for querying the system
- Streaming responses for real-time answer generation
- Support for multiple Ollama models

## Requirements

- Python 3.8+
- Ollama installed and running locally
- Required Python packages (see below)

## Installation

1. Install the required packages:
   ```
   pip install flask duckduckgo-search langchain langchain-community ollama requests
   ```

2. Ensure Ollama is installed and running with the required models:
   - An LLM model (e.g., `llama2`, `mistral`, etc.)

## Usage

### Web Interface

Start the Flask application:

```
python web_search_app.py
```

Then open your browser and navigate to `http://localhost:5000` to access the web interface.

The web interface allows you to:
- Enter questions or search queries
- Select different Ollama models
- See search results from DuckDuckGo
- Get streaming responses in real-time

## How It Works

1. **User Query**: The user enters a question or search query.

2. **Web Search**: The system searches the web using DuckDuckGo to find relevant information.

3. **Context Retrieval**: The most relevant search results are retrieved and formatted.

4. **Answer Generation**: The Ollama LLM generates an answer based on the retrieved information.

5. **Streaming Response**: The answer is streamed back to the user in real-time.

## Project Structure

- `web_search_app.py`: Flask application for the web search RAG system
- `src/duckduckgo_search.py`: Module for searching the web using DuckDuckGo
- `src/web_rag_system.py`: RAG system that integrates web search and LLM
- `src/ollama_client.py`: Ollama LLM integration with streaming support
- `templates/web_search.html`: HTML template for the web interface

## Customization

You can customize the system by modifying the parameters in the `WebRAGSystem` class:

- `llm_model`: Ollama LLM model name
- `max_results`: Maximum number of search results to use
- `region`: Region for search results
- `safesearch`: SafeSearch setting

## Limitations

- The system relies on DuckDuckGo's search results, which may not always be comprehensive.
- The quality of answers depends on the LLM model used and the relevance of search results.
- Web search may be slower than using a local vector database.

## License

MIT
