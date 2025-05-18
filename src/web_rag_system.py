"""
Web RAG System Module.
This module integrates web search and LLM components for a web-based RAG system.
"""

import os
from typing import List, Dict, Any, Optional, Generator

from src.duckduckgo_search import DuckDuckGoSearch
from src.ollama_client import OllamaClient

class WebRAGSystem:
    """Class for the web-based RAG system."""
    
    def __init__(
        self,
        llm_model: str = "llama2",
        max_results: int = 5,
        region: str = "wt-wt",
        safesearch: str = "moderate"
    ):
        """
        Initialize the web RAG system.
        
        Args:
            llm_model: Ollama LLM model name
            max_results: Maximum number of search results to use
            region: Region for search results
            safesearch: SafeSearch setting
        """
        self.max_results = max_results
        
        # Initialize components
        self.search_engine = DuckDuckGoSearch(
            max_results=max_results,
            region=region,
            safesearch=safesearch
        )
        self.ollama_client = OllamaClient(model_name=llm_model)
    
    def query(self, question: str) -> Dict[str, Any]:
        """
        Process a query through the web RAG system.
        
        Args:
            question: User question
            
        Returns:
            Dictionary with answer and retrieved documents
        """
        # Search for relevant information
        search_query = self._generate_search_query(question)
        retrieved_docs = self.search_engine.search(search_query, self.max_results)
        
        # Generate answer using RAG
        answer = self.ollama_client.answer_with_rag(question, retrieved_docs)
        
        return {
            "question": question,
            "search_query": search_query,
            "answer": answer,
            "retrieved_documents": retrieved_docs
        }
    
    def stream_query(self, question: str) -> Generator[str, None, None]:
        """
        Process a query through the web RAG system with streaming response.
        
        Args:
            question: User question
            
        Yields:
            Chunks of the generated answer
        """
        # Search for relevant information
        search_query = self._generate_search_query(question)
        retrieved_docs = self.search_engine.search(search_query, self.max_results)
        
        # Stream answer using RAG
        yield from self.ollama_client.stream_answer_with_rag(question, retrieved_docs)
    
    def get_retrieved_docs(self, question: str) -> List[Dict[str, Any]]:
        """
        Get retrieved documents for a question without generating an answer.
        
        Args:
            question: User question
            
        Returns:
            List of retrieved documents
        """
        # Search for relevant information
        search_query = self._generate_search_query(question)
        return self.search_engine.search(search_query, self.max_results)
    
    def _generate_search_query(self, question: str) -> str:
        """
        Generate an optimized search query from the user question.
        
        Args:
            question: User question
            
        Returns:
            Optimized search query
        """
        # For now, just use the question as the search query
        # In a more advanced implementation, this could use an LLM to generate a better search query
        return question
