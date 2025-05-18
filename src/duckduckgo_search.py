"""
DuckDuckGo Search Module for RAG System.
This module handles web search using the DuckDuckGo API.
"""

from typing import List, Dict, Any, Optional
from duckduckgo_search import DDGS

class DuckDuckGoSearch:
    """Class for searching the web using DuckDuckGo."""
    
    def __init__(self, max_results: int = 10, region: str = "wt-wt", safesearch: str = "moderate"):
        """
        Initialize the DuckDuckGo search.
        
        Args:
            max_results: Maximum number of search results to return
            region: Region for search results (default: worldwide)
            safesearch: SafeSearch setting ("off", "moderate", or "strict")
        """
        self.max_results = max_results
        self.region = region
        self.safesearch = safesearch
        self.ddgs = DDGS()
    
    def search(self, query: str, max_results: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Search the web using DuckDuckGo.
        
        Args:
            query: Search query
            max_results: Optional override for max_results
            
        Returns:
            List of search results with title, body, href, and source
        """
        if max_results is None:
            max_results = self.max_results
            
        try:
            # Perform the search
            results = list(self.ddgs.text(
                query,
                region=self.region,
                safesearch=self.safesearch,
                max_results=max_results
            ))
            
            # Format the results
            formatted_results = []
            for i, result in enumerate(results):
                formatted_results.append({
                    "title": result.get("title", ""),
                    "content": result.get("body", ""),
                    "url": result.get("href", ""),
                    "source": "web",
                    "index": i + 1,
                    "metadata": {
                        "source": "DuckDuckGo",
                        "url": result.get("href", ""),
                        "title": result.get("title", "")
                    }
                })
            
            return formatted_results
        except Exception as e:
            print(f"Error searching DuckDuckGo: {str(e)}")
            return []
    
    def get_snippets(self, query: str, max_results: Optional[int] = None) -> str:
        """
        Get search results as a formatted text for context.
        
        Args:
            query: Search query
            max_results: Optional override for max_results
            
        Returns:
            Formatted string with search results
        """
        results = self.search(query, max_results)
        
        if not results:
            return "No search results found."
        
        snippets = []
        for result in results:
            snippet = f"Title: {result['title']}\nURL: {result['url']}\nContent: {result['content']}\n"
            snippets.append(snippet)
        
        return "\n".join(snippets)
