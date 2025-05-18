"""
Vector Store Module for RAG System.
This module handles vector embeddings and FAISS database operations.
"""

import os
import pickle
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import faiss
from langchain_community.embeddings import OllamaEmbeddings

class VectorStore:
    """Class for managing vector embeddings and FAISS database."""
    
    def __init__(self, embedding_model_name: str = "nomic-embed-text"):
        """
        Initialize the vector store.
        
        Args:
            embedding_model_name: Name of the Ollama embedding model to use
        """
        self.embedding_model_name = embedding_model_name
        self.embeddings = OllamaEmbeddings(model=embedding_model_name)
        self.index = None
        self.documents = []
        self.dimension = None
    
    def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Get embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            NumPy array of embeddings
        """
        embeddings = self.embeddings.embed_documents(texts)
        return np.array(embeddings, dtype=np.float32)
    
    def create_index(self, documents: List[Dict[str, Any]]) -> None:
        """
        Create a FAISS index from documents.
        
        Args:
            documents: List of document dictionaries with 'content' and 'metadata'
        """
        self.documents = documents
        texts = [doc["content"] for doc in documents]
        
        # Get embeddings
        embeddings = self._get_embeddings(texts)
        self.dimension = embeddings.shape[1]
        
        # Create FAISS index
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(embeddings)
        
        print(f"Created FAISS index with {len(documents)} documents and dimension {self.dimension}")
    
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search the vector store for documents similar to the query.
        
        Args:
            query: Query string
            k: Number of results to return
            
        Returns:
            List of document dictionaries with similarity scores
        """
        if self.index is None:
            raise ValueError("Index has not been created yet")
        
        # Get query embedding
        query_embedding = self._get_embeddings([query])
        
        # Search FAISS index
        distances, indices = self.index.search(query_embedding, k)
        
        # Get results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.documents):  # Ensure index is valid
                results.append({
                    "content": self.documents[idx]["content"],
                    "metadata": self.documents[idx]["metadata"],
                    "score": float(distances[0][i])
                })
        
        return results
    
    def save(self, directory_path: str, name: str = "faiss_index") -> None:
        """
        Save the FAISS index and documents to disk.
        
        Args:
            directory_path: Directory to save the index
            name: Base name for the saved files
        """
        if self.index is None:
            raise ValueError("Index has not been created yet")
        
        os.makedirs(directory_path, exist_ok=True)
        
        # Save FAISS index
        index_path = os.path.join(directory_path, f"{name}.index")
        faiss.write_index(self.index, index_path)
        
        # Save documents and metadata
        docs_path = os.path.join(directory_path, f"{name}.pkl")
        with open(docs_path, 'wb') as f:
            pickle.dump({
                "documents": self.documents,
                "dimension": self.dimension,
                "embedding_model": self.embedding_model_name
            }, f)
        
        print(f"Saved index to {index_path} and documents to {docs_path}")
    
    def load(self, directory_path: str, name: str = "faiss_index") -> None:
        """
        Load a FAISS index and documents from disk.
        
        Args:
            directory_path: Directory containing the saved index
            name: Base name of the saved files
        """
        index_path = os.path.join(directory_path, f"{name}.index")
        docs_path = os.path.join(directory_path, f"{name}.pkl")
        
        if not os.path.exists(index_path) or not os.path.exists(docs_path):
            raise FileNotFoundError(f"Index or documents file not found in {directory_path}")
        
        # Load FAISS index
        self.index = faiss.read_index(index_path)
        
        # Load documents and metadata
        with open(docs_path, 'rb') as f:
            data = pickle.load(f)
            self.documents = data["documents"]
            self.dimension = data["dimension"]
            self.embedding_model_name = data.get("embedding_model", self.embedding_model_name)
            
            # Reinitialize embeddings if model changed
            if self.embedding_model_name != self.embeddings.model:
                self.embeddings = OllamaEmbeddings(model=self.embedding_model_name)
        
        print(f"Loaded index with {len(self.documents)} documents and dimension {self.dimension}")
