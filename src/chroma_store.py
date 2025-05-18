"""
ChromaDB Vector Store Module for RAG System.
This module handles vector embeddings and ChromaDB operations.
"""

import os
import chromadb
from typing import List, Dict, Any, Optional
from langchain_community.embeddings import OllamaEmbeddings

class ChromaStore:
    """Class for managing vector embeddings and ChromaDB."""

    def __init__(self,
                 embedding_model_name: str = "nomic-embed-text",
                 collection_name: str = "pdf_documents",
                 persist_directory: str = "data/chroma_db"):
        """
        Initialize the ChromaDB vector store.

        Args:
            embedding_model_name: Name of the Ollama embedding model to use
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist the ChromaDB
        """
        self.embedding_model_name = embedding_model_name
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embeddings = OllamaEmbeddings(model=embedding_model_name)

        # Create directory if it doesn't exist
        os.makedirs(persist_directory, exist_ok=True)

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=persist_directory)

        # Try to get collection or create it if it doesn't exist
        try:
            self.collection = self.client.get_collection(name=collection_name)
            print(f"Loaded existing collection '{collection_name}' with {self.collection.count()} documents")
        except Exception as e:
            print(f"Collection not found, creating new one: {str(e)}")
            self.collection = self.client.create_collection(name=collection_name)
            print(f"Created new collection '{collection_name}'")

    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings for a list of texts.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embeddings
        """
        return self.embeddings.embed_documents(texts)

    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Add documents to the ChromaDB collection.

        Args:
            documents: List of document dictionaries with 'content' and 'metadata'
        """
        if not documents:
            print("No documents to add")
            return

        # Extract document content, metadata, and create IDs
        contents = [doc["content"] for doc in documents]
        metadatas = [doc["metadata"] for doc in documents]
        ids = [f"doc_{i}" for i in range(len(documents))]

        # Get embeddings
        embeddings = self._get_embeddings(contents)

        # Add documents to collection
        self.collection.add(
            documents=contents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )

        print(f"Added {len(documents)} documents to ChromaDB collection")

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search the vector store for documents similar to the query.

        Args:
            query: Query string
            k: Number of results to return

        Returns:
            List of document dictionaries with similarity scores
        """
        # Get query embedding
        query_embedding = self._get_embeddings([query])[0]

        # Search ChromaDB collection
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"]
        )

        # Format results
        formatted_results = []
        for i in range(len(results["documents"][0])):
            formatted_results.append({
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "score": results["distances"][0][i]
            })

        return formatted_results

    def clear(self) -> None:
        """Clear all documents from the collection."""
        self.collection.delete(where={})
        print(f"Cleared all documents from collection '{self.collection_name}'")

    def count(self) -> int:
        """
        Get the number of documents in the collection.

        Returns:
            Number of documents
        """
        return self.collection.count()
