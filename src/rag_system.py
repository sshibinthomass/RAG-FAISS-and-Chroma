"""
RAG System Module.
This module integrates PDF processing, vector store, and LLM components.
"""

import os
import enum
from typing import List, Dict, Any, Optional, Union

from src.pdf_processor import PDFProcessor
from src.vector_store import VectorStore
from src.chroma_store import ChromaStore
from src.ollama_client import OllamaClient

class VectorStoreType(enum.Enum):
    """Enum for vector store types."""
    FAISS = "faiss"
    CHROMA = "chroma"

class RAGSystem:
    """Class for the complete RAG system."""

    def __init__(
        self,
        pdf_dir: str = "data/pdfs",
        index_dir: str = "data/index",
        chroma_dir: str = "data/chroma_db",
        llm_model: str = "llama2",
        embedding_model: str = "nomic-embed-text",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        top_k: int = 5,
        vector_store_type: VectorStoreType = VectorStoreType.FAISS
    ):
        """
        Initialize the RAG system.

        Args:
            pdf_dir: Directory containing PDF files
            index_dir: Directory to store the FAISS index
            chroma_dir: Directory to store the ChromaDB
            llm_model: Ollama LLM model name
            embedding_model: Ollama embedding model name
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            top_k: Number of documents to retrieve for each query
            vector_store_type: Type of vector store to use (FAISS or ChromaDB)
        """
        self.pdf_dir = pdf_dir
        self.index_dir = index_dir
        self.chroma_dir = chroma_dir
        self.top_k = top_k
        self.vector_store_type = vector_store_type

        # Initialize components
        self.pdf_processor = PDFProcessor(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.ollama_client = OllamaClient(model_name=llm_model)

        # Initialize vector store based on type
        if vector_store_type == VectorStoreType.FAISS:
            self.vector_store = VectorStore(embedding_model_name=embedding_model)
        else:
            self.vector_store = ChromaStore(
                embedding_model_name=embedding_model,
                persist_directory=chroma_dir
            )

        # Create directories if they don't exist
        os.makedirs(pdf_dir, exist_ok=True)
        os.makedirs(index_dir, exist_ok=True)
        os.makedirs(chroma_dir, exist_ok=True)

    def index_documents(self, force_reindex: bool = False) -> None:
        """
        Index all PDF documents in the PDF directory.

        Args:
            force_reindex: Whether to force reindexing even if index exists
        """
        if self.vector_store_type == VectorStoreType.FAISS:
            self._index_documents_faiss(force_reindex)
        else:
            self._index_documents_chroma(force_reindex)

    def _index_documents_faiss(self, force_reindex: bool = False) -> None:
        """
        Index all PDF documents using FAISS.

        Args:
            force_reindex: Whether to force reindexing even if index exists
        """
        index_path = os.path.join(self.index_dir, "faiss_index.index")

        # Check if index already exists
        if os.path.exists(index_path) and not force_reindex:
            print("Loading existing FAISS index...")
            self.vector_store.load(self.index_dir)
            return

        print(f"Indexing documents from {self.pdf_dir} using FAISS...")
        documents = self.pdf_processor.process_directory(self.pdf_dir)

        if not documents:
            print("No documents found to index.")
            return

        print(f"Creating FAISS index with {len(documents)} document chunks...")
        self.vector_store.create_index(documents)

        print("Saving FAISS index...")
        self.vector_store.save(self.index_dir)

        print("FAISS indexing complete.")

    def _index_documents_chroma(self, force_reindex: bool = False) -> None:
        """
        Index all PDF documents using ChromaDB.

        Args:
            force_reindex: Whether to force reindexing even if index exists
        """
        # Clear collection if force reindex
        if force_reindex:
            print("Clearing ChromaDB collection...")
            self.vector_store.clear()

        # Check if collection already has documents
        if not force_reindex and self.vector_store.count() > 0:
            print(f"ChromaDB collection already has {self.vector_store.count()} documents.")
            return

        print(f"Indexing documents from {self.pdf_dir} using ChromaDB...")
        documents = self.pdf_processor.process_directory(self.pdf_dir)

        if not documents:
            print("No documents found to index.")
            return

        print(f"Adding {len(documents)} document chunks to ChromaDB...")
        self.vector_store.add_documents(documents)

        print("ChromaDB indexing complete.")

    def query(self, question: str) -> Dict[str, Any]:
        """
        Process a query through the RAG system.

        Args:
            question: User question

        Returns:
            Dictionary with answer and retrieved documents
        """
        # Ensure documents are indexed
        self.index_documents()

        # Search for relevant documents
        retrieved_docs = self.vector_store.search(question, k=self.top_k)

        # Generate answer using RAG
        answer = self.ollama_client.answer_with_rag(question, retrieved_docs)

        return {
            "question": question,
            "answer": answer,
            "retrieved_documents": retrieved_docs,
            "vector_store_type": self.vector_store_type.value
        }

    def stream_query(self, question: str):
        """
        Process a query through the RAG system with streaming response.

        Args:
            question: User question

        Yields:
            Chunks of the generated answer
        """
        # Ensure documents are indexed
        self.index_documents()

        # Search for relevant documents
        retrieved_docs = self.vector_store.search(question, k=self.top_k)

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
        # Ensure documents are indexed
        self.index_documents()

        # Search for relevant documents
        return self.vector_store.search(question, k=self.top_k)

    def switch_vector_store(self, vector_store_type: VectorStoreType, embedding_model: str = None) -> None:
        """
        Switch the vector store type.

        Args:
            vector_store_type: Type of vector store to use
            embedding_model: Optional embedding model name to use
        """
        if vector_store_type == self.vector_store_type:
            print(f"Already using {vector_store_type.value} vector store.")
            return

        # Update embedding model if provided
        if embedding_model:
            embedding_model_name = embedding_model
        else:
            # Use current embedding model
            embedding_model_name = getattr(self.vector_store, "embedding_model_name", "nomic-embed-text")

        # Initialize new vector store
        if vector_store_type == VectorStoreType.FAISS:
            self.vector_store = VectorStore(embedding_model_name=embedding_model_name)
        else:
            self.vector_store = ChromaStore(
                embedding_model_name=embedding_model_name,
                persist_directory=self.chroma_dir
            )

        # Update vector store type
        self.vector_store_type = vector_store_type

        # Load or create index
        self.index_documents()

    def add_pdf(self, pdf_path: str, reindex: bool = True) -> None:
        """
        Add a new PDF to the system.

        Args:
            pdf_path: Path to the PDF file
            reindex: Whether to reindex after adding
        """
        # Copy PDF to the PDF directory if it's not already there
        filename = os.path.basename(pdf_path)
        target_path = os.path.join(self.pdf_dir, filename)

        if pdf_path != target_path:
            import shutil
            shutil.copy2(pdf_path, target_path)
            print(f"Copied {pdf_path} to {target_path}")

        if reindex:
            self.index_documents(force_reindex=True)
