"""
PDF Processing Module for RAG System.
This module handles loading, parsing, and chunking PDF documents.
"""

import os
from typing import List, Dict, Any
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

class PDFProcessor:
    """Class for processing PDF documents."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the PDF processor.
        
        Args:
            chunk_size: Size of text chunks for vectorization
            chunk_overlap: Overlap between chunks to maintain context
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
        )
    
    def load_pdf(self, pdf_path: str) -> str:
        """
        Load and extract text from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text from the PDF
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        try:
            reader = PdfReader(pdf_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            raise Exception(f"Error extracting text from PDF: {str(e)}")
    
    def load_pdfs_from_directory(self, directory_path: str) -> Dict[str, str]:
        """
        Load all PDFs from a directory.
        
        Args:
            directory_path: Path to directory containing PDFs
            
        Returns:
            Dictionary mapping filenames to extracted text
        """
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        pdf_texts = {}
        for filename in os.listdir(directory_path):
            if filename.lower().endswith('.pdf'):
                file_path = os.path.join(directory_path, filename)
                try:
                    pdf_texts[filename] = self.load_pdf(file_path)
                except Exception as e:
                    print(f"Error processing {filename}: {str(e)}")
        
        return pdf_texts
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks for vectorization.
        
        Args:
            text: Text to be chunked
            
        Returns:
            List of text chunks
        """
        return self.text_splitter.split_text(text)
    
    def process_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Process a single PDF file: load, extract text, and chunk.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of document chunks with metadata
        """
        filename = os.path.basename(pdf_path)
        text = self.load_pdf(pdf_path)
        chunks = self.chunk_text(text)
        
        documents = []
        for i, chunk in enumerate(chunks):
            documents.append({
                "content": chunk,
                "metadata": {
                    "source": filename,
                    "chunk": i,
                    "filepath": pdf_path
                }
            })
        
        return documents
    
    def process_directory(self, directory_path: str) -> List[Dict[str, Any]]:
        """
        Process all PDFs in a directory.
        
        Args:
            directory_path: Path to directory containing PDFs
            
        Returns:
            List of document chunks with metadata from all PDFs
        """
        all_documents = []
        
        for filename in os.listdir(directory_path):
            if filename.lower().endswith('.pdf'):
                file_path = os.path.join(directory_path, filename)
                try:
                    documents = self.process_pdf(file_path)
                    all_documents.extend(documents)
                except Exception as e:
                    print(f"Error processing {filename}: {str(e)}")
        
        return all_documents
