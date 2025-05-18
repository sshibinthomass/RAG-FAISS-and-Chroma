"""
Main application for PDF RAG System.
This script provides a command-line interface to the RAG system.
"""

import os
import argparse
import sys
from typing import List, Optional

from src.rag_system import RAGSystem

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="PDF RAG System using Ollama and FAISS")
    
    # Main command subparsers
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Query the RAG system")
    query_parser.add_argument("question", type=str, help="Question to ask")
    query_parser.add_argument("--model", type=str, default="llama2", help="Ollama model to use")
    query_parser.add_argument("--top-k", type=int, default=5, help="Number of documents to retrieve")
    
    # Add PDF command
    add_parser = subparsers.add_parser("add", help="Add a PDF to the system")
    add_parser.add_argument("pdf_path", type=str, help="Path to the PDF file")
    
    # Index command
    index_parser = subparsers.add_parser("index", help="Index all PDFs in the directory")
    index_parser.add_argument("--force", action="store_true", help="Force reindexing")
    
    # Interactive mode
    interactive_parser = subparsers.add_parser("interactive", help="Start interactive mode")
    interactive_parser.add_argument("--model", type=str, default="llama2", help="Ollama model to use")
    
    return parser.parse_args()

def interactive_mode(rag_system: RAGSystem):
    """Run the RAG system in interactive mode."""
    print("\n=== PDF RAG System Interactive Mode ===")
    print("Type 'exit' or 'quit' to exit.")
    print("Type 'add <pdf_path>' to add a PDF.")
    print("Type 'index' to reindex all PDFs.")
    print("Type anything else to query the system.\n")
    
    while True:
        try:
            user_input = input("\nEnter your query: ").strip()
            
            if user_input.lower() in ["exit", "quit"]:
                print("Exiting interactive mode.")
                break
            
            elif user_input.lower().startswith("add "):
                pdf_path = user_input[4:].strip()
                if os.path.exists(pdf_path):
                    rag_system.add_pdf(pdf_path)
                    print(f"Added {pdf_path} to the system and reindexed.")
                else:
                    print(f"Error: File {pdf_path} not found.")
            
            elif user_input.lower() == "index":
                rag_system.index_documents(force_reindex=True)
                print("Reindexed all PDFs.")
            
            else:
                print("\nProcessing query...")
                result = rag_system.query(user_input)
                
                print("\n=== Answer ===")
                print(result["answer"])
                
                print("\n=== Retrieved Documents ===")
                for i, doc in enumerate(result["retrieved_documents"]):
                    print(f"\nDocument {i+1} (Source: {doc['metadata']['source']})")
                    print(f"Score: {doc['score']:.4f}")
                    print("-" * 40)
                    print(doc["content"][:200] + "..." if len(doc["content"]) > 200 else doc["content"])
                    print("-" * 40)
        
        except KeyboardInterrupt:
            print("\nExiting interactive mode.")
            break
        except Exception as e:
            print(f"Error: {str(e)}")

def main():
    """Main function."""
    args = parse_args()
    
    # Initialize RAG system
    rag_system = RAGSystem(
        llm_model=getattr(args, "model", "llama2"),
        top_k=getattr(args, "top_k", 5)
    )
    
    # Process command
    if args.command == "query":
        # Ensure documents are indexed
        rag_system.index_documents()
        
        # Process query
        result = rag_system.query(args.question)
        
        # Print answer
        print("\n=== Answer ===")
        print(result["answer"])
        
        # Print retrieved documents
        print("\n=== Retrieved Documents ===")
        for i, doc in enumerate(result["retrieved_documents"]):
            print(f"\nDocument {i+1} (Source: {doc['metadata']['source']})")
            print(f"Score: {doc['score']:.4f}")
            print("-" * 40)
            print(doc["content"][:200] + "..." if len(doc["content"]) > 200 else doc["content"])
            print("-" * 40)
    
    elif args.command == "add":
        if not os.path.exists(args.pdf_path):
            print(f"Error: File {args.pdf_path} not found.")
            return
        
        rag_system.add_pdf(args.pdf_path)
        print(f"Added {args.pdf_path} to the system and reindexed.")
    
    elif args.command == "index":
        rag_system.index_documents(force_reindex=args.force)
        print("Indexing complete.")
    
    elif args.command == "interactive":
        interactive_mode(rag_system)
    
    else:
        print("Please specify a command. Use --help for more information.")

if __name__ == "__main__":
    main()
