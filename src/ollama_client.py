"""
Ollama Client Module for RAG System.
This module handles interactions with Ollama LLM models.
"""

from typing import List, Dict, Any, Optional, Generator, Callable
import requests
import json
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

class OllamaClient:
    """Class for interacting with Ollama LLM models."""

    def __init__(self, model_name: str = "llama2", api_base: str = "http://localhost:11434"):
        """
        Initialize the Ollama client.

        Args:
            model_name: Name of the Ollama model to use
            api_base: Base URL for the Ollama API
        """
        self.model_name = model_name
        self.api_base = api_base
        self.llm = Ollama(model=model_name, base_url=api_base)

    def generate_response(self, prompt: str) -> str:
        """
        Generate a response from the LLM.

        Args:
            prompt: Prompt text

        Returns:
            Generated response
        """
        return self.llm.invoke(prompt)

    def create_rag_chain(self) -> LLMChain:
        """
        Create a LangChain chain for RAG.

        Returns:
            LLMChain for RAG
        """
        template = """
        You are a helpful assistant that provides accurate information based on the context provided.

        Context information is below:
        ---------------------
        {context}
        ---------------------

        Given the context information and not prior knowledge, answer the question: {question}

        If the answer cannot be determined from the context, say "I don't have enough information to answer this question."
        """

        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=template
        )

        return LLMChain(llm=self.llm, prompt=prompt)

    def answer_with_rag(self, question: str, context_docs: List[Dict[str, Any]]) -> str:
        """
        Answer a question using RAG.

        Args:
            question: User question
            context_docs: List of context documents from vector search

        Returns:
            Generated answer
        """
        # Format context from documents
        context_text = self._format_context(context_docs)

        # Create and run chain
        chain = self.create_rag_chain()
        response = chain.invoke({
            "context": context_text,
            "question": question
        })

        return response["text"]

    def _format_context(self, context_docs: List[Dict[str, Any]]) -> str:
        """Format context documents into a string."""
        return "\n\n".join([f"Document {i+1} (Source: {doc['metadata']['source']}):\n{doc['content']}"
                           for i, doc in enumerate(context_docs)])

    def stream_answer_with_rag(self, question: str, context_docs: List[Dict[str, Any]]) -> Generator[str, None, None]:
        """
        Stream an answer using RAG.

        Args:
            question: User question
            context_docs: List of context documents from vector search

        Yields:
            Chunks of the generated answer
        """
        try:
            # Format context from documents
            context_text = self._format_context(context_docs)

            # Create prompt
            prompt = f"""
            You are a helpful assistant that provides accurate information based on the context provided.

            Context information is below:
            ---------------------
            {context_text}
            ---------------------

            Given the context information and not prior knowledge, answer the question: {question}

            If the answer cannot be determined from the context, say "I don't have enough information to answer this question."
            """

            # Stream response directly from Ollama API
            url = f"{self.api_base}/api/generate"
            data = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": True
            }

            print(f"Sending request to Ollama API at {url}")
            print(f"Using model: {self.model_name}")

            try:
                response = requests.post(url, json=data, stream=True, timeout=10)

                if response.status_code == 200:
                    for line in response.iter_lines():
                        if line:
                            try:
                                chunk = json.loads(line)
                                if "response" in chunk:
                                    yield chunk["response"]
                            except json.JSONDecodeError as e:
                                print(f"JSON decode error: {e}, line: {line}")
                                continue
                else:
                    error_msg = f"Ollama API error: {response.status_code} - {response.text}"
                    print(error_msg)
                    yield f"Error: {error_msg}"
            except requests.exceptions.RequestException as e:
                error_msg = f"Request to Ollama API failed: {str(e)}"
                print(error_msg)
                yield f"Error: {error_msg}"
        except Exception as e:
            error_msg = f"Unexpected error in stream_answer_with_rag: {str(e)}"
            print(error_msg)
            yield f"Error: {error_msg}"

    def stream_with_callback(self, question: str, context_docs: List[Dict[str, Any]],
                            callback: Callable[[str], None]) -> None:
        """
        Stream an answer with a callback function.

        Args:
            question: User question
            context_docs: List of context documents from vector search
            callback: Function to call with each chunk of text
        """
        for chunk in self.stream_answer_with_rag(question, context_docs):
            callback(chunk)
