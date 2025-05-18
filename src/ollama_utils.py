"""
Ollama Utilities Module.
This module provides utility functions for interacting with Ollama.
"""

import requests
from typing import List, Dict, Any, Optional

def get_available_models(api_base: str = "http://localhost:11434") -> List[str]:
    """
    Get a list of available models from Ollama.
    
    Args:
        api_base: Base URL for the Ollama API
        
    Returns:
        List of available model names
    """
    try:
        # Call Ollama API to get list of models
        response = requests.get(f"{api_base}/api/tags")
        
        if response.status_code == 200:
            data = response.json()
            # Extract model names from the response
            models = [model["name"] for model in data.get("models", [])]
            return models
        else:
            print(f"Error getting models from Ollama: {response.status_code} - {response.text}")
            # Return a default list if we can't get models
            return ["llama2"]
    except Exception as e:
        print(f"Exception getting models from Ollama: {str(e)}")
        # Return a default list if we can't get models
        return ["llama2"]
