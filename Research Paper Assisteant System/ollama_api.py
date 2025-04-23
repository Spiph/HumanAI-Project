"""
Ollama API Module

This module contains the Ollama API interaction functionality from the original ollama_chatv35.py file.
It maintains the exact same functionality without any modifications.
"""

import requests
import json

# Define the Ollama API endpoint
OLLAMA_API_URL = "http://localhost:11434/api/chat"

def query_ollama_model(prompt, model_name="gemma3:1b", context=None, stream=True):
    """
    Send a prompt to the Ollama model and retrieve the response.
    
    Args:
        prompt: The prompt to send to the model
        model_name: The name of the model to use (default: "gemma3:1b")
        context: Optional system context message
        stream: Whether to stream the response (default: True)
        
    Yields:
        Response chunks with role and content keys
    """
    messages = [
        {"role": "user", "content": prompt}
    ]
    if context:
        messages.insert(0, {"role": "system", "content": context})  # Add context as system message
    payload = {
        "model": model_name,
        "messages": messages,
        "stream": stream  # Enable or disable streaming
    }
    try:
        response = requests.post(OLLAMA_API_URL, json=payload, stream=stream)
        response.raise_for_status()  # Raise an error for bad responses
        if stream:
            full_response = ""
            for line in response.iter_lines():
                if line:
                    chunk = line.decode("utf-8")
                    data = json.loads(chunk)
                    if "message" in data and "content" in data["message"]:
                        full_response += data["message"]["content"]
                        yield {"role": "assistant", "content": full_response}  # Yield the cumulative response
        else:
            full_response = ""
            for line in response.iter_lines():
                if line:
                    chunk = line.decode("utf-8")
                    data = json.loads(chunk)
                    if "message" in data and "content" in data["message"]:
                        full_response += data["message"]["content"]
            yield {"role": "assistant", "content": full_response}  # Yield the complete response
    except requests.exceptions.RequestException as e:
        yield {"role": "assistant", "content": f"Error communicating with the Ollama API: {str(e)}"}
