import os
import json
import requests
from google import genai
from typing import Optional, Dict, Any
from dotenv import load_dotenv

load_dotenv()

class BasePredictor:
    def predict(self, prompt: str) -> str:
        raise NotImplementedError

class GeminiPredictor(BasePredictor):
    def __init__(self, model_name: str = "gemini-3.1-flash-lite-preview"):
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY not found in environment")
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    def predict(self, prompt: str) -> str:
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt
        )
        return response.text
    
    def get_model_name(self) -> str:
        return self.model_name

class OllamaPredictor(BasePredictor):
    def __init__(self, model_name: str = "llama3", host: str = "http://localhost:11434"):
        self.model_name = model_name
        self.host = host

    def predict(self, prompt: str) -> str:
        url = f"{self.host}/api/generate"
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False
        }
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json().get("response", "")
    
    def get_model_name(self) -> str:
        return self.model_name

def get_predictor(provider: str = "gemini", **kwargs) -> BasePredictor:
    if provider.lower() == "gemini":
        return GeminiPredictor(**kwargs)
    elif provider.lower() == "ollama":
        return OllamaPredictor(**kwargs)
    else:
        raise ValueError(f"Unknown provider: {provider}")
