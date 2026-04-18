import os
import json
import requests
import time
import threading
from typing import Optional, Dict, Any, List
from google import genai
from dotenv import load_dotenv

load_dotenv()

class RateLimiter:
    def __init__(self, rpm: int):
        self.interval = 60.0 / rpm if rpm > 0 else 0
        self.last_call = 0.0
        self.lock = threading.Lock()

    def wait(self):
        if self.interval <= 0:
            return
        with self.lock:
            now = time.time()
            elapsed = now - self.last_call
            if elapsed < self.interval:
                time.sleep(self.interval - elapsed)
            self.last_call = time.time()

class BasePredictor:
    def predict(self, prompt: str) -> str:
        raise NotImplementedError
    
    def predict_json(self, prompt: str) -> Dict[str, Any]:
        """
        Request a JSON response from the model and return it as a dictionary.
        """
        raw = self.predict(prompt + "\nIMPORTANT: Your entire response must be a valid JSON object or list. Do not include any markdown formatting or explanations.")
        try:
            # Clean up potential markdown blocks
            clean_raw = raw.strip()
            if clean_raw.startswith("```json"):
                clean_raw = clean_raw[7:]
            if clean_raw.endswith("```"):
                clean_raw = clean_raw[:-3]
            return json.loads(clean_raw.strip())
        except Exception as e:
            print(f"Failed to parse JSON response: {raw[:200]}...")
            raise e

    def get_model_name(self) -> str:
        raise NotImplementedError

class GeminiPredictor(BasePredictor):
    def __init__(self, model_name: str = "gemini-3.1-flash-lite-preview", rpm: int = 15):
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY not found in environment")
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.rate_limiter = RateLimiter(rpm)

    def predict(self, prompt: str) -> str:
        self.rate_limiter.wait()
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt
        )
        return response.text
    
    def predict_json(self, prompt: str) -> Dict[str, Any]:
        self.rate_limiter.wait()
        # Gemini 1.5+ supports constrained output
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config={
                "response_mime_type": "application/json"
            }
        )
        return json.loads(response.text)

    def get_model_name(self) -> str:
        return self.model_name

class OllamaPredictor(BasePredictor):
    def __init__(self, model_name: Optional[str] = None, host: Optional[str] = None):
        self.model_name = model_name or os.getenv("OLLAMA_MODEL") or "llama3"
        self.host = host or os.getenv("OLLAMA_HOST") or "http://localhost:11434"

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

    def predict_json(self, prompt: str) -> Dict[str, Any]:
        url = f"{self.host}/api/generate"
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "format": "json" # Ollama's native JSON mode
        }
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return json.loads(response.json().get("response", "{}"))
    
    def get_model_name(self) -> str:
        return self.model_name

class FinBertPredictor(BasePredictor):
    def __init__(self, model_name: str = "ProsusAI/finbert"):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Check for Apple Silicon (MPS)
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            # print("Using MPS (Metal) for FinBERT acceleration.")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
            
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def predict(self, text: str) -> str:
        """Returns the raw label for single headline."""
        import torch
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            # labels: 0: positive, 1: negative, 2: neutral
            conf, idx = torch.max(probs, dim=-1)
            labels = ["positive", "negative", "neutral"]
            return f"{labels[idx.item()]} (conf: {conf.item():.4f})"

    def predict_json(self, text: str) -> Dict[str, Any]:
        """Returns structured sentiment info with -1 to 1 score."""
        import torch
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # ProsusAI/finbert labels: 0 -> positive, 1 -> negative, 2 -> neutral
            p_pos = probs[0][0].item()
            p_neg = probs[0][1].item()
            p_neu = probs[0][2].item()
            
            # Calculate continuous score: -1 to 1
            # Simple approach: Pos probability - Neg probability
            sentiment_score = p_pos - p_neg
            
            # Map back to buy/sell for consistency with existing code
            sentiment_label = "buy" if sentiment_score > 0 else "sell"
            
            return {
                "sentiment_score": round(sentiment_score, 4),
                "sentiment": sentiment_label,
                "confidence": round(max(p_pos, p_neg, p_neu), 4),
                "reasoning": f"FinBERT Score: {sentiment_score:.2f} (Pos: {p_pos:.2f}, Neg: {p_neg:.2f}, Neu: {p_neu:.2f})"
            }

    def predict_batch_json(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Efficient batch processing for FinBERT."""
        import torch
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            results = []
            for i in range(len(texts)):
                p_pos = probs[i][0].item()
                p_neg = probs[i][1].item()
                p_neu = probs[i][2].item()
                
                score = p_pos - p_neg
                results.append({
                    "sentiment_score": round(score, 4),
                    "sentiment": "buy" if score > 0 else "sell",
                    "confidence": round(max(p_pos, p_neg, p_neu), 4),
                    "reasoning": f"FinBERT Score: {score:.2f}"
                })
            return results

    def get_model_name(self) -> str:
        return self.model_name

def get_predictor(provider: str = "gemini", **kwargs) -> BasePredictor:
    if provider.lower() == "gemini":
        return GeminiPredictor(**kwargs)
    elif provider.lower() == "ollama":
        return OllamaPredictor(**kwargs)
    elif provider.lower() == "finbert":
        return FinBertPredictor(**kwargs)
    else:
        raise ValueError(f"Unknown provider: {provider}")
