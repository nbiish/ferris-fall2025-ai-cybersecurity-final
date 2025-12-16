"""
Embedding Provider for RAG system.

Purpose: Generate embeddings for text chunks using various providers
Inputs: Text content, provider configuration
Outputs: Numpy arrays of embeddings

Supports:
- IBM Granite embeddings via Ollama (granite-embedding:30m, granite-embedding:278m)
- IBM Granite embeddings via HuggingFace (granite-embedding-english-r2)
- OpenAI embeddings (text-embedding-3-small)
- Local sentence-transformers
- Custom provider endpoints

Recommended for offline/local use:
- granite-embedding:278m via Ollama (multilingual, 768 dims)
- granite-embedding:30m via Ollama (English only, faster)
"""

from typing import List, Optional
import numpy as np
from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    """Base class for embedding providers."""
    
    @abstractmethod
    def embed(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts."""
        pass
    
    @abstractmethod
    def embed_single(self, text: str) -> np.ndarray:
        """Generate embedding for a single text."""
        pass
    
    @property
    def embedding_dim(self) -> int:
        """Return the embedding dimension."""
        return 768  # Default, override in subclasses


class GraniteOllamaEmbeddings(EmbeddingProvider):
    """
    IBM Granite embedding provider via Ollama.
    
    Models available:
    - granite-embedding:30m (63MB, English only, 512 context, 768 dims)
    - granite-embedding:278m (563MB, multilingual, 512 context, 768 dims)
    
    Requires Ollama running locally with granite-embedding model pulled:
        ollama pull granite-embedding:278m
    """
    
    def __init__(
        self,
        model: str = "granite-embedding:278m",
        base_url: str = "http://localhost:11434",
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._embedding_dim = 768
    
    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim
    
    def embed(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for multiple texts via Ollama."""
        import httpx
        
        if not texts:
            return np.array([])
        
        embeddings = []
        for text in texts:
            response = httpx.post(
                f"{self.base_url}/api/embed",
                json={
                    "model": self.model,
                    "input": text,
                },
                timeout=60.0,
            )
            response.raise_for_status()
            data = response.json()
            
            # Ollama returns embeddings in 'embeddings' array
            if "embeddings" in data and len(data["embeddings"]) > 0:
                embeddings.append(data["embeddings"][0])
            elif "embedding" in data:
                embeddings.append(data["embedding"])
            else:
                raise ValueError(f"Unexpected Ollama response format: {data.keys()}")
        
        return np.array(embeddings)
    
    def embed_single(self, text: str) -> np.ndarray:
        """Generate embedding for a single text."""
        return self.embed([text])[0]


class GraniteHuggingFaceEmbeddings(EmbeddingProvider):
    """
    IBM Granite embedding provider via HuggingFace Transformers.
    
    Models available:
    - ibm-granite/granite-embedding-english-r2 (latest, English)
    - ibm-granite/granite-embedding-278m-multilingual (multilingual)
    
    Runs fully locally without internet after initial download.
    """
    
    def __init__(self, model_name: str = "ibm-granite/granite-embedding-english-r2"):
        self.model_name = model_name
        self._model = None
        self._tokenizer = None
        self._embedding_dim = 768
    
    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim
    
    def _load_model(self):
        """Lazy load the model and tokenizer."""
        if self._model is None:
            try:
                from transformers import AutoModel, AutoTokenizer
                import torch
                
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self._model = AutoModel.from_pretrained(self.model_name)
                self._model.eval()
                
                # Move to GPU if available
                if torch.cuda.is_available():
                    self._model = self._model.cuda()
                    
            except ImportError:
                raise ImportError(
                    "transformers and torch required: pip install transformers torch"
                )
    
    def embed(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings using HuggingFace Granite model."""
        import torch
        
        if not texts:
            return np.array([])
        
        self._load_model()
        
        # Tokenize
        inputs = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        
        # Move to same device as model
        if next(self._model.parameters()).is_cuda:
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Generate embeddings
        with torch.no_grad():
            outputs = self._model(**inputs)
            # Use CLS token embedding or mean pooling
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        return embeddings
    
    def embed_single(self, text: str) -> np.ndarray:
        """Generate embedding for a single text."""
        return self.embed([text])[0]


class OpenAIEmbeddings(EmbeddingProvider):
    """OpenAI embedding provider using text-embedding-3-small."""
    
    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        self.api_key = api_key
        self.model = model
        self._client = None
    
    @property
    def client(self):
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError("openai package required: pip install openai")
        return self._client
    
    def embed(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for multiple texts."""
        if not texts:
            return np.array([])
        
        response = self.client.embeddings.create(
            model=self.model,
            input=texts
        )
        embeddings = [item.embedding for item in response.data]
        return np.array(embeddings)
    
    def embed_single(self, text: str) -> np.ndarray:
        """Generate embedding for a single text."""
        return self.embed([text])[0]


class LocalEmbeddings(EmbeddingProvider):
    """Local embedding provider using sentence-transformers."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None
    
    @property
    def model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
            except ImportError:
                raise ImportError(
                    "sentence-transformers required: pip install sentence-transformers"
                )
        return self._model
    
    def embed(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for multiple texts."""
        if not texts:
            return np.array([])
        return self.model.encode(texts, convert_to_numpy=True)
    
    def embed_single(self, text: str) -> np.ndarray:
        """Generate embedding for a single text."""
        return self.model.encode(text, convert_to_numpy=True)


class CustomEndpointEmbeddings(EmbeddingProvider):
    """Custom endpoint embedding provider for ZenMux/OpenRouter compatible APIs."""
    
    def __init__(self, base_url: str, api_key: str, model: str = "text-embedding-3-small"):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
    
    def embed(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings via custom endpoint."""
        import httpx
        
        if not texts:
            return np.array([])
        
        response = httpx.post(
            f"{self.base_url}/embeddings",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "input": texts,
            },
            timeout=60.0,
        )
        response.raise_for_status()
        
        data = response.json()
        embeddings = [item["embedding"] for item in data["data"]]
        return np.array(embeddings)
    
    def embed_single(self, text: str) -> np.ndarray:
        """Generate embedding for a single text."""
        return self.embed([text])[0]


def create_embedding_provider(
    provider_type: str = "granite-ollama",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
) -> EmbeddingProvider:
    """
    Factory function to create embedding providers.
    
    Args:
        provider_type: Provider type, one of:
            - "granite-ollama" (default): IBM Granite via Ollama (local, no internet)
            - "granite-hf": IBM Granite via HuggingFace (local after download)
            - "openai": OpenAI embeddings (requires API key)
            - "local": sentence-transformers (local)
            - "custom": Custom OpenAI-compatible endpoint
        api_key: API key for OpenAI or custom providers
        base_url: Base URL for Ollama or custom providers
        model: Model name/ID to use
        
    Returns:
        Configured EmbeddingProvider instance
        
    Examples:
        # Local Granite via Ollama (recommended for offline use)
        provider = create_embedding_provider("granite-ollama")
        
        # Granite via HuggingFace
        provider = create_embedding_provider("granite-hf")
        
        # OpenAI (requires internet)
        provider = create_embedding_provider("openai", api_key="sk-...")
    """
    if provider_type == "granite-ollama":
        return GraniteOllamaEmbeddings(
            model=model or "granite-embedding:278m",
            base_url=base_url or "http://localhost:11434",
        )
    
    elif provider_type == "granite-hf":
        return GraniteHuggingFaceEmbeddings(
            model_name=model or "ibm-granite/granite-embedding-english-r2"
        )
    
    elif provider_type == "openai":
        if not api_key:
            raise ValueError("API key required for OpenAI embeddings")
        return OpenAIEmbeddings(api_key, model or "text-embedding-3-small")
    
    elif provider_type == "local":
        return LocalEmbeddings(model or "all-MiniLM-L6-v2")
    
    elif provider_type == "custom":
        if not base_url or not api_key:
            raise ValueError("Base URL and API key required for custom embeddings")
        return CustomEndpointEmbeddings(base_url, api_key, model or "text-embedding-3-small")
    
    else:
        raise ValueError(f"Unknown provider type: {provider_type}")


# Granite model configurations for easy reference
GRANITE_MODELS = {
    "ollama": {
        "embedding": {
            "30m": "granite-embedding:30m",  # 63MB, English only
            "278m": "granite-embedding:278m",  # 563MB, multilingual
        },
        "llm": {
            "2b": "granite3.1-dense:2b",  # 1.6GB, 128K context
            "8b": "granite3.1-dense:8b",  # 5.0GB, 128K context
        },
    },
    "huggingface": {
        "embedding": {
            "english-r2": "ibm-granite/granite-embedding-english-r2",
            "278m-multilingual": "ibm-granite/granite-embedding-278m-multilingual",
        },
        "llm": {
            "4.0-micro": "ibm-granite/granite-4.0-micro",  # 3B
            "4.0-tiny": "ibm-granite/granite-4.0-tiny-preview",  # 7B
            "4.0-small": "ibm-granite/granite-4.0-h-small",  # 32B
        },
    },
}
