"""
RAG (Retrieval-Augmented Generation) module with Pyversity diversification.

Purpose: Provide diversified context retrieval for agent conversations
Inputs: Agent outputs, conversation history, document folders
Outputs: Diversified, relevant context chunks for LLM prompts

Uses SSD (Sliding Spectrum Decomposition) strategy from Pyversity for
sequence-aware diversification in conversational contexts.

Recommended Setup for Local/Offline Use:
    1. Install Ollama: https://ollama.com/download
    2. Pull Granite embedding model: ollama pull granite-embedding:278m
    3. Run RAG server: python -m rag.api --embedding-provider granite-ollama

Or use the setup script:
    python -m rag.granite_setup --embedding
"""

from .service import RAGService, RetrievalResult
from .embeddings import (
    EmbeddingProvider,
    GraniteOllamaEmbeddings,
    GraniteHuggingFaceEmbeddings,
    create_embedding_provider,
    GRANITE_MODELS,
)
from .indexer import ContentIndexer

__all__ = [
    "RAGService",
    "RetrievalResult",
    "EmbeddingProvider",
    "GraniteOllamaEmbeddings",
    "GraniteHuggingFaceEmbeddings",
    "create_embedding_provider",
    "ContentIndexer",
    "GRANITE_MODELS",
]
