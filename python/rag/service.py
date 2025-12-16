"""
RAG Service with Pyversity diversification.

Purpose: Main service for diversified context retrieval
Inputs: Query text, recent context, retrieval parameters
Outputs: Diversified relevant chunks using SSD strategy

Uses Pyversity's SSD (Sliding Spectrum Decomposition) for sequence-aware
diversification - ideal for conversational AI to avoid redundant context.
"""

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np

from .embeddings import EmbeddingProvider, create_embedding_provider
from .indexer import ContentIndexer, Chunk


@dataclass
class RetrievalResult:
    """Result from RAG retrieval."""
    chunks: List[Chunk]
    scores: List[float]
    diversified_indices: List[int]
    strategy_used: str
    query: str
    
    def to_context_string(self, max_chunks: int = 5) -> str:
        """Convert to a context string for LLM prompts."""
        context_parts = []
        for i, chunk in enumerate(self.chunks[:max_chunks]):
            source = chunk.source_path or chunk.source_type
            context_parts.append(f"[Source: {source}]\n{chunk.content}")
        return "\n\n---\n\n".join(context_parts)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "chunks": [c.to_dict() for c in self.chunks],
            "scores": self.scores,
            "diversified_indices": self.diversified_indices,
            "strategy_used": self.strategy_used,
            "query": self.query,
        }


class RAGService:
    """
    RAG service with Pyversity diversification.
    
    Provides diversified context retrieval using various strategies:
    - SSD: Sequence-aware diversification (best for conversations)
    - MMR: Maximal Marginal Relevance (general purpose)
    - DPP: Determinantal Point Process (high diversity)
    - COVER: Coverage-based (ensures topic coverage)
    """
    
    def __init__(
        self,
        embedding_provider: Optional[EmbeddingProvider] = None,
        provider_type: str = "local",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        embedding_model: Optional[str] = None,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
    ):
        """
        Initialize RAG service.
        
        Args:
            embedding_provider: Pre-configured embedding provider
            provider_type: Type of embedding provider if not provided
            api_key: API key for embedding provider
            base_url: Base URL for custom embedding provider
            embedding_model: Model to use for embeddings
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        if embedding_provider is None:
            embedding_provider = create_embedding_provider(
                provider_type=provider_type,
                api_key=api_key,
                base_url=base_url,
                model=embedding_model,
            )
        
        self.embedding_provider = embedding_provider
        self.indexer = ContentIndexer(
            embedding_provider=embedding_provider,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        
        # Track recent embeddings for SSD strategy
        self.recent_embeddings: Optional[np.ndarray] = None
        self.recent_window_size: int = 10
        
        # Pyversity availability
        self._pyversity_available = self._check_pyversity()
    
    def _check_pyversity(self) -> bool:
        """Check if pyversity is available."""
        try:
            import pyversity
            return True
        except ImportError:
            print("[RAG] Warning: pyversity not installed. Using fallback diversification.")
            return False
    
    def index_agent_conversation(
        self,
        agent_id: str,
        messages: List[Dict[str, Any]],
    ) -> int:
        """
        Index an agent's conversation history.
        
        Returns number of chunks created.
        """
        chunk_ids = self.indexer.index_conversation(messages, agent_id)
        return len(chunk_ids)
    
    def index_folder(
        self,
        folder_path: str,
        agent_id: Optional[str] = None,
        extensions: Optional[List[str]] = None,
    ) -> int:
        """
        Index a folder of documents.
        
        Returns number of chunks created.
        """
        chunk_ids = self.indexer.index_folder(folder_path, extensions, agent_id)
        return len(chunk_ids)
    
    def index_agent_output(
        self,
        agent_id: str,
        output: str,
        output_type: str = "response",
    ) -> int:
        """
        Index a single agent output.
        
        Returns number of chunks created.
        """
        chunk_ids = self.indexer.index_agent_output(agent_id, output, output_type)
        return len(chunk_ids)
    
    def retrieve(
        self,
        query: str,
        k: int = 5,
        diversity: float = 0.5,
        strategy: str = "ssd",
        agent_filter: Optional[str] = None,
    ) -> RetrievalResult:
        """
        Retrieve diversified context for a query.
        
        Args:
            query: Query text
            k: Number of chunks to retrieve
            diversity: Diversity parameter (0.0-1.0, higher = more diverse)
            strategy: Diversification strategy ("ssd", "mmr", "dpp", "cover")
            agent_filter: Optional agent ID to filter results
            
        Returns:
            RetrievalResult with diversified chunks
        """
        # Get query embedding
        query_embedding = self.embedding_provider.embed_single(query)
        
        # Get all embeddings from indexer
        embeddings, chunk_ids = self.indexer.get_all_embeddings()
        
        if len(chunk_ids) == 0:
            return RetrievalResult(
                chunks=[],
                scores=[],
                diversified_indices=[],
                strategy_used=strategy,
                query=query,
            )
        
        # Apply agent filter if specified
        if agent_filter:
            filtered_indices = []
            for i, cid in enumerate(chunk_ids):
                chunk = self.indexer.chunks.get(cid)
                if chunk and chunk.agent_id == agent_filter:
                    filtered_indices.append(i)
            
            if not filtered_indices:
                return RetrievalResult(
                    chunks=[],
                    scores=[],
                    diversified_indices=[],
                    strategy_used=strategy,
                    query=query,
                )
            
            embeddings = embeddings[filtered_indices]
            chunk_ids = [chunk_ids[i] for i in filtered_indices]
        
        # Calculate similarity scores
        scores = self._cosine_similarity(query_embedding, embeddings)
        
        # Apply diversification
        if self._pyversity_available:
            diversified_indices = self._diversify_pyversity(
                embeddings=embeddings,
                scores=scores,
                k=min(k, len(chunk_ids)),
                diversity=diversity,
                strategy=strategy,
            )
        else:
            diversified_indices = self._diversify_fallback(
                scores=scores,
                k=min(k, len(chunk_ids)),
            )
        
        # Get diversified chunks
        diversified_chunk_ids = [chunk_ids[i] for i in diversified_indices]
        chunks = self.indexer.get_chunks_by_ids(diversified_chunk_ids)
        diversified_scores = [float(scores[i]) for i in diversified_indices]
        
        # Update recent embeddings for SSD
        if strategy == "ssd" and chunks:
            selected_embeddings = embeddings[diversified_indices]
            self._update_recent_embeddings(selected_embeddings)
        
        return RetrievalResult(
            chunks=chunks,
            scores=diversified_scores,
            diversified_indices=diversified_indices,
            strategy_used=strategy,
            query=query,
        )
    
    def _cosine_similarity(self, query: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
        """Calculate cosine similarity between query and embeddings."""
        query_norm = query / (np.linalg.norm(query) + 1e-8)
        embeddings_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
        return np.dot(embeddings_norm, query_norm)
    
    def _diversify_pyversity(
        self,
        embeddings: np.ndarray,
        scores: np.ndarray,
        k: int,
        diversity: float,
        strategy: str,
    ) -> List[int]:
        """Apply pyversity diversification."""
        from pyversity import diversify, Strategy
        
        strategy_map = {
            "ssd": Strategy.SSD,
            "mmr": Strategy.MMR,
            "dpp": Strategy.DPP,
            "cover": Strategy.COVER,
            "msd": Strategy.MSD,
        }
        
        pyversity_strategy = strategy_map.get(strategy.lower(), Strategy.SSD)
        
        # Build kwargs for diversify
        kwargs = {
            "embeddings": embeddings,
            "scores": scores,
            "k": k,
            "strategy": pyversity_strategy,
            "diversity": diversity,
        }
        
        # Add recent embeddings for SSD strategy
        if strategy.lower() == "ssd" and self.recent_embeddings is not None:
            kwargs["recent_embeddings"] = self.recent_embeddings
        
        result = diversify(**kwargs)
        return list(result.indices)
    
    def _diversify_fallback(self, scores: np.ndarray, k: int) -> List[int]:
        """Fallback diversification using top-k scores."""
        return list(np.argsort(scores)[-k:][::-1])
    
    def _update_recent_embeddings(self, new_embeddings: np.ndarray) -> None:
        """Update the rolling window of recent embeddings for SSD."""
        if self.recent_embeddings is None:
            self.recent_embeddings = new_embeddings
        else:
            combined = np.vstack([self.recent_embeddings, new_embeddings])
            self.recent_embeddings = combined[-self.recent_window_size:]
    
    def clear_recent_context(self) -> None:
        """Clear the recent embeddings context."""
        self.recent_embeddings = None
    
    def get_indexed_sources(self) -> Dict[str, Dict[str, Any]]:
        """Get information about indexed sources."""
        return self.indexer.get_indexed_sources()
    
    def remove_source(self, source_key: str) -> int:
        """Remove an indexed source."""
        return self.indexer.remove_source(source_key)
    
    def clear_index(self) -> None:
        """Clear all indexed content."""
        self.indexer.clear()
        self.recent_embeddings = None
    
    def stats(self) -> Dict[str, Any]:
        """Get service statistics."""
        indexer_stats = self.indexer.stats()
        return {
            **indexer_stats,
            "pyversity_available": self._pyversity_available,
            "recent_context_size": len(self.recent_embeddings) if self.recent_embeddings is not None else 0,
        }


class AgentRAGContext:
    """
    Per-agent RAG context manager.
    
    Manages RAG context for individual agents, tracking their
    conversations and outputs separately.
    """
    
    def __init__(self, rag_service: RAGService, agent_id: str):
        self.rag_service = rag_service
        self.agent_id = agent_id
        self.conversation_indexed = False
        self.output_folders: List[str] = []
    
    def index_conversation(self, messages: List[Dict[str, Any]]) -> int:
        """Index this agent's conversation."""
        count = self.rag_service.index_agent_conversation(self.agent_id, messages)
        self.conversation_indexed = True
        return count
    
    def index_output_folder(self, folder_path: str) -> int:
        """Index an output folder for this agent."""
        count = self.rag_service.index_folder(folder_path, agent_id=self.agent_id)
        if folder_path not in self.output_folders:
            self.output_folders.append(folder_path)
        return count
    
    def index_output(self, output: str, output_type: str = "response") -> int:
        """Index a single output from this agent."""
        return self.rag_service.index_agent_output(self.agent_id, output, output_type)
    
    def retrieve(
        self,
        query: str,
        k: int = 5,
        diversity: float = 0.5,
        strategy: str = "ssd",
        include_other_agents: bool = False,
    ) -> RetrievalResult:
        """
        Retrieve context for this agent.
        
        Args:
            query: Query text
            k: Number of chunks
            diversity: Diversity parameter
            strategy: Diversification strategy
            include_other_agents: If True, search all agents; if False, only this agent
        """
        agent_filter = None if include_other_agents else self.agent_id
        return self.rag_service.retrieve(
            query=query,
            k=k,
            diversity=diversity,
            strategy=strategy,
            agent_filter=agent_filter,
        )
