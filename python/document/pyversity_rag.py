"""
Pyversity Integration for RAG diversification.

Purpose: Diversify retrieval results to avoid redundancy
Inputs: Embeddings and relevance scores
Outputs: Diversified result indices
"""

from typing import Any, Dict, List, Optional
import numpy as np


class PyversityDiversifier:
    """
    Diversifier using Pyversity for RAG result diversification.
    
    Pyversity implements several diversification strategies:
    - MMR (Maximal Marginal Relevance)
    - MSD (Maximum Sum Dispersion)
    - DPP (Determinantal Point Process)
    - COVER (Facility-Location)
    - SSD (Sliding Spectrum Decomposition)
    """
    
    def __init__(self, strategy: str = "MMR", diversity: float = 0.5):
        """
        Initialize diversifier.
        
        Args:
            strategy: Diversification strategy (MMR, MSD, DPP, COVER, SSD)
            diversity: Trade-off between relevance and diversity (0.0-1.0)
        """
        self.strategy = strategy
        self.diversity = diversity
        self._pyversity = None
    
    def _load_pyversity(self):
        """Lazy load pyversity."""
        if self._pyversity is None:
            try:
                import pyversity
                self._pyversity = pyversity
            except ImportError:
                raise ImportError("pyversity not installed. Run: pip install pyversity")
    
    def diversify(
        self,
        embeddings: np.ndarray,
        scores: np.ndarray,
        k: int = 10,
        recent_embeddings: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Diversify retrieval results.
        
        Args:
            embeddings: Document embeddings (n_docs, embedding_dim)
            scores: Relevance scores for each document
            k: Number of items to select
            recent_embeddings: For SSD strategy, embeddings from recent context
            
        Returns:
            Dictionary with diversified indices and metadata
        """
        self._load_pyversity()
        
        # Map strategy string to enum
        strategy_map = {
            "MMR": self._pyversity.Strategy.MMR,
            "MSD": self._pyversity.Strategy.MSD,
            "DPP": self._pyversity.Strategy.DPP,
            "COVER": self._pyversity.Strategy.COVER,
            "SSD": self._pyversity.Strategy.SSD,
        }
        
        strategy = strategy_map.get(self.strategy, self._pyversity.Strategy.MMR)
        
        # Build kwargs
        kwargs = {
            "embeddings": embeddings,
            "scores": scores,
            "k": k,
            "strategy": strategy,
            "diversity": self.diversity,
        }
        
        # Add recent embeddings for SSD
        if self.strategy == "SSD" and recent_embeddings is not None:
            kwargs["recent_embeddings"] = recent_embeddings
        
        # Diversify
        result = self._pyversity.diversify(**kwargs)
        
        return {
            "indices": result.indices.tolist(),
            "selection_scores": result.selection_scores.tolist() if hasattr(result, "selection_scores") else None,
            "strategy": self.strategy,
            "diversity": self.diversity,
            "k": k,
        }
    
    def diversify_chunks(
        self,
        chunks: List[str],
        embeddings: np.ndarray,
        scores: np.ndarray,
        k: int = 10,
    ) -> List[str]:
        """
        Diversify and return actual chunk content.
        
        Args:
            chunks: List of text chunks
            embeddings: Chunk embeddings
            scores: Relevance scores
            k: Number of chunks to select
            
        Returns:
            List of diversified chunks
        """
        result = self.diversify(embeddings, scores, k)
        indices = result["indices"]
        return [chunks[i] for i in indices]
    
    def set_strategy(self, strategy: str) -> None:
        """Update the diversification strategy."""
        valid_strategies = ["MMR", "MSD", "DPP", "COVER", "SSD"]
        if strategy not in valid_strategies:
            raise ValueError(f"Strategy must be one of {valid_strategies}")
        self.strategy = strategy
    
    def set_diversity(self, diversity: float) -> None:
        """Update the diversity parameter."""
        if not 0.0 <= diversity <= 1.0:
            raise ValueError("Diversity must be between 0.0 and 1.0")
        self.diversity = diversity
