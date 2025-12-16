"""
Document Processing Pipeline.

Purpose: End-to-end document processing with OCR and diversification
Inputs: Document file paths
Outputs: Processed, diversified document chunks for RAG
"""

from typing import Any, Dict, List, Optional
from pathlib import Path
import numpy as np

from .hunyuan_ocr import HunyuanOCRProcessor
from .pyversity_rag import PyversityDiversifier


class DocumentPipeline:
    """
    End-to-end document processing pipeline.
    
    Combines:
    1. HunyuanOCR for text extraction
    2. Text chunking
    3. Embedding generation
    4. Pyversity diversification for RAG
    """
    
    def __init__(
        self,
        ocr_device: str = "auto",
        diversify_strategy: str = "MMR",
        diversity: float = 0.5,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ):
        self.ocr = HunyuanOCRProcessor(device=ocr_device)
        self.diversifier = PyversityDiversifier(
            strategy=diversify_strategy,
            diversity=diversity
        )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._embedding_model = None
    
    def process_document(self, file_path: str) -> Dict[str, Any]:
        """
        Process a document through the full pipeline.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Dictionary with extracted text, chunks, and metadata
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")
        
        # Step 1: Extract text with OCR
        ocr_result = self.ocr.process_image(str(path))
        extracted_text = ocr_result["text"]
        
        # Step 2: Chunk the text
        chunks = self._chunk_text(extracted_text)
        
        # Step 3: Generate embeddings (placeholder)
        embeddings = self._generate_embeddings(chunks)
        
        # Step 4: Generate relevance scores (placeholder - would use query)
        scores = np.ones(len(chunks))  # Equal scores without query
        
        return {
            "source": file_path,
            "extracted_text": extracted_text,
            "chunks": chunks,
            "embeddings": embeddings,
            "num_chunks": len(chunks),
        }
    
    def get_diversified_chunks(
        self,
        chunks: List[str],
        embeddings: np.ndarray,
        query_embedding: Optional[np.ndarray] = None,
        k: int = 10,
    ) -> List[str]:
        """
        Get diversified chunks for a query.
        
        Args:
            chunks: All document chunks
            embeddings: Chunk embeddings
            query_embedding: Query embedding for relevance scoring
            k: Number of chunks to return
            
        Returns:
            List of diversified chunks
        """
        # Calculate relevance scores
        if query_embedding is not None:
            scores = self._calculate_similarity(embeddings, query_embedding)
        else:
            scores = np.ones(len(chunks))
        
        # Diversify
        return self.diversifier.diversify_chunks(chunks, embeddings, scores, k)
    
    def _chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks."""
        if not text:
            return []
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            
            # Try to break at sentence boundary
            if end < len(text):
                last_period = chunk.rfind(".")
                last_newline = chunk.rfind("\n")
                break_point = max(last_period, last_newline)
                if break_point > self.chunk_size // 2:
                    chunk = chunk[:break_point + 1]
                    end = start + break_point + 1
            
            chunks.append(chunk.strip())
            start = end - self.chunk_overlap
        
        return [c for c in chunks if c]  # Filter empty chunks
    
    def _generate_embeddings(self, chunks: List[str]) -> np.ndarray:
        """Generate embeddings for chunks (placeholder)."""
        # Would use actual embedding model in production
        # For now, return random embeddings
        return np.random.randn(len(chunks), 256).astype(np.float32)
    
    def _calculate_similarity(
        self, 
        embeddings: np.ndarray, 
        query_embedding: np.ndarray
    ) -> np.ndarray:
        """Calculate cosine similarity between embeddings and query."""
        # Normalize
        embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        
        # Cosine similarity
        return np.dot(embeddings_norm, query_norm)
