"""
Content Indexer for RAG system.

Purpose: Index agent outputs, conversations, and document folders
Inputs: File paths, text content, agent conversation histories
Outputs: Indexed chunks with embeddings stored in memory/disk

Supports:
- Agent conversation indexing
- Output folder scanning
- Document chunking with overlap
"""

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import json
import hashlib
import numpy as np
from datetime import datetime

from .embeddings import EmbeddingProvider


@dataclass
class Chunk:
    """Represents a text chunk with metadata."""
    id: str
    content: str
    embedding: Optional[np.ndarray] = None
    source_type: str = "unknown"  # "conversation", "file", "agent_output"
    source_path: Optional[str] = None
    agent_id: Optional[str] = None
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "content": self.content,
            "source_type": self.source_type,
            "source_path": self.source_path,
            "agent_id": self.agent_id,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


class ContentIndexer:
    """
    Indexes content from various sources for RAG retrieval.
    
    Maintains an in-memory index of chunks with embeddings,
    supporting incremental updates and source tracking.
    """
    
    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
    ):
        self.embedding_provider = embedding_provider
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        self.chunks: Dict[str, Chunk] = {}
        self.embeddings: Optional[np.ndarray] = None
        self.chunk_ids: List[str] = []
        
        # Track indexed sources
        self.indexed_sources: Dict[str, Dict[str, Any]] = {}
    
    def _generate_chunk_id(self, content: str, source: str) -> str:
        """Generate unique ID for a chunk."""
        hash_input = f"{content}:{source}".encode()
        return hashlib.sha256(hash_input).hexdigest()[:16]
    
    def _split_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks."""
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []
        
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            
            # Try to break at sentence/word boundary
            if end < len(text):
                last_period = chunk.rfind(".")
                last_newline = chunk.rfind("\n")
                last_space = chunk.rfind(" ")
                
                break_point = max(last_period, last_newline, last_space)
                if break_point > self.chunk_size // 2:
                    chunk = chunk[:break_point + 1]
                    end = start + break_point + 1
            
            if chunk.strip():
                chunks.append(chunk.strip())
            
            start = end - self.chunk_overlap
        
        return chunks
    
    def index_text(
        self,
        text: str,
        source_type: str = "text",
        source_path: Optional[str] = None,
        agent_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """
        Index a text string, splitting into chunks.
        
        Returns list of chunk IDs created.
        """
        text_chunks = self._split_text(text)
        if not text_chunks:
            return []
        
        # Generate embeddings
        embeddings = self.embedding_provider.embed(text_chunks)
        
        chunk_ids = []
        for i, (content, embedding) in enumerate(zip(text_chunks, embeddings)):
            source = f"{source_path or 'text'}:{i}"
            chunk_id = self._generate_chunk_id(content, source)
            
            chunk = Chunk(
                id=chunk_id,
                content=content,
                embedding=embedding,
                source_type=source_type,
                source_path=source_path,
                agent_id=agent_id,
                metadata=metadata or {},
            )
            
            self.chunks[chunk_id] = chunk
            chunk_ids.append(chunk_id)
        
        self._rebuild_embedding_matrix()
        return chunk_ids
    
    def index_conversation(
        self,
        messages: List[Dict[str, Any]],
        agent_id: str,
    ) -> List[str]:
        """
        Index an agent's conversation history.
        
        Args:
            messages: List of conversation messages with role/content
            agent_id: ID of the agent
            
        Returns:
            List of chunk IDs created
        """
        chunk_ids = []
        
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            timestamp = msg.get("timestamp", datetime.now().timestamp())
            
            if not content.strip():
                continue
            
            # Format message with role context
            formatted = f"[{role.upper()}]: {content}"
            
            ids = self.index_text(
                text=formatted,
                source_type="conversation",
                source_path=f"agent:{agent_id}",
                agent_id=agent_id,
                metadata={
                    "role": role,
                    "original_timestamp": timestamp,
                    "thinking": msg.get("thinking"),
                },
            )
            chunk_ids.extend(ids)
        
        # Track indexed source
        self.indexed_sources[f"conversation:{agent_id}"] = {
            "type": "conversation",
            "agent_id": agent_id,
            "message_count": len(messages),
            "indexed_at": datetime.now().isoformat(),
        }
        
        return chunk_ids
    
    def index_folder(
        self,
        folder_path: str,
        extensions: Optional[List[str]] = None,
        agent_id: Optional[str] = None,
    ) -> List[str]:
        """
        Index all files in a folder.
        
        Args:
            folder_path: Path to folder to index
            extensions: File extensions to include (default: common text files)
            agent_id: Optional agent ID if this is an agent output folder
            
        Returns:
            List of chunk IDs created
        """
        if extensions is None:
            extensions = [".txt", ".md", ".json", ".py", ".js", ".ts", ".log"]
        
        folder = Path(folder_path)
        if not folder.exists():
            raise ValueError(f"Folder not found: {folder_path}")
        
        chunk_ids = []
        files_indexed = 0
        
        for ext in extensions:
            for file_path in folder.rglob(f"*{ext}"):
                try:
                    content = file_path.read_text(encoding="utf-8")
                    ids = self.index_text(
                        text=content,
                        source_type="file",
                        source_path=str(file_path),
                        agent_id=agent_id,
                        metadata={
                            "filename": file_path.name,
                            "extension": ext,
                            "folder": folder_path,
                        },
                    )
                    chunk_ids.extend(ids)
                    files_indexed += 1
                except Exception as e:
                    print(f"Error indexing {file_path}: {e}")
        
        # Track indexed source
        self.indexed_sources[f"folder:{folder_path}"] = {
            "type": "folder",
            "path": folder_path,
            "files_indexed": files_indexed,
            "agent_id": agent_id,
            "indexed_at": datetime.now().isoformat(),
        }
        
        return chunk_ids
    
    def index_agent_output(
        self,
        agent_id: str,
        output_content: str,
        output_type: str = "response",
    ) -> List[str]:
        """
        Index a single agent output/response.
        
        Args:
            agent_id: ID of the agent
            output_content: The output text to index
            output_type: Type of output (response, thought, code, etc.)
            
        Returns:
            List of chunk IDs created
        """
        return self.index_text(
            text=output_content,
            source_type="agent_output",
            source_path=f"agent:{agent_id}:{output_type}",
            agent_id=agent_id,
            metadata={"output_type": output_type},
        )
    
    def _rebuild_embedding_matrix(self) -> None:
        """Rebuild the embedding matrix from chunks."""
        if not self.chunks:
            self.embeddings = None
            self.chunk_ids = []
            return
        
        self.chunk_ids = list(self.chunks.keys())
        embeddings_list = [
            self.chunks[cid].embedding
            for cid in self.chunk_ids
            if self.chunks[cid].embedding is not None
        ]
        
        if embeddings_list:
            self.embeddings = np.vstack(embeddings_list)
        else:
            self.embeddings = None
    
    def get_all_embeddings(self) -> Tuple[np.ndarray, List[str]]:
        """Get all embeddings and their chunk IDs."""
        if self.embeddings is None:
            return np.array([]), []
        return self.embeddings, self.chunk_ids
    
    def get_chunks_by_ids(self, chunk_ids: List[str]) -> List[Chunk]:
        """Get chunks by their IDs."""
        return [self.chunks[cid] for cid in chunk_ids if cid in self.chunks]
    
    def get_chunks_by_agent(self, agent_id: str) -> List[Chunk]:
        """Get all chunks associated with an agent."""
        return [c for c in self.chunks.values() if c.agent_id == agent_id]
    
    def get_indexed_sources(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all indexed sources."""
        return self.indexed_sources.copy()
    
    def remove_source(self, source_key: str) -> int:
        """
        Remove all chunks from a source.
        
        Args:
            source_key: Key from indexed_sources (e.g., "folder:/path" or "conversation:agent-1")
            
        Returns:
            Number of chunks removed
        """
        source_info = self.indexed_sources.get(source_key)
        if not source_info:
            return 0
        
        # Determine which chunks to remove
        chunks_to_remove = []
        source_type = source_info.get("type")
        
        for chunk_id, chunk in self.chunks.items():
            if source_type == "folder" and chunk.source_path and chunk.source_path.startswith(source_info.get("path", "")):
                chunks_to_remove.append(chunk_id)
            elif source_type == "conversation" and chunk.agent_id == source_info.get("agent_id"):
                chunks_to_remove.append(chunk_id)
        
        # Remove chunks
        for chunk_id in chunks_to_remove:
            del self.chunks[chunk_id]
        
        # Remove source tracking
        del self.indexed_sources[source_key]
        
        # Rebuild embedding matrix
        self._rebuild_embedding_matrix()
        
        return len(chunks_to_remove)
    
    def clear(self) -> None:
        """Clear all indexed content."""
        self.chunks.clear()
        self.embeddings = None
        self.chunk_ids = []
        self.indexed_sources.clear()
    
    def stats(self) -> Dict[str, Any]:
        """Get indexer statistics."""
        return {
            "total_chunks": len(self.chunks),
            "total_sources": len(self.indexed_sources),
            "embedding_dim": self.embeddings.shape[1] if self.embeddings is not None else 0,
            "sources": list(self.indexed_sources.keys()),
        }
