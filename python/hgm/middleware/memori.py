"""
Memori Memory Middleware for DeepHGM Agents.

Integrates MemoriLabs/Memori SQL-native memory layer for efficient agent memory.
Each agent gets its own session with entity/process attribution for isolated memory.

Architecture:
- Entity: The agent itself (agent_id)
- Process: The task/workflow being executed
- Session: Groups related interactions within a task

Reference: https://github.com/MemoriLabs/Memori
"""

from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
import sqlite3
import os
import json
import uuid

from langchain_core.tools import BaseTool
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from .base import DeepHGMMiddleware, MiddlewareState, InterruptRequest


@dataclass
class MemoriConfig:
    """Configuration for Memori memory integration."""
    
    db_path: Optional[str] = None
    entity_id: Optional[str] = None
    process_id: str = "hgm-agent"
    enable_advanced_augmentation: bool = True
    memory_types: List[str] = field(default_factory=lambda: [
        "facts", "preferences", "events", "skills", "relationships"
    ])
    max_context_memories: int = 10
    semantic_search_enabled: bool = True
    

@dataclass
class MemoryEntry:
    """A single memory entry in the Memori system."""
    
    id: str
    entity_id: str
    process_id: str
    session_id: str
    memory_type: str  # facts, preferences, events, skills, relationships, rules
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    importance: float = 0.5
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "entity_id": self.entity_id,
            "process_id": self.process_id,
            "session_id": self.session_id,
            "memory_type": self.memory_type,
            "content": self.content,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "importance": self.importance,
        }


class MemoriStorageBackend:
    """
    SQLite-based storage backend for Memori memories.
    
    Implements a compatible schema with MemoriLabs/Memori for agent memories.
    Supports semantic search via embeddings when available.
    """
    
    SCHEMA = """
    -- Core memories table (third normal form)
    CREATE TABLE IF NOT EXISTS memories (
        id TEXT PRIMARY KEY,
        entity_id TEXT NOT NULL,
        process_id TEXT NOT NULL,
        session_id TEXT NOT NULL,
        memory_type TEXT NOT NULL,
        content TEXT NOT NULL,
        metadata TEXT NOT NULL DEFAULT '{}',
        embedding BLOB,
        created_at TEXT NOT NULL,
        importance REAL DEFAULT 0.5,
        UNIQUE(entity_id, process_id, content)
    );
    
    CREATE INDEX IF NOT EXISTS idx_memories_entity ON memories(entity_id);
    CREATE INDEX IF NOT EXISTS idx_memories_session ON memories(session_id);
    CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type);
    CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories(importance DESC);
    
    -- Semantic triples for knowledge graph
    CREATE TABLE IF NOT EXISTS semantic_triples (
        id TEXT PRIMARY KEY,
        subject TEXT NOT NULL,
        predicate TEXT NOT NULL,
        object TEXT NOT NULL,
        entity_id TEXT NOT NULL,
        session_id TEXT NOT NULL,
        confidence REAL DEFAULT 1.0,
        created_at TEXT NOT NULL
    );
    
    CREATE INDEX IF NOT EXISTS idx_triples_subject ON semantic_triples(subject);
    CREATE INDEX IF NOT EXISTS idx_triples_entity ON semantic_triples(entity_id);
    
    -- Sessions tracking
    CREATE TABLE IF NOT EXISTS memori_sessions (
        session_id TEXT PRIMARY KEY,
        entity_id TEXT NOT NULL,
        process_id TEXT NOT NULL,
        started_at TEXT NOT NULL,
        last_active TEXT NOT NULL,
        memory_count INTEGER DEFAULT 0,
        metadata TEXT DEFAULT '{}'
    );
    
    CREATE INDEX IF NOT EXISTS idx_sessions_entity ON memori_sessions(entity_id);
    """
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._ensure_directory()
        self._init_db()
    
    def _ensure_directory(self) -> None:
        """Ensure parent directory exists."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
    
    def _init_db(self) -> None:
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(self.SCHEMA)
            conn.commit()
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection with row factory."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def store_memory(self, memory: MemoryEntry) -> str:
        """Store a memory entry."""
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO memories 
                (id, entity_id, process_id, session_id, memory_type, 
                 content, metadata, created_at, importance)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    memory.id,
                    memory.entity_id,
                    memory.process_id,
                    memory.session_id,
                    memory.memory_type,
                    memory.content,
                    json.dumps(memory.metadata),
                    memory.created_at,
                    memory.importance,
                ),
            )
            
            # Update session stats
            conn.execute(
                """
                INSERT INTO memori_sessions (session_id, entity_id, process_id, started_at, last_active, memory_count)
                VALUES (?, ?, ?, ?, ?, 1)
                ON CONFLICT(session_id) DO UPDATE SET
                    last_active = excluded.last_active,
                    memory_count = memory_count + 1
                """,
                (
                    memory.session_id,
                    memory.entity_id,
                    memory.process_id,
                    memory.created_at,
                    memory.created_at,
                ),
            )
            conn.commit()
        
        return memory.id
    
    def store_triple(
        self,
        subject: str,
        predicate: str,
        obj: str,
        entity_id: str,
        session_id: str,
        confidence: float = 1.0,
    ) -> str:
        """Store a semantic triple for knowledge graph."""
        triple_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat()
        
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO semantic_triples 
                (id, subject, predicate, object, entity_id, session_id, confidence, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (triple_id, subject, predicate, obj, entity_id, session_id, confidence, timestamp),
            )
            conn.commit()
        
        return triple_id
    
    def get_memories(
        self,
        entity_id: str,
        memory_types: Optional[List[str]] = None,
        session_id: Optional[str] = None,
        limit: int = 50,
    ) -> List[MemoryEntry]:
        """Retrieve memories for an entity."""
        with self._get_connection() as conn:
            query = "SELECT * FROM memories WHERE entity_id = ?"
            params: List[Any] = [entity_id]
            
            if memory_types:
                placeholders = ",".join("?" * len(memory_types))
                query += f" AND memory_type IN ({placeholders})"
                params.extend(memory_types)
            
            if session_id:
                query += " AND session_id = ?"
                params.append(session_id)
            
            query += " ORDER BY importance DESC, created_at DESC LIMIT ?"
            params.append(limit)
            
            rows = conn.execute(query, params).fetchall()
            
            return [
                MemoryEntry(
                    id=row["id"],
                    entity_id=row["entity_id"],
                    process_id=row["process_id"],
                    session_id=row["session_id"],
                    memory_type=row["memory_type"],
                    content=row["content"],
                    metadata=json.loads(row["metadata"]),
                    created_at=row["created_at"],
                    importance=row["importance"],
                )
                for row in rows
            ]
    
    def get_related_triples(
        self,
        entity_id: str,
        subject: Optional[str] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Get semantic triples related to an entity."""
        with self._get_connection() as conn:
            query = "SELECT * FROM semantic_triples WHERE entity_id = ?"
            params: List[Any] = [entity_id]
            
            if subject:
                query += " AND subject = ?"
                params.append(subject)
            
            query += " ORDER BY confidence DESC, created_at DESC LIMIT ?"
            params.append(limit)
            
            rows = conn.execute(query, params).fetchall()
            return [dict(row) for row in rows]
    
    def search_memories(
        self,
        entity_id: str,
        query: str,
        limit: int = 10,
    ) -> List[MemoryEntry]:
        """
        Search memories using text matching.
        
        For production, integrate with embedding-based semantic search.
        """
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM memories 
                WHERE entity_id = ? AND content LIKE ?
                ORDER BY importance DESC, created_at DESC
                LIMIT ?
                """,
                (entity_id, f"%{query}%", limit),
            ).fetchall()
            
            return [
                MemoryEntry(
                    id=row["id"],
                    entity_id=row["entity_id"],
                    process_id=row["process_id"],
                    session_id=row["session_id"],
                    memory_type=row["memory_type"],
                    content=row["content"],
                    metadata=json.loads(row["metadata"]),
                    created_at=row["created_at"],
                    importance=row["importance"],
                )
                for row in rows
            ]


class MemoriMiddleware(DeepHGMMiddleware):
    """
    Memori Memory Middleware for DeepHGM agents.
    
    Integrates MemoriLabs/Memori patterns for efficient agent memory:
    - Entity attribution (agent_id)
    - Process attribution (task/workflow)
    - Session management (grouped interactions)
    - Multi-level memory (facts, preferences, events, skills, relationships)
    - Semantic triples for knowledge graph
    
    Each agent maintains isolated memory with its own session.
    """
    
    def __init__(
        self,
        config: Optional[MemoriConfig] = None,
        priority: int = 50,
    ):
        super().__init__("memori", priority)
        self.config = config or MemoriConfig()
        self._storage: Optional[MemoriStorageBackend] = None
        self._current_session_id: Optional[str] = None
        self._entity_id: Optional[str] = None
        self._memory_cache: List[MemoryEntry] = []
        
        # Initialize storage if db_path provided
        if self.config.db_path:
            self._init_storage(self.config.db_path)
    
    def _init_storage(self, db_path: str) -> None:
        """Initialize storage backend."""
        self._storage = MemoriStorageBackend(db_path)
    
    def configure_for_agent(
        self,
        agent_id: str,
        workspace_path: str,
        process_id: Optional[str] = None,
    ) -> str:
        """
        Configure middleware for a specific agent.
        
        Returns:
            session_id: New session ID for this agent's work
        """
        self._entity_id = agent_id
        self.config.entity_id = agent_id
        
        if process_id:
            self.config.process_id = process_id
        
        # Initialize storage in agent's workspace
        if not self._storage:
            db_path = os.path.join(workspace_path, "agents", agent_id, "memori.db")
            self._init_storage(db_path)
        
        # Create new session
        self._current_session_id = str(uuid.uuid4())
        
        return self._current_session_id
    
    def new_session(self) -> str:
        """Start a new session, keeping the same entity/process attribution."""
        self._current_session_id = str(uuid.uuid4())
        self._memory_cache = []
        return self._current_session_id
    
    def set_session(self, session_id: str) -> None:
        """Resume an existing session."""
        self._current_session_id = session_id
        self._memory_cache = []
    
    @property
    def session_id(self) -> Optional[str]:
        """Current session ID."""
        return self._current_session_id
    
    def get_system_prompt_addition(self) -> Optional[str]:
        """Add memory context to system prompt."""
        if not self._storage or not self._entity_id:
            return None
        
        # Retrieve relevant memories
        memories = self._storage.get_memories(
            entity_id=self._entity_id,
            memory_types=self.config.memory_types,
            limit=self.config.max_context_memories,
        )
        
        if not memories:
            return None
        
        # Format memories for context injection
        memory_sections = {}
        for mem in memories:
            if mem.memory_type not in memory_sections:
                memory_sections[mem.memory_type] = []
            memory_sections[mem.memory_type].append(mem.content)
        
        context_parts = ["## Agent Memory Context"]
        for mem_type, contents in memory_sections.items():
            context_parts.append(f"\n### {mem_type.title()}")
            for content in contents[:3]:  # Limit per type
                context_parts.append(f"- {content}")
        
        return "\n".join(context_parts)
    
    def add_memory(
        self,
        content: str,
        memory_type: str = "facts",
        metadata: Optional[Dict[str, Any]] = None,
        importance: float = 0.5,
    ) -> Optional[str]:
        """
        Add a memory entry.
        
        Args:
            content: Memory content
            memory_type: Type (facts, preferences, events, skills, relationships, rules)
            metadata: Additional metadata
            importance: Memory importance (0-1)
            
        Returns:
            Memory ID or None if storage not initialized
        """
        if not self._storage or not self._entity_id or not self._current_session_id:
            return None
        
        memory = MemoryEntry(
            id=str(uuid.uuid4()),
            entity_id=self._entity_id,
            process_id=self.config.process_id,
            session_id=self._current_session_id,
            memory_type=memory_type,
            content=content,
            metadata=metadata or {},
            importance=importance,
        )
        
        self._memory_cache.append(memory)
        return self._storage.store_memory(memory)
    
    def add_fact(self, content: str, importance: float = 0.5) -> Optional[str]:
        """Add a fact memory."""
        return self.add_memory(content, "facts", importance=importance)
    
    def add_preference(self, content: str, importance: float = 0.6) -> Optional[str]:
        """Add a preference memory."""
        return self.add_memory(content, "preferences", importance=importance)
    
    def add_event(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Add an event memory."""
        return self.add_memory(content, "events", metadata=metadata, importance=0.4)
    
    def add_skill(self, content: str, importance: float = 0.7) -> Optional[str]:
        """Add a skill memory."""
        return self.add_memory(content, "skills", importance=importance)
    
    def add_relationship(self, content: str, importance: float = 0.6) -> Optional[str]:
        """Add a relationship memory."""
        return self.add_memory(content, "relationships", importance=importance)
    
    def add_triple(
        self,
        subject: str,
        predicate: str,
        obj: str,
        confidence: float = 1.0,
    ) -> Optional[str]:
        """
        Add a semantic triple to the knowledge graph.
        
        Example: add_triple("agent", "learned", "python parsing")
        """
        if not self._storage or not self._entity_id or not self._current_session_id:
            return None
        
        return self._storage.store_triple(
            subject=subject,
            predicate=predicate,
            obj=obj,
            entity_id=self._entity_id,
            session_id=self._current_session_id,
            confidence=confidence,
        )
    
    def recall(
        self,
        query: Optional[str] = None,
        memory_types: Optional[List[str]] = None,
        limit: int = 10,
    ) -> List[MemoryEntry]:
        """
        Recall memories relevant to a query.
        
        Args:
            query: Search query (if None, returns recent memories)
            memory_types: Filter by memory types
            limit: Maximum memories to return
            
        Returns:
            List of relevant memories
        """
        if not self._storage or not self._entity_id:
            return []
        
        if query:
            return self._storage.search_memories(
                entity_id=self._entity_id,
                query=query,
                limit=limit,
            )
        else:
            return self._storage.get_memories(
                entity_id=self._entity_id,
                memory_types=memory_types or self.config.memory_types,
                limit=limit,
            )
    
    def get_knowledge_graph(self, subject: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get semantic triples from knowledge graph."""
        if not self._storage or not self._entity_id:
            return []
        
        return self._storage.get_related_triples(
            entity_id=self._entity_id,
            subject=subject,
        )
    
    def pre_model(self, state: MiddlewareState) -> MiddlewareState:
        """Inject memory context before model call."""
        # Memory context is already injected via get_system_prompt_addition
        # Here we can add dynamic context based on recent messages
        
        if not self._storage or not self._entity_id:
            return state
        
        # Extract key terms from recent messages for memory search
        messages = state.get("messages", [])
        if messages:
            recent_content = " ".join(
                msg.content for msg in messages[-3:]
                if hasattr(msg, "content")
            )
            
            # Search for relevant memories
            if recent_content:
                relevant_memories = self.recall(query=recent_content[:200], limit=5)
                
                # Store in context for potential use
                context = state.get("context", {})
                context["relevant_memories"] = [m.to_dict() for m in relevant_memories]
                state["context"] = context
        
        return state
    
    def post_model(self, state: MiddlewareState, response: Any) -> MiddlewareState:
        """Extract and store memories from model response."""
        if not self._storage or not self._entity_id:
            return state
        
        # Extract potential memories from response
        # In production, use LLM to extract structured memories
        if hasattr(response, "content") and response.content:
            content = response.content
            
            # Simple heuristic: store significant responses as events
            if len(content) > 100:
                self.add_event(
                    f"Generated response: {content[:200]}...",
                    metadata={"full_length": len(content)},
                )
        
        return state
    
    def post_tool(self, state: MiddlewareState, tool_name: str, result: Any) -> MiddlewareState:
        """Record tool usage as skills/events."""
        if not self._storage or not self._entity_id:
            return state
        
        # Record successful tool usage as a skill
        self.add_skill(
            f"Used tool '{tool_name}' successfully",
            importance=0.6,
        )
        
        # Add semantic triple
        self.add_triple(
            subject="agent",
            predicate="used_tool",
            obj=tool_name,
            confidence=1.0,
        )
        
        return state
    
    def on_error(self, state: MiddlewareState, error: Exception) -> MiddlewareState:
        """Record errors for learning."""
        if self._storage and self._entity_id:
            self.add_event(
                f"Encountered error: {str(error)[:200]}",
                metadata={"error_type": type(error).__name__},
            )
        
        return state


# Factory function
def create_memori_middleware(
    agent_id: str,
    workspace_path: str,
    process_id: str = "hgm-agent",
) -> MemoriMiddleware:
    """
    Create a configured Memori middleware for an agent.
    
    Args:
        agent_id: Unique agent identifier
        workspace_path: Root workspace path
        process_id: Process/workflow identifier
        
    Returns:
        Configured MemoriMiddleware with new session
    """
    config = MemoriConfig(
        entity_id=agent_id,
        process_id=process_id,
    )
    
    middleware = MemoriMiddleware(config=config)
    middleware.configure_for_agent(agent_id, workspace_path, process_id)
    
    return middleware
