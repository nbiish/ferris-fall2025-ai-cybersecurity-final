"""
HGM Agent Persistence Module.

Purpose: Persistent conversation storage for HGM agents using LangGraph checkpointers.
Each agent gets its own SQLite database file in its dedicated folder.

Architecture:
- Per-agent SQLite database: {workspace}/agents/{agent_id}/checkpoints.db
- Conversation history stored as LangGraph checkpoints
- Thread-based isolation for multiple conversation sessions
- CMP history and metrics persistence
"""

import os
import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
import uuid

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage


@dataclass
class AgentCheckpoint:
    """Checkpoint data for an HGM agent conversation."""
    
    checkpoint_id: str
    thread_id: str
    agent_id: str
    timestamp: str
    messages: List[Dict[str, Any]]
    cmp_score: float
    cmp_history: List[float]
    generation: int
    tools_used: List[str]
    iteration: int
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentCheckpoint":
        return cls(**data)


class HGMCheckpointer:
    """
    SQLite-based checkpointer for HGM agents.
    
    Each agent has its own SQLite database file for conversation persistence.
    Supports multiple threads (conversation sessions) per agent.
    """
    
    SCHEMA = """
    CREATE TABLE IF NOT EXISTS checkpoints (
        checkpoint_id TEXT PRIMARY KEY,
        thread_id TEXT NOT NULL,
        agent_id TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        messages TEXT NOT NULL,
        cmp_score REAL NOT NULL,
        cmp_history TEXT NOT NULL,
        generation INTEGER NOT NULL,
        tools_used TEXT NOT NULL,
        iteration INTEGER NOT NULL,
        metadata TEXT NOT NULL,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    );
    
    CREATE INDEX IF NOT EXISTS idx_thread_id ON checkpoints(thread_id);
    CREATE INDEX IF NOT EXISTS idx_agent_id ON checkpoints(agent_id);
    CREATE INDEX IF NOT EXISTS idx_timestamp ON checkpoints(timestamp DESC);
    
    CREATE TABLE IF NOT EXISTS agent_state (
        agent_id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        generation INTEGER NOT NULL,
        cmp_score REAL NOT NULL,
        cmp_history TEXT NOT NULL,
        total_successes INTEGER DEFAULT 0,
        total_failures INTEGER DEFAULT 0,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT DEFAULT CURRENT_TIMESTAMP
    );
    
    CREATE TABLE IF NOT EXISTS threads (
        thread_id TEXT PRIMARY KEY,
        agent_id TEXT NOT NULL,
        title TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        last_active TEXT DEFAULT CURRENT_TIMESTAMP,
        message_count INTEGER DEFAULT 0
    );
    
    CREATE INDEX IF NOT EXISTS idx_threads_agent ON threads(agent_id);
    """
    
    def __init__(self, db_path: str):
        """
        Initialize checkpointer with database path.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._ensure_directory()
        self._init_db()
    
    def _ensure_directory(self) -> None:
        """Ensure the directory for the database exists."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
    
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
    
    def put(
        self,
        thread_id: str,
        agent_id: str,
        messages: List[BaseMessage],
        cmp_score: float,
        cmp_history: List[float],
        generation: int,
        tools_used: List[str],
        iteration: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Store a checkpoint.
        
        Returns:
            checkpoint_id: Unique ID for this checkpoint
        """
        checkpoint_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat()
        
        # Serialize messages
        serialized_messages = [
            {
                "type": type(msg).__name__,
                "content": msg.content,
                "additional_kwargs": getattr(msg, "additional_kwargs", {}),
            }
            for msg in messages
        ]
        
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO checkpoints 
                (checkpoint_id, thread_id, agent_id, timestamp, messages, 
                 cmp_score, cmp_history, generation, tools_used, iteration, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    checkpoint_id,
                    thread_id,
                    agent_id,
                    timestamp,
                    json.dumps(serialized_messages),
                    cmp_score,
                    json.dumps(cmp_history),
                    generation,
                    json.dumps(tools_used),
                    iteration,
                    json.dumps(metadata or {}),
                ),
            )
            
            # Update thread info
            conn.execute(
                """
                INSERT INTO threads (thread_id, agent_id, last_active, message_count)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(thread_id) DO UPDATE SET
                    last_active = excluded.last_active,
                    message_count = message_count + 1
                """,
                (thread_id, agent_id, timestamp, len(messages)),
            )
            
            conn.commit()
        
        return checkpoint_id
    
    def get(self, thread_id: str) -> Optional[AgentCheckpoint]:
        """
        Get the latest checkpoint for a thread.
        
        Returns:
            Latest checkpoint or None if not found
        """
        with self._get_connection() as conn:
            row = conn.execute(
                """
                SELECT * FROM checkpoints 
                WHERE thread_id = ? 
                ORDER BY timestamp DESC 
                LIMIT 1
                """,
                (thread_id,),
            ).fetchone()
            
            if not row:
                return None
            
            return AgentCheckpoint(
                checkpoint_id=row["checkpoint_id"],
                thread_id=row["thread_id"],
                agent_id=row["agent_id"],
                timestamp=row["timestamp"],
                messages=json.loads(row["messages"]),
                cmp_score=row["cmp_score"],
                cmp_history=json.loads(row["cmp_history"]),
                generation=row["generation"],
                tools_used=json.loads(row["tools_used"]),
                iteration=row["iteration"],
                metadata=json.loads(row["metadata"]),
            )
    
    def get_by_id(self, checkpoint_id: str) -> Optional[AgentCheckpoint]:
        """Get a specific checkpoint by ID."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM checkpoints WHERE checkpoint_id = ?",
                (checkpoint_id,),
            ).fetchone()
            
            if not row:
                return None
            
            return AgentCheckpoint(
                checkpoint_id=row["checkpoint_id"],
                thread_id=row["thread_id"],
                agent_id=row["agent_id"],
                timestamp=row["timestamp"],
                messages=json.loads(row["messages"]),
                cmp_score=row["cmp_score"],
                cmp_history=json.loads(row["cmp_history"]),
                generation=row["generation"],
                tools_used=json.loads(row["tools_used"]),
                iteration=row["iteration"],
                metadata=json.loads(row["metadata"]),
            )
    
    def list_checkpoints(
        self, 
        thread_id: str, 
        limit: int = 50
    ) -> List[AgentCheckpoint]:
        """List checkpoints for a thread (time-travel support)."""
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM checkpoints 
                WHERE thread_id = ? 
                ORDER BY timestamp DESC 
                LIMIT ?
                """,
                (thread_id, limit),
            ).fetchall()
            
            return [
                AgentCheckpoint(
                    checkpoint_id=row["checkpoint_id"],
                    thread_id=row["thread_id"],
                    agent_id=row["agent_id"],
                    timestamp=row["timestamp"],
                    messages=json.loads(row["messages"]),
                    cmp_score=row["cmp_score"],
                    cmp_history=json.loads(row["cmp_history"]),
                    generation=row["generation"],
                    tools_used=json.loads(row["tools_used"]),
                    iteration=row["iteration"],
                    metadata=json.loads(row["metadata"]),
                )
                for row in rows
            ]
    
    def list_threads(self, agent_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all threads, optionally filtered by agent."""
        with self._get_connection() as conn:
            if agent_id:
                rows = conn.execute(
                    """
                    SELECT * FROM threads 
                    WHERE agent_id = ? 
                    ORDER BY last_active DESC
                    """,
                    (agent_id,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM threads ORDER BY last_active DESC"
                ).fetchall()
            
            return [dict(row) for row in rows]
    
    def save_agent_state(
        self,
        agent_id: str,
        name: str,
        generation: int,
        cmp_score: float,
        cmp_history: List[float],
        total_successes: int = 0,
        total_failures: int = 0,
    ) -> None:
        """Save or update agent state."""
        timestamp = datetime.utcnow().isoformat()
        
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO agent_state 
                (agent_id, name, generation, cmp_score, cmp_history, 
                 total_successes, total_failures, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(agent_id) DO UPDATE SET
                    name = excluded.name,
                    generation = excluded.generation,
                    cmp_score = excluded.cmp_score,
                    cmp_history = excluded.cmp_history,
                    total_successes = excluded.total_successes,
                    total_failures = excluded.total_failures,
                    updated_at = excluded.updated_at
                """,
                (
                    agent_id,
                    name,
                    generation,
                    cmp_score,
                    json.dumps(cmp_history),
                    total_successes,
                    total_failures,
                    timestamp,
                ),
            )
            conn.commit()
    
    def get_agent_state(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get saved agent state."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM agent_state WHERE agent_id = ?",
                (agent_id,),
            ).fetchone()
            
            if not row:
                return None
            
            result = dict(row)
            result["cmp_history"] = json.loads(result["cmp_history"])
            return result
    
    def deserialize_messages(
        self, 
        serialized: List[Dict[str, Any]]
    ) -> List[BaseMessage]:
        """Deserialize messages from checkpoint format."""
        messages = []
        for msg_data in serialized:
            msg_type = msg_data.get("type", "HumanMessage")
            content = msg_data.get("content", "")
            
            if msg_type == "HumanMessage":
                messages.append(HumanMessage(content=content))
            elif msg_type == "AIMessage":
                messages.append(AIMessage(content=content))
            elif msg_type == "SystemMessage":
                messages.append(SystemMessage(content=content))
            else:
                # Default to HumanMessage for unknown types
                messages.append(HumanMessage(content=content))
        
        return messages


class AgentFolderManager:
    """
    Manages per-agent folder structure in the workspace.
    
    Folder structure:
    {workspace}/agents/{agent_id}/
    ├── checkpoints.db      # SQLite database for conversation persistence
    ├── config.json         # Agent configuration
    ├── documents/          # Agent-specific documents
    ├── outputs/            # Generated outputs
    └── voice/              # Voice recordings and transcripts
    """
    
    SUBFOLDERS = ["documents", "outputs", "voice"]
    
    def __init__(self, workspace_path: str):
        """
        Initialize folder manager.
        
        Args:
            workspace_path: Root workspace path
        """
        self.workspace_path = workspace_path
        self.agents_root = os.path.join(workspace_path, "agents")
    
    def get_agent_folder(self, agent_id: str) -> str:
        """Get the folder path for an agent."""
        return os.path.join(self.agents_root, agent_id)
    
    def ensure_agent_folder(self, agent_id: str) -> str:
        """
        Ensure agent folder exists with all subfolders.
        
        Returns:
            Path to agent folder
        """
        agent_folder = self.get_agent_folder(agent_id)
        
        # Create main folder
        Path(agent_folder).mkdir(parents=True, exist_ok=True)
        
        # Create subfolders
        for subfolder in self.SUBFOLDERS:
            Path(os.path.join(agent_folder, subfolder)).mkdir(exist_ok=True)
        
        return agent_folder
    
    def get_checkpointer(self, agent_id: str) -> HGMCheckpointer:
        """
        Get checkpointer for an agent.
        
        Creates agent folder if it doesn't exist.
        """
        agent_folder = self.ensure_agent_folder(agent_id)
        db_path = os.path.join(agent_folder, "checkpoints.db")
        return HGMCheckpointer(db_path)
    
    def save_agent_config(
        self, 
        agent_id: str, 
        config: Dict[str, Any]
    ) -> None:
        """Save agent configuration to JSON file."""
        agent_folder = self.ensure_agent_folder(agent_id)
        config_path = os.path.join(agent_folder, "config.json")
        
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
    
    def load_agent_config(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Load agent configuration from JSON file."""
        config_path = os.path.join(
            self.get_agent_folder(agent_id), "config.json"
        )
        
        if not os.path.exists(config_path):
            return None
        
        with open(config_path, "r") as f:
            return json.load(f)
    
    def list_agents(self) -> List[str]:
        """List all agent IDs with folders."""
        if not os.path.exists(self.agents_root):
            return []
        
        return [
            name for name in os.listdir(self.agents_root)
            if os.path.isdir(os.path.join(self.agents_root, name))
        ]
    
    def get_voice_folder(self, agent_id: str) -> str:
        """Get voice folder path for an agent."""
        return os.path.join(self.get_agent_folder(agent_id), "voice")
    
    def get_documents_folder(self, agent_id: str) -> str:
        """Get documents folder path for an agent."""
        return os.path.join(self.get_agent_folder(agent_id), "documents")
    
    def get_outputs_folder(self, agent_id: str) -> str:
        """Get outputs folder path for an agent."""
        return os.path.join(self.get_agent_folder(agent_id), "outputs")


# Factory function for easy instantiation
def create_persistent_agent_manager(workspace_path: str) -> AgentFolderManager:
    """
    Create an agent folder manager for the workspace.
    
    Args:
        workspace_path: Root workspace path
        
    Returns:
        AgentFolderManager instance
    """
    return AgentFolderManager(workspace_path)
