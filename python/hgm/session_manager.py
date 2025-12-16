"""
HGM Session Manager for Self-Improving Agents.

Implements session-based workspace isolation where each agent runs in its own
isolated space with dedicated folder structure, generations tracking, and
configuration management.

Architecture (per agent session):
{workspace}/agents/{agent_id}/
├── memori.db           # Memori memory database
├── checkpoints.db      # Conversation persistence
├── config.json         # Agent configuration
├── session.json        # Current session state
├── generations/        # Evolution history
│   ├── gen_0/          # Initial generation
│   ├── gen_1/          # First evolution
│   └── ...
├── documents/          # Agent-specific documents
├── outputs/            # Generated outputs (code, artifacts)
├── voice/              # Voice recordings
└── logs/               # Execution logs

Reference: https://github.com/metauto-ai/HGM (Huxley-Gödel Machine)
"""

import os
import json
import uuid
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum

from .agent import HGMAgent
from .persistence import HGMCheckpointer, AgentFolderManager


class SessionStatus(Enum):
    """Status of an agent session."""
    INITIALIZING = "initializing"
    ACTIVE = "active"
    PAUSED = "paused"
    EVOLVING = "evolving"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class GenerationRecord:
    """Record of an agent generation (evolution snapshot)."""
    
    generation: int
    agent_id: str
    parent_id: Optional[str]
    cmp_score: float
    created_at: str
    modifications: List[str]
    prompt_template: str
    metrics: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GenerationRecord":
        return cls(**data)


@dataclass
class SessionState:
    """
    State of an agent session.
    
    Tracks all session-specific data for workspace isolation.
    """
    
    session_id: str
    agent_id: str
    agent_name: str
    generation: int
    status: SessionStatus
    
    # Workspace paths
    workspace_path: str
    agent_folder: str
    
    # Memory/persistence IDs
    memori_session_id: Optional[str] = None
    checkpoint_thread_id: Optional[str] = None
    
    # Evolution tracking
    cmp_score: float = 0.5
    cmp_history: List[float] = field(default_factory=list)
    generation_history: List[str] = field(default_factory=list)  # Generation record IDs
    
    # Execution metrics
    total_tasks: int = 0
    successful_tasks: int = 0
    failed_tasks: int = 0
    tools_used: List[str] = field(default_factory=list)
    
    # Timestamps
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    last_active: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    # Configuration
    config: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionState":
        data["status"] = SessionStatus(data["status"])
        return cls(**data)
    
    def update_activity(self) -> None:
        """Update last activity timestamp."""
        self.last_active = datetime.utcnow().isoformat()
    
    def record_task_result(self, success: bool) -> None:
        """Record task execution result."""
        self.total_tasks += 1
        if success:
            self.successful_tasks += 1
        else:
            self.failed_tasks += 1
        self.update_activity()
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_tasks == 0:
            return 0.5
        return self.successful_tasks / self.total_tasks


class HGMSessionManager:
    """
    Manages HGM agent sessions with workspace isolation.
    
    Each agent gets:
    - Unique session ID for tracking
    - Dedicated folder structure
    - Isolated memory (via Memori)
    - Generation tracking for self-improvement
    - Configuration persistence
    
    Key features from metauto-ai/HGM:
    - CMP (Clade Metaproductivity) tracking
    - Generation evolution with lineage
    - Self-improvement through prompt evolution
    """
    
    SESSION_FILE = "session.json"
    CONFIG_FILE = "config.json"
    
    SUBFOLDERS = [
        "documents",
        "outputs",
        "voice",
        "logs",
        "generations",
    ]
    
    def __init__(self, workspace_path: str):
        """
        Initialize session manager.
        
        Args:
            workspace_path: Root workspace path for all agents
        """
        self.workspace_path = workspace_path
        self.agents_root = os.path.join(workspace_path, "agents")
        self._active_sessions: Dict[str, SessionState] = {}
        
        # Ensure agents root exists
        Path(self.agents_root).mkdir(parents=True, exist_ok=True)
    
    def _get_agent_folder(self, agent_id: str) -> str:
        """Get folder path for an agent."""
        return os.path.join(self.agents_root, agent_id)
    
    def _ensure_agent_structure(self, agent_id: str) -> str:
        """
        Ensure complete folder structure for an agent.
        
        Returns:
            Path to agent folder
        """
        agent_folder = self._get_agent_folder(agent_id)
        
        # Create main folder and subfolders
        for subfolder in [""] + self.SUBFOLDERS:
            Path(os.path.join(agent_folder, subfolder)).mkdir(parents=True, exist_ok=True)
        
        return agent_folder
    
    def create_session(
        self,
        agent_name: str,
        config: Optional[Dict[str, Any]] = None,
        agent_id: Optional[str] = None,
    ) -> SessionState:
        """
        Create a new agent session with isolated workspace.
        
        Args:
            agent_name: Human-readable agent name
            config: Initial configuration
            agent_id: Optional specific agent ID (generates UUID if not provided)
            
        Returns:
            New SessionState with initialized workspace
        """
        # Generate IDs
        agent_id = agent_id or str(uuid.uuid4())
        session_id = str(uuid.uuid4())
        
        # Ensure folder structure
        agent_folder = self._ensure_agent_structure(agent_id)
        
        # Create session state
        state = SessionState(
            session_id=session_id,
            agent_id=agent_id,
            agent_name=agent_name,
            generation=0,
            status=SessionStatus.INITIALIZING,
            workspace_path=self.workspace_path,
            agent_folder=agent_folder,
            checkpoint_thread_id=str(uuid.uuid4()),
            config=config or {},
        )
        
        # Save initial state
        self._save_session_state(state)
        self._save_agent_config(agent_id, {
            "name": agent_name,
            "created_at": state.created_at,
            "initial_config": config or {},
        })
        
        # Create initial generation record
        self._create_generation_record(state, agent_id)
        
        # Mark as active
        state.status = SessionStatus.ACTIVE
        self._active_sessions[session_id] = state
        self._save_session_state(state)
        
        return state
    
    def resume_session(self, session_id: str) -> Optional[SessionState]:
        """
        Resume an existing session.
        
        Args:
            session_id: Session ID to resume
            
        Returns:
            SessionState if found, None otherwise
        """
        # Check active sessions first
        if session_id in self._active_sessions:
            state = self._active_sessions[session_id]
            state.status = SessionStatus.ACTIVE
            state.update_activity()
            return state
        
        # Search for session file in agent folders
        for agent_id in self.list_agents():
            state = self._load_session_state(agent_id)
            if state and state.session_id == session_id:
                state.status = SessionStatus.ACTIVE
                state.update_activity()
                self._active_sessions[session_id] = state
                self._save_session_state(state)
                return state
        
        return None
    
    def resume_agent(self, agent_id: str) -> Optional[SessionState]:
        """
        Resume session for a specific agent.
        
        Args:
            agent_id: Agent ID
            
        Returns:
            SessionState if found, None otherwise
        """
        state = self._load_session_state(agent_id)
        if state:
            state.status = SessionStatus.ACTIVE
            state.update_activity()
            self._active_sessions[state.session_id] = state
            self._save_session_state(state)
        return state
    
    def get_session(self, session_id: str) -> Optional[SessionState]:
        """Get session state by ID."""
        return self._active_sessions.get(session_id)
    
    def pause_session(self, session_id: str) -> bool:
        """
        Pause a session.
        
        Returns:
            True if session was paused, False if not found
        """
        if session_id not in self._active_sessions:
            return False
        
        state = self._active_sessions[session_id]
        state.status = SessionStatus.PAUSED
        state.update_activity()
        self._save_session_state(state)
        
        return True
    
    def complete_session(self, session_id: str, success: bool = True) -> bool:
        """
        Mark session as completed.
        
        Args:
            session_id: Session to complete
            success: Whether session completed successfully
            
        Returns:
            True if session was completed
        """
        if session_id not in self._active_sessions:
            return False
        
        state = self._active_sessions[session_id]
        state.status = SessionStatus.COMPLETED if success else SessionStatus.FAILED
        state.update_activity()
        self._save_session_state(state)
        
        # Remove from active sessions
        del self._active_sessions[session_id]
        
        return True
    
    def evolve_agent(
        self,
        session_id: str,
        new_cmp_score: float,
        modifications: List[str],
        new_prompt_template: Optional[str] = None,
    ) -> Optional[SessionState]:
        """
        Evolve an agent to a new generation.
        
        Implements HGM self-improvement through prompt evolution.
        
        Args:
            session_id: Current session
            new_cmp_score: Updated CMP score
            modifications: List of modifications made
            new_prompt_template: Optional new prompt template
            
        Returns:
            Updated SessionState with new generation
        """
        if session_id not in self._active_sessions:
            return None
        
        state = self._active_sessions[session_id]
        state.status = SessionStatus.EVOLVING
        
        # Record current generation
        self._create_generation_record(
            state,
            state.agent_id,
            modifications=modifications,
        )
        
        # Update state
        state.generation += 1
        state.cmp_score = new_cmp_score
        state.cmp_history.append(new_cmp_score)
        
        # Update config with new prompt if provided
        if new_prompt_template:
            state.config["prompt_template"] = new_prompt_template
        
        state.config["modifications"] = state.config.get("modifications", []) + modifications
        
        # Save and return to active
        state.status = SessionStatus.ACTIVE
        state.update_activity()
        self._save_session_state(state)
        self._save_agent_config(state.agent_id, state.config)
        
        return state
    
    def update_cmp(self, session_id: str, cmp_score: float) -> None:
        """Update CMP score for a session."""
        if session_id in self._active_sessions:
            state = self._active_sessions[session_id]
            state.cmp_score = cmp_score
            state.cmp_history.append(cmp_score)
            state.update_activity()
            self._save_session_state(state)
    
    def record_task(self, session_id: str, success: bool, tools_used: Optional[List[str]] = None) -> None:
        """Record task execution for a session."""
        if session_id in self._active_sessions:
            state = self._active_sessions[session_id]
            state.record_task_result(success)
            if tools_used:
                state.tools_used.extend(tools_used)
            self._save_session_state(state)
    
    def _save_session_state(self, state: SessionState) -> None:
        """Save session state to file."""
        session_path = os.path.join(state.agent_folder, self.SESSION_FILE)
        with open(session_path, "w") as f:
            json.dump(state.to_dict(), f, indent=2)
    
    def _load_session_state(self, agent_id: str) -> Optional[SessionState]:
        """Load session state from file."""
        session_path = os.path.join(self._get_agent_folder(agent_id), self.SESSION_FILE)
        
        if not os.path.exists(session_path):
            return None
        
        try:
            with open(session_path, "r") as f:
                data = json.load(f)
            return SessionState.from_dict(data)
        except (json.JSONDecodeError, KeyError):
            return None
    
    def _save_agent_config(self, agent_id: str, config: Dict[str, Any]) -> None:
        """Save agent configuration."""
        config_path = os.path.join(self._get_agent_folder(agent_id), self.CONFIG_FILE)
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
    
    def load_agent_config(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Load agent configuration."""
        config_path = os.path.join(self._get_agent_folder(agent_id), self.CONFIG_FILE)
        
        if not os.path.exists(config_path):
            return None
        
        with open(config_path, "r") as f:
            return json.load(f)
    
    def _create_generation_record(
        self,
        state: SessionState,
        agent_id: str,
        modifications: Optional[List[str]] = None,
    ) -> str:
        """Create a generation record snapshot."""
        record = GenerationRecord(
            generation=state.generation,
            agent_id=agent_id,
            parent_id=state.config.get("parent_id"),
            cmp_score=state.cmp_score,
            created_at=datetime.utcnow().isoformat(),
            modifications=modifications or [],
            prompt_template=state.config.get("prompt_template", ""),
            metrics={
                "total_tasks": state.total_tasks,
                "success_rate": state.success_rate,
                "tools_used": list(set(state.tools_used)),
            },
        )
        
        # Save to generations folder
        gen_folder = os.path.join(state.agent_folder, "generations", f"gen_{state.generation}")
        Path(gen_folder).mkdir(parents=True, exist_ok=True)
        
        record_path = os.path.join(gen_folder, "record.json")
        with open(record_path, "w") as f:
            json.dump(record.to_dict(), f, indent=2)
        
        # Track in state
        state.generation_history.append(f"gen_{state.generation}")
        
        return f"gen_{state.generation}"
    
    def get_generation_history(self, agent_id: str) -> List[GenerationRecord]:
        """Get all generation records for an agent."""
        gen_folder = os.path.join(self._get_agent_folder(agent_id), "generations")
        
        if not os.path.exists(gen_folder):
            return []
        
        records = []
        for gen_dir in sorted(os.listdir(gen_folder)):
            record_path = os.path.join(gen_folder, gen_dir, "record.json")
            if os.path.exists(record_path):
                with open(record_path, "r") as f:
                    records.append(GenerationRecord.from_dict(json.load(f)))
        
        return records
    
    def list_agents(self) -> List[str]:
        """List all agent IDs in workspace."""
        if not os.path.exists(self.agents_root):
            return []
        
        return [
            name for name in os.listdir(self.agents_root)
            if os.path.isdir(os.path.join(self.agents_root, name))
        ]
    
    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all sessions (active and inactive)."""
        sessions = []
        
        # Active sessions
        for session_id, state in self._active_sessions.items():
            sessions.append({
                "session_id": session_id,
                "agent_id": state.agent_id,
                "agent_name": state.agent_name,
                "status": state.status.value,
                "generation": state.generation,
                "cmp_score": state.cmp_score,
                "last_active": state.last_active,
            })
        
        # Inactive sessions from disk
        for agent_id in self.list_agents():
            state = self._load_session_state(agent_id)
            if state and state.session_id not in self._active_sessions:
                sessions.append({
                    "session_id": state.session_id,
                    "agent_id": state.agent_id,
                    "agent_name": state.agent_name,
                    "status": state.status.value,
                    "generation": state.generation,
                    "cmp_score": state.cmp_score,
                    "last_active": state.last_active,
                })
        
        return sessions
    
    def get_agent_outputs_folder(self, agent_id: str) -> str:
        """Get outputs folder for an agent (for IDE access)."""
        return os.path.join(self._get_agent_folder(agent_id), "outputs")
    
    def get_agent_workspace_info(self, agent_id: str) -> Dict[str, str]:
        """
        Get workspace paths for an agent.
        
        Use this to open agent workspace in IDE.
        """
        agent_folder = self._get_agent_folder(agent_id)
        return {
            "root": agent_folder,
            "outputs": os.path.join(agent_folder, "outputs"),
            "documents": os.path.join(agent_folder, "documents"),
            "generations": os.path.join(agent_folder, "generations"),
            "logs": os.path.join(agent_folder, "logs"),
        }
    
    def export_agent(self, agent_id: str, export_path: str) -> str:
        """
        Export agent workspace to a zip file.
        
        Args:
            agent_id: Agent to export
            export_path: Destination path for zip file
            
        Returns:
            Path to created zip file
        """
        agent_folder = self._get_agent_folder(agent_id)
        
        if not os.path.exists(agent_folder):
            raise ValueError(f"Agent {agent_id} not found")
        
        # Create zip archive
        shutil.make_archive(
            export_path.replace(".zip", ""),
            "zip",
            agent_folder,
        )
        
        return export_path if export_path.endswith(".zip") else f"{export_path}.zip"


# Factory function
def create_session_manager(workspace_path: str) -> HGMSessionManager:
    """
    Create a session manager for the workspace.
    
    Args:
        workspace_path: Root workspace path
        
    Returns:
        Configured HGMSessionManager
    """
    return HGMSessionManager(workspace_path)
