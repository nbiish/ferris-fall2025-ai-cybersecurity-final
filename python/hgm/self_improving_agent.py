"""
Self-Improving HGM Agent with One-Prompt Activation.

Implements the complete Huxley-Gödel Machine architecture from metauto-ai/HGM
with MemoriLabs/Memori integration for efficient agent memory.

One prompt activates a fully configured self-improving agent with:
- Isolated session workspace
- Memori memory system
- CMP-based self-evaluation
- UCB-Air branch selection
- Thompson Sampling action selection
- Automatic evolution tracking

Reference:
- https://github.com/metauto-ai/HGM (Self-improving architecture)
- https://github.com/MemoriLabs/Memori (Efficient agent memory)
"""

from typing import Any, AsyncIterator, Dict, List, Optional, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime
import uuid
import json
import os
import asyncio

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import BaseTool

from .agent import HGMAgent
from .deep_hgm_agent import (
    DeepHGMAgent,
    DeepHGMState,
    UCBAir,
    ThompsonSampler,
    CladeMetrics,
    ActionType,
    PlanningMiddleware,
    SelfImprovementMiddleware,
)
from .session_manager import (
    HGMSessionManager,
    SessionState,
    SessionStatus,
    create_session_manager,
)
from .middleware.memori import (
    MemoriMiddleware,
    MemoriConfig,
    create_memori_middleware,
)
from .middleware.base import MiddlewareChain, DeepHGMMiddleware
from .persistence import HGMCheckpointer, AgentFolderManager


@dataclass
class AgentConfig:
    """Configuration for a self-improving HGM agent."""
    
    name: str
    workspace_path: str
    
    # LLM configuration
    model: str = "gpt-4"
    temperature: float = 0.7
    max_tokens: int = 4096
    
    # Self-improvement settings
    enable_evolution: bool = True
    evolution_threshold: float = 0.7  # CMP threshold for evolution
    max_generations: int = 10
    
    # Memory settings
    enable_memori: bool = True
    memori_max_context: int = 10
    
    # Middleware settings
    enable_planning: bool = True
    enable_human_in_loop: bool = False
    
    # Tools
    tools: List[BaseTool] = field(default_factory=list)
    
    # Custom prompt template
    prompt_template: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "workspace_path": self.workspace_path,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "enable_evolution": self.enable_evolution,
            "evolution_threshold": self.evolution_threshold,
            "max_generations": self.max_generations,
            "enable_memori": self.enable_memori,
            "memori_max_context": self.memori_max_context,
            "enable_planning": self.enable_planning,
            "enable_human_in_loop": self.enable_human_in_loop,
            "prompt_template": self.prompt_template,
        }


class SelfImprovingHGMAgent:
    """
    Self-Improving HGM Agent with full Huxley-Gödel Machine architecture.
    
    Features:
    - **One-prompt activation**: Single entry point activates complete agent
    - **Session isolation**: Each agent runs in dedicated workspace
    - **Memori integration**: Efficient SQL-native memory layer
    - **Self-improvement**: CMP-based evolution through generations
    - **UCB-Air**: Macro-decision branch selection
    - **Thompson Sampling**: Micro-decision action selection
    
    Architecture from metauto-ai/HGM:
    1. Agents iteratively rewrite themselves
    2. CMP (Clade Metaproductivity) estimates promise of subtrees
    3. UCB-Air decides which self-modifications to expand
    4. Thompson Sampling selects actions within branches
    """
    
    SYSTEM_PROMPT = """You are a self-improving AI agent implementing the Huxley-Gödel Machine architecture.

## Agent Identity
- **Name**: {agent_name}
- **Generation**: {generation}
- **Session**: {session_id}
- **CMP Score**: {cmp_score:.4f}

## Capabilities
- Self-evaluation and improvement through CMP optimization
- Memory persistence across sessions (Memori)
- Planning and task decomposition
- Tool usage when appropriate

## Available Tools
{tool_names}

## Workspace
{workspace_path}

## Execution Rules
1. **Plan first**: Decompose complex tasks into steps
2. **Execute efficiently**: Use tools only when necessary
3. **Evaluate honestly**: Report success/failure for CMP tracking
4. **Learn continuously**: Store important learnings in memory
5. **Propose improvements**: When patterns emerge, suggest self-modifications

## Memory Context
{memory_context}

## Current Task
Execute the user's request while optimizing for long-term productivity (CMP).
Be concise, action-oriented, and honest about results."""

    def __init__(
        self,
        config: AgentConfig,
        session_manager: Optional[HGMSessionManager] = None,
    ):
        """
        Initialize self-improving agent.
        
        Args:
            config: Agent configuration
            session_manager: Optional session manager (creates new if not provided)
        """
        self.config = config
        self.session_manager = session_manager or create_session_manager(config.workspace_path)
        
        # Core components (initialized on activation)
        self._session: Optional[SessionState] = None
        self._hgm_agent: Optional[HGMAgent] = None
        self._deep_agent: Optional[DeepHGMAgent] = None
        self._memori: Optional[MemoriMiddleware] = None
        self._checkpointer: Optional[HGMCheckpointer] = None
        
        # Self-improvement components
        self._ucb = UCBAir(exploration_constant=2.0)
        self._thompson = ThompsonSampler()
        self._clades: Dict[str, CladeMetrics] = {}
        
        # Middleware chain
        self._middleware_chain: Optional[MiddlewareChain] = None
        
        # State
        self._activated = False
        self._current_branch: str = "root"
    
    @property
    def session_id(self) -> Optional[str]:
        """Current session ID."""
        return self._session.session_id if self._session else None
    
    @property
    def agent_id(self) -> Optional[str]:
        """Current agent ID."""
        return self._session.agent_id if self._session else None
    
    @property
    def generation(self) -> int:
        """Current generation."""
        return self._session.generation if self._session else 0
    
    @property
    def cmp_score(self) -> float:
        """Current CMP score."""
        return self._session.cmp_score if self._session else 0.5
    
    @property
    def workspace_path(self) -> Optional[str]:
        """Agent workspace path."""
        return self._session.agent_folder if self._session else None
    
    def activate(
        self,
        agent_id: Optional[str] = None,
        resume_session: Optional[str] = None,
    ) -> "SelfImprovingHGMAgent":
        """
        Activate the agent with one call.
        
        This is the main entry point that sets up:
        - Session with isolated workspace
        - Memori memory system
        - Self-improvement middleware
        - Planning capabilities
        
        Args:
            agent_id: Specific agent ID (generates new if not provided)
            resume_session: Session ID to resume (creates new if not provided)
            
        Returns:
            Self for method chaining
        """
        # Resume existing session or create new
        if resume_session:
            self._session = self.session_manager.resume_session(resume_session)
            if not self._session:
                raise ValueError(f"Session {resume_session} not found")
        elif agent_id:
            self._session = self.session_manager.resume_agent(agent_id)
            if not self._session:
                # Create new session for existing agent
                self._session = self.session_manager.create_session(
                    agent_name=self.config.name,
                    config=self.config.to_dict(),
                    agent_id=agent_id,
                )
        else:
            # Create completely new agent and session
            self._session = self.session_manager.create_session(
                agent_name=self.config.name,
                config=self.config.to_dict(),
            )
        
        # Initialize HGM agent
        self._hgm_agent = HGMAgent(
            id=self._session.agent_id,
            name=self._session.agent_name,
            generation=self._session.generation,
            cmp_score=self._session.cmp_score,
            prompt_template=self.config.prompt_template or "",
        )
        
        # Initialize Memori if enabled
        if self.config.enable_memori:
            self._memori = create_memori_middleware(
                agent_id=self._session.agent_id,
                workspace_path=self.config.workspace_path,
                process_id=f"hgm-{self._session.agent_name}",
            )
            self._session.memori_session_id = self._memori.session_id
        
        # Initialize checkpointer
        folder_manager = AgentFolderManager(self.config.workspace_path)
        self._checkpointer = folder_manager.get_checkpointer(self._session.agent_id)
        
        # Build middleware chain
        middlewares: List[DeepHGMMiddleware] = []
        
        if self.config.enable_planning:
            middlewares.append(PlanningMiddleware())
        
        middlewares.append(SelfImprovementMiddleware())
        
        if self._memori:
            middlewares.append(self._memori)
        
        self._middleware_chain = MiddlewareChain(middlewares)
        
        # Create DeepHGM agent
        self._deep_agent = DeepHGMAgent(
            agent=self._hgm_agent,
            tools=self.config.tools,
            workspace_path=self._session.agent_folder,
            middlewares=middlewares,
        )
        
        # Initialize branch
        self._current_branch = f"clade-{self._session.agent_id[:8]}"
        self._clades[self._current_branch] = CladeMetrics(clade_id=self._current_branch)
        
        self._activated = True
        
        # Record activation in memory
        if self._memori:
            self._memori.add_event(
                f"Agent activated: {self._session.agent_name} (Gen {self._session.generation})",
                metadata={"session_id": self._session.session_id},
            )
        
        return self
    
    def _ensure_activated(self) -> None:
        """Ensure agent is activated before operations."""
        if not self._activated:
            raise RuntimeError("Agent not activated. Call activate() first.")
    
    def _get_system_message(self) -> SystemMessage:
        """Generate system message with current state."""
        self._ensure_activated()
        
        tool_names = ", ".join(t.name for t in self.config.tools) if self.config.tools else "none"
        
        # Get memory context from Memori
        memory_context = "No memory context available."
        if self._memori:
            prompt_addition = self._memori.get_system_prompt_addition()
            if prompt_addition:
                memory_context = prompt_addition
        
        return SystemMessage(content=self.SYSTEM_PROMPT.format(
            agent_name=self._session.agent_name,
            generation=self._session.generation,
            session_id=self._session.session_id,
            cmp_score=self._session.cmp_score,
            tool_names=tool_names,
            workspace_path=self._session.agent_folder,
            memory_context=memory_context,
        ))
    
    def invoke(self, prompt: str) -> Dict[str, Any]:
        """
        Execute a task with the self-improving agent.
        
        This is the main execution method:
        1. Processes user prompt
        2. Uses UCB-Air for branch selection
        3. Uses Thompson Sampling for action selection
        4. Evaluates results and updates CMP
        5. Triggers evolution if warranted
        
        Args:
            prompt: User's task/request
            
        Returns:
            Response dict with result, state, and metrics
        """
        self._ensure_activated()
        
        # Select branch using UCB-Air
        available_branches = list(self._clades.keys())
        self._current_branch = self._ucb.select_branch(available_branches)
        
        # Create input for deep agent
        messages = [
            self._get_system_message(),
            HumanMessage(content=prompt),
        ]
        
        # Invoke deep agent
        result = self._deep_agent.invoke({
            "messages": [{"role": "user", "content": prompt}],
        })
        
        # Evaluate success
        success = result.get("success", False)
        response_text = result.get("response", "")
        
        # Update CMP
        new_cmp = self._calculate_cmp(success)
        self._session.cmp_score = new_cmp
        self._hgm_agent.cmp_score = new_cmp
        
        # Update branch metrics
        self._ucb.update(self._current_branch, 1.0 if success else 0.0)
        
        if self._current_branch in self._clades:
            self._clades[self._current_branch].add_descendant_result(success, new_cmp)
        
        # Record task
        tools_used = result.get("state", {}).get("tools_used", [])
        self.session_manager.record_task(self._session.session_id, success, tools_used)
        
        # Store in memory
        if self._memori:
            self._memori.add_fact(
                f"Task: {prompt[:100]}... Result: {'success' if success else 'failed'}",
                importance=0.6 if success else 0.4,
            )
        
        # Save checkpoint
        self._save_checkpoint(messages + [AIMessage(content=response_text)])
        
        # Check for evolution
        evolution_triggered = False
        if self.config.enable_evolution and self._should_evolve():
            self._evolve()
            evolution_triggered = True
        
        return {
            "response": response_text,
            "success": success,
            "session_id": self._session.session_id,
            "agent_id": self._session.agent_id,
            "generation": self._session.generation,
            "cmp_score": new_cmp,
            "branch": self._current_branch,
            "tools_used": tools_used,
            "evolution_triggered": evolution_triggered,
            "workspace": self._session.agent_folder,
        }
    
    async def astream(
        self,
        prompt: str,
        provider=None,
        chat_model=None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Stream response from the self-improving agent.
        
        Args:
            prompt: User's task/request
            provider: Optional OpenAI-compatible provider
            chat_model: Optional LangChain chat model
            
        Yields:
            Dict with event type and content
        """
        self._ensure_activated()
        
        # Select branch
        available_branches = list(self._clades.keys())
        self._current_branch = self._ucb.select_branch(available_branches)
        
        yield {
            "type": "session_info",
            "content": "",
            "metadata": {
                "session_id": self._session.session_id,
                "agent_id": self._session.agent_id,
                "generation": self._session.generation,
                "branch": self._current_branch,
            },
        }
        
        # Stream from deep agent
        full_response = ""
        tools_used = []
        success = True
        
        async for event in self._deep_agent.astream(
            {"user_input": prompt},
            provider=provider,
            chat_model=chat_model,
        ):
            yield event
            
            if event.get("type") == "token":
                full_response += event.get("content", "")
            elif event.get("type") == "tool_end":
                tools_used.append(event.get("content", ""))
            elif event.get("type") == "error":
                success = False
        
        # Update CMP and metrics
        new_cmp = self._calculate_cmp(success)
        self._session.cmp_score = new_cmp
        self._hgm_agent.cmp_score = new_cmp
        
        self._ucb.update(self._current_branch, 1.0 if success else 0.0)
        self.session_manager.record_task(self._session.session_id, success, tools_used)
        
        # Store in memory
        if self._memori:
            self._memori.add_fact(
                f"Streamed task: {prompt[:100]}... Result: {'success' if success else 'failed'}",
                importance=0.5,
            )
        
        # Check evolution
        evolution_triggered = False
        if self.config.enable_evolution and self._should_evolve():
            self._evolve()
            evolution_triggered = True
        
        yield {
            "type": "complete",
            "content": "",
            "metadata": {
                "success": success,
                "cmp_score": new_cmp,
                "tools_used": tools_used,
                "evolution_triggered": evolution_triggered,
            },
        }
    
    async def astream_text(
        self,
        prompt: str,
        provider=None,
        chat_model=None,
    ) -> AsyncIterator[str]:
        """Stream just text tokens."""
        async for event in self.astream(prompt, provider, chat_model):
            if event.get("type") == "token":
                yield event.get("content", "")
    
    def _calculate_cmp(self, success: bool) -> float:
        """Calculate updated CMP score."""
        current = self._session.cmp_score
        alpha = 0.3  # Learning rate
        
        reward = 1.0 if success else 0.0
        new_cmp = alpha * reward + (1 - alpha) * current
        
        return max(0.0, min(1.0, new_cmp))
    
    def _should_evolve(self) -> bool:
        """Determine if agent should evolve."""
        if self._session.generation >= self.config.max_generations:
            return False
        
        # Evolve if CMP is below threshold (room for improvement)
        if self._session.cmp_score < self.config.evolution_threshold:
            # Only evolve if we have enough data
            if self._session.total_tasks >= 3:
                return True
        
        return False
    
    def _evolve(self) -> None:
        """Trigger agent evolution."""
        modifications = [
            f"Evolved at CMP={self._session.cmp_score:.4f}",
            f"After {self._session.total_tasks} tasks ({self._session.success_rate:.1%} success)",
        ]
        
        # Record in memory before evolution
        if self._memori:
            self._memori.add_event(
                f"Evolution triggered: Gen {self._session.generation} -> {self._session.generation + 1}",
                metadata={
                    "cmp_score": self._session.cmp_score,
                    "success_rate": self._session.success_rate,
                },
            )
        
        # Evolve through session manager
        self.session_manager.evolve_agent(
            self._session.session_id,
            new_cmp_score=self._session.cmp_score,
            modifications=modifications,
        )
        
        # Update local state
        self._hgm_agent.generation = self._session.generation
        self._hgm_agent.add_modification(modifications[0])
    
    def _save_checkpoint(self, messages: List[BaseMessage]) -> None:
        """Save conversation checkpoint."""
        if self._checkpointer:
            self._checkpointer.put(
                thread_id=self._session.checkpoint_thread_id,
                agent_id=self._session.agent_id,
                messages=messages,
                cmp_score=self._session.cmp_score,
                cmp_history=self._session.cmp_history,
                generation=self._session.generation,
                tools_used=self._session.tools_used,
                iteration=self._session.total_tasks,
            )
    
    def recall_memories(self, query: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Recall memories from Memori."""
        self._ensure_activated()
        
        if not self._memori:
            return []
        
        memories = self._memori.recall(query=query, limit=limit)
        return [m.to_dict() for m in memories]
    
    def add_memory(self, content: str, memory_type: str = "facts") -> Optional[str]:
        """Add a memory entry."""
        self._ensure_activated()
        
        if not self._memori:
            return None
        
        return self._memori.add_memory(content, memory_type)
    
    def get_session_info(self) -> Dict[str, Any]:
        """Get current session information."""
        self._ensure_activated()
        
        return {
            "session_id": self._session.session_id,
            "agent_id": self._session.agent_id,
            "agent_name": self._session.agent_name,
            "generation": self._session.generation,
            "cmp_score": self._session.cmp_score,
            "status": self._session.status.value,
            "total_tasks": self._session.total_tasks,
            "success_rate": self._session.success_rate,
            "workspace": self._session.agent_folder,
            "memori_session": self._session.memori_session_id,
        }
    
    def get_workspace_paths(self) -> Dict[str, str]:
        """Get workspace paths for IDE access."""
        self._ensure_activated()
        
        return self.session_manager.get_agent_workspace_info(self._session.agent_id)
    
    def deactivate(self, complete: bool = False) -> None:
        """
        Deactivate the agent.
        
        Args:
            complete: If True, marks session as completed
        """
        if not self._activated:
            return
        
        if self._memori:
            self._memori.add_event(
                f"Agent deactivated: {self._session.agent_name}",
                metadata={"completed": complete},
            )
        
        if complete:
            self.session_manager.complete_session(self._session.session_id)
        else:
            self.session_manager.pause_session(self._session.session_id)
        
        self._activated = False


# One-prompt activation factory
def create_self_improving_agent(
    name: str,
    workspace_path: str,
    model: str = "gpt-4",
    tools: Optional[List[BaseTool]] = None,
    enable_memori: bool = True,
    enable_evolution: bool = True,
    **kwargs,
) -> SelfImprovingHGMAgent:
    """
    Create and activate a self-improving HGM agent with one call.
    
    This is the main entry point for the HGM + Memori integration.
    
    Args:
        name: Agent name
        workspace_path: Root workspace path
        model: LLM model to use
        tools: Optional list of tools
        enable_memori: Enable Memori memory system
        enable_evolution: Enable self-improvement evolution
        **kwargs: Additional AgentConfig parameters
        
    Returns:
        Activated SelfImprovingHGMAgent ready for use
        
    Example:
        ```python
        agent = create_self_improving_agent(
            name="CodeReviewer",
            workspace_path="/path/to/workspace",
            model="gpt-4",
            enable_memori=True,
        )
        
        # Single prompt activates and runs
        result = agent.invoke("Review this code for security issues...")
        
        # Access workspace in IDE
        print(agent.get_workspace_paths())
        ```
    """
    config = AgentConfig(
        name=name,
        workspace_path=workspace_path,
        model=model,
        tools=tools or [],
        enable_memori=enable_memori,
        enable_evolution=enable_evolution,
        **kwargs,
    )
    
    agent = SelfImprovingHGMAgent(config)
    agent.activate()
    
    return agent


def resume_agent(
    workspace_path: str,
    session_id: Optional[str] = None,
    agent_id: Optional[str] = None,
) -> Optional[SelfImprovingHGMAgent]:
    """
    Resume an existing agent session.
    
    Args:
        workspace_path: Root workspace path
        session_id: Session ID to resume
        agent_id: Agent ID to resume (uses latest session)
        
    Returns:
        Resumed agent or None if not found
    """
    session_manager = create_session_manager(workspace_path)
    
    # Find session
    if session_id:
        session = session_manager.resume_session(session_id)
    elif agent_id:
        session = session_manager.resume_agent(agent_id)
    else:
        return None
    
    if not session:
        return None
    
    # Load config
    config_data = session_manager.load_agent_config(session.agent_id)
    
    config = AgentConfig(
        name=session.agent_name,
        workspace_path=workspace_path,
        **(config_data.get("initial_config", {}) if config_data else {}),
    )
    
    agent = SelfImprovingHGMAgent(config, session_manager=session_manager)
    agent.activate(agent_id=session.agent_id, resume_session=session.session_id)
    
    return agent


def list_agents(workspace_path: str) -> List[Dict[str, Any]]:
    """
    List all agents in a workspace.
    
    Args:
        workspace_path: Root workspace path
        
    Returns:
        List of agent info dictionaries
    """
    session_manager = create_session_manager(workspace_path)
    return session_manager.list_sessions()
