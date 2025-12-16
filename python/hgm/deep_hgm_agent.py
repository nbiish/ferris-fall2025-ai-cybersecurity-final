"""
DeepHGM Agent: Chimera of LangChain DeepAgents + Huxley-Gödel Machine.

Purpose: State-of-the-art self-improving agent architecture combining:
- DeepAgents: Planning, filesystem backend, sub-agent delegation, middleware
- HGM: CMP scoring, UCB-Air branch selection, Thompson Sampling, self-evolution

Architecture:
┌─────────────────────────────────────────────────────────────────────┐
│                        DeepHGM Agent                                │
├─────────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                 │
│  │   Planner   │  │  Executor   │  │  Evaluator  │                 │
│  │  (DeepAgent)│  │  (Tools)    │  │  (HGM CMP)  │                 │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘                 │
│         │                │                │                         │
│  ┌──────▼────────────────▼────────────────▼──────┐                 │
│  │              State Manager                     │                 │
│  │  (LangGraph StateGraph + HGM Tree)            │                 │
│  └──────────────────────┬────────────────────────┘                 │
│                         │                                           │
│  ┌──────────────────────▼────────────────────────┐                 │
│  │           Self-Improvement Engine              │                 │
│  │  UCB-Air (branch) + Thompson (action)         │                 │
│  └───────────────────────────────────────────────┘                 │
└─────────────────────────────────────────────────────────────────────┘
"""

from typing import Any, AsyncIterator, Dict, List, Optional, TypedDict, Annotated, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import operator
import math
import random
import uuid
import json
from datetime import datetime

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, AIMessageChunk
from langchain_core.tools import BaseTool

from .agent import HGMAgent
from .middleware.base import (
    DeepHGMMiddleware as BaseMiddleware,
    MiddlewareChain,
    MiddlewareState,
    InterruptRequest,
    InterruptDecision,
)
from .middleware.hitl import HumanInTheLoopMiddleware, HITLConfig, DecisionType
from .middleware.mcp_tools import MCPToolMiddleware, MCPServerConfig
from .middleware.cli_tools import CLIToolMiddleware, CLIProvider, ExecutionMode
from .middleware.todo_list import TodoListMiddleware
from .middleware.filesystem import FilesystemMiddleware


class ActionType(Enum):
    """Types of actions the agent can take."""
    PLAN = "plan"
    EXECUTE = "execute"
    DELEGATE = "delegate"
    REFLECT = "reflect"
    EVOLVE = "evolve"


class DeepHGMState(TypedDict):
    """State for DeepHGM agent combining DeepAgents + HGM concepts."""
    messages: Annotated[List[BaseMessage], operator.add]
    agent_id: str
    generation: int
    cmp_score: float
    
    # DeepAgents features
    plan: List[Dict[str, Any]]
    current_step: int
    filesystem_context: Dict[str, Any]
    subagent_results: List[Dict[str, Any]]
    
    # HGM features
    clade_id: str
    branch_history: List[str]
    action_history: List[Dict[str, Any]]
    ucb_scores: Dict[str, float]
    thompson_params: Dict[str, Dict[str, float]]
    
    # Execution state
    tools_used: List[str]
    iteration: int
    workspace_path: Optional[str]
    last_action: Optional[ActionType]
    success_indicators: List[bool]


@dataclass
class CladeMetrics:
    """Clade Metaproductivity (CMP) metrics for agent lineage evaluation."""
    
    clade_id: str
    agent_ids: List[str] = field(default_factory=list)
    total_successes: int = 0
    total_failures: int = 0
    descendant_scores: List[float] = field(default_factory=list)
    
    def cmp_score(self) -> float:
        """
        Calculate Clade Metaproductivity score.
        
        CMP evaluates long-term productivity of descendants rather than
        short-term benchmark scores.
        """
        if not self.descendant_scores:
            # Prior: Beta(1, 1) = uniform
            return 0.5
        
        # Weighted average favoring recent descendants
        weights = [1.0 / (i + 1) for i in range(len(self.descendant_scores))]
        total_weight = sum(weights)
        weighted_sum = sum(s * w for s, w in zip(reversed(self.descendant_scores), weights))
        
        return weighted_sum / total_weight if total_weight > 0 else 0.5
    
    def add_descendant_result(self, success: bool, score: float) -> None:
        """Record a descendant's performance."""
        if success:
            self.total_successes += 1
        else:
            self.total_failures += 1
        self.descendant_scores.append(score)


@dataclass
class PlanStep:
    """A step in the agent's plan (DeepAgents pattern)."""
    
    id: str
    description: str
    status: str = "pending"  # pending, in_progress, completed, failed
    tool_required: Optional[str] = None
    subagent_required: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    result: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "status": self.status,
            "tool_required": self.tool_required,
            "subagent_required": self.subagent_required,
            "dependencies": self.dependencies,
            "result": self.result,
        }


class UCBAir:
    """
    UCB-Air for macro-decisions (branch switching).
    
    Decides whether to exploit current branch or explore new branches
    in the agent evolution tree.
    """
    
    def __init__(self, exploration_constant: float = 2.0):
        self.c = exploration_constant
        self.branch_visits: Dict[str, int] = {}
        self.branch_rewards: Dict[str, float] = {}
        self.total_visits: int = 0
    
    def ucb_score(self, branch_id: str) -> float:
        """Calculate UCB score for a branch."""
        if branch_id not in self.branch_visits or self.branch_visits[branch_id] == 0:
            return float('inf')  # Unexplored branches get priority
        
        n_i = self.branch_visits[branch_id]
        avg_reward = self.branch_rewards.get(branch_id, 0) / n_i
        exploration_bonus = self.c * math.sqrt(math.log(self.total_visits + 1) / n_i)
        
        return avg_reward + exploration_bonus
    
    def select_branch(self, available_branches: List[str]) -> str:
        """Select best branch using UCB-Air."""
        if not available_branches:
            return "root"
        
        scores = {b: self.ucb_score(b) for b in available_branches}
        return max(scores, key=scores.get)
    
    def update(self, branch_id: str, reward: float) -> None:
        """Update branch statistics after evaluation."""
        self.total_visits += 1
        self.branch_visits[branch_id] = self.branch_visits.get(branch_id, 0) + 1
        self.branch_rewards[branch_id] = self.branch_rewards.get(branch_id, 0) + reward


class ThompsonSampler:
    """
    Thompson Sampling for micro-decisions (action selection).
    
    Uses Bayesian approach over estimated CMP to select actions
    within a chosen branch.
    """
    
    def __init__(self):
        # Beta distribution parameters for each action type
        self.action_params: Dict[str, Dict[str, float]] = {
            action.value: {"alpha": 1.0, "beta": 1.0}
            for action in ActionType
        }
    
    def sample(self, action: ActionType) -> float:
        """Sample from Beta distribution for action."""
        params = self.action_params[action.value]
        return random.betavariate(params["alpha"], params["beta"])
    
    def select_action(self, available_actions: List[ActionType]) -> ActionType:
        """Select action using Thompson Sampling."""
        if not available_actions:
            return ActionType.EXECUTE
        
        samples = {a: self.sample(a) for a in available_actions}
        return max(samples, key=samples.get)
    
    def update(self, action: ActionType, success: bool) -> None:
        """Update Beta parameters after action result."""
        params = self.action_params[action.value]
        if success:
            params["alpha"] += 1
        else:
            params["beta"] += 1


class DeepHGMMiddleware:
    """
    Middleware pattern from DeepAgents for extensibility.
    
    Allows injection of tools, prompt modifications, and lifecycle hooks.
    """
    
    def __init__(self, name: str):
        self.name = name
        self.tools: List[BaseTool] = []
    
    def pre_execute(self, state: DeepHGMState) -> DeepHGMState:
        """Hook before action execution."""
        return state
    
    def post_execute(self, state: DeepHGMState, result: Any) -> DeepHGMState:
        """Hook after action execution."""
        return state
    
    def inject_context(self, messages: List[BaseMessage]) -> List[BaseMessage]:
        """Inject additional context into messages."""
        return messages


class PlanningMiddleware(DeepHGMMiddleware):
    """Middleware for planning capabilities (DeepAgents pattern)."""
    
    def __init__(self):
        super().__init__("planning")
    
    def create_plan(self, task: str, context: Dict[str, Any]) -> List[PlanStep]:
        """Create execution plan for a task."""
        # In production, this would use LLM to generate plan
        plan_id = str(uuid.uuid4())[:8]
        return [
            PlanStep(
                id=f"{plan_id}-analyze",
                description=f"Analyze task: {task[:50]}...",
                status="pending",
            ),
            PlanStep(
                id=f"{plan_id}-execute",
                description="Execute primary action",
                status="pending",
                dependencies=[f"{plan_id}-analyze"],
            ),
            PlanStep(
                id=f"{plan_id}-evaluate",
                description="Evaluate results and propose improvements",
                status="pending",
                dependencies=[f"{plan_id}-execute"],
            ),
        ]


class SelfImprovementMiddleware(DeepHGMMiddleware):
    """Middleware for HGM self-improvement capabilities."""
    
    def __init__(self):
        super().__init__("self_improvement")
        self.ucb = UCBAir()
        self.thompson = ThompsonSampler()
        self.clade_metrics: Dict[str, CladeMetrics] = {}
    
    def get_or_create_clade(self, clade_id: str) -> CladeMetrics:
        """Get or create clade metrics."""
        if clade_id not in self.clade_metrics:
            self.clade_metrics[clade_id] = CladeMetrics(clade_id=clade_id)
        return self.clade_metrics[clade_id]
    
    def select_branch(self, available_branches: List[str]) -> str:
        """Select branch using UCB-Air."""
        return self.ucb.select_branch(available_branches)
    
    def select_action(self, available_actions: List[ActionType]) -> ActionType:
        """Select action using Thompson Sampling."""
        return self.thompson.select_action(available_actions)
    
    def record_result(
        self,
        branch_id: str,
        action: ActionType,
        success: bool,
        cmp_score: float,
    ) -> None:
        """Record action result for learning."""
        reward = 1.0 if success else 0.0
        self.ucb.update(branch_id, reward)
        self.thompson.update(action, success)
        
        clade = self.get_or_create_clade(branch_id)
        clade.add_descendant_result(success, cmp_score)


class DeepHGMAgent:
    """
    DeepHGM Agent: Chimera of LangChain DeepAgents + Huxley-Gödel Machine.
    
    Key features:
    - Planning with sub-agent delegation (DeepAgents)
    - CMP-based self-evaluation (HGM)
    - UCB-Air branch selection (HGM)
    - Thompson Sampling action selection (HGM)
    - Middleware extensibility (DeepAgents)
    - Filesystem backend for persistence (DeepAgents)
    """
    
    SYSTEM_PROMPT = """You are a DeepHGM self-improving AI agent combining:
- DeepAgents: Planning, delegation, filesystem access
- HGM: Self-evolution through CMP optimization

Current state:
- Agent: {agent_name} (Gen {generation})
- CMP Score: {cmp_score:.4f}
- Branch: {branch_id}
- Iteration: {iteration}

Available tools: {tool_names}
Workspace: {workspace}

Execution rules:
1. Plan before executing complex tasks
2. Delegate to sub-agents when appropriate
3. Evaluate results honestly for CMP tracking
4. Propose self-improvements when patterns emerge
5. Be concise and action-oriented"""
    
    def __init__(
        self,
        agent: HGMAgent,
        tools: Optional[List[BaseTool]] = None,
        workspace_path: Optional[str] = None,
        middlewares: Optional[List[DeepHGMMiddleware]] = None,
    ):
        self.agent = agent
        self.tools = tools or []
        self.workspace_path = workspace_path
        self.middlewares = middlewares or [
            PlanningMiddleware(),
            SelfImprovementMiddleware(),
        ]
        
        self.clade_id = f"clade-{agent.id[:8]}"
        self.branch_history: List[str] = [self.clade_id]
        
        # Get self-improvement middleware
        self._si_middleware = next(
            (m for m in self.middlewares if isinstance(m, SelfImprovementMiddleware)),
            SelfImprovementMiddleware()
        )
        
        # Get planning middleware
        self._plan_middleware = next(
            (m for m in self.middlewares if isinstance(m, PlanningMiddleware)),
            PlanningMiddleware()
        )
    
    def get_system_message(self, state: DeepHGMState) -> SystemMessage:
        """Generate system message with current state context."""
        tool_names = ", ".join(t.name for t in self.tools) or "none"
        
        return SystemMessage(content=self.SYSTEM_PROMPT.format(
            agent_name=self.agent.name,
            generation=self.agent.generation,
            cmp_score=self.agent.cmp_score,
            branch_id=state.get("clade_id", self.clade_id),
            iteration=state.get("iteration", 0),
            tool_names=tool_names,
            workspace=self.workspace_path or "default",
        ))
    
    def create_initial_state(self, user_input: str) -> DeepHGMState:
        """Create initial state for agent execution."""
        return DeepHGMState(
            messages=[
                self.get_system_message({}),
                HumanMessage(content=user_input),
            ],
            agent_id=self.agent.id,
            generation=self.agent.generation,
            cmp_score=self.agent.cmp_score,
            
            # DeepAgents features
            plan=[],
            current_step=0,
            filesystem_context={},
            subagent_results=[],
            
            # HGM features
            clade_id=self.clade_id,
            branch_history=self.branch_history.copy(),
            action_history=[],
            ucb_scores={},
            thompson_params={},
            
            # Execution state
            tools_used=[],
            iteration=0,
            workspace_path=self.workspace_path,
            last_action=None,
            success_indicators=[],
        )
    
    def plan(self, state: DeepHGMState, task: str) -> DeepHGMState:
        """Create execution plan (DeepAgents pattern)."""
        plan_steps = self._plan_middleware.create_plan(task, state.get("filesystem_context", {}))
        
        new_state = dict(state)
        new_state["plan"] = [step.to_dict() for step in plan_steps]
        new_state["current_step"] = 0
        new_state["last_action"] = ActionType.PLAN
        new_state["action_history"] = state.get("action_history", []) + [{
            "action": ActionType.PLAN.value,
            "timestamp": datetime.now().isoformat(),
            "task": task[:100],
        }]
        
        return DeepHGMState(**new_state)
    
    def execute_tool(self, state: DeepHGMState, tool_name: str, **kwargs) -> DeepHGMState:
        """Execute a tool and update state."""
        tool = next((t for t in self.tools if t.name == tool_name), None)
        
        new_messages = list(state["messages"])
        tools_used = list(state.get("tools_used", []))
        success_indicators = list(state.get("success_indicators", []))
        
        if tool:
            try:
                result = tool._run(**kwargs)
                tools_used.append(tool_name)
                success_indicators.append(True)
                new_messages.append(AIMessage(
                    content=f"[Tool: {tool_name}] Executed successfully.\nResult: {str(result)[:500]}"
                ))
            except Exception as e:
                success_indicators.append(False)
                new_messages.append(AIMessage(
                    content=f"[Tool: {tool_name}] Failed: {str(e)}"
                ))
        else:
            success_indicators.append(False)
            new_messages.append(AIMessage(
                content=f"[Error] Tool '{tool_name}' not found."
            ))
        
        new_state = dict(state)
        new_state["messages"] = new_messages
        new_state["tools_used"] = tools_used
        new_state["success_indicators"] = success_indicators
        new_state["last_action"] = ActionType.EXECUTE
        
        return DeepHGMState(**new_state)
    
    def reflect(self, state: DeepHGMState) -> DeepHGMState:
        """Reflect on performance and update CMP (HGM pattern)."""
        success_indicators = state.get("success_indicators", [])
        
        # Calculate success rate
        if success_indicators:
            success_rate = sum(success_indicators) / len(success_indicators)
        else:
            success_rate = 0.5
        
        # Update agent's CMP score using exponential moving average
        alpha = 0.3  # Learning rate
        new_cmp = alpha * success_rate + (1 - alpha) * self.agent.cmp_score
        self.agent.cmp_score = new_cmp
        
        # Record in self-improvement middleware
        last_action = state.get("last_action", ActionType.EXECUTE)
        self._si_middleware.record_result(
            branch_id=state.get("clade_id", self.clade_id),
            action=last_action if isinstance(last_action, ActionType) else ActionType.EXECUTE,
            success=success_rate > 0.5,
            cmp_score=new_cmp,
        )
        
        # Generate reflection message
        reflection = f"""[Self-Reflection]
- Success Rate: {success_rate:.2%}
- Updated CMP: {new_cmp:.4f}
- Actions Taken: {len(state.get('action_history', []))}
- Tools Used: {', '.join(state.get('tools_used', [])) or 'none'}
- Improvement Potential: {'High' if success_rate < 0.7 else 'Moderate' if success_rate < 0.9 else 'Low'}"""
        
        new_messages = list(state["messages"])
        new_messages.append(AIMessage(content=reflection))
        
        new_state = dict(state)
        new_state["messages"] = new_messages
        new_state["cmp_score"] = new_cmp
        new_state["last_action"] = ActionType.REFLECT
        
        return DeepHGMState(**new_state)
    
    def evolve(self, state: DeepHGMState) -> Optional["DeepHGMAgent"]:
        """
        Create evolved child agent (HGM pattern).
        
        Returns new agent if evolution is warranted, None otherwise.
        """
        cmp_score = state.get("cmp_score", self.agent.cmp_score)
        
        # Only evolve if CMP indicates room for improvement
        if cmp_score > 0.9:
            return None  # Already performing well
        
        # Create child agent
        child_agent = HGMAgent.from_parent(self.agent)
        child_agent.add_modification(
            f"Evolved from {self.agent.name} at CMP={cmp_score:.4f}"
        )
        
        # Create new DeepHGM agent with inherited state
        child = DeepHGMAgent(
            agent=child_agent,
            tools=self.tools,
            workspace_path=self.workspace_path,
            middlewares=self.middlewares,
        )
        
        # Inherit branch history
        child.branch_history = self.branch_history + [child.clade_id]
        
        return child
    
    def step(self, state: DeepHGMState, user_input: str) -> DeepHGMState:
        """
        Execute one step of the DeepHGM agent.
        
        Uses Thompson Sampling to select action, then executes it.
        """
        # Apply pre-execute hooks
        for middleware in self.middlewares:
            state = middleware.pre_execute(state)
        
        # Select action using Thompson Sampling
        available_actions = [ActionType.PLAN, ActionType.EXECUTE, ActionType.REFLECT]
        
        # Add EVOLVE if CMP is low
        if state.get("cmp_score", 0.5) < 0.7:
            available_actions.append(ActionType.EVOLVE)
        
        selected_action = self._si_middleware.select_action(available_actions)
        
        # Execute selected action
        if selected_action == ActionType.PLAN:
            state = self.plan(state, user_input)
        elif selected_action == ActionType.EXECUTE:
            # Select tool based on input
            tool = self._select_tool(user_input)
            if tool:
                state = self.execute_tool(state, tool.name, input=user_input)
            else:
                # No tool needed, generate response
                new_messages = list(state["messages"])
                new_messages.append(AIMessage(
                    content=f"Processed request: {user_input[:200]}..."
                ))
                state = DeepHGMState(**{**dict(state), "messages": new_messages})
        elif selected_action == ActionType.REFLECT:
            state = self.reflect(state)
        
        # Update iteration
        new_state = dict(state)
        new_state["iteration"] = state.get("iteration", 0) + 1
        state = DeepHGMState(**new_state)
        
        # Apply post-execute hooks
        for middleware in self.middlewares:
            state = middleware.post_execute(state, None)
        
        return state
    
    def _select_tool(self, query: str) -> Optional[BaseTool]:
        """Select most relevant tool for query."""
        query_lower = query.lower()
        
        tool_keywords = {
            "ocr_extract": ["ocr", "extract", "document", "pdf", "image", "text", "scan"],
            "web_search": ["search", "find", "lookup", "web", "internet"],
            "file_read": ["read", "file", "open", "load"],
            "file_write": ["write", "save", "create", "output"],
        }
        
        for tool in self.tools:
            keywords = tool_keywords.get(tool.name, [])
            if any(kw in query_lower for kw in keywords):
                return tool
        
        return None
    
    def invoke(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main entry point for agent invocation.
        
        Compatible with LangChain agent interface.
        """
        messages = input_data.get("messages", [])
        user_input = ""
        
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_input = msg.get("content", "")
                break
        
        if not user_input and messages:
            user_input = messages[-1].get("content", "")
        
        # Create or update state
        state = self.create_initial_state(user_input)
        
        # Execute step
        state = self.step(state, user_input)
        
        # Reflect on results
        state = self.reflect(state)
        
        # Extract response
        response_messages = [
            msg.content for msg in state["messages"]
            if isinstance(msg, AIMessage)
        ]
        
        return {
            "response": "\n\n".join(response_messages[-3:]),  # Last 3 AI messages
            "state": {
                "agent_id": state["agent_id"],
                "generation": state["generation"],
                "cmp_score": state["cmp_score"],
                "iteration": state["iteration"],
                "tools_used": state["tools_used"],
                "plan": state["plan"],
            },
            "success": len(state.get("success_indicators", [])) > 0 and 
                      sum(state.get("success_indicators", [])) / len(state.get("success_indicators", [])) > 0.5,
        }


def create_deep_hgm_agent(
    agent: Optional[HGMAgent] = None,
    tools: Optional[List[BaseTool]] = None,
    workspace_path: Optional[str] = None,
    mcp_config_path: Optional[str] = None,
    interrupt_on: Optional[Dict[str, Any]] = None,
    enabled_cli_providers: Optional[List[str]] = None,
) -> "DeepHGMAgentV2":
    """
    Factory function to create DeepHGM agent with full middleware stack.
    
    Args:
        agent: HGM agent instance (creates initial if None)
        tools: List of additional tools to provide
        workspace_path: Agent workspace path
        mcp_config_path: Path to mcp.json for MCP server configuration
        interrupt_on: Dict mapping tool names to HITL configs
        enabled_cli_providers: List of CLI providers to enable (qwen, gemini)
        
    Returns:
        Configured DeepHGMAgentV2 with full middleware chain
    """
    if agent is None:
        agent = HGMAgent.initial()
    
    return DeepHGMAgentV2(
        agent=agent,
        tools=tools or [],
        workspace_path=workspace_path,
        mcp_config_path=mcp_config_path,
        interrupt_on=interrupt_on,
        enabled_cli_providers=enabled_cli_providers,
    )


class DeepHGMAgentV2:
    """
    DeepHGM Agent V2: Full middleware integration.
    
    Implements the complete LangChain DeepAgents middleware pattern:
    - HumanInTheLoopMiddleware: Interrupt/resume for human approval
    - MCPToolMiddleware: MCP server tool integration
    - CLIToolMiddleware: CLI tool (qwen, gemini) integration
    - TodoListMiddleware: Task planning and tracking
    - FilesystemMiddleware: File operations and context offloading
    
    Combined with HGM self-improvement:
    - CMP-based evaluation
    - UCB-Air branch selection
    - Thompson Sampling action selection
    """
    
    SYSTEM_PROMPT = """You are a DeepHGM self-improving AI agent.

## Architecture
- **DeepAgents**: Planning, delegation, filesystem access, human-in-the-loop
- **HGM**: Self-evolution through CMP optimization

## Current State
- Agent: {agent_name} (Gen {generation})
- CMP Score: {cmp_score:.4f}
- Branch: {branch_id}
- Iteration: {iteration}

## Available Tools
{tool_list}

{middleware_prompts}

## Execution Rules
1. Plan before executing complex tasks using write_todos
2. Request human approval for sensitive operations
3. Delegate to sub-agents when appropriate
4. Evaluate results honestly for CMP tracking
5. Use filesystem tools to manage large context
6. Be concise and action-oriented"""
    
    def __init__(
        self,
        agent: HGMAgent,
        tools: Optional[List[BaseTool]] = None,
        workspace_path: Optional[str] = None,
        mcp_config_path: Optional[str] = None,
        interrupt_on: Optional[Dict[str, Any]] = None,
        enabled_cli_providers: Optional[List[str]] = None,
    ):
        self.agent = agent
        self.workspace_path = workspace_path or "."
        self.clade_id = f"clade-{agent.id[:8]}"
        self.branch_history: List[str] = [self.clade_id]
        
        # Initialize middleware chain
        self.middleware_chain = MiddlewareChain()
        
        # Add HITL middleware (highest priority, reads config from mcp.json)
        self.hitl_middleware = HumanInTheLoopMiddleware(
            interrupt_on=interrupt_on or {},
            mcp_config_path=mcp_config_path,
        )
        self.middleware_chain.add(self.hitl_middleware)
        
        # Add TodoList middleware
        self.todo_middleware = TodoListMiddleware()
        self.middleware_chain.add(self.todo_middleware)
        
        # Add Filesystem middleware
        self.fs_middleware = FilesystemMiddleware(
            workspace_root=self.workspace_path,
        )
        self.middleware_chain.add(self.fs_middleware)
        
        # Add MCP middleware
        self.mcp_middleware = MCPToolMiddleware(
            mcp_config_path=mcp_config_path,
        )
        self.middleware_chain.add(self.mcp_middleware)
        
        # Add CLI middleware (reads config from mcp.json for frontend toggle sync)
        self.cli_middleware = CLIToolMiddleware(
            workspace_root=self.workspace_path,
            enabled_providers=enabled_cli_providers,
            mcp_config_path=mcp_config_path,
        )
        self.middleware_chain.add(self.cli_middleware)
        
        # Collect all tools
        self.tools = list(tools or [])
        self.tools.extend(self.middleware_chain.all_tools)
        
        # HGM components
        self.ucb = UCBAir()
        self.thompson = ThompsonSampler()
        self.clade_metrics: Dict[str, CladeMetrics] = {}
        
        # State tracking
        self._current_state: Optional[MiddlewareState] = None
        self._pending_interrupt: bool = False
    
    def get_system_prompt(self) -> str:
        """Generate system prompt with middleware additions."""
        tool_list = "\n".join(
            f"- **{t.name}**: {t.description[:80]}..."
            for t in self.tools
        ) or "No tools available"
        
        middleware_prompts = self.middleware_chain.get_system_prompt_additions()
        
        return self.SYSTEM_PROMPT.format(
            agent_name=self.agent.name,
            generation=self.agent.generation,
            cmp_score=self.agent.cmp_score,
            branch_id=self.clade_id,
            iteration=0,
            tool_list=tool_list,
            middleware_prompts=middleware_prompts,
        )
    
    def configure_hitl(self, tool_name: str, config: Any) -> None:
        """Configure human-in-the-loop for a tool."""
        if config is True:
            self.hitl_middleware.configure_tool(tool_name, enabled=True)
        elif config is False:
            self.hitl_middleware.configure_tool(tool_name, enabled=False)
        elif isinstance(config, dict):
            self.hitl_middleware.configure_tool(
                tool_name,
                enabled=True,
                allowed_decisions=config.get("allowed_decisions"),
            )
    
    def invoke(
        self,
        input_data: Dict[str, Any],
        thread_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Main entry point for agent invocation.
        
        Supports interrupt/resume pattern for human-in-the-loop.
        
        Args:
            input_data: Dict with messages and optional decisions for resume
            thread_id: Thread ID for state persistence
            
        Returns:
            Response dict, potentially with __interrupt__ for HITL
        """
        messages = input_data.get("messages", [])
        decisions = input_data.get("decisions", [])
        
        # Extract user input
        user_input = ""
        for msg in reversed(messages):
            if isinstance(msg, dict) and msg.get("role") == "user":
                user_input = msg.get("content", "")
                break
            elif hasattr(msg, "content"):
                user_input = msg.content
                break
        
        # Create or restore state
        state: MiddlewareState = {
            "messages": messages,
            "tools": self.tools,
            "pending_interrupts": [],
            "decisions": decisions,
            "context": {},
            "agent_id": self.agent.id,
            "thread_id": thread_id or str(uuid.uuid4()),
        }
        
        # Import MiddlewareHook locally to avoid circular import
        from .middleware.base import MiddlewareHook
        
        # Run pre-model hooks
        state = self.middleware_chain.run_hook(
            MiddlewareHook.PRE_MODEL, state
        )
        
        # Select action using Thompson Sampling
        available_actions = [ActionType.PLAN, ActionType.EXECUTE, ActionType.REFLECT]
        selected_action = self.thompson.select_action(available_actions)
        
        # Execute action
        tools_used = []
        success = True
        response_content = ""
        
        if selected_action == ActionType.PLAN:
            # Use todo middleware for planning
            todo_progress = self.todo_middleware.get_progress()
            response_content = f"[Planning Phase]\nCurrent progress: {todo_progress}"
            
        elif selected_action == ActionType.EXECUTE:
            # Select and execute tool
            tool = self._select_tool(user_input)
            if tool:
                # Check for HITL interrupt
                state, tool_args = self.hitl_middleware.pre_tool(
                    state, tool.name, {"input": user_input}
                )
                
                # Check if we need human approval
                if self.hitl_middleware.has_pending_interrupts(state):
                    self._pending_interrupt = True
                    self._current_state = state
                    
                    interrupt_response = self.hitl_middleware.get_interrupt_response(state)
                    return {
                        **interrupt_response,
                        "state": {
                            "agent_id": self.agent.id,
                            "cmp_score": self.agent.cmp_score,
                            "pending_approval": True,
                        },
                    }
                
                # Execute tool
                try:
                    if "__rejected__" in tool_args:
                        response_content = f"[Tool Rejected] {tool.name}: {tool_args.get('reason', 'User rejected')}"
                        success = False
                    else:
                        result = tool._run(**tool_args)
                        tools_used.append(tool.name)
                        response_content = f"[Tool: {tool.name}] {str(result)[:500]}"
                except Exception as e:
                    response_content = f"[Tool Error: {tool.name}] {str(e)}"
                    success = False
            else:
                response_content = f"Processed: {user_input[:200]}..."
        
        elif selected_action == ActionType.REFLECT:
            # Self-reflection with CMP update
            alpha = 0.3
            new_cmp = alpha * (1.0 if success else 0.0) + (1.0 - alpha) * self.agent.cmp_score
            self.agent.cmp_score = new_cmp
            
            response_content = f"""[Self-Reflection]
- CMP Score: {new_cmp:.4f}
- Action: {selected_action.value}
- Tools Used: {', '.join(tools_used) or 'none'}
- Branch: {self.clade_id}"""
        
        # Update Thompson Sampling
        self.thompson.update(selected_action, success)
        
        # Run post-model hooks
        state = self.middleware_chain.run_hook(
            MiddlewareHook.POST_MODEL, state, response=response_content
        )
        
        return {
            "response": response_content,
            "state": {
                "agent_id": self.agent.id,
                "generation": self.agent.generation,
                "cmp_score": self.agent.cmp_score,
                "iteration": 1,
                "tools_used": tools_used,
                "plan": [step.to_dict() for step in self.todo_middleware.get_todos()],
            },
            "success": success,
            "branch": self.clade_id,
            "clade_cmp": self.agent.cmp_score,
        }
    
    def resume(self, decisions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Resume execution after human-in-the-loop interrupt.
        
        Args:
            decisions: List of decisions from human review
            
        Returns:
            Response dict from continued execution
        """
        if not self._current_state:
            return {"error": "No pending interrupt to resume"}
        
        # Apply decisions
        state = self.hitl_middleware.apply_decisions(self._current_state, decisions)
        
        # Clear interrupt state
        self._pending_interrupt = False
        self._current_state = None
        
        # Continue execution
        return self.invoke({"messages": state.get("messages", []), "decisions": decisions})
    
    def _select_tool(self, query: str) -> Optional[BaseTool]:
        """Select most relevant tool for query using keyword matching."""
        query_lower = query.lower()
        
        # Tool keyword mappings
        keywords = {
            "write_todos": ["plan", "todo", "task", "steps", "breakdown"],
            "read_todos": ["progress", "status", "todos", "tasks"],
            "ls": ["list", "directory", "files", "folder"],
            "read_file": ["read", "open", "view", "content"],
            "write_file": ["write", "save", "create", "output"],
            "edit_file": ["edit", "modify", "change", "update"],
            "glob_search": ["find", "search", "pattern", "glob"],
            "grep_search": ["grep", "search", "regex", "find text"],
            "qwen": ["qwen", "summarize", "analyze", "quick"],
            "gemini": ["gemini", "design", "document", "complex", "code"],
            "orchestrate_agents": ["orchestrate", "multi-agent", "parallel"],
            "web_search": ["search", "web", "internet", "lookup"],
        }
        
        for tool in self.tools:
            tool_keywords = keywords.get(tool.name, [])
            if any(kw in query_lower for kw in tool_keywords):
                return tool
        
        return None
    
    @property
    def has_pending_interrupt(self) -> bool:
        """Check if there's a pending HITL interrupt."""
        return self._pending_interrupt
    
    async def astream(
        self,
        input_data: Dict[str, Any],
        thread_id: Optional[str] = None,
        provider=None,
        chat_model=None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Stream agent response token by token.
        
        Args:
            input_data: Dict with messages
            thread_id: Optional thread ID for state persistence
            provider: Optional OpenAICompatibleProvider for streaming
            chat_model: Optional LangChain chat model with streaming
            
        Yields:
            Dict with event type and content:
            - {"type": "token", "content": "..."}
            - {"type": "tool_start", "content": "tool_name", "metadata": {...}}
            - {"type": "tool_end", "content": "result"}
            - {"type": "metadata", "content": "", "metadata": {...}}
            - {"type": "error", "content": "error message"}
        """
        messages = input_data.get("messages", [])
        decisions = input_data.get("decisions", [])
        
        # Extract user input
        user_input = ""
        for msg in reversed(messages):
            if isinstance(msg, dict) and msg.get("role") == "user":
                user_input = msg.get("content", "")
                break
            elif hasattr(msg, "content"):
                user_input = msg.content
                break
        
        # Create state
        state: MiddlewareState = {
            "messages": messages,
            "tools": self.tools,
            "pending_interrupts": [],
            "decisions": decisions,
            "context": {},
            "agent_id": self.agent.id,
            "thread_id": thread_id or str(uuid.uuid4()),
        }
        
        # Run pre-model hooks
        from .middleware.base import MiddlewareHook
        state = self.middleware_chain.run_hook(MiddlewareHook.PRE_MODEL, state)
        
        # Select action
        available_actions = [ActionType.PLAN, ActionType.EXECUTE, ActionType.REFLECT]
        selected_action = self.thompson.select_action(available_actions)
        
        tools_used = []
        success = True
        
        if selected_action == ActionType.EXECUTE:
            tool = self._select_tool(user_input)
            if tool:
                # Yield tool start event
                yield {
                    "type": "tool_start",
                    "content": tool.name,
                    "metadata": {"input": user_input[:200]},
                }
                
                # Check for HITL interrupt
                state, tool_args = self.hitl_middleware.pre_tool(
                    state, tool.name, {"input": user_input}
                )
                
                if self.hitl_middleware.has_pending_interrupts(state):
                    yield {
                        "type": "interrupt",
                        "content": "Human approval required",
                        "metadata": self.hitl_middleware.get_interrupt_response(state),
                    }
                    return
                
                # Execute tool
                try:
                    if "__rejected__" in tool_args:
                        yield {"type": "error", "content": f"Tool rejected: {tool_args.get('reason', 'User rejected')}"}
                        success = False
                    else:
                        result = tool._run(**tool_args)
                        tools_used.append(tool.name)
                        
                        # Yield tool end event
                        yield {
                            "type": "tool_end",
                            "content": str(result)[:500],
                        }
                except Exception as e:
                    yield {"type": "error", "content": f"Tool error: {str(e)}"}
                    success = False
        
        # Stream LLM response if provider or chat_model available
        if provider:
            # Format messages for provider
            formatted_messages = self._format_messages_for_streaming(messages, user_input)
            
            try:
                async for chunk in provider.astream(
                    messages=formatted_messages,
                    temperature=0.7,
                ):
                    yield {"type": "token", "content": chunk}
            except Exception as e:
                yield {"type": "error", "content": str(e)}
                success = False
                
        elif chat_model:
            # Use LangChain chat model streaming
            lc_messages = self._convert_to_langchain_messages(messages)
            
            try:
                async for chunk in chat_model.astream(lc_messages):
                    if isinstance(chunk, AIMessageChunk) and chunk.content:
                        yield {"type": "token", "content": chunk.content}
                    elif hasattr(chunk, 'content') and chunk.content:
                        yield {"type": "token", "content": chunk.content}
            except Exception as e:
                yield {"type": "error", "content": str(e)}
                success = False
        else:
            # Fallback: generate response and simulate streaming
            response_content = self._generate_response(selected_action, user_input, tools_used, success)
            
            # Yield tokens character by character for simulation
            for char in response_content:
                yield {"type": "token", "content": char}
                await asyncio.sleep(0.005)  # Small delay for simulation
        
        # Update Thompson Sampling
        self.thompson.update(selected_action, success)
        
        # Yield final metadata
        yield {
            "type": "metadata",
            "content": "",
            "metadata": {
                "agent_id": self.agent.id,
                "generation": self.agent.generation,
                "cmp_score": self.agent.cmp_score,
                "tools_used": tools_used,
                "success": success,
                "branch": self.clade_id,
            },
        }
    
    async def astream_text(
        self,
        input_data: Dict[str, Any],
        thread_id: Optional[str] = None,
        provider=None,
        chat_model=None,
    ) -> AsyncIterator[str]:
        """
        Stream just the text content (tokens only).
        
        Args:
            input_data: Dict with messages
            thread_id: Optional thread ID
            provider: Optional OpenAICompatibleProvider
            chat_model: Optional LangChain chat model
            
        Yields:
            Token strings
        """
        async for event in self.astream(input_data, thread_id, provider, chat_model):
            if event.get("type") == "token":
                yield event.get("content", "")
    
    def _format_messages_for_streaming(
        self,
        messages: List[Union[Dict, BaseMessage]],
        user_input: str,
    ) -> List[Dict[str, str]]:
        """Format messages for OpenAI-compatible provider streaming."""
        formatted = []
        
        # Add system prompt
        formatted.append({
            "role": "system",
            "content": self.get_system_prompt(),
        })
        
        # Add conversation history
        for msg in messages:
            if isinstance(msg, dict):
                formatted.append({
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", ""),
                })
            elif isinstance(msg, BaseMessage):
                role = "assistant" if isinstance(msg, AIMessage) else "user"
                if hasattr(msg, '__class__') and 'System' in msg.__class__.__name__:
                    role = "system"
                formatted.append({
                    "role": role,
                    "content": msg.content,
                })
        
        return formatted
    
    def _convert_to_langchain_messages(
        self,
        messages: List[Union[Dict, BaseMessage]],
    ) -> List[BaseMessage]:
        """Convert messages to LangChain format."""
        lc_messages = [SystemMessage(content=self.get_system_prompt())]
        
        for msg in messages:
            if isinstance(msg, dict):
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "system":
                    lc_messages.append(SystemMessage(content=content))
                elif role == "assistant":
                    lc_messages.append(AIMessage(content=content))
                else:
                    lc_messages.append(HumanMessage(content=content))
            elif isinstance(msg, BaseMessage):
                lc_messages.append(msg)
        
        return lc_messages
    
    def _generate_response(
        self,
        action: ActionType,
        user_input: str,
        tools_used: List[str],
        success: bool,
    ) -> str:
        """Generate response content for fallback streaming."""
        if action == ActionType.PLAN:
            todo_progress = self.todo_middleware.get_progress()
            return f"[Planning Phase]\nCurrent progress: {todo_progress}"
        elif action == ActionType.REFLECT:
            return f"""[Self-Reflection]
- CMP Score: {self.agent.cmp_score:.4f}
- Action: {action.value}
- Tools Used: {', '.join(tools_used) or 'none'}
- Branch: {self.clade_id}"""
        else:
            if tools_used:
                return f"[Executed tools: {', '.join(tools_used)}]\nProcessed: {user_input[:200]}..."
            return f"Processed: {user_input[:200]}..."
