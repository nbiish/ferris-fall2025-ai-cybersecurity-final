"""
Base middleware classes for DeepHGM architecture.

Implements the middleware pattern from LangChain DeepAgents with
extensions for HGM self-improvement capabilities.
"""

from typing import Any, Dict, List, Optional, Callable, TypedDict, Union
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import uuid
from datetime import datetime

from langchain_core.tools import BaseTool
from langchain_core.messages import BaseMessage


class MiddlewareHook(Enum):
    """Hooks available in the middleware lifecycle."""
    PRE_MODEL = "pre_model"
    POST_MODEL = "post_model"
    PRE_TOOL = "pre_tool"
    POST_TOOL = "post_tool"
    ON_INTERRUPT = "on_interrupt"
    ON_RESUME = "on_resume"
    ON_ERROR = "on_error"


@dataclass
class InterruptRequest:
    """
    Request for human-in-the-loop interrupt.
    
    Based on LangChain's interrupt pattern for DeepAgents.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    tool_name: str = ""
    tool_args: Dict[str, Any] = field(default_factory=dict)
    allowed_decisions: List[str] = field(default_factory=lambda: ["approve", "edit", "reject"])
    reason: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "tool_name": self.tool_name,
            "tool_args": self.tool_args,
            "allowed_decisions": self.allowed_decisions,
            "reason": self.reason,
            "timestamp": self.timestamp,
        }


@dataclass
class InterruptDecision:
    """
    Human decision in response to an interrupt.
    
    Types:
    - approve: Execute tool with original args
    - edit: Execute tool with modified args
    - reject: Skip tool execution
    """
    type: str  # "approve", "edit", "reject"
    edited_args: Optional[Dict[str, Any]] = None
    feedback: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "edited_args": self.edited_args,
            "feedback": self.feedback,
        }


class MiddlewareState(TypedDict, total=False):
    """State passed through middleware chain."""
    messages: List[BaseMessage]
    tools: List[BaseTool]
    pending_interrupts: List[InterruptRequest]
    decisions: List[InterruptDecision]
    context: Dict[str, Any]
    agent_id: str
    thread_id: str


class DeepHGMMiddleware(ABC):
    """
    Base middleware class for DeepHGM architecture.
    
    Implements the middleware pattern from LangChain DeepAgents:
    - Node-style hooks: Run sequentially at specific execution points
    - Wrap-style hooks: Run around each model or tool call
    - Tool injection: Add tools to the agent
    - Context injection: Modify messages/prompts
    
    Extended with HGM capabilities:
    - CMP tracking integration
    - Self-improvement hooks
    """
    
    def __init__(self, name: str, priority: int = 100):
        """
        Initialize middleware.
        
        Args:
            name: Unique middleware identifier
            priority: Execution order (lower = earlier)
        """
        self.name = name
        self.priority = priority
        self._tools: List[BaseTool] = []
        self._enabled = True
    
    @property
    def tools(self) -> List[BaseTool]:
        """Tools provided by this middleware."""
        return self._tools
    
    @property
    def enabled(self) -> bool:
        """Whether middleware is active."""
        return self._enabled
    
    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value
    
    def get_system_prompt_addition(self) -> Optional[str]:
        """
        Return text to append to system prompt.
        
        Override to inject middleware-specific instructions.
        """
        return None
    
    def pre_model(self, state: MiddlewareState) -> MiddlewareState:
        """
        Hook before model call.
        
        Use for:
        - Modifying messages
        - Injecting context
        - Validating state
        """
        return state
    
    def post_model(self, state: MiddlewareState, response: Any) -> MiddlewareState:
        """
        Hook after model call.
        
        Use for:
        - Processing model output
        - Logging
        - State updates
        """
        return state
    
    def pre_tool(self, state: MiddlewareState, tool_name: str, tool_args: Dict[str, Any]) -> tuple[MiddlewareState, Dict[str, Any]]:
        """
        Hook before tool execution.
        
        Use for:
        - Validating tool calls
        - Modifying arguments
        - Creating interrupts
        
        Returns:
            Updated state and potentially modified tool args
        """
        return state, tool_args
    
    def post_tool(self, state: MiddlewareState, tool_name: str, result: Any) -> MiddlewareState:
        """
        Hook after tool execution.
        
        Use for:
        - Processing tool results
        - Logging
        - CMP tracking
        """
        return state
    
    def on_interrupt(self, state: MiddlewareState, interrupt: InterruptRequest) -> MiddlewareState:
        """
        Hook when interrupt is triggered.
        
        Use for:
        - Preparing interrupt data
        - Logging pending actions
        """
        return state
    
    def on_resume(self, state: MiddlewareState, decision: InterruptDecision) -> MiddlewareState:
        """
        Hook when execution resumes after interrupt.
        
        Use for:
        - Processing human feedback
        - Updating state based on decision
        """
        return state
    
    def on_error(self, state: MiddlewareState, error: Exception) -> MiddlewareState:
        """
        Hook when error occurs.
        
        Use for:
        - Error recovery
        - Logging
        - CMP failure tracking
        """
        return state
    
    def wrap_model_call(
        self,
        state: MiddlewareState,
        model_fn: Callable[[MiddlewareState], Any],
    ) -> Any:
        """
        Wrap model call for full control.
        
        Default implementation calls pre_model, model_fn, post_model.
        Override for custom behavior like retries, fallbacks, etc.
        """
        state = self.pre_model(state)
        response = model_fn(state)
        state = self.post_model(state, response)
        return response
    
    def wrap_tool_call(
        self,
        state: MiddlewareState,
        tool_name: str,
        tool_args: Dict[str, Any],
        tool_fn: Callable[[str, Dict[str, Any]], Any],
    ) -> Any:
        """
        Wrap tool call for full control.
        
        Default implementation calls pre_tool, tool_fn, post_tool.
        Override for custom behavior like interrupts, retries, etc.
        """
        state, modified_args = self.pre_tool(state, tool_name, tool_args)
        result = tool_fn(tool_name, modified_args)
        state = self.post_tool(state, tool_name, result)
        return result


class MiddlewareChain:
    """
    Chain of middleware executed in priority order.
    
    Manages middleware lifecycle and provides unified interface
    for the DeepHGM agent.
    """
    
    def __init__(self, middlewares: Optional[List[DeepHGMMiddleware]] = None):
        self._middlewares: List[DeepHGMMiddleware] = []
        if middlewares:
            for m in middlewares:
                self.add(m)
    
    def add(self, middleware: DeepHGMMiddleware) -> None:
        """Add middleware to chain, maintaining priority order."""
        self._middlewares.append(middleware)
        self._middlewares.sort(key=lambda m: m.priority)
    
    def remove(self, name: str) -> bool:
        """Remove middleware by name."""
        for i, m in enumerate(self._middlewares):
            if m.name == name:
                self._middlewares.pop(i)
                return True
        return False
    
    def get(self, name: str) -> Optional[DeepHGMMiddleware]:
        """Get middleware by name."""
        for m in self._middlewares:
            if m.name == name:
                return m
        return None
    
    @property
    def all_tools(self) -> List[BaseTool]:
        """Collect tools from all enabled middleware."""
        tools = []
        for m in self._middlewares:
            if m.enabled:
                tools.extend(m.tools)
        return tools
    
    def get_system_prompt_additions(self) -> str:
        """Collect system prompt additions from all middleware."""
        additions = []
        for m in self._middlewares:
            if m.enabled:
                addition = m.get_system_prompt_addition()
                if addition:
                    additions.append(addition)
        return "\n\n".join(additions)
    
    def run_hook(self, hook: MiddlewareHook, state: MiddlewareState, **kwargs) -> MiddlewareState:
        """Run a hook through all enabled middleware."""
        for m in self._middlewares:
            if not m.enabled:
                continue
            
            if hook == MiddlewareHook.PRE_MODEL:
                state = m.pre_model(state)
            elif hook == MiddlewareHook.POST_MODEL:
                state = m.post_model(state, kwargs.get("response"))
            elif hook == MiddlewareHook.PRE_TOOL:
                state, kwargs["tool_args"] = m.pre_tool(
                    state, kwargs.get("tool_name", ""), kwargs.get("tool_args", {})
                )
            elif hook == MiddlewareHook.POST_TOOL:
                state = m.post_tool(state, kwargs.get("tool_name", ""), kwargs.get("result"))
            elif hook == MiddlewareHook.ON_INTERRUPT:
                state = m.on_interrupt(state, kwargs.get("interrupt"))
            elif hook == MiddlewareHook.ON_RESUME:
                state = m.on_resume(state, kwargs.get("decision"))
            elif hook == MiddlewareHook.ON_ERROR:
                state = m.on_error(state, kwargs.get("error"))
        
        return state
