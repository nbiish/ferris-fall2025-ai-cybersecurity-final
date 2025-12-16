"""
Human-in-the-Loop Middleware for DeepHGM.

Implements LangChain's interrupt/resume pattern for human approval
of sensitive tool operations via the Agent Monitor prompt box.

Follows LangChain DeepAgents middleware pattern:
- Lifecycle hooks: `pre_tool`, `on_interrupt`, `on_resume`
- `interrupt_on` configuration for tool-specific approval requirements
- Returns `__interrupt__` format for frontend integration
"""

from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import uuid
import json
import os

from langchain_core.tools import BaseTool

from .base import (
    DeepHGMMiddleware,
    MiddlewareState,
    InterruptRequest,
    InterruptDecision,
)


class DecisionType(Enum):
    """Types of decisions a human can make."""
    APPROVE = "approve"
    EDIT = "edit"
    REJECT = "reject"


@dataclass
class HITLConfig:
    """
    Configuration for a tool's human-in-the-loop behavior.
    
    Based on LangChain's interrupt_on configuration pattern.
    """
    tool_name: str
    enabled: bool = True
    allowed_decisions: List[str] = field(
        default_factory=lambda: ["approve", "edit", "reject"]
    )
    require_reason: bool = False
    auto_approve_after_seconds: Optional[int] = None
    
    @classmethod
    def from_dict(cls, tool_name: str, config: Any) -> "HITLConfig":
        """Create config from interrupt_on dict value."""
        if config is True:
            return cls(tool_name=tool_name, enabled=True)
        elif config is False:
            return cls(tool_name=tool_name, enabled=False)
        elif isinstance(config, dict):
            return cls(
                tool_name=tool_name,
                enabled=True,
                allowed_decisions=config.get("allowed_decisions", ["approve", "edit", "reject"]),
                require_reason=config.get("require_reason", False),
                auto_approve_after_seconds=config.get("auto_approve_after_seconds"),
            )
        return cls(tool_name=tool_name, enabled=False)


class HumanInTheLoopMiddleware(DeepHGMMiddleware):
    """
    Middleware for human-in-the-loop approval workflows.
    
    Implements LangChain DeepAgents' interrupt pattern:
    1. Configure which tools require approval via interrupt_on
    2. When tool is called, create InterruptRequest
    3. Pause execution and return interrupt to frontend
    4. Frontend shows prompt box for human decision
    5. Human approves/edits/rejects
    6. Resume execution with decision
    
    Configuration can be provided via:
    - Constructor argument (interrupt_on dict)
    - mcp.json file (humanInTheLoop.interruptOn section)
    
    Usage:
        middleware = HumanInTheLoopMiddleware(
            interrupt_on={
                "delete_file": True,  # Default: approve, edit, reject
                "read_file": False,   # No interrupts
                "send_email": {"allowed_decisions": ["approve", "reject"]},
            }
        )
        
        # Or load from mcp.json:
        middleware = HumanInTheLoopMiddleware(mcp_config_path="./mcp.json")
    """
    
    SYSTEM_PROMPT_ADDITION = """
## Human-in-the-Loop

Some tools require human approval before execution. When you call these tools,
execution will pause and wait for human review. The human can:
- **Approve**: Execute the tool as proposed
- **Edit**: Modify the tool arguments before execution
- **Reject**: Skip the tool execution entirely

Tools requiring approval: {tool_list}

Always explain why you're calling sensitive tools so the human can make an informed decision.
"""
    
    def __init__(
        self,
        interrupt_on: Optional[Dict[str, Any]] = None,
        default_allowed_decisions: Optional[List[str]] = None,
        mcp_config_path: Optional[str] = None,
    ):
        """
        Initialize HITL middleware.
        
        Args:
            interrupt_on: Dict mapping tool names to interrupt configs
            default_allowed_decisions: Default decisions for True configs
            mcp_config_path: Path to mcp.json for loading config
        """
        super().__init__("human_in_the_loop", priority=10)
        
        self._mcp_config_path = mcp_config_path
        self._interrupt_on = interrupt_on or {}
        self._default_decisions = default_allowed_decisions or ["approve", "edit", "reject"]
        self._configs: Dict[str, HITLConfig] = {}
        self._pending_interrupts: Dict[str, InterruptRequest] = {}
        
        # Load config from mcp.json if available
        self._load_config_from_mcp_json()
        
        # Parse configs from constructor (overrides mcp.json)
        for tool_name, config in self._interrupt_on.items():
            self._configs[tool_name] = HITLConfig.from_dict(tool_name, config)
    
    def _load_config_from_mcp_json(self) -> None:
        """
        Load HITL configuration from mcp.json.
        
        Reads the humanInTheLoop.interruptOn section to configure
        which tools require human approval.
        """
        config_paths = [
            self._mcp_config_path,
            "mcp.json",
            "../mcp.json",
        ]
        
        for path in config_paths:
            if path and os.path.exists(path):
                try:
                    with open(path, "r") as f:
                        config = json.load(f)
                    
                    hitl_config = config.get("humanInTheLoop", {})
                    
                    if not hitl_config.get("enabled", True):
                        return  # HITL disabled globally
                    
                    # Load default decisions
                    default_decisions = hitl_config.get("defaultAllowedDecisions")
                    if default_decisions:
                        self._default_decisions = default_decisions
                    
                    # Load interrupt_on configs
                    interrupt_on = hitl_config.get("interruptOn", {})
                    for tool_name, tool_config in interrupt_on.items():
                        if isinstance(tool_config, dict):
                            enabled = tool_config.get("enabled", True)
                            if enabled:
                                self._configs[tool_name] = HITLConfig(
                                    tool_name=tool_name,
                                    enabled=True,
                                    allowed_decisions=tool_config.get(
                                        "allowedDecisions", 
                                        self._default_decisions
                                    ),
                                )
                        elif tool_config is True:
                            self._configs[tool_name] = HITLConfig(
                                tool_name=tool_name,
                                enabled=True,
                                allowed_decisions=self._default_decisions,
                            )
                    
                    return  # Successfully loaded
                except Exception as e:
                    print(f"Warning: Failed to load HITL config from {path}: {e}")
    
    def reload_config(self) -> None:
        """Reload configuration from mcp.json."""
        self._configs.clear()
        self._load_config_from_mcp_json()
    
    def get_system_prompt_addition(self) -> Optional[str]:
        """Return HITL instructions for system prompt."""
        enabled_tools = [
            name for name, cfg in self._configs.items()
            if cfg.enabled
        ]
        
        if not enabled_tools:
            return None
        
        return self.SYSTEM_PROMPT_ADDITION.format(
            tool_list=", ".join(enabled_tools)
        )
    
    def configure_tool(
        self,
        tool_name: str,
        enabled: bool = True,
        allowed_decisions: Optional[List[str]] = None,
    ) -> None:
        """Configure HITL for a specific tool."""
        self._configs[tool_name] = HITLConfig(
            tool_name=tool_name,
            enabled=enabled,
            allowed_decisions=allowed_decisions or self._default_decisions,
        )
    
    def requires_approval(self, tool_name: str) -> bool:
        """Check if tool requires human approval."""
        config = self._configs.get(tool_name)
        return config is not None and config.enabled
    
    def get_allowed_decisions(self, tool_name: str) -> List[str]:
        """Get allowed decisions for a tool."""
        config = self._configs.get(tool_name)
        if config:
            return config.allowed_decisions
        return self._default_decisions
    
    def pre_tool(
        self,
        state: MiddlewareState,
        tool_name: str,
        tool_args: Dict[str, Any],
    ) -> tuple[MiddlewareState, Dict[str, Any]]:
        """
        Check if tool requires approval and create interrupt if needed.
        """
        if not self.requires_approval(tool_name):
            return state, tool_args
        
        # Check if we have a decision for this tool call
        decisions = state.get("decisions", [])
        pending = state.get("pending_interrupts", [])
        
        # Find matching pending interrupt
        for i, interrupt in enumerate(pending):
            if interrupt.tool_name == tool_name:
                # Check if we have a decision
                if i < len(decisions):
                    decision = decisions[i]
                    
                    if decision.type == "reject":
                        # Skip tool execution by returning empty args
                        # The agent will handle this gracefully
                        return state, {"__rejected__": True, "reason": decision.feedback}
                    
                    elif decision.type == "edit" and decision.edited_args:
                        # Use edited arguments
                        return state, decision.edited_args
                    
                    # Approve: use original args
                    return state, tool_args
        
        # No decision yet - create interrupt
        interrupt = InterruptRequest(
            tool_name=tool_name,
            tool_args=tool_args,
            allowed_decisions=self.get_allowed_decisions(tool_name),
            reason=f"Tool '{tool_name}' requires human approval",
        )
        
        # Add to pending interrupts
        pending_list = list(state.get("pending_interrupts", []))
        pending_list.append(interrupt)
        
        new_state = dict(state)
        new_state["pending_interrupts"] = pending_list
        
        return MiddlewareState(**new_state), tool_args
    
    def has_pending_interrupts(self, state: MiddlewareState) -> bool:
        """Check if there are pending interrupts awaiting decisions."""
        pending = state.get("pending_interrupts", [])
        decisions = state.get("decisions", [])
        return len(pending) > len(decisions)
    
    def get_interrupt_response(self, state: MiddlewareState) -> Dict[str, Any]:
        """
        Get interrupt response for frontend.
        
        Returns dict compatible with LangChain's __interrupt__ format.
        """
        pending = state.get("pending_interrupts", [])
        decisions = state.get("decisions", [])
        
        # Get interrupts without decisions
        unanswered = pending[len(decisions):]
        
        return {
            "__interrupt__": [{
                "value": {
                    "action_requests": [
                        {
                            "id": i.id,
                            "name": i.tool_name,
                            "args": i.tool_args,
                        }
                        for i in unanswered
                    ],
                    "review_configs": [
                        {
                            "action_name": i.tool_name,
                            "allowed_decisions": i.allowed_decisions,
                        }
                        for i in unanswered
                    ],
                }
            }]
        }
    
    def apply_decisions(
        self,
        state: MiddlewareState,
        decisions: List[Dict[str, Any]],
    ) -> MiddlewareState:
        """
        Apply human decisions to state.
        
        Args:
            state: Current middleware state
            decisions: List of decision dicts with type and optional edited_args
            
        Returns:
            Updated state with decisions applied
        """
        decision_objects = [
            InterruptDecision(
                type=d.get("type", "approve"),
                edited_args=d.get("edited_args"),
                feedback=d.get("feedback"),
            )
            for d in decisions
        ]
        
        new_state = dict(state)
        existing = list(state.get("decisions", []))
        existing.extend(decision_objects)
        new_state["decisions"] = existing
        
        return MiddlewareState(**new_state)
    
    def clear_interrupts(self, state: MiddlewareState) -> MiddlewareState:
        """Clear all pending interrupts and decisions."""
        new_state = dict(state)
        new_state["pending_interrupts"] = []
        new_state["decisions"] = []
        return MiddlewareState(**new_state)
    
    def on_interrupt(
        self,
        state: MiddlewareState,
        interrupt: InterruptRequest,
    ) -> MiddlewareState:
        """
        Hook when interrupt is triggered.
        
        Prepares interrupt data for frontend display per LangChain DeepAgents pattern.
        Logs pending actions for audit trail.
        """
        # Track interrupt in context for CMP scoring
        context = dict(state.get("context", {}))
        interrupt_history = context.get("hitl_interrupts", [])
        interrupt_history.append({
            "id": interrupt.id,
            "tool_name": interrupt.tool_name,
            "timestamp": interrupt.timestamp,
            "reason": interrupt.reason,
        })
        context["hitl_interrupts"] = interrupt_history[-20:]  # Keep last 20
        context["hitl_pending_count"] = len(state.get("pending_interrupts", [])) + 1
        
        new_state = dict(state)
        new_state["context"] = context
        return MiddlewareState(**new_state)
    
    def on_resume(
        self,
        state: MiddlewareState,
        decision: InterruptDecision,
    ) -> MiddlewareState:
        """
        Hook when execution resumes after interrupt.
        
        Processes human feedback for CMP tracking per LangChain DeepAgents pattern.
        """
        # Track decision in context for CMP scoring
        context = dict(state.get("context", {}))
        decision_history = context.get("hitl_decisions", [])
        decision_history.append({
            "type": decision.type,
            "feedback": decision.feedback,
            "edited": decision.edited_args is not None,
        })
        context["hitl_decisions"] = decision_history[-20:]  # Keep last 20
        
        # Update approval stats for CMP
        stats = context.get("hitl_stats", {"approved": 0, "rejected": 0, "edited": 0})
        if decision.type == "approve":
            stats["approved"] += 1
        elif decision.type == "reject":
            stats["rejected"] += 1
        elif decision.type == "edit":
            stats["edited"] += 1
        context["hitl_stats"] = stats
        
        # Decrement pending count
        context["hitl_pending_count"] = max(0, context.get("hitl_pending_count", 1) - 1)
        
        new_state = dict(state)
        new_state["context"] = context
        return MiddlewareState(**new_state)
    
    def post_tool(
        self,
        state: MiddlewareState,
        tool_name: str,
        result: Any,
    ) -> MiddlewareState:
        """
        Hook after tool execution.
        
        Tracks tool execution results for auditing per LangChain DeepAgents pattern.
        """
        if self.requires_approval(tool_name):
            # Track execution of approved tools
            context = dict(state.get("context", {}))
            executed_tools = context.get("hitl_executed_tools", [])
            executed_tools.append({
                "tool_name": tool_name,
                "required_approval": True,
            })
            context["hitl_executed_tools"] = executed_tools[-20:]  # Keep last 20
            
            new_state = dict(state)
            new_state["context"] = context
            return MiddlewareState(**new_state)
        
        return state
