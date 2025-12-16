"""
CLI Tools for Agent Workspace Operations.

Purpose: Provide qwen and gemini CLI tools for agent workspace manipulation
Inputs: Agent workspace path, prompts
Outputs: CLI execution results

Each agent is isolated to its own folder. CLI tools allow agents to:
- Create complex projects
- Modify files
- Run builds and tests
- Generate documentation

Based on OSA.md patterns for multi-agent CLI orchestration.
"""

from typing import Any, Callable, Dict, List, Optional, Set
from dataclasses import dataclass
from enum import Enum
import subprocess
import os
import json

from langchain_core.tools import BaseTool, tool


class CLIProvider(Enum):
    """CLI agent providers."""
    QWEN = "qwen"
    GEMINI = "gemini"


class ExecutionMode(Enum):
    """CLI execution modes."""
    NORMAL = "normal"      # Standard development
    FULL_AUTO = "full_auto"  # -y flag, trusted automation
    SANDBOX = "sandbox"    # -s flag, safe execution


@dataclass
class CLIToolConfig:
    """Configuration for a CLI tool."""
    
    name: str
    provider: CLIProvider
    enabled: bool = True
    sandbox_mode: bool = False
    timeout_seconds: int = 120
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "provider": self.provider.value,
            "enabled": self.enabled,
            "sandbox_mode": self.sandbox_mode,
            "timeout_seconds": self.timeout_seconds,
        }


@dataclass
class CLICallRecord:
    """Record of a CLI tool call for tracking."""
    
    id: str
    provider: str
    prompt: str
    status: str  # pending, running, success, error, timeout
    started_at: float
    workspace_path: str
    completed_at: Optional[float] = None
    result: Optional[str] = None
    error: Optional[str] = None
    latency_ms: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "provider": self.provider,
            "prompt": self.prompt[:200] + "..." if len(self.prompt) > 200 else self.prompt,
            "status": self.status,
            "started_at": self.started_at,
            "workspace_path": self.workspace_path,
            "completed_at": self.completed_at,
            "result": self.result[:500] if self.result else None,
            "error": self.error,
            "latency_ms": self.latency_ms,
        }


class CLIToolManager:
    """
    Manages CLI tools for agents with per-agent configuration.
    
    Provides qwen and gemini CLI tools that operate within an agent's
    isolated workspace folder.
    """
    
    def __init__(
        self,
        workspace_path: str,
        default_mode: ExecutionMode = ExecutionMode.FULL_AUTO,
        on_call_update: Optional[Callable[[CLICallRecord], None]] = None,
    ):
        """
        Initialize CLI tool manager.
        
        Args:
            workspace_path: Agent's isolated workspace directory
            default_mode: Default execution mode (FULL_AUTO for -y flag)
            on_call_update: Callback for CLI call status updates
        """
        self.workspace_path = workspace_path
        self.default_mode = default_mode
        self.on_call_update = on_call_update
        self._call_counter = 0
        self._call_records: List[CLICallRecord] = []
        
        # Tool configurations
        self._tools: Dict[CLIProvider, CLIToolConfig] = {
            CLIProvider.QWEN: CLIToolConfig(
                name="qwen",
                provider=CLIProvider.QWEN,
                enabled=True,
            ),
            CLIProvider.GEMINI: CLIToolConfig(
                name="gemini",
                provider=CLIProvider.GEMINI,
                enabled=True,
            ),
        }
        
        # Ensure workspace exists
        os.makedirs(workspace_path, exist_ok=True)
    
    def enable_tool(self, provider: CLIProvider) -> bool:
        """Enable a CLI tool."""
        if provider in self._tools:
            self._tools[provider].enabled = True
            return True
        return False
    
    def disable_tool(self, provider: CLIProvider) -> bool:
        """Disable a CLI tool."""
        if provider in self._tools:
            self._tools[provider].enabled = False
            return True
        return False
    
    def set_sandbox_mode(self, provider: CLIProvider, sandbox: bool) -> bool:
        """Set sandbox mode for a CLI tool."""
        if provider in self._tools:
            self._tools[provider].sandbox_mode = sandbox
            return True
        return False
    
    def execute_qwen(
        self,
        prompt: str,
        output_file: Optional[str] = None,
        sandbox: Optional[bool] = None,
    ) -> str:
        """
        Execute qwen CLI command in the agent's workspace.
        
        Syntax: qwen [-y] [-s] "{prompt}"
        - -y: Full-auto mode (skip confirmations)
        - -s: Sandbox mode (isolated execution)
        
        Args:
            prompt: Task description for qwen
            output_file: Optional file to write output to
            sandbox: Override sandbox mode
        """
        import time
        
        config = self._tools[CLIProvider.QWEN]
        
        # Create call record
        self._call_counter += 1
        record = CLICallRecord(
            id=f"cli-qwen-{self._call_counter}-{int(time.time() * 1000)}",
            provider="qwen",
            prompt=prompt,
            status="pending",
            started_at=time.time(),
            workspace_path=self.workspace_path,
        )
        self._call_records.append(record)
        self._notify(record)
        
        # Build command
        cmd = ["qwen"]
        
        use_sandbox = sandbox if sandbox is not None else config.sandbox_mode
        if use_sandbox:
            cmd.append("-s")
        elif self.default_mode == ExecutionMode.FULL_AUTO:
            cmd.append("-y")
        
        # Enhance prompt with output file if specified
        full_prompt = prompt
        if output_file:
            # Make path relative to workspace
            if not os.path.isabs(output_file):
                output_file = os.path.join(self.workspace_path, output_file)
            full_prompt = f"{prompt}\n\nWrite output to: {output_file}"
        
        cmd.append(full_prompt)
        
        record.status = "running"
        self._notify(record)
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=config.timeout_seconds,
                cwd=self.workspace_path,
            )
            
            if result.returncode != 0:
                record.status = "error"
                record.error = result.stderr.strip() or f"Exit code: {result.returncode}"
                record.completed_at = time.time()
                record.latency_ms = (record.completed_at - record.started_at) * 1000
                self._notify(record)
                return f"[qwen error] {record.error}"
            
            record.status = "success"
            record.result = result.stdout.strip() or "[qwen] Task completed"
            record.completed_at = time.time()
            record.latency_ms = (record.completed_at - record.started_at) * 1000
            self._notify(record)
            return record.result
            
        except subprocess.TimeoutExpired:
            record.status = "timeout"
            record.error = f"Timeout after {config.timeout_seconds}s"
            record.completed_at = time.time()
            record.latency_ms = (record.completed_at - record.started_at) * 1000
            self._notify(record)
            return f"[qwen error] {record.error}"
            
        except FileNotFoundError:
            record.status = "error"
            record.error = "qwen CLI not installed"
            record.completed_at = time.time()
            record.latency_ms = (record.completed_at - record.started_at) * 1000
            self._notify(record)
            return "[qwen error] CLI not installed. Install: npm i -g @anthropics/qwen-cli"
            
        except Exception as e:
            record.status = "error"
            record.error = str(e)
            record.completed_at = time.time()
            record.latency_ms = (record.completed_at - record.started_at) * 1000
            self._notify(record)
            return f"[qwen error] {str(e)}"
    
    def execute_gemini(
        self,
        prompt: str,
        output_file: Optional[str] = None,
        sandbox: Optional[bool] = None,
    ) -> str:
        """
        Execute gemini CLI command in the agent's workspace.
        
        Syntax: gemini [-y] [-s] -p "{prompt}"
        - -y: Full-auto mode (skip confirmations)
        - -s: Sandbox mode (isolated, no network)
        - -p: Prompt flag (required)
        
        Args:
            prompt: Task description for gemini
            output_file: Optional file to write output to
            sandbox: Override sandbox mode
        """
        import time
        
        config = self._tools[CLIProvider.GEMINI]
        
        # Create call record
        self._call_counter += 1
        record = CLICallRecord(
            id=f"cli-gemini-{self._call_counter}-{int(time.time() * 1000)}",
            provider="gemini",
            prompt=prompt,
            status="pending",
            started_at=time.time(),
            workspace_path=self.workspace_path,
        )
        self._call_records.append(record)
        self._notify(record)
        
        # Build command
        cmd = ["gemini"]
        
        use_sandbox = sandbox if sandbox is not None else config.sandbox_mode
        if use_sandbox:
            cmd.append("-s")
        elif self.default_mode == ExecutionMode.FULL_AUTO:
            cmd.append("-y")
        
        cmd.append("-p")
        
        # Enhance prompt with output file if specified
        full_prompt = prompt
        if output_file:
            if not os.path.isabs(output_file):
                output_file = os.path.join(self.workspace_path, output_file)
            full_prompt = f"{prompt}\n\nWrite output to: {output_file}"
        
        cmd.append(full_prompt)
        
        record.status = "running"
        self._notify(record)
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=config.timeout_seconds,
                cwd=self.workspace_path,
            )
            
            if result.returncode != 0:
                record.status = "error"
                record.error = result.stderr.strip() or f"Exit code: {result.returncode}"
                record.completed_at = time.time()
                record.latency_ms = (record.completed_at - record.started_at) * 1000
                self._notify(record)
                return f"[gemini error] {record.error}"
            
            record.status = "success"
            record.result = result.stdout.strip() or "[gemini] Task completed"
            record.completed_at = time.time()
            record.latency_ms = (record.completed_at - record.started_at) * 1000
            self._notify(record)
            return record.result
            
        except subprocess.TimeoutExpired:
            record.status = "timeout"
            record.error = f"Timeout after {config.timeout_seconds}s"
            record.completed_at = time.time()
            record.latency_ms = (record.completed_at - record.started_at) * 1000
            self._notify(record)
            return f"[gemini error] {record.error}"
            
        except FileNotFoundError:
            record.status = "error"
            record.error = "gemini CLI not installed"
            record.completed_at = time.time()
            record.latency_ms = (record.completed_at - record.started_at) * 1000
            self._notify(record)
            return "[gemini error] CLI not installed. Install: npm i -g @anthropics/gemini-cli"
            
        except Exception as e:
            record.status = "error"
            record.error = str(e)
            record.completed_at = time.time()
            record.latency_ms = (record.completed_at - record.started_at) * 1000
            self._notify(record)
            return f"[gemini error] {str(e)}"
    
    def _notify(self, record: CLICallRecord) -> None:
        """Notify callback of call update."""
        if self.on_call_update:
            try:
                self.on_call_update(record)
            except Exception:
                pass
    
    def get_call_records(self) -> List[Dict[str, Any]]:
        """Get all call records."""
        return [r.to_dict() for r in self._call_records]
    
    def clear_records(self) -> None:
        """Clear call records."""
        self._call_records.clear()
    
    def get_langchain_tools(self, enabled_tools: Optional[Set[str]] = None) -> List[BaseTool]:
        """
        Get LangChain-compatible tools for enabled CLI providers.
        
        Args:
            enabled_tools: Set of tool names to include (None = all enabled)
            
        Returns:
            List of LangChain tools
        """
        tools = []
        manager = self  # Capture for closure
        
        # Check if qwen should be included
        qwen_config = self._tools[CLIProvider.QWEN]
        if qwen_config.enabled and (enabled_tools is None or "qwen" in enabled_tools):
            @tool
            def qwen(prompt: str, output_file: Optional[str] = None) -> str:
                """
                Fast reasoning CLI agent. Best for: summarization, analysis, quick coding tasks.
                
                Works in the agent's isolated workspace folder. Can create and modify files.
                
                Args:
                    prompt: Task description
                    output_file: Optional file path to write output (relative to workspace)
                """
                return manager.execute_qwen(prompt, output_file)
            
            tools.append(qwen)
        
        # Check if gemini should be included
        gemini_config = self._tools[CLIProvider.GEMINI]
        if gemini_config.enabled and (enabled_tools is None or "gemini" in enabled_tools):
            @tool
            def gemini(prompt: str, output_file: Optional[str] = None) -> str:
                """
                Multimodal CLI agent. Best for: complex code, documentation, design, large context.
                
                Works in the agent's isolated workspace folder. Can create and modify files.
                
                Args:
                    prompt: Task description
                    output_file: Optional file path to write output (relative to workspace)
                """
                return manager.execute_gemini(prompt, output_file)
            
            tools.append(gemini)
        
        return tools
    
    def get_tool_status(self) -> List[Dict[str, Any]]:
        """Get status of all CLI tools."""
        return [
            {
                "name": config.name,
                "provider": config.provider.value,
                "enabled": config.enabled,
                "sandbox_mode": config.sandbox_mode,
                "timeout_seconds": config.timeout_seconds,
            }
            for config in self._tools.values()
        ]


def create_cli_tools_for_agent(
    workspace_path: str,
    enabled_tools: Optional[Set[str]] = None,
    on_call_update: Optional[Callable[[CLICallRecord], None]] = None,
) -> List[BaseTool]:
    """
    Create CLI tools for a specific agent instance.
    
    Args:
        workspace_path: Agent's isolated workspace directory
        enabled_tools: Set of tool names to enable ("qwen", "gemini")
        on_call_update: Callback for CLI call status updates
        
    Returns:
        List of LangChain tools for the enabled CLI providers
    """
    manager = CLIToolManager(
        workspace_path=workspace_path,
        on_call_update=on_call_update,
    )
    return manager.get_langchain_tools(enabled_tools)
