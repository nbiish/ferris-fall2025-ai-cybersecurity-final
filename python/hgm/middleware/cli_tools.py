"""
CLI Tool Middleware for DeepHGM.

Integrates qwen and gemini CLI tools into the DeepAgents middleware.
Based on OSA.md patterns for multi-agent CLI orchestration.

Follows LangChain DeepAgents middleware pattern:
- Tool injection via `tools` property
- System prompt additions via `get_system_prompt_addition()`
- Lifecycle hooks: `pre_tool`, `post_tool` for execution tracking

CLI Syntax:
- qwen: `qwen -y "{prompt}"` (fast reasoning, summarization)
- gemini: `gemini -y -p "{prompt}"` (multimodal, large context, coding)

Modes:
- Full-auto: -y flag (trusted repos, automation)
- Sandbox: -s flag (untrusted code, safe execution)
- Normal: no flags (standard development)
"""

from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import subprocess
import os
import json
import time

from langchain_core.tools import BaseTool, tool

from .base import DeepHGMMiddleware, MiddlewareState


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
class CLIExecutionRecord:
    """Record of a CLI tool execution for tracking."""
    id: str
    provider: str
    prompt: str
    started_at: float
    completed_at: Optional[float] = None
    success: bool = False
    output: Optional[str] = None
    error: Optional[str] = None
    latency_ms: Optional[float] = None
    sandbox: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "provider": self.provider,
            "prompt": self.prompt[:100] + "..." if len(self.prompt) > 100 else self.prompt,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "success": self.success,
            "output": self.output[:500] if self.output else None,
            "error": self.error,
            "latency_ms": self.latency_ms,
            "sandbox": self.sandbox,
        }


@dataclass
class CLIConfig:
    """CLI provider configuration."""
    provider: CLIProvider
    enabled: bool = True
    mode: ExecutionMode = ExecutionMode.FULL_AUTO
    timeout_seconds: int = 120
    output_dir: str = "./tmp"
    
    @classmethod
    def qwen_default(cls) -> "CLIConfig":
        return cls(provider=CLIProvider.QWEN, enabled=True)
    
    @classmethod
    def gemini_default(cls) -> "CLIConfig":
        return cls(provider=CLIProvider.GEMINI, enabled=True)


class CLIToolMiddleware(DeepHGMMiddleware):
    """
    CLI tool middleware for qwen and gemini agents.
    
    Provides concise tool wrappers for CLI-based agentic compute.
    Reads configuration from mcp.json to sync with frontend toggles.
    
    Per LangChain DeepAgents docs:
    - Tools are dynamically added/removed based on cliTools config in mcp.json
    - System prompt additions explain tool capabilities
    - post_tool hook tracks execution results for CMP scoring
    
    Usage:
        middleware = CLIToolMiddleware(
            workspace_root="/path/to/workspace",
            enabled_providers=["qwen", "gemini"],
            on_execution_complete=lambda record: print(f"CLI: {record.provider} completed"),
        )
    """
    
    SYSTEM_PROMPT_ADDITION = """## CLI Agents

You have access to CLI-based AI agents for specialized tasks:

- **qwen**: Fast reasoning agent. Use for:
  - Quick analysis and summarization
  - Code review and explanations
  - Rapid prototyping and iterations
  
- **gemini**: Multimodal agent with large context. Use for:
  - Complex code generation and refactoring
  - Design documents and architecture
  - Tasks requiring visual understanding

Both tools accept a `sandbox` parameter for safe execution of untrusted code."""
    
    def __init__(
        self,
        workspace_root: Optional[str] = None,
        enabled_providers: Optional[List[str]] = None,
        default_mode: ExecutionMode = ExecutionMode.FULL_AUTO,
        mcp_config_path: Optional[str] = None,
        on_execution_complete: Optional[Callable[[CLIExecutionRecord], None]] = None,
    ):
        super().__init__("cli_tools", priority=60)
        
        self._workspace_root = workspace_root or os.getcwd()
        self._default_mode = default_mode
        self._mcp_config_path = mcp_config_path
        self._on_execution_complete = on_execution_complete
        
        # Execution tracking
        self._execution_records: List[CLIExecutionRecord] = []
        self._execution_counter = 0
        
        # Initialize providers with defaults
        self._providers: Dict[CLIProvider, CLIConfig] = {
            CLIProvider.QWEN: CLIConfig.qwen_default(),
            CLIProvider.GEMINI: CLIConfig.gemini_default(),
        }
        
        # Load config from mcp.json if available (syncs with frontend toggles)
        self._load_config_from_mcp_json()
        
        # Apply enabled filter from constructor (overrides mcp.json if provided)
        if enabled_providers:
            for p in self._providers.values():
                p.enabled = p.provider.value in enabled_providers
        
        self._initialize_tools()
    
    def _load_config_from_mcp_json(self) -> None:
        """
        Load CLI tool configuration from mcp.json.
        
        This syncs the middleware with frontend toggle state, following
        the LangChain DeepAgents pattern for dynamic tool management.
        """
        config_paths = [
            self._mcp_config_path,
            os.path.join(self._workspace_root, "mcp.json"),
            "mcp.json",
            "../mcp.json",
        ]
        
        for path in config_paths:
            if path and os.path.exists(path):
                try:
                    with open(path, "r") as f:
                        config = json.load(f)
                    
                    cli_tools = config.get("cliTools", {})
                    
                    # Update qwen config
                    if "qwen" in cli_tools:
                        qwen_cfg = cli_tools["qwen"]
                        self._providers[CLIProvider.QWEN].enabled = qwen_cfg.get("enabled", True)
                        if qwen_cfg.get("sandbox", False):
                            self._providers[CLIProvider.QWEN].mode = ExecutionMode.SANDBOX
                    
                    # Update gemini config
                    if "gemini" in cli_tools:
                        gemini_cfg = cli_tools["gemini"]
                        self._providers[CLIProvider.GEMINI].enabled = gemini_cfg.get("enabled", True)
                        if gemini_cfg.get("sandbox", False):
                            self._providers[CLIProvider.GEMINI].mode = ExecutionMode.SANDBOX
                    
                    return  # Successfully loaded
                except Exception as e:
                    print(f"Warning: Failed to load CLI config from {path}: {e}")
    
    def reload_config(self) -> None:
        """
        Reload configuration from mcp.json and reinitialize tools.
        
        Call this method to sync with frontend toggle changes at runtime.
        """
        self._load_config_from_mcp_json()
        self._initialize_tools()
    
    def _initialize_tools(self) -> None:
        """Initialize CLI tools."""
        self._tools = []
        
        if self._providers[CLIProvider.QWEN].enabled:
            self._tools.append(self._create_qwen_tool())
        
        if self._providers[CLIProvider.GEMINI].enabled:
            self._tools.append(self._create_gemini_tool())
    
    def _create_qwen_tool(self) -> BaseTool:
        """Create qwen CLI tool with detailed description per LangChain DeepAgents pattern."""
        middleware = self
        
        @tool
        def qwen(prompt: str, sandbox: bool = False, output_file: Optional[str] = None) -> str:
            """
            Fast reasoning CLI agent for quick tasks.
            
            Use qwen for:
            - Summarization and analysis
            - Code review and explanations
            - Quick prototyping
            - Simple transformations
            
            Args:
                prompt: The task description or question
                sandbox: If True, runs in isolated sandbox mode (safe for untrusted code)
                output_file: Optional path to save output
                
            Returns:
                Agent's response or error message
            """
            return middleware._execute_qwen(prompt, sandbox, output_file)
        
        return qwen
    
    def _create_gemini_tool(self) -> BaseTool:
        """Create gemini CLI tool with detailed description per LangChain DeepAgents pattern."""
        middleware = self
        
        @tool
        def gemini(prompt: str, sandbox: bool = False, output_file: Optional[str] = None) -> str:
            """
            Multimodal CLI agent with large context window.
            
            Use gemini for:
            - Complex code generation and refactoring
            - Architecture and design documents
            - Tasks requiring large context (multiple files)
            - Visual/multimodal understanding
            
            Args:
                prompt: The task description or question
                sandbox: If True, runs in isolated sandbox mode (safe for untrusted code)
                output_file: Optional path to save output
                
            Returns:
                Agent's response or error message
            """
            return middleware._execute_gemini(prompt, sandbox, output_file)
        
        return gemini
    
    def _create_execution_record(
        self,
        provider: str,
        prompt: str,
        sandbox: bool = False,
    ) -> CLIExecutionRecord:
        """Create execution record for tracking."""
        self._execution_counter += 1
        record = CLIExecutionRecord(
            id=f"cli-{self._execution_counter}-{int(time.time() * 1000)}",
            provider=provider,
            prompt=prompt,
            started_at=time.time(),
            sandbox=sandbox,
        )
        self._execution_records.append(record)
        return record
    
    def _complete_execution_record(
        self,
        record: CLIExecutionRecord,
        success: bool,
        output: Optional[str] = None,
        error: Optional[str] = None,
    ) -> None:
        """Complete execution record and notify callback."""
        record.completed_at = time.time()
        record.success = success
        record.output = output
        record.error = error
        record.latency_ms = (record.completed_at - record.started_at) * 1000
        
        if self._on_execution_complete:
            try:
                self._on_execution_complete(record)
            except Exception as e:
                print(f"Warning: CLI execution callback failed: {e}")
    
    def get_execution_records(self) -> List[Dict[str, Any]]:
        """Get all execution records for serialization."""
        return [r.to_dict() for r in self._execution_records]
    
    def get_recent_executions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent execution records."""
        return [r.to_dict() for r in self._execution_records[-limit:]]
    
    def _execute_qwen(
        self,
        prompt: str,
        sandbox: bool = False,
        output_file: Optional[str] = None,
    ) -> str:
        """
        Execute qwen CLI command.
        
        Syntax: qwen [-y] [-s] "{prompt}"
        - -y: Full-auto mode (skip confirmations)
        - -s: Sandbox mode (isolated execution)
        """
        config = self._providers[CLIProvider.QWEN]
        record = self._create_execution_record("qwen", prompt, sandbox)
        
        # Build command
        cmd = ["qwen"]
        
        if sandbox:
            cmd.append("-s")
        elif self._default_mode == ExecutionMode.FULL_AUTO:
            cmd.append("-y")
        
        # Add output redirect if specified
        if output_file:
            full_prompt = f"{prompt}, output to {output_file}"
        else:
            full_prompt = prompt
        
        cmd.append(full_prompt)
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=config.timeout_seconds,
                cwd=self._workspace_root,
            )
            
            if result.returncode != 0:
                error_msg = f"[qwen error] {result.stderr.strip()}"
                self._complete_execution_record(record, False, error=error_msg)
                return error_msg
            
            output = result.stdout.strip() or "[qwen] Task completed"
            self._complete_execution_record(record, True, output=output)
            return output
            
        except subprocess.TimeoutExpired:
            error_msg = f"[qwen error] Timeout after {config.timeout_seconds}s"
            self._complete_execution_record(record, False, error=error_msg)
            return error_msg
        except FileNotFoundError:
            error_msg = "[qwen error] CLI not installed. Install: npm i -g @anthropics/qwen-cli"
            self._complete_execution_record(record, False, error=error_msg)
            return error_msg
        except Exception as e:
            error_msg = f"[qwen error] {str(e)}"
            self._complete_execution_record(record, False, error=error_msg)
            return error_msg
    
    def _execute_gemini(
        self,
        prompt: str,
        sandbox: bool = False,
        output_file: Optional[str] = None,
    ) -> str:
        """
        Execute gemini CLI command.
        
        Syntax: gemini [-y] [-s] -p "{prompt}"
        - -y: Full-auto mode (skip confirmations)
        - -s/--sandbox: Sandbox mode (isolated, no network)
        - -p: Prompt flag (required)
        """
        config = self._providers[CLIProvider.GEMINI]
        record = self._create_execution_record("gemini", prompt, sandbox)
        
        # Build command
        cmd = ["gemini"]
        
        if sandbox:
            cmd.append("-s")
        elif self._default_mode == ExecutionMode.FULL_AUTO:
            cmd.append("-y")
        
        cmd.append("-p")
        
        # Add output redirect if specified
        if output_file:
            full_prompt = f"{prompt}, output to {output_file}"
        else:
            full_prompt = prompt
        
        cmd.append(full_prompt)
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=config.timeout_seconds,
                cwd=self._workspace_root,
            )
            
            if result.returncode != 0:
                error_msg = f"[gemini error] {result.stderr.strip()}"
                self._complete_execution_record(record, False, error=error_msg)
                return error_msg
            
            output = result.stdout.strip() or "[gemini] Task completed"
            self._complete_execution_record(record, True, output=output)
            return output
            
        except subprocess.TimeoutExpired:
            error_msg = f"[gemini error] Timeout after {config.timeout_seconds}s"
            self._complete_execution_record(record, False, error=error_msg)
            return error_msg
        except FileNotFoundError:
            error_msg = "[gemini error] CLI not installed. Install: npm i -g @anthropics/gemini-cli"
            self._complete_execution_record(record, False, error=error_msg)
            return error_msg
        except Exception as e:
            error_msg = f"[gemini error] {str(e)}"
            self._complete_execution_record(record, False, error=error_msg)
            return error_msg
    
    def get_system_prompt_addition(self) -> Optional[str]:
        """Return concise CLI tools description."""
        enabled = [p for p in self._providers.values() if p.enabled]
        if not enabled:
            return None
        return self.SYSTEM_PROMPT_ADDITION
    
    def set_mode(self, mode: ExecutionMode) -> None:
        """Set default execution mode for all providers."""
        self._default_mode = mode
    
    def enable_provider(self, provider: CLIProvider) -> bool:
        """Enable a CLI provider."""
        if provider in self._providers:
            self._providers[provider].enabled = True
            self._initialize_tools()
            return True
        return False
    
    def disable_provider(self, provider: CLIProvider) -> bool:
        """Disable a CLI provider."""
        if provider in self._providers:
            self._providers[provider].enabled = False
            self._initialize_tools()
            return True
        return False
    
    def post_tool(
        self,
        state: MiddlewareState,
        tool_name: str,
        result: Any,
    ) -> MiddlewareState:
        """
        Hook after CLI tool execution.
        
        Tracks execution results for CMP scoring per LangChain DeepAgents pattern.
        """
        if tool_name in ("qwen", "gemini"):
            # Get the most recent execution record
            if self._execution_records:
                latest = self._execution_records[-1]
                
                # Update context with execution info for CMP tracking
                context = dict(state.get("context", {}))
                cli_history = context.get("cli_executions", [])
                cli_history.append(latest.to_dict())
                context["cli_executions"] = cli_history[-10:]  # Keep last 10
                
                new_state = dict(state)
                new_state["context"] = context
                return MiddlewareState(**new_state)
        
        return state
