"""
MCP Tool Middleware for DeepHGM.

Integrates Model Context Protocol (MCP) servers as tools in the
DeepHGM middleware chain, following LangChain's MCP adapter pattern.

Follows LangChain DeepAgents middleware pattern:
- Tool injection via `tools` property
- System prompt additions via `get_system_prompt_addition()`
- Lifecycle hooks: `pre_tool`, `post_tool` for execution tracking
- Real-time call status tracking with callbacks for UI integration
"""

from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import subprocess
import asyncio
import time
from datetime import datetime

from langchain_core.tools import BaseTool, tool

from .base import DeepHGMMiddleware, MiddlewareState


class MCPCallStatus(str, Enum):
    """Status of an MCP tool call."""
    PENDING = "pending"
    CONNECTING = "connecting"
    RUNNING = "running"
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"


@dataclass
class MCPCallRecord:
    """Record of an MCP tool call for tracking and display."""
    id: str
    server_name: str
    tool_name: str
    args: Dict[str, Any]
    status: MCPCallStatus
    started_at: float
    completed_at: Optional[float] = None
    result: Optional[str] = None
    error: Optional[str] = None
    latency_ms: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "server_name": self.server_name,
            "tool_name": self.tool_name,
            "args": self.args,
            "status": self.status.value,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "result": self.result[:500] if self.result else None,  # Truncate for display
            "error": self.error,
            "latency_ms": self.latency_ms,
        }


@dataclass
class MCPServerConfig:
    """Configuration for an MCP server."""
    name: str
    command: str
    args: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)
    description: str = ""
    enabled: bool = True
    
    @classmethod
    def from_dict(cls, name: str, config: Dict[str, Any]) -> "MCPServerConfig":
        """Create from mcp.json server config."""
        return cls(
            name=name,
            command=config.get("command", ""),
            args=config.get("args", []),
            env=config.get("env", {}),
            description=config.get("description", ""),
            enabled=config.get("enabled", True),
        )


class MCPToolWrapper(BaseTool):
    """
    Wrapper to expose MCP server tools as LangChain tools.
    
    This follows the langchain-mcp-adapters pattern for converting
    MCP tools to LangChain-compatible tools.
    """
    
    name: str = ""
    description: str = ""
    server_name: str = ""
    mcp_tool_name: str = ""
    _invoke_fn: Optional[Callable] = None
    
    def __init__(
        self,
        name: str,
        description: str,
        server_name: str,
        mcp_tool_name: str,
        invoke_fn: Optional[Callable] = None,
    ):
        super().__init__()
        self.name = name
        self.description = description
        self.server_name = server_name
        self.mcp_tool_name = mcp_tool_name
        self._invoke_fn = invoke_fn
    
    def _run(self, **kwargs) -> str:
        """Execute the MCP tool."""
        if self._invoke_fn:
            return self._invoke_fn(self.server_name, self.mcp_tool_name, kwargs)
        return f"MCP tool {self.mcp_tool_name} not connected"


class MCPToolMiddleware(DeepHGMMiddleware):
    """
    Middleware for MCP server tool integration.
    
    Loads tools from configured MCP servers and exposes them to the
    DeepHGM agent. Supports the giizhendam-aabajichiganan-mcp server
    for multi-agent CLI orchestration.
    
    Features:
    - Auto-discovery of MCP server tools
    - Tool interceptors for runtime context
    - Integration with HITL for sensitive MCP operations
    - Real-time call status tracking with callbacks
    
    Usage:
        middleware = MCPToolMiddleware(
            servers=[
                MCPServerConfig(
                    name="giizhendam",
                    command="npx",
                    args=["-y", "@nbiish/giizhendam-aabajichiganan-mcp@latest"],
                    enabled=True,
                ),
            ],
            on_call_update=lambda record: print(f"MCP: {record.status}"),
        )
    """
    
    SYSTEM_PROMPT_ADDITION = """
## MCP Tools

You have access to tools from Model Context Protocol (MCP) servers:
{server_list}

These tools provide extended capabilities like:
- Multi-agent CLI orchestration (sequential/parallel execution)
- Expert content generation for RAG systems
- Web search and research capabilities

Use MCP tools when you need capabilities beyond your built-in tools.
"""
    
    def __init__(
        self,
        servers: Optional[List[MCPServerConfig]] = None,
        mcp_config_path: Optional[str] = None,
        agent_folder: Optional[str] = None,
        on_call_update: Optional[Callable[[MCPCallRecord], None]] = None,
        timeout_seconds: float = 30.0,
    ):
        """
        Initialize MCP middleware.
        
        Args:
            servers: List of MCP server configs
            mcp_config_path: Path to mcp.json file
            agent_folder: Agent's working folder for MCP output
            on_call_update: Callback for MCP call status updates (for UI)
            timeout_seconds: Timeout for MCP calls
        """
        super().__init__("mcp_tools", priority=50)
        
        self._servers: Dict[str, MCPServerConfig] = {}
        self._server_tools: Dict[str, List[BaseTool]] = {}
        self._server_processes: Dict[str, subprocess.Popen] = {}
        self._call_records: List[MCPCallRecord] = []
        self._agent_folder = agent_folder
        self._on_call_update = on_call_update
        self._timeout_seconds = timeout_seconds
        self._call_counter = 0
        
        # Load from config file if provided
        if mcp_config_path:
            self._load_from_config(mcp_config_path)
        
        # Add explicitly provided servers
        if servers:
            for server in servers:
                self._servers[server.name] = server
        
        # Initialize tools
        self._initialize_tools()
    
    def _load_from_config(self, config_path: str) -> None:
        """Load MCP servers from mcp.json config file."""
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
            
            mcp_servers = config.get("mcpServers", {})
            for name, server_config in mcp_servers.items():
                self._servers[name] = MCPServerConfig.from_dict(name, server_config)
        except Exception as e:
            print(f"Warning: Failed to load MCP config from {config_path}: {e}")
    
    def _initialize_tools(self) -> None:
        """Initialize tools from enabled MCP servers."""
        self._tools = []
        
        for name, server in self._servers.items():
            if not server.enabled:
                continue
            
            # Create tools based on server type
            if "giizhendam" in name.lower():
                self._tools.extend(self._create_giizhendam_tools(server))
            elif "tavily" in name.lower():
                self._tools.extend(self._create_tavily_tools(server))
            else:
                # Generic MCP tool placeholder
                self._tools.append(self._create_generic_mcp_tool(server))
    
    def _create_giizhendam_tools(self, server: MCPServerConfig) -> List[BaseTool]:
        """Create tools for giizhendam-aabajichiganan-mcp server."""
        tools = []
        
        # Multi-agent orchestration tool
        @tool
        def orchestrate_agents(
            task: str,
            execution_style: str = "sequential",
            agents: Optional[List[str]] = None,
        ) -> str:
            """
            Orchestrate multiple CLI agents to complete a task.
            
            Args:
                task: The task description for agents to complete
                execution_style: 'sequential' or 'parallel' execution
                agents: Optional list of specific agents to use
                
            Returns:
                Combined output from all agents
            """
            return self._invoke_mcp_tool(
                server.name,
                "orchestrate_agents",
                {
                    "task": task,
                    "execution_style": execution_style,
                    "agents": agents or [],
                }
            )
        
        # Expert content generation tool
        @tool
        def generate_expert_content(
            topic: str,
            content_type: str = "analysis",
            output_path: Optional[str] = None,
        ) -> str:
            """
            Generate expert content for RAG systems.
            
            Args:
                topic: The topic to generate content about
                content_type: Type of content (analysis, summary, report)
                output_path: Optional path to save generated content
                
            Returns:
                Generated expert content
            """
            return self._invoke_mcp_tool(
                server.name,
                "generate_expert_content",
                {
                    "topic": topic,
                    "content_type": content_type,
                    "output_path": output_path,
                }
            )
        
        tools.append(orchestrate_agents)
        tools.append(generate_expert_content)
        
        return tools
    
    def _create_tavily_tools(self, server: MCPServerConfig) -> List[BaseTool]:
        """Create tools for tavily-remote-mcp server."""
        tools = []
        
        @tool
        def web_search(query: str, max_results: int = 5) -> str:
            """
            Search the web for information.
            
            Args:
                query: Search query
                max_results: Maximum number of results to return
                
            Returns:
                Search results with snippets and URLs
            """
            return self._invoke_mcp_tool(
                server.name,
                "tavily_search",
                {"query": query, "max_results": max_results}
            )
        
        @tool
        def extract_content(urls: List[str]) -> str:
            """
            Extract content from web pages.
            
            Args:
                urls: List of URLs to extract content from
                
            Returns:
                Extracted content from the URLs
            """
            return self._invoke_mcp_tool(
                server.name,
                "tavily_extract",
                {"urls": urls}
            )
        
        tools.append(web_search)
        tools.append(extract_content)
        
        return tools
    
    def _create_generic_mcp_tool(self, server: MCPServerConfig) -> BaseTool:
        """Create a generic tool for unknown MCP servers."""
        return MCPToolWrapper(
            name=f"mcp_{server.name}",
            description=server.description or f"Tool from MCP server: {server.name}",
            server_name=server.name,
            mcp_tool_name="invoke",
            invoke_fn=self._invoke_mcp_tool,
        )
    
    def _create_call_record(
        self,
        server_name: str,
        tool_name: str,
        args: Dict[str, Any],
    ) -> MCPCallRecord:
        """Create a new call record and notify listeners."""
        self._call_counter += 1
        record = MCPCallRecord(
            id=f"mcp-{self._call_counter}-{int(time.time() * 1000)}",
            server_name=server_name,
            tool_name=tool_name,
            args=args,
            status=MCPCallStatus.PENDING,
            started_at=time.time(),
        )
        self._call_records.append(record)
        self._notify_call_update(record)
        return record
    
    def _update_call_record(
        self,
        record: MCPCallRecord,
        status: MCPCallStatus,
        result: Optional[str] = None,
        error: Optional[str] = None,
    ) -> None:
        """Update a call record and notify listeners."""
        record.status = status
        if result is not None:
            record.result = result
        if error is not None:
            record.error = error
        if status in (MCPCallStatus.SUCCESS, MCPCallStatus.ERROR, MCPCallStatus.TIMEOUT):
            record.completed_at = time.time()
            record.latency_ms = (record.completed_at - record.started_at) * 1000
        self._notify_call_update(record)
    
    def _notify_call_update(self, record: MCPCallRecord) -> None:
        """Notify callback of call status update."""
        if self._on_call_update:
            try:
                self._on_call_update(record)
            except Exception as e:
                print(f"Warning: MCP call update callback failed: {e}")
    
    def get_call_records(self) -> List[Dict[str, Any]]:
        """Get all call records as dictionaries for serialization."""
        return [r.to_dict() for r in self._call_records]
    
    def get_recent_calls(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent call records."""
        return [r.to_dict() for r in self._call_records[-limit:]]
    
    def clear_call_records(self) -> None:
        """Clear all call records."""
        self._call_records.clear()
    
    def _invoke_mcp_tool(
        self,
        server_name: str,
        tool_name: str,
        args: Dict[str, Any],
    ) -> str:
        """
        Invoke an MCP tool with status tracking.
        
        Uses subprocess to spawn the MCP server and communicate via stdio.
        Tracks call status for UI feedback.
        """
        server = self._servers.get(server_name)
        if not server:
            return json.dumps({
                "status": "error",
                "error": f"MCP server '{server_name}' not found",
            })
        
        if not server.enabled:
            return json.dumps({
                "status": "error",
                "error": f"MCP server '{server_name}' is disabled",
            })
        
        # Create call record for tracking
        record = self._create_call_record(server_name, tool_name, args)
        
        try:
            # Update status to connecting
            self._update_call_record(record, MCPCallStatus.CONNECTING)
            
            # Build environment with server config
            env = {**server.env}
            if self._agent_folder:
                env["AGENT_OUTPUT_DIR"] = f"{self._agent_folder}/outputs"
            
            # Build the MCP request
            mcp_request = {
                "jsonrpc": "2.0",
                "id": record.id,
                "method": "tools/call",
                "params": {
                    "name": tool_name,
                    "arguments": args,
                }
            }
            
            # Update status to running
            self._update_call_record(record, MCPCallStatus.RUNNING)
            
            # Execute via subprocess (stdio transport)
            cmd = [server.command] + server.args
            
            try:
                process = subprocess.Popen(
                    cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env={**subprocess.os.environ, **env},
                    text=True,
                )
                
                # Send request and get response
                stdout, stderr = process.communicate(
                    input=json.dumps(mcp_request) + "\n",
                    timeout=self._timeout_seconds,
                )
                
                if process.returncode != 0:
                    error_msg = stderr.strip() or f"Process exited with code {process.returncode}"
                    self._update_call_record(record, MCPCallStatus.ERROR, error=error_msg)
                    return json.dumps({
                        "status": "error",
                        "error": error_msg,
                        "server": server_name,
                        "tool": tool_name,
                    })
                
                # Parse response
                try:
                    response = json.loads(stdout.strip()) if stdout.strip() else {}
                    result_str = json.dumps(response)
                    self._update_call_record(record, MCPCallStatus.SUCCESS, result=result_str)
                    return result_str
                except json.JSONDecodeError:
                    # Return raw output if not JSON
                    self._update_call_record(record, MCPCallStatus.SUCCESS, result=stdout)
                    return json.dumps({
                        "status": "success",
                        "server": server_name,
                        "tool": tool_name,
                        "result": stdout.strip(),
                    })
                    
            except subprocess.TimeoutExpired:
                process.kill()
                self._update_call_record(record, MCPCallStatus.TIMEOUT, error="Request timed out")
                return json.dumps({
                    "status": "timeout",
                    "error": f"MCP call timed out after {self._timeout_seconds}s",
                    "server": server_name,
                    "tool": tool_name,
                })
            except FileNotFoundError:
                error_msg = f"MCP server command not found: {server.command}"
                self._update_call_record(record, MCPCallStatus.ERROR, error=error_msg)
                return json.dumps({
                    "status": "error",
                    "error": error_msg,
                    "server": server_name,
                    "tool": tool_name,
                })
                
        except Exception as e:
            error_msg = str(e)
            self._update_call_record(record, MCPCallStatus.ERROR, error=error_msg)
            return json.dumps({
                "status": "error",
                "error": error_msg,
                "server": server_name,
                "tool": tool_name,
            })
    
    def get_system_prompt_addition(self) -> Optional[str]:
        """Return MCP tools instructions for system prompt."""
        enabled_servers = [
            f"- **{name}**: {server.description}"
            for name, server in self._servers.items()
            if server.enabled
        ]
        
        if not enabled_servers:
            return None
        
        return self.SYSTEM_PROMPT_ADDITION.format(
            server_list="\n".join(enabled_servers)
        )
    
    def add_server(self, server: MCPServerConfig) -> None:
        """Add an MCP server configuration."""
        self._servers[server.name] = server
        self._initialize_tools()
    
    def remove_server(self, name: str) -> bool:
        """Remove an MCP server."""
        if name in self._servers:
            del self._servers[name]
            self._initialize_tools()
            return True
        return False
    
    def enable_server(self, name: str) -> bool:
        """Enable an MCP server."""
        if name in self._servers:
            self._servers[name].enabled = True
            self._initialize_tools()
            return True
        return False
    
    def disable_server(self, name: str) -> bool:
        """Disable an MCP server."""
        if name in self._servers:
            self._servers[name].enabled = False
            self._initialize_tools()
            return True
        return False
    
    def pre_tool(
        self,
        state: MiddlewareState,
        tool_name: str,
        tool_args: Dict[str, Any],
    ) -> tuple[MiddlewareState, Dict[str, Any]]:
        """
        Hook before MCP tool execution.
        
        Validates MCP tool calls and prepares state per LangChain DeepAgents pattern.
        """
        # Check if this is an MCP tool
        mcp_tools = ["orchestrate_agents", "generate_expert_content", "web_search", "extract_content"]
        mcp_tools.extend([f"mcp_{name}" for name in self._servers.keys()])
        
        if tool_name in mcp_tools:
            # Add MCP context to state
            context = dict(state.get("context", {}))
            context["mcp_tool_pending"] = {
                "tool_name": tool_name,
                "args": tool_args,
            }
            new_state = dict(state)
            new_state["context"] = context
            return MiddlewareState(**new_state), tool_args
        
        return state, tool_args
    
    def post_tool(
        self,
        state: MiddlewareState,
        tool_name: str,
        result: Any,
    ) -> MiddlewareState:
        """
        Hook after MCP tool execution.
        
        Tracks MCP call results for CMP scoring per LangChain DeepAgents pattern.
        """
        # Check if this is an MCP tool
        mcp_tools = ["orchestrate_agents", "generate_expert_content", "web_search", "extract_content"]
        mcp_tools.extend([f"mcp_{name}" for name in self._servers.keys()])
        
        if tool_name in mcp_tools:
            # Get the most recent call record
            if self._call_records:
                latest = self._call_records[-1]
                
                # Update context with MCP call info for CMP tracking
                context = dict(state.get("context", {}))
                mcp_history = context.get("mcp_calls", [])
                mcp_history.append(latest.to_dict())
                context["mcp_calls"] = mcp_history[-10:]  # Keep last 10
                
                # Clear pending
                context.pop("mcp_tool_pending", None)
                
                new_state = dict(state)
                new_state["context"] = context
                return MiddlewareState(**new_state)
        
        return state
