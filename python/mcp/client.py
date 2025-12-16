"""
MCP Client for Direct Python Integration.

Purpose: Handle MCP server communication without Rust/Tauri layer
Inputs: MCP server configurations from settings
Outputs: LangChain-compatible tools for agents

Replaces the Tauri mcp_commands.rs functionality with pure Python.
Supports per-agent tool enabling/disabling at runtime.
"""

from typing import Any, Callable, Dict, List, Optional, Set
from dataclasses import dataclass, field
import json
import subprocess
import asyncio
import time

from langchain_core.tools import BaseTool, tool


@dataclass
class MCPCallRecord:
    """Record of an MCP tool call for tracking."""
    
    id: str
    server_name: str
    tool_name: str
    args: Dict[str, Any]
    status: str  # pending, running, success, error, timeout
    started_at: float
    completed_at: Optional[float] = None
    result: Optional[str] = None
    error: Optional[str] = None
    latency_ms: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "server_name": self.server_name,
            "tool_name": self.tool_name,
            "args": self.args,
            "status": self.status,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "result": self.result[:500] if self.result else None,
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
    is_hardcoded: bool = False


class MCPClient:
    """
    Client for invoking MCP server tools.
    
    Handles subprocess communication with MCP servers using stdio transport.
    """
    
    def __init__(
        self,
        timeout_seconds: float = 30.0,
        on_call_update: Optional[Callable[[MCPCallRecord], None]] = None,
    ):
        self.timeout_seconds = timeout_seconds
        self.on_call_update = on_call_update
        self._call_counter = 0
        self._call_records: List[MCPCallRecord] = []
    
    def invoke(
        self,
        server: MCPServerConfig,
        tool_name: str,
        args: Dict[str, Any],
    ) -> str:
        """
        Invoke an MCP tool synchronously.
        
        Args:
            server: Server configuration
            tool_name: Name of the MCP tool to invoke
            args: Arguments for the tool
            
        Returns:
            JSON string with result or error
        """
        # Create call record
        self._call_counter += 1
        record = MCPCallRecord(
            id=f"mcp-{self._call_counter}-{int(time.time() * 1000)}",
            server_name=server.name,
            tool_name=tool_name,
            args=args,
            status="pending",
            started_at=time.time(),
        )
        self._call_records.append(record)
        self._notify(record)
        
        try:
            # Update status
            record.status = "running"
            self._notify(record)
            
            # Build MCP request
            mcp_request = {
                "jsonrpc": "2.0",
                "id": record.id,
                "method": "tools/call",
                "params": {
                    "name": tool_name,
                    "arguments": args,
                }
            }
            
            # Build command
            cmd = [server.command] + server.args
            
            # Build environment
            env = {**subprocess.os.environ, **server.env}
            
            # Execute
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                text=True,
            )
            
            stdout, stderr = process.communicate(
                input=json.dumps(mcp_request) + "\n",
                timeout=self.timeout_seconds,
            )
            
            if process.returncode != 0:
                error_msg = stderr.strip() or f"Process exited with code {process.returncode}"
                record.status = "error"
                record.error = error_msg
                record.completed_at = time.time()
                record.latency_ms = (record.completed_at - record.started_at) * 1000
                self._notify(record)
                return json.dumps({"status": "error", "error": error_msg})
            
            # Parse response
            try:
                response = json.loads(stdout.strip()) if stdout.strip() else {}
                result_str = json.dumps(response)
            except json.JSONDecodeError:
                result_str = stdout.strip()
            
            record.status = "success"
            record.result = result_str
            record.completed_at = time.time()
            record.latency_ms = (record.completed_at - record.started_at) * 1000
            self._notify(record)
            return result_str
            
        except subprocess.TimeoutExpired:
            record.status = "timeout"
            record.error = f"Timeout after {self.timeout_seconds}s"
            record.completed_at = time.time()
            record.latency_ms = (record.completed_at - record.started_at) * 1000
            self._notify(record)
            return json.dumps({"status": "timeout", "error": record.error})
            
        except FileNotFoundError:
            record.status = "error"
            record.error = f"Command not found: {server.command}"
            record.completed_at = time.time()
            record.latency_ms = (record.completed_at - record.started_at) * 1000
            self._notify(record)
            return json.dumps({"status": "error", "error": record.error})
            
        except Exception as e:
            record.status = "error"
            record.error = str(e)
            record.completed_at = time.time()
            record.latency_ms = (record.completed_at - record.started_at) * 1000
            self._notify(record)
            return json.dumps({"status": "error", "error": str(e)})
    
    def _notify(self, record: MCPCallRecord) -> None:
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


def create_tavily_tools(
    api_key: str,
    client: Optional[MCPClient] = None,
) -> List[BaseTool]:
    """
    Create Tavily search tools.
    
    Args:
        api_key: Tavily API key
        client: Optional MCP client for tracking
        
    Returns:
        List of LangChain tools for Tavily
    """
    if not api_key:
        return []
    
    client = client or MCPClient()
    
    server = MCPServerConfig(
        name="tavily",
        command="npx",
        args=["-y", "mcp-remote", f"https://mcp.tavily.com/mcp/?tavilyApiKey={api_key}"],
        description="Tavily Search API",
        enabled=True,
        is_hardcoded=True,
    )
    
    tools = []
    
    @tool
    def web_search(query: str, max_results: int = 5) -> str:
        """
        Search the web for information using Tavily.
        
        Args:
            query: Search query
            max_results: Maximum number of results (default 5)
            
        Returns:
            Search results with snippets and URLs
        """
        return client.invoke(server, "tavily_search", {
            "query": query,
            "max_results": max_results,
        })
    
    @tool
    def extract_content(urls: List[str]) -> str:
        """
        Extract content from web pages using Tavily.
        
        Args:
            urls: List of URLs to extract content from
            
        Returns:
            Extracted content from the URLs
        """
        return client.invoke(server, "tavily_extract", {"urls": urls})
    
    tools.append(web_search)
    tools.append(extract_content)
    
    return tools


class MCPToolManager:
    """
    Manages MCP tools for agents with per-agent enable/disable.
    
    Each agent instance can have different tools enabled based on
    runtime toggles in the UI.
    """
    
    def __init__(self, client: Optional[MCPClient] = None):
        self.client = client or MCPClient()
        self._servers: Dict[str, MCPServerConfig] = {}
        self._server_tools: Dict[str, List[BaseTool]] = {}
    
    def add_server(self, server: MCPServerConfig) -> None:
        """Add an MCP server configuration."""
        self._servers[server.name] = server
        if server.enabled:
            self._server_tools[server.name] = self._create_tools_for_server(server)
    
    def remove_server(self, name: str) -> bool:
        """Remove an MCP server."""
        if name in self._servers and not self._servers[name].is_hardcoded:
            del self._servers[name]
            if name in self._server_tools:
                del self._server_tools[name]
            return True
        return False
    
    def enable_server(self, name: str) -> bool:
        """Enable an MCP server."""
        if name in self._servers:
            self._servers[name].enabled = True
            self._server_tools[name] = self._create_tools_for_server(self._servers[name])
            return True
        return False
    
    def disable_server(self, name: str) -> bool:
        """Disable an MCP server (removes tools from active set)."""
        if name in self._servers:
            self._servers[name].enabled = False
            if name in self._server_tools:
                del self._server_tools[name]
            return True
        return False
    
    def get_enabled_tools(self) -> List[BaseTool]:
        """Get all tools from enabled servers."""
        tools = []
        for server_tools in self._server_tools.values():
            tools.extend(server_tools)
        return tools
    
    def get_tools_for_servers(self, server_names: Set[str]) -> List[BaseTool]:
        """Get tools only from specified enabled servers."""
        tools = []
        for name in server_names:
            if name in self._server_tools:
                tools.extend(self._server_tools[name])
        return tools
    
    def _create_tools_for_server(self, server: MCPServerConfig) -> List[BaseTool]:
        """Create LangChain tools for an MCP server."""
        if "tavily" in server.name.lower():
            # Tavily has special tool definitions
            return self._create_tavily_tools(server)
        else:
            # Generic MCP tool
            return [self._create_generic_tool(server)]
    
    def _create_tavily_tools(self, server: MCPServerConfig) -> List[BaseTool]:
        """Create Tavily-specific tools."""
        client = self.client
        
        @tool
        def web_search(query: str, max_results: int = 5) -> str:
            """Search the web for information using Tavily."""
            return client.invoke(server, "tavily_search", {
                "query": query,
                "max_results": max_results,
            })
        
        @tool
        def extract_content(urls: List[str]) -> str:
            """Extract content from web pages using Tavily."""
            return client.invoke(server, "tavily_extract", {"urls": urls})
        
        return [web_search, extract_content]
    
    def _create_generic_tool(self, server: MCPServerConfig) -> BaseTool:
        """Create a generic tool wrapper for an MCP server."""
        client = self.client
        server_copy = server
        
        @tool
        def mcp_invoke(tool_name: str, args: Dict[str, Any]) -> str:
            f"""
            Invoke a tool from the {server_copy.name} MCP server.
            
            Args:
                tool_name: Name of the tool to invoke
                args: Arguments for the tool
                
            Returns:
                Tool execution result
            """
            return client.invoke(server_copy, tool_name, args)
        
        # Rename the tool to be server-specific
        mcp_invoke.name = f"mcp_{server.name}"
        mcp_invoke.description = server.description or f"Tool from MCP server: {server.name}"
        
        return mcp_invoke
    
    def get_server_list(self) -> List[Dict[str, Any]]:
        """Get list of all servers with their status."""
        return [
            {
                "name": s.name,
                "description": s.description,
                "enabled": s.enabled,
                "is_hardcoded": s.is_hardcoded,
            }
            for s in self._servers.values()
        ]


def create_mcp_tools_for_agent(
    enabled_servers: Set[str],
    tavily_api_key: str,
    custom_servers: Optional[List[MCPServerConfig]] = None,
    client: Optional[MCPClient] = None,
) -> List[BaseTool]:
    """
    Create MCP tools for a specific agent instance.
    
    Only includes tools from servers that are in enabled_servers.
    This allows per-agent tool toggling without affecting other agents.
    
    Args:
        enabled_servers: Set of server names to enable for this agent
        tavily_api_key: Tavily API key for the hardcoded server
        custom_servers: Additional custom MCP servers
        client: Optional shared MCP client
        
    Returns:
        List of LangChain tools for the enabled servers
    """
    manager = MCPToolManager(client)
    
    # Add hardcoded Tavily server if API key provided
    if tavily_api_key and "tavily" in enabled_servers:
        manager.add_server(MCPServerConfig(
            name="tavily",
            command="npx",
            args=["-y", "mcp-remote", f"https://mcp.tavily.com/mcp/?tavilyApiKey={tavily_api_key}"],
            description="Tavily Search API - Web search and research",
            enabled=True,
            is_hardcoded=True,
        ))
    
    # Add custom servers
    if custom_servers:
        for server in custom_servers:
            if server.name in enabled_servers:
                # Skip hardcoded Tavily if we already added it with API key
                if server.name == "tavily" and tavily_api_key and server.is_hardcoded:
                    continue
                
                server.enabled = True
                manager.add_server(server)
    
    return manager.get_enabled_tools()
