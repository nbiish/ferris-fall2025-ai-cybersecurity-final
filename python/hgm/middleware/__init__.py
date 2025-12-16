"""
DeepHGM Middleware Package.

Implements LangChain DeepAgents-style middleware with HGM self-improvement:
- HumanInTheLoopMiddleware: Interrupt/resume for human approval
- MCPToolMiddleware: MCP server tool integration
- CLIToolMiddleware: CLI tool (qwen, gemini) integration
- TodoListMiddleware: Task planning and tracking
- FilesystemMiddleware: File operations and context offloading
- MemoriMiddleware: SQL-native memory layer (MemoriLabs/Memori integration)
"""

from .base import DeepHGMMiddleware, MiddlewareHook, InterruptRequest, InterruptDecision, MiddlewareChain
from .hitl import HumanInTheLoopMiddleware, HITLConfig, DecisionType
from .mcp_tools import MCPToolMiddleware, MCPServerConfig, MCPCallStatus, MCPCallRecord
from .cli_tools import CLIToolMiddleware, CLIProvider, ExecutionMode, CLIConfig, CLIExecutionRecord
from .todo_list import TodoListMiddleware
from .filesystem import FilesystemMiddleware
from .memori import (
    MemoriMiddleware,
    MemoriConfig,
    MemoriStorageBackend,
    MemoryEntry,
    create_memori_middleware,
)

__all__ = [
    # Base
    "DeepHGMMiddleware",
    "MiddlewareHook",
    "InterruptRequest",
    "InterruptDecision",
    "MiddlewareChain",
    # HITL
    "HumanInTheLoopMiddleware",
    "HITLConfig",
    "DecisionType",
    # MCP
    "MCPToolMiddleware",
    "MCPServerConfig",
    "MCPCallStatus",
    "MCPCallRecord",
    # CLI
    "CLIToolMiddleware",
    "CLIProvider",
    "ExecutionMode",
    "CLIConfig",
    "CLIExecutionRecord",
    # Planning
    "TodoListMiddleware",
    # Filesystem
    "FilesystemMiddleware",
    # Memori (MemoriLabs/Memori integration)
    "MemoriMiddleware",
    "MemoriConfig",
    "MemoriStorageBackend",
    "MemoryEntry",
    "create_memori_middleware",
]
