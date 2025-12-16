from .agent import HGMAgent
from .evolution import HGMEvolutionManager
from .langgraph_agent import HGMLangGraphAgent, HGMAgentState, HGMToolConfig, create_hgm_langgraph_agent
from .persistence import (
    HGMCheckpointer,
    AgentCheckpoint,
    AgentFolderManager,
    create_persistent_agent_manager,
)
from .deep_hgm_agent import (
    DeepHGMAgent,
    DeepHGMAgentV2,
    DeepHGMState,
    DeepHGMMiddleware,
    PlanningMiddleware,
    SelfImprovementMiddleware,
    UCBAir,
    ThompsonSampler,
    CladeMetrics,
    ActionType,
    create_deep_hgm_agent,
)
from .completion_loop import (
    CompletionStatus,
    LoopAction,
    CompletionCriteria,
    LoopState,
    CMPEvaluator,
    HGMCompletionLoop,
    create_completion_loop,
)
from .streaming import (
    StreamEventType,
    StreamEvent,
    AsyncStreamingCallbackHandler,
    StreamingMixin,
    StreamingAgentWrapper,
    stream_openai_provider,
    stream_langchain_chat,
    create_streaming_agent,
)

# Session management (HGM self-improving workspace isolation)
from .session_manager import (
    HGMSessionManager,
    SessionState,
    SessionStatus,
    GenerationRecord,
    create_session_manager,
)

# Self-improving agent (One-prompt activation with Memori integration)
from .self_improving_agent import (
    SelfImprovingHGMAgent,
    AgentConfig,
    create_self_improving_agent,
    resume_agent,
    list_agents,
)

# Middleware exports
from .middleware import (
    HumanInTheLoopMiddleware,
    HITLConfig,
    DecisionType,
    MCPToolMiddleware,
    MCPServerConfig,
    MCPCallStatus,
    MCPCallRecord,
    CLIToolMiddleware,
    CLIProvider,
    ExecutionMode,
    CLIConfig,
    TodoListMiddleware,
    FilesystemMiddleware,
    InterruptRequest,
    InterruptDecision,
    MiddlewareChain,
    # Memori integration
    MemoriMiddleware,
    MemoriConfig,
    MemoriStorageBackend,
    MemoryEntry,
    create_memori_middleware,
)

__all__ = [
    "HGMAgent",
    "HGMEvolutionManager",
    "HGMLangGraphAgent",
    "HGMAgentState",
    "HGMToolConfig",
    "create_hgm_langgraph_agent",
    # Persistence
    "HGMCheckpointer",
    "AgentCheckpoint",
    "AgentFolderManager",
    "create_persistent_agent_manager",
    # DeepHGM Chimera
    "DeepHGMAgent",
    "DeepHGMAgentV2",
    "DeepHGMState",
    "DeepHGMMiddleware",
    "PlanningMiddleware",
    "SelfImprovementMiddleware",
    "UCBAir",
    "ThompsonSampler",
    "CladeMetrics",
    "ActionType",
    "create_deep_hgm_agent",
    # Completion Loop
    "CompletionStatus",
    "LoopAction",
    "CompletionCriteria",
    "LoopState",
    "CMPEvaluator",
    "HGMCompletionLoop",
    "create_completion_loop",
    # Middleware
    "HumanInTheLoopMiddleware",
    "HITLConfig",
    "DecisionType",
    "MCPToolMiddleware",
    "MCPServerConfig",
    "MCPCallStatus",
    "MCPCallRecord",
    "CLIToolMiddleware",
    "CLIProvider",
    "ExecutionMode",
    "CLIConfig",
    "TodoListMiddleware",
    "FilesystemMiddleware",
    "InterruptRequest",
    "InterruptDecision",
    "MiddlewareChain",
    # Streaming
    "StreamEventType",
    "StreamEvent",
    "AsyncStreamingCallbackHandler",
    "StreamingMixin",
    "StreamingAgentWrapper",
    "stream_openai_provider",
    "stream_langchain_chat",
    "create_streaming_agent",
    # Session Management (HGM workspace isolation)
    "HGMSessionManager",
    "SessionState",
    "SessionStatus",
    "GenerationRecord",
    "create_session_manager",
    # Self-Improving Agent (One-prompt activation)
    "SelfImprovingHGMAgent",
    "AgentConfig",
    "create_self_improving_agent",
    "resume_agent",
    "list_agents",
    # Memori Integration (MemoriLabs/Memori)
    "MemoriMiddleware",
    "MemoriConfig",
    "MemoriStorageBackend",
    "MemoryEntry",
    "create_memori_middleware",
]
