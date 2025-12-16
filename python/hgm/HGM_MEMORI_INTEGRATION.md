# HGM + Memori Integration

**Self-Improving Agents with Efficient Memory**

This integration combines the [Huxley-Gödel Machine](https://github.com/metauto-ai/HGM) self-improving agent architecture with [MemoriLabs/Memori](https://github.com/MemoriLabs/Memori) SQL-native memory layer for production-ready AI agents.

## Overview

### Key Features

- **One-Prompt Activation**: Single call creates fully configured self-improving agent
- **Session Isolation**: Each agent runs in dedicated workspace with unique session ID
- **Memori Memory**: SQL-native memory layer with entity/process attribution
- **Self-Improvement**: CMP-based evolution through generations
- **IDE Integration**: Agent workspaces can be opened directly in your IDE

### Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    SelfImprovingHGMAgent                            │
├─────────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                 │
│  │   Planner   │  │  Executor   │  │  Evaluator  │                 │
│  │  (DeepAgent)│  │  (Tools)    │  │  (HGM CMP)  │                 │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘                 │
│         │                │                │                         │
│  ┌──────▼────────────────▼────────────────▼──────┐                 │
│  │              Session Manager                   │                 │
│  │  (Workspace isolation, Generation tracking)   │                 │
│  └──────────────────────┬────────────────────────┘                 │
│                         │                                           │
│  ┌──────────────────────▼────────────────────────┐                 │
│  │              Memori Memory Layer               │                 │
│  │  (Facts, Skills, Events, Relationships)       │                 │
│  └──────────────────────┬────────────────────────┘                 │
│                         │                                           │
│  ┌──────────────────────▼────────────────────────┐                 │
│  │           Self-Improvement Engine              │                 │
│  │  UCB-Air (branch) + Thompson (action)         │                 │
│  └───────────────────────────────────────────────┘                 │
└─────────────────────────────────────────────────────────────────────┘
```

### Workspace Structure (Per Agent)

```
{workspace}/agents/{agent_id}/
├── memori.db           # Memori SQL memory database
├── checkpoints.db      # Conversation persistence
├── config.json         # Agent configuration
├── session.json        # Current session state
├── generations/        # Evolution history
│   ├── gen_0/          # Initial generation
│   │   └── record.json
│   ├── gen_1/          # First evolution
│   └── ...
├── documents/          # Agent-specific documents
├── outputs/            # Generated code/artifacts
├── voice/              # Voice recordings
└── logs/               # Execution logs
```

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### One-Prompt Activation

```python
from hgm import create_self_improving_agent

# One line creates and activates a fully configured agent
agent = create_self_improving_agent(
    name="CodeReviewer",
    workspace_path="/path/to/workspace",
    model="gpt-4",
    enable_memori=True,
    enable_evolution=True,
)

# Execute tasks - agent learns and improves over time
result = agent.invoke("Review this code for security issues: ...")

# Access workspace in IDE
print(agent.get_workspace_paths())
# {'root': '/path/to/workspace/agents/abc123',
#  'outputs': '/path/to/workspace/agents/abc123/outputs', ...}
```

### Resume Existing Agent

```python
from hgm import resume_agent, list_agents

# List all agents in workspace
agents = list_agents("/path/to/workspace")
for info in agents:
    print(f"{info['agent_name']} - Gen {info['generation']}")

# Resume by session ID
agent = resume_agent(
    workspace_path="/path/to/workspace",
    session_id="abc-123-def-456",
)

# Or resume by agent ID (uses latest session)
agent = resume_agent(
    workspace_path="/path/to/workspace",
    agent_id="agent-uuid",
)
```

### Streaming Responses

```python
import asyncio

async def stream_example():
    agent = create_self_improving_agent(name="StreamAgent", workspace_path="...")
    
    async for text in agent.astream_text("Explain recursion briefly"):
        print(text, end="", flush=True)

asyncio.run(stream_example())
```

## Core Concepts

### Session Management

Each agent session provides:
- **Unique Session ID**: Track work across invocations
- **Isolated Workspace**: Dedicated folder structure
- **Generation Tracking**: Evolution history preserved
- **CMP Metrics**: Performance tracking over time

```python
# Get session info
info = agent.get_session_info()
# {
#     'session_id': 'uuid-...',
#     'agent_id': 'uuid-...',
#     'generation': 2,
#     'cmp_score': 0.75,
#     'total_tasks': 15,
#     'success_rate': 0.8,
#     'workspace': '/path/to/agents/uuid-...',
# }
```

### Memori Memory System

The Memori integration provides multi-level memory:

| Memory Type | Description | Use Case |
|-------------|-------------|----------|
| `facts` | Factual information | Project details, requirements |
| `preferences` | User/system preferences | Coding style, language choice |
| `events` | Actions and occurrences | Task completions, errors |
| `skills` | Learned capabilities | Tool usage, patterns |
| `relationships` | Entity connections | Team members, dependencies |
| `rules` | Behavioral guidelines | Code standards, policies |

```python
# Add memories manually
agent.add_memory("User prefers async Python", "preferences")
agent.add_memory("Learned effective error handling", "skills")

# Recall memories
memories = agent.recall_memories(query="Python", limit=10)
for mem in memories:
    print(f"[{mem['memory_type']}] {mem['content']}")
```

### Self-Improvement (HGM)

The agent self-improves through:

1. **CMP (Clade Metaproductivity)**: Measures long-term productivity
2. **UCB-Air**: Selects which branches (strategies) to explore
3. **Thompson Sampling**: Selects actions within a branch
4. **Evolution**: Creates new generation when CMP indicates room for improvement

```python
# Configure evolution behavior
from hgm import AgentConfig, SelfImprovingHGMAgent

config = AgentConfig(
    name="EvolvingAgent",
    workspace_path="/workspace",
    enable_evolution=True,
    evolution_threshold=0.7,  # Evolve when CMP < 0.7
    max_generations=10,
)

agent = SelfImprovingHGMAgent(config)
agent.activate()
```

## Advanced Usage

### Custom Middleware

Add custom middleware to the agent:

```python
from hgm import (
    SelfImprovingHGMAgent,
    AgentConfig,
    MemoriMiddleware,
    MCPToolMiddleware,
    HumanInTheLoopMiddleware,
)

config = AgentConfig(
    name="CustomAgent",
    workspace_path="/workspace",
    enable_human_in_loop=True,  # Require approval for dangerous actions
)

agent = SelfImprovingHGMAgent(config)
agent.activate()
```

### Multiple Specialized Agents

Create specialized agents for different tasks:

```python
# Create multiple agents in same workspace
agents = {}
for role in ["SecurityAnalyzer", "PerformanceOptimizer", "DocWriter"]:
    agents[role] = create_self_improving_agent(
        name=role,
        workspace_path="/workspace",
    )

# Each agent has isolated session and memory
result = agents["SecurityAnalyzer"].invoke("Check for SQL injection...")
result = agents["PerformanceOptimizer"].invoke("Optimize this query...")
```

### Integration with LangChain

```python
from langchain_openai import ChatOpenAI
from hgm import create_self_improving_agent

# Use with LangChain chat model for streaming
chat_model = ChatOpenAI(model="gpt-4", streaming=True)

agent = create_self_improving_agent(
    name="LangChainAgent",
    workspace_path="/workspace",
)

# Stream with LangChain model
async for event in agent.astream("Task...", chat_model=chat_model):
    if event["type"] == "token":
        print(event["content"], end="")
```

## API Reference

### `create_self_improving_agent()`

Main entry point for creating agents.

```python
def create_self_improving_agent(
    name: str,
    workspace_path: str,
    model: str = "gpt-4",
    tools: Optional[List[BaseTool]] = None,
    enable_memori: bool = True,
    enable_evolution: bool = True,
    **kwargs,
) -> SelfImprovingHGMAgent
```

### `SelfImprovingHGMAgent`

Main agent class.

| Method | Description |
|--------|-------------|
| `activate()` | Activate agent with session |
| `invoke(prompt)` | Execute task synchronously |
| `astream(prompt)` | Stream response asynchronously |
| `astream_text(prompt)` | Stream text tokens only |
| `recall_memories(query, limit)` | Search memories |
| `add_memory(content, type)` | Add memory entry |
| `get_session_info()` | Get session state |
| `get_workspace_paths()` | Get workspace paths for IDE |
| `deactivate(complete)` | Pause or complete session |

### `HGMSessionManager`

Manages agent sessions and workspaces.

| Method | Description |
|--------|-------------|
| `create_session(name, config)` | Create new session |
| `resume_session(session_id)` | Resume existing session |
| `resume_agent(agent_id)` | Resume agent's latest session |
| `evolve_agent(session_id, ...)` | Trigger agent evolution |
| `list_agents()` | List all agents |
| `list_sessions()` | List all sessions |
| `get_agent_workspace_info(agent_id)` | Get workspace paths |

### `MemoriMiddleware`

Memory layer integration.

| Method | Description |
|--------|-------------|
| `configure_for_agent(agent_id, ...)` | Configure for specific agent |
| `new_session()` | Start new memory session |
| `add_memory(content, type, ...)` | Add memory |
| `add_fact(content)` | Add fact memory |
| `add_skill(content)` | Add skill memory |
| `add_triple(s, p, o)` | Add semantic triple |
| `recall(query, limit)` | Search memories |
| `get_knowledge_graph()` | Get semantic triples |

## References

- [Huxley-Gödel Machine Paper](https://arxiv.org/abs/2510.21614)
- [metauto-ai/HGM Repository](https://github.com/metauto-ai/HGM)
- [MemoriLabs/Memori Repository](https://github.com/MemoriLabs/Memori)
- [LangChain DeepAgents](https://python.langchain.com/)
