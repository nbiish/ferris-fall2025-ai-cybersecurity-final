#!/usr/bin/env python3
"""
Self-Improving HGM Agent Example.

Demonstrates the complete Huxley-Gödel Machine + Memori integration:
- One-prompt agent activation
- Session-based workspace isolation
- Memori memory system
- CMP-based self-improvement
- IDE-accessible agent workspaces

Reference:
- https://github.com/metauto-ai/HGM (Self-improving architecture)
- https://github.com/MemoriLabs/Memori (Efficient agent memory)

Usage:
    python -m examples.self_improving_agent_example
"""

import os
import sys
import asyncio
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from hgm import (
    create_self_improving_agent,
    resume_agent,
    list_agents,
    SelfImprovingHGMAgent,
    AgentConfig,
)


def example_basic_usage():
    """
    Basic usage: Create and use a self-improving agent with one call.
    """
    print("\n" + "=" * 60)
    print("Example 1: Basic One-Prompt Activation")
    print("=" * 60)
    
    # Get workspace path (use temp directory for example)
    workspace = os.path.join(os.path.dirname(__file__), "..", "..", "workspace")
    os.makedirs(workspace, exist_ok=True)
    
    # One line creates and activates a fully configured self-improving agent
    agent = create_self_improving_agent(
        name="CodeReviewer",
        workspace_path=workspace,
        model="gpt-4",
        enable_memori=True,
        enable_evolution=True,
    )
    
    print(f"\n✅ Agent activated!")
    print(f"   Session ID: {agent.session_id}")
    print(f"   Agent ID: {agent.agent_id}")
    print(f"   Generation: {agent.generation}")
    print(f"   CMP Score: {agent.cmp_score:.4f}")
    
    # Get workspace paths for IDE access
    paths = agent.get_workspace_paths()
    print(f"\n📁 Agent Workspace (open in IDE):")
    for name, path in paths.items():
        print(f"   {name}: {path}")
    
    # Execute a task
    print("\n🚀 Executing task...")
    result = agent.invoke("Analyze this code for potential issues: def add(a, b): return a + b")
    
    print(f"\n📊 Result:")
    print(f"   Success: {result['success']}")
    print(f"   CMP Score: {result['cmp_score']:.4f}")
    print(f"   Evolution Triggered: {result['evolution_triggered']}")
    print(f"\n   Response: {result['response'][:200]}...")
    
    # Check memory
    memories = agent.recall_memories(limit=3)
    print(f"\n🧠 Agent Memories ({len(memories)} total):")
    for mem in memories[:3]:
        print(f"   [{mem['memory_type']}] {mem['content'][:60]}...")
    
    # Deactivate (pause, don't complete)
    agent.deactivate(complete=False)
    print("\n⏸️  Agent paused (session preserved)")
    
    return agent.session_id, agent.agent_id


def example_resume_session(session_id: str, workspace: str):
    """
    Resume an existing agent session.
    """
    print("\n" + "=" * 60)
    print("Example 2: Resume Existing Session")
    print("=" * 60)
    
    # Resume the agent
    agent = resume_agent(
        workspace_path=workspace,
        session_id=session_id,
    )
    
    if agent:
        print(f"\n✅ Agent resumed!")
        print(f"   Session ID: {agent.session_id}")
        print(f"   Generation: {agent.generation}")
        print(f"   Previous tasks: {agent.get_session_info()['total_tasks']}")
        
        # Continue working
        result = agent.invoke("What improvements would you suggest for the previous code?")
        print(f"\n   Response: {result['response'][:150]}...")
        
        agent.deactivate()
    else:
        print(f"\n❌ Session {session_id} not found")


def example_multiple_agents():
    """
    Create multiple agents with isolated workspaces.
    """
    print("\n" + "=" * 60)
    print("Example 3: Multiple Isolated Agents")
    print("=" * 60)
    
    workspace = os.path.join(os.path.dirname(__file__), "..", "..", "workspace")
    
    # Create multiple specialized agents
    agents = []
    agent_configs = [
        ("SecurityAnalyzer", "Analyze code for security vulnerabilities"),
        ("PerformanceOptimizer", "Optimize code for better performance"),
        ("DocumentationWriter", "Generate documentation for code"),
    ]
    
    for name, task in agent_configs:
        agent = create_self_improving_agent(
            name=name,
            workspace_path=workspace,
            enable_memori=True,
        )
        
        print(f"\n🤖 Created {name}:")
        print(f"   Agent ID: {agent.agent_id}")
        print(f"   Workspace: {agent.workspace_path}")
        
        # Each agent works independently
        result = agent.invoke(f"{task}: def factorial(n): return 1 if n <= 1 else n * factorial(n-1)")
        print(f"   Task result: {'✓' if result['success'] else '✗'}")
        
        agents.append(agent)
    
    # List all agents in workspace
    print(f"\n📋 All agents in workspace:")
    all_agents = list_agents(workspace)
    for info in all_agents:
        print(f"   - {info['agent_name']} (Gen {info['generation']}, CMP: {info['cmp_score']:.3f})")
    
    # Cleanup
    for agent in agents:
        agent.deactivate()


async def example_streaming():
    """
    Stream responses from the agent.
    """
    print("\n" + "=" * 60)
    print("Example 4: Streaming Responses")
    print("=" * 60)
    
    workspace = os.path.join(os.path.dirname(__file__), "..", "..", "workspace")
    
    agent = create_self_improving_agent(
        name="StreamingAgent",
        workspace_path=workspace,
    )
    
    print(f"\n🌊 Streaming response...")
    print("-" * 40)
    
    async for text in agent.astream_text("Explain recursion in 2 sentences."):
        print(text, end="", flush=True)
    
    print("\n" + "-" * 40)
    
    agent.deactivate()


def example_evolution_tracking():
    """
    Demonstrate agent evolution and generation tracking.
    """
    print("\n" + "=" * 60)
    print("Example 5: Evolution and Generation Tracking")
    print("=" * 60)
    
    workspace = os.path.join(os.path.dirname(__file__), "..", "..", "workspace")
    
    # Create agent with lower evolution threshold for demo
    config = AgentConfig(
        name="EvolvingAgent",
        workspace_path=workspace,
        enable_evolution=True,
        evolution_threshold=0.9,  # High threshold = evolve more often
        max_generations=5,
    )
    
    agent = SelfImprovingHGMAgent(config)
    agent.activate()
    
    print(f"\n🧬 Starting agent (Gen {agent.generation})")
    
    # Run multiple tasks to trigger evolution
    tasks = [
        "Write a function to reverse a string",
        "Write a function to check if a number is prime",
        "Write a function to find the factorial",
        "Write a function to sort a list",
    ]
    
    for i, task in enumerate(tasks):
        print(f"\n   Task {i+1}: {task[:40]}...")
        result = agent.invoke(task)
        print(f"   Gen: {result['generation']}, CMP: {result['cmp_score']:.4f}, Evolved: {result['evolution_triggered']}")
    
    # Check generation history
    info = agent.get_session_info()
    print(f"\n📊 Final Stats:")
    print(f"   Generation: {info['generation']}")
    print(f"   CMP Score: {info['cmp_score']:.4f}")
    print(f"   Total Tasks: {info['total_tasks']}")
    print(f"   Success Rate: {info['success_rate']:.1%}")
    
    agent.deactivate()


def example_memory_operations():
    """
    Demonstrate Memori memory operations.
    """
    print("\n" + "=" * 60)
    print("Example 6: Memori Memory Operations")
    print("=" * 60)
    
    workspace = os.path.join(os.path.dirname(__file__), "..", "..", "workspace")
    
    agent = create_self_improving_agent(
        name="MemoryAgent",
        workspace_path=workspace,
        enable_memori=True,
        memori_max_context=15,
    )
    
    # Manually add memories
    print("\n📝 Adding memories...")
    agent.add_memory("User prefers Python over JavaScript", "preferences")
    agent.add_memory("Learned to use list comprehensions effectively", "skills")
    agent.add_memory("Project uses FastAPI for the backend", "facts")
    
    # Recall memories
    print("\n🔍 Recalling memories...")
    
    # All memories
    all_memories = agent.recall_memories(limit=10)
    print(f"\n   All memories ({len(all_memories)}):")
    for mem in all_memories:
        print(f"   - [{mem['memory_type']}] {mem['content'][:50]}...")
    
    # Search specific memories
    python_memories = agent.recall_memories(query="Python", limit=5)
    print(f"\n   Python-related ({len(python_memories)}):")
    for mem in python_memories:
        print(f"   - {mem['content'][:50]}...")
    
    agent.deactivate()


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("Self-Improving HGM Agent + Memori Integration Examples")
    print("=" * 60)
    print("\nThis demonstrates the complete integration of:")
    print("  - metauto-ai/HGM: Self-improving agent architecture")
    print("  - MemoriLabs/Memori: SQL-native memory layer")
    print("  - Session-based workspace isolation")
    print("  - One-prompt agent activation")
    
    workspace = os.path.join(os.path.dirname(__file__), "..", "..", "workspace")
    
    # Run examples
    try:
        # Example 1: Basic usage
        session_id, agent_id = example_basic_usage()
        
        # Example 2: Resume session
        example_resume_session(session_id, workspace)
        
        # Example 3: Multiple agents
        example_multiple_agents()
        
        # Example 4: Streaming (async)
        asyncio.run(example_streaming())
        
        # Example 5: Evolution tracking
        example_evolution_tracking()
        
        # Example 6: Memory operations
        example_memory_operations()
        
        print("\n" + "=" * 60)
        print("✅ All examples completed!")
        print("=" * 60)
        print(f"\n📁 Check agent workspaces at: {workspace}/agents/")
        print("   Each agent folder can be opened in your IDE to inspect:")
        print("   - outputs/     : Generated code and artifacts")
        print("   - generations/ : Evolution history")
        print("   - documents/   : Agent-specific documents")
        print("   - memori.db    : Memory database")
        print("   - session.json : Session state")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
