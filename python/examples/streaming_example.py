#!/usr/bin/env python3
"""
Streaming Example: Demonstrates token streaming for all HGM agent types.

Usage:
    # With OpenAI provider
    python streaming_example.py --provider openai
    
    # With ZenMux provider
    python streaming_example.py --provider zenmux
    
    # With OpenRouter provider
    python streaming_example.py --provider openrouter

Requires:
    - OPENAI_API_KEY, ZENMUX_API_KEY, or OPENROUTER_API_KEY environment variable
    - langchain, langchain-core, openai packages installed
"""

import asyncio
import argparse
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from providers import OpenAICompatibleProvider, ProviderConfig, ProviderType
from hgm import (
    HGMAgent,
    DeepHGMAgentV2,
    HGMLangGraphAgent,
    create_deep_hgm_agent,
    create_hgm_langgraph_agent,
    create_streaming_agent,
    StreamEvent,
    StreamEventType,
)
from agents.orchestrator import DeepAgentOrchestrator


def get_provider(provider_type: str) -> OpenAICompatibleProvider:
    """Create provider based on type."""
    if provider_type == "openai":
        return OpenAICompatibleProvider.openai()
    elif provider_type == "zenmux":
        return OpenAICompatibleProvider.zenmux()
    elif provider_type == "openrouter":
        return OpenAICompatibleProvider.openrouter()
    else:
        raise ValueError(f"Unknown provider type: {provider_type}")


async def demo_provider_streaming(provider: OpenAICompatibleProvider):
    """Demo direct provider streaming."""
    print("\n" + "=" * 60)
    print("1. Direct Provider Streaming")
    print("=" * 60)
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Be concise."},
        {"role": "user", "content": "What is the capital of France? Answer in one sentence."},
    ]
    
    print("\nStreaming response: ", end="", flush=True)
    async for chunk in provider.astream(messages):
        print(chunk, end="", flush=True)
    print("\n")


async def demo_deep_hgm_streaming(provider: OpenAICompatibleProvider):
    """Demo DeepHGMAgentV2 streaming."""
    print("\n" + "=" * 60)
    print("2. DeepHGMAgentV2 Streaming")
    print("=" * 60)
    
    agent = create_deep_hgm_agent()
    
    input_data = {
        "messages": [{"role": "user", "content": "Explain what machine learning is in 2 sentences."}]
    }
    
    print("\nStreaming response: ", end="", flush=True)
    async for event in agent.astream(input_data, provider=provider):
        if event.get("type") == "token":
            print(event.get("content", ""), end="", flush=True)
        elif event.get("type") == "tool_start":
            print(f"\n[Tool: {event.get('content')}] ", end="", flush=True)
        elif event.get("type") == "metadata":
            metadata = event.get("metadata", {})
            print(f"\n\n[CMP: {metadata.get('cmp_score', 'N/A'):.4f}]")
    print()


async def demo_langgraph_streaming(provider: OpenAICompatibleProvider):
    """Demo HGMLangGraphAgent streaming."""
    print("\n" + "=" * 60)
    print("3. HGMLangGraphAgent Streaming")
    print("=" * 60)
    
    agent = create_hgm_langgraph_agent()
    
    input_data = {
        "user_input": "What are the three primary colors?"
    }
    
    print("\nStreaming response: ", end="", flush=True)
    async for event in agent.astream(input_data, provider=provider):
        if event.get("type") == "token":
            print(event.get("content", ""), end="", flush=True)
        elif event.get("type") == "metadata":
            metadata = event.get("metadata", {})
            print(f"\n\n[Agent ID: {metadata.get('agent_id', 'N/A')}]")
    print()


async def demo_orchestrator_streaming(provider: OpenAICompatibleProvider):
    """Demo DeepAgentOrchestrator streaming."""
    print("\n" + "=" * 60)
    print("4. DeepAgentOrchestrator Streaming")
    print("=" * 60)
    
    orchestrator = DeepAgentOrchestrator()
    
    print("\nStreaming response: ", end="", flush=True)
    async for event in orchestrator.astream_with_deep_hgm(
        agent_id="demo-agent",
        agent_name="Demo Agent",
        request="What is 2 + 2?",
        provider=provider,
    ):
        if event.get("type") == "token":
            print(event.get("content", ""), end="", flush=True)
        elif event.get("type") == "branch_selected":
            print(f"[Branch: {event.get('content')}] ", end="", flush=True)
        elif event.get("type") == "orchestrator_complete":
            metadata = event.get("metadata", {})
            print(f"\n\n[Clade CMP: {metadata.get('clade_cmp', 'N/A'):.4f}]")
    print()


async def demo_streaming_wrapper(provider: OpenAICompatibleProvider):
    """Demo StreamingAgentWrapper."""
    print("\n" + "=" * 60)
    print("5. StreamingAgentWrapper (Universal)")
    print("=" * 60)
    
    # Create any agent
    base_agent = create_deep_hgm_agent()
    
    # Wrap with streaming
    streaming_agent = create_streaming_agent(base_agent, provider=provider)
    
    input_data = {
        "messages": [{"role": "user", "content": "Name three planets in our solar system."}]
    }
    
    print("\nStreaming text only: ", end="", flush=True)
    async for text in streaming_agent.astream_text(input_data):
        print(text, end="", flush=True)
    print("\n")


async def demo_text_only_streaming(provider: OpenAICompatibleProvider):
    """Demo text-only streaming (convenience method)."""
    print("\n" + "=" * 60)
    print("6. Text-Only Streaming (astream_text)")
    print("=" * 60)
    
    agent = create_deep_hgm_agent()
    
    input_data = {
        "messages": [{"role": "user", "content": "Say hello in three languages."}]
    }
    
    print("\nStreaming: ", end="", flush=True)
    async for text in agent.astream_text(input_data, provider=provider):
        print(text, end="", flush=True)
    print("\n")


async def main():
    parser = argparse.ArgumentParser(description="HGM Streaming Demo")
    parser.add_argument(
        "--provider",
        choices=["openai", "zenmux", "openrouter"],
        default="zenmux",
        help="Provider to use for streaming",
    )
    parser.add_argument(
        "--demo",
        choices=["all", "provider", "deep_hgm", "langgraph", "orchestrator", "wrapper", "text"],
        default="all",
        help="Which demo to run",
    )
    args = parser.parse_args()
    
    # Get provider
    try:
        provider = get_provider(args.provider)
        print(f"Using provider: {args.provider}")
        print(f"Model: {provider.config.model}")
    except ValueError as e:
        print(f"Error: {e}")
        return
    
    # Run demos
    demos = {
        "provider": demo_provider_streaming,
        "deep_hgm": demo_deep_hgm_streaming,
        "langgraph": demo_langgraph_streaming,
        "orchestrator": demo_orchestrator_streaming,
        "wrapper": demo_streaming_wrapper,
        "text": demo_text_only_streaming,
    }
    
    if args.demo == "all":
        for name, demo_fn in demos.items():
            try:
                await demo_fn(provider)
            except Exception as e:
                print(f"\nError in {name} demo: {e}")
    else:
        await demos[args.demo](provider)
    
    print("\n" + "=" * 60)
    print("Streaming demos complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
