"""
HGM Streaming Support: Token-level streaming for all providers and agents.

Purpose: Enable real-time token streaming for DeepHGM agents across all providers
Supports: OpenAI, ZenMux, OpenRouter, and any OpenAI-compatible endpoint

Architecture:
┌─────────────────────────────────────────────────────────────────────┐
│                    Streaming Pipeline                                │
├─────────────────────────────────────────────────────────────────────┤
│  Provider (OpenAI SDK)  →  StreamingCallback  →  AsyncIterator      │
│         ↓                        ↓                    ↓             │
│  stream=True             on_llm_new_token      yield chunk          │
└─────────────────────────────────────────────────────────────────────┘
"""

from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Union
from dataclasses import dataclass, field
import asyncio
import json
from enum import Enum

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.messages import BaseMessage, AIMessage, AIMessageChunk


class StreamEventType(Enum):
    """Types of streaming events."""
    TOKEN = "token"
    TOOL_START = "tool_start"
    TOOL_END = "tool_end"
    AGENT_ACTION = "agent_action"
    AGENT_FINISH = "agent_finish"
    ERROR = "error"
    METADATA = "metadata"


@dataclass
class StreamEvent:
    """A streaming event with type and content."""
    
    event_type: StreamEventType
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.event_type.value,
            "content": self.content,
            "metadata": self.metadata,
        }
    
    def to_sse(self) -> str:
        """Format as Server-Sent Event."""
        return f"data: {json.dumps(self.to_dict())}\n\n"


class AsyncStreamingCallbackHandler(AsyncCallbackHandler):
    """
    Async callback handler for LangChain streaming.
    
    Collects tokens and pushes them to an async queue for consumption.
    Compatible with ChatOpenAI, ChatAnthropic, and other LangChain chat models.
    """
    
    def __init__(self):
        self.queue: asyncio.Queue[Optional[StreamEvent]] = asyncio.Queue()
        self.done = asyncio.Event()
        self._error: Optional[Exception] = None
    
    async def on_llm_new_token(self, token: str, **kwargs) -> None:
        """Called when a new token is generated."""
        await self.queue.put(StreamEvent(
            event_type=StreamEventType.TOKEN,
            content=token,
        ))
    
    async def on_llm_end(self, response, **kwargs) -> None:
        """Called when LLM finishes."""
        self.done.set()
        await self.queue.put(None)  # Signal end
    
    async def on_llm_error(self, error: Exception, **kwargs) -> None:
        """Called on LLM error."""
        self._error = error
        await self.queue.put(StreamEvent(
            event_type=StreamEventType.ERROR,
            content=str(error),
        ))
        self.done.set()
        await self.queue.put(None)
    
    async def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        **kwargs,
    ) -> None:
        """Called when a tool starts."""
        await self.queue.put(StreamEvent(
            event_type=StreamEventType.TOOL_START,
            content=serialized.get("name", "unknown"),
            metadata={"input": input_str[:200]},
        ))
    
    async def on_tool_end(self, output: str, **kwargs) -> None:
        """Called when a tool ends."""
        await self.queue.put(StreamEvent(
            event_type=StreamEventType.TOOL_END,
            content=output[:500] if output else "",
        ))
    
    async def on_agent_action(self, action, **kwargs) -> None:
        """Called on agent action."""
        await self.queue.put(StreamEvent(
            event_type=StreamEventType.AGENT_ACTION,
            content=str(action.tool) if hasattr(action, 'tool') else str(action),
            metadata={"tool_input": str(action.tool_input)[:200] if hasattr(action, 'tool_input') else ""},
        ))
    
    async def on_agent_finish(self, finish, **kwargs) -> None:
        """Called when agent finishes."""
        await self.queue.put(StreamEvent(
            event_type=StreamEventType.AGENT_FINISH,
            content=str(finish.return_values) if hasattr(finish, 'return_values') else str(finish),
        ))
    
    async def __aiter__(self) -> AsyncIterator[StreamEvent]:
        """Iterate over stream events."""
        while True:
            if self.done.is_set() and self.queue.empty():
                break
            try:
                event = await asyncio.wait_for(self.queue.get(), timeout=60.0)
                if event is None:
                    break
                yield event
            except asyncio.TimeoutError:
                break
    
    @property
    def error(self) -> Optional[Exception]:
        return self._error


class StreamingMixin:
    """
    Mixin class to add streaming capabilities to HGM agents.
    
    Add this to DeepHGMAgent, DeepHGMAgentV2, or HGMLangGraphAgent
    to enable token streaming.
    """
    
    async def astream(
        self,
        input_data: Dict[str, Any],
        thread_id: Optional[str] = None,
    ) -> AsyncIterator[StreamEvent]:
        """
        Stream agent response token by token.
        
        Args:
            input_data: Dict with messages
            thread_id: Optional thread ID for state persistence
            
        Yields:
            StreamEvent objects with tokens and metadata
        """
        raise NotImplementedError("Subclass must implement astream")
    
    async def astream_text(
        self,
        input_data: Dict[str, Any],
        thread_id: Optional[str] = None,
    ) -> AsyncIterator[str]:
        """
        Stream just the text content.
        
        Args:
            input_data: Dict with messages
            thread_id: Optional thread ID
            
        Yields:
            Token strings
        """
        async for event in self.astream(input_data, thread_id):
            if event.event_type == StreamEventType.TOKEN:
                yield event.content


async def stream_openai_provider(
    provider,
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    temperature: float = 0.7,
    **kwargs,
) -> AsyncIterator[StreamEvent]:
    """
    Stream from OpenAICompatibleProvider.
    
    Args:
        provider: OpenAICompatibleProvider instance
        messages: Chat messages
        model: Optional model override
        temperature: Sampling temperature
        
    Yields:
        StreamEvent objects
    """
    try:
        async for chunk in provider.astream(
            messages=messages,
            model=model,
            temperature=temperature,
            **kwargs,
        ):
            yield StreamEvent(
                event_type=StreamEventType.TOKEN,
                content=chunk,
            )
    except Exception as e:
        yield StreamEvent(
            event_type=StreamEventType.ERROR,
            content=str(e),
        )


async def stream_langchain_chat(
    chat_model,
    messages: List[BaseMessage],
    **kwargs,
) -> AsyncIterator[StreamEvent]:
    """
    Stream from any LangChain chat model with streaming support.
    
    Args:
        chat_model: LangChain chat model (ChatOpenAI, etc.)
        messages: List of BaseMessage objects
        
    Yields:
        StreamEvent objects
    """
    try:
        async for chunk in chat_model.astream(messages, **kwargs):
            if isinstance(chunk, AIMessageChunk):
                content = chunk.content
                if content:
                    yield StreamEvent(
                        event_type=StreamEventType.TOKEN,
                        content=content,
                    )
            elif hasattr(chunk, 'content') and chunk.content:
                yield StreamEvent(
                    event_type=StreamEventType.TOKEN,
                    content=chunk.content,
                )
    except Exception as e:
        yield StreamEvent(
            event_type=StreamEventType.ERROR,
            content=str(e),
        )


class StreamingAgentWrapper:
    """
    Wrapper to add streaming to any HGM agent.
    
    Usage:
        agent = DeepHGMAgentV2(...)
        streaming_agent = StreamingAgentWrapper(agent, provider)
        
        async for event in streaming_agent.astream({"messages": [...]}):
            print(event.content, end="", flush=True)
    """
    
    def __init__(
        self,
        agent,
        provider=None,
        chat_model=None,
    ):
        """
        Initialize streaming wrapper.
        
        Args:
            agent: HGM agent instance (DeepHGMAgent, DeepHGMAgentV2, etc.)
            provider: Optional OpenAICompatibleProvider for streaming
            chat_model: Optional LangChain chat model for streaming
        """
        self.agent = agent
        self.provider = provider
        self.chat_model = chat_model
    
    async def astream(
        self,
        input_data: Dict[str, Any],
        thread_id: Optional[str] = None,
    ) -> AsyncIterator[StreamEvent]:
        """
        Stream agent response.
        
        Uses provider or chat_model for actual streaming,
        falls back to simulated streaming if neither available.
        """
        messages = input_data.get("messages", [])
        
        # Convert to appropriate format
        if self.provider:
            # Use OpenAI-compatible provider streaming
            formatted_messages = self._format_messages_for_provider(messages)
            async for event in stream_openai_provider(self.provider, formatted_messages):
                yield event
                
        elif self.chat_model:
            # Use LangChain chat model streaming
            from langchain_core.messages import HumanMessage, SystemMessage, AIMessage as LCAIMessage
            
            lc_messages = []
            for msg in messages:
                if isinstance(msg, dict):
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    if role == "system":
                        lc_messages.append(SystemMessage(content=content))
                    elif role == "assistant":
                        lc_messages.append(LCAIMessage(content=content))
                    else:
                        lc_messages.append(HumanMessage(content=content))
                elif isinstance(msg, BaseMessage):
                    lc_messages.append(msg)
            
            async for event in stream_langchain_chat(self.chat_model, lc_messages):
                yield event
        else:
            # Fallback: invoke and simulate streaming
            result = self.agent.invoke(input_data, thread_id) if thread_id else self.agent.invoke(input_data)
            response = result.get("response", "")
            
            # Simulate streaming by yielding character by character
            for char in response:
                yield StreamEvent(
                    event_type=StreamEventType.TOKEN,
                    content=char,
                )
                await asyncio.sleep(0.01)  # Small delay for simulation
            
            # Yield metadata at end
            yield StreamEvent(
                event_type=StreamEventType.METADATA,
                content="",
                metadata=result.get("state", {}),
            )
    
    async def astream_text(
        self,
        input_data: Dict[str, Any],
        thread_id: Optional[str] = None,
    ) -> AsyncIterator[str]:
        """Stream just text content."""
        async for event in self.astream(input_data, thread_id):
            if event.event_type == StreamEventType.TOKEN:
                yield event.content
    
    def _format_messages_for_provider(
        self,
        messages: List[Union[Dict, BaseMessage]],
    ) -> List[Dict[str, str]]:
        """Format messages for OpenAI-compatible provider."""
        formatted = []
        for msg in messages:
            if isinstance(msg, dict):
                formatted.append({
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", ""),
                })
            elif isinstance(msg, BaseMessage):
                role = "assistant" if isinstance(msg, AIMessage) else "user"
                if hasattr(msg, '__class__') and 'System' in msg.__class__.__name__:
                    role = "system"
                formatted.append({
                    "role": role,
                    "content": msg.content,
                })
        return formatted
    
    def invoke(self, input_data: Dict[str, Any], thread_id: Optional[str] = None) -> Dict[str, Any]:
        """Non-streaming invoke (delegates to wrapped agent)."""
        if thread_id:
            return self.agent.invoke(input_data, thread_id)
        return self.agent.invoke(input_data)


def create_streaming_agent(
    agent,
    provider=None,
    chat_model=None,
) -> StreamingAgentWrapper:
    """
    Factory function to create a streaming-enabled agent.
    
    Args:
        agent: HGM agent instance
        provider: Optional OpenAICompatibleProvider
        chat_model: Optional LangChain chat model
        
    Returns:
        StreamingAgentWrapper with streaming capabilities
    """
    return StreamingAgentWrapper(
        agent=agent,
        provider=provider,
        chat_model=chat_model,
    )


__all__ = [
    "StreamEventType",
    "StreamEvent",
    "AsyncStreamingCallbackHandler",
    "StreamingMixin",
    "StreamingAgentWrapper",
    "stream_openai_provider",
    "stream_langchain_chat",
    "create_streaming_agent",
]
