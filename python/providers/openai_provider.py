"""
OpenAI-Compatible Provider.

Purpose: Unified interface for OpenAI SDK-compatible providers
Inputs: Provider configuration, chat messages
Outputs: LLM responses

Supports: OpenAI, ZenMux, OpenRouter, and custom endpoints
"""

from typing import Any, Dict, Iterator, List, Optional, AsyncIterator
import os
from openai import OpenAI, AsyncOpenAI

from .config import ProviderConfig, ProviderType


class OpenAICompatibleProvider:
    """
    Unified provider for OpenAI SDK-compatible APIs.
    
    Works with:
    - OpenAI direct API
    - ZenMux (OpenAI-compatible aggregator)
    - OpenRouter (multi-provider aggregator)
    - Any custom OpenAI-compatible endpoint
    """
    
    def __init__(self, config: ProviderConfig):
        """
        Initialize provider with configuration.
        
        Args:
            config: Provider configuration with API key, base URL, and model
        """
        self.config = config
        self._client: Optional[OpenAI] = None
        self._async_client: Optional[AsyncOpenAI] = None
    
    @classmethod
    def zenmux(cls, api_key: Optional[str] = None, model: str = "gpt-4o") -> "OpenAICompatibleProvider":
        """Create a ZenMux provider."""
        return cls(ProviderConfig.zenmux(api_key, model))
    
    @classmethod
    def openai(cls, api_key: Optional[str] = None, model: str = "gpt-4o") -> "OpenAICompatibleProvider":
        """Create an OpenAI provider."""
        return cls(ProviderConfig.openai(api_key, model))
    
    @classmethod
    def openrouter(cls, api_key: Optional[str] = None, model: str = "openai/gpt-4o") -> "OpenAICompatibleProvider":
        """Create an OpenRouter provider."""
        return cls(ProviderConfig.openrouter(api_key, model))
    
    @classmethod
    def custom(cls, api_key: str, base_url: str, model: str) -> "OpenAICompatibleProvider":
        """Create a custom provider."""
        return cls(ProviderConfig.custom(api_key, base_url, model))
    
    def _get_client(self) -> OpenAI:
        """Get or create synchronous client."""
        if self._client is None:
            if not self.config.api_key:
                raise ValueError(f"API key not configured for {self.config.provider_type.value}")
            self._client = OpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                organization=self.config.organization_id,
            )
        return self._client
    
    def _get_async_client(self) -> AsyncOpenAI:
        """Get or create async client."""
        if self._async_client is None:
            if not self.config.api_key:
                raise ValueError(f"API key not configured for {self.config.provider_type.value}")
            self._async_client = AsyncOpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                organization=self.config.organization_id,
            )
        return self._async_client
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Synchronous chat completion.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Override model (uses config model if not specified)
            temperature: Sampling temperature
            max_tokens: Maximum response tokens
            **kwargs: Additional parameters passed to the API
            
        Returns:
            Dictionary with response content and metadata
        """
        client = self._get_client()
        
        response = client.chat.completions.create(
            model=model or self.config.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        
        return {
            "content": response.choices[0].message.content,
            "model": response.model,
            "finish_reason": response.choices[0].finish_reason,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            } if response.usage else None,
        }
    
    async def achat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Async chat completion.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Override model (uses config model if not specified)
            temperature: Sampling temperature
            max_tokens: Maximum response tokens
            **kwargs: Additional parameters passed to the API
            
        Returns:
            Dictionary with response content and metadata
        """
        client = self._get_async_client()
        
        response = await client.chat.completions.create(
            model=model or self.config.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        
        return {
            "content": response.choices[0].message.content,
            "model": response.model,
            "finish_reason": response.choices[0].finish_reason,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            } if response.usage else None,
        }
    
    def stream(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        **kwargs,
    ) -> Iterator[str]:
        """
        Synchronous streaming chat completion.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Override model (uses config model if not specified)
            temperature: Sampling temperature
            **kwargs: Additional parameters passed to the API
            
        Yields:
            Response content chunks
        """
        client = self._get_client()
        
        stream = client.chat.completions.create(
            model=model or self.config.model,
            messages=messages,
            temperature=temperature,
            stream=True,
            **kwargs,
        )
        
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    
    async def astream(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        **kwargs,
    ) -> AsyncIterator[str]:
        """
        Async streaming chat completion.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Override model (uses config model if not specified)
            temperature: Sampling temperature
            **kwargs: Additional parameters passed to the API
            
        Yields:
            Response content chunks
        """
        client = self._get_async_client()
        
        stream = await client.chat.completions.create(
            model=model or self.config.model,
            messages=messages,
            temperature=temperature,
            stream=True,
            **kwargs,
        )
        
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    
    def simple_chat(self, prompt: str, system: Optional[str] = None) -> str:
        """
        Simple synchronous chat with optional system prompt.
        
        Args:
            prompt: User message
            system: Optional system prompt
            
        Returns:
            Response content string
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        result = self.chat(messages)
        return result["content"]
    
    async def asimple_chat(self, prompt: str, system: Optional[str] = None) -> str:
        """
        Simple async chat with optional system prompt.
        
        Args:
            prompt: User message
            system: Optional system prompt
            
        Returns:
            Response content string
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        result = await self.achat(messages)
        return result["content"]
    
    def set_model(self, model: str) -> None:
        """Update the default model."""
        self.config.model = model
    
    def set_api_key(self, api_key: str) -> None:
        """Update API key and reset clients."""
        self.config.api_key = api_key
        self._client = None
        self._async_client = None
    
    def get_info(self) -> Dict[str, Any]:
        """Get provider info (without exposing API key)."""
        return self.config.to_dict()
