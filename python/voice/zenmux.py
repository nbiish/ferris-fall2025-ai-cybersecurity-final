"""
ZenMux Provider Integration.

Purpose: LLM backend for voice conversations via OpenAI SDK
Inputs: Conversation messages
Outputs: LLM responses for TTS

Security: API keys should be stored securely and never hardcoded.
Use environment variables or secure credential storage.
"""

from typing import Any, Dict, List, Optional
import os
from openai import OpenAI, AsyncOpenAI


class ZenMuxProvider:
    """
    ZenMux LLM provider using OpenAI SDK compatibility.
    
    Provides LLM responses that can be synthesized with VibeVoice.
    """
    
    DEFAULT_BASE_URL = "https://api.zenmux.ai/v1"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "gpt-4o",
        organization_id: Optional[str] = None,
    ):
        """
        Initialize ZenMux provider.
        
        Args:
            api_key: ZenMux API key (falls back to ZENMUX_API_KEY env var)
            base_url: API base URL (defaults to ZenMux endpoint)
            model: Model to use for chat
            organization_id: Optional organization ID
        """
        self.api_key = api_key or os.environ.get("ZENMUX_API_KEY")
        self.base_url = base_url or self.DEFAULT_BASE_URL
        self.model = model
        self.organization_id = organization_id
        self._client: Optional[OpenAI] = None
        self._async_client: Optional[AsyncOpenAI] = None
    
    def _get_client(self) -> OpenAI:
        """Get or create synchronous OpenAI client."""
        if self._client is None:
            if not self.api_key:
                raise ValueError("API key not provided. Set ZENMUX_API_KEY env var or pass api_key.")
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                organization=self.organization_id,
            )
        return self._client
    
    def _get_async_client(self) -> AsyncOpenAI:
        """Get or create async OpenAI client."""
        if self._async_client is None:
            if not self.api_key:
                raise ValueError("API key not provided. Set ZENMUX_API_KEY env var or pass api_key.")
            self._async_client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                organization=self.organization_id,
            )
        return self._async_client
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 500,
    ) -> Dict[str, Any]:
        """
        Generate chat response.
        
        Args:
            messages: Conversation history
            temperature: Sampling temperature
            max_tokens: Maximum response tokens
            
        Returns:
            Dictionary with response and metadata
        """
        client = self._get_client()
        
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            
            return {
                "response": response.choices[0].message.content,
                "model": self.model,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                },
            }
        except Exception as e:
            return {
                "response": f"Error: {str(e)}",
                "error": True,
            }
    
    async def chat_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
    ):
        """
        Stream chat response for real-time TTS.
        
        Args:
            messages: Conversation history
            temperature: Sampling temperature
            
        Yields:
            Response text chunks
        """
        client = self._get_async_client()
        
        try:
            stream = await client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                stream=True,
            )
            
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            yield f"Error: {str(e)}"
    
    async def achat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 500,
    ) -> Dict[str, Any]:
        """
        Async generate chat response.
        
        Args:
            messages: Conversation history
            temperature: Sampling temperature
            max_tokens: Maximum response tokens
            
        Returns:
            Dictionary with response and metadata
        """
        client = self._get_async_client()
        
        try:
            response = await client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            
            return {
                "response": response.choices[0].message.content,
                "model": self.model,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                },
            }
        except Exception as e:
            return {
                "response": f"Error: {str(e)}",
                "error": True,
            }
    
    def set_model(self, model: str) -> None:
        """Update the model."""
        self.model = model
    
    def set_api_key(self, api_key: str) -> None:
        """Update the API key and reset clients."""
        self.api_key = api_key
        self._client = None
        self._async_client = None
    
    def set_base_url(self, base_url: str) -> None:
        """Update the base URL and reset clients."""
        self.base_url = base_url
        self._client = None
        self._async_client = None
    
    @classmethod
    def from_env(cls, model: str = "gpt-4o") -> "ZenMuxProvider":
        """Create provider from environment variables."""
        return cls(
            api_key=os.environ.get("ZENMUX_API_KEY"),
            base_url=os.environ.get("ZENMUX_BASE_URL", cls.DEFAULT_BASE_URL),
            model=model,
        )
