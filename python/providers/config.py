"""
Provider Configuration.

Purpose: Unified configuration for OpenAI-compatible providers
Inputs: API keys, base URLs, model names
Outputs: Provider configuration objects

Security: API keys should never be logged or exposed.
Use environment variables for production deployments.
"""

from typing import Optional
from enum import Enum
from dataclasses import dataclass
import os


class ProviderType(Enum):
    """Supported provider types."""
    OPENAI = "openai"
    ZENMUX = "zenmux"
    OPENROUTER = "openrouter"
    CUSTOM = "custom"


@dataclass
class ProviderConfig:
    """Configuration for an OpenAI-compatible provider."""
    
    provider_type: ProviderType
    api_key: str
    base_url: str
    model: str
    organization_id: Optional[str] = None
    
    # Default base URLs for known providers
    PROVIDER_URLS = {
        ProviderType.OPENAI: "https://api.openai.com/v1",
        ProviderType.ZENMUX: "https://api.zenmux.ai/v1",
        ProviderType.OPENROUTER: "https://openrouter.ai/api/v1",
    }
    
    # Environment variable names for API keys
    ENV_KEYS = {
        ProviderType.OPENAI: "OPENAI_API_KEY",
        ProviderType.ZENMUX: "ZENMUX_API_KEY",
        ProviderType.OPENROUTER: "OPENROUTER_API_KEY",
    }
    
    @classmethod
    def zenmux(cls, api_key: Optional[str] = None, model: str = "gpt-4o") -> "ProviderConfig":
        """Create ZenMux provider configuration."""
        key = api_key or os.environ.get("ZENMUX_API_KEY", "")
        return cls(
            provider_type=ProviderType.ZENMUX,
            api_key=key,
            base_url=cls.PROVIDER_URLS[ProviderType.ZENMUX],
            model=model,
        )
    
    @classmethod
    def openai(cls, api_key: Optional[str] = None, model: str = "gpt-4o") -> "ProviderConfig":
        """Create OpenAI provider configuration."""
        key = api_key or os.environ.get("OPENAI_API_KEY", "")
        return cls(
            provider_type=ProviderType.OPENAI,
            api_key=key,
            base_url=cls.PROVIDER_URLS[ProviderType.OPENAI],
            model=model,
        )
    
    @classmethod
    def openrouter(cls, api_key: Optional[str] = None, model: str = "openai/gpt-4o") -> "ProviderConfig":
        """Create OpenRouter provider configuration."""
        key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        return cls(
            provider_type=ProviderType.OPENROUTER,
            api_key=key,
            base_url=cls.PROVIDER_URLS[ProviderType.OPENROUTER],
            model=model,
        )
    
    @classmethod
    def custom(cls, api_key: str, base_url: str, model: str) -> "ProviderConfig":
        """Create custom provider configuration."""
        return cls(
            provider_type=ProviderType.CUSTOM,
            api_key=api_key,
            base_url=base_url,
            model=model,
        )
    
    @classmethod
    def from_env(cls, provider_type: ProviderType = ProviderType.ZENMUX, model: str = "gpt-4o") -> "ProviderConfig":
        """Create configuration from environment variables."""
        env_key = cls.ENV_KEYS.get(provider_type, "API_KEY")
        api_key = os.environ.get(env_key, "")
        base_url = cls.PROVIDER_URLS.get(provider_type, "")
        
        return cls(
            provider_type=provider_type,
            api_key=api_key,
            base_url=base_url,
            model=model,
        )
    
    def validate(self) -> bool:
        """Validate the configuration."""
        if not self.api_key:
            return False
        if not self.base_url:
            return False
        if not self.model:
            return False
        return True
    
    def to_dict(self) -> dict:
        """Convert to dictionary (without exposing API key)."""
        return {
            "provider_type": self.provider_type.value,
            "base_url": self.base_url,
            "model": self.model,
            "organization_id": self.organization_id,
            "has_api_key": bool(self.api_key),
        }
