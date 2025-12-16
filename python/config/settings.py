"""
Settings Management with Encrypted Storage.

Purpose: Secure storage for API keys and provider configuration
Inputs: User configuration via Gradio UI
Outputs: Encrypted JSON files in ~/.nanoboozhoo/

Security: API keys encrypted with Fernet (AES-128-CBC + HMAC-SHA256).
Key derived from machine-specific data for portability.
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field, asdict
from pathlib import Path
import json
import os
import base64
import hashlib

from cryptography.fernet import Fernet


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class ProviderSettings:
    """Settings for an LLM provider."""
    
    id: str
    name: str
    provider_type: str  # openai, zenmux, openrouter, nebius, ollama, custom
    api_key: str = ""
    base_url: str = ""
    default_model: str = ""
    enabled: bool = True
    
    # Provider-specific defaults
    PROVIDER_DEFAULTS = {
        "openai": {
            "base_url": "https://api.openai.com/v1",
            "default_model": "gpt-4o",
        },
        "zenmux": {
            "base_url": "https://api.zenmux.ai/v1",
            "default_model": "zenmux/auto",  # Auto-router
        },
        "openrouter": {
            "base_url": "https://openrouter.ai/api/v1",
            "default_model": "openai/gpt-4o",
        },
        "nebius": {
            "base_url": "https://api.studio.nebius.ai/v1",
            "default_model": "meta-llama/Llama-3.3-70B-Instruct",
        },
        "ollama": {
            "base_url": "http://localhost:11434/v1",
            "default_model": "llama3.2",
        },
    }
    
    @classmethod
    def create_default(cls, provider_type: str, provider_id: Optional[str] = None) -> "ProviderSettings":
        """Create provider with default settings."""
        defaults = cls.PROVIDER_DEFAULTS.get(provider_type, {})
        return cls(
            id=provider_id or provider_type,
            name=provider_type.title(),
            provider_type=provider_type,
            base_url=defaults.get("base_url", ""),
            default_model=defaults.get("default_model", ""),
            enabled=True,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProviderSettings":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class MCPServerSettings:
    """Settings for an MCP server."""
    
    name: str
    command: str
    args: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)
    description: str = ""
    enabled: bool = True
    is_hardcoded: bool = False  # True for Tavily (cannot be deleted)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MCPServerSettings":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class CLIToolSettings:
    """Settings for CLI tools (qwen, gemini)."""
    
    name: str
    enabled: bool = True
    sandbox_mode: bool = False
    timeout_seconds: int = 120
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CLIToolSettings":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class WorkspaceSettings:
    """Workspace configuration."""
    
    root_directory: str = ""
    agents_directory: str = ""
    knowledge_base_directory: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkspaceSettings":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class Settings:
    """Complete application settings."""
    
    providers: List[ProviderSettings] = field(default_factory=list)
    mcp_servers: List[MCPServerSettings] = field(default_factory=list)
    cli_tools: List[CLIToolSettings] = field(default_factory=list)
    workspace: WorkspaceSettings = field(default_factory=WorkspaceSettings)
    active_provider_id: str = ""
    tavily_api_key: str = ""  # Stored separately for hardcoded Tavily server
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "providers": [p.to_dict() for p in self.providers],
            "mcp_servers": [s.to_dict() for s in self.mcp_servers],
            "cli_tools": [t.to_dict() for t in self.cli_tools],
            "workspace": self.workspace.to_dict(),
            "active_provider_id": self.active_provider_id,
            "tavily_api_key": self.tavily_api_key,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Settings":
        """Create from dictionary."""
        return cls(
            providers=[ProviderSettings.from_dict(p) for p in data.get("providers", [])],
            mcp_servers=[MCPServerSettings.from_dict(s) for s in data.get("mcp_servers", [])],
            cli_tools=[CLIToolSettings.from_dict(t) for t in data.get("cli_tools", [])],
            workspace=WorkspaceSettings.from_dict(data.get("workspace", {})),
            active_provider_id=data.get("active_provider_id", ""),
            tavily_api_key=data.get("tavily_api_key", ""),
        )
    
    @classmethod
    def create_default(cls) -> "Settings":
        """Create settings with default providers."""
        return cls(
            providers=[
                ProviderSettings.create_default("zenmux"),
                ProviderSettings.create_default("openai"),
                ProviderSettings.create_default("openrouter"),
                ProviderSettings.create_default("nebius"),
                ProviderSettings.create_default("ollama"),
            ],
            mcp_servers=[
                MCPServerSettings(
                    name="tavily",
                    command="npx",
                    args=["-y", "mcp-remote", "https://mcp.tavily.com/mcp/"],
                    description="Tavily Search API - Web search and research capabilities",
                    enabled=True,
                    is_hardcoded=True,
                ),
            ],
            cli_tools=[
                CLIToolSettings(name="qwen", enabled=True),
                CLIToolSettings(name="gemini", enabled=True),
            ],
            workspace=WorkspaceSettings(),
            active_provider_id="zenmux",
            tavily_api_key="",
        )
    
    def get_provider(self, provider_id: str) -> Optional[ProviderSettings]:
        """Get provider by ID."""
        for p in self.providers:
            if p.id == provider_id:
                return p
        return None
    
    def get_active_provider(self) -> Optional[ProviderSettings]:
        """Get the active provider."""
        return self.get_provider(self.active_provider_id)
    
    def get_enabled_mcp_servers(self) -> List[MCPServerSettings]:
        """Get all enabled MCP servers."""
        return [s for s in self.mcp_servers if s.enabled]
    
    def get_enabled_cli_tools(self) -> List[CLIToolSettings]:
        """Get all enabled CLI tools."""
        return [t for t in self.cli_tools if t.enabled]


# ============================================================================
# Encryption Utilities
# ============================================================================

def _get_encryption_key() -> bytes:
    """
    Derive encryption key from machine-specific data.
    
    Uses a combination of username and home directory to create
    a machine-specific key. This isn't perfect security but provides
    reasonable protection for API keys at rest.
    """
    # Combine machine-specific data
    machine_data = f"{os.getlogin()}:{Path.home()}:nanoboozhoo-v1"
    
    # Derive key using SHA256
    key_material = hashlib.sha256(machine_data.encode()).digest()
    
    # Fernet requires URL-safe base64-encoded 32-byte key
    return base64.urlsafe_b64encode(key_material)


def _get_fernet() -> Fernet:
    """Get Fernet instance for encryption/decryption."""
    return Fernet(_get_encryption_key())


def _encrypt_value(value: str) -> str:
    """Encrypt a string value."""
    if not value:
        return ""
    fernet = _get_fernet()
    encrypted = fernet.encrypt(value.encode())
    return base64.urlsafe_b64encode(encrypted).decode()


def _decrypt_value(encrypted: str) -> str:
    """Decrypt an encrypted string value."""
    if not encrypted:
        return ""
    try:
        fernet = _get_fernet()
        encrypted_bytes = base64.urlsafe_b64decode(encrypted.encode())
        decrypted = fernet.decrypt(encrypted_bytes)
        return decrypted.decode()
    except Exception:
        # Return empty string if decryption fails (key changed, corrupted data)
        return ""


# ============================================================================
# Settings Storage
# ============================================================================

def _get_settings_dir() -> Path:
    """Get the settings directory path."""
    settings_dir = Path.home() / ".nanoboozhoo"
    settings_dir.mkdir(parents=True, exist_ok=True)
    return settings_dir


def _get_settings_path() -> Path:
    """Get the settings file path."""
    return _get_settings_dir() / "settings.json"


def _encrypt_settings(settings: Settings) -> Dict[str, Any]:
    """Encrypt sensitive fields in settings."""
    data = settings.to_dict()
    
    # Encrypt API keys in providers
    for provider in data["providers"]:
        if provider.get("api_key"):
            provider["api_key"] = _encrypt_value(provider["api_key"])
    
    # Encrypt env values in MCP servers (may contain API keys)
    for server in data["mcp_servers"]:
        if server.get("env"):
            encrypted_env = {}
            for key, value in server["env"].items():
                # Only encrypt values that look like secrets
                if "key" in key.lower() or "secret" in key.lower() or "token" in key.lower():
                    encrypted_env[key] = _encrypt_value(value)
                else:
                    encrypted_env[key] = value
            server["env"] = encrypted_env
    
    # Encrypt Tavily API key
    if data.get("tavily_api_key"):
        data["tavily_api_key"] = _encrypt_value(data["tavily_api_key"])
    
    return data


def _decrypt_settings(data: Dict[str, Any]) -> Settings:
    """Decrypt sensitive fields in settings data."""
    # Decrypt API keys in providers
    for provider in data.get("providers", []):
        if provider.get("api_key"):
            provider["api_key"] = _decrypt_value(provider["api_key"])
    
    # Decrypt env values in MCP servers
    for server in data.get("mcp_servers", []):
        if server.get("env"):
            decrypted_env = {}
            for key, value in server["env"].items():
                if "key" in key.lower() or "secret" in key.lower() or "token" in key.lower():
                    decrypted_env[key] = _decrypt_value(value)
                else:
                    decrypted_env[key] = value
            server["env"] = decrypted_env
    
    # Decrypt Tavily API key
    if data.get("tavily_api_key"):
        data["tavily_api_key"] = _decrypt_value(data["tavily_api_key"])
    
    return Settings.from_dict(data)


def load_settings() -> Settings:
    """Load settings from disk, or create defaults if not found."""
    settings_path = _get_settings_path()
    
    if not settings_path.exists():
        # Create default settings
        settings = Settings.create_default()
        save_settings(settings)
        return settings
    
    try:
        with open(settings_path, "r") as f:
            data = json.load(f)
        return _decrypt_settings(data)
    except Exception as e:
        print(f"Warning: Failed to load settings: {e}")
        return Settings.create_default()


def save_settings(settings: Settings) -> bool:
    """Save settings to disk with encryption."""
    settings_path = _get_settings_path()
    
    try:
        encrypted_data = _encrypt_settings(settings)
        with open(settings_path, "w") as f:
            json.dump(encrypted_data, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving settings: {e}")
        return False


# ============================================================================
# Model Fetching
# ============================================================================

async def fetch_models_for_provider(provider: ProviderSettings) -> List[str]:
    """
    Fetch available models from a provider.
    
    Uses the OpenAI /models endpoint which most providers support.
    """
    import httpx
    
    if not provider.api_key or not provider.base_url:
        return []
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            headers = {"Authorization": f"Bearer {provider.api_key}"}
            
            # Handle different provider API patterns
            models_url = f"{provider.base_url.rstrip('/')}/models"
            
            response = await client.get(models_url, headers=headers)
            
            if response.status_code != 200:
                return []
            
            data = response.json()
            models = []
            
            # Handle different response formats
            if "data" in data:
                # OpenAI-style response
                for model in data["data"]:
                    model_id = model.get("id", "")
                    if model_id:
                        models.append(model_id)
            elif "models" in data:
                # Alternative format
                for model in data["models"]:
                    if isinstance(model, str):
                        models.append(model)
                    elif isinstance(model, dict):
                        models.append(model.get("id", model.get("name", "")))
            
            return sorted(models)
            
    except Exception as e:
        print(f"Error fetching models for {provider.name}: {e}")
        return []


def get_default_models_for_provider(provider_type: str) -> List[str]:
    """Get default model list for a provider type (fallback when API fails)."""
    defaults = {
        "zenmux": [
            "zenmux/auto",  # Auto-router - picks best model
            "z-ai/claude-sonnet-4.0",
            "z-ai/gemini-flash-preview",
            "z-ai/glm-4.6v-flash",
            "z-ai/grok-3-fast",
            "deepseek/deepseek-v3.2-speciale",
            "anthropic/claude-sonnet-4-20250514",
            "openai/gpt-4.1-turbo",
            "google/gemini-2.5-flash-preview-05-20",
        ],
        "openai": [
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4-turbo",
            "gpt-4",
            "gpt-3.5-turbo",
            "o1-preview",
            "o1-mini",
        ],
        "openrouter": [
            "openai/gpt-4o",
            "anthropic/claude-3.5-sonnet",
            "google/gemini-pro-1.5",
            "meta-llama/llama-3.1-405b-instruct",
            "deepseek/deepseek-chat",
        ],
        "nebius": [
            "meta-llama/Llama-3.3-70B-Instruct",
            "meta-llama/Llama-3.1-405B-Instruct",
            "deepseek-ai/DeepSeek-V3",
            "Qwen/Qwen2.5-72B-Instruct",
        ],
        "ollama": [
            "llama3.2",
            "llama3.1",
            "mistral",
            "codellama",
            "qwen2.5",
        ],
    }
    return defaults.get(provider_type, [])
