"""
Nanoboozhoo Gradio Application.

A unified interface for:
- Agent Monitor: Run and monitor DeepHGM agents
- RAG Chat: Realtime retrieval-augmented conversation
- Settings: API keys, providers, MCP servers, CLI tools

Run with: python -m python.app
"""

import gradio as gr
import asyncio
import uuid
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from .config import (
    Settings,
    ProviderSettings,
    MCPServerSettings,
    WorkspaceSettings,
    load_settings,
    save_settings,
    fetch_models_for_provider,
    get_default_models_for_provider,
)
from .providers import OpenAICompatibleProvider, ProviderConfig, ProviderType
from .rag import RAGService, RetrievalResult
from .mcp import MCPClient, MCPToolManager, MCPServerConfig, create_mcp_tools_for_agent
from .tools import CLIToolManager, create_cli_tools_for_agent


# ============================================================================
# Global State
# ============================================================================

_settings: Optional[Settings] = None
_agents: Dict[str, "AgentInstance"] = {}


def get_settings() -> Settings:
    """Get or load application settings."""
    global _settings
    if _settings is None:
        _settings = load_settings()
    return _settings


def update_settings(settings: Settings) -> bool:
    """Update and save settings."""
    global _settings
    _settings = settings
    return save_settings(settings)


# ============================================================================
# Agent Instance Management
# ============================================================================

@dataclass
class AgentMetrics:
    """Metrics for an agent instance."""
    
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: float = 0
    cmp_score: float = 0.5
    rag_retrievals: int = 0
    mcp_calls: int = 0
    cli_calls: int = 0


@dataclass
class AgentInstance:
    """Running agent instance with its state."""
    
    id: str
    name: str
    workspace_path: str
    provider_id: str
    model_id: str
    created_at: float
    
    # Tool toggles (per-agent)
    enabled_mcp_servers: Set[str] = field(default_factory=set)
    enabled_cli_tools: Set[str] = field(default_factory=set)
    
    # Conversation
    messages: List[Dict[str, str]] = field(default_factory=list)
    
    # Metrics
    metrics: AgentMetrics = field(default_factory=AgentMetrics)
    
    # Status
    status: str = "idle"  # idle, running, error
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for display."""
        return {
            "id": self.id,
            "name": self.name,
            "workspace_path": self.workspace_path,
            "provider_id": self.provider_id,
            "model_id": self.model_id,
            "created_at": datetime.fromtimestamp(self.created_at).isoformat(),
            "enabled_mcp_servers": list(self.enabled_mcp_servers),
            "enabled_cli_tools": list(self.enabled_cli_tools),
            "message_count": len(self.messages),
            "metrics": {
                "tokens_in": self.metrics.tokens_in,
                "tokens_out": self.metrics.tokens_out,
                "latency_ms": round(self.metrics.latency_ms, 2),
                "cmp_score": round(self.metrics.cmp_score, 3),
                "rag_retrievals": self.metrics.rag_retrievals,
                "mcp_calls": self.metrics.mcp_calls,
                "cli_calls": self.metrics.cli_calls,
            },
            "status": self.status,
        }


def create_agent(
    name: str,
    provider_id: str,
    model_id: str,
    workspace_root: str,
    enabled_mcp_servers: Optional[List[str]] = None,
    enabled_cli_tools: Optional[List[str]] = None,
) -> AgentInstance:
    """Create a new agent instance."""
    agent_id = str(uuid.uuid4())[:8]
    workspace_path = os.path.join(workspace_root, "agents", f"agent-{agent_id}")
    os.makedirs(workspace_path, exist_ok=True)
    
    settings = get_settings()
    
    # Default tool enablement if not provided
    if enabled_mcp_servers is None:
        enabled_mcp = {"tavily"} if settings.tavily_api_key else set()
    else:
        enabled_mcp = set(enabled_mcp_servers)
        
    if enabled_cli_tools is None:
        enabled_cli = {"qwen", "gemini"}
    else:
        enabled_cli = set(enabled_cli_tools)
    
    agent = AgentInstance(
        id=agent_id,
        name=name or f"Agent {agent_id}",
        workspace_path=workspace_path,
        provider_id=provider_id,
        model_id=model_id,
        created_at=time.time(),
        enabled_mcp_servers=enabled_mcp,
        enabled_cli_tools=enabled_cli,
    )
    
    _agents[agent_id] = agent
    return agent


def update_agent_tools(
    agent_id: str,
    enabled_mcp: List[str],
    enabled_cli: List[str],
) -> str:
    """Update tools for an existing agent."""
    agent = get_agent(agent_id)
    if not agent:
        return "Agent not found"
    
    agent.enabled_mcp_servers = set(enabled_mcp)
    agent.enabled_cli_tools = set(enabled_cli)
    return f"Updated tools for {agent.name}"


def get_agent(agent_id: str) -> Optional[AgentInstance]:
    """Get an agent by ID."""
    return _agents.get(agent_id)


def list_agents() -> List[Dict[str, Any]]:
    """List all agents."""
    return [a.to_dict() for a in _agents.values()]


def delete_agent(agent_id: str) -> str:
    """Delete an agent and its workspace."""
    if agent_id not in _agents:
        return "Agent not found"
    
    agent = _agents[agent_id]
    
    # Remove from memory
    del _agents[agent_id]
    
    # Remove workspace
    try:
        if os.path.exists(agent.workspace_path):
            shutil.rmtree(agent.workspace_path)
        return f"Deleted agent {agent.name} and workspace"
    except Exception as e:
        return f"Deleted agent from memory, but failed to delete workspace: {str(e)}"


# ============================================================================
# Chat Functions
# ============================================================================

async def stream_chat_response(
    agent: AgentInstance,
    user_message: str,
):
    """
    Stream a chat response from the agent.
    
    Uses the configured provider with token-by-token streaming.
    Yields response chunks for Gradio streaming.
    """
    settings = get_settings()
    provider_settings = settings.get_provider(agent.provider_id)
    
    if not provider_settings:
        yield "[Error] Provider not configured"
        return
    
    if not provider_settings.api_key:
        yield "[Error] API key not set for provider"
        return
    
    # Create provider
    config = ProviderConfig(
        provider_type=ProviderType.CUSTOM,
        api_key=provider_settings.api_key,
        base_url=provider_settings.base_url,
        model=agent.model_id,
    )
    provider = OpenAICompatibleProvider(config)
    
    # Build messages
    messages = [{"role": m["role"], "content": m["content"]} for m in agent.messages]
    messages.append({"role": "user", "content": user_message})
    
    # Track metrics
    start_time = time.time()
    response_text = ""
    
    try:
        agent.status = "running"
        
        # Stream response
        async for chunk in provider.astream(messages, model=agent.model_id):
            response_text += chunk
            yield response_text
        
        # Update metrics
        agent.metrics.latency_ms = (time.time() - start_time) * 1000
        agent.metrics.tokens_out += len(response_text.split())  # Rough estimate
        agent.metrics.tokens_in += len(user_message.split())
        
        # Save to conversation
        agent.messages.append({"role": "user", "content": user_message})
        agent.messages.append({"role": "assistant", "content": response_text})
        
        agent.status = "idle"
        
    except Exception as e:
        agent.status = "error"
        yield f"[Error] {str(e)}"


def chat_with_agent(
    agent_id: str,
    user_message: str,
    history: List[Dict[str, str]],
) -> Tuple[str, List[Dict[str, str]]]:
    """
    Synchronous chat wrapper for Gradio.
    
    Returns updated history for the chatbot component.
    """
    agent = get_agent(agent_id)
    if not agent:
        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": "[Error] Agent not found"})
        return "", history
    
    # Run async streaming
    async def _run():
        response = ""
        async for chunk in stream_chat_response(agent, user_message):
            response = chunk
        return response
    
    try:
        response = asyncio.run(_run())
        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": response})
    except Exception as e:
        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": f"[Error] {str(e)}"})
    
    return "", history


# ============================================================================
# RAG Functions
# ============================================================================

_rag_service: Optional[RAGService] = None
_voice_client: Optional["VibeVoiceRealtime"] = None


def get_rag_service() -> RAGService:
    """Get or create RAG service."""
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
    return _rag_service


def get_voice_client():
    """Get or create voice client for STT/TTS."""
    global _voice_client
    if _voice_client is None:
        from .voice import VibeVoiceRealtime
        _voice_client = VibeVoiceRealtime()
    return _voice_client


def get_hgm_sessions(workspace_root: str) -> List[Dict[str, Any]]:
    """Get all HGM agent sessions from workspace."""
    from .hgm import create_session_manager
    try:
        manager = create_session_manager(workspace_root)
        return manager.list_sessions()
    except Exception:
        return []


async def rag_chat_with_llm(
    query: str,
    history: List[Dict[str, str]],
    provider_id: str,
    model_id: str,
    agent_session_id: Optional[str],
    strategy: str = "ssd",
    k: int = 5,
    diversity: float = 0.5,
    recent_window: int = 10,
) -> Tuple[str, List[Dict[str, str]], str]:
    """
    RAG-enhanced chat with LLM integration.
    
    Retrieves context from RAG, optionally filters by agent session,
    and calls the selected LLM provider/model for the response.
    
    Uses pyversity diversification strategies:
    - SSD: Sequence-aware (tracks recent chunks for novelty)
    - MMR: Maximal Marginal Relevance (avoid near-duplicates)
    - MSD: Max Sum of Distances (wider topic spread)
    - DPP: Determinantal Point Process (probabilistic diversity)
    - COVER: Facility-Location (topic coverage)
    
    Returns: (cleared input, updated history, sources markdown)
    """
    rag = get_rag_service()
    rag.recent_window_size = recent_window  # Update SSD window size
    settings = get_settings()
    
    try:
        # Retrieve relevant context (optionally filtered by agent)
        result = rag.retrieve(
            query=query,
            k=k,
            diversity=diversity,
            strategy=strategy,
            agent_filter=agent_session_id if agent_session_id else None,
        )
        
        # Build context string
        context = result.to_context_string(max_chunks=k)
        
        # Format sources markdown
        sources_md = "### Retrieved Sources\n\n"
        for i, (chunk, score) in enumerate(zip(result.chunks, result.scores)):
            source = chunk.source_path or chunk.source_type
            sources_md += f"**{i+1}. {source}** (score: {score:.3f})\n"
            sources_md += f"> {chunk.content[:200]}...\n\n"
        
        if not result.chunks:
            sources_md = "No relevant sources found. Index some content first."
        
        # Get provider settings
        provider_settings = settings.get_provider(provider_id)
        if not provider_settings or not provider_settings.api_key:
            response = f"[Error] Provider '{provider_id}' not configured or missing API key.\n\n**Retrieved Context:**\n{context}"
            history.append({"role": "user", "content": query})
            history.append({"role": "assistant", "content": response})
            return "", history, sources_md
        
        # Build RAG-augmented prompt
        system_prompt = """You are a helpful AI assistant with access to retrieved context.
Use the provided context to answer the user's question accurately.
If the context doesn't contain relevant information, say so and provide your best answer.
Always cite sources when using information from the context."""
        
        rag_prompt = f"""### Retrieved Context:
{context}

### User Question:
{query}

Please provide a helpful response based on the context above."""
        
        # Create provider and get response
        config = ProviderConfig(
            provider_type=ProviderType.CUSTOM,
            api_key=provider_settings.api_key,
            base_url=provider_settings.base_url,
            model=model_id,
        )
        provider = OpenAICompatibleProvider(config)
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": rag_prompt},
        ]
        
        # Stream response
        response_text = ""
        async for chunk in provider.astream(messages, model=model_id):
            response_text += chunk
        
        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": response_text})
        return "", history, sources_md
        
    except Exception as e:
        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": f"[Error] {str(e)}"})
        return "", history, f"Error: {str(e)}"


def rag_chat_sync(
    query: str,
    history: List[Dict[str, str]],
    provider_id: str,
    model_id: str,
    agent_session_id: Optional[str],
    strategy: str = "ssd",
    k: int = 5,
    diversity: float = 0.5,
    recent_window: int = 10,
) -> Tuple[str, List[Dict[str, str]], str]:
    """Synchronous wrapper for RAG chat with LLM."""
    return asyncio.run(rag_chat_with_llm(
        query, history, provider_id, model_id, agent_session_id,
        strategy, k, diversity, recent_window
    ))


def transcribe_audio(audio_path: str) -> str:
    """Transcribe audio file to text using VibeVoice STT."""
    if not audio_path:
        return ""
    try:
        voice = get_voice_client()
        result = voice.transcribe(audio_path)
        return result.get("text", "")
    except Exception as e:
        return f"[Transcription Error] {str(e)}"


def synthesize_response(text: str, speaker: str = "Carter") -> Optional[Tuple[int, Any]]:
    """Synthesize text to speech using VibeVoice TTS."""
    if not text or text.startswith("[Error]"):
        return None
    try:
        voice = get_voice_client()
        result = voice.synthesize(text, speaker=speaker)
        if result.get("audio") is not None:
            return (result["sample_rate"], result["audio"])
        return None
    except Exception:
        return None


def index_folder(folder_path: str) -> str:
    """Index a folder for RAG."""
    if not folder_path or not os.path.isdir(folder_path):
        return "Invalid folder path"
    
    rag = get_rag_service()
    try:
        count = rag.index_folder(folder_path)
        return f"Indexed {count} chunks from {folder_path}"
    except Exception as e:
        return f"Error indexing: {str(e)}"


def index_agent_session(agent_id: str, workspace_root: str) -> str:
    """Index an agent session's documents and outputs for RAG."""
    from .hgm import create_session_manager
    try:
        manager = create_session_manager(workspace_root)
        paths = manager.get_agent_workspace_info(agent_id)
        
        rag = get_rag_service()
        total = 0
        
        for folder_name in ["documents", "outputs"]:
            folder_path = paths.get(folder_name, "")
            if folder_path and os.path.isdir(folder_path):
                count = rag.index_folder(folder_path, agent_id=agent_id)
                total += count
        
        return f"Indexed {total} chunks from agent {agent_id}"
    except Exception as e:
        return f"Error indexing agent: {str(e)}"


# ============================================================================
# Settings Functions
# ============================================================================

def get_providers_table() -> List[List[str]]:
    """Get providers as table data."""
    settings = get_settings()
    return [
        [p.id, p.name, p.provider_type, p.base_url, "✓" if p.enabled else "✗", "***" if p.api_key else ""]
        for p in settings.providers
    ]


def get_mcp_servers_table() -> List[List[str]]:
    """Get MCP servers as table data."""
    settings = get_settings()
    return [
        [s.name, s.command, " ".join(s.args[:2]) + "...", "✓" if s.enabled else "✗", "🔒" if s.is_hardcoded else ""]
        for s in settings.mcp_servers
    ]


def save_provider_key(provider_id: str, api_key: str) -> str:
    """Save API key for a provider."""
    settings = get_settings()
    for p in settings.providers:
        if p.id == provider_id:
            p.api_key = api_key
            update_settings(settings)
            return f"API key saved for {provider_id}"
    return f"Provider {provider_id} not found"


def save_tavily_key(api_key: str) -> str:
    """Save Tavily API key."""
    settings = get_settings()
    settings.tavily_api_key = api_key
    update_settings(settings)
    return "Tavily API key saved"


def add_custom_provider(
    provider_id: str,
    name: str,
    base_url: str,
    api_key: str,
    default_model: str,
) -> Tuple[str, List[List[str]]]:
    """Add a custom provider."""
    settings = get_settings()
    
    # Check for duplicate
    if any(p.id == provider_id for p in settings.providers):
        return f"Provider {provider_id} already exists", get_providers_table()
    
    settings.providers.append(ProviderSettings(
        id=provider_id,
        name=name or provider_id,
        provider_type="custom",
        api_key=api_key,
        base_url=base_url,
        default_model=default_model,
        enabled=True,
    ))
    
    update_settings(settings)
    return f"Added provider: {name}", get_providers_table()


def add_mcp_server(
    name: str,
    command: str,
    args: str,
    description: str,
) -> Tuple[str, List[List[str]]]:
    """Add a custom MCP server."""
    settings = get_settings()
    
    # Check for duplicate
    if any(s.name == name for s in settings.mcp_servers):
        return f"Server {name} already exists", get_mcp_servers_table()
    
    settings.mcp_servers.append(MCPServerSettings(
        name=name,
        command=command,
        args=args.split() if args else [],
        description=description,
        enabled=True,
        is_hardcoded=False,
    ))
    
    update_settings(settings)
    return f"Added MCP server: {name}", get_mcp_servers_table()


# ============================================================================
# Gradio Interface
# ============================================================================

def create_agent_monitor_tab():
    """Create the Agent Monitor tab."""
    with gr.Tab("🤖 Agent Monitor"):
        with gr.Row():
            # Left sidebar - Agent setup
            with gr.Column(scale=1):
                gr.Markdown("### Create Agent")
                
                agent_name = gr.Textbox(label="Agent Name", placeholder="My Agent")
                
                settings = get_settings()
                provider_choices = [(p.name, p.id) for p in settings.providers]
                provider_dropdown = gr.Dropdown(
                    label="Provider",
                    choices=provider_choices,
                    value=settings.active_provider_id,
                )
                
                model_dropdown = gr.Dropdown(
                    label="Model",
                    choices=get_default_models_for_provider(settings.active_provider_id),
                )
                
                workspace_input = gr.Textbox(
                    label="Workspace Root",
                    value=str(Path.home() / "nanoboozhoo-workspace"),
                )
                
                # Tool choices
                mcp_choices = [s.name for s in settings.mcp_servers if s.enabled]
                if settings.tavily_api_key and "tavily" not in mcp_choices:
                    mcp_choices.append("tavily")
                
                cli_choices = ["qwen", "gemini"]
                
                gr.Markdown("### Tool Configuration")
                mcp_toggles = gr.CheckboxGroup(
                    label="MCP Servers",
                    choices=mcp_choices,
                    value=mcp_choices,
                )
                
                cli_toggles = gr.CheckboxGroup(
                    label="CLI Tools",
                    choices=cli_choices,
                    value=cli_choices,
                )
                
                create_btn = gr.Button("Create Agent", variant="primary")
                
                gr.Markdown("### Active Agents")
                agent_selector = gr.Dropdown(label="Select Agent", choices=[], interactive=True)
                with gr.Row():
                    refresh_agents_btn = gr.Button("Refresh List")
                    delete_agent_btn = gr.Button("Delete Agent", variant="stop")
            
            # Main chat area
            with gr.Column(scale=3):
                gr.Markdown("### Agent Chat")
                
                current_agent_id = gr.State(value="")
                
                chatbot = gr.Chatbot(
                    label="Conversation",
                    height=500,
                )
                
                with gr.Row():
                    chat_input = gr.Textbox(
                        label="Message",
                        placeholder="Type your message...",
                        scale=4,
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)
                
                gr.Markdown("### Metrics")
                metrics_json = gr.JSON(label="Agent Metrics", value={})
        
        # Event handlers
        def on_provider_change(provider_id):
            models = get_default_models_for_provider(provider_id)
            return gr.update(choices=models, value=models[0] if models else "")
        
        provider_dropdown.change(
            on_provider_change,
            inputs=[provider_dropdown],
            outputs=[model_dropdown],
        )
        
        def on_create_agent(name, provider_id, model_id, workspace_root, mcp_tools, cli_tools):
            agent = create_agent(name, provider_id, model_id, workspace_root, mcp_tools, cli_tools)
            agent_list = [(a.name, a.id) for a in _agents.values()]
            return agent.id, gr.update(choices=agent_list, value=agent.id), []
        
        create_btn.click(
            on_create_agent,
            inputs=[agent_name, provider_dropdown, model_dropdown, workspace_input, mcp_toggles, cli_toggles],
            outputs=[current_agent_id, agent_selector, chatbot],
        )
        
        def on_refresh_agents():
            agent_list = [(a.name, a.id) for a in _agents.values()]
            return gr.update(choices=agent_list)

        refresh_agents_btn.click(
            on_refresh_agents,
            outputs=[agent_selector],
        )
        
        def on_delete_agent(agent_id):
            if not agent_id:
                return gr.update(), gr.update(), gr.update()
            
            msg = delete_agent(agent_id)
            gr.Info(msg)
            
            agent_list = [(a.name, a.id) for a in _agents.values()]
            return gr.update(choices=agent_list, value=None), gr.update(value=""), gr.update(value=[])

        delete_agent_btn.click(
            on_delete_agent,
            inputs=[agent_selector],
            outputs=[agent_selector, current_agent_id, chatbot],
        )
        
        def on_agent_select(agent_id):
            agent = get_agent(agent_id)
            if not agent:
                return [], [], [], {}
            
            history = list(agent.messages)
            
            metrics = agent.to_dict()["metrics"]
            
            return (
                list(agent.enabled_mcp_servers),
                list(agent.enabled_cli_tools),
                history,
                metrics
            )

        agent_selector.change(
            on_agent_select,
            inputs=[agent_selector],
            outputs=[mcp_toggles, cli_toggles, chatbot, metrics_json],
        ).then(
            lambda id: id, inputs=[agent_selector], outputs=[current_agent_id]
        )
        
        def on_tools_change(agent_id, mcp_tools, cli_tools):
            if not agent_id:
                return
            update_agent_tools(agent_id, mcp_tools, cli_tools)
            
        mcp_toggles.change(
            on_tools_change,
            inputs=[current_agent_id, mcp_toggles, cli_toggles],
        )
        
        cli_toggles.change(
            on_tools_change,
            inputs=[current_agent_id, mcp_toggles, cli_toggles],
        )
        
        def on_send_message(agent_id, message, history):
            if not agent_id:
                history = history or []
                history.append({"role": "user", "content": message})
                history.append({"role": "assistant", "content": "[Error] No agent selected. Create an agent first."})
                return "", history, {}
            
            _, history = chat_with_agent(agent_id, message, history or [])
            agent = get_agent(agent_id)
            metrics = agent.to_dict()["metrics"] if agent else {}
            return "", history, metrics
        
        send_btn.click(
            on_send_message,
            inputs=[current_agent_id, chat_input, chatbot],
            outputs=[chat_input, chatbot, metrics_json],
        )
        
        chat_input.submit(
            on_send_message,
            inputs=[current_agent_id, chat_input, chatbot],
            outputs=[chat_input, chatbot, metrics_json],
        )


def create_rag_tab():
    """Create the RAG Chat tab with agent session selection, provider/model selection, and voice."""
    with gr.Tab("📚 RAG Chat"):
        with gr.Row():
            # Left sidebar - Configuration
            with gr.Column(scale=1):
                gr.Markdown("### Agent Session")
                
                workspace_input = gr.Textbox(
                    label="Workspace Root",
                    value=str(Path.home() / "nanoboozhoo-workspace"),
                )
                
                agent_session_dropdown = gr.Dropdown(
                    label="Select Agent Session",
                    choices=[],
                    value=None,
                    allow_custom_value=True,
                    info="Select an agent session to RAG against, or leave empty for all",
                )
                
                refresh_sessions_btn = gr.Button("🔄 Refresh Sessions", size="sm")
                
                gr.Markdown("### LLM Provider")
                settings = get_settings()
                provider_choices = [(p.name, p.id) for p in settings.providers if p.enabled]
                
                rag_provider_dropdown = gr.Dropdown(
                    label="Provider",
                    choices=provider_choices,
                    value=settings.active_provider_id,
                )
                
                rag_model_dropdown = gr.Dropdown(
                    label="Model",
                    choices=get_default_models_for_provider(settings.active_provider_id),
                    value=get_default_models_for_provider(settings.active_provider_id)[0] if get_default_models_for_provider(settings.active_provider_id) else "",
                )
                
                gr.Markdown("### Voice Settings")
                from .voice import AVAILABLE_SPEAKERS
                
                voice_enabled = gr.Checkbox(
                    label="🎤 Enable Voice Mode",
                    value=False,
                    info="Toggle voice input/output with VibeVoice-Realtime",
                )
                
                speaker_dropdown = gr.Dropdown(
                    label="TTS Speaker",
                    choices=AVAILABLE_SPEAKERS,
                    value="Carter",
                    visible=False,
                )
                
                gr.Markdown("### RAG Settings")
                strategy_dropdown = gr.Dropdown(
                    label="Diversification Strategy",
                    choices=[
                        ("SSD - Sequence-aware (best for chat)", "ssd"),
                        ("MMR - Maximal Marginal Relevance", "mmr"),
                        ("MSD - Max Sum of Distances", "msd"),
                        ("DPP - Determinantal Point Process", "dpp"),
                        ("COVER - Topic Coverage", "cover"),
                    ],
                    value="ssd",
                    info="Pyversity diversification strategy",
                )
                
                with gr.Row():
                    k_slider = gr.Slider(
                        label="Results (k)",
                        minimum=1,
                        maximum=20,
                        value=5,
                        step=1,
                        info="Number of chunks to retrieve",
                    )
                    diversity_slider = gr.Slider(
                        label="Diversity",
                        minimum=0.0,
                        maximum=1.0,
                        value=0.5,
                        step=0.1,
                        info="0.0=relevance only, 1.0=max diversity",
                    )
                
                recent_window_slider = gr.Slider(
                    label="SSD Recent Window",
                    minimum=5,
                    maximum=50,
                    value=10,
                    step=5,
                    info="Track recent chunks for SSD sequence-aware diversification",
                    visible=True,
                )
            
            # Main chat area
            with gr.Column(scale=2):
                gr.Markdown("### RAG-Enhanced Conversation")
                
                rag_chatbot = gr.Chatbot(
                    label="Conversation",
                    height=450,
                )
                
                # Text input row
                with gr.Row(visible=True) as text_input_row:
                    rag_input = gr.Textbox(
                        label="Query",
                        placeholder="Ask a question about your agent sessions...",
                        scale=4,
                    )
                    rag_send_btn = gr.Button("Send", variant="primary", scale=1)
                
                # Voice input row (hidden by default)
                with gr.Row(visible=False) as voice_input_row:
                    audio_input = gr.Audio(
                        label="🎤 Voice Input",
                        sources=["microphone"],
                        type="filepath",
                        scale=3,
                    )
                    voice_send_btn = gr.Button("🎤 Send Voice", variant="primary", scale=1)
                
                # Audio output for TTS
                audio_output = gr.Audio(
                    label="🔊 Response Audio",
                    visible=False,
                    autoplay=True,
                )
                
                # Clear button
                clear_btn = gr.Button("🗑️ Clear Conversation")
            
            # Right sidebar - Sources & Indexing
            with gr.Column(scale=1):
                gr.Markdown("### Retrieved Sources")
                sources_md = gr.Markdown("*Send a query to see sources*")
                
                gr.Markdown("### Index Content")
                
                with gr.Accordion("Index Folder", open=False):
                    folder_input = gr.Textbox(
                        label="Folder Path",
                        placeholder="/path/to/documents",
                    )
                    index_btn = gr.Button("Index Folder")
                    index_status = gr.Textbox(label="Status", interactive=False)
                
                with gr.Accordion("Index Agent Session", open=False):
                    index_agent_btn = gr.Button("Index Selected Agent")
                    index_agent_status = gr.Textbox(label="Status", interactive=False)
                
                gr.Markdown("### RAG Stats")
                rag_stats = gr.JSON(label="Statistics", value={})
                stats_btn = gr.Button("Refresh Stats", size="sm")
        
        # Event handlers
        def on_provider_change(provider_id):
            models = get_default_models_for_provider(provider_id)
            return gr.update(choices=models, value=models[0] if models else "")
        
        rag_provider_dropdown.change(
            on_provider_change,
            inputs=[rag_provider_dropdown],
            outputs=[rag_model_dropdown],
        )
        
        def on_refresh_sessions(workspace_root):
            sessions = get_hgm_sessions(workspace_root)
            choices = [(f"{s['agent_name']} (Gen {s['generation']})", s['agent_id']) for s in sessions]
            choices.insert(0, ("All Sessions", None))
            return gr.update(choices=choices, value=None)
        
        refresh_sessions_btn.click(
            on_refresh_sessions,
            inputs=[workspace_input],
            outputs=[agent_session_dropdown],
        )
        
        def on_voice_toggle(enabled):
            return (
                gr.update(visible=not enabled),  # text_input_row
                gr.update(visible=enabled),       # voice_input_row
                gr.update(visible=enabled),       # speaker_dropdown
                gr.update(visible=enabled),       # audio_output
            )
        
        voice_enabled.change(
            on_voice_toggle,
            inputs=[voice_enabled],
            outputs=[text_input_row, voice_input_row, speaker_dropdown, audio_output],
        )
        
        def on_text_send(query, history, provider_id, model_id, agent_session, strategy, k, diversity, recent_window):
            if not query.strip():
                return "", history or [], "*Send a query to see sources*"
            return rag_chat_sync(
                query, history or [], provider_id, model_id, agent_session,
                strategy, int(k), diversity, int(recent_window)
            )
        
        rag_send_btn.click(
            on_text_send,
            inputs=[rag_input, rag_chatbot, rag_provider_dropdown, rag_model_dropdown,
                    agent_session_dropdown, strategy_dropdown, k_slider, diversity_slider,
                    recent_window_slider],
            outputs=[rag_input, rag_chatbot, sources_md],
        )
        
        rag_input.submit(
            on_text_send,
            inputs=[rag_input, rag_chatbot, rag_provider_dropdown, rag_model_dropdown,
                    agent_session_dropdown, strategy_dropdown, k_slider, diversity_slider,
                    recent_window_slider],
            outputs=[rag_input, rag_chatbot, sources_md],
        )
        
        def on_voice_send(audio_path, history, provider_id, model_id, agent_session,
                          strategy, k, diversity, recent_window, speaker, voice_mode):
            if not audio_path or not voice_mode:
                return history or [], "*No audio input*", None
            
            # Transcribe audio
            transcribed_text = transcribe_audio(audio_path)
            if not transcribed_text or transcribed_text.startswith("["):
                return history or [], f"*Transcription failed: {transcribed_text}*", None
            
            # Get RAG response
            _, updated_history, sources = rag_chat_sync(
                transcribed_text, history or [], provider_id, model_id, agent_session,
                strategy, int(k), diversity, int(recent_window)
            )
            
            # Synthesize response to audio
            if updated_history:
                last_response = updated_history[-1]["content"]
                audio_data = synthesize_response(last_response, speaker=speaker)
                return updated_history, sources, audio_data
            
            return updated_history, sources, None
        
        voice_send_btn.click(
            on_voice_send,
            inputs=[audio_input, rag_chatbot, rag_provider_dropdown, rag_model_dropdown,
                    agent_session_dropdown, strategy_dropdown, k_slider, diversity_slider,
                    recent_window_slider, speaker_dropdown, voice_enabled],
            outputs=[rag_chatbot, sources_md, audio_output],
        )
        
        def on_clear():
            return [], "*Send a query to see sources*"
        
        clear_btn.click(
            on_clear,
            outputs=[rag_chatbot, sources_md],
        )
        
        index_btn.click(
            index_folder,
            inputs=[folder_input],
            outputs=[index_status],
        )
        
        def on_index_agent(agent_id, workspace_root):
            if not agent_id:
                return "No agent selected"
            return index_agent_session(agent_id, workspace_root)
        
        index_agent_btn.click(
            on_index_agent,
            inputs=[agent_session_dropdown, workspace_input],
            outputs=[index_agent_status],
        )
        
        stats_btn.click(
            lambda: get_rag_service().stats(),
            outputs=[rag_stats],
        )


def create_settings_tab():
    """Create the Settings tab."""
    with gr.Tab("⚙️ Settings"):
        gr.Markdown("## API Keys & Providers")
        
        with gr.Accordion("Provider API Keys", open=True):
            settings = get_settings()
            
            gr.Markdown("### Built-in Providers")
            
            for provider in settings.providers:
                if provider.provider_type != "custom":
                    with gr.Row():
                        gr.Markdown(f"**{provider.name}** ({provider.base_url})")
                        key_input = gr.Textbox(
                            label=f"{provider.name} API Key",
                            type="password",
                            value="",
                            placeholder="Enter API key...",
                        )
                        save_key_btn = gr.Button("Save", size="sm")
                        
                        save_key_btn.click(
                            lambda k, pid=provider.id: save_provider_key(pid, k),
                            inputs=[key_input],
                            outputs=[gr.Textbox(visible=False)],
                        )
        
        with gr.Accordion("Tavily Search (Hardcoded MCP)", open=True):
            gr.Markdown("""
            Tavily is the default web search provider for agents.
            Get your API key at [tavily.com](https://tavily.com)
            """)
            
            tavily_key = gr.Textbox(
                label="Tavily API Key",
                type="password",
                placeholder="Enter Tavily API key...",
            )
            tavily_save_btn = gr.Button("Save Tavily Key")
            tavily_status = gr.Textbox(label="Status", interactive=False)
            
            tavily_save_btn.click(
                save_tavily_key,
                inputs=[tavily_key],
                outputs=[tavily_status],
            )
        
        with gr.Accordion("Custom Provider", open=False):
            gr.Markdown("Add a custom OpenAI-compatible provider")
            
            with gr.Row():
                custom_id = gr.Textbox(label="Provider ID", placeholder="my-provider")
                custom_name = gr.Textbox(label="Display Name", placeholder="My Provider")
            
            with gr.Row():
                custom_url = gr.Textbox(label="Base URL", placeholder="https://api.example.com/v1")
                custom_model = gr.Textbox(label="Default Model", placeholder="gpt-4")
            
            custom_key = gr.Textbox(label="API Key", type="password")
            
            add_provider_btn = gr.Button("Add Provider", variant="primary")
            provider_status = gr.Textbox(label="Status", interactive=False)
            
            providers_table = gr.Dataframe(
                headers=["ID", "Name", "Type", "Base URL", "Enabled", "Key Set"],
                value=get_providers_table(),
                label="All Providers",
            )
            
            add_provider_btn.click(
                add_custom_provider,
                inputs=[custom_id, custom_name, custom_url, custom_key, custom_model],
                outputs=[provider_status, providers_table],
            )
        
        gr.Markdown("## MCP Servers")
        
        with gr.Accordion("Add Custom MCP Server", open=False):
            gr.Markdown("Add additional MCP servers (Tavily is always available)")
            
            with gr.Row():
                mcp_name = gr.Textbox(label="Server Name", placeholder="my-server")
                mcp_command = gr.Textbox(label="Command", placeholder="npx")
            
            mcp_args = gr.Textbox(label="Arguments", placeholder="-y @my-org/my-mcp-server")
            mcp_desc = gr.Textbox(label="Description", placeholder="My custom MCP server")
            
            add_mcp_btn = gr.Button("Add MCP Server", variant="primary")
            mcp_status = gr.Textbox(label="Status", interactive=False)
            
            mcp_table = gr.Dataframe(
                headers=["Name", "Command", "Args", "Enabled", "Hardcoded"],
                value=get_mcp_servers_table(),
                label="MCP Servers",
            )
            
            add_mcp_btn.click(
                add_mcp_server,
                inputs=[mcp_name, mcp_command, mcp_args, mcp_desc],
                outputs=[mcp_status, mcp_table],
            )
        
        gr.Markdown("## Workspace")
        
        with gr.Accordion("Workspace Settings", open=False):
            workspace_root = gr.Textbox(
                label="Workspace Root Directory",
                value=str(Path.home() / "nanoboozhoo-workspace"),
            )
            
            def save_workspace(path):
                settings = get_settings()
                settings.workspace.root_directory = path
                update_settings(settings)
                return "Workspace saved"
            
            save_workspace_btn = gr.Button("Save Workspace Settings")
            workspace_status = gr.Textbox(label="Status", interactive=False)
            
            save_workspace_btn.click(
                save_workspace,
                inputs=[workspace_root],
                outputs=[workspace_status],
            )


def create_app() -> gr.Blocks:
    """Create the main Gradio application."""
    with gr.Blocks(
        title="Nanoboozhoo",
    ) as app:
        gr.Markdown("""
        # 🦊 Nanoboozhoo
        
        **AI Agent Platform** with DeepHGM agents, RAG-enhanced conversation, and tool integration.
        """)
        
        create_agent_monitor_tab()
        create_rag_tab()
        create_settings_tab()
        
        gr.Markdown("""
        ---
        *Nanoboozhoo v2.0 - Gradio Edition*
        """)
    
    return app


# ============================================================================
# Entry Point
# ============================================================================

def main():
    """Run the Gradio application."""
    app = create_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )


if __name__ == "__main__":
    main()
