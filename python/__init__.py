"""
Nanoboozhoo - AI Agent Platform.

A Gradio-based application for running DeepHGM agents with:
- RAG-enhanced conversation using Pyversity diversification
- MCP server integration (Tavily hardcoded + custom)
- CLI tool integration (qwen, gemini)
- Multi-provider support (ZenMux, OpenAI, OpenRouter, Nebius, Ollama)

Usage:
    python -m python.app  # Run the Gradio app
"""

__version__ = "2.0.0"
__author__ = "nbiish"


def create_app():
    """Lazily import and create the Gradio app."""
    from .app import create_app as _create_app
    return _create_app()


def main():
    """Lazily import and run the main function."""
    from .app import main as _main
    return _main()


__all__ = ["create_app", "main", "__version__"]
