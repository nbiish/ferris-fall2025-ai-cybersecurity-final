"""
IBM Granite Model Setup Script.

Purpose: Download and configure IBM Granite models for local RAG
Inputs: Model selection, Ollama configuration
Outputs: Ready-to-use local Granite models

Usage:
    python -m rag.granite_setup --all          # Install all recommended models
    python -m rag.granite_setup --embedding    # Install embedding model only
    python -m rag.granite_setup --llm          # Install LLM only
"""

import subprocess
import sys
from typing import List, Optional


# Recommended Granite models for RAG
GRANITE_EMBEDDING_MODELS = [
    ("granite-embedding:278m", "Multilingual embedding (563MB, 768 dims)"),
    ("granite-embedding:30m", "English-only embedding (63MB, 768 dims, faster)"),
]

GRANITE_LLM_MODELS = [
    ("granite3.1-dense:2b", "2B parameter LLM (1.6GB, 128K context)"),
    ("granite3.1-dense:8b", "8B parameter LLM (5.0GB, 128K context)"),
]


def check_ollama_installed() -> bool:
    """Check if Ollama is installed and running."""
    try:
        result = subprocess.run(
            ["ollama", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def check_ollama_running() -> bool:
    """Check if Ollama server is running."""
    try:
        import httpx
        response = httpx.get("http://localhost:11434/api/tags", timeout=5.0)
        return response.status_code == 200
    except Exception:
        return False


def pull_model(model_name: str) -> bool:
    """Pull a model from Ollama."""
    print(f"  Pulling {model_name}...")
    try:
        result = subprocess.run(
            ["ollama", "pull", model_name],
            capture_output=False,
            timeout=600,  # 10 minute timeout for large models
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"  ⚠ Timeout pulling {model_name}")
        return False
    except Exception as e:
        print(f"  ✗ Error pulling {model_name}: {e}")
        return False


def list_installed_models() -> List[str]:
    """List installed Ollama models."""
    try:
        import httpx
        response = httpx.get("http://localhost:11434/api/tags", timeout=10.0)
        if response.status_code == 200:
            data = response.json()
            return [m["name"] for m in data.get("models", [])]
    except Exception:
        pass
    return []


def setup_granite_models(
    install_embedding: bool = True,
    install_llm: bool = False,
    embedding_model: str = "278m",
    llm_model: str = "2b",
) -> bool:
    """
    Set up IBM Granite models for local RAG.
    
    Args:
        install_embedding: Install embedding model
        install_llm: Install LLM model
        embedding_model: Embedding model size ("278m" or "30m")
        llm_model: LLM model size ("2b" or "8b")
        
    Returns:
        True if all requested models installed successfully
    """
    print("=" * 60)
    print("IBM Granite Model Setup for Local RAG")
    print("=" * 60)
    
    # Check Ollama
    if not check_ollama_installed():
        print("\n✗ Ollama is not installed.")
        print("  Please install Ollama from: https://ollama.com/download")
        return False
    
    print("\n✓ Ollama is installed")
    
    if not check_ollama_running():
        print("\n⚠ Ollama server is not running.")
        print("  Starting Ollama...")
        try:
            subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            import time
            time.sleep(3)
            if not check_ollama_running():
                print("  ✗ Failed to start Ollama server")
                return False
        except Exception as e:
            print(f"  ✗ Error starting Ollama: {e}")
            return False
    
    print("✓ Ollama server is running")
    
    # Check installed models
    installed = list_installed_models()
    print(f"\nCurrently installed models: {len(installed)}")
    for m in installed:
        print(f"  - {m}")
    
    success = True
    
    # Install embedding model
    if install_embedding:
        model_name = f"granite-embedding:{embedding_model}"
        print(f"\n📦 Installing embedding model: {model_name}")
        
        if any(model_name in m for m in installed):
            print(f"  ✓ {model_name} already installed")
        else:
            if pull_model(model_name):
                print(f"  ✓ {model_name} installed successfully")
            else:
                print(f"  ✗ Failed to install {model_name}")
                success = False
    
    # Install LLM
    if install_llm:
        model_name = f"granite3.1-dense:{llm_model}"
        print(f"\n📦 Installing LLM: {model_name}")
        
        if any(model_name in m for m in installed):
            print(f"  ✓ {model_name} already installed")
        else:
            if pull_model(model_name):
                print(f"  ✓ {model_name} installed successfully")
            else:
                print(f"  ✗ Failed to install {model_name}")
                success = False
    
    # Summary
    print("\n" + "=" * 60)
    if success:
        print("✓ Granite model setup complete!")
        print("\nYou can now run the RAG server with:")
        print("  python -m rag.api --embedding-provider granite-ollama")
    else:
        print("⚠ Some models failed to install. Check the errors above.")
    print("=" * 60)
    
    return success


def test_embedding() -> bool:
    """Test that Granite embedding is working."""
    print("\n🧪 Testing Granite embedding...")
    
    try:
        from .embeddings import GraniteOllamaEmbeddings
        
        provider = GraniteOllamaEmbeddings(model="granite-embedding:278m")
        embedding = provider.embed_single("Hello, this is a test.")
        
        print(f"  ✓ Embedding generated successfully")
        print(f"  ✓ Embedding dimension: {len(embedding)}")
        return True
    except Exception as e:
        print(f"  ✗ Embedding test failed: {e}")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Set up IBM Granite models for local RAG"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Install all recommended models (embedding + LLM)",
    )
    parser.add_argument(
        "--embedding",
        action="store_true",
        help="Install embedding model only",
    )
    parser.add_argument(
        "--llm",
        action="store_true",
        help="Install LLM only",
    )
    parser.add_argument(
        "--embedding-size",
        choices=["278m", "30m"],
        default="278m",
        help="Embedding model size (default: 278m multilingual)",
    )
    parser.add_argument(
        "--llm-size",
        choices=["2b", "8b"],
        default="2b",
        help="LLM size (default: 2b)",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test embedding after setup",
    )
    
    args = parser.parse_args()
    
    # Default to embedding only if no flags specified
    install_embedding = args.embedding or args.all or (not args.llm)
    install_llm = args.llm or args.all
    
    success = setup_granite_models(
        install_embedding=install_embedding,
        install_llm=install_llm,
        embedding_model=args.embedding_size,
        llm_model=args.llm_size,
    )
    
    if success and args.test:
        test_embedding()
    
    sys.exit(0 if success else 1)
