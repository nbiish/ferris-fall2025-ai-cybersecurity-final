"""
RAG HTTP API Server.

Purpose: Expose RAG service via HTTP for frontend communication
Inputs: HTTP requests for indexing, retrieval, and management
Outputs: JSON responses with results and status

Run with: python -m rag.api
"""

import asyncio
import json
from typing import Any, Dict, List, Optional
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import threading

from .service import RAGService
from .embeddings import create_embedding_provider


class RAGAPIHandler(BaseHTTPRequestHandler):
    """HTTP request handler for RAG API."""
    
    rag_service: Optional[RAGService] = None
    
    def _send_json(self, data: Any, status: int = 200) -> None:
        """Send JSON response."""
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())
    
    def _read_json(self) -> Dict[str, Any]:
        """Read JSON request body."""
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length == 0:
            return {}
        body = self.rfile.read(content_length)
        return json.loads(body.decode())
    
    def do_OPTIONS(self) -> None:
        """Handle CORS preflight."""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
    
    def do_GET(self) -> None:
        """Handle GET requests."""
        parsed = urlparse(self.path)
        path = parsed.path
        
        if path == "/api/rag/stats":
            self._handle_stats()
        elif path == "/api/rag/sources":
            self._handle_get_sources()
        elif path == "/api/rag/health":
            self._send_json({"status": "ok"})
        else:
            self._send_json({"error": "Not found"}, 404)
    
    def do_POST(self) -> None:
        """Handle POST requests."""
        parsed = urlparse(self.path)
        path = parsed.path
        
        try:
            data = self._read_json()
            
            if path == "/api/rag/index/conversation":
                self._handle_index_conversation(data)
            elif path == "/api/rag/index/folder":
                self._handle_index_folder(data)
            elif path == "/api/rag/index/output":
                self._handle_index_output(data)
            elif path == "/api/rag/retrieve":
                self._handle_retrieve(data)
            elif path == "/api/rag/clear":
                self._handle_clear()
            elif path == "/api/rag/clear-recent":
                self._handle_clear_recent()
            else:
                self._send_json({"error": "Not found"}, 404)
        except Exception as e:
            self._send_json({"error": str(e)}, 500)
    
    def do_DELETE(self) -> None:
        """Handle DELETE requests."""
        parsed = urlparse(self.path)
        path = parsed.path
        
        if path.startswith("/api/rag/sources/"):
            source_key = path.replace("/api/rag/sources/", "")
            self._handle_remove_source(source_key)
        else:
            self._send_json({"error": "Not found"}, 404)
    
    def _handle_stats(self) -> None:
        """Get RAG statistics."""
        if not self.rag_service:
            self._send_json({"error": "RAG service not initialized"}, 500)
            return
        
        stats = self.rag_service.stats()
        self._send_json({
            "totalChunks": stats.get("total_chunks", 0),
            "totalSources": stats.get("total_sources", 0),
            "embeddingDim": stats.get("embedding_dim", 0),
            "pyversityAvailable": stats.get("pyversity_available", False),
            "recentContextSize": stats.get("recent_context_size", 0),
        })
    
    def _handle_get_sources(self) -> None:
        """Get indexed sources."""
        if not self.rag_service:
            self._send_json({})
            return
        
        sources = self.rag_service.get_indexed_sources()
        self._send_json(sources)
    
    def _handle_index_conversation(self, data: Dict[str, Any]) -> None:
        """Index agent conversation."""
        if not self.rag_service:
            self._send_json({"error": "RAG service not initialized"}, 500)
            return
        
        agent_id = data.get("agentId", "")
        messages = data.get("messages", [])
        
        chunk_count = self.rag_service.index_agent_conversation(agent_id, messages)
        self._send_json({"chunkCount": chunk_count})
    
    def _handle_index_folder(self, data: Dict[str, Any]) -> None:
        """Index folder."""
        if not self.rag_service:
            self._send_json({"error": "RAG service not initialized"}, 500)
            return
        
        folder_path = data.get("folderPath", "")
        agent_id = data.get("agentId")
        extensions = data.get("extensions")
        
        chunk_count = self.rag_service.index_folder(folder_path, agent_id, extensions)
        self._send_json({"chunkCount": chunk_count})
    
    def _handle_index_output(self, data: Dict[str, Any]) -> None:
        """Index agent output."""
        if not self.rag_service:
            self._send_json({"error": "RAG service not initialized"}, 500)
            return
        
        agent_id = data.get("agentId", "")
        output = data.get("output", "")
        output_type = data.get("outputType", "response")
        
        chunk_count = self.rag_service.index_agent_output(agent_id, output, output_type)
        self._send_json({"chunkCount": chunk_count})
    
    def _handle_retrieve(self, data: Dict[str, Any]) -> None:
        """Retrieve diversified context."""
        if not self.rag_service:
            self._send_json({"error": "RAG service not initialized"}, 500)
            return
        
        query = data.get("query", "")
        k = data.get("k", 5)
        diversity = data.get("diversity", 0.5)
        strategy = data.get("strategy", "ssd")
        agent_filter = data.get("agentFilter")
        
        result = self.rag_service.retrieve(
            query=query,
            k=k,
            diversity=diversity,
            strategy=strategy,
            agent_filter=agent_filter,
        )
        
        self._send_json({
            "chunks": [
                {
                    "id": c.id,
                    "content": c.content,
                    "sourceType": c.source_type,
                    "sourcePath": c.source_path,
                    "agentId": c.agent_id,
                    "score": score,
                    "metadata": c.metadata,
                }
                for c, score in zip(result.chunks, result.scores)
            ],
            "query": result.query,
            "strategy": result.strategy_used,
        })
    
    def _handle_remove_source(self, source_key: str) -> None:
        """Remove indexed source."""
        if not self.rag_service:
            self._send_json({"error": "RAG service not initialized"}, 500)
            return
        
        from urllib.parse import unquote
        source_key = unquote(source_key)
        removed_count = self.rag_service.remove_source(source_key)
        self._send_json({"removedCount": removed_count})
    
    def _handle_clear(self) -> None:
        """Clear all indexed content."""
        if not self.rag_service:
            self._send_json({"error": "RAG service not initialized"}, 500)
            return
        
        self.rag_service.clear_index()
        self._send_json({"status": "cleared"})
    
    def _handle_clear_recent(self) -> None:
        """Clear recent context."""
        if not self.rag_service:
            self._send_json({"error": "RAG service not initialized"}, 500)
            return
        
        self.rag_service.clear_recent_context()
        self._send_json({"status": "cleared"})
    
    def log_message(self, format: str, *args) -> None:
        """Suppress default logging."""
        pass


def create_server(
    host: str = "localhost",
    port: int = 8765,
    embedding_provider: str = "local",
    api_key: Optional[str] = None,
) -> HTTPServer:
    """
    Create and configure the RAG API server.
    
    Args:
        host: Server host
        port: Server port
        embedding_provider: Type of embedding provider
        api_key: API key for embedding provider
        
    Returns:
        Configured HTTPServer instance
    """
    # Initialize RAG service
    provider = create_embedding_provider(
        provider_type=embedding_provider,
        api_key=api_key,
    )
    RAGAPIHandler.rag_service = RAGService(embedding_provider=provider)
    
    server = HTTPServer((host, port), RAGAPIHandler)
    return server


def run_server(
    host: str = "localhost",
    port: int = 8765,
    embedding_provider: str = "granite-ollama",
    api_key: Optional[str] = None,
) -> None:
    """
    Run the RAG API server.
    
    Args:
        host: Server host
        port: Server port
        embedding_provider: Type of embedding provider
        api_key: API key for embedding provider
    """
    server = create_server(host, port, embedding_provider, api_key)
    print(f"[RAG API] Starting server on http://{host}:{port}")
    print(f"[RAG API] Embedding provider: {embedding_provider}")
    print("[RAG API] Press Ctrl+C to stop")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[RAG API] Shutting down...")
        server.shutdown()


if __name__ == "__main__":
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description="RAG API Server")
    parser.add_argument("--host", default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=8765, help="Server port")
    parser.add_argument(
        "--embedding-provider",
        choices=["granite-ollama", "granite-hf", "local", "openai", "custom"],
        default="granite-ollama",
        help="Embedding provider type (granite-ollama recommended for local use)",
    )
    parser.add_argument("--api-key", help="API key for embedding provider")
    
    args = parser.parse_args()
    
    # Try to get API key from environment if not provided
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    
    run_server(
        host=args.host,
        port=args.port,
        embedding_provider=args.embedding_provider,
        api_key=api_key,
    )
