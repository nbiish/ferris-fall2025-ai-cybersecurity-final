"""
Supervisor Agent for LangChain DeepAgents orchestration.

Purpose: Central coordinator for multi-agent workflows
Inputs: User requests, agent states
Outputs: Delegated tasks to worker agents
"""

from typing import Any, Dict, List, Optional
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool
from pydantic import BaseModel


class AgentState(BaseModel):
    messages: List[Dict[str, Any]] = []
    current_task: Optional[str] = None
    completed_tasks: List[str] = []
    active_workers: List[str] = []


class SupervisorAgent:
    """
    Central supervisor agent that coordinates specialized worker agents.
    
    Implements the supervisor pattern from LangChain multi-agent architecture:
    - Creates specialized sub-agents for different domains
    - Wraps sub-agents as tools for centralized orchestration
    - Manages task delegation and result aggregation
    """
    
    def __init__(self, model: str = "gpt-4", workers: Optional[List[Any]] = None):
        self.model = model
        self.workers = workers or []
        self.state = AgentState()
        self._worker_tools = self._create_worker_tools()
    
    def _create_worker_tools(self) -> List[Any]:
        """Wrap each worker agent as a callable tool."""
        tools = []
        for worker in self.workers:
            @tool(
                name=worker.name,
                description=worker.description
            )
            def call_worker(query: str, worker=worker) -> str:
                result = worker.invoke({"messages": [{"role": "user", "content": query}]})
                return result.get("response", "")
            tools.append(call_worker)
        return tools
    
    def add_worker(self, worker: Any) -> None:
        """Add a worker agent to the supervisor's pool."""
        self.workers.append(worker)
        self._worker_tools = self._create_worker_tools()
    
    def invoke(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input and delegate to appropriate workers.
        
        Args:
            input_data: Dictionary containing messages and task context
            
        Returns:
            Dictionary with response and updated state
        """
        messages = input_data.get("messages", [])
        self.state.messages.extend(messages)
        
        # Analyze task and determine which workers to invoke
        task = self._analyze_task(messages)
        self.state.current_task = task
        
        # Delegate to workers based on task type
        results = []
        for worker in self.workers:
            if self._should_invoke_worker(worker, task):
                self.state.active_workers.append(worker.name)
                result = worker.invoke({"messages": messages, "task": task})
                results.append(result)
        
        # Aggregate results
        response = self._aggregate_results(results)
        self.state.completed_tasks.append(task)
        self.state.current_task = None
        self.state.active_workers = []
        
        return {
            "response": response,
            "state": self.state.model_dump(),
        }
    
    def _analyze_task(self, messages: List[Dict[str, Any]]) -> str:
        """Analyze messages to determine the task type."""
        if not messages:
            return "unknown"
        
        last_message = messages[-1].get("content", "").lower()
        
        if any(kw in last_message for kw in ["document", "ocr", "extract", "pdf"]):
            return "document_processing"
        elif any(kw in last_message for kw in ["voice", "speak", "audio", "tts"]):
            return "voice_interaction"
        elif any(kw in last_message for kw in ["code", "execute", "run", "command"]):
            return "cli_execution"
        else:
            return "general"
    
    def _should_invoke_worker(self, worker: Any, task: str) -> bool:
        """Determine if a worker should handle the given task."""
        worker_tasks = getattr(worker, "supported_tasks", [])
        return task in worker_tasks or "general" in worker_tasks
    
    def _aggregate_results(self, results: List[Dict[str, Any]]) -> str:
        """Combine results from multiple workers into a coherent response."""
        if not results:
            return "No workers were able to process the request."
        
        responses = [r.get("response", "") for r in results if r.get("response")]
        return "\n\n".join(responses) if responses else "Task completed."
