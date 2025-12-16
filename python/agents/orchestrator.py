"""
DeepAgent Orchestrator for HGM integration.

Purpose: Bridge between HGM self-improving engine and LangChain agents
Inputs: HGM agent configurations, task requests
Outputs: Orchestrated multi-agent responses

Architecture: DeepHGM Chimera (LangChain DeepAgents + Huxley-Gödel Machine)
"""

from typing import Any, AsyncIterator, Dict, List, Optional
from .supervisor import SupervisorAgent
from .workers import DocumentWorker, VoiceWorker, CLIWorker
from .tools import HunyuanOCRTool

# Import DeepHGM components
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hgm import (
    HGMAgent,
    DeepHGMAgent,
    create_deep_hgm_agent,
    UCBAir,
    ThompsonSampler,
    CladeMetrics,
    ActionType,
)


class DeepAgentOrchestrator:
    """
    Orchestrates LangChain DeepAgents with HGM self-improving capabilities.
    
    This orchestrator implements the DeepHGM Chimera architecture:
    1. DeepAgents: Planning, filesystem backend, sub-agent delegation
    2. HGM: CMP scoring, UCB-Air branch selection, Thompson Sampling
    3. Self-improvement through agent evolution
    4. Minimal-context tools (OCR, etc.) for efficiency
    """
    
    def __init__(self, model: str = "gpt-4", workspace_path: Optional[str] = None):
        self.model = model
        self.workspace_path = workspace_path
        self.tools = self._create_tools()
        self.supervisor = self._create_supervisor()
        self.task_history: List[Dict[str, Any]] = []
        self.performance_metrics: Dict[str, float] = {}
        
        # DeepHGM components
        self.ucb = UCBAir(exploration_constant=2.0)
        self.thompson = ThompsonSampler()
        self.clades: Dict[str, CladeMetrics] = {}
        self.active_agents: Dict[str, DeepHGMAgent] = {}
    
    def _create_tools(self) -> List[Any]:
        """Create minimal tool set for HGM agents."""
        return [
            HunyuanOCRTool(workspace_path=self.workspace_path),
        ]
    
    def _create_supervisor(self) -> SupervisorAgent:
        """Create supervisor with default workers."""
        workers = [
            DocumentWorker(),
            VoiceWorker(),
            CLIWorker(),
        ]
        return SupervisorAgent(model=self.model, workers=workers)
    
    def process_request(self, request: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a user request through the agent hierarchy.
        
        Args:
            request: User's request string
            context: Optional context from previous interactions
            
        Returns:
            Response dictionary with results and metadata
        """
        input_data = {
            "messages": [{"role": "user", "content": request}],
            "context": context or {},
        }
        
        # Invoke supervisor
        result = self.supervisor.invoke(input_data)
        
        # Track for HGM metrics
        self._track_task(request, result)
        
        return result
    
    def _track_task(self, request: str, result: Dict[str, Any]) -> None:
        """Track task execution for HGM performance metrics."""
        task_record = {
            "request": request,
            "response": result.get("response", ""),
            "success": self._evaluate_success(result),
        }
        self.task_history.append(task_record)
        
        # Update performance metrics
        successes = sum(1 for t in self.task_history if t["success"])
        self.performance_metrics["success_rate"] = successes / len(self.task_history)
    
    def _evaluate_success(self, result: Dict[str, Any]) -> bool:
        """Evaluate if the task was successful."""
        # Simple heuristic - would be more sophisticated in production
        response = result.get("response", "")
        failure_indicators = ["error", "failed", "unable", "cannot"]
        return not any(ind in response.lower() for ind in failure_indicators)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics for HGM."""
        return {
            "total_tasks": len(self.task_history),
            "success_rate": self.performance_metrics.get("success_rate", 0.0),
            "active_workers": len(self.supervisor.workers),
        }
    
    def update_from_hgm(self, agent_config: Dict[str, Any]) -> None:
        """
        Update orchestrator based on HGM agent evolution.
        
        Args:
            agent_config: New agent configuration from HGM
        """
        # Apply prompt template updates
        if "prompt_template" in agent_config:
            # Would update worker prompts based on HGM evolution
            pass
        
        # Apply worker configuration updates
        if "worker_config" in agent_config:
            # Would reconfigure workers based on HGM evolution
            pass
    
    def get_tools(self) -> List[Any]:
        """Get available tools for HGM agents."""
        return self.tools
    
    def invoke_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """
        Invoke a specific tool by name.
        
        Args:
            tool_name: Name of the tool to invoke
            **kwargs: Tool-specific arguments
            
        Returns:
            Tool execution result
        """
        for tool in self.tools:
            if tool.name == tool_name:
                return tool._run(**kwargs)
        return {"error": f"Tool '{tool_name}' not found", "success": False}
    
    def create_deep_hgm_agent(self, agent_id: str, agent_name: str) -> DeepHGMAgent:
        """
        Create a DeepHGM agent for the given agent configuration.
        
        Args:
            agent_id: Unique agent identifier
            agent_name: Human-readable agent name
            
        Returns:
            Configured DeepHGMAgent
        """
        hgm_agent = HGMAgent(
            id=agent_id,
            name=agent_name,
        )
        
        deep_agent = create_deep_hgm_agent(
            agent=hgm_agent,
            tools=self.tools,
            workspace_path=self.workspace_path,
        )
        
        self.active_agents[agent_id] = deep_agent
        return deep_agent
    
    def get_or_create_agent(self, agent_id: str, agent_name: str) -> DeepHGMAgent:
        """Get existing agent or create new one."""
        if agent_id not in self.active_agents:
            return self.create_deep_hgm_agent(agent_id, agent_name)
        return self.active_agents[agent_id]
    
    def process_with_deep_hgm(
        self,
        agent_id: str,
        agent_name: str,
        request: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Process request using DeepHGM agent architecture.
        
        This is the main entry point for the chimera architecture:
        1. Gets or creates DeepHGM agent
        2. Uses UCB-Air for branch selection
        3. Uses Thompson Sampling for action selection
        4. Tracks CMP for self-improvement
        
        Args:
            agent_id: Agent identifier
            agent_name: Agent name
            request: User request
            context: Optional context
            
        Returns:
            Response with agent state and metrics
        """
        agent = self.get_or_create_agent(agent_id, agent_name)
        
        # Select branch using UCB-Air
        available_branches = list(self.clades.keys()) or [agent.clade_id]
        selected_branch = self.ucb.select_branch(available_branches)
        
        # Invoke agent
        result = agent.invoke({
            "messages": [{"role": "user", "content": request}],
            "context": context or {},
            "branch": selected_branch,
        })
        
        # Update UCB with result
        success = result.get("success", False)
        self.ucb.update(selected_branch, 1.0 if success else 0.0)
        
        # Update clade metrics
        if selected_branch not in self.clades:
            self.clades[selected_branch] = CladeMetrics(clade_id=selected_branch)
        self.clades[selected_branch].add_descendant_result(
            success=success,
            score=result.get("state", {}).get("cmp_score", 0.5),
        )
        
        # Track task
        self._track_task(request, result)
        
        return {
            **result,
            "branch": selected_branch,
            "clade_cmp": self.clades[selected_branch].cmp_score(),
            "ucb_scores": {b: self.ucb.ucb_score(b) for b in available_branches},
        }
    
    def evolve_agent(self, agent_id: str) -> Optional[DeepHGMAgent]:
        """
        Attempt to evolve an agent based on its CMP score.
        
        Returns new agent if evolution occurred, None otherwise.
        """
        if agent_id not in self.active_agents:
            return None
        
        parent = self.active_agents[agent_id]
        state = parent.create_initial_state("")
        
        child = parent.evolve(state)
        if child:
            self.active_agents[child.agent.id] = child
            
            # Create new clade for child
            self.clades[child.clade_id] = CladeMetrics(
                clade_id=child.clade_id,
                agent_ids=[child.agent.id],
            )
            
        return child
    
    async def astream_with_deep_hgm(
        self,
        agent_id: str,
        agent_name: str,
        request: str,
        context: Optional[Dict[str, Any]] = None,
        provider=None,
        chat_model=None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Stream response using DeepHGM agent architecture.
        
        Token-level streaming for real-time response delivery.
        
        Args:
            agent_id: Agent identifier
            agent_name: Agent name
            request: User request
            context: Optional context
            provider: Optional OpenAICompatibleProvider for streaming
            chat_model: Optional LangChain chat model for streaming
            
        Yields:
            Dict with event type and content:
            - {"type": "token", "content": "..."}
            - {"type": "tool_start", "content": "tool_name"}
            - {"type": "tool_end", "content": "result"}
            - {"type": "metadata", "metadata": {...}}
        """
        agent = self.get_or_create_agent(agent_id, agent_name)
        
        # Select branch using UCB-Air
        available_branches = list(self.clades.keys()) or [agent.clade_id]
        selected_branch = self.ucb.select_branch(available_branches)
        
        # Yield branch selection info
        yield {
            "type": "branch_selected",
            "content": selected_branch,
            "metadata": {"ucb_scores": {b: self.ucb.ucb_score(b) for b in available_branches}},
        }
        
        # Stream agent response
        full_response = ""
        tools_used = []
        success = True
        
        async for event in agent.astream(
            {
                "messages": [{"role": "user", "content": request}],
                "context": context or {},
                "branch": selected_branch,
            },
            provider=provider,
            chat_model=chat_model,
        ):
            yield event
            
            if event.get("type") == "token":
                full_response += event.get("content", "")
            elif event.get("type") == "tool_end":
                tools_used.append(event.get("content", ""))
            elif event.get("type") == "error":
                success = False
            elif event.get("type") == "metadata":
                # Extract success from metadata
                metadata = event.get("metadata", {})
                success = metadata.get("success", success)
        
        # Update UCB with result
        self.ucb.update(selected_branch, 1.0 if success else 0.0)
        
        # Update clade metrics
        if selected_branch not in self.clades:
            self.clades[selected_branch] = CladeMetrics(clade_id=selected_branch)
        
        cmp_score = agent.agent.cmp_score if hasattr(agent, 'agent') else 0.5
        self.clades[selected_branch].add_descendant_result(
            success=success,
            score=cmp_score,
        )
        
        # Track task
        self._track_task(request, {"response": full_response, "success": success})
        
        # Yield final orchestrator metadata
        yield {
            "type": "orchestrator_complete",
            "content": "",
            "metadata": {
                "branch": selected_branch,
                "clade_cmp": self.clades[selected_branch].cmp_score(),
                "ucb_scores": {b: self.ucb.ucb_score(b) for b in available_branches},
                "tools_used": tools_used,
                "success": success,
            },
        }
    
    async def astream_text(
        self,
        agent_id: str,
        agent_name: str,
        request: str,
        context: Optional[Dict[str, Any]] = None,
        provider=None,
        chat_model=None,
    ) -> AsyncIterator[str]:
        """
        Stream just text tokens from DeepHGM agent.
        
        Convenience method for simple text streaming.
        """
        async for event in self.astream_with_deep_hgm(
            agent_id, agent_name, request, context, provider, chat_model
        ):
            if event.get("type") == "token":
                yield event.get("content", "")
