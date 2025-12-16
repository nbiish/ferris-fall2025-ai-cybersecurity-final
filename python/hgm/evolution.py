"""
HGM Evolution Manager.

Purpose: Manage agent evolution and self-improvement in Python
Inputs: Agent archive, evaluation results
Outputs: Evolved agents, CMP estimates
"""

from typing import Any, Dict, List, Optional
import random
import math

from .agent import HGMAgent


class HGMEvolutionManager:
    """
    Manages HGM agent evolution process.
    
    Implements:
    - Clade-Metaproductivity (CMP) estimation
    - Thompson Sampling for agent selection
    - Expansion and evaluation policies
    """
    
    def __init__(self, alpha: float = 0.5, epsilon: float = 0.1):
        """
        Initialize evolution manager.
        
        Args:
            alpha: UCB-Air parameter for expansion/evaluation balance
            epsilon: Percentile for best-belief agent selection
        """
        self.alpha = alpha
        self.epsilon = epsilon
        self.agents: Dict[str, HGMAgent] = {}
        self.root_id: Optional[str] = None
        self.iteration = 0
    
    def add_agent(self, agent: HGMAgent) -> None:
        """Add an agent to the archive."""
        if self.root_id is None:
            self.root_id = agent.id
        self.agents[agent.id] = agent
    
    def get_agent(self, agent_id: str) -> Optional[HGMAgent]:
        """Get an agent by ID."""
        return self.agents.get(agent_id)
    
    def get_all_agents(self) -> List[HGMAgent]:
        """Get all agents in the archive."""
        return list(self.agents.values())
    
    def estimate_cmp(self, agent_id: str) -> float:
        """
        Estimate Clade-Metaproductivity for an agent.
        
        CMP = weighted sum of empirical means in clade,
        where weight = number of evaluations for each agent.
        """
        clade = self._get_clade(agent_id)
        if not clade:
            return 0.5
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for agent in clade:
            weight = agent.total_evaluations()
            if weight > 0:
                weighted_sum += weight * agent.empirical_mean()
                total_weight += weight
        
        if total_weight == 0:
            return 0.5
        
        return weighted_sum / total_weight
    
    def _get_clade(self, agent_id: str) -> List[HGMAgent]:
        """Get all agents in a clade (agent and all descendants)."""
        clade = []
        self._collect_clade(agent_id, clade)
        return clade
    
    def _collect_clade(self, agent_id: str, clade: List[HGMAgent]) -> None:
        """Recursively collect clade members."""
        agent = self.agents.get(agent_id)
        if agent is None:
            return
        
        clade.append(agent)
        
        # Find children
        for other in self.agents.values():
            if other.parent_id == agent_id:
                self._collect_clade(other.id, clade)
    
    def thompson_sample(self, agents: List[HGMAgent], tau: float = 1.0) -> Optional[HGMAgent]:
        """
        Select an agent using Thompson Sampling.
        
        Args:
            agents: List of agents to sample from
            tau: Temperature parameter for exploration-exploitation
            
        Returns:
            Selected agent
        """
        if not agents:
            return None
        
        samples = []
        for agent in agents:
            alpha = 1.0 + agent.success_count * tau
            beta = 1.0 + agent.failure_count * tau
            
            # Sample from Beta distribution
            sample = random.betavariate(max(alpha, 0.01), max(beta, 0.01))
            samples.append((sample, agent))
        
        # Return agent with highest sample
        samples.sort(key=lambda x: x[0], reverse=True)
        return samples[0][1]
    
    def should_expand(self) -> bool:
        """
        Determine whether to expand or evaluate.
        
        Uses UCB-Air strategy: expand when N^alpha >= m
        """
        threshold = self.iteration ** self.alpha
        return threshold >= len(self.agents)
    
    def expand(self) -> Optional[HGMAgent]:
        """
        Expand the agent tree by creating a new agent.
        
        Returns:
            Newly created child agent
        """
        agents = self.get_all_agents()
        parent = self.thompson_sample(agents)
        
        if parent is None:
            return None
        
        child = HGMAgent.from_parent(parent)
        self.add_agent(child)
        
        return child
    
    def evaluate(self, agent_id: str, success: bool) -> None:
        """
        Record an evaluation result for an agent.
        
        Args:
            agent_id: ID of the agent to evaluate
            success: Whether the evaluation was successful
        """
        agent = self.agents.get(agent_id)
        if agent is None:
            return
        
        if success:
            agent.record_success()
        else:
            agent.record_failure()
        
        # Update CMP scores for the clade
        self._update_clade_cmp(agent_id)
    
    def _update_clade_cmp(self, agent_id: str) -> None:
        """Update CMP scores for all ancestors."""
        current_id = agent_id
        
        while current_id is not None:
            agent = self.agents.get(current_id)
            if agent is None:
                break
            
            agent.cmp_score = self.estimate_cmp(current_id)
            current_id = agent.parent_id
    
    def step(self) -> Dict[str, Any]:
        """
        Execute one step of the HGM algorithm.
        
        Returns:
            Dictionary with step results
        """
        self.iteration += 1
        
        if self.should_expand():
            child = self.expand()
            return {
                "action": "expand",
                "agent_id": child.id if child else None,
                "iteration": self.iteration,
            }
        else:
            # Select agent for evaluation
            agents = self.get_all_agents()
            agent = self.thompson_sample(agents)
            
            if agent:
                # Simulate evaluation (would be actual task in production)
                success = random.random() > 0.3
                self.evaluate(agent.id, success)
                
                return {
                    "action": "evaluate",
                    "agent_id": agent.id,
                    "success": success,
                    "iteration": self.iteration,
                }
        
        return {"action": "none", "iteration": self.iteration}
    
    def get_best_agent(self) -> Optional[HGMAgent]:
        """Get the best-belief agent based on CMP scores."""
        if not self.agents:
            return None
        
        return max(self.agents.values(), key=lambda a: a.cmp_score)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current evolution metrics."""
        agents = self.get_all_agents()
        best = self.get_best_agent()
        
        return {
            "iteration": self.iteration,
            "num_agents": len(agents),
            "best_cmp": best.cmp_score if best else 0.0,
            "best_agent_id": best.id if best else None,
            "total_evaluations": sum(a.total_evaluations() for a in agents),
        }
