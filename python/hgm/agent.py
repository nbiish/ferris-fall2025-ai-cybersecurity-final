"""
HGM Agent Python Interface.

Purpose: Python representation of HGM agents for LangChain integration
Inputs: Agent configuration from Rust backend
Outputs: Agent behavior and self-modification capabilities
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
import uuid


@dataclass
class HGMAgent:
    """
    Python representation of an HGM agent.
    
    Mirrors the Rust Agent struct for cross-language compatibility.
    """
    
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "Agent"
    generation: int = 0
    parent_id: Optional[str] = None
    success_count: int = 0
    failure_count: int = 0
    cmp_score: float = 0.5
    prompt_template: str = ""
    modifications: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.prompt_template:
            self.prompt_template = self._default_prompt_template()
    
    @classmethod
    def initial(cls) -> "HGMAgent":
        """Create the initial root agent."""
        return cls(
            name="Initial Agent",
            generation=0,
        )
    
    @classmethod
    def from_parent(cls, parent: "HGMAgent") -> "HGMAgent":
        """Create a child agent from a parent."""
        agent_id = str(uuid.uuid4())
        return cls(
            id=agent_id,
            name=f"Agent-{agent_id[:8]}",
            generation=parent.generation + 1,
            parent_id=parent.id,
            cmp_score=parent.cmp_score,
            prompt_template=parent.prompt_template,
            modifications=parent.modifications.copy(),
        )
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HGMAgent":
        """Create agent from dictionary (e.g., from Rust backend)."""
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            name=data.get("name", "Agent"),
            generation=data.get("generation", 0),
            parent_id=data.get("parent_id"),
            success_count=data.get("success_count", 0),
            failure_count=data.get("failure_count", 0),
            cmp_score=data.get("cmp_score", 0.5),
            prompt_template=data.get("prompt_template", ""),
            modifications=data.get("modifications", []),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert agent to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "generation": self.generation,
            "parent_id": self.parent_id,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "cmp_score": self.cmp_score,
            "prompt_template": self.prompt_template,
            "modifications": self.modifications,
        }
    
    def empirical_mean(self) -> float:
        """Calculate empirical success rate."""
        total = self.success_count + self.failure_count
        if total == 0:
            return 0.5
        return self.success_count / total
    
    def total_evaluations(self) -> int:
        """Get total number of evaluations."""
        return self.success_count + self.failure_count
    
    def record_success(self) -> None:
        """Record a successful evaluation."""
        self.success_count += 1
    
    def record_failure(self) -> None:
        """Record a failed evaluation."""
        self.failure_count += 1
    
    def add_modification(self, modification: str) -> None:
        """Add a self-modification to the agent's history."""
        self.modifications.append(modification)
    
    def update_prompt(self, new_template: str) -> None:
        """Update the agent's prompt template."""
        self.prompt_template = new_template
        self.add_modification(f"Updated prompt template")
    
    def _default_prompt_template(self) -> str:
        """Get the default prompt template."""
        return """You are a self-improving coding agent. Your task is to:
1. Analyze the given problem
2. Generate a solution
3. Evaluate the solution
4. Propose improvements to your own approach

Current task: {task}
Previous attempts: {history}
"""
