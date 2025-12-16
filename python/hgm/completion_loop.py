"""
HGM Completion Loop: Self-improving task execution until completion.

Purpose: Drive DeepHGM agents through iterative self-improvement until
task completion criteria are met (CMP threshold or explicit completion).

Architecture:
┌─────────────────────────────────────────────────────────────────────┐
│                    HGM Completion Loop                               │
├─────────────────────────────────────────────────────────────────────┤
│  1. PLAN    → Break task into steps                                 │
│  2. EXECUTE → Run current step with tools                           │
│  3. EVALUATE → CMP scoring of result                                │
│  4. DECIDE  → Complete? Continue? Evolve?                           │
│  5. ITERATE → Loop until completion or max iterations               │
└─────────────────────────────────────────────────────────────────────┘

Based on: https://github.com/metauto-ai/HGM
"""

from typing import Any, AsyncIterator, Dict, List, Optional, Callable, TypedDict
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import json
import time
from datetime import datetime

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage


class CompletionStatus(Enum):
    """Status of task completion."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    NEEDS_IMPROVEMENT = "needs_improvement"
    MAX_ITERATIONS = "max_iterations"


class LoopAction(Enum):
    """Actions the loop can take."""
    CONTINUE = "continue"
    COMPLETE = "complete"
    EVOLVE = "evolve"
    RETRY = "retry"
    ABORT = "abort"


@dataclass
class CompletionCriteria:
    """Criteria for determining task completion."""
    
    cmp_threshold: float = 0.85
    max_iterations: int = 10
    min_iterations: int = 1
    require_explicit_completion: bool = False
    success_streak_required: int = 2
    
    def is_complete(
        self,
        cmp_score: float,
        iteration: int,
        success_streak: int,
        explicit_complete: bool = False,
    ) -> bool:
        """Check if completion criteria are met."""
        if iteration < self.min_iterations:
            return False
        
        if self.require_explicit_completion and not explicit_complete:
            return False
        
        if cmp_score >= self.cmp_threshold and success_streak >= self.success_streak_required:
            return True
        
        return False


@dataclass
class LoopState:
    """State maintained across loop iterations."""
    
    task: str
    iteration: int = 0
    cmp_score: float = 0.5
    cmp_history: List[float] = field(default_factory=list)
    success_streak: int = 0
    status: CompletionStatus = CompletionStatus.PENDING
    plan: List[Dict[str, Any]] = field(default_factory=list)
    current_step: int = 0
    results: List[Dict[str, Any]] = field(default_factory=list)
    messages: List[Dict[str, Any]] = field(default_factory=list)
    tools_used: List[str] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    last_action: Optional[LoopAction] = None
    error_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "task": self.task,
            "iteration": self.iteration,
            "cmp_score": self.cmp_score,
            "cmp_history": self.cmp_history,
            "success_streak": self.success_streak,
            "status": self.status.value,
            "plan": self.plan,
            "current_step": self.current_step,
            "results": self.results,
            "tools_used": self.tools_used,
            "elapsed_seconds": time.time() - self.start_time,
            "last_action": self.last_action.value if self.last_action else None,
            "error_count": self.error_count,
        }


class CMPEvaluator:
    """
    Clade Metaproductivity (CMP) Evaluator.
    
    Evaluates agent responses for quality and task completion.
    Uses multiple heuristics combined with optional LLM evaluation.
    """
    
    def __init__(
        self,
        llm_evaluator: Optional[Callable[[str, str], float]] = None,
        alpha: float = 0.3,
    ):
        self.llm_evaluator = llm_evaluator
        self.alpha = alpha  # Learning rate for EMA
    
    def evaluate(
        self,
        task: str,
        response: str,
        previous_cmp: float,
        tools_used: List[str],
    ) -> Dict[str, Any]:
        """
        Evaluate response quality and compute new CMP score.
        
        Returns dict with:
        - cmp_score: Updated CMP score
        - success: Whether this iteration was successful
        - feedback: Evaluation feedback
        - improvement_suggestions: List of suggestions
        """
        scores = {}
        
        # Heuristic 1: Response length (not too short, not too long)
        response_len = len(response)
        if response_len < 50:
            scores["length"] = 0.3
        elif response_len < 200:
            scores["length"] = 0.6
        elif response_len < 2000:
            scores["length"] = 0.9
        else:
            scores["length"] = 0.7  # Too verbose
        
        # Heuristic 2: Tool usage (using tools is generally good)
        if tools_used:
            scores["tool_usage"] = min(1.0, 0.5 + len(tools_used) * 0.15)
        else:
            scores["tool_usage"] = 0.5
        
        # Heuristic 3: Completion indicators
        completion_phrases = [
            "completed", "done", "finished", "successfully",
            "here is", "the result", "solution", "answer",
        ]
        completion_count = sum(1 for p in completion_phrases if p in response.lower())
        scores["completion_indicators"] = min(1.0, 0.4 + completion_count * 0.1)
        
        # Heuristic 4: Error indicators (negative)
        error_phrases = ["error", "failed", "cannot", "unable", "sorry"]
        error_count = sum(1 for p in error_phrases if p in response.lower())
        scores["no_errors"] = max(0.2, 1.0 - error_count * 0.15)
        
        # Heuristic 5: Task relevance (simple keyword matching)
        task_words = set(task.lower().split())
        response_words = set(response.lower().split())
        overlap = len(task_words & response_words)
        scores["relevance"] = min(1.0, 0.3 + overlap * 0.1)
        
        # LLM evaluation if available
        if self.llm_evaluator:
            try:
                llm_score = self.llm_evaluator(task, response)
                scores["llm_eval"] = llm_score
            except Exception:
                pass
        
        # Weighted average of scores
        weights = {
            "length": 0.1,
            "tool_usage": 0.2,
            "completion_indicators": 0.25,
            "no_errors": 0.25,
            "relevance": 0.2,
        }
        
        if "llm_eval" in scores:
            weights["llm_eval"] = 0.3
            # Normalize other weights
            total = sum(v for k, v in weights.items() if k != "llm_eval")
            for k in weights:
                if k != "llm_eval":
                    weights[k] *= 0.7 / total
        
        raw_score = sum(scores.get(k, 0.5) * w for k, w in weights.items())
        
        # Apply EMA to smooth score changes
        new_cmp = self.alpha * raw_score + (1 - self.alpha) * previous_cmp
        
        # Determine success
        success = raw_score >= 0.6 and scores.get("no_errors", 1.0) >= 0.7
        
        # Generate feedback
        feedback_parts = []
        if scores["length"] < 0.5:
            feedback_parts.append("Response too short")
        if scores["no_errors"] < 0.7:
            feedback_parts.append("Contains error indicators")
        if scores["relevance"] < 0.5:
            feedback_parts.append("May not address task directly")
        if scores["completion_indicators"] < 0.5:
            feedback_parts.append("No clear completion signal")
        
        # Improvement suggestions
        suggestions = []
        if not tools_used:
            suggestions.append("Consider using available tools")
        if scores["length"] < 0.5:
            suggestions.append("Provide more detailed response")
        if scores["relevance"] < 0.5:
            suggestions.append("Focus more on the specific task")
        
        return {
            "cmp_score": new_cmp,
            "raw_score": raw_score,
            "success": success,
            "component_scores": scores,
            "feedback": "; ".join(feedback_parts) if feedback_parts else "Good response",
            "improvement_suggestions": suggestions,
        }


class HGMCompletionLoop:
    """
    Self-improving completion loop for HGM agents.
    
    Drives agent execution through iterative improvement until
    task completion criteria are met.
    """
    
    def __init__(
        self,
        agent_invoke: Callable[[Dict[str, Any]], Dict[str, Any]],
        criteria: Optional[CompletionCriteria] = None,
        evaluator: Optional[CMPEvaluator] = None,
        on_iteration: Optional[Callable[[LoopState], None]] = None,
        on_complete: Optional[Callable[[LoopState], None]] = None,
    ):
        """
        Initialize completion loop.
        
        Args:
            agent_invoke: Function to invoke agent with messages
            criteria: Completion criteria
            evaluator: CMP evaluator
            on_iteration: Callback after each iteration
            on_complete: Callback on completion
        """
        self.agent_invoke = agent_invoke
        self.criteria = criteria or CompletionCriteria()
        self.evaluator = evaluator or CMPEvaluator()
        self.on_iteration = on_iteration
        self.on_complete = on_complete
    
    def _create_planning_prompt(self, task: str) -> str:
        """Create prompt for planning phase."""
        return f"""Analyze this task and create a step-by-step plan:

TASK: {task}

Create a concise plan with 2-5 steps. For each step, specify:
1. What action to take
2. What tool to use (if any)
3. Expected outcome

Format your response as a numbered list of steps."""
    
    def _create_execution_prompt(
        self,
        task: str,
        plan: List[Dict[str, Any]],
        current_step: int,
        previous_results: List[Dict[str, Any]],
    ) -> str:
        """Create prompt for execution phase."""
        plan_text = "\n".join(
            f"{i+1}. {step.get('description', step)}"
            for i, step in enumerate(plan)
        )
        
        results_text = ""
        if previous_results:
            results_text = "\n\nPrevious results:\n" + "\n".join(
                f"- Step {r.get('step', '?')}: {r.get('summary', r.get('result', ''))[:200]}"
                for r in previous_results[-3:]
            )
        
        current = plan[current_step] if current_step < len(plan) else {"description": "Complete task"}
        
        return f"""Execute the current step of this task:

TASK: {task}

PLAN:
{plan_text}

CURRENT STEP ({current_step + 1}/{len(plan)}): {current.get('description', current)}
{results_text}

Execute this step and provide the result. Be specific and actionable."""
    
    def _create_improvement_prompt(
        self,
        task: str,
        feedback: str,
        suggestions: List[str],
        previous_response: str,
    ) -> str:
        """Create prompt for self-improvement iteration."""
        suggestions_text = "\n".join(f"- {s}" for s in suggestions) if suggestions else "None"
        
        return f"""Improve your previous response based on feedback:

TASK: {task}

PREVIOUS RESPONSE:
{previous_response[:500]}...

FEEDBACK: {feedback}

IMPROVEMENT SUGGESTIONS:
{suggestions_text}

Provide an improved response that addresses the feedback."""
    
    def _parse_plan(self, response: str) -> List[Dict[str, Any]]:
        """Parse plan from agent response."""
        lines = response.strip().split("\n")
        plan = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Try to parse numbered items
            if line[0].isdigit() and ("." in line[:3] or ")" in line[:3]):
                # Remove number prefix
                content = line.split(".", 1)[-1].strip() if "." in line[:3] else line.split(")", 1)[-1].strip()
                plan.append({
                    "id": f"step-{len(plan)+1}",
                    "description": content,
                    "status": "pending",
                })
        
        # If no numbered items found, treat whole response as single step
        if not plan:
            plan.append({
                "id": "step-1",
                "description": response[:200],
                "status": "pending",
            })
        
        return plan
    
    def run(self, task: str, initial_cmp: float = 0.5) -> LoopState:
        """
        Run the completion loop synchronously.
        
        Args:
            task: Task description
            initial_cmp: Initial CMP score
            
        Returns:
            Final LoopState
        """
        state = LoopState(task=task, cmp_score=initial_cmp)
        state.status = CompletionStatus.IN_PROGRESS
        
        # Phase 1: Planning
        planning_prompt = self._create_planning_prompt(task)
        state.messages.append({"role": "user", "content": planning_prompt})
        
        try:
            plan_response = self.agent_invoke({
                "messages": [{"role": "user", "content": planning_prompt}]
            })
            plan_text = plan_response.get("response", "")
            state.plan = self._parse_plan(plan_text)
            state.messages.append({"role": "assistant", "content": plan_text})
            state.tools_used.extend(plan_response.get("state", {}).get("tools_used", []))
        except Exception as e:
            state.status = CompletionStatus.FAILED
            state.messages.append({"role": "system", "content": f"Planning failed: {e}"})
            return state
        
        # Phase 2: Iterative Execution
        while state.iteration < self.criteria.max_iterations:
            state.iteration += 1
            
            # Check if all plan steps completed
            if state.current_step >= len(state.plan):
                # All steps done, check completion
                if self.criteria.is_complete(
                    state.cmp_score,
                    state.iteration,
                    state.success_streak,
                ):
                    state.status = CompletionStatus.COMPLETED
                    break
                else:
                    # Need improvement
                    state.status = CompletionStatus.NEEDS_IMPROVEMENT
            
            # Execute current step or improve
            if state.status == CompletionStatus.NEEDS_IMPROVEMENT and state.results:
                # Improvement iteration
                last_result = state.results[-1]
                prompt = self._create_improvement_prompt(
                    task,
                    last_result.get("feedback", ""),
                    last_result.get("suggestions", []),
                    last_result.get("response", ""),
                )
            else:
                # Normal execution
                prompt = self._create_execution_prompt(
                    task,
                    state.plan,
                    state.current_step,
                    state.results,
                )
            
            state.messages.append({"role": "user", "content": prompt})
            
            try:
                response = self.agent_invoke({
                    "messages": state.messages
                })
                response_text = response.get("response", "")
                tools_used = response.get("state", {}).get("tools_used", [])
                state.tools_used.extend(tools_used)
                state.messages.append({"role": "assistant", "content": response_text})
                
                # Evaluate response
                eval_result = self.evaluator.evaluate(
                    task,
                    response_text,
                    state.cmp_score,
                    tools_used,
                )
                
                # Update state
                state.cmp_score = eval_result["cmp_score"]
                state.cmp_history.append(state.cmp_score)
                
                if eval_result["success"]:
                    state.success_streak += 1
                    state.current_step += 1
                    state.status = CompletionStatus.IN_PROGRESS
                    
                    # Mark plan step as completed
                    if state.current_step <= len(state.plan):
                        state.plan[state.current_step - 1]["status"] = "completed"
                else:
                    state.success_streak = 0
                    state.status = CompletionStatus.NEEDS_IMPROVEMENT
                
                # Record result
                state.results.append({
                    "iteration": state.iteration,
                    "step": state.current_step,
                    "response": response_text,
                    "cmp_score": state.cmp_score,
                    "success": eval_result["success"],
                    "feedback": eval_result["feedback"],
                    "suggestions": eval_result["improvement_suggestions"],
                    "tools_used": tools_used,
                })
                
                # Callback
                if self.on_iteration:
                    self.on_iteration(state)
                
                # Check completion
                if self.criteria.is_complete(
                    state.cmp_score,
                    state.iteration,
                    state.success_streak,
                ):
                    state.status = CompletionStatus.COMPLETED
                    break
                    
            except Exception as e:
                state.error_count += 1
                state.messages.append({"role": "system", "content": f"Error: {e}"})
                
                if state.error_count >= 3:
                    state.status = CompletionStatus.FAILED
                    break
        
        # Check if we hit max iterations
        if state.iteration >= self.criteria.max_iterations and state.status != CompletionStatus.COMPLETED:
            state.status = CompletionStatus.MAX_ITERATIONS
        
        # Final callback
        if self.on_complete:
            self.on_complete(state)
        
        return state
    
    async def run_async(self, task: str, initial_cmp: float = 0.5) -> LoopState:
        """
        Run the completion loop asynchronously.
        
        Wraps synchronous run in executor for non-blocking operation.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.run, task, initial_cmp)
    
    async def astream(
        self,
        task: str,
        initial_cmp: float = 0.5,
        provider=None,
        chat_model=None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Stream completion loop execution with token-level streaming.
        
        Args:
            task: Task description
            initial_cmp: Initial CMP score
            provider: Optional OpenAICompatibleProvider for streaming
            chat_model: Optional LangChain chat model for streaming
            
        Yields:
            Dict with event type and content:
            - {"type": "iteration_start", "iteration": N, "phase": "..."}
            - {"type": "token", "content": "..."}
            - {"type": "evaluation", "cmp_score": X, "success": bool}
            - {"type": "iteration_end", "iteration": N, "status": "..."}
            - {"type": "complete", "state": {...}}
        """
        state = LoopState(task=task, cmp_score=initial_cmp)
        state.status = CompletionStatus.IN_PROGRESS
        
        # Phase 1: Planning
        yield {
            "type": "iteration_start",
            "iteration": 0,
            "phase": "planning",
        }
        
        planning_prompt = self._create_planning_prompt(task)
        state.messages.append({"role": "user", "content": planning_prompt})
        
        # Stream planning response
        plan_text = ""
        if hasattr(self.agent_invoke, 'astream'):
            async for event in self.agent_invoke.astream(
                {"messages": [{"role": "user", "content": planning_prompt}]},
                provider=provider,
                chat_model=chat_model,
            ):
                if event.get("type") == "token":
                    plan_text += event.get("content", "")
                    yield event
        else:
            # Fallback to sync invoke
            try:
                plan_response = self.agent_invoke({
                    "messages": [{"role": "user", "content": planning_prompt}]
                })
                plan_text = plan_response.get("response", "")
                # Simulate streaming
                for char in plan_text:
                    yield {"type": "token", "content": char}
                    await asyncio.sleep(0.002)
            except Exception as e:
                yield {"type": "error", "content": str(e)}
                state.status = CompletionStatus.FAILED
                yield {"type": "complete", "state": state.to_dict()}
                return
        
        state.plan = self._parse_plan(plan_text)
        state.messages.append({"role": "assistant", "content": plan_text})
        
        yield {
            "type": "iteration_end",
            "iteration": 0,
            "phase": "planning",
            "plan_steps": len(state.plan),
        }
        
        # Phase 2: Iterative Execution with streaming
        while state.iteration < self.criteria.max_iterations:
            state.iteration += 1
            
            yield {
                "type": "iteration_start",
                "iteration": state.iteration,
                "phase": "execution" if state.status != CompletionStatus.NEEDS_IMPROVEMENT else "improvement",
                "current_step": state.current_step,
            }
            
            # Check if all plan steps completed
            if state.current_step >= len(state.plan):
                if self.criteria.is_complete(
                    state.cmp_score,
                    state.iteration,
                    state.success_streak,
                ):
                    state.status = CompletionStatus.COMPLETED
                    break
                else:
                    state.status = CompletionStatus.NEEDS_IMPROVEMENT
            
            # Create prompt
            if state.status == CompletionStatus.NEEDS_IMPROVEMENT and state.results:
                last_result = state.results[-1]
                prompt = self._create_improvement_prompt(
                    task,
                    last_result.get("feedback", ""),
                    last_result.get("suggestions", []),
                    last_result.get("response", ""),
                )
            else:
                prompt = self._create_execution_prompt(
                    task,
                    state.plan,
                    state.current_step,
                    state.results,
                )
            
            state.messages.append({"role": "user", "content": prompt})
            
            # Stream execution response
            response_text = ""
            tools_used = []
            
            if hasattr(self.agent_invoke, 'astream'):
                async for event in self.agent_invoke.astream(
                    {"messages": state.messages},
                    provider=provider,
                    chat_model=chat_model,
                ):
                    if event.get("type") == "token":
                        response_text += event.get("content", "")
                        yield event
                    elif event.get("type") in ("tool_start", "tool_end"):
                        yield event
                        if event.get("type") == "tool_end":
                            tools_used.append(event.get("metadata", {}).get("tool", "unknown"))
            else:
                # Fallback to sync
                try:
                    response = self.agent_invoke({"messages": state.messages})
                    response_text = response.get("response", "")
                    tools_used = response.get("state", {}).get("tools_used", [])
                    
                    for char in response_text:
                        yield {"type": "token", "content": char}
                        await asyncio.sleep(0.002)
                except Exception as e:
                    yield {"type": "error", "content": str(e)}
                    state.error_count += 1
                    if state.error_count >= 3:
                        state.status = CompletionStatus.FAILED
                        break
                    continue
            
            state.tools_used.extend(tools_used)
            state.messages.append({"role": "assistant", "content": response_text})
            
            # Evaluate
            eval_result = self.evaluator.evaluate(
                task,
                response_text,
                state.cmp_score,
                tools_used,
            )
            
            state.cmp_score = eval_result["cmp_score"]
            state.cmp_history.append(state.cmp_score)
            
            yield {
                "type": "evaluation",
                "cmp_score": state.cmp_score,
                "success": eval_result["success"],
                "feedback": eval_result["feedback"],
            }
            
            if eval_result["success"]:
                state.success_streak += 1
                state.current_step += 1
                state.status = CompletionStatus.IN_PROGRESS
                if state.current_step <= len(state.plan):
                    state.plan[state.current_step - 1]["status"] = "completed"
            else:
                state.success_streak = 0
                state.status = CompletionStatus.NEEDS_IMPROVEMENT
            
            state.results.append({
                "iteration": state.iteration,
                "step": state.current_step,
                "response": response_text,
                "cmp_score": state.cmp_score,
                "success": eval_result["success"],
                "feedback": eval_result["feedback"],
                "suggestions": eval_result["improvement_suggestions"],
                "tools_used": tools_used,
            })
            
            yield {
                "type": "iteration_end",
                "iteration": state.iteration,
                "status": state.status.value,
                "cmp_score": state.cmp_score,
            }
            
            if self.on_iteration:
                self.on_iteration(state)
            
            if self.criteria.is_complete(
                state.cmp_score,
                state.iteration,
                state.success_streak,
            ):
                state.status = CompletionStatus.COMPLETED
                break
        
        if state.iteration >= self.criteria.max_iterations and state.status != CompletionStatus.COMPLETED:
            state.status = CompletionStatus.MAX_ITERATIONS
        
        if self.on_complete:
            self.on_complete(state)
        
        yield {"type": "complete", "state": state.to_dict()}


def create_completion_loop(
    agent,
    criteria: Optional[CompletionCriteria] = None,
    on_iteration: Optional[Callable[[LoopState], None]] = None,
    on_complete: Optional[Callable[[LoopState], None]] = None,
) -> HGMCompletionLoop:
    """
    Factory function to create completion loop for an agent.
    
    Args:
        agent: DeepHGMAgent or DeepHGMAgentV2 instance
        criteria: Completion criteria
        on_iteration: Iteration callback
        on_complete: Completion callback
        
    Returns:
        Configured HGMCompletionLoop
    """
    return HGMCompletionLoop(
        agent_invoke=agent.invoke,
        criteria=criteria,
        on_iteration=on_iteration,
        on_complete=on_complete,
    )


__all__ = [
    "CompletionStatus",
    "LoopAction",
    "CompletionCriteria",
    "LoopState",
    "CMPEvaluator",
    "HGMCompletionLoop",
    "create_completion_loop",
]
