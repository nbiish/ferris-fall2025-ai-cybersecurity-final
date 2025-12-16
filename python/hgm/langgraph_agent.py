"""
LangGraph-based HGM Agent with minimal tool context.

Purpose: State-of-the-art agent orchestration using LangGraph
Design: Minimal context overhead, focused tool descriptions, self-improving
"""

from typing import Any, AsyncIterator, Dict, List, Optional, TypedDict, Annotated, Union
from dataclasses import dataclass
import asyncio
import operator

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, AIMessageChunk
from langchain_core.tools import BaseTool

from .agent import HGMAgent


class HGMAgentState(TypedDict):
    """State for LangGraph HGM agent."""
    messages: Annotated[List[BaseMessage], operator.add]
    agent_id: str
    generation: int
    cmp_score: float
    tools_used: List[str]
    iteration: int
    workspace_path: Optional[str]


@dataclass
class HGMToolConfig:
    """Configuration for HGM agent tools - minimal context design."""
    
    name: str
    description: str
    enabled: bool = True
    
    @classmethod
    def ocr_tool(cls) -> "HGMToolConfig":
        """OCR tool with concise description."""
        return cls(
            name="ocr_extract",
            description="Extract text from image/PDF. Returns structured text.",
        )


class HGMLangGraphAgent:
    """
    LangGraph-based HGM agent with self-improvement capabilities.
    
    Key features:
    - Minimal tool context (low token overhead)
    - State-aware execution
    - CMP-based self-evaluation
    - Tool selector middleware pattern
    """
    
    SYSTEM_PROMPT = """You are a self-improving AI agent. Execute tasks efficiently.
Available tools: {tool_names}
Workspace: {workspace}

Rules:
1. Use tools only when necessary
2. Be concise in responses
3. Report success/failure clearly"""
    
    def __init__(
        self,
        agent: HGMAgent,
        tools: Optional[List[BaseTool]] = None,
        workspace_path: Optional[str] = None,
    ):
        self.agent = agent
        self.tools = tools or []
        self.workspace_path = workspace_path
        self.tool_configs = self._create_tool_configs()
    
    def _create_tool_configs(self) -> List[HGMToolConfig]:
        """Create minimal tool configurations."""
        configs = []
        for tool in self.tools:
            configs.append(HGMToolConfig(
                name=tool.name,
                description=tool.description[:80],  # Truncate for minimal context
            ))
        return configs
    
    def get_system_message(self) -> SystemMessage:
        """Generate system message with minimal context."""
        tool_names = ", ".join(c.name for c in self.tool_configs if c.enabled)
        return SystemMessage(content=self.SYSTEM_PROMPT.format(
            tool_names=tool_names or "none",
            workspace=self.workspace_path or "default",
        ))
    
    def create_initial_state(self, user_input: str) -> HGMAgentState:
        """Create initial state for agent execution."""
        return HGMAgentState(
            messages=[
                self.get_system_message(),
                HumanMessage(content=user_input),
            ],
            agent_id=self.agent.id,
            generation=self.agent.generation,
            cmp_score=self.agent.cmp_score,
            tools_used=[],
            iteration=0,
            workspace_path=self.workspace_path,
        )
    
    def select_tool(self, query: str) -> Optional[BaseTool]:
        """
        Select most relevant tool for query (tool selector middleware pattern).
        
        Uses keyword matching for minimal overhead.
        """
        query_lower = query.lower()
        
        tool_keywords = {
            "ocr_extract": ["ocr", "extract", "document", "pdf", "image", "text", "scan"],
        }
        
        for tool in self.tools:
            keywords = tool_keywords.get(tool.name, [])
            if any(kw in query_lower for kw in keywords):
                return tool
        
        return None
    
    def invoke_tool(self, tool: BaseTool, **kwargs) -> Dict[str, Any]:
        """Invoke tool and track usage."""
        try:
            result = tool._run(**kwargs)
            return {"success": True, "result": result}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def evaluate_response(self, response: str) -> bool:
        """Evaluate if response indicates success (for CMP)."""
        failure_indicators = ["error", "failed", "unable", "cannot", "exception"]
        return not any(ind in response.lower() for ind in failure_indicators)
    
    def step(self, state: HGMAgentState, user_input: str) -> HGMAgentState:
        """
        Execute one step of the HGM agent.
        
        Returns updated state.
        """
        # Check if tool is needed
        tool = self.select_tool(user_input)
        
        new_messages = list(state["messages"])
        tools_used = list(state["tools_used"])
        
        if tool:
            # Extract parameters from user input (simplified)
            result = self.invoke_tool(tool, file_path=user_input)
            tools_used.append(tool.name)
            
            if result["success"]:
                new_messages.append(AIMessage(
                    content=f"Tool '{tool.name}' executed successfully."
                ))
            else:
                new_messages.append(AIMessage(
                    content=f"Tool '{tool.name}' failed: {result.get('error', 'unknown')}"
                ))
        else:
            new_messages.append(AIMessage(
                content=f"Processed request: {user_input[:100]}..."
            ))
        
        # Update CMP based on evaluation
        last_response = new_messages[-1].content if new_messages else ""
        success = self.evaluate_response(last_response)
        
        if success:
            self.agent.record_success()
        else:
            self.agent.record_failure()
        
        return HGMAgentState(
            messages=new_messages,
            agent_id=state["agent_id"],
            generation=state["generation"],
            cmp_score=self.agent.cmp_score,
            tools_used=tools_used,
            iteration=state["iteration"] + 1,
            workspace_path=state["workspace_path"],
        )
    
    def get_tool_descriptions(self) -> str:
        """Get minimal tool descriptions for context."""
        return "\n".join(
            f"- {c.name}: {c.description}"
            for c in self.tool_configs
            if c.enabled
        )
    
    async def astream(
        self,
        input_data: Dict[str, Any],
        provider=None,
        chat_model=None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Stream agent response token by token.
        
        Args:
            input_data: Dict with messages or user_input
            provider: Optional OpenAICompatibleProvider for streaming
            chat_model: Optional LangChain chat model with streaming
            
        Yields:
            Dict with event type and content
        """
        # Extract user input
        user_input = input_data.get("user_input", "")
        if not user_input:
            messages = input_data.get("messages", [])
            for msg in reversed(messages):
                if isinstance(msg, dict) and msg.get("role") == "user":
                    user_input = msg.get("content", "")
                    break
                elif hasattr(msg, "content"):
                    user_input = msg.content
                    break
        
        # Check if tool is needed
        tool = self.select_tool(user_input)
        tools_used = []
        
        if tool:
            yield {
                "type": "tool_start",
                "content": tool.name,
                "metadata": {"input": user_input[:200]},
            }
            
            result = self.invoke_tool(tool, file_path=user_input)
            tools_used.append(tool.name)
            
            yield {
                "type": "tool_end",
                "content": str(result.get("result", result.get("error", "")))[:500],
            }
        
        # Stream LLM response
        if provider:
            formatted_messages = self._format_messages_for_streaming(user_input)
            
            try:
                async for chunk in provider.astream(
                    messages=formatted_messages,
                    temperature=0.7,
                ):
                    yield {"type": "token", "content": chunk}
            except Exception as e:
                yield {"type": "error", "content": str(e)}
                
        elif chat_model:
            lc_messages = [
                self.get_system_message(),
                HumanMessage(content=user_input),
            ]
            
            try:
                async for chunk in chat_model.astream(lc_messages):
                    if isinstance(chunk, AIMessageChunk) and chunk.content:
                        yield {"type": "token", "content": chunk.content}
                    elif hasattr(chunk, 'content') and chunk.content:
                        yield {"type": "token", "content": chunk.content}
            except Exception as e:
                yield {"type": "error", "content": str(e)}
        else:
            # Fallback: simulate streaming
            response = f"Processed request: {user_input[:100]}..."
            if tools_used:
                response = f"[Tools: {', '.join(tools_used)}] {response}"
            
            for char in response:
                yield {"type": "token", "content": char}
                await asyncio.sleep(0.005)
        
        # Yield metadata
        yield {
            "type": "metadata",
            "content": "",
            "metadata": {
                "agent_id": self.agent.id,
                "cmp_score": self.agent.cmp_score,
                "tools_used": tools_used,
            },
        }
    
    async def astream_text(
        self,
        input_data: Dict[str, Any],
        provider=None,
        chat_model=None,
    ) -> AsyncIterator[str]:
        """Stream just text tokens."""
        async for event in self.astream(input_data, provider, chat_model):
            if event.get("type") == "token":
                yield event.get("content", "")
    
    def _format_messages_for_streaming(
        self,
        user_input: str,
    ) -> List[Dict[str, str]]:
        """Format messages for provider streaming."""
        tool_names = ", ".join(c.name for c in self.tool_configs if c.enabled)
        system_content = self.SYSTEM_PROMPT.format(
            tool_names=tool_names or "none",
            workspace=self.workspace_path or "default",
        )
        
        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_input},
        ]


def create_hgm_langgraph_agent(
    agent: Optional[HGMAgent] = None,
    tools: Optional[List[BaseTool]] = None,
    workspace_path: Optional[str] = None,
) -> HGMLangGraphAgent:
    """
    Factory function to create HGM LangGraph agent.
    
    Args:
        agent: HGM agent instance (creates initial if None)
        tools: List of tools to provide
        workspace_path: Agent workspace path
        
    Returns:
        Configured HGMLangGraphAgent
    """
    if agent is None:
        agent = HGMAgent.initial()
    
    return HGMLangGraphAgent(
        agent=agent,
        tools=tools or [],
        workspace_path=workspace_path,
    )
