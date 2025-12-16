"""
TodoList Middleware for DeepHGM.

Implements LangChain DeepAgents' TodoListMiddleware pattern for
task planning and progress tracking.
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum
import uuid
from datetime import datetime

from langchain_core.tools import BaseTool, tool

from .base import DeepHGMMiddleware, MiddlewareState


class TodoStatus(Enum):
    """Status of a todo item."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"


@dataclass
class TodoItem:
    """A single todo item in the planning list."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    description: str = ""
    status: TodoStatus = TodoStatus.PENDING
    priority: int = 0  # Lower = higher priority
    dependencies: List[str] = field(default_factory=list)
    notes: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "status": self.status.value,
            "priority": self.priority,
            "dependencies": self.dependencies,
            "notes": self.notes,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TodoItem":
        return cls(
            id=data.get("id", str(uuid.uuid4())[:8]),
            description=data.get("description", ""),
            status=TodoStatus(data.get("status", "pending")),
            priority=data.get("priority", 0),
            dependencies=data.get("dependencies", []),
            notes=data.get("notes", ""),
            created_at=data.get("created_at", datetime.now().isoformat()),
            completed_at=data.get("completed_at"),
        )


class TodoListMiddleware(DeepHGMMiddleware):
    """
    Middleware for task planning and progress tracking.
    
    Implements the TodoListMiddleware pattern from LangChain DeepAgents:
    - Provides write_todos and read_todos tools
    - Enables agents to break down complex tasks
    - Tracks progress through multi-step workflows
    - Integrates with HGM CMP for success tracking
    
    Usage:
        middleware = TodoListMiddleware(
            system_prompt="Use write_todos to plan complex tasks...",
            max_todos=50,
        )
    """
    
    SYSTEM_PROMPT_ADDITION = """
## Task Planning

You have access to a todo list for planning and tracking complex tasks:
- Use `write_todos` to create or update your task plan
- Use `read_todos` to review current progress

Best practices:
1. Break down complex tasks into discrete, actionable steps
2. Update todo status as you complete each step
3. Add notes to document important findings
4. Mark blocked items when waiting for dependencies
5. Prioritize items (lower number = higher priority)

Always plan before executing multi-step tasks.
"""
    
    def __init__(
        self,
        system_prompt: Optional[str] = None,
        max_todos: int = 50,
        persist_to_filesystem: bool = False,
    ):
        """
        Initialize TodoList middleware.
        
        Args:
            system_prompt: Custom addition to system prompt
            max_todos: Maximum number of todo items
            persist_to_filesystem: Whether to persist todos to filesystem
        """
        super().__init__("todo_list", priority=20)
        
        self._custom_prompt = system_prompt
        self._max_todos = max_todos
        self._persist = persist_to_filesystem
        self._todos: Dict[str, TodoItem] = {}
        
        # Initialize tools
        self._initialize_tools()
    
    def _initialize_tools(self) -> None:
        """Initialize todo list tools."""
        self._tools = [
            self._create_write_todos_tool(),
            self._create_read_todos_tool(),
        ]
    
    def _create_write_todos_tool(self) -> BaseTool:
        """Create the write_todos tool."""
        middleware = self
        
        @tool
        def write_todos(
            todos: List[Dict[str, Any]],
            clear_existing: bool = False,
        ) -> str:
            """
            Create or update the todo list for task planning.
            
            Args:
                todos: List of todo items, each with:
                    - description (str): What needs to be done
                    - status (str): pending, in_progress, completed, blocked, cancelled
                    - priority (int): Lower = higher priority (default: 0)
                    - dependencies (list): IDs of todos this depends on
                    - notes (str): Additional context or findings
                clear_existing: If True, replace all todos; if False, merge/update
                
            Returns:
                Summary of the updated todo list
            """
            if clear_existing:
                middleware._todos.clear()
            
            updated_count = 0
            created_count = 0
            
            for todo_data in todos:
                if len(middleware._todos) >= middleware._max_todos:
                    break
                
                # Check if updating existing todo
                todo_id = todo_data.get("id")
                if todo_id and todo_id in middleware._todos:
                    # Update existing
                    existing = middleware._todos[todo_id]
                    if "description" in todo_data:
                        existing.description = todo_data["description"]
                    if "status" in todo_data:
                        existing.status = TodoStatus(todo_data["status"])
                        if existing.status == TodoStatus.COMPLETED:
                            existing.completed_at = datetime.now().isoformat()
                    if "priority" in todo_data:
                        existing.priority = todo_data["priority"]
                    if "dependencies" in todo_data:
                        existing.dependencies = todo_data["dependencies"]
                    if "notes" in todo_data:
                        existing.notes = todo_data["notes"]
                    updated_count += 1
                else:
                    # Create new
                    new_todo = TodoItem.from_dict(todo_data)
                    middleware._todos[new_todo.id] = new_todo
                    created_count += 1
            
            # Generate summary
            total = len(middleware._todos)
            completed = sum(1 for t in middleware._todos.values() if t.status == TodoStatus.COMPLETED)
            in_progress = sum(1 for t in middleware._todos.values() if t.status == TodoStatus.IN_PROGRESS)
            pending = sum(1 for t in middleware._todos.values() if t.status == TodoStatus.PENDING)
            
            return f"""Todo list updated:
- Created: {created_count}
- Updated: {updated_count}
- Total: {total}

Progress: {completed}/{total} completed, {in_progress} in progress, {pending} pending"""
        
        return write_todos
    
    def _create_read_todos_tool(self) -> BaseTool:
        """Create the read_todos tool."""
        middleware = self
        
        @tool
        def read_todos(
            status_filter: Optional[str] = None,
            include_completed: bool = True,
        ) -> str:
            """
            Read the current todo list.
            
            Args:
                status_filter: Filter by status (pending, in_progress, completed, blocked)
                include_completed: Whether to include completed items
                
            Returns:
                Formatted todo list with status and progress
            """
            todos = list(middleware._todos.values())
            
            # Apply filters
            if status_filter:
                todos = [t for t in todos if t.status.value == status_filter]
            if not include_completed:
                todos = [t for t in todos if t.status != TodoStatus.COMPLETED]
            
            # Sort by priority, then status
            status_order = {
                TodoStatus.IN_PROGRESS: 0,
                TodoStatus.PENDING: 1,
                TodoStatus.BLOCKED: 2,
                TodoStatus.COMPLETED: 3,
                TodoStatus.CANCELLED: 4,
            }
            todos.sort(key=lambda t: (t.priority, status_order.get(t.status, 5)))
            
            if not todos:
                return "No todos found."
            
            # Format output
            lines = ["## Current Todo List\n"]
            
            for todo in todos:
                status_icon = {
                    TodoStatus.PENDING: "⬜",
                    TodoStatus.IN_PROGRESS: "🔄",
                    TodoStatus.COMPLETED: "✅",
                    TodoStatus.BLOCKED: "🚫",
                    TodoStatus.CANCELLED: "❌",
                }.get(todo.status, "⬜")
                
                line = f"{status_icon} [{todo.id}] {todo.description}"
                if todo.priority > 0:
                    line += f" (P{todo.priority})"
                if todo.notes:
                    line += f"\n   Notes: {todo.notes}"
                if todo.dependencies:
                    line += f"\n   Depends on: {', '.join(todo.dependencies)}"
                
                lines.append(line)
            
            # Add summary
            total = len(middleware._todos)
            completed = sum(1 for t in middleware._todos.values() if t.status == TodoStatus.COMPLETED)
            
            lines.append(f"\n---\nProgress: {completed}/{total} ({100*completed//total if total else 0}%)")
            
            return "\n".join(lines)
        
        return read_todos
    
    def get_system_prompt_addition(self) -> Optional[str]:
        """Return todo list instructions for system prompt."""
        if self._custom_prompt:
            return self._custom_prompt
        return self.SYSTEM_PROMPT_ADDITION
    
    def get_todos(self) -> List[TodoItem]:
        """Get all todo items."""
        return list(self._todos.values())
    
    def get_progress(self) -> Dict[str, Any]:
        """Get progress summary."""
        total = len(self._todos)
        if total == 0:
            return {"total": 0, "completed": 0, "percentage": 0}
        
        completed = sum(1 for t in self._todos.values() if t.status == TodoStatus.COMPLETED)
        return {
            "total": total,
            "completed": completed,
            "percentage": 100 * completed // total,
            "in_progress": sum(1 for t in self._todos.values() if t.status == TodoStatus.IN_PROGRESS),
            "pending": sum(1 for t in self._todos.values() if t.status == TodoStatus.PENDING),
            "blocked": sum(1 for t in self._todos.values() if t.status == TodoStatus.BLOCKED),
        }
    
    def clear(self) -> None:
        """Clear all todos."""
        self._todos.clear()
