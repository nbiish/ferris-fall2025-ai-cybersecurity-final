"""
Filesystem Middleware for DeepHGM.

Implements LangChain DeepAgents' FilesystemMiddleware pattern for
file operations and context offloading.
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
import os
import glob as glob_module
import re

from langchain_core.tools import BaseTool, tool

from .base import DeepHGMMiddleware, MiddlewareState


class FilesystemMiddleware(DeepHGMMiddleware):
    """
    Middleware for file system operations and context management.
    
    Implements the FilesystemMiddleware pattern from LangChain DeepAgents:
    - Provides ls, read_file, write_file, edit_file, glob, grep tools
    - Enables context offloading for large tool results
    - Supports both short-term (state) and long-term (store) backends
    
    Usage:
        middleware = FilesystemMiddleware(
            workspace_root="/workspace",
            auto_save_threshold=20000,  # Auto-save results > 20K tokens
        )
    """
    
    SYSTEM_PROMPT_ADDITION = """
## Filesystem Tools

You have access to filesystem tools for context management:
- `ls`: List files and directories
- `read_file`: Read file contents (supports line ranges)
- `write_file`: Write content to a file
- `edit_file`: Edit specific parts of a file
- `glob_search`: Find files matching patterns
- `grep_search`: Search file contents with regex

Use these tools to:
1. Offload large context to files to prevent overflow
2. Store intermediate results for later reference
3. Organize outputs in the workspace directory

Workspace root: {workspace_root}
"""
    
    def __init__(
        self,
        workspace_root: Optional[str] = None,
        auto_save_threshold: int = 20000,
        allowed_extensions: Optional[List[str]] = None,
    ):
        """
        Initialize Filesystem middleware.
        
        Args:
            workspace_root: Root directory for file operations
            auto_save_threshold: Token count to trigger auto-save
            allowed_extensions: List of allowed file extensions
        """
        super().__init__("filesystem", priority=30)
        
        self._workspace_root = workspace_root or os.getcwd()
        self._auto_save_threshold = auto_save_threshold
        self._allowed_extensions = allowed_extensions
        
        # Virtual filesystem for state-based storage
        self._virtual_fs: Dict[str, str] = {}
        
        # Initialize tools
        self._initialize_tools()
    
    def _initialize_tools(self) -> None:
        """Initialize filesystem tools."""
        self._tools = [
            self._create_ls_tool(),
            self._create_read_file_tool(),
            self._create_write_file_tool(),
            self._create_edit_file_tool(),
            self._create_glob_tool(),
            self._create_grep_tool(),
        ]
    
    def _resolve_path(self, path: str) -> str:
        """Resolve path relative to workspace root."""
        if os.path.isabs(path):
            return path
        return os.path.join(self._workspace_root, path)
    
    def _is_allowed_path(self, path: str) -> bool:
        """Check if path is within workspace and has allowed extension."""
        resolved = self._resolve_path(path)
        
        # Must be within workspace
        if not resolved.startswith(os.path.abspath(self._workspace_root)):
            return False
        
        # Check extension if restrictions exist
        if self._allowed_extensions:
            ext = os.path.splitext(path)[1].lstrip(".")
            if ext and ext not in self._allowed_extensions:
                return False
        
        return True
    
    def _create_ls_tool(self) -> BaseTool:
        """Create the ls tool."""
        middleware = self
        
        @tool
        def ls(path: str = ".", show_hidden: bool = False) -> str:
            """
            List files and directories in a path.
            
            Args:
                path: Directory path (relative to workspace)
                show_hidden: Whether to show hidden files
                
            Returns:
                Formatted directory listing
            """
            resolved = middleware._resolve_path(path)
            
            if not os.path.exists(resolved):
                return f"Error: Path '{path}' does not exist"
            
            if not os.path.isdir(resolved):
                return f"Error: '{path}' is not a directory"
            
            try:
                entries = os.listdir(resolved)
                if not show_hidden:
                    entries = [e for e in entries if not e.startswith(".")]
                
                entries.sort()
                
                lines = [f"Contents of {path}:\n"]
                for entry in entries:
                    full_path = os.path.join(resolved, entry)
                    if os.path.isdir(full_path):
                        lines.append(f"📁 {entry}/")
                    else:
                        size = os.path.getsize(full_path)
                        lines.append(f"📄 {entry} ({size} bytes)")
                
                return "\n".join(lines)
                
            except PermissionError:
                return f"Error: Permission denied for '{path}'"
        
        return ls
    
    def _create_read_file_tool(self) -> BaseTool:
        """Create the read_file tool."""
        middleware = self
        
        @tool
        def read_file(
            path: str,
            start_line: Optional[int] = None,
            end_line: Optional[int] = None,
        ) -> str:
            """
            Read contents of a file.
            
            Args:
                path: File path (relative to workspace)
                start_line: Optional starting line (1-indexed)
                end_line: Optional ending line (inclusive)
                
            Returns:
                File contents or specified line range
            """
            # Check virtual filesystem first
            if path in middleware._virtual_fs:
                content = middleware._virtual_fs[path]
            else:
                resolved = middleware._resolve_path(path)
                
                if not os.path.exists(resolved):
                    return f"Error: File '{path}' does not exist"
                
                if not os.path.isfile(resolved):
                    return f"Error: '{path}' is not a file"
                
                try:
                    with open(resolved, "r", encoding="utf-8") as f:
                        content = f.read()
                except UnicodeDecodeError:
                    return f"Error: '{path}' is not a text file"
                except PermissionError:
                    return f"Error: Permission denied for '{path}'"
            
            # Apply line range if specified
            if start_line is not None or end_line is not None:
                lines = content.split("\n")
                start = (start_line or 1) - 1
                end = end_line or len(lines)
                
                selected = lines[start:end]
                numbered = [f"{i+start+1}: {line}" for i, line in enumerate(selected)]
                return "\n".join(numbered)
            
            return content
        
        return read_file
    
    def _create_write_file_tool(self) -> BaseTool:
        """Create the write_file tool."""
        middleware = self
        
        @tool
        def write_file(path: str, content: str, append: bool = False) -> str:
            """
            Write content to a file.
            
            Args:
                path: File path (relative to workspace)
                content: Content to write
                append: If True, append to file; if False, overwrite
                
            Returns:
                Confirmation message
            """
            if not middleware._is_allowed_path(path):
                return f"Error: Path '{path}' is not allowed"
            
            resolved = middleware._resolve_path(path)
            
            try:
                # Create directories if needed
                os.makedirs(os.path.dirname(resolved), exist_ok=True)
                
                mode = "a" if append else "w"
                with open(resolved, mode, encoding="utf-8") as f:
                    f.write(content)
                
                # Also store in virtual filesystem
                if append and path in middleware._virtual_fs:
                    middleware._virtual_fs[path] += content
                else:
                    middleware._virtual_fs[path] = content
                
                action = "Appended to" if append else "Wrote"
                return f"{action} {len(content)} characters to '{path}'"
                
            except PermissionError:
                return f"Error: Permission denied for '{path}'"
            except Exception as e:
                return f"Error writing to '{path}': {str(e)}"
        
        return write_file
    
    def _create_edit_file_tool(self) -> BaseTool:
        """Create the edit_file tool."""
        middleware = self
        
        @tool
        def edit_file(
            path: str,
            old_string: str,
            new_string: str,
            replace_all: bool = False,
        ) -> str:
            """
            Edit a file by replacing text.
            
            Args:
                path: File path (relative to workspace)
                old_string: Text to find and replace
                new_string: Replacement text
                replace_all: If True, replace all occurrences
                
            Returns:
                Confirmation with number of replacements
            """
            # Read current content
            if path in middleware._virtual_fs:
                content = middleware._virtual_fs[path]
            else:
                resolved = middleware._resolve_path(path)
                
                if not os.path.exists(resolved):
                    return f"Error: File '{path}' does not exist"
                
                try:
                    with open(resolved, "r", encoding="utf-8") as f:
                        content = f.read()
                except Exception as e:
                    return f"Error reading '{path}': {str(e)}"
            
            # Count occurrences
            count = content.count(old_string)
            if count == 0:
                return f"Error: '{old_string[:50]}...' not found in '{path}'"
            
            # Replace
            if replace_all:
                new_content = content.replace(old_string, new_string)
                replacements = count
            else:
                new_content = content.replace(old_string, new_string, 1)
                replacements = 1
            
            # Write back
            resolved = middleware._resolve_path(path)
            try:
                with open(resolved, "w", encoding="utf-8") as f:
                    f.write(new_content)
                
                middleware._virtual_fs[path] = new_content
                
                return f"Made {replacements} replacement(s) in '{path}'"
                
            except Exception as e:
                return f"Error writing to '{path}': {str(e)}"
        
        return edit_file
    
    def _create_glob_tool(self) -> BaseTool:
        """Create the glob search tool."""
        middleware = self
        
        @tool
        def glob_search(pattern: str, max_results: int = 50) -> str:
            """
            Find files matching a glob pattern.
            
            Args:
                pattern: Glob pattern (e.g., '**/*.py', 'src/**/*.ts')
                max_results: Maximum number of results to return
                
            Returns:
                List of matching file paths
            """
            search_path = os.path.join(middleware._workspace_root, pattern)
            
            try:
                matches = glob_module.glob(search_path, recursive=True)
                matches = matches[:max_results]
                
                # Make paths relative to workspace
                relative = [
                    os.path.relpath(m, middleware._workspace_root)
                    for m in matches
                ]
                
                if not relative:
                    return f"No files matching '{pattern}'"
                
                return f"Found {len(relative)} files:\n" + "\n".join(relative)
                
            except Exception as e:
                return f"Error searching for '{pattern}': {str(e)}"
        
        return glob_search
    
    def _create_grep_tool(self) -> BaseTool:
        """Create the grep search tool."""
        middleware = self
        
        @tool
        def grep_search(
            pattern: str,
            path: str = ".",
            include: Optional[str] = None,
            max_results: int = 50,
        ) -> str:
            """
            Search file contents with regex.
            
            Args:
                pattern: Regex pattern to search for
                path: Directory to search in
                include: Glob pattern to filter files (e.g., '*.py')
                max_results: Maximum number of results
                
            Returns:
                Matching lines with file paths and line numbers
            """
            resolved = middleware._resolve_path(path)
            
            if not os.path.exists(resolved):
                return f"Error: Path '{path}' does not exist"
            
            try:
                regex = re.compile(pattern)
            except re.error as e:
                return f"Error: Invalid regex pattern: {str(e)}"
            
            results = []
            files_searched = 0
            
            # Get files to search
            if os.path.isfile(resolved):
                files = [resolved]
            else:
                if include:
                    glob_pattern = os.path.join(resolved, "**", include)
                    files = glob_module.glob(glob_pattern, recursive=True)
                else:
                    files = glob_module.glob(os.path.join(resolved, "**", "*"), recursive=True)
                files = [f for f in files if os.path.isfile(f)]
            
            for filepath in files:
                if len(results) >= max_results:
                    break
                
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        for line_num, line in enumerate(f, 1):
                            if regex.search(line):
                                rel_path = os.path.relpath(filepath, middleware._workspace_root)
                                results.append(f"{rel_path}:{line_num}: {line.strip()}")
                                if len(results) >= max_results:
                                    break
                    files_searched += 1
                except (UnicodeDecodeError, PermissionError):
                    continue
            
            if not results:
                return f"No matches for '{pattern}' in {files_searched} files"
            
            header = f"Found {len(results)} matches in {files_searched} files:\n"
            return header + "\n".join(results)
        
        return grep_search
    
    def get_system_prompt_addition(self) -> Optional[str]:
        """Return filesystem instructions for system prompt."""
        return self.SYSTEM_PROMPT_ADDITION.format(
            workspace_root=self._workspace_root
        )
    
    def set_workspace_root(self, path: str) -> None:
        """Set the workspace root directory."""
        self._workspace_root = path
    
    def get_virtual_file(self, path: str) -> Optional[str]:
        """Get content from virtual filesystem."""
        return self._virtual_fs.get(path)
    
    def set_virtual_file(self, path: str, content: str) -> None:
        """Set content in virtual filesystem."""
        self._virtual_fs[path] = content
    
    def clear_virtual_fs(self) -> None:
        """Clear the virtual filesystem."""
        self._virtual_fs.clear()
