"""
HunyuanOCR Tool for HGM Agents.

Purpose: Minimal-context OCR tool for self-improving agents
Design: Concise descriptions, low token overhead, focused functionality
"""

from typing import Any, Dict, Optional, Type
from pathlib import Path
from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun
from pydantic import BaseModel, Field


class HunyuanOCRInput(BaseModel):
    """Input schema for HunyuanOCR tool."""
    
    file_path: str = Field(description="Path to image or PDF file")
    agent_workspace: Optional[str] = Field(
        default=None,
        description="Agent workspace path for output storage"
    )


class HunyuanOCRTool(BaseTool):
    """
    Extract text from documents using HunyuanOCR.
    
    Tool for HGM agents - optimized for minimal context overhead.
    """
    
    name: str = "ocr_extract"
    description: str = "Extract text from image/PDF. Returns structured text with coordinates."
    args_schema: Type[BaseModel] = HunyuanOCRInput
    
    _processor: Any = None
    _workspace_path: Optional[str] = None
    
    def __init__(self, workspace_path: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self._workspace_path = workspace_path
    
    def _get_processor(self):
        """Lazy load OCR processor."""
        if self._processor is None:
            from document.hunyuan_ocr import HunyuanOCRProcessor
            self._processor = HunyuanOCRProcessor()
        return self._processor
    
    def _run(
        self,
        file_path: str,
        agent_workspace: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Dict[str, Any]:
        """Execute OCR extraction."""
        path = Path(file_path)
        
        if not path.exists():
            return {"error": f"File not found: {file_path}", "success": False}
        
        if not path.suffix.lower() in {".png", ".jpg", ".jpeg", ".pdf", ".tiff", ".bmp", ".webp"}:
            return {"error": f"Unsupported format: {path.suffix}", "success": False}
        
        try:
            processor = self._get_processor()
            result = processor.process_image(str(path))
            
            output = {
                "success": True,
                "text": result.get("text", ""),
                "source": str(path),
                "char_count": len(result.get("text", "")),
            }
            
            workspace = agent_workspace or self._workspace_path
            if workspace:
                output["workspace"] = workspace
                output_path = self._save_to_workspace(result, path, workspace)
                if output_path:
                    output["saved_to"] = output_path
            
            return output
            
        except Exception as e:
            return {"error": str(e), "success": False, "source": str(path)}
    
    def _save_to_workspace(
        self, 
        result: Dict[str, Any], 
        source_path: Path, 
        workspace: str
    ) -> Optional[str]:
        """Save OCR result to agent workspace."""
        try:
            workspace_path = Path(workspace)
            documents_dir = workspace_path / "documents" / "ocr"
            documents_dir.mkdir(parents=True, exist_ok=True)
            
            output_file = documents_dir / f"{source_path.stem}_ocr.txt"
            output_file.write_text(result.get("text", ""), encoding="utf-8")
            
            return str(output_file)
        except Exception:
            return None


def create_ocr_tool(workspace_path: Optional[str] = None) -> HunyuanOCRTool:
    """
    Factory function to create HunyuanOCR tool for HGM agents.
    
    Args:
        workspace_path: Optional agent workspace for output storage
        
    Returns:
        Configured HunyuanOCRTool instance
    """
    return HunyuanOCRTool(workspace_path=workspace_path)
