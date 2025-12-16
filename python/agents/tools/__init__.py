"""
HGM Agent Tools Module.

Purpose: Minimal, focused tools for HGM self-improving agents
Design: Low token overhead, concise descriptions per LangChain best practices
"""

from .hunyuan_ocr_tool import HunyuanOCRTool

__all__ = [
    "HunyuanOCRTool",
]
