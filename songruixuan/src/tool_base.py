"""工具基类 — 参考 Mini-Agent (MIT License)"""

from typing import Any

from pydantic import BaseModel


class ToolResult(BaseModel):
    """工具执行结果"""
    success: bool
    content: str = ""
    error: str | None = None


class Tool:
    """所有工具的基类"""

    @property
    def name(self) -> str:
        raise NotImplementedError

    @property
    def description(self) -> str:
        raise NotImplementedError

    @property
    def parameters(self) -> dict[str, Any]:
        """参数 JSON Schema"""
        raise NotImplementedError

    async def execute(self, **kwargs) -> ToolResult:
        raise NotImplementedError

    def to_openai_schema(self) -> dict[str, Any]:
        """转换为 OpenAI 工具格式"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }
