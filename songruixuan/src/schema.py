"""数据模型定义 — 参考 Mini-Agent (MIT License)"""

from typing import Any

from pydantic import BaseModel


class FunctionCall(BaseModel):
    """函数调用详情"""
    name: str
    arguments: dict[str, Any]


class ToolCall(BaseModel):
    """工具调用结构"""
    id: str
    type: str  # "function"
    function: FunctionCall


class Message(BaseModel):
    """对话消息"""
    role: str  # "system" | "user" | "assistant" | "tool"
    content: str | list[dict[str, Any]]
    thinking: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None
    name: str | None = None


class TokenUsage(BaseModel):
    """Token 用量统计"""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class LLMResponse(BaseModel):
    """LLM 响应"""
    content: str
    thinking: str | None = None
    tool_calls: list[ToolCall] | None = None
    finish_reason: str = "stop"
    usage: TokenUsage | None = None
