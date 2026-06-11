"""LLM 客户端 — DeepSeek (OpenAI 兼容协议)"""

import json
from typing import Any

from openai import AsyncOpenAI

from .schema import FunctionCall, LLMResponse, Message, TokenUsage, ToolCall


class LLMClient:
    """统一的 LLM 客户端"""

    def __init__(
        self,
        api_key: str,
        api_base: str,
        model: str,
    ):
        self.api_key = api_key
        self.api_base = api_base
        self.model = model
        self.client = AsyncOpenAI(api_key=api_key, base_url=api_base)

    async def generate(
        self,
        messages: list[Message],
        tools: list[Any] | None = None,
    ) -> LLMResponse:
        """调用 LLM 生成回复"""
        api_messages = self._convert_messages(messages)
        openai_tools = self._convert_tools(tools) if tools else None

        params: dict[str, Any] = {
            "model": self.model,
            "messages": api_messages,
        }
        if openai_tools:
            params["tools"] = openai_tools

        response = await self.client.chat.completions.create(**params)
        return self._parse_response(response)

    # ── 内部方法 ──────────────────────────────────

    def _convert_messages(self, messages: list[Message]) -> list[dict]:
        api_msgs = []
        for msg in messages:
            if msg.role == "system":
                api_msgs.append({"role": "system", "content": msg.content})
            elif msg.role == "user":
                api_msgs.append({"role": "user", "content": msg.content})
            elif msg.role == "assistant":
                m: dict[str, Any] = {"role": "assistant"}
                if msg.content:
                    m["content"] = msg.content
                if msg.tool_calls:
                    m["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": json.dumps(tc.function.arguments),
                            },
                        }
                        for tc in msg.tool_calls
                    ]
                api_msgs.append(m)
            elif msg.role == "tool":
                api_msgs.append({
                    "role": "tool",
                    "tool_call_id": msg.tool_call_id,
                    "content": msg.content,
                })
        return api_msgs

    def _convert_tools(self, tools: list[Any]) -> list[dict]:
        result = []
        for tool in tools:
            if hasattr(tool, "to_openai_schema"):
                result.append(tool.to_openai_schema())
            elif isinstance(tool, dict):
                result.append(tool)
        return result

    def _parse_response(self, response: Any) -> LLMResponse:
        msg = response.choices[0].message
        tool_calls = None
        if msg.tool_calls:
            tool_calls = [
                ToolCall(
                    id=tc.id,
                    type="function",
                    function=FunctionCall(
                        name=tc.function.name,
                        arguments=json.loads(tc.function.arguments),
                    ),
                )
                for tc in msg.tool_calls
            ]

        usage = None
        if hasattr(response, "usage") and response.usage:
            usage = TokenUsage(
                prompt_tokens=response.usage.prompt_tokens or 0,
                completion_tokens=response.usage.completion_tokens or 0,
                total_tokens=response.usage.total_tokens or 0,
            )

        return LLMResponse(
            content=msg.content or "",
            tool_calls=tool_calls,
            usage=usage,
        )
