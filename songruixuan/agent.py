"""
🌸 SONGRUIXUAN Agent 执行循环
"""

import asyncio
import json
from pathlib import Path
from typing import Any, Optional

from src.llm_client import LLMClient
from src.schema import Message
from src.tool_base import Tool, ToolResult


class Agent:
    """SONGRUIXUAN 核心 Agent"""

    def __init__(
        self,
        llm_clients: dict[str, LLMClient],
        system_prompt: str,
        tools: list[Tool],
        workspace_dir: str = "./workspace",
        max_steps: int = 50,
        default_llm: str = "deepseek",
    ):
        self.llm_clients = llm_clients
        self.default_llm = default_llm
        self.tools = {t.name: t for t in tools}
        self.max_steps = max_steps

        self.workspace_dir = Path(workspace_dir)
        self.workspace_dir.mkdir(parents=True, exist_ok=True)

        self.system_prompt = system_prompt
        self.messages: list[Message] = [
            Message(role="system", content=system_prompt)
        ]

        # 当前活跃的 LLM
        self.active_llm = default_llm

    def add_user_message(self, content: str):
        self.messages.append(Message(role="user", content=content))

    def get_active_client(self) -> LLMClient:
        return self.llm_clients[self.active_llm]

    def switch_llm(self, name: str):
        """切换 LLM（DeepSeek ↔ MiniMax）"""
        if name in self.llm_clients:
            self.active_llm = name

    def _route_task(self, user_input: str) -> str:
        """根据任务特征路由到合适的 LLM"""
        # 当前只有 DeepSeek，后续加新 LLM 时在此扩展路由
        return self.default_llm

    async def run(self, task: Optional[str] = None) -> str:
        """执行 Agent 循环"""
        if task:
            self.add_user_message(task)

        # 路由选择 LLM
        last_user_msg = next(
            (m.content for m in reversed(self.messages)
             if m.role == "user"), ""
        )
        if isinstance(last_user_msg, str):
            self.active_llm = self._route_task(last_user_msg)

        step = 0
        while step < self.max_steps:
            step += 1
            client = self.get_active_client()
            tool_list = list(self.tools.values())

            # LLM 思考
            response = await client.generate(self.messages, tool_list)

            # 记录助手消息
            self.messages.append(Message(
                role="assistant",
                content=response.content,
                tool_calls=response.tool_calls,
            ))

            # 没有工具调用 → 任务完成
            if not response.tool_calls:
                return response.content

            # 执行工具调用
            for tc in response.tool_calls:
                tool_name = tc.function.name
                args = tc.function.arguments

                if tool_name not in self.tools:
                    result = ToolResult(
                        success=False,
                        error=f"未知工具: {tool_name}"
                    )
                else:
                    try:
                        result = await self.tools[tool_name].execute(**args)
                    except Exception as e:
                        result = ToolResult(
                            success=False,
                            error=f"工具执行失败: {e}"
                        )

                # 记录工具结果
                self.messages.append(Message(
                    role="tool",
                    content=result.content if result.success
                             else f"Error: {result.error}",
                    tool_call_id=tc.id,
                    name=tool_name,
                ))

        return f"任务在 {self.max_steps} 步内未完成"

    def get_history(self) -> list[Message]:
        return self.messages.copy()
