"""🐚 Shell 执行工具"""

import asyncio
import os
from pathlib import Path

from src.tool_base import Tool, ToolResult


class BashTool(Tool):
    """执行 shell 命令 — Agent 用来自主完成任务"""

    def __init__(self, workspace_dir: str):
        self.workspace = Path(workspace_dir)

    @property
    def name(self) -> str:
        return "run_shell"

    @property
    def description(self) -> str:
        return (
            "执行终端命令。用于运行 Python 脚本生成文档、安装依赖等。"
            "完成任务后应删除临时脚本，只保留最终文件。"
            "示例：python gen_pdf.py && rm gen_pdf.py"
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "要执行的命令",
                },
            },
            "required": ["command"],
        }

    async def execute(self, command: str) -> ToolResult:
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                cwd=str(self.workspace),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            )
            stdout, stderr = await proc.communicate()

            out = stdout.decode("utf-8", errors="replace").strip()
            err = stderr.decode("utf-8", errors="replace").strip()

            if proc.returncode == 0:
                msg = out or "✅ 执行成功"
                return ToolResult(success=True, content=msg)
            else:
                return ToolResult(
                    success=False,
                    error=err or out or f"退出码: {proc.returncode}",
                )

        except Exception as e:
            return ToolResult(success=False, error=f"执行失败: {e}")
