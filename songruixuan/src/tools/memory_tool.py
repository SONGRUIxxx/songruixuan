"""SONGRUIXUAN 记忆工具 — 跨会话记忆"""

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import yaml as _yaml

from src.tool_base import Tool, ToolResult

if TYPE_CHECKING:
    from src.tools.vector_memory import VectorMemory


class SaveMemoryTool(Tool):
    """保存记忆 — 索引到 ChromaDB，下次对话可语义搜索"""

    def __init__(self, memory_dir: str, vector_memory: "VectorMemory" = None):
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.vector_memory = vector_memory

    @property
    def name(self) -> str:
        return "save_memory"

    @property
    def description(self) -> str:
        return (
            "将重要信息保存到记忆库，下次对话可以语义搜索到。"
            "用于保存创作决策、偏好、灵感、待办事项等。"
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "记忆标题"},
                "content": {"type": "string", "description": "要保存的完整内容"},
                "tags": {"type": "string", "description": "逗号分隔的标签"},
            },
            "required": ["title", "content"],
        }

    async def execute(self, title: str, content: str, tags: str = "") -> ToolResult:
        tag_list = [t.strip() for t in tags.split(",") if t.strip()]
        tag_line = " ".join(f"`#{t}`" for t in tag_list) if tag_list else ""

        full_content = f"# {title}\n\n"
        if tag_line:
            full_content += f"{tag_line}\n\n"
        full_content += f"*保存于 {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n\n"
        full_content += content

        if self.vector_memory:
            self.vector_memory.add_memory(title, full_content)

        return ToolResult(success=True, content=f"✅ 已保存: {title}")


class ReadMemoryTool(Tool):
    """读取记忆 — 查看 conversations.yaml 中的对话记录"""

    @property
    def name(self) -> str:
        return "read_memory"

    @property
    def description(self) -> str:
        return (
            "读取记忆库中的对话列表或搜索指定内容。"
            "用于回顾之前的创作决策、偏好和灵感。"
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "留空则列出所有对话，填入关键词则搜索匹配的对话",
                },
            },
            "required": [],
        }

    def __init__(self, memory_dir: str):
        self.memory_dir = Path(memory_dir)

    async def execute(self, query: str = "") -> ToolResult:
        yaml_path = self.memory_dir / "conversations.yaml"
        if not yaml_path.exists():
            return ToolResult(success=True, content="📭 记忆库为空。")

        try:
            data = _yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
        except Exception:
            return ToolResult(success=True, content="📭 记忆库为空。")

        sessions = data.get("sessions", [])
        if not sessions:
            return ToolResult(success=True, content="📭 记忆库为空。")

        if not query:
            lines = [f"📚 记忆列表（共 {len(sessions)} 次对话）:\n"]
            for s in sessions:
                sid = s.get("id", "?")
                date = s.get("date", "?")
                summary = s.get("summary", "")[:80]
                lines.append(f"  🌿 第{sid}次对话 ({date}): {summary}")
            return ToolResult(success=True, content="\n".join(lines))

        found = []
        for s in sessions:
            transcript = s.get("transcript", "")
            summary = s.get("summary", "")
            if query.lower() in (summary + transcript[:5000]).lower():
                sid = s.get("id", "?")
                date = s.get("date", "?")
                found.append(f"📄 第{sid}次对话 ({date})\n\n{transcript[:3000]}\n---")

        if not found:
            return ToolResult(success=True, content=f"🔍 未找到与「{query}」相关的记忆。")

        return ToolResult(success=True, content=f"🔍 找到 {len(found)} 条:\n\n" + "\n".join(found))


class RecallContextTool(Tool):
    """回忆上下文 — 加载最近对话摘要"""

    @property
    def name(self) -> str:
        return "recall_context"

    @property
    def description(self) -> str:
        return "加载最近的对话摘要，用于快速回忆上下文。"

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "count": {"type": "integer", "description": "加载最近几次对话（默认 5）"},
            },
            "required": [],
        }

    def __init__(self, memory_dir: str):
        self.memory_dir = Path(memory_dir)

    async def execute(self, count: int = 5) -> ToolResult:
        yaml_path = self.memory_dir / "conversations.yaml"
        if not yaml_path.exists():
            return ToolResult(success=True, content="📭 暂无记忆。")

        try:
            data = _yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
        except Exception:
            return ToolResult(success=True, content="📭 暂无记忆。")

        sessions = data.get("sessions", [])
        if not sessions:
            return ToolResult(success=True, content="📭 暂无记忆。")

        recent = sessions[-count:]
        lines = [f"🧠 最近 {len(recent)} 次对话:\n"]
        for s in reversed(recent):
            sid = s.get("id", "?")
            date = s.get("date", "?")
            summary = s.get("summary", "")[:300]
            lines.append(f"### 第{sid}次对话 ({date})\n{summary}\n")

        return ToolResult(success=True, content="\n".join(lines))
