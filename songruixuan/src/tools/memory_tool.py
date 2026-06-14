"""SONGRUIXUAN 记忆工具 — 跨会话记忆"""

from datetime import datetime
from pathlib import Path
from typing import Optional, TYPE_CHECKING

from src.tool_base import Tool, ToolResult

if TYPE_CHECKING:
    from src.tools.vector_memory import VectorMemory


class SaveMemoryTool(Tool):
    """保存记忆 — 将重要信息持久化到 memory/conversations/，并自动索引到 ChromaDB"""

    def __init__(self, memory_dir: str, vector_memory: Optional["VectorMemory"] = None):
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.vector_memory = vector_memory

    @property
    def name(self) -> str:
        return "save_memory"

    @property
    def description(self) -> str:
        return (
            "将重要信息保存到记忆库，下次对话可以读取。"
            "用于保存创作决策、偏好、灵感、待办事项等。"
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "记忆标题（简洁描述内容）",
                },
                "content": {
                    "type": "string",
                    "description": "要保存的完整内容",
                },
                "tags": {
                    "type": "string",
                    "description": "逗号分隔的标签，如：旁白,成都老街,抒情",
                },
            },
            "required": ["title", "content"],
        }

    async def execute(
        self, title: str, content: str, tags: str = ""
    ) -> ToolResult:
        date_str = datetime.now().strftime("%Y-%m-%d_%H%M")
        safe_title = "".join(c for c in title if c.isalnum() or c in "_- ")
        safe_title = safe_title.strip()[:50]
        filename = f"{date_str}_{safe_title}.md"

        tag_list = [t.strip() for t in tags.split(",") if t.strip()]
        tag_line = " ".join(f"`#{t}`" for t in tag_list) if tag_list else ""

        full_content = f"# {title}\n\n"
        if tag_line:
            full_content += f"{tag_line}\n\n"
        full_content += f"*保存于 {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n\n"
        full_content += content

        # 追加到 journal.md
        journal = self.memory_dir / "journal.md"
        journal.parent.mkdir(parents=True, exist_ok=True)
        with open(journal, "a", encoding="utf-8") as f:
            if journal.stat().st_size > 0:
                f.write(f"\n\n---\n\n{full_content}")
            else:
                f.write(full_content)

        # 如果向量记忆可用，自动索引到 ChromaDB
        if self.vector_memory:
            self.vector_memory.add_memory(title, full_content)

        return ToolResult(
            success=True,
            content=f"✅ 已保存到对话日记: {title}",
        )


class ReadMemoryTool(Tool):
    """读取记忆 — 查看之前的记忆内容"""

    @property
    def name(self) -> str:
        return "read_memory"

    @property
    def description(self) -> str:
        return (
            "读取记忆库中的文件列表或指定记忆内容。"
            "用于回顾之前的创作决策、偏好和灵感。"
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "留空则列出所有记忆文件，"
                        "填入关键词则搜索并读取匹配的记忆"
                    ),
                },
            },
            "required": [],
        }

    def __init__(self, memory_dir: str):
        self.memory_dir = Path(memory_dir)

    async def execute(self, query: str = "") -> ToolResult:
        if not self.memory_dir.exists():
            return ToolResult(
                success=True,
                content="📭 记忆库为空。",
            )

        files = sorted(
            self.memory_dir.glob("*.md"),
            key=lambda f: f.stat().st_mtime,
            reverse=True,
        )

        if not files:
            return ToolResult(
                success=True,
                content="📭 记忆库为空。",
            )

        # 没有关键词 → 列出所有
        if not query:
            lines = ["📚 记忆列表（最新在前）:\n"]
            for f in files:
                mtime = datetime.fromtimestamp(f.stat().st_mtime)
                date_str = mtime.strftime("%m-%d %H:%M")
                lines.append(f"  📄 {f.stem}  ({date_str})")
            return ToolResult(
                success=True,
                content="\n".join(lines),
            )

        # 有关键词 → 搜索并读取
        found = []
        for f in files:
            content = f.read_text(encoding="utf-8")
            if query.lower() in content.lower() or query.lower() in f.stem.lower():
                found.append(f"📄 {f.stem}\n\n{content}\n---")

        if not found:
            return ToolResult(
                success=True,
                content=f"🔍 未找到与「{query}」相关的记忆。",
            )

        return ToolResult(
            success=True,
            content=f"🔍 找到 {len(found)} 条相关记忆:\n\n" + "\n".join(found),
        )


class RecallContextTool(Tool):
    """回忆上下文 — 加载最近的记忆作为对话背景"""

    @property
    def name(self) -> str:
        return "recall_context"

    @property
    def description(self) -> str:
        return "加载最近的记忆摘要，用于在新对话开始时快速回忆上下文。"

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "count": {
                    "type": "integer",
                    "description": "加载最近几条记忆（默认 5）",
                },
            },
            "required": [],
        }

    def __init__(self, memory_dir: str):
        self.memory_dir = Path(memory_dir)

    async def execute(self, count: int = 5) -> ToolResult:
        if not self.memory_dir.exists():
            return ToolResult(
                success=True,
                content="📭 暂无记忆。",
            )

        files = sorted(
            self.memory_dir.glob("*.md"),
            key=lambda f: f.stat().st_mtime,
            reverse=True,
        )

        if not files:
            return ToolResult(
                success=True,
                content="📭 暂无记忆。",
            )

        lines = ["🧠 最近的记忆:\n"]
        for f in files[:count]:
            content = f.read_text(encoding="utf-8")
            lines.append(f"### {f.stem}\n{content}\n")

        return ToolResult(
            success=True,
            content="\n".join(lines),
        )
