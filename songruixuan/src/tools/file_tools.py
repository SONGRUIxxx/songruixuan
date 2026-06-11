"""SONGRUIXUAN 文件工具 — 最小定制版"""

from pathlib import Path

import httpx

from src.tool_base import Tool, ToolResult


class ReadTool(Tool):
    """读取文件内容"""

    @property
    def name(self) -> str:
        return "read_file"

    @property
    def description(self) -> str:
        return "读取指定文件的内容。用于查看笔记、脚本、参考卡片等。"

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "要读取的文件路径（相对或绝对路径）",
                }
            },
            "required": ["file_path"],
        }

    def __init__(self, workspace_dir: str):
        self.workspace = Path(workspace_dir)

    async def execute(self, file_path: str) -> ToolResult:
        path = Path(file_path)
        if not path.is_absolute():
            path = self.workspace / path

        if not path.exists():
            return ToolResult(
                success=False,
                error=f"文件不存在: {path}",
            )

        try:
            content = path.read_text(encoding="utf-8")
            return ToolResult(
                success=True,
                content=f"📄 {path.name}\n\n{content}",
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"读取失败: {e}",
            )


class WriteTool(Tool):
    """写入文件"""

    @property
    def name(self) -> str:
        return "write_file"

    @property
    def description(self) -> str:
        return "将内容写入文件。用于保存脚本、笔记、卡片等。会覆盖已有文件。"

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "要写入的文件路径",
                },
                "content": {
                    "type": "string",
                    "description": "要写入的内容",
                },
            },
            "required": ["file_path", "content"],
        }

    def __init__(self, workspace_dir: str):
        self.workspace = Path(workspace_dir)

    async def execute(self, file_path: str, content: str) -> ToolResult:
        path = self._resolve(file_path)

        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            path.write_text(content, encoding="utf-8")
            return ToolResult(
                success=True,
                content=f"✅ 已保存: {path}",
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"写入失败: {e}",
            )

    def _resolve(self, file_path: str) -> Path:
        path = Path(file_path)
        if not path.is_absolute():
            path = self.workspace / file_path
        return path


class MoveTool(Tool):
    """移动/重命名文件或文件夹"""

    @property
    def name(self) -> str:
        return "move_file"

    @property
    def description(self) -> str:
        return "移动文件到另一个位置，或重命名文件。相当于 mv 命令。"

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "source": {
                    "type": "string",
                    "description": "源文件路径",
                },
                "dest": {
                    "type": "string",
                    "description": "目标路径",
                },
            },
            "required": ["source", "dest"],
        }

    def __init__(self, workspace_dir: str):
        self.workspace = Path(workspace_dir)

    async def execute(self, source: str, dest: str) -> ToolResult:
        src = self._resolve(source)
        dst = self._resolve(dest)

        if not src.exists():
            return ToolResult(success=False, error=f"文件不存在: {src}")

        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            src.rename(dst)
            return ToolResult(success=True, content=f"✅ 已移动: {src} → {dst}")
        except Exception as e:
            return ToolResult(success=False, error=f"移动失败: {e}")

    def _resolve(self, file_path: str) -> Path:
        path = Path(file_path)
        if not path.is_absolute():
            path = self.workspace / file_path
        return path


class DeleteTool(Tool):
    """删除文件"""

    @property
    def name(self) -> str:
        return "delete_file"

    @property
    def description(self) -> str:
        return "删除文件。⚠️ 不可恢复，请谨慎使用。"

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "要删除的文件路径",
                },
            },
            "required": ["file_path"],
        }

    def __init__(self, workspace_dir: str):
        self.workspace = Path(workspace_dir)

    async def execute(self, file_path: str) -> ToolResult:
        path = self._resolve(file_path)

        if not path.exists():
            return ToolResult(success=False, error=f"文件不存在: {path}")

        try:
            path.unlink()
            return ToolResult(success=True, content=f"🗑️ 已删除: {path}")
        except Exception as e:
            return ToolResult(success=False, error=f"删除失败: {e}")

    def _resolve(self, file_path: str) -> Path:
        path = Path(file_path)
        if not path.is_absolute():
            path = self.workspace / file_path
        return path


class DownloadTool(Tool):
    """下载文件 — 从 URL 保存到本地"""

    @property
    def name(self) -> str:
        return "download"

    @property
    def description(self) -> str:
        return "从 URL 下载文件保存到本地。用于下载生成的音乐、图片等。"

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "下载链接",
                },
                "save_path": {
                    "type": "string",
                    "description": "保存路径，如 output/song.mp3",
                },
            },
            "required": ["url", "save_path"],
        }

    def __init__(self, workspace_dir: str):
        self.workspace = Path(workspace_dir)

    async def execute(self, url: str, save_path: str) -> ToolResult:
        path = Path(save_path)
        if not path.is_absolute():
            path = self.workspace / save_path
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            async with httpx.AsyncClient(timeout=120) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                path.write_bytes(resp.content)

            size_kb = round(len(resp.content) / 1024, 1)
            return ToolResult(
                success=True,
                content=f"✅ 已下载: {path} ({size_kb} KB)",
            )
        except Exception as e:
            return ToolResult(success=False, error=f"下载失败: {e}")


class ListDirTool(Tool):
    """列出目录内容"""

    @property
    def name(self) -> str:
        return "list_dir"

    @property
    def description(self) -> str:
        return "列出指定目录下的文件和子目录。用于浏览素材库、影库、乐库等。"

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "dir_path": {
                    "type": "string",
                    "description": "要列出的目录路径",
                }
            },
            "required": ["dir_path"],
        }

    def __init__(self, workspace_dir: str):
        self.workspace = Path(workspace_dir)

    async def execute(self, dir_path: str) -> ToolResult:
        path = Path(dir_path)
        if not path.is_absolute():
            path = self.workspace / dir_path

        if not path.exists() or not path.is_dir():
            return ToolResult(
                success=False,
                error=f"目录不存在: {path}",
            )

        items = []
        for item in sorted(path.iterdir()):
            prefix = "📁" if item.is_dir() else "📄"
            items.append(f"  {prefix} {item.name}")

        return ToolResult(
            success=True,
            content=f"📂 {path}\n" + "\n".join(items) if items else "(空目录)",
        )
