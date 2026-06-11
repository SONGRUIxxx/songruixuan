"""🔍 搜索工具 — DuckDuckGo 免费（无需 API Key）"""

import httpx

from src.tool_base import Tool, ToolResult


class WebSearchTool(Tool):
    """DuckDuckGo 网页搜索"""

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return (
            "搜索网页获取信息。用于查电影分析、音乐理论、摄影教程、"
            "纪录片参考资料等。返回标题、链接和摘要。"
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜索关键词"},
            },
            "required": ["query"],
        }

    async def execute(self, query: str) -> ToolResult:
        try:
            async with httpx.AsyncClient(timeout=20) as client:
                resp = await client.get(
                    "https://lite.duckduckgo.com/lite/",
                    params={"q": query},
                    headers={"User-Agent": "SONGRUIXUAN/0.2"},
                )
                resp.raise_for_status()

            results = []
            for line in resp.text.split("\n"):
                stripped = line.strip()
                if stripped and not stripped.startswith("<") and len(stripped) > 20:
                    results.append(stripped)

            if not results:
                return ToolResult(success=True, content=f"🔍「{query}」无结果。")

            lines = [f"🔍「{query}」:\n"]
            for i, s in enumerate(results, 1):
                lines.append(f"{i}. {s[:300]}")
                lines.append("")

            return ToolResult(success=True, content="\n".join(lines))

        except Exception as e:
            return ToolResult(success=False, error=f"搜索失败: {e}")


class ImageSearchTool(Tool):
    """DuckDuckGo 图片搜索 — 搜参考图"""

    @property
    def name(self) -> str:
        return "image_search"

    @property
    def description(self) -> str:
        return (
            "搜索网络图片，返回图片链接。用于找构图参考、色板灵感、"
            "电影截图风格参考等。结果可直接用于 vision 工具分析。"
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜索关键词"},
                "count": {
                    "type": "integer",
                    "description": "返回图片数量（默认 6）",
                },
            },
            "required": ["query"],
        }

    async def execute(self, query: str, count: int = 6) -> ToolResult:
        try:
            async with httpx.AsyncClient(timeout=20) as client:
                resp = await client.get(
                    "https://lite.duckduckgo.com/lite/",
                    params={"q": query, "iax": "images", "ia": "images"},
                    headers={"User-Agent": "SONGRUIXUAN/0.2"},
                )
                resp.raise_for_status()

            # 解析图片搜索结果
            text = resp.text
            results = []
            for line in text.split("\n"):
                stripped = line.strip()
                if stripped and len(stripped) > 10:
                    results.append(stripped)

            if not results:
                return ToolResult(success=True, content=f"🖼️「{query}」无结果。")

            lines = [f"🖼️「{query}」图片搜索结果:\n"]
            for i, s in enumerate(results[:count], 1):
                lines.append(f"{i}. {s}")
                lines.append("")

            return ToolResult(success=True, content="\n".join(lines))

        except Exception as e:
            return ToolResult(success=False, error=f"图片搜索失败: {e}")
