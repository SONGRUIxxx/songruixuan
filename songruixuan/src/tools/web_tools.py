"""🌐 网页抓取 — 获取文章全文"""

import httpx
from bs4 import BeautifulSoup

from src.tool_base import Tool, ToolResult


class WebFetchTool(Tool):
    """抓取网页全文"""

    @property
    def name(self) -> str:
        return "web_fetch"

    @property
    def description(self) -> str:
        return (
            "抓取指定 URL 的网页内容，提取正文文字。"
            "用于获取豆瓣影评、知乎回答、电影分析文章等完整内容。"
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "要抓取的网页 URL"},
            },
            "required": ["url"],
        }

    async def execute(self, url: str) -> ToolResult:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
        }
        try:
            async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
                resp = await client.get(url, headers=headers)
                resp.raise_for_status()
                html = resp.text

            soup = BeautifulSoup(html, "html.parser")
            # 去掉脚本和样式
            for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
                tag.decompose()

            text = soup.get_text(separator="\n")
            lines = [l.strip() for l in text.split("\n") if l.strip()]
            # 去重相邻重复行
            clean = []
            for l in lines:
                if not clean or l != clean[-1]:
                    clean.append(l)
            content = "\n".join(clean)

            if len(content) < 100:
                return ToolResult(success=False, error="页面内容过短，可能被反爬")

            return ToolResult(success=True, content=content)

        except httpx.HTTPStatusError as e:
            return ToolResult(success=False, error=f"HTTP {e.response.status_code}")
        except Exception as e:
            return ToolResult(success=False, error=f"抓取失败: {e}")
