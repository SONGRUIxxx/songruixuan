"""🔍 搜索工具 — 必应（免 API Key，国内可用）"""

import re
import httpx
from bs4 import BeautifulSoup

from src.tool_base import Tool, ToolResult


class WebSearchTool(Tool):
    """必应网页搜索"""

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
            async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
                resp = await client.get(
                    "https://cn.bing.com/search",
                    params={"q": query, "ensearch": 1},
                    headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
                )
                resp.raise_for_status()

            soup = BeautifulSoup(resp.text, "html.parser")
            results = []
            for li in soup.select("li.b_algo"):
                title_el = li.select_one("h2 a")
                snippet_el = li.select_one(".b_caption p, .b_lineclamp2")
                if title_el:
                    title = title_el.get_text(strip=True)
                    raw_url = title_el.get("href", "")
                    # Bing 重定向 URL 中提取真实链接
                    import base64
                    real_url = raw_url
                    m = re.search(r'[?&]u=a1([^&]+)', raw_url)
                    if m:
                        try:
                            real_url = base64.b64decode(m.group(1) + "==").decode("utf-8")
                        except Exception:
                            pass
                    snippet = snippet_el.get_text(strip=True)[:300] if snippet_el else ""
                    results.append(f"{len(results)+1}. **{title}**\n{real_url}\n{snippet}")

            if not results:
                return ToolResult(success=True, content=f"🔍「{query}」无结果。")
            return ToolResult(success=True, content=f"🔍「{query}」:\n\n" + "\n\n".join(results[:6]))

        except Exception as e:
            return ToolResult(success=False, error=f"搜索失败: {e}")


class ImageSearchTool(Tool):
    """必应图片搜索 — 搜参考图"""

    @property
    def name(self) -> str:
        return "image_search"

    @property
    def description(self) -> str:
        return (
            "搜索网络图片，返回图片链接。用于找构图参考、色板灵感、"
            "电影截图风格参考等。"
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜索关键词"},
                "count": {"type": "integer", "description": "返回数量（默认 6）"},
            },
            "required": ["query"],
        }

    async def execute(self, query: str, count: int = 6) -> ToolResult:
        try:
            async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
                resp = await client.get(
                    "https://cn.bing.com/images/search",
                    params={"q": query},
                    headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
                )
                resp.raise_for_status()

            soup = BeautifulSoup(resp.text, "html.parser")
            results = []
            for img in soup.select(".imgpt a, a[href*='images']"):
                href = img.get("href", "")
                title = img.get("title", img.get_text(strip=True))[:100]
                if href and "http" in href.lower():
                    results.append(f"{title}\n{href}")

            if not results:
                # fallback: 从 murl 属性提取
                imgs = re.findall(r'"murl"\s*:\s*"([^"]+)"', resp.text)
                for u in imgs[:count]:
                    results.append(u)

            if not results:
                return ToolResult(success=True, content=f"🖼️「{query}」无结果。")
            return ToolResult(success=True, content=f"🖼️「{query}」:\n\n" + "\n\n".join(results[:count]))

        except Exception as e:
            return ToolResult(success=False, error=f"图片搜索失败: {e}")
