"""👁️ 视觉理解 — 火山引擎豆包 Seed Vision"""

import base64
from pathlib import Path

import httpx

from src.tool_base import Tool, ToolResult


class VisionTool(Tool):
    """豆包 Seed Vision 视觉理解"""

    def __init__(self, api_key: str, model: str = "doubao-seed-2-1-pro-260628"):
        self.api_key = api_key
        self.model = model
        self.endpoint = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"

    @property
    def name(self) -> str:
        return "vision"

    @property
    def description(self) -> str:
        return "分析图片：构图、色调、光线、情绪。用于电影镜头分析、摄影作品描述。"

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "image": {"type": "string", "description": "图片路径（本地文件或 http URL）"},
                "prompt": {"type": "string", "description": "想问什么"},
            },
            "required": ["image"],
        }

    async def execute(self, image: str, prompt: str = "") -> ToolResult:
        question = prompt or "用简洁的中文描述这张图片的内容、构图和色调。"

        if image.startswith("http://") or image.startswith("https://"):
            image_url = image
        else:
            path = Path(image)
            if not path.exists():
                return ToolResult(success=False, error=f"图片不存在: {image}")
            image_url = self._encode(path)

        try:
            async with httpx.AsyncClient(timeout=120) as client:
                resp = await client.post(
                    self.endpoint,
                    json={
                        "model": self.model,
                        "messages": [{
                            "role": "user",
                            "content": [
                                {"type": "image_url", "image_url": {"url": image_url}},
                                {"type": "text", "text": question},
                            ],
                        }],
                    },
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                )
                resp.raise_for_status()
                data = resp.json()

            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            if not content:
                return ToolResult(success=False, error="VLM 返回空内容")

            return ToolResult(success=True, content=f"👁️ {content}")

        except Exception as e:
            detail = str(e)
            try:
                detail = e.response.json()
            except Exception:
                try:
                    detail = e.response.text[:500]
                except Exception:
                    pass
            return ToolResult(success=False, error=f"图片分析失败: {detail}")

    def _encode(self, path: Path) -> str:
        ext = path.suffix.lower()
        mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
                    ".gif": "image/gif", ".webp": "image/webp", ".bmp": "image/bmp"}
        mime = mime_map.get(ext, "image/jpeg")
        data = base64.b64encode(path.read_bytes()).decode("utf-8")
        return f"data:{mime};base64,{data}"
