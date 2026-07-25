"""🎨 生成工具 — Seedream 图片 + Seedance 视频（火山引擎）"""

import base64
import time
from pathlib import Path

import httpx

from src.tool_base import Tool, ToolResult


class SeedreamTool(Tool):
    """豆包 Seedream 5.0 Pro 图片生成"""

    def __init__(self, api_key: str, output_dir: str = r"D:\Workspace"):
        self.api_key = api_key
        self.output_dir = Path(output_dir)
        self.endpoint = "https://ark.cn-beijing.volces.com/api/v3/images/generations"
        self.model = "doubao-seedream-5-0-pro-260628"

    @property
    def name(self) -> str:
        return "generate_image"

    @property
    def description(self) -> str:
        return "生成图片（Seedream 5.0 Pro）。参数：prompt（描述词），images（参考图路径列表，可选）。"

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "description": "图片描述/提示词"},
                "images": {"type": "array", "items": {"type": "string"}, "description": "参考图片路径列表（可选）"},
            },
            "required": ["prompt"],
        }

    async def execute(self, prompt: str, images: list[str] | None = None) -> ToolResult:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        body: dict = {
            "model": self.model,
            "prompt": prompt,
            "size": "2K",
            "output_format": "png",
            "watermark": False,
        }
        if images:
            refs = []
            for img in images[:10]:
                if img.startswith("http://") or img.startswith("https://"):
                    refs.append(img)  # URL 直接用
                else:
                    p = Path(img)
                    if p.exists():
                        b64 = base64.b64encode(p.read_bytes()).decode("utf-8")
                        ext = p.suffix.lower().replace(".jpeg", "jpg").lstrip(".")
                        refs.append(f"data:image/{ext};base64,{b64}")
            if refs:
                body["image"] = refs if len(refs) > 1 else refs[0]

        try:
            async with httpx.AsyncClient(timeout=120) as client:
                resp = await client.post(
                    self.endpoint,
                    json=body,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                )
                resp.raise_for_status()
                data = resp.json()

            img_data = None
            for item in data.get("data", []):
                if "url" in item:
                    async with httpx.AsyncClient(timeout=60) as dl:
                        r = await dl.get(item["url"])
                        r.raise_for_status()
                        img_data = r.content
                        break
                if "b64_json" in item:
                    img_data = base64.b64decode(item["b64_json"])
                    break

            if not img_data:
                return ToolResult(success=False, error="Seedream 返回无图片数据")

            ts = int(time.time())
            out_path = self.output_dir / f"seedream_{ts}.png"
            out_path.write_bytes(img_data)
            return ToolResult(success=True, content=f"✅ 图片已保存: {out_path}")

        except Exception as e:
            return ToolResult(success=False, error=f"Seedream 失败: {e}")


class SeedanceTool(Tool):
    """豆包 Seedance 2.0 视频生成"""

    def __init__(self, api_key: str, output_dir: str = r"D:\Workspace"):
        self.api_key = api_key
        self.output_dir = Path(output_dir)
        self.base_url = "https://ark.cn-beijing.volces.com/api/v3"
        self.model = "doubao-seedance-2-0-260128"

    @property
    def name(self) -> str:
        return "generate_video"

    @property
    def description(self) -> str:
        return "生成视频（Seedance 2.0）。参数：prompt（描述词），images（参考图路径列表，可选）。"

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "description": "视频描述/脚本"},
                "images": {"type": "array", "items": {"type": "string"}, "description": "参考图片路径列表（可选）"},
            },
            "required": ["prompt"],
        }

    async def execute(self, prompt: str, images: list[str] | None = None, refs: list | None = None) -> ToolResult:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        content_list = []
        if prompt:
            content_list.append({"type": "text", "text": prompt})

        # 处理混合参考（元组格式：(type, url)）
        items = []
        if refs:
            items = refs  # [(type, url), ...]
        elif images:
            items = [("image", u) for u in images]

        for item in items[:9]:
            mtype, url = item if isinstance(item, tuple) else ("image", item)
            url = url if isinstance(url, str) else item if isinstance(item, str) else ""
            if not url:
                continue
            if mtype == "video":
                content_list.append({
                    "type": "video_url",
                    "video_url": {"url": url},
                    "role": "reference_video",
                })
            else:
                if url.startswith("http://") or url.startswith("https://"):
                    content_list.append({
                        "type": "image_url",
                        "image_url": {"url": url},
                        "role": "reference_image",
                    })
                else:
                    p = Path(url)
                    if p.exists():
                        b64 = base64.b64encode(p.read_bytes()).decode("utf-8")
                        ext = p.suffix.lower().replace(".jpeg", "jpg").lstrip(".")
                        content_list.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:image/{ext};base64,{b64}"},
                            "role": "reference_image",
                        })

        body = {
            "model": self.model,
            "content": content_list,
            "duration": 10,
            "ratio": "16:9",
            "generate_audio": False,
            "watermark": False,
        }

        try:
            async with httpx.AsyncClient(timeout=300) as client:
                resp = await client.post(
                    f"{self.base_url}/contents/generations/tasks",
                    json=body,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                )
                resp.raise_for_status()
                data = resp.json()

            task_id = data.get("id", data.get("task_id", ""))
            if not task_id:
                return ToolResult(success=False, error=f"Seedance 未返回任务 ID: {data}")

            # 轮询等待完成
            for _ in range(120):
                await __import__("asyncio").sleep(5)
                async with httpx.AsyncClient(timeout=60) as poll:
                    r = await poll.get(
                        f"{self.base_url}/contents/generations/tasks/{task_id}",
                        headers={"Authorization": f"Bearer {self.api_key}"},
                    )
                    r.raise_for_status()
                    task_data = r.json()
                status = task_data.get("status", "")
                if status == "succeeded":
                    td = task_data if isinstance(task_data, dict) else {}
                    video_url = ""
                    # content 是 dict 时：content.video_url
                    content = td.get("content", {})
                    if isinstance(content, dict):
                        video_url = content.get("video_url", "")
                    # content 是 list 时
                    elif isinstance(content, list):
                        for item in content:
                            if isinstance(item, dict) and item.get("type") == "video_url":
                                vu = item.get("video_url", "")
                                video_url = vu if isinstance(vu, str) else vu.get("url", "")
                                break
                    # 兜底：顶层 video_url
                    if not video_url:
                        video_url = td.get("video_url", "")
                        if isinstance(video_url, dict):
                            video_url = video_url.get("url", "")
                    if not video_url:
                        return ToolResult(success=False, error=f"Seedance 完成但无下载链接。原始数据: {str(td)[:500]}")
                    if video_url:
                        async with httpx.AsyncClient(timeout=120) as dl:
                            vr = await dl.get(video_url)
                            vr.raise_for_status()
                            ts = int(time.time())
                            out_path = self.output_dir / f"seedance_{ts}.mp4"
                            out_path.write_bytes(vr.content)
                            return ToolResult(success=True, content=f"✅ 视频已保存: {out_path}")
                    return ToolResult(success=False, error=f"Seedance 完成但无下载链接: {task_data}")
                if status in ("failed", "error", "cancelled", "expired"):
                    err = task_data.get("error", task_data)
                    return ToolResult(success=False, error=f"Seedance {status}: {err}")

            return ToolResult(success=False, error="Seedance 超时未完成")

        except Exception as e:
            import traceback
            detail = f"{e}\n{traceback.format_exc()[-500:]}"
            try:
                detail += f"\nResponse: {e.response.text[:800]}"
            except Exception:
                pass
            return ToolResult(success=False, error=f"Seedance 失败: {detail}")
