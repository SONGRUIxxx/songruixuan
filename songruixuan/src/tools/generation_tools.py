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
        return "生成图片（Seedream 5.0 Pro）。参数：prompt（描述词）。返回保存路径。"

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "description": "图片描述/提示词"},
            },
            "required": ["prompt"],
        }

    async def execute(self, prompt: str) -> ToolResult:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        try:
            async with httpx.AsyncClient(timeout=120) as client:
                resp = await client.post(
                    self.endpoint,
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "size": "2K",
                        "output_format": "png",
                        "watermark": False,
                    },
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                )
                resp.raise_for_status()
                data = resp.json()

            # 下载图片
            img_data = None
            # 优先取 b64_json
            for item in data.get("data", []):
                if "b64_json" in item:
                    img_data = base64.b64decode(item["b64_json"])
                    break
                if "url" in item:
                    async with httpx.AsyncClient(timeout=60) as dl:
                        r = await dl.get(item["url"])
                        r.raise_for_status()
                        img_data = r.content
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
        self.endpoint = "https://ark.cn-beijing.volces.com/api/v3/contents/generations"
        self.model = "doubao-seedance-2-0-260128"

    @property
    def name(self) -> str:
        return "generate_video"

    @property
    def description(self) -> str:
        return "生成视频（Seedance 2.0）。参数：prompt（描述词）。返回任务 ID 和保存路径。"

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "description": "视频描述/脚本"},
            },
            "required": ["prompt"],
        }

    async def execute(self, prompt: str) -> ToolResult:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        try:
            async with httpx.AsyncClient(timeout=300) as client:
                # 提交任务
                resp = await client.post(
                    self.endpoint,
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "size": "720P",
                        "duration": 5,
                    },
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                )
                resp.raise_for_status()
                data = resp.json()

            task_id = data.get("id", data.get("task_id", ""))
            if not task_id:
                return ToolResult(success=False, error="Seedance 未返回任务 ID")

            # 轮询等待完成
            for _ in range(60):
                await __import__("asyncio").sleep(5)
                async with httpx.AsyncClient(timeout=60) as poll:
                    r = await poll.get(
                        f"https://ark.cn-beijing.volces.com/api/v3/contents/generations/{task_id}",
                        headers={"Authorization": f"Bearer {self.api_key}"},
                    )
                    r.raise_for_status()
                    task_data = r.json()
                status = task_data.get("status", "")
                if status in ("succeeded", "completed", "done"):
                    video_url = ""
                    for item in task_data.get("data", task_data.get("output", {}).get("data", [])):
                        video_url = item.get("url", "")
                        if video_url:
                            break
                    if not video_url:
                        video_url = task_data.get("video_url", task_data.get("url", ""))

                    if video_url:
                        async with httpx.AsyncClient(timeout=120) as dl:
                            vr = await dl.get(video_url)
                            vr.raise_for_status()
                            ts = int(time.time())
                            out_path = self.output_dir / f"seedance_{ts}.mp4"
                            out_path.write_bytes(vr.content)
                            return ToolResult(success=True, content=f"✅ 视频已保存: {out_path}")
                    return ToolResult(success=False, error="Seedance 完成但无下载链接")
                if status in ("failed", "error"):
                    return ToolResult(success=False, error=f"Seedance 失败: {task_data}")

            return ToolResult(success=False, error="Seedance 超时未完成")

        except Exception as e:
            return ToolResult(success=False, error=f"Seedance 失败: {e}")
