"""🎵 乐库 — 扁平大库，无限编号"""

import os
import yaml
from pathlib import Path
from typing import TYPE_CHECKING

from src.tool_base import Tool, ToolResult

if TYPE_CHECKING:
    from src.tools.vector_memory import VectorMemory
    from src.tools.vision_tool import VisionTool


class MusicImportTool(Tool):
    """导入音乐：扫根目录 → 继续编号 → vision 分析封面 → 追加 index.yaml → ChromaDB"""

    def __init__(self, music_dir: str, vector_memory=None, vision_tool=None):
        self.base = Path(music_dir)
        self.vm = vector_memory
        self.vision = vision_tool

    @property
    def name(self) -> str:
        return "music_import"

    @property
    def description(self) -> str:
        return (
            "【必须调用此工具】导入乐库文件（音频或封面图）。"
            "扫描 music_library_weekly 目录，继续编号，对封面图片用 vision 分析，写入 index.yaml，索引到 ChromaDB。"
            "用户说「导入音乐」「导入专辑」「导入封面」时调用。"
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "album": {"type": "string", "description": "专辑/作品名称，用于 ChromaDB 标记"},
                "label": {"type": "string", "description": "可选标签，如：电影原声、电子、古典"},
            },
            "required": ["album"],
        }

    async def execute(self, album: str, label: str = "") -> ToolResult:
        self.base.mkdir(parents=True, exist_ok=True)

        # 加载现有索引
        idx_path = self.base / "index.yaml"
        existing = {}
        max_num = 0
        if idx_path.exists():
            try:
                existing = yaml.safe_load(idx_path.read_text(encoding="utf-8")) or {}
            except Exception:
                pass
            for tr in existing.get("tracks", []):
                try:
                    n = int("".join(c for c in tr.get("file", "") if c.isdigit()))
                    if n > max_num:
                        max_num = n
                except Exception:
                    pass

        # 收集已导入文件（跳过已有描述的）
        existing_desc = set()
        for tr in existing.get("tracks", []):
            if tr.get("desc", "").strip() and tr.get("file"):
                existing_desc.add(tr["file"])

        # 支持的文件格式
        audio_exts = {".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg", ".wma"}
        image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

        files = sorted(
            [f for f in self.base.iterdir()
             if f.is_file() and f.suffix.lower() in (audio_exts | image_exts)
             and f.name not in ("index.yaml", "card.md")
             and f.name not in existing_desc],
            key=lambda f: f.stat().st_mtime,
        )
        if not files:
            return ToolResult(success=False, error="没有新文件")

        tracks = existing.get("tracks", [])
        processed = 0

        for f in files:
            ext = f.suffix.lower()
            max_num += 1
            new_name = f"{max_num:05d}{ext}"
            new_path = self.base / new_name
            os.rename(str(f), str(new_path))

            desc = ""
            # 封面图片 → VLM 分析
            if ext in image_exts and self.vision:
                vr = await self.vision.execute(
                    image=str(new_path),
                    prompt="用中文描述这张封面图的视觉风格、色调和情绪氛围。给出 3-5 个标签。",
                )
                if vr.success:
                    desc = vr.content.replace("👁️ ", "").strip()
            elif ext in audio_exts:
                desc = f"(音频文件: {ext})"

            tracks.append({
                "file": new_name,
                "album": album,
                "title": "",
                "label": label,
                "desc": desc,
            })
            processed += 1

            # 每处理一个立刻写索引
            idx_path.write_text(
                yaml.dump({"tracks": tracks}, allow_unicode=True, default_flow_style=False, sort_keys=False),
                encoding="utf-8",
            )

        # ChromaDB
        indexed = 0
        if self.vm:
            import hashlib
            mc = self.vm.get_or_create_music_collection()
            exist_ids = set(mc.get()["ids"]) if mc.count() > 0 else set()
            for tr in tracks[-processed:]:
                tname = tr["file"]
                doc_id = f"{album}__{tname}"
                th = hashlib.md5((tr.get("desc", "")[:200]).encode()).hexdigest()[:8]
                chroma_id = f"{doc_id}__{th}"
                if chroma_id not in exist_ids:
                    mc.add(
                        ids=[chroma_id],
                        documents=[f"《{album}》{tname}: {tr.get('desc','')}"[:1200]],
                        metadatas=[{"album": album, "track": tname, "desc": tr.get("desc","")[:400], "label": label}],
                    )
                    indexed += 1

        return ToolResult(
            success=True,
            content=f"OK {processed} analyzed, {indexed} indexed"
        )
