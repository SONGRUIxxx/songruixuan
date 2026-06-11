"""📸 摄影库 — 扁平大库，无限编号"""

import os
import yaml
from pathlib import Path
from typing import Optional, TYPE_CHECKING

from src.tool_base import Tool, ToolResult

if TYPE_CHECKING:
    from src.tools.vector_memory import VectorMemory
    from src.tools.vision_tool import VisionTool


class PhotoImportTool(Tool):
    """导入照片：扫根目录 → 继续编号 → vision 分析 → 追加 index.yaml → ChromaDB"""

    def __init__(self, photo_dir: str, vector_memory=None, vision_tool=None):
        self.base = Path(photo_dir)
        self.vm = vector_memory
        self.vision = vision_tool

    @property
    def name(self) -> str:
        return "photo_import"

    @property
    def description(self) -> str:
        return "【必须调用此工具】用户说「处理照片」「生成索引」时你必须调用 photo_import，不要自己逐张调 vision。工具会自动扫描目录、分析每张、写入 index.yaml。"

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "label": {"type": "string", "description": "可选标签"},
                "range": {"type": "string", "description": "可选范围，如 00001-00100"},
            },
            "required": [],
        }

    async def execute(self, label: str = "", range: str = "") -> ToolResult:
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
            for fr in existing.get("photos", []):
                try:
                    n = int("".join(c for c in fr.get("file", "") if c.isdigit()))
                    if n > max_num:
                        max_num = n
                except Exception:
                    pass
        # 没有索引 → 从已编号文件名推断
        if max_num == 0:
            for f in self.base.iterdir():
                try:
                    n = int("".join(c for c in f.stem if c.isdigit()))
                    if n > max_num:
                        max_num = n
                except Exception:
                    pass

        # 收集已描述图片（用于跳过）
        existing_desc = set()
        for p in existing.get("photos", []):
            if p.get("desc", "").strip() and p.get("file"):
                existing_desc.add(p["file"])

        valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif",
                      ".tiff", ".tif", ".cr2", ".nef", ".arw", ".dng", ".heic", ".heif"}

        images = sorted(
            [f for f in self.base.iterdir()
             if f.is_file() and f.suffix.lower() in valid_exts
             and f.name not in ("index.yaml", "card.md")
             and f.name not in existing_desc],  # 已有描述的跳过
            key=lambda f: f.stat().st_mtime,
        )
        if not images:
            return ToolResult(success=False, error="没有新图片")

        # 范围过滤
        if range:
            parts = range.split("-")
            r_start = int(parts[0]) if len(parts) >= 1 else None
            r_end = int(parts[1]) if len(parts) >= 2 else None
            if r_start is not None:
                images = [i for i in images if int("".join(c for c in i.stem if c.isdigit()) or "0") >= r_start]
            if r_end is not None:
                images = [i for i in images if int("".join(c for c in i.stem if c.isdigit()) or "0") <= r_end]
            if not images:
                return ToolResult(success=False, error=f"范围 {range} 内没有匹配的图片")

        # 视觉兼容格式
        vision_ok = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

        photos = existing.get("photos", [])
        processed_count = 0
        analyzed_count = 0

        for img in images:
            ext = img.suffix.lower()
            if img.stem.isdigit():
                # 已编号的不重命名，直接分析
                new_path = img
                new_name = img.name
            else:
                max_num += 1
                new_name = f"{max_num:05d}{ext}"
                new_path = self.base / new_name
                os.rename(str(img), str(new_path))

            processed_count += 1

            desc = ""
            vision_path = str(new_path)

            # RAW 文件：临时转 JPEG → VLM 分析 → 删临时文件
            raw_exts = {".cr2", ".nef", ".arw", ".dng"}
            tmp_path = None
            if ext in raw_exts and self.vision:
                try:
                    import rawpy, tempfile
                    with rawpy.imread(str(new_path)) as raw:
                        rgb = raw.postprocess()
                    from PIL import Image
                    tmp_path = os.path.join(tempfile.gettempdir(), f"_srx_photo_{max_num:04d}.jpg")
                    Image.fromarray(rgb).save(tmp_path, "JPEG", quality=85)
                    vision_path = tmp_path
                except Exception:
                    vision_path = ""  # 转换失败，跳过分析

            if vision_path and ext in (vision_ok | raw_exts) and self.vision:
                vr = await self.vision.execute(
                    image=vision_path,
                    prompt="用中文描述这张照片的画面、构图、色调和情绪氛围。给出 3-5 个标签。",
                )
                if vr.success:
                    desc = vr.content.replace("👁️ ", "").strip()
                # 删临时文件
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except Exception:
                        pass
            elif ext == ".gif":
                desc = "(GIF 动图)"
            elif ext in (".tiff", ".tif", ".heic", ".heif"):
                desc = f"(高规格格式: {ext})"

            if desc:
                analyzed_count += 1

            photos.append({
                "file": new_name,
                "format": ext,
                "label": label,
                "desc": desc,
            })

            # 每张立刻写索引，不怕中断
            idx_path.write_text(
                yaml.dump({"photos": photos}, allow_unicode=True, default_flow_style=False, sort_keys=False),
                encoding="utf-8",
            )

        # ChromaDB
        indexed = 0
        if self.vm:
            pc = self.vm.get_or_create_photo_collection()
            exist_ids = set(pc.get()["ids"]) if pc.count() > 0 else set()
            for fr in photos[-processed_count:]:
                fname = fr["file"]
                pid = f"photo_{fname}"
                import hashlib
                ph = hashlib.md5((fr.get("desc", "")[:200]).encode()).hexdigest()[:8]
                chroma_id = f"{pid}__{ph}"
                if chroma_id not in exist_ids:
                    pc.add(
                        ids=[chroma_id],
                        documents=[f"{fname}: {fr.get('desc','')}"[:1200]],
                        metadatas=[{"file": fname, "desc": fr.get("desc", "")[:400], "label": label}],
                    )
                    indexed += 1

        return ToolResult(
            success=True,
            content=f"OK {processed_count} analyzed {analyzed_count} indexed {indexed}"
        )
