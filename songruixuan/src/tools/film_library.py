"""🎬 影库 — 扁平大库，无子目录"""

import os
import yaml
from pathlib import Path
from typing import Optional, TYPE_CHECKING

from src.tool_base import Tool, ToolResult

if TYPE_CHECKING:
    from src.tools.vector_memory import VectorMemory
    from src.tools.vision_tool import VisionTool


class FilmCreateTool(Tool):
    """在 notes.md 追加新电影的内容大纲"""

    def __init__(self, film_library_dir: str):
        self.base = Path(film_library_dir)

    @property
    def name(self) -> str:
        return "film_create"

    @property
    def description(self) -> str:
        return (
            "在影库 notes.md 中追加一部新电影的笔记。说「学习某电影」时调用。"
            "不建子目录，所有电影的笔记在同一个 notes.md 里。"
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {"name": {"type": "string", "description": "电影名称"}},
            "required": ["name"],
        }

    async def execute(self, name: str) -> ToolResult:
        notes = self.base / "notes.md"
        self.base.mkdir(parents=True, exist_ok=True)

        # 检查是否已有
        if notes.exists() and f"# {name}" in notes.read_text(encoding="utf-8"):
            return ToolResult(success=False, error=f"「{name}」已在 notes.md 中")

        entry = (
            f"\n\n---\n\n"
            f"# {name}\n\n"
            f"## 内容大纲\n\n\n"
            f"## 深度解析\n\n\n"
        )
        with open(notes, "a", encoding="utf-8") as f:
            f.write(entry)

        return ToolResult(
            success=True,
            content=f"🎬 「{name}」已追加到 notes.md\n请用 web_search 搜索该电影，填写内容大纲。"
        )


class FilmAnalyzeTool(Tool):
    """读 notes.md 中影评 → 糅合正反面"""

    def __init__(self, film_library_dir: str):
        self.base = Path(film_library_dir)

    @property
    def name(self) -> str:
        return "film_analyze"

    @property
    def description(self) -> str:
        return (
            "读取 notes.md 中某部电影的深度解析部分（用户粘贴的影评），"
            "将其糅合成一篇完整分析（正面+反面）。说「深度分析某电影」时调用。"
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {"film_name": {"type": "string", "description": "电影名称"}},
            "required": ["film_name"],
        }

    async def execute(self, film_name: str) -> ToolResult:
        notes = self.base / "notes.md"
        if not notes.exists():
            return ToolResult(success=False, error="notes.md 不存在")

        content = notes.read_text(encoding="utf-8")
        # 找到对应电影的段落
        marker = f"# {film_name}"
        if marker not in content:
            return ToolResult(success=False, error=f"notes.md 中没有「{film_name}」")

        sections = content.split("\n\n---\n\n")
        film_section = ""
        for s in sections:
            if s.strip().startswith(marker):
                film_section = s
                break

        return ToolResult(
            success=True,
            content=(
                f"📄 {film_name} 当前笔记:\n\n{film_section}\n\n"
                "---\n请将「深度解析」下的影评原文糅合成一篇完整分析"
                "（正面+反面，语言连贯），替换掉深度解析部分。"
            ),
        )


class FilmImportTool(Tool):
    """导入镜头：扫描根目录 → 继续编号 → vision 分析 → 追加 index.yaml → ChromaDB"""

    def __init__(self, film_library_dir: str, vector_memory=None, vision_tool=None):
        self.base = Path(film_library_dir)
        self.vm = vector_memory
        self.vision = vision_tool

    @property
    def name(self) -> str:
        return "film_import"

    @property
    def description(self) -> str:
        return "导入影库镜头：扫根目录新图片 → 接着最大编号命名 → vision 分析 → 追加 index.yaml → ChromaDB"

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "film_name": {"type": "string", "description": "电影名，用于 ChromaDB 标记"},
            },
            "required": ["film_name"],
        }

    async def execute(self, film_name: str) -> ToolResult:
        self.base.mkdir(parents=True, exist_ok=True)

        images = sorted(
            [f for f in self.base.iterdir()
             if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif")
             and f.name not in ("notes.md", "index.yaml")],
            key=lambda f: f.stat().st_mtime,
        )
        if not images:
            return ToolResult(success=False, error="没有新图片")

        # 读已有索引找最大编号
        idx_path = self.base / "index.yaml"
        existing = {}
        max_num = 0
        if idx_path.exists():
            try:
                existing = yaml.safe_load(idx_path.read_text(encoding="utf-8")) or {}
            except Exception:
                pass
            for fr in existing.get("frames", []):
                if isinstance(fr, dict):
                    fn = fr.get("file", "")
                    try:
                        n = int("".join(c for c in fn if c.isdigit()))
                        if n > max_num:
                            max_num = n
                    except Exception:
                        pass

        # 重命名 + 分析
        frames = existing.get("frames", [])
        analyzed = 0
        for img in images:
            max_num += 1
            ext = img.suffix.lower().replace("jpeg", "jpg")
            new_name = f"{max_num:05d}{ext}"
            new_path = self.base / new_name
            os.rename(str(img), str(new_path))

            desc = ""
            scene = ""
            tags = []
            if self.vision and ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
                # 读 notes.md 获取电影风格参考
                context_hint = ""
                notes_path = self.base / "notes.md"
                if notes_path.exists():
                    notes = notes_path.read_text(encoding="utf-8")
                    # 找该电影相关段落（前后 500 字）
                    marker = f"# {film_name}"
                    if marker in notes:
                        idx = notes.find(marker)
                        snippet = notes[max(0,idx-200):idx+800]
                        context_hint = f"参考这部电影的整体风格：{snippet[:500]}。"
                vr = await self.vision.execute(
                    image=str(new_path),
                    prompt=f"{context_hint}用中文描述这个镜头的构图、色调、光线和情绪氛围。给出3-5个标签。",
                )
                if vr.success:
                    desc = vr.content.replace("👁️ ", "").strip()
                else:
                    desc = f"分析失败: {vr.error}"
            elif ext == ".gif":
                desc = "(GIF 动图)"
            else:
                desc = f"(未分析: {ext})"

            frames.append({
                "file": new_name,
                "film": film_name,
                "scene": scene,
                "desc": desc,
                "tags": tags,
            })
            analyzed += 1

        idx_path.write_text(
            yaml.dump({"frames": frames}, allow_unicode=True, default_flow_style=False, sort_keys=False),
            encoding="utf-8",
        )

        # ChromaDB
        indexed = 0
        if self.vm:
            fc = self.vm.get_or_create_film_collection()
            exist_ids = set(fc.get()["ids"]) if fc.count() > 0 else set()
            for fr in frames[-analyzed:]:
                fname = fr["file"]
                doc_id = f"{film_name}__{fname}"
                f_hash = __import__('hashlib').md5((fr.get("desc","")[:200]).encode()).hexdigest()[:12]
                chroma_id = f"{doc_id}__{f_hash}"
                if chroma_id not in exist_ids:
                    for old in exist_ids:
                        if old.startswith(f"{film_name}__{fname}"):
                            fc.delete(ids=[old]); break
                    fc.add(
                        ids=[chroma_id],
                        documents=[f"《{film_name}》{fname}: {fr.get('desc','')}"[:1200]],
                        metadatas=[{"film": film_name, "frame": fname, "desc": fr.get("desc","")[:400]}],
                    )
                    indexed += 1

        return ToolResult(
            success=True,
            content=f"🎬 {film_name}: {len(images)} 张 → {max_num-len(images)+1:03d}~{max_num:03d}, {analyzed} 已分析, {indexed} 入 ChromaDB"
        )
