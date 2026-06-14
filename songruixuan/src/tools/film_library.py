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
    """在 notes.yaml 追加新电影的内容大纲"""

    def __init__(self, film_library_dir: str):
        self.base = Path(film_library_dir)

    @property
    def name(self) -> str:
        return "film_create"

    @property
    def description(self) -> str:
        return (
            "在影库 notes.yaml 中追加一部新电影的笔记。说「学习某电影」时调用。"
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {"name": {"type": "string", "description": "电影名称"}},
            "required": ["name"],
        }

    async def execute(self, name: str) -> ToolResult:
        yaml_path = self.base / "notes.yaml"
        self.base.mkdir(parents=True, exist_ok=True)

        # 读已有数据
        data = {}
        if yaml_path.exists():
            try:
                data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
            except Exception:
                pass

        films = data.get("films", [])

        # 检查是否已有
        for f in films:
            if f.get("name") == name:
                return ToolResult(success=False, error=f"「{name}」已在 notes.yaml 中")

        films.append({
            "name": name,
            "content": "## 内容大纲\n\n\n## 深度解析\n\n\n",
        })

        yaml_path.write_text(
            yaml.dump({"films": films}, allow_unicode=True, default_flow_style=False, sort_keys=False),
            encoding="utf-8",
        )

        return ToolResult(
            success=True,
            content=f"🎬 「{name}」已追加到 notes.yaml\n请用 web_search 搜索该电影，填写内容大纲。"
        )


class FilmAnalyzeTool(Tool):
    """读 notes.yaml 中影评 → 糅合正反面"""

    def __init__(self, film_library_dir: str):
        self.base = Path(film_library_dir)

    @property
    def name(self) -> str:
        return "film_analyze"

    @property
    def description(self) -> str:
        return (
            "读取 notes.yaml 中某部电影的笔记（含深度解析）。"
            "用于检索影库中的电影分析、参考已有笔记。说「深度分析某电影」「参考影库」「看某电影的笔记」时调用。"
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {"film_name": {"type": "string", "description": "电影名称"}},
            "required": ["film_name"],
        }

    async def execute(self, film_name: str) -> ToolResult:
        yaml_path = self.base / "notes.yaml"
        if not yaml_path.exists():
            return ToolResult(success=False, error="notes.yaml 不存在")

        try:
            data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
        except Exception:
            return ToolResult(success=False, error="notes.yaml 读取失败")

        films = data.get("films", [])
        film_content = ""
        for f in films:
            if f.get("name") == film_name:
                film_content = f.get("content", "")
                break

        if not film_content:
            return ToolResult(success=False, error=f"notes.yaml 中没有「{film_name}」")

        return ToolResult(
            success=True,
            content=(
                f"📄 《{film_name}》当前笔记:\n\n{film_content}\n\n"
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
             and f.name not in ("notes.yaml", "notes.md", "index.yaml")],
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
                # ── 从 notes.yaml 提取电影的视觉风格上下文 ──
                context_hint = ""
                notes_path = self.base / "notes.yaml"
                if notes_path.exists():
                    try:
                        data = yaml.safe_load(notes_path.read_text(encoding="utf-8")) or {}
                        film_section = ""
                        for f in data.get("films", []):
                            if f.get("name") == film_name:
                                film_section = f.get("content", "")
                                break

                        context_parts = []

                        # 辅助函数：从文本中提取以 ### 开头的子章节
                        def _extract_subsection(text: str, *keywords: str) -> str:
                            """提取 ### 标题匹配任一关键词的子章节，到下一个同级/上级标题为止"""
                            lines = text.split('\n')
                            for i, line in enumerate(lines):
                                stripped = line.strip()
                                if stripped.startswith('###') and any(kw in stripped for kw in keywords):
                                    # 找到起止：从标题行到下一个 ### / ## / #
                                    body_start = i
                                    body_end = len(lines)
                                    for j in range(i + 1, len(lines)):
                                        if lines[j].strip().startswith('###') or lines[j].strip().startswith('##') or lines[j].strip().startswith('# '):
                                            body_end = j
                                            break
                                    return '\n'.join(lines[body_start:body_end]).strip()
                            return ""

                        # 1. 视觉风格
                        vis = _extract_subsection(film_section, "视觉风格", "四、视觉")
                        if vis:
                            context_parts.append(vis)

                        # 2. 关键场景索引
                        scenes_idx = _extract_subsection(film_section, "关键场景索引", "八、关键场景")
                        if scenes_idx:
                            context_parts.append(scenes_idx)

                        # 3. 色彩系统 / 光线相关章节（如果有独立子章节的话）
                        color = _extract_subsection(film_section, "色彩系统", "光线")
                        if color and color != vis:
                            context_parts.append(color)

                        if context_parts:
                            context_hint = (
                                f"以下是电影《{film_name}》的视觉风格与关键场景参考，"
                                f"请用其中的视觉词汇来描述这张截图：\n\n"
                                + "\n\n".join(context_parts)
                                + "\n\n---\n"
                            )
                    except Exception:
                        pass

                prompt = (
                    f"{context_hint}"
                    f"分三步描述这个镜头：\n"
                    f"第一步——画面内容：描述画面上具体有什么。人物是谁、在做什么动作、穿什么、什么表情。"
                    f"画面里有哪些物件、在什么位置。空间是怎样的（室内/室外、什么房间/场景）。"
                    f"第二步——视觉分析：用上面电影笔记里的视觉词汇来分析这个镜头的构图、光线、色调。"
                    f"如果画面里有笔记中提到的视觉标志（例如特定的布光方式、色彩、道具），一定要指出来。"
                    f"第三步——场景判断：这个帧可能对应笔记里的哪个关键场景？"
                    f"\n要求：平实、准确、有细节。禁止使用「这张图片展示了」「一个经典的电影场景」「充满了...的氛围」等万能套话。"
                    f"禁止给描述加小标题分段（如### 构图）。用一个自然段落写完。"
                    f"\n最后单独一行给出3-5个标签，格式：标签：标签1, 标签2, 标签3"
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

        # ChromaDB: 增量索引新帧（2000/500 滑动窗口分块），不动已有条目
        indexed = 0
        if self.vm:
            fc = self.vm.get_or_create_film_collection()
            exist_ids = set(fc.get()["ids"]) if fc.count() > 0 else set()
            cs, ol = 2000, 500
            for fr in frames[-analyzed:]:
                fname = fr["file"]
                desc = fr.get("desc", "")
                if not desc:
                    continue
                doc_id = f"{film_name}__{fname}"
                full_text = f"《{film_name}》{fname}: {desc}"

                # 清除该帧旧记录
                for old_id in list(exist_ids):
                    if old_id.startswith(doc_id):
                        fc.delete(ids=[old_id])
                        exist_ids.discard(old_id)

                if len(full_text) <= cs:
                    f_hash = __import__('hashlib').md5(desc[:200].encode()).hexdigest()[:12]
                    chroma_id = f"{doc_id}__{f_hash}"
                    fc.add(
                        ids=[chroma_id],
                        documents=[full_text[:1200]],
                        metadatas=[{"film": film_name, "frame": fname, "desc": desc[:800], "source": str(idx_path)}],
                    )
                    indexed += 1
                else:
                    for i, p in enumerate(range(0, len(full_text), cs - ol)):
                        chunk = full_text[p:p + cs]
                        if not chunk.strip():
                            continue
                        c_hash = __import__('hashlib').md5(chunk[:200].encode()).hexdigest()[:8]
                        chroma_id = f"{doc_id}_chunk_{i}__{c_hash}"
                        fc.add(
                            ids=[chroma_id],
                            documents=[chunk[:1200]],
                            metadatas=[{"film": film_name, "frame": fname, "desc": chunk[:800], "source": str(idx_path), "chunk_index": i}],
                        )
                        indexed += 1

        return ToolResult(
            success=True,
            content=f"🎬 {film_name}: {len(images)} 张 → {max_num-len(images)+1:03d}~{max_num:03d}, {analyzed} 已分析, {indexed} 条已入 ChromaDB（含分块）"
        )
