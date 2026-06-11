"""📖 文档读取 — Word / PDF / PPT / MD 全支持（扫描版 PDF 用 MiniMax VLM 识别）"""

from pathlib import Path

from src.tool_base import Tool, ToolResult


class ReadDocumentTool(Tool):
    """读取任意文档格式，提取文本内容。扫描版 PDF 自动用 Tesseract OCR。"""

    def __init__(self, workspace_dir: str, vision_tool=None):
        self.workspace = Path(workspace_dir)
        self.vision = vision_tool

    @property
    def name(self) -> str:
        return "read_document"

    @property
    def description(self) -> str:
        return (
            "读取 Word(.docx) / PDF(.pdf) / PPT(.pptx) / Markdown(.md) 文件。"
            "扫描版 PDF 自动 OCR 识别。"
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "文档路径"},
            },
            "required": ["file_path"],
        }

    async def execute(self, file_path: str) -> ToolResult:
        path = Path(file_path)
        if not path.is_absolute():
            path = self.workspace / file_path
        if not path.exists():
            return ToolResult(success=False, error=f"文件不存在: {path}")

        ext = path.suffix.lower()

        try:
            if ext == ".md" or ext == ".txt":
                text = path.read_text(encoding="utf-8")

            elif ext == ".docx":
                from docx import Document
                doc = Document(str(path))
                text = "\n".join(p.text for p in doc.paragraphs)

            elif ext == ".pdf":
                from pypdf import PdfReader
                reader = PdfReader(str(path))
                text = "\n".join(page.extract_text() or "" for page in reader.pages)
                if len(text.strip()) < 200:
                    text = await self._ocr_pdf(path)

            elif ext == ".pptx":
                from pptx import Presentation
                prs = Presentation(str(path))
                lines = []
                for slide in prs.slides:
                    for shape in slide.shapes:
                        if shape.has_text_frame:
                            for p in shape.text_frame.paragraphs:
                                if p.text.strip():
                                    lines.append(p.text)
                text = "\n".join(lines)

            else:
                return ToolResult(
                    success=False,
                    error=f"不支持的格式: {ext}。支持: md, txt, docx, pdf, pptx",
                )

            if not text.strip():
                return ToolResult(success=True, content=f"📄 {path.name}（内容为空）")

            return ToolResult(success=True, content=f"📄 {path.name}:\n\n{text}")

        except ImportError as e:
            return ToolResult(
                success=False,
                error=f"缺少库: {e}。pip install python-docx pypdf python-pptx",
            )
        except Exception as e:
            return ToolResult(success=False, error=f"读取失败: {e}")

    async def _ocr_pdf(self, pdf_path) -> str:
        """扫描版 PDF 用 MiniMax VLM 逐页识别"""
        import tempfile, os
        from pdf2image import convert_from_path

        if not self.vision:
            return "⚠️ VLM 未启用，无法识别扫描版 PDF"

        images = convert_from_path(str(pdf_path), dpi=150)
        parts = []
        for i, img in enumerate(images):
            tmp = os.path.join(tempfile.gettempdir(), f"_srx_ocr_{i}.jpg")
            img.save(tmp, "JPEG", quality=70)
            try:
                r = await self.vision.execute(
                    image=tmp,
                    prompt="请逐字识别并输出这张图片中的所有文字内容，保持原文格式和段落",
                )
                if r.success:
                    text = r.content.replace("👁️ 图片分析:\n\n", "").strip()
                    parts.append(text)
            except Exception:
                pass
            finally:
                try: os.remove(tmp)
                except: pass
        return "\n\n".join(parts)
