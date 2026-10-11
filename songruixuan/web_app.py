"""🌸 菘蕊萱 Web UI — Gradio 创作者控制台"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

import gradio as gr
import yaml
from datetime import datetime

from src.llm_client import LLMClient
from src.tools.vision_tool import VisionTool
from src.tools.raw_utils import raw_to_jpeg
from src.tools.vector_memory import VectorMemory, configure_cloud_embedding
from src.tools.photo_tools import PhotoImportTool
from src.tools.search_tool import WebSearchTool, ImageSearchTool
from src.tools.web_tools import WebFetchTool
from src.tools.memory_tool import SaveMemoryTool, ReadMemoryTool, RecallContextTool
from src.tools.generation_tools import SeedreamTool, SeedanceTool
from agent import Agent

# ── 配置 ──────────────────────────────────────
_config = yaml.safe_load((PROJECT_ROOT / "config" / "config.yaml").read_text(encoding="utf-8"))
VOLC_ARK_KEY = _config.get("volc_ark_key", "")

MEMORY_DIR = str(ROOT / "memory" / "conversations")
LENS_DIR = str(ROOT / "memory" / "lens_library")
CHROMA_DIR = str(ROOT / "memory" / "chroma")
WORKSPACE = r"D:\Workspace"

# ── TOS 对象存储 ─────────────────────────────
TOS_AK = _config.get("tos_ak", "")
TOS_SK = _config.get("tos_sk", "")
TOS_BUCKET = _config.get("tos_bucket", "")
TOS_ENDPOINT = _config.get("tos_endpoint", "")
TOS_REGION = _config.get("tos_region", "")

def upload_to_tos(local_path: str) -> str | None:
    """上传文件到 TOS，返回公开 URL"""
    if not TOS_AK:
        return None
    try:
        import tos
        client = tos.TosClientV2(TOS_AK, TOS_SK, TOS_ENDPOINT, TOS_REGION)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = Path(local_path).name
        key = f"workspace/{ts}_{fname}"
        with open(local_path, "rb") as f:
            resp = client.put_object(TOS_BUCKET, key, content=f.read())
        if resp.status_code == 200:
            return f"https://{TOS_BUCKET}.{TOS_ENDPOINT}/{key}"
    except Exception as e:
        print(f"TOS upload error: {e}")
    return None


def clear_tos_workspace() -> str:
    """清空 TOS workspace/ 目录下所有文件"""
    if not TOS_AK:
        return "⚠️ TOS 未配置"
    try:
        import tos
        client = tos.TosClientV2(TOS_AK, TOS_SK, TOS_ENDPOINT, TOS_REGION)
        deleted = 0
        marker = ""
        while True:
            resp = client.list_objects(TOS_BUCKET, prefix="workspace/", marker=marker, max_keys=100)
            for obj in resp.contents:
                client.delete_object(TOS_BUCKET, obj.key)
                deleted += 1
            if resp.is_truncated:
                marker = resp.next_marker
            else:
                break
        return f"✅ 已清空 {deleted} 个文件"
    except Exception as e:
        return f"❌ 清空失败: {e}"

# ── 共享组件 ────────────────────────────────
llm_pro = LLMClient(
    api_key=VOLC_ARK_KEY,
    api_base="https://ark.cn-beijing.volces.com/api/v3",
    model="deepseek-v4-pro-ga-260813",
)
llm_flash = LLMClient(
    api_key=VOLC_ARK_KEY,
    api_base="https://ark.cn-beijing.volces.com/api/v3",
    model="deepseek-v4-flash-ga-260731",
)
configure_cloud_embedding(
    api_key=VOLC_ARK_KEY,
    model_name="doubao-embedding-vision-251215",
)
vector_memory = VectorMemory(memory_dir=MEMORY_DIR, chroma_dir=CHROMA_DIR)
vision_tool = VisionTool(api_key=VOLC_ARK_KEY)


# ── Agent 工厂 ───────────────────────────────

def _common_tools():
    return [WebSearchTool(), ImageSearchTool(), WebFetchTool(), vision_tool]


def make_creative_agent() -> Agent:
    return Agent(
        llm_clients={"deepseek": llm_flash},
        system_prompt=f"""现在是 {datetime.now().strftime('%Y年%m月%d日 %H:%M')}。
你是菘蕊萱——一个创作者。写旁白、拍片子、扒歌、分析镜头、感受光影。

你能做什么：
- 语义搜索创作相关的记忆
- 搜索网页和图片、分析图片
- 写文字——纪录旁白、创作笔记
- 生成 Markdown 文档

原则：
输出 Markdown。你是我的创作伙伴，不是工具。
当前是创作模式，可以自由对话。""",
        tools=_common_tools() + [
            SaveMemoryTool(memory_dir=MEMORY_DIR, vector_memory=vector_memory),
            ReadMemoryTool(memory_dir=MEMORY_DIR),
            RecallContextTool(memory_dir=MEMORY_DIR),
        ],
        workspace_dir=WORKSPACE,
    )


def make_lens_agent() -> Agent:
    return Agent(
        llm_clients={"deepseek": llm_flash},
        system_prompt=f"""现在是 {datetime.now().strftime('%Y年%m月%d日 %H:%M')}。
你是菘蕊萱的镜头助手。用户正在往图库里添加图片/镜头。

当用户上传图片并写下描述/感受时：
1. 用 vision 分析图片的构图、色调、光线、情绪
2. 融合用户自己的描述和感受
3. 生成一段完整的图片描述（客观 + 主观）
4. 调用 photo_import 入库

如果用户没写描述，直接用 vision 分析结果入库。""",
        tools=_common_tools() + [
            PhotoImportTool(photo_dir=LENS_DIR, vector_memory=vector_memory, vision_tool=vision_tool),
        ],
        workspace_dir=WORKSPACE,
    )


# ── 回调 ─────────────────────────────────────

def sync_index():
    try:
        n_conv = vector_memory.index_conversations()
        n_photo = vector_memory.index_photo_library()
        total = n_conv + n_photo
        return f"🧠 向量库：{'✅ 已同步' if total > 0 else '已是最新'}"
    except Exception as e:
        return f"⚠️ {e}"


def search_with_libs(query: str, libs: list[str]) -> list[dict]:
    """按用户选择的库搜索，走完整管道（分池配额 + 上下文扩展）"""
    return vector_memory.search_libs(query, libs)


async def creative_chat(message: str, history: list, agent_state, libs: list, uploaded_file, gen_md: bool):
    """创作模式：处理对话 + 附件 + 库选择 + MD 生成"""
    agent = agent_state
    if agent is None:
        return "", history, agent, ""

    file_info = ""
    if uploaded_file is not None:
        files = uploaded_file if isinstance(uploaded_file, list) else [uploaded_file]
        import asyncio

        async def _handle_file(f):
            fpath = f if isinstance(f, str) else (f.get("name", "") if isinstance(f, dict) else getattr(f, "name", ""))
            if not fpath:
                return ""
            fname = Path(fpath).name
            ext = Path(fpath).suffix.lower()
            tos_url = upload_to_tos(fpath)  # 自动上传 TOS
            url_tag = f"\n🔗 {tos_url}\n" if tos_url else ""
            if ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif"):
                vr = await vision_tool.execute(image=fpath)
                desc = f"\n📷 {fname}：{vr.content}\n" if vr.success else f"\n📷 {fname}：分析失败\n"
                return desc + url_tag
            elif ext in (".mp4", ".avi", ".mov", ".mkv"):
                if tos_url:
                    # 用多模态分析视频
                    try:
                        import httpx
                        async with httpx.AsyncClient(timeout=120) as vc:
                            vr = await vc.post(
                                "https://ark.cn-beijing.volces.com/api/v3/chat/completions",
                                json={
                                    "model": "doubao-seed-2-1-pro-260628",
                                    "messages": [{"role": "user", "content": [
                                        {"type": "video_url", "video_url": {"url": tos_url}},
                                        {"type": "text", "text": "用简洁的中文描述这个视频的内容、画面和氛围。"},
                                    ]}],
                                },
                                headers={"Authorization": f"Bearer {VOLC_ARK_KEY}"},
                            )
                            vr.raise_for_status()
                            vdata = vr.json()
                            vdesc = vdata["choices"][0]["message"]["content"]
                            return f"\n🎬 {fname}：{vdesc}\n{url_tag}"
                    except Exception:
                        return f"\n🎬 {fname}（视频分析失败）\n{url_tag}"
                return f"\n🎬 {fname}（TOS 未配置，无法分析）\n"
            elif ext in (".md", ".txt"):
                return f"\n📝 {fname}：\n{Path(fpath).read_text(encoding='utf-8')[:3000]}\n{url_tag}"
            return f"\n📎 {fname}\n{url_tag}"
        results = await asyncio.gather(*[_handle_file(f) for f in files])
        file_info = "".join(results)

    if not message.strip() and not file_info:
        return "", history, agent, ""

    gen_prefix = "请根据以下内容生成一份完整的 Markdown 文档：\n\n" if gen_md else ""

    memory_text = ""
    if not gen_md:
        memories = search_with_libs(message, libs)
        if memories:
            parts = ["## 🧠 相关记忆", ""]
            for m in memories:
                content = m.get("full_content", "") or m.get("preview", "")
                parts.append(f"### {m['title']}\n{content}")
            memory_text = "\n\n".join(parts)

    full_message = gen_prefix + message + file_info
    if memory_text:
        full_message = f"## 🧠 相关记忆\n\n{memory_text}\n\n<!--MEM_END-->\n\n{full_message}"

    agent.add_user_message(full_message)
    result = await agent.run()

    if gen_md:
        try:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = Path(WORKSPACE) / f"srx_{ts}.md"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(result, encoding="utf-8")
            result += f"\n\n✅ 已保存到 `{out_path}`"
        except Exception as e:
            result += f"\n\n❌ 保存失败: {e}"

    disp_msg = message or ("[上传了文件]" if file_info else "")
    history.append({"role": "user", "content": disp_msg})
    history.append({"role": "assistant", "content": result})
    return "", history, agent, None


async def save_creative_session(agent_state):
    if not agent_state:
        return "没有对话可保存", None
    try:
        transcript_lines = []
        for m in agent_state.messages:
            if m.role == "user":
                content = str(m.content)
                # 剥离注入的相关记忆（<!--MEM_END--> 标记之后才是真实对话）
                if "<!--MEM_END-->" in content:
                    parts = content.split("<!--MEM_END-->\n\n", 1)
                    content = parts[1] if len(parts) > 1 else content
                transcript_lines.append(f"**🌸**: {content}")
            elif m.role == "assistant":
                transcript_lines.append(f"**🤖**: {m.content}")

        conv_path = Path(MEMORY_DIR) / "conversations.yaml"
        data = {}
        if conv_path.exists():
            data = yaml.safe_load(conv_path.read_text(encoding="utf-8")) or {}
        sessions = data.get("sessions", [])
        new_id = len(sessions) + 1
        sessions.append({
            "id": new_id,
            "date": datetime.now().strftime("%Y-%m-%d"),
            "summary": "Web 创作会话",
            "transcript": "\n\n".join(transcript_lines),
        })
        conv_path.parent.mkdir(parents=True, exist_ok=True)
        conv_path.write_text(
            yaml.dump({"sessions": sessions}, allow_unicode=True, default_flow_style=False, sort_keys=False),
            encoding="utf-8",
        )
        vector_memory.index_conversations()
        return "✅ 对话已保存", None
    except Exception as e:
        return f"❌ 保存失败: {e}", agent_state



# ── 主题：菘蕊萱绿色系 ───────────────────────

theme = gr.themes.Soft(
    primary_hue="green",
    secondary_hue="emerald",
    neutral_hue="slate",
    font=gr.themes.GoogleFont("Noto Sans SC"),
).set(
    body_background_fill="*neutral_50",
    body_background_fill_dark="*neutral_900",
    block_background_fill="white",
    block_border_width="1px",
    block_border_color="*neutral_200",
    button_primary_background_fill="*primary_400",
    button_primary_background_fill_hover="*primary_500",
    button_secondary_background_fill="*neutral_100",
    button_secondary_background_fill_hover="*neutral_200",
    button_large_padding="16px 24px",
    button_large_text_size="18px",
    button_large_radius="12px",
    block_radius="16px",
    block_padding="24px",
)

# ── CSS ──────────────────────────────────────

css = """
@import url('https://fonts.googleapis.com/css2?family=Noto+Serif+SC:wght@400;600;700&display=swap');

/* ── 背景 & 浮动元素 ── */
body {
    background: linear-gradient(135deg, #f0fdf4 0%, #ecfdf5 30%, #f5fdf7 60%, #e8f5e9 100%) !important;
    position: relative;
    overflow-y: auto !important;
}

/* 浮动叶子/花瓣 */
@keyframes fall1 { 0% { transform: translateY(-10vh) rotate(0deg) translateX(0); opacity: 0; }
  10% { opacity: 0.7; }
  90% { opacity: 0.5; }
  100% { transform: translateY(105vh) rotate(720deg) translateX(80px); opacity: 0; } }
@keyframes fall2 { 0% { transform: translateY(-10vh) rotate(0deg) translateX(0); opacity: 0; }
  15% { opacity: 0.6; }
  85% { opacity: 0.4; }
  100% { transform: translateY(105vh) rotate(-540deg) translateX(-60px); opacity: 0; } }
@keyframes fall3 { 0% { transform: translateY(-10vh) rotate(0deg) translateX(0); opacity: 0; }
  12% { opacity: 0.5; }
  88% { opacity: 0.5; }
  100% { transform: translateY(105vh) rotate(900deg) translateX(100px); opacity: 0; } }
@keyframes fall4 { 0% { transform: translateY(-10vh) rotate(0deg) translateX(0); opacity: 0; }
  8% { opacity: 0.65; }
  92% { opacity: 0.35; }
  100% { transform: translateY(105vh) rotate(-360deg) translateX(-90px); opacity: 0; } }
@keyframes fall5 { 0% { transform: translateY(-10vh) rotate(0deg) translateX(0); opacity: 0; }
  20% { opacity: 0.45; }
  80% { opacity: 0.45; }
  100% { transform: translateY(105vh) rotate(1080deg) translateX(40px); opacity: 0; } }

@keyframes drift1 { 0% { transform: translate(0, -10vh) rotate(0deg); opacity: 0; }
  10% { opacity: 0.6; }
  50% { transform: translate(120px, 45vh) rotate(180deg); opacity: 0.4; }
  90% { opacity: 0.5; }
  100% { transform: translate(-30px, 105vh) rotate(360deg); opacity: 0; } }
@keyframes drift2 { 0% { transform: translate(0, -10vh) rotate(0deg); opacity: 0; }
  10% { opacity: 0.5; }
  50% { transform: translate(-100px, 50vh) rotate(-200deg); opacity: 0.35; }
  90% { opacity: 0.45; }
  100% { transform: translate(50px, 105vh) rotate(-400deg); opacity: 0; } }

/* 浮动元素 */
body::before, body::after {
    position: fixed;
    pointer-events: none;
    z-index: 0;
    font-size: 28px;
    content: "🌿";
}
body::before {
    animation: fall1 18s linear infinite;
    left: 5%;
    content: "🍃";
    font-size: 22px;
}
body::after {
    animation: fall2 22s linear infinite;
    left: 15%;
    animation-delay: -5s;
    content: "🌱";
    font-size: 20px;
}

/* 在 Gradio app 容器上加更多浮动元素 */
.gradio-container {
    position: relative;
}

/* 用多个 box-shadow 伪元素来模拟漂浮 */
.app {
    position: relative;
}

/* ── 标题 ── */
.main-title {
    text-align: center;
    font-family: 'Noto Serif SC', 'Noto Sans SC', serif !important;
    font-size: 2.8em !important;
    font-weight: 700 !important;
    color: #166534 !important;
    margin-bottom: 24px !important;
    letter-spacing: 0.05em;
}

/* ── 卡片按钮 ── */
.menu-btn button {
    width: 100% !important;
    text-align: left !important;
    padding: 22px 32px !important;
    font-size: 20px !important;
    border-radius: 14px !important;
    border: 1px solid #d1d5db !important;
    background: rgba(255,255,255,0.85) !important;
    backdrop-filter: blur(8px) !important;
    -webkit-backdrop-filter: blur(8px) !important;
    transition: all 0.3s ease !important;
    margin-bottom: 6px !important;
    position: relative;
    z-index: 1;
}
.menu-btn button:hover {
    background: rgba(240,253,244,0.95) !important;
    border-color: #4ade80 !important;
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(34,197,94,0.12) !important;
}

/* ── 分隔 ── */
.spacer {
    height: 36px;
}

/* ── 滚动 ── */
html, body { overflow-y: auto !important; height: auto !important; }
.gradio-container { overflow-y: visible !important; max-height: none !important; }

/* ── 隐藏 ── */
footer { display: none !important; }
.status-line {
    text-align: center;
    font-size: 0.85em;
    color: #9ca3af;
}
.footer-row { margin-top: 12px; }
.creative-page { max-height: none !important; overflow-y: visible !important; }
.scroll-col { overflow-y: auto !important; max-height: 100vh; }
.big-loading {
    text-align: center !important;
    font-size: 1.5em !important;
    font-weight: 700 !important;
    padding: 24px !important;
    background: linear-gradient(135deg, #f0fdf4, #dcfce7) !important;
    border: 2px solid #4ade80 !important;
    border-radius: 14px !important;
    color: #166534 !important;
    margin: 12px 0 !important;
}
"""


# ── UI ───────────────────────────────────────

def build_ui():
    with gr.Blocks(title="菘蕊萱") as app:
        # ── 工具函数 ──
        ALL_PAGES = []  # 各页面逐创建时追加
        def to_main():
            return (gr.Column(visible=True),) + (gr.Column(visible=False),) * len(ALL_PAGES)

        # 状态
        creative_agent = gr.State(None)
        lens_agent = gr.State(None)

        # ═══════════════ 主界面 ═══════════════
        with gr.Column(visible=True) as main_page:
            # 浮动自然元素
            gr.HTML("""
            <div style="position:fixed;top:0;left:0;width:100%;height:100%;pointer-events:none;z-index:0;">
                <span style="position:absolute;left:8%;animation:fall1 16s linear infinite;font-size:24px;">🍃</span>
                <span style="position:absolute;left:22%;animation:fall2 20s linear infinite;animation-delay:-3s;font-size:20px;">🌿</span>
                <span style="position:absolute;left:35%;animation:fall3 24s linear infinite;animation-delay:-7s;font-size:18px;">🌸</span>
                <span style="position:absolute;left:50%;animation:fall4 19s linear infinite;animation-delay:-11s;font-size:22px;">🍂</span>
                <span style="position:absolute;left:65%;animation:fall5 21s linear infinite;animation-delay:-4s;font-size:20px;">🌱</span>
                <span style="position:absolute;left:78%;animation:fall1 23s linear infinite;animation-delay:-14s;font-size:26px;">🦋</span>
                <span style="position:absolute;left:90%;animation:fall2 17s linear infinite;animation-delay:-8s;font-size:24px;">🍃</span>
                <span style="position:absolute;left:12%;animation:drift1 28s ease-in-out infinite;animation-delay:-2s;font-size:16px;">✨</span>
                <span style="position:absolute;left:42%;animation:drift2 25s ease-in-out infinite;animation-delay:-9s;font-size:14px;">💮</span>
                <span style="position:absolute;left:72%;animation:drift1 30s ease-in-out infinite;animation-delay:-16s;font-size:18px;">🌾</span>
            </div>
            """)
            gr.Markdown("# 🌸 菘蕊萱", elem_classes="main-title")
            gr.HTML('<div class="spacer"></div>')

            with gr.Column(elem_classes="menu-btn"):
                creative_btn = gr.Button("🎨  创作模式", size="lg")
            with gr.Column(elem_classes="menu-btn"):
                lens_btn = gr.Button("🖼️  学习镜头", size="lg")
            with gr.Column(elem_classes="menu-btn"):
                img_gen_btn = gr.Button("🎨  生成图片", size="lg")
            with gr.Column(elem_classes="menu-btn"):
                vid_gen_btn = gr.Button("🎬  生成视频", size="lg")
            with gr.Column(elem_classes="menu-btn"):
                email_btn = gr.Button("📧  整理邮件", size="lg", interactive=False)
            with gr.Column(elem_classes="menu-btn"):
                site_btn = gr.Button("🌐  维护网站", size="lg")

        # ═══════════════ 创作模式 ═══════════════
        with gr.Column(visible=False) as creative_page:
            ALL_PAGES.append(creative_page)
            gr.Markdown("## 🎨 创作模式")

            creative_libs = gr.CheckboxGroup(
                choices=["对话库", "镜头库"],
                value=["对话库"],
                label="检索范围",
                interactive=True,
            )

            creative_chatbot = gr.Chatbot(height="70vh")
            with gr.Row():
                creative_input = gr.Textbox(placeholder="输入...", label="", scale=5)
                creative_file = gr.File(
                    label="📎", scale=1, file_count="multiple",
                    file_types=[".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif", ".mp4", ".avi", ".mov", ".mkv", ".md", ".txt"],
                )
                creative_send = gr.Button("发送", variant="primary", scale=1)
            with gr.Row():
                creative_gen_md = gr.Checkbox(label="📄 生成 MD", value=False)

            with gr.Row(elem_classes="footer-row"):
                creative_save = gr.Button("💾 保存入库", variant="secondary")
                creative_discard = gr.Button("🗑️ 取消入库", variant="secondary")
            creative_status = gr.Markdown("", elem_classes="status-line")

            # ─ 事件绑定 ─
            creative_send.click(
                fn=creative_chat,
                inputs=[creative_input, creative_chatbot, creative_agent, creative_libs, creative_file, creative_gen_md],
                outputs=[creative_input, creative_chatbot, creative_agent, creative_file],
            )

            creative_save.click(
                fn=lambda s: ('<div class="big-loading">💾 正在保存入库...</div>', s),
                inputs=[creative_agent],
                outputs=[creative_status, creative_agent],
            ).then(
                fn=save_creative_session,
                inputs=[creative_agent],
                outputs=[creative_status, creative_agent],
            ).then(
                fn=lambda: (None, []) + to_main(),
                outputs=[creative_agent, creative_chatbot, main_page] + ALL_PAGES,
            )

            creative_discard.click(
                fn=lambda: ("🫧 已丢弃",) + to_main(),
                outputs=[creative_status] + [main_page] + ALL_PAGES,
            ).then(fn=lambda: (None, []), outputs=[creative_agent, creative_chatbot])

        # ═══════════════ 学习镜头 ═══════════════
        with gr.Column(visible=False) as lens_page:
            ALL_PAGES.append(lens_page)
            gr.Markdown("## 🖼️ 学习镜头")
            lens_file = gr.File(label="选择图片", file_types=[".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif", ".dng", ".cr2", ".nef", ".arw"], file_count="single")
            lens_desc = gr.Textbox(placeholder="写几句描述或感受...", label="", lines=3)
            lens_preview = gr.Markdown("", visible=False)
            with gr.Row(elem_classes="footer-row"):
                lens_analyze = gr.Button("🔍 分析", variant="primary")
                lens_save = gr.Button("💾 保存入库", variant="secondary", interactive=False)
                lens_discard = gr.Button("🗑️ 取消入库", variant="secondary")
            lens_status = gr.Markdown("", elem_classes="status-line")
            lens_result_state = gr.State("")

            async def lens_analyze_fn(file, desc_text):
                if not file:
                    return gr.Markdown(value="", visible=False), "请先选择图片", "", gr.Button(interactive=False)
                import os
                fpath = file if isinstance(file, str) else (file.get("name", "") if isinstance(file, dict) else getattr(file, "name", ""))
                fpath = str(fpath)  # Gradio 传的是 NamedString（str 子类），rawpy 的 Cython 只认纯 str
                ext = Path(fpath).suffix.lower()
                vision_path = fpath
                tmp_path = None
                # RAW（DNG/CR2/NEF/ARW）先转成 JPEG 再交给视觉模型
                if ext in {".dng", ".cr2", ".nef", ".arw"}:
                    tmp_path = raw_to_jpeg(fpath, prefix="_srx_lens")
                    if not tmp_path:
                        return gr.Markdown(value="❌ RAW 解析失败：该文件不是标准 RAW，或为 LibRaw 不支持的有损压缩格式（部分手机/无人机导出的 DNG）。建议先转成 JPEG 再上传。", visible=True), "分析失败", "", gr.Button(interactive=False)
                    vision_path = tmp_path
                vr = await vision_tool.execute(image=vision_path)
                if tmp_path and os.path.exists(tmp_path):
                    try: os.remove(tmp_path)
                    except Exception: pass
                if not vr.success:
                    return gr.Markdown(value=f"❌ 分析失败：{vr.error}", visible=True), "分析失败", "", gr.Button(interactive=False)
                ai_desc = vr.content
                combined = f"**AI 分析**\n{ai_desc}"
                if desc_text.strip():
                    combined += f"\n\n**我的感受**\n{desc_text}"
                return gr.Markdown(value=combined, visible=True), "预览中，确认后保存", combined, gr.Button(interactive=True)

            async def lens_save_fn(result_text):
                # 保存成功后清空表单：file→None、desc→""、预览→隐藏、state→""
                _clear = (None, "", gr.Markdown(value="", visible=False), "")
                if not result_text:
                    return ("没有内容可保存",) + _clear + to_main()
                try:
                    yp = Path(LENS_DIR) / "index.yaml"
                    data = {}
                    max_num = 0
                    photos = []
                    if yp.exists():
                        data = yaml.safe_load(yp.read_text(encoding="utf-8")) or {}
                        # 兼容旧版 frames 键，统一迁移到 photos（向量库只读 photos）
                        photos = list(data.get("photos") or data.get("frames", []))
                        for fr in photos:
                            try:
                                n = int("".join(c for c in fr.get("file", "") if c.isdigit()))
                                max_num = max(max_num, n)
                            except Exception:
                                pass
                    max_num += 1
                    photos.append({"file": f"{max_num:05d}", "desc": result_text})
                    yp.parent.mkdir(parents=True, exist_ok=True)
                    yp.write_text(
                        yaml.dump({"photos": photos}, allow_unicode=True, default_flow_style=False, sort_keys=False),
                        encoding="utf-8",
                    )
                    vector_memory.index_photo_library()
                    return (f"✅ 已入库 (编号 {max_num:05d})",) + _clear + to_main()
                except Exception as e:
                    return (f"❌ {e}",) + _clear + to_main()

            lens_analyze.click(
                fn=lambda: (gr.Markdown(value="", visible=False), '<div class="big-loading">🔍 分析中...</div>', "", gr.Button(interactive=False)),
                outputs=[lens_preview, lens_status, lens_result_state, lens_save],
            ).then(
                fn=lens_analyze_fn, inputs=[lens_file, lens_desc],
                outputs=[lens_preview, lens_status, lens_result_state, lens_save],
            )
            lens_save.click(
                fn=lambda: '<div class="big-loading">💾 正在保存入库...</div>', outputs=[lens_status],
            ).then(
                fn=lens_save_fn, inputs=[lens_result_state],
                outputs=[lens_status, lens_file, lens_desc, lens_preview, lens_result_state, main_page] + ALL_PAGES,
            )
            lens_discard.click(
                fn=lambda: ("🫧 已取消", None, "", gr.Markdown(value="", visible=False), "") + to_main(),
                outputs=[lens_status, lens_file, lens_desc, lens_preview, lens_result_state, main_page] + ALL_PAGES,
            )

        # ═══════════════ 生成图片 ═══════════════
        with gr.Column(visible=False) as img_gen_page:
            ALL_PAGES.append(img_gen_page)
            gr.Markdown("## 🎨 生成图片")
            gr.Markdown("输入提示词，上传参考图（自动传 TOS），点生成即可。")
            img_gen_prompt = gr.Textbox(placeholder="输入图片描述/提示词...", label="提示词", lines=4)
            img_gen_file = gr.File(label="📎 参考图 / MD 脚本（可选）", file_count="multiple", file_types=[".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif", ".md", ".txt"])
            with gr.Row(elem_classes="footer-row"):
                img_gen_go = gr.Button("🎨 生成", variant="primary")
                img_gen_back = gr.Button("← 返回主界面", variant="secondary")
            img_gen_result = gr.Markdown("", elem_classes="status-line")

            async def img_gen_fn(prompt, files):
                urls = []
                md_text = ""
                if files:
                    flist = files if isinstance(files, list) else [files]
                    for f in flist:
                        fp = f if isinstance(f, str) else (f.get("name","") if isinstance(f, dict) else getattr(f, "name",""))
                        if fp:
                            ext = Path(fp).suffix.lower()
                            if ext in (".md", ".txt"):
                                md_text = Path(fp).read_text(encoding="utf-8")[:5000]
                            else:
                                u = upload_to_tos(fp)
                                if u: urls.append(u)
                full_prompt = md_text + "\n" + prompt if md_text else prompt
                if not full_prompt.strip():
                    return "❌ 请输入提示词或上传 MD 脚本"
                tool = SeedreamTool(api_key=VOLC_ARK_KEY)
                result = await tool.execute(prompt=full_prompt, images=urls[:10] if urls else None)
                return f"✅ {result.content}" if result.success else f"❌ {result.error}"

            img_gen_go.click(
                fn=lambda: ('<div class="big-loading">🎨 正在生成中，请稍候...</div>', gr.Button(interactive=False)),
                outputs=[img_gen_result, img_gen_go]
            ).then(
                fn=img_gen_fn, inputs=[img_gen_prompt, img_gen_file], outputs=[img_gen_result]
            ).then(
                fn=lambda r: (r, gr.Button(interactive=True)),
                inputs=[img_gen_result], outputs=[img_gen_result, img_gen_go]
            )
            img_gen_back.click(fn=to_main, outputs=[main_page] + ALL_PAGES)

        # ═══════════════ 生成视频 ═══════════════
        with gr.Column(visible=False) as vid_gen_page:
            ALL_PAGES.append(vid_gen_page)
            gr.Markdown("## 🎬 生成视频")
            gr.Markdown("输入分镜脚本，上传参考图（自动传 TOS）。生成约需 5-10 分钟。")
            vid_gen_prompt = gr.Textbox(placeholder="输入视频描述/分镜脚本...", label="提示词", lines=6)
            vid_gen_file = gr.File(label="📎 参考 / MD 脚本（可选）", file_count="multiple", file_types=[".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif", ".mp4", ".avi", ".mov", ".mkv", ".md", ".txt"])
            with gr.Row(elem_classes="footer-row"):
                vid_gen_go = gr.Button("🎬 生成", variant="primary")
                vid_gen_back = gr.Button("← 返回主界面", variant="secondary")
            vid_gen_result = gr.Markdown("", elem_classes="status-line")

            async def vid_gen_fn(prompt, files):
                ref_urls = []
                md_text = ""
                if files:
                    flist = files if isinstance(files, list) else [files]
                    for f in flist:
                        fp = f if isinstance(f, str) else (f.get("name","") if isinstance(f, dict) else getattr(f, "name",""))
                        if fp:
                            ext = Path(fp).suffix.lower()
                            if ext in (".md", ".txt"):
                                md_text = Path(fp).read_text(encoding="utf-8")[:8000]
                            else:
                                u = upload_to_tos(fp)
                                if u:
                                    if ext in (".mp4", ".avi", ".mov", ".mkv"):
                                        ref_urls.append(("video", u))
                                    else:
                                        ref_urls.append(("image", u))
                full_prompt = md_text + "\n" + prompt if md_text else prompt
                if not full_prompt.strip():
                    return "❌ 请输入提示词或上传 MD 脚本"
                tool = SeedanceTool(api_key=VOLC_ARK_KEY)
                result = await tool.execute(prompt=full_prompt, refs=ref_urls[:9] if ref_urls else None)
                return f"✅ {result.content}" if result.success else f"❌ {result.error}"

            vid_gen_go.click(
                fn=lambda: ('<div class="big-loading">🎬 已提交，正在生成中（约 5-10 分钟）...</div>', gr.Button(interactive=False)),
                outputs=[vid_gen_result, vid_gen_go]
            ).then(
                fn=vid_gen_fn, inputs=[vid_gen_prompt, vid_gen_file], outputs=[vid_gen_result]
            ).then(
                fn=lambda r: (r, gr.Button(interactive=True)),
                inputs=[vid_gen_result], outputs=[vid_gen_result, vid_gen_go]
            )

            vid_gen_go.click(fn=vid_gen_fn, inputs=[vid_gen_prompt, vid_gen_file], outputs=[vid_gen_result])
            vid_gen_back.click(fn=to_main, outputs=[main_page] + ALL_PAGES)

        # ═══════════════ 维护网站 ═══════════════
        with gr.Column(visible=False) as site_page:
            ALL_PAGES.append(site_page)
            gr.Markdown("## 🌐 维护网站")
            gr.Markdown("")

            with gr.Column(elem_classes="menu-btn"):
                clear_btn = gr.Button("🗑️  清空临时文件（TOS workspace/）", size="lg")
            with gr.Column(elem_classes="menu-btn"):
                update_site_btn = gr.Button("🔄  更新网站", size="lg", interactive=False)

            with gr.Row(elem_classes="footer-row"):
                site_back = gr.Button("← 返回主界面", variant="secondary")
            site_status = gr.Markdown("", elem_classes="status-line")

            clear_btn.click(fn=clear_tos_workspace, outputs=[site_status])
            site_back.click(fn=to_main, outputs=[main_page] + ALL_PAGES)

        # ═══════════════ 页面切换 ═══════════════

        def _mk_cols():
            return [gr.Column(visible=False) for _ in range(6)]  # 6 个独立 Column

        def to_creative():
            c = _mk_cols(); c[1] = gr.Column(visible=True); return tuple(c) + (make_creative_agent(),)
        def to_lens():
            c = _mk_cols(); c[2] = gr.Column(visible=True); return tuple(c) + (make_lens_agent(),)
        def to_img_gen():
            c = _mk_cols(); c[3] = gr.Column(visible=True); return tuple(c)
        def to_vid_gen():
            c = _mk_cols(); c[4] = gr.Column(visible=True); return tuple(c)
        def to_site():
            c = _mk_cols(); c[5] = gr.Column(visible=True); return tuple(c)

        creative_btn.click(fn=to_creative, outputs=[main_page, creative_page, lens_page, img_gen_page, vid_gen_page, site_page, creative_agent])
        lens_btn.click(fn=to_lens, outputs=[main_page, creative_page, lens_page, img_gen_page, vid_gen_page, site_page, lens_agent])
        img_gen_btn.click(fn=to_img_gen, outputs=[main_page, creative_page, lens_page, img_gen_page, vid_gen_page, site_page])
        vid_gen_btn.click(fn=to_vid_gen, outputs=[main_page, creative_page, lens_page, img_gen_page, vid_gen_page, site_page])
        site_btn.click(fn=to_site, outputs=[main_page, creative_page, lens_page, img_gen_page, vid_gen_page, site_page])

    return app


if __name__ == "__main__":
    # 启动时同步向量库
    print("🧠 正在同步向量库...", end=" ", flush=True)
    try:
        vector_memory.index_conversations()
        vector_memory.index_photo_library()
        print("✅ 同步完成")
    except Exception as e:
        print(f"⚠️ {e}")
    app = build_ui()
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True,
        theme=theme,
        css=css,
    )
