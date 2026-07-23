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
from src.tools.vector_memory import VectorMemory, configure_cloud_embedding
from src.tools.film_library import FilmCreateTool, FilmAnalyzeTool
from src.tools.music_tools import MusicCreateTool, MusicAnalyzeTool
from src.tools.photo_tools import PhotoImportTool
from src.tools.search_tool import WebSearchTool, ImageSearchTool
from src.tools.web_tools import WebFetchTool
from src.tools.memory_tool import SaveMemoryTool, ReadMemoryTool, RecallContextTool
from src.tools.generation_tools import SeedreamTool, SeedanceTool
from agent import Agent

# ── 配置 ──────────────────────────────────────
_config = yaml.safe_load((PROJECT_ROOT / "config" / "config.yaml").read_text(encoding="utf-8"))
DEEPSEEK_KEY = _config["deepseek_key"]
VOLC_ARK_KEY = _config.get("volc_ark_key", "")

MEMORY_DIR = str(ROOT / "memory" / "conversations")
FILM_DIR = str(ROOT / "memory" / "film_library_daily")
MUSIC_DIR = str(ROOT / "memory" / "music_library_daily")
LENS_DIR = str(ROOT / "memory" / "lens_library")
CHROMA_DIR = str(ROOT / "memory" / "chroma")
WORKSPACE = r"D:\Workspace"

# ── 共享组件 ────────────────────────────────
llm_client = LLMClient(
    api_key=DEEPSEEK_KEY,
    api_base="https://api.deepseek.com/v1",
    model="deepseek-v4-pro",
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
        llm_clients={"deepseek": llm_client},
        system_prompt=f"""现在是 {datetime.now().strftime('%Y年%m月%d日 %H:%M')}。
你是菘蕊萱——一个创作者。写旁白、拍片子、扒歌、分析镜头、感受光影。

你能做什么：
- 语义搜索创作相关的记忆
- 搜索网页和图片、分析图片
- 生成图片（调用 generate_image）
- 生成视频（调用 generate_video）
- 写文字——纪录旁白、创作笔记

原则：
输出 Markdown。你是我的创作伙伴，不是工具。
当前是创作模式，可以自由对话、生成图片/视频。
如果用户要求生成图片，请调用 generate_image 工具。
如果用户要求生成视频，请调用 generate_video 工具。""",
        tools=_common_tools() + [
            SaveMemoryTool(memory_dir=MEMORY_DIR, vector_memory=vector_memory),
            ReadMemoryTool(memory_dir=MEMORY_DIR),
            RecallContextTool(memory_dir=MEMORY_DIR),
            SeedreamTool(api_key=VOLC_ARK_KEY),
            SeedanceTool(api_key=VOLC_ARK_KEY),
        ],
        workspace_dir=WORKSPACE,
    )


def make_film_agent() -> Agent:
    return Agent(
        llm_clients={"deepseek": llm_client},
        system_prompt=f"""现在是 {datetime.now().strftime('%Y年%m月%d日 %H:%M')}。
你是菘蕊萱的电影学习助手。用户点击「学习电影」进入此模式。

### 你的工作流程：

**第一步：搜集信息**
用户告诉你电影名后，立刻用 web_search 搜索，生成结构化的「内容大纲」，包含：
- 基本信息（片名、上映年份、导演、主演、片长）
- 故事梗概
- 叙事结构
- 视觉风格
- 声音设计
- 核心主题
- 值得学习的创作手法

**第三步：和用户讨论**
把大纲展示给用户，问用户有什么补充或个人感受。在对话中逐步收集用户的观影笔记。

**第四步：等待整合**
当用户按「确认结束，整理笔记」时，把以上所有内容整合为最终格式输出：
## 内容大纲
（客观信息，你整理的）
## 深度解析
（结合用户的主观笔记和你的分析，糅合成连贯的深度分析）

禁止擅自保存文件。""",
        tools=_common_tools() + [
            FilmCreateTool(film_library_dir=FILM_DIR),
            FilmAnalyzeTool(film_library_dir=FILM_DIR),
        ],
        workspace_dir=WORKSPACE,
    )


def make_music_agent() -> Agent:
    return Agent(
        llm_clients={"deepseek": llm_client},
        system_prompt=f"""现在是 {datetime.now().strftime('%Y年%m月%d日 %H:%M')}。
你是菘蕊萱的音乐学习助手。用户点击「学习音乐」进入此模式。

### 你的工作流程：

**第一步：搜集信息**
用户告诉你曲目名后，立刻用 web_search 搜索，生成结构化的「曲目信息」，包含：
- 基本信息（歌名、艺人、发行年份、专辑）
- 风格流派
- 乐器编排
- 和弦进行 / 编曲分析
- 歌词主题
- 制作背景

**第三步：和用户讨论**
把曲目信息展示给用户，问用户的听歌感受。在对话中讨论乐器、和弦、编曲、风格等，逐步收集用户的个人听歌笔记。

**第四步：等待整合**
当用户按「确认结束，整理笔记」时，把以上所有内容整合为最终格式输出：
## 曲目信息
（客观信息，你整理的）
## 听歌笔记
（结合用户的主观感受和你的分析，糅合成连贯的深度笔记）

禁止擅自保存文件。""",
        tools=_common_tools() + [
            MusicCreateTool(music_dir=MUSIC_DIR),
            MusicAnalyzeTool(music_dir=MUSIC_DIR),
        ],
        workspace_dir=WORKSPACE,
    )


def make_lens_agent() -> Agent:
    return Agent(
        llm_clients={"deepseek": llm_client},
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
        n_notes = vector_memory.index_notes()
        n_music_notes = vector_memory.index_music_notes()
        n_photo = vector_memory.index_photo_library()
        total = n_conv + n_notes + n_music_notes + n_photo
        return f"🧠 向量库：{'✅ 已同步' if total > 0 else '已是最新'}"
    except Exception as e:
        return f"⚠️ {e}"


def search_with_libs(query: str, libs: list[str]) -> list[dict]:
    """按用户选择的库搜索，不依赖关键词路由"""
    results = []
    if not libs:
        results = vector_memory.search(query, 20)
    else:
        for lib in libs:
            if lib == "对话库":
                results.extend(vector_memory.search(query, 10))
            elif lib == "电影库":
                results.extend(vector_memory.search_notes(query, 5))
            elif lib == "音乐库":
                results.extend(vector_memory.search_music_notes(query, 5))
            elif lib == "镜头库":
                results.extend(vector_memory.search_photos(query, 5))
    # 去重
    seen = set()
    deduped = []
    for r in results:
        key = (r.get("source", ""), r.get("title", ""))
        if key not in seen:
            seen.add(key)
            deduped.append(r)
    return deduped[:20]


async def creative_chat(message: str, history: list, agent_state, libs: list, uploaded_file, gen_opts: list):
    """创作模式：处理对话 + 附件 + 库选择 + 生成选项"""
    agent = agent_state
    if agent is None:
        return "", history, agent, ""

    file_info = ""
    fname = ""
    if uploaded_file is not None:
        fpath = uploaded_file if isinstance(uploaded_file, str) else uploaded_file.name
        fname = Path(fpath).name
        ext = Path(fpath).suffix.lower()
        if ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif"):
            vr = await vision_tool.execute(image=fpath)
            file_info = f"\n📷 图片分析：{vr.content}\n" if vr.success else ""
        elif ext == ".pdf":
            file_info = "\n📄 PDF 已上传，暂不支持解析\n"
        elif ext in (".mp4", ".avi", ".mov", ".mkv"):
            file_info = "\n🎬 视频已上传，暂不支持解析\n"
        elif ext == ".md":
            file_info = f"\n📝 MD 文档：\n{Path(fpath).read_text(encoding='utf-8')[:3000]}\n"
        else:
            file_info = f"\n📎 文件已上传: {fname}\n"

    if not message.strip() and not file_info:
        return "", history, agent, ""

    # 生成选项前缀
    gen_prefix = ""
    if "🎨 生成图片" in gen_opts:
        gen_prefix = "请调用 generate_image 工具，根据以下描述生成图片：\n\n"
    elif "🎬 生成视频" in gen_opts:
        gen_prefix = "请调用 generate_video 工具，根据以下描述生成视频：\n\n"
    elif "📄 生成 MD" in gen_opts:
        gen_prefix = "请根据以下内容生成一份完整的 Markdown 文档：\n\n"

    # 注入记忆
    memories = search_with_libs(message, libs)
    memory_text = ""
    if memories and gen_prefix == "":  # 生成模式不注入记忆避免干扰
        parts = ["## 🧠 相关记忆", ""]
        for m in memories:
            content = m.get("full_content", "") or m.get("preview", "")
            parts.append(f"### {m['title']}\n{content}")
        memory_text = "\n\n".join(parts)

    full_message = gen_prefix + message + file_info
    if memory_text:
        full_message = f"{memory_text}\n\n---\n\n{full_message}"

    agent.add_user_message(full_message)
    result = await agent.run()

    # MD 生成模式：保存到 Workspace
    if "📄 生成 MD" in gen_opts:
        try:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = Path(WORKSPACE) / f"srx_{ts}.md"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(result, encoding="utf-8")
            result += f"\n\n✅ 已保存到 `{out_path}`"
        except Exception as e:
            result += f"\n\n❌ 保存失败: {e}"

    disp_msg = message or ""
    if file_info:
        disp_msg = f"[上传: {fname}]\n{disp_msg}" if fname else disp_msg
    if gen_prefix:
        disp_msg = f"[生成模式]\n{disp_msg}"

    history.append({"role": "user", "content": disp_msg})
    history.append({"role": "assistant", "content": result})
    return "", history, agent, None


async def simple_chat(message: str, history: list, agent_state):
    """电影/音乐等模式用的简单对话，不走库选择和附件"""
    result = await creative_chat(message, history, agent_state, [], None, [])
    return result[:3]  # msg, history, agent


async def save_creative_session(agent_state):
    if not agent_state:
        return "没有对话可保存", None
    try:
        transcript_lines = []
        for m in agent_state.messages:
            if m.role == "user":
                transcript_lines.append(f"**🌸**: {m.content}")
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
"""


# ── UI ───────────────────────────────────────

def build_ui():
    with gr.Blocks(title="菘蕊萱", theme=theme, css=css) as app:
        # 状态
        creative_agent = gr.State(None)
        film_agent = gr.State(None)
        music_agent = gr.State(None)
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
                film_btn = gr.Button("🎬  学习电影", size="lg")
            with gr.Column(elem_classes="menu-btn"):
                music_btn = gr.Button("🎵  学习音乐", size="lg")
            with gr.Column(elem_classes="menu-btn"):
                lens_btn = gr.Button("🖼️  学习镜头", size="lg")
            with gr.Column(elem_classes="menu-btn"):
                email_btn = gr.Button("📧  整理邮件", size="lg", interactive=False)
            with gr.Column(elem_classes="menu-btn"):
                site_btn = gr.Button("🌐  维护网站", size="lg", interactive=False)

        # ═══════════════ 创作模式 ═══════════════
        with gr.Column(visible=False) as creative_page:
            gr.Markdown("## 🎨 创作模式")

            creative_libs = gr.CheckboxGroup(
                choices=["对话库", "电影库", "音乐库", "镜头库"],
                value=["对话库"],
                label="检索范围",
                interactive=True,
            )

            creative_chatbot = gr.Chatbot(height="70vh")
            with gr.Row():
                creative_input = gr.Textbox(placeholder="输入...", label="", scale=5)
                creative_file = gr.File(
                    label="📎", scale=1, file_count="single",
                    file_types=[".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif", ".pdf", ".mp4", ".avi", ".mov", ".mkv", ".md"],
                )
                creative_send = gr.Button("发送", variant="primary", scale=1)
            with gr.Row():
                creative_gen_opts = gr.CheckboxGroup(
                    choices=["🎨 生成图片", "🎬 生成视频", "📄 生成 MD"],
                    value=[],
                    label="输出模式",
                    interactive=True,
                )

            with gr.Row(elem_classes="footer-row"):
                creative_save = gr.Button("💾 保存入库", variant="secondary")
                creative_discard = gr.Button("🗑️ 取消入库", variant="secondary")
            creative_status = gr.Markdown("", elem_classes="status-line")

            # ─ 事件绑定 ─
            creative_send.click(
                fn=creative_chat,
                inputs=[creative_input, creative_chatbot, creative_agent, creative_libs, creative_file, creative_gen_opts],
                outputs=[creative_input, creative_chatbot, creative_agent, creative_file],
            )

            creative_save.click(
                fn=save_creative_session,
                inputs=[creative_agent],
                outputs=[creative_status, creative_agent],
            ).then(
                fn=lambda: (None, [], gr.Column(visible=True), gr.Column(visible=False)),
                outputs=[creative_agent, creative_chatbot, main_page, creative_page],
            )

            creative_discard.click(
                fn=lambda: ("🫧 已丢弃", None, [], gr.Column(visible=True), gr.Column(visible=False)),
                outputs=[creative_status, creative_agent, creative_chatbot, main_page, creative_page],
            )

        # ═══════════════ 学习电影 ═══════════════
        with gr.Column(visible=False) as film_page:
            gr.Markdown("## 🎬 学习电影")
            film_chatbot = gr.Chatbot(height=450)
            with gr.Row():
                film_input = gr.Textbox(placeholder="聊聊这部电影...", label="", scale=5)
                film_send = gr.Button("发送", variant="primary", scale=1)
            with gr.Row():
                film_finish = gr.Button("✅ 确认结束，整理笔记")
                film_preview = gr.Markdown("", visible=False)
            with gr.Row(elem_classes="footer-row"):
                film_save = gr.Button("💾 保存入库", variant="secondary", interactive=False)
                film_discard = gr.Button("🗑️ 取消入库", variant="secondary")
            film_status = gr.Markdown("", elem_classes="status-line")

            async def film_finish_fn(agent_state, history):
                if not agent_state:
                    return "", "", gr.Button(interactive=False), gr.Button(interactive=True), agent_state
                agent_state.add_user_message("请将以上所有讨论整合成电影笔记，格式：\n## 内容大纲\n（客观信息）\n## 深度解析\n（主观感受+分析）")
                result = await agent_state.run()
                history.append({"role": "user", "content": "[确认结束，整理笔记]"})
                history.append({"role": "assistant", "content": result})
                return result, "", gr.Button(interactive=True), gr.Button(interactive=False), agent_state

            async def film_save_fn(agent_state):
                if not agent_state:
                    return "没有笔记可保存", gr.Column(visible=True), gr.Column(visible=False)
                try:
                    # 收集笔记内容
                    film_name = ""
                    content = ""
                    for m in agent_state.messages:
                        if m.role == "assistant" and "## 内容大纲" in str(m.content):
                            content = str(m.content)
                    if not content:
                        return "未找到整理好的笔记，请先点「确认结束」", gr.Column(visible=True), gr.Column(visible=False)
                    # 存到 YAML
                    yp = Path(FILM_DIR) / "notes.yaml"
                    data = {}
                    if yp.exists():
                        data = yaml.safe_load(yp.read_text(encoding="utf-8")) or {}
                    films = data.get("films", [])
                    # 尝试从对话中提取电影名
                    for m in agent_state.messages:
                        if m.role == "user" and "电影" in str(m.content):
                            film_name = str(m.content)[:50]
                            break
                    if not film_name:
                        film_name = f"电影笔记 {datetime.now().strftime('%m%d')}"
                    films.append({"name": film_name, "content": content})
                    yp.parent.mkdir(parents=True, exist_ok=True)
                    yp.write_text(
                        yaml.dump({"films": films}, allow_unicode=True, default_flow_style=False, sort_keys=False),
                        encoding="utf-8",
                    )
                    vector_memory.index_notes()
                    return "✅ 已保存到电影库", gr.Column(visible=True), gr.Column(visible=False)
                except Exception as e:
                    return f"❌ {e}", gr.Column(visible=True), gr.Column(visible=False)

            film_send.click(
                fn=simple_chat, inputs=[film_input, film_chatbot, film_agent],
                outputs=[film_input, film_chatbot, film_agent],
            ).then(lambda: "", None, film_input)

            film_finish.click(
                fn=film_finish_fn, inputs=[film_agent, film_chatbot],
                outputs=[film_preview, film_status, film_save, film_discard, film_agent],
            )
            film_save.click(
                fn=film_save_fn, inputs=[film_agent],
                outputs=[film_status, main_page, film_page],
            )
            film_discard.click(
                fn=lambda: ("🫧 已丢弃", None, [], gr.Column(visible=True), gr.Column(visible=False)),
                outputs=[film_status, film_agent, film_chatbot, main_page, film_page],
            )

        # ═══════════════ 学习音乐 ═══════════════
        with gr.Column(visible=False) as music_page:
            gr.Markdown("## 🎵 学习音乐")
            music_chatbot = gr.Chatbot(height=450)
            with gr.Row():
                music_input = gr.Textbox(placeholder="聊聊这首歌...", label="", scale=5)
                music_send = gr.Button("发送", variant="primary", scale=1)
            with gr.Row():
                music_finish = gr.Button("✅ 确认结束，整理笔记")
                music_preview = gr.Markdown("", visible=False)
            with gr.Row(elem_classes="footer-row"):
                music_save = gr.Button("💾 保存入库", variant="secondary", interactive=False)
                music_discard = gr.Button("🗑️ 取消入库", variant="secondary")
            music_status = gr.Markdown("", elem_classes="status-line")

            async def music_finish_fn(agent_state, history):
                if not agent_state:
                    return "", "", gr.Button(interactive=False), gr.Button(interactive=True), agent_state
                agent_state.add_user_message("请将以上所有讨论整合成音乐笔记，格式：\n## 曲目信息\n（客观信息）\n## 听歌笔记\n（主观感受+分析）")
                result = await agent_state.run()
                history.append({"role": "user", "content": "[确认结束，整理笔记]"})
                history.append({"role": "assistant", "content": result})
                return result, "", gr.Button(interactive=True), gr.Button(interactive=False), agent_state

            async def music_save_fn(agent_state):
                if not agent_state:
                    return "没有笔记可保存", gr.Column(visible=True), gr.Column(visible=False)
                try:
                    content = ""
                    for m in agent_state.messages:
                        if m.role == "assistant" and "## 曲目信息" in str(m.content):
                            content = str(m.content)
                    if not content:
                        return "未找到整理好的笔记", gr.Column(visible=True), gr.Column(visible=False)
                    yp = Path(MUSIC_DIR) / "notes.yaml"
                    data = {}
                    if yp.exists():
                        data = yaml.safe_load(yp.read_text(encoding="utf-8")) or {}
                    tracks = data.get("tracks", [])
                    track_name = f"音乐笔记 {datetime.now().strftime('%m%d')}"
                    for m in agent_state.messages:
                        if m.role == "user" and ("歌" in str(m.content) or "专辑" in str(m.content)):
                            track_name = str(m.content)[:50]
                            break
                    tracks.append({"name": track_name, "content": content})
                    yp.parent.mkdir(parents=True, exist_ok=True)
                    yp.write_text(
                        yaml.dump({"tracks": tracks}, allow_unicode=True, default_flow_style=False, sort_keys=False),
                        encoding="utf-8",
                    )
                    vector_memory.index_music_notes()
                    return "✅ 已保存到音乐库", gr.Column(visible=True), gr.Column(visible=False)
                except Exception as e:
                    return f"❌ {e}", gr.Column(visible=True), gr.Column(visible=False)

            music_send.click(
                fn=simple_chat, inputs=[music_input, music_chatbot, music_agent],
                outputs=[music_input, music_chatbot, music_agent],
            ).then(lambda: "", None, music_input)

            music_finish.click(
                fn=music_finish_fn, inputs=[music_agent, music_chatbot],
                outputs=[music_preview, music_status, music_save, music_discard, music_agent],
            )
            music_save.click(
                fn=music_save_fn, inputs=[music_agent],
                outputs=[music_status, main_page, music_page],
            )
            music_discard.click(
                fn=lambda: ("🫧 已丢弃", None, [], gr.Column(visible=True), gr.Column(visible=False)),
                outputs=[music_status, music_agent, music_chatbot, main_page, music_page],
            )

        # ═══════════════ 学习镜头 ═══════════════
        with gr.Column(visible=False) as lens_page:
            gr.Markdown("## 🖼️ 学习镜头")
            lens_file = gr.File(label="选择图片", file_types=["image"], file_count="single")
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
                    return "", "请先选择图片", "", gr.Button(interactive=False)
                vr = await vision_tool.execute(image=file)
                if not vr.success:
                    return "", f"分析失败: {vr.error}", "", gr.Button(interactive=False)
                ai_desc = vr.content
                combined = f"**AI 分析**\n{ai_desc}"
                if desc_text.strip():
                    combined += f"\n\n**我的感受**\n{desc_text}"
                return combined, "预览中，确认后保存", combined, gr.Button(interactive=True)

            async def lens_save_fn(result_text):
                if not result_text:
                    return "没有内容可保存", gr.Column(visible=True), gr.Column(visible=False)
                try:
                    yp = Path(LENS_DIR) / "index.yaml"
                    data = {}
                    max_num = 0
                    if yp.exists():
                        data = yaml.safe_load(yp.read_text(encoding="utf-8")) or {}
                        for fr in data.get("frames", data.get("photos", [])):
                            try:
                                n = int("".join(c for c in fr.get("file", "") if c.isdigit()))
                                max_num = max(max_num, n)
                            except Exception:
                                pass
                    frames = data.get("frames", data.get("photos", []))
                    max_num += 1
                    frames.append({"file": f"{max_num:05d}", "desc": result_text})
                    yp.parent.mkdir(parents=True, exist_ok=True)
                    yp.write_text(
                        yaml.dump({"frames": frames}, allow_unicode=True, default_flow_style=False, sort_keys=False),
                        encoding="utf-8",
                    )
                    vector_memory.index_photo_library()
                    return f"✅ 已入库 (编号 {max_num:05d})", gr.Column(visible=True), gr.Column(visible=False)
                except Exception as e:
                    return f"❌ {e}", gr.Column(visible=True), gr.Column(visible=False)

            lens_analyze.click(
                fn=lens_analyze_fn, inputs=[lens_file, lens_desc],
                outputs=[lens_preview, lens_status, lens_result_state, lens_save],
            )
            lens_save.click(
                fn=lens_save_fn, inputs=[lens_result_state],
                outputs=[lens_status, main_page, lens_page],
            )
            lens_discard.click(
                fn=lambda: ("🫧 已取消", gr.Column(visible=True), gr.Column(visible=False)),
                outputs=[lens_status, main_page, lens_page],
            )

        # ═══════════════ 页面切换 ═══════════════

        def to_creative():
            return (gr.Column(visible=False), gr.Column(visible=True), gr.Column(visible=False),
                    gr.Column(visible=False), gr.Column(visible=False), make_creative_agent())

        def to_film():
            return (gr.Column(visible=False), gr.Column(visible=False), gr.Column(visible=True),
                    gr.Column(visible=False), gr.Column(visible=False), make_film_agent())

        def to_music():
            return (gr.Column(visible=False), gr.Column(visible=False), gr.Column(visible=False),
                    gr.Column(visible=True), gr.Column(visible=False), make_music_agent())

        def to_lens():
            return (gr.Column(visible=False), gr.Column(visible=False), gr.Column(visible=False),
                    gr.Column(visible=False), gr.Column(visible=True), make_lens_agent())

        creative_btn.click(fn=to_creative, outputs=[main_page, creative_page, film_page, music_page, lens_page, creative_agent])
        film_btn.click(fn=to_film, outputs=[main_page, creative_page, film_page, music_page, lens_page, film_agent])
        music_btn.click(fn=to_music, outputs=[main_page, creative_page, film_page, music_page, lens_page, music_agent])
        lens_btn.click(fn=to_lens, outputs=[main_page, creative_page, film_page, music_page, lens_page, lens_agent])

    return app


if __name__ == "__main__":
    app = build_ui()
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True,
    )
