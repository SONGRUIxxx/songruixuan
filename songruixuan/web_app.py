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
    api_key=DEEPSEEK_KEY,
    api_base="https://api.deepseek.com/v1",
    model="deepseek-v4-pro",
)
llm_flash = LLMClient(
    api_key=DEEPSEEK_KEY,
    api_base="https://api.deepseek.com/v1",
    model="deepseek-v4-flash",
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


def make_film_agent() -> Agent:
    return Agent(
        llm_clients={"deepseek": llm_pro},
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
        llm_clients={"deepseek": llm_pro},
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
        n_notes = vector_memory.index_notes()
        n_music_notes = vector_memory.index_music_notes()
        n_photo = vector_memory.index_photo_library()
        total = n_conv + n_notes + n_music_notes + n_photo
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


async def simple_chat(message: str, history: list, agent_state):
    """电影/音乐等模式用的简单对话，不走库选择和附件"""
    result = await creative_chat(message, history, agent_state, [], None, False)
    return result[:3]  # msg, history, agent


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
                choices=["对话库", "电影库", "音乐库", "镜头库"],
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

        # ═══════════════ 学习电影 ═══════════════
        with gr.Column(visible=False) as film_page:
            ALL_PAGES.append(film_page)
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
                    return ("没有笔记可保存",) + to_main()
                try:
                    # 收集笔记内容
                    film_name = ""
                    content = ""
                    for m in agent_state.messages:
                        if m.role == "assistant" and "## 内容大纲" in str(m.content):
                            content = str(m.content)
                    if not content:
                        return ("未找到整理好的笔记，请先点「确认结束」",) + to_main()
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
                    return ("✅ 已保存到电影库",) + to_main()
                except Exception as e:
                    return (f"❌ {e}",) + to_main()

            film_send.click(
                fn=simple_chat, inputs=[film_input, film_chatbot, film_agent],
                outputs=[film_input, film_chatbot, film_agent],
            ).then(lambda: "", None, film_input)

            film_finish.click(
                fn=film_finish_fn, inputs=[film_agent, film_chatbot],
                outputs=[film_preview, film_status, film_save, film_discard, film_agent],
            )
            film_save.click(
                fn=lambda: '<div class="big-loading">💾 正在保存入库...</div>', outputs=[film_status],
            ).then(
                fn=film_save_fn, inputs=[film_agent],
                outputs=[film_status, main_page] + ALL_PAGES,
            )
            film_discard.click(
                fn=lambda: ("🫧 已丢弃",) + to_main(),
                outputs=[film_status] + [main_page] + ALL_PAGES,
            ).then(fn=lambda: (None, []), outputs=[film_agent, film_chatbot])

        # ═══════════════ 学习音乐 ═══════════════
        with gr.Column(visible=False) as music_page:
            ALL_PAGES.append(music_page)
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
                    return ("没有笔记可保存",) + to_main()
                try:
                    content = ""
                    for m in agent_state.messages:
                        if m.role == "assistant" and "## 曲目信息" in str(m.content):
                            content = str(m.content)
                    if not content:
                        return ("未找到整理好的笔记",) + to_main()
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
                    return ("✅ 已保存到音乐库",) + to_main()
                except Exception as e:
                    return (f"❌ {e}",) + to_main()

            music_send.click(
                fn=simple_chat, inputs=[music_input, music_chatbot, music_agent],
                outputs=[music_input, music_chatbot, music_agent],
            ).then(lambda: "", None, music_input)

            music_finish.click(
                fn=music_finish_fn, inputs=[music_agent, music_chatbot],
                outputs=[music_preview, music_status, music_save, music_discard, music_agent],
            )
            music_save.click(
                fn=lambda: '<div class="big-loading">💾 正在保存入库...</div>', outputs=[music_status],
            ).then(
                fn=music_save_fn, inputs=[music_agent],
                outputs=[music_status, main_page] + ALL_PAGES,
            )
            music_discard.click(
                fn=lambda: ("🫧 已丢弃",) + to_main(),
                outputs=[music_status] + [main_page] + ALL_PAGES,
            ).then(fn=lambda: (None, []), outputs=[music_agent, music_chatbot])

        # ═══════════════ 学习镜头 ═══════════════
        with gr.Column(visible=False) as lens_page:
            ALL_PAGES.append(lens_page)
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
                import shutil
                fpath = file if isinstance(file, str) else (file.get("name", "") if isinstance(file, dict) else getattr(file, "name", ""))
                saved = Path(WORKSPACE) / f"lens_{Path(fpath).name}"
                shutil.copy2(fpath, str(saved))
                vr = await vision_tool.execute(image=str(saved))
                if not vr.success:
                    return "", f"分析失败: {vr.error}", "", gr.Button(interactive=False)
                ai_desc = vr.content
                combined = f"**AI 分析**\n{ai_desc}"
                if desc_text.strip():
                    combined += f"\n\n**我的感受**\n{desc_text}"
                return combined, "预览中，确认后保存", combined, gr.Button(interactive=True)

            async def lens_save_fn(result_text):
                if not result_text:
                    return ("没有内容可保存",) + to_main()
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
                    return (f"✅ 已入库 (编号 {max_num:05d})",) + to_main()
                except Exception as e:
                    return (f"❌ {e}",) + to_main()

            lens_analyze.click(
                fn=lens_analyze_fn, inputs=[lens_file, lens_desc],
                outputs=[lens_preview, lens_status, lens_result_state, lens_save],
            )
            lens_save.click(
                fn=lambda: '<div class="big-loading">💾 正在保存入库...</div>', outputs=[lens_status],
            ).then(
                fn=lens_save_fn, inputs=[lens_result_state],
                outputs=[lens_status, main_page] + ALL_PAGES,
            )
            lens_discard.click(
                fn=lambda: ("🫧 已取消",) + to_main(),
                outputs=[lens_status] + [main_page] + ALL_PAGES,
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
            return [gr.Column(visible=False) for _ in range(8)]  # 8 个独立 Column

        def to_creative():
            c = _mk_cols(); c[1] = gr.Column(visible=True); return tuple(c) + (make_creative_agent(),)
        def to_film():
            c = _mk_cols(); c[2] = gr.Column(visible=True); return tuple(c) + (make_film_agent(),)
        def to_music():
            c = _mk_cols(); c[3] = gr.Column(visible=True); return tuple(c) + (make_music_agent(),)
        def to_lens():
            c = _mk_cols(); c[4] = gr.Column(visible=True); return tuple(c) + (make_lens_agent(),)
        def to_img_gen():
            c = _mk_cols(); c[5] = gr.Column(visible=True); return tuple(c)
        def to_vid_gen():
            c = _mk_cols(); c[6] = gr.Column(visible=True); return tuple(c)
        def to_site():
            c = _mk_cols(); c[7] = gr.Column(visible=True); return tuple(c)

        creative_btn.click(fn=to_creative, outputs=[main_page, creative_page, film_page, music_page, lens_page, img_gen_page, vid_gen_page, site_page, creative_agent])
        film_btn.click(fn=to_film, outputs=[main_page, creative_page, film_page, music_page, lens_page, img_gen_page, vid_gen_page, site_page, film_agent])
        music_btn.click(fn=to_music, outputs=[main_page, creative_page, film_page, music_page, lens_page, img_gen_page, vid_gen_page, site_page, music_agent])
        lens_btn.click(fn=to_lens, outputs=[main_page, creative_page, film_page, music_page, lens_page, img_gen_page, vid_gen_page, site_page, lens_agent])
        img_gen_btn.click(fn=to_img_gen, outputs=[main_page, creative_page, film_page, music_page, lens_page, img_gen_page, vid_gen_page, site_page])
        vid_gen_btn.click(fn=to_vid_gen, outputs=[main_page, creative_page, film_page, music_page, lens_page, img_gen_page, vid_gen_page, site_page])
        site_btn.click(fn=to_site, outputs=[main_page, creative_page, film_page, music_page, lens_page, img_gen_page, vid_gen_page, site_page])

    return app


if __name__ == "__main__":
    # 启动时同步向量库
    print("🧠 正在同步向量库...", end=" ", flush=True)
    try:
        vector_memory.index_conversations()
        vector_memory.index_notes()
        vector_memory.index_music_notes()
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
