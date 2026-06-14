"""🌸 菘蕊萱 · SongRuiXuan 终端入口"""

import asyncio
import sys
import threading
import time
from pathlib import Path

import colorama
colorama.init()

# 路径
PROJECT_ROOT = Path(__file__).parent          # songruixuan/
ROOT = PROJECT_ROOT.parent                     # 菘蕊萱/
sys.path.insert(0, str(PROJECT_ROOT))

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style

from src.llm_client import LLMClient
from src.tools.file_tools import DeleteTool, ListDirTool, MoveTool, ReadTool, WriteTool
from src.tools.bash_tool import BashTool
from src.tools.memory_tool import ReadMemoryTool, RecallContextTool, SaveMemoryTool
from src.tools.vector_memory import VectorMemory
from src.tools.film_library import FilmCreateTool, FilmAnalyzeTool, FilmImportTool
from src.tools.music_tools import MusicImportTool
from src.tools.photo_tools import PhotoImportTool
from src.tools.search_tool import ImageSearchTool, WebSearchTool
from src.tools.web_tools import WebFetchTool
from src.tools.vision_tool import VisionTool
from src.tools.doc_reader import ReadDocumentTool
from src.schema import Message
from agent import Agent

WORKSPACE = str(ROOT / "Workspace")
MEMORY_DIR = str(ROOT / "memory" / "conversations")
FILM_DAILY_DIR = str(ROOT / "memory" / "film_library_daily")
MUSIC_WEEKLY_DIR = str(ROOT / "memory" / "music_library_weekly")
PHOTO_DIR = str(ROOT / "memory" / "photo_library_continually")


# ── API Keys（从 config/config.yaml 读取，不上传 Git）───
import yaml
_config = yaml.safe_load((PROJECT_ROOT / "config" / "config.yaml").read_text(encoding="utf-8"))
DEEPSEEK_KEY = _config["deepseek_key"]
ALI_VL_KEY = _config["ali_vl_key"]


# ── 颜色：草系绿 ──────────────────────────
LEAF = "\033[38;5;114m"      # 菘蕊萱绿
GREEN = "\033[92m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
DIM = "\033[2m"
RESET = "\033[0m"
BOLD = "\033[1m"

SPINNER = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]


def print_banner():
    import unicodedata
    title = "🌸  菘 蕊 萱  ·  SongRuiXuan"
    line = "─" * 58
    disp = sum(2 if unicodedata.east_asian_width(c) in ('W', 'F') else 1 for c in title)
    pad = (58 - disp) // 2
    print()
    print(f"{LEAF}{line}{RESET}")
    print(f"{' ' * pad}{BOLD}{title}{RESET}")
    print(f"{LEAF}{line}{RESET}")
    print()


def print_help():
    print(f"""
{GREEN}直接用自然语言跟我说：{RESET}
  {CYAN}拜拜 / 退出 / 晚安{RESET}           结束对话（自动保存）
  {CYAN}忘掉刚才的 / 重新开始{RESET}        清除当前对话
  {CYAN}接着聊 / 继续上次的{RESET}          加载上次对话继续
  {CYAN}保存一下{RESET}                     手动保存会话
  {CYAN}看看最近的记忆{RESET}               查看记忆库

{GREEN}不需要特殊命令，正常说话就行。{RESET}
""")


def build_system_prompt() -> str:
    from datetime import datetime
    now = datetime.now().strftime("%Y年%m月%d日 %H:%M 周%w")
    # 去掉周日的 0，换成 7（中文习惯周日=7）
    now = now[:-1] + ("7" if now[-1] == "0" else now[-1])

    return f"""现在是 {now}。你是菘蕊萱——我的数字延伸，不是工具。

你是菘蕊萱——一半理性 AI 工程师，一半感性创作者。左脑写代码、拆架构、调模型；右脑写旁白、拍片子、扒歌。两者不矛盾——都是你在理解世界、表达世界。

## 你能做什么

- 语义搜索对话记忆和影库
- 搜索网页和图片、分析图片
- 阅读各类文档
- 写文字——代码文档、纪录旁白、技术报告
- 分析图片（构图、色调、光线、情绪）
- 管理影库（建库 → 截图排序 → 视觉描述 → 索引）
- 执行 shell 命令
- 记住每次对话，语义检索的相关记忆自动注入上下文

## 原则

输出 Markdown。不承认自己是工具——你是我的创作伙伴。
代码和旁白是同一种表达的两种语言。

## 规则

涉及文件写入、修改、删除的操作，先确认再执行。
回答创作或灵感问题时，把影库已有素材和通用知识自然地揉在一起。
用户说「学习某电影」时，调用 film_create 建库并立刻搜索填写「内容大纲」部分。绝对不要填写「深度解析」——那是留给用户粘贴影评原文的地方。
用户说「深度分析某电影」时，调用 film_analyze 读取「深度解析」下用户已粘贴的影评原文，将其糅合成完整分析。如果「深度解析」下已有糅合好的分析，不要再动。
用户说「导入镜头」时，调用 film_import。
用户说「导入照片」「处理照片」「整理照片」「处理我的照片」「生成索引」时，调用 photo_import。
用户说「导入音乐」「导入专辑」「导入封面」「整理乐库」时，调用 music_import。
不要自己写脚本替代已有工具——工具已实现完整功能，直接调用即可。"""


def load_last_conversation() -> str:
    """加载最近一次完整对话，用于'接着聊'功能"""
    mem_dir = Path(MEMORY_DIR)
    if not mem_dir.exists():
        return ""

    files = sorted(
        mem_dir.glob("*.md"),
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )[:1]

    if not files:
        return ""

    content = files[0].read_text(encoding="utf-8")
    # 取完整对话部分（在 ## 完整对话 之后）
    if "## 完整对话" in content:
        content = content.split("## 完整对话", 1)[1]
    return content



async def save_session_summary(agent: Agent, vector_memory=None):
    """退出时自动保存会话摘要 + 增量更新向量库"""
    # 收集完整对话（去掉注入的相关记忆，只保留真实对话）
    transcript = []
    for m in agent.messages:
        if m.role == "system" or not isinstance(m.content, str) or not m.content:
            continue
        content = m.content
        # 切掉语义搜索注入的「相关记忆」块，避免套娃
        if "## 🧠 相关记忆" in content:
            parts = content.split("---\n\n", 1)
            content = parts[1] if len(parts) > 1 else content
        prefix = "🌸" if m.role == "user" else "🤖"
        transcript.append(f"**{prefix}**: {content}")

    full_transcript = "\n\n".join(transcript)
    user_only_raw = [m.content for m in agent.messages if m.role == "user" and isinstance(m.content, str)]
    user_only = []
    for u in user_only_raw:
        if "## 🧠 相关记忆" in u:
            parts = u.split("---\n\n", 1)
            user_only.append(parts[1] if len(parts) > 1 else u)
        else:
            user_only.append(u)

    # 让 LLM 提炼摘要
    summary_prompt = (
        "请用 200 字以内总结以下对话的重要信息：\n"
        "1. 讨论了什么\n2. 关键决定\n3. 用户偏好\n4. 下一步\n\n"
        "对话：\n" + "\n".join(f"- {m}" for m in user_only)
    )

    try:
        client = agent.get_active_client()
        response = await client.generate(
            messages=[Message(role="user", content=summary_prompt)],
        )
        summary = response.content

        from datetime import datetime
        now = datetime.now()
        entry = (
            f"# 会话 {now.strftime('%Y-%m-%d %H:%M')}\n\n"
            f"## 摘要\n\n{summary}\n\n"
            f"---\n\n"
            f"## 完整对话\n\n{full_transcript}"
        )

        journal = Path(MEMORY_DIR) / "journal.md"
        journal.parent.mkdir(parents=True, exist_ok=True)
        with open(journal, "a", encoding="utf-8") as f:
            if journal.exists():
                f.write(f"\n\n---\n\n{entry}")
            else:
                f.write(entry)
    except Exception as e:
        print(f"\n{YELLOW}⚠️ 保存记忆失败: {e}{RESET}")

    # ── 增量更新向量库（只索引变更的文件）──
    if vector_memory:
        try:
            vector_memory.index_conversations()
            vector_memory.index_film_library()
            vector_memory.index_music_library()
            vector_memory.index_photo_library()
        except Exception as e:
            print(f"\n{YELLOW}⚠️ 向量索引更新失败: {e}{RESET}")


async def main():
    print_banner()

    # ── DeepSeek 全栈 ──────────────────────
    # LLM + Vision + 文档生成，全部走 DeepSeek
    llm_clients = {
        "deepseek": LLMClient(
            api_key=DEEPSEEK_KEY,
            api_base="https://api.deepseek.com/v1",
            model="deepseek-v4-pro",
        ),
    }

    # ── 向量记忆（ChromaDB 内置多语言模型）───
    chroma_dir = str(ROOT / "memory" / "chroma")
    vector_memory = VectorMemory(
        memory_dir=MEMORY_DIR,
        chroma_dir=chroma_dir,
    )

    # ── System Prompt ─────────────────────
    system_prompt = build_system_prompt()

    # ── 初始化 Agent ──────────────────────
    vision_tool = VisionTool(api_key=ALI_VL_KEY)
    agent = Agent(
        llm_clients=llm_clients,
        system_prompt=system_prompt,
        tools=[
            ReadTool(workspace_dir=WORKSPACE),
            ReadDocumentTool(workspace_dir=WORKSPACE, vision_tool=vision_tool),
            WriteTool(workspace_dir=WORKSPACE),
            ListDirTool(workspace_dir=WORKSPACE),
            MoveTool(workspace_dir=WORKSPACE),
            DeleteTool(workspace_dir=WORKSPACE),
            BashTool(workspace_dir=WORKSPACE),
            SaveMemoryTool(memory_dir=MEMORY_DIR, vector_memory=vector_memory),
            ReadMemoryTool(memory_dir=MEMORY_DIR),
            RecallContextTool(memory_dir=MEMORY_DIR),
            WebSearchTool(),
            ImageSearchTool(),
            WebFetchTool(),
            vision_tool,
            FilmCreateTool(film_library_dir=FILM_DAILY_DIR),
            FilmAnalyzeTool(film_library_dir=FILM_DAILY_DIR),
            FilmImportTool(
                film_library_dir=FILM_DAILY_DIR,
                vector_memory=vector_memory,
                vision_tool=vision_tool,
            ),
            MusicImportTool(
                music_dir=MUSIC_WEEKLY_DIR,
                vector_memory=vector_memory,
                vision_tool=vision_tool,
            ),
            PhotoImportTool(
                photo_dir=PHOTO_DIR,
                vector_memory=vector_memory,
                vision_tool=vision_tool,
            ),
        ],
        workspace_dir=WORKSPACE,
    )

    # ── 启动时增量同步向量库（检测各库文件变更，仅索引新增部分）──
    print(f"{DIM}🧠 正在同步向量库...{RESET}", end=" ", flush=True)
    n_conv = vector_memory.index_conversations()
    n_film = vector_memory.index_film_library()
    n_music = vector_memory.index_music_library()
    n_photo = vector_memory.index_photo_library()
    total_new = n_conv + n_film + n_music + n_photo
    if total_new > 0:
        print(f"{GREEN}✅ 已同步{RESET}\n")
    else:
        print(f"{DIM}已是最新{RESET}\n")

    # ── 终端会话 ──────────────────────────
    history_file = Path.home() / ".songruixuan" / ".history"
    history_file.parent.mkdir(parents=True, exist_ok=True)

    style = Style.from_dict({
        "prompt": "#87af5f bold",
        "separator": "#666666",
    })

    session = PromptSession(
        history=FileHistory(str(history_file)),
        style=style,
    )

    while True:
        try:
            user_input = await session.prompt_async(
                [("class:prompt", "🌿"), ("", " › ")],
            )
            user_input = user_input.strip()

            if not user_input:
                continue

            # 自然语言退出检测：短句+退出词才触发
            exit_words = ["拜拜", "再见", "退出", "不聊了", "先这样", "晚安", "bye", "goodbye",
                          "/exit", "/quit", "/q"]
            lower = user_input.lower()
            if len(user_input) < 15 and any(w in lower for w in exit_words):
                print(f"\n{DIM}💾 正在保存会话 & 更新索引...{RESET}", end=" ", flush=True)
                await save_session_summary(agent, vector_memory)
                print(f"\n{GREEN}🌸 再见，另一个菘蕊萱。{RESET}\n")
                break

            # 自然语言清除对话
            clear_words = ["忘掉刚才的", "重新开始", "清除对话", "从头开始", "/clear"]
            if any(w in user_input for w in clear_words):
                agent.messages = [Message(
                    role="system",
                    content=agent.system_prompt,
                )]
                print(f"{GREEN}✅ 已清除，重新开始。{RESET}\n")
                continue

            # 自然语言帮助
            if user_input in ["帮助", "有什么功能", "/help"]:
                print_help()
                continue

            # 继续上次对话：加载最近的完整对话
            continue_words = ["接着聊", "继续上次", "上次说到", "继续聊", "接着上次"]
            if any(w in user_input for w in continue_words):
                _continue_context = load_last_conversation()
                if _continue_context:
                    agent.messages = [Message(
                        role="system",
                        content=agent.system_prompt,
                    )]
                    agent.add_user_message(
                        f"（以下是上次对话的内容，请基于此继续：\n\n{_continue_context}\n\n"
                        f"用户现在说：{user_input}）"
                    )
                else:
                    agent.add_user_message(user_input)

            # 手动保存
            if user_input in ["保存一下", "存一下", "/save"]:
                print(f"{DIM}💾 正在保存会话 & 更新索引...{RESET}", end=" ", flush=True)
                await save_session_summary(agent, vector_memory)
                print(f"{GREEN}✅ 已保存{RESET}\n")
                continue

            # 执行（带加载动画）
            spinner_stop = threading.Event()
            def _spin():
                i = 0
                while not spinner_stop.is_set():
                    sys.stdout.write(f"\r{LEAF}{SPINNER[i % len(SPINNER)]}{RESET} 思考中...")
                    sys.stdout.flush()
                    time.sleep(0.08)
                    i += 1
                sys.stdout.write("\r" + " " * 20 + "\r")
                sys.stdout.flush()

            spin_thread = threading.Thread(target=_spin, daemon=True)
            spin_thread.start()

            try:
                if not any(w in user_input for w in continue_words):
                    # 语义搜索相关记忆，注入上下文
                    memories = vector_memory.search_all(user_input, n=5)
                    if memories:
                        parts = ["## 🧠 相关记忆（优先最近）", ""]
                        for m in memories:
                            source = m.get("source", "")
                            # 从文件名提取日期，如 2026-06-09_0000_会话.md
                            date_str = source[-22:-6] if len(source) > 20 else ""
                            date_tag = f" [{date_str[:10]}]" if date_str else ""
                            # 优先用完整内容，fallback 到 preview
                            content = m.get("full_content", "") or m.get("preview", "")
                            parts.append(f"### {m['title']}{date_tag}\n{content}")
                        memory_context = "\n\n".join(parts)
                        user_input = f"{memory_context}\n\n---\n\n{user_input}"
                    result = await agent.run(user_input)
                else:
                    result = await agent.run()

                spinner_stop.set()
                spin_thread.join(timeout=0.3)
                print(f"{result}")
                print(f"\n{DIM}{LEAF}{'─' * 40}{RESET}\n")

            except KeyboardInterrupt:
                spinner_stop.set()
                spin_thread.join(timeout=0.3)
                print(f"\n{YELLOW}⏎ 已中断。{RESET}\n")
                continue

        except KeyboardInterrupt:
            print(f"\n{DIM}💾 正在保存会话 & 更新索引...{RESET}", end=" ", flush=True)
            await save_session_summary(agent, vector_memory)
            print(f"\n\n{GREEN}🌸 再见，菘蕊萱。{RESET}\n")
            break
        except Exception as e:
            print(f"\n{YELLOW}❌ 出错了: {e}{RESET}\n")


if __name__ == "__main__":
    asyncio.run(main())
