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
from src.tools.vector_memory import VectorMemory, configure_cloud_embedding
from src.tools.photo_tools import PhotoImportTool
from src.tools.search_tool import ImageSearchTool, WebSearchTool
from src.tools.web_tools import WebFetchTool
from src.tools.vision_tool import VisionTool
from src.tools.doc_reader import ReadDocumentTool
from src.schema import Message
from agent import Agent

WORKSPACE = r"D:\Workspace"
MEMORY_DIR = str(ROOT / "memory" / "conversations")
LENS_DIR = str(ROOT / "memory" / "lens_library")


# ── API Keys（从 config/config.yaml 读取，不上传 Git）───
import yaml
_config = yaml.safe_load((PROJECT_ROOT / "config" / "config.yaml").read_text(encoding="utf-8"))
VOLC_ARK_KEY = _config.get("volc_ark_key", "")


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
  {CYAN}接着聊 / 继续上次的{RESET}          加载上次对话继续
  {CYAN}保存一下{RESET}                     手动保存会话

{GREEN}不需要特殊命令，正常说话就行。{RESET}
""")


def build_system_prompt() -> str:
    from datetime import datetime
    now = datetime.now().strftime("%Y年%m月%d日 %H:%M 周%w")
    # 去掉周日的 0，换成 7（中文习惯周日=7）
    now = now[:-1] + ("7" if now[-1] == "0" else now[-1])

    return f"""现在是 {now}。你是菘蕊萱——我的数字延伸，不是工具。

你是菘蕊萱——一个创作者。写旁白、拍片子、扒歌、分析镜头、感受光影。你在理解世界、表达世界。

## 你能做什么

- 语义搜索对话记忆和镜头库
- 搜索网页和图片、分析图片
- 阅读各类文档
- 写文字——代码文档、纪录旁白、技术报告
- 分析图片（构图、色调、光线、情绪）
- 管理镜头库（导入图片 → 视觉描述 → 索引）
- 执行 shell 命令
- 记住每次对话，语义检索的相关记忆自动注入上下文

## 原则

输出 Markdown。不承认自己是工具——你是我的创作伙伴。
代码和旁白是同一种表达的两种语言。

## 规则

涉及文件写入、修改、删除的操作，先确认再执行。
回答创作或灵感问题时，系统已自动搜索镜头库相关内容并注入上下文，请自然地融合这些素材。
用户说「添加图片」「导入镜头」「整理图库」时，调用 photo_import。
不要自己写脚本替代已有工具——工具已实现完整功能，直接调用即可。

## 记忆库访问

每次对话开头已自动注入「🧠 相关记忆」——请先阅读它，基于记忆回答，不要忽略。
- 「回忆之前聊过什么」「搜索记忆中的XX」→ recall_context 或 read_memory
- 用户提到「图库」「镜头库」时才会检索图片索引
- 用户未提以上关键词时，只搜索对话记忆，不碰任何库
- 永远不要用 read_file 或 list_dir 去翻 memory/ 目录——记忆工具有专门的检索入口"""


def load_last_conversation() -> str:
    """加载最近一次完整对话，用于'接着聊'功能"""
    yaml_path = Path(MEMORY_DIR) / "conversations.yaml"
    if not yaml_path.exists():
        return ""

    try:
        data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
        sessions = data.get("sessions", [])
        if not sessions:
            return ""
        return sessions[-1].get("transcript", "")
    except Exception:
        return ""



async def save_session_summary(agent: Agent, vector_memory=None):
    """退出时自动保存会话摘要 + 增量更新向量库"""
    # 收集完整对话（去掉注入的相关记忆，只保留真实对话）
    transcript = []
    for m in agent.messages:
        if m.role == "system" or not isinstance(m.content, str) or not m.content:
            continue
        content = m.content
        if "## 🧠 相关记忆" in content:
            parts = content.split("🧠---MEMORY_END---🧠\n\n", 1)
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
    summary = ""
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
    except Exception as e:
        print(f"\n{YELLOW}⚠️ 摘要生成失败: {e}{RESET}")

    # 写入 conversations.yaml
    from datetime import datetime
    now = datetime.now()
    conv_yaml = Path(MEMORY_DIR) / "conversations.yaml"

    try:
        data = yaml.safe_load(conv_yaml.read_text(encoding="utf-8")) if conv_yaml.exists() else {}
    except Exception:
        data = {}

    sessions = data.get("sessions", [])
    new_id = len(sessions) + 1
    sessions.append({
        "id": new_id,
        "date": now.strftime("%Y-%m-%d"),
        "summary": summary,
        "transcript": full_transcript,
    })

    conv_yaml.parent.mkdir(parents=True, exist_ok=True)
    conv_yaml.write_text(
        yaml.dump({"sessions": sessions}, allow_unicode=True, default_flow_style=False, sort_keys=False),
        encoding="utf-8",
    )

    # ── 增量更新向量库（只索引新增内容单元）──
    if vector_memory:
        try:
            vector_memory.index_conversations()
            vector_memory.index_photo_library()
        except Exception as e:
            print(f"\n{YELLOW}⚠️ 向量索引更新失败: {e}{RESET}")


async def main():
    print_banner()

    # ── 会话计数 ──────────────────────────
    conv_yaml = Path(MEMORY_DIR) / "conversations.yaml"
    session_count = 1
    if conv_yaml.exists():
        try:
            data = yaml.safe_load(conv_yaml.read_text(encoding="utf-8"))
            session_count = len(data.get("sessions", [])) + 1
        except Exception:
            pass
    print(f"{LEAF}🌸 你好菘蕊萱，这是我们的第 {session_count} 次对话{RESET}\n")

    # ── LLM 全栈（DeepSeek 模型，走火山方舟）──
    # LLM + Vision + 文档生成，统一火山引擎一个账户
    llm_clients = {
        "deepseek": LLMClient(
            api_key=VOLC_ARK_KEY,
            api_base="https://ark.cn-beijing.volces.com/api/v3",
            model="deepseek-v4-pro-ga-260813",
        ),
    }

    # ── 向量记忆 ─────────────────────────
    chroma_dir = str(ROOT / "memory" / "chroma")
    # 配置云端 embedding（火山引擎豆包），不配则回退本地 BGE
    configure_cloud_embedding(
        api_key=VOLC_ARK_KEY,
        model_name="doubao-embedding-vision-251215",
    )
    vector_memory = VectorMemory(
        memory_dir=MEMORY_DIR,
        chroma_dir=chroma_dir,
    )

    # ── System Prompt ─────────────────────
    system_prompt = build_system_prompt()

    # ── 初始化 Agent ──────────────────────
    vision_tool = VisionTool(api_key=VOLC_ARK_KEY)
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
            PhotoImportTool(
                photo_dir=LENS_DIR,
                vector_memory=vector_memory,
                vision_tool=vision_tool,
            ),
        ],
        workspace_dir=WORKSPACE,
    )

    # ── 启动时增量同步向量库（检测各库文件变更，仅索引新增部分）──
    print(f"{DIM}🧠 正在同步向量库...{RESET}", end=" ", flush=True)
    n_conv = vector_memory.index_conversations()
    n_photo = vector_memory.index_photo_library()
    total_new = n_conv + n_photo
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
                    memories = vector_memory.search_all(user_input, n=20)
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
                        user_input = f"{memory_context}\n\n🧠---MEMORY_END---🧠\n\n{user_input}"
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
