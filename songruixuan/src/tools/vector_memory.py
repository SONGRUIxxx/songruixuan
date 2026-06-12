"""🧠 向量记忆 — 中英文语义检索（自动管理模型，零手动）"""

import hashlib
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions


_ef = None


def _get_ef():
    """延迟加载模型，避免阻塞启动"""
    global _ef
    if _ef is None:
        _ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="BAAI/bge-small-zh-v1.5",
        )
    return _ef


class VectorMemory:
    """语义记忆索引 — 首次启动自动下载多语言模型，之后离线秒级响应"""

    def __init__(self, memory_dir: str, chroma_dir: str):
        self.memory_dir = Path(memory_dir)
        self.chroma_dir = Path(chroma_dir)
        self.chroma_dir.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(path=str(self.chroma_dir))
        self.collection = self.client.get_or_create_collection(
            name="conversations",
            embedding_function=_get_ef(),
            metadata={"hnsw:space": "cosine"},
        )
        self._film_collection = None
        self._music_collection = None
        self._photo_collection = None

    # ── Collection 懒加载 ──────────────────────────

    def get_or_create_film_collection(self):
        """影库 collection"""
        if self._film_collection is None:
            self._film_collection = self.client.get_or_create_collection(
                name="film_library_daily",
                embedding_function=_get_ef(),
                metadata={"hnsw:space": "cosine"},
            )
        return self._film_collection

    def get_or_create_music_collection(self):
        """乐库 collection"""
        if self._music_collection is None:
            self._music_collection = self.client.get_or_create_collection(
                name="music_library_weekly",
                embedding_function=_get_ef(),
                metadata={"hnsw:space": "cosine"},
            )
        return self._music_collection

    def get_or_create_photo_collection(self):
        """摄影库 collection"""
        if self._photo_collection is None:
            self._photo_collection = self.client.get_or_create_collection(
                name="photo_library_continually",
                embedding_function=_get_ef(),
                metadata={"hnsw:space": "cosine"},
            )
        return self._photo_collection

    # ── 通用工具 ──────────────────────────────────

    def _file_hash(self, filepath: Path) -> str:
        stat = filepath.stat()
        raw = f"{stat.st_mtime}_{stat.st_size}".encode()
        return hashlib.md5(raw).hexdigest()[:12]

    def _text_hash(self, text: str) -> str:
        return hashlib.md5(text[:200].encode()).hexdigest()[:12]

    @staticmethod
    def _search_col(collection, query: str, n: int) -> list[dict]:
        if collection.count() == 0:
            return []
        results = collection.query(
            query_texts=[query],
            n_results=min(n, collection.count()),
        )
        memories = []
        if results["metadatas"] and results["metadatas"][0]:
            for meta in results["metadatas"][0]:
                memories.append({
                    "title": meta.get("title", meta.get("film", meta.get("album", ""))),
                    "preview": meta.get("preview", meta.get("desc", "")),
                    "source": meta.get("source", meta.get("film", meta.get("album", ""))),
                })
        return memories

    # ── 对话记忆索引 ──────────────────────────────

    def index_conversations(self) -> int:
        if not self.memory_dir.exists():
            return 0

        existing_ids = set(self.collection.get()["ids"]) if self.collection.count() > 0 else set()
        indexed = 0

        # 1. 索引单次会话 .md
        for md_file in sorted(self.memory_dir.glob("*.md")):
            if md_file.name == "journal.md":
                continue
            self._index_one_file(md_file, md_file.stem, existing_ids)
            indexed += 1

        # 2. 索引 journal.md：每次启动清空旧索引，从头分块重建
        journal = self.memory_dir / "journal.md"
        if journal.exists():
            old_ids = self.collection.get()["ids"]
            for oid in old_ids:
                if oid.startswith("j_"):
                    self.collection.delete(ids=[oid])
            content = journal.read_text(encoding="utf-8")
            cs, ol = 2000, 500
            for i, p in enumerate(range(0, len(content), cs - ol)):
                chunk = content[p:p + cs]
                if not chunk.strip():
                    continue
                chroma_id = f"j_{i}__{hashlib.md5(chunk[:200].encode()).hexdigest()[:8]}"
                self.collection.add(
                    ids=[chroma_id],
                    documents=[chunk[:1200]],
                    metadatas=[{"source": str(journal), "title": f"journal-{i+1}", "preview": chunk[:400]}],
                )
                indexed += 1

        return indexed

    def _index_one_file(self, md_file, doc_id, existing_ids):
        current_hash = self._file_hash(md_file)
        chroma_id = f"{doc_id}__{current_hash}"
        if chroma_id in existing_ids:
            return
        for old_id in existing_ids:
            if old_id.startswith(f"{doc_id}__"):
                self.collection.delete(ids=[old_id]); break
        content = md_file.read_text(encoding="utf-8")
        self.collection.add(
            ids=[chroma_id], documents=[content[:1200]],
            metadatas=[{"source": str(md_file), "title": doc_id, "preview": content[:400]}],
        )

    def search(self, query: str, n: int = 5) -> list[dict]:
        """搜对话记忆"""
        return self._search_col(self.collection, query, n)

    def add_memory(self, title: str, content: str):
        chroma_id = f"{title}__{self._text_hash(content)}"
        self.collection.add(
            ids=[chroma_id],
            documents=[f"# {title}\n\n{content}"[:1200]],
            metadatas=[{"title": title, "preview": content[:400]}],
        )

    # ── 影库索引 ──────────────────────────────────

    def index_film_library(self) -> int:
        """扫描扁平影库：读根目录 index.yaml"""
        film_dir = self.memory_dir.parent / "film_library_daily"
        if not film_dir.exists():
            return 0

        idx_file = film_dir / "index.yaml"
        if not idx_file.exists():
            return 0

        import yaml
        try:
            data = yaml.safe_load(idx_file.read_text(encoding="utf-8")) or {}
        except Exception:
            return 0

        fc = self.get_or_create_film_collection()
        exist_ids = set(fc.get()["ids"]) if fc.count() > 0 else set()
        indexed = 0

        for fr in data.get("frames", []):
            fname = fr.get("file", "")
            film = fr.get("film", "")
            desc = fr.get("desc", "")
            if not desc or not fname:
                continue
            doc_id = f"{film}__{fname}"
            f_hash = hashlib.md5(desc[:200].encode()).hexdigest()[:12]
            chroma_id = f"{doc_id}__{f_hash}"
            if chroma_id not in exist_ids:
                for old in exist_ids:
                    if old.startswith(doc_id):
                        fc.delete(ids=[old]); break
                fc.add(
                    ids=[chroma_id],
                    documents=[f"《{film}》{fname}: {desc}"[:1200]],
                    metadatas=[{"film": film, "frame": fname, "desc": desc[:400]}],
                )
                indexed += 1

        return indexed

    def search_films(self, query: str, n: int = 5) -> list[dict]:
        """搜影库"""
        return self._search_col(self.get_or_create_film_collection(), query, n)

    # ── 乐库索引 ──────────────────────────────────

    def index_music_library(self) -> int:
        """扫描扁平乐库：读根目录 index.yaml"""
        music_dir = self.memory_dir.parent / "music_library_weekly"
        if not music_dir.exists():
            return 0

        idx_file = music_dir / "index.yaml"
        if not idx_file.exists():
            return 0

        import yaml
        try:
            data = yaml.safe_load(idx_file.read_text(encoding="utf-8")) or {}
        except Exception:
            return 0

        mc = self.get_or_create_music_collection()
        exist_ids = set(mc.get()["ids"]) if mc.count() > 0 else set()
        indexed = 0

        for tr in data.get("tracks", []):
            tname = tr.get("file", tr.get("title", ""))
            album = tr.get("album", "")
            desc = tr.get("desc", "")
            if not desc or not tname:
                continue
            doc_id = f"{album}__{tname}"
            t_hash = hashlib.md5(desc[:200].encode()).hexdigest()[:12]
            chroma_id = f"{doc_id}__{t_hash}"
            if chroma_id not in exist_ids:
                for old in exist_ids:
                    if old.startswith(doc_id):
                        mc.delete(ids=[old]); break
                mc.add(
                    ids=[chroma_id],
                    documents=[f"《{album}》{tname}: {desc}"[:1200]],
                    metadatas=[{"album": album, "track": tname, "desc": desc[:400]}],
                )
                indexed += 1

        return indexed

    def search_music(self, query: str, n: int = 5) -> list[dict]:
        """搜乐库"""
        return self._search_col(self.get_or_create_music_collection(), query, n)

    # ── 摄影库索引 ────────────────────────────────

    def index_photo_library(self) -> int:
        """扫描扁平摄影库：读根目录 index.yaml"""
        photo_dir = self.memory_dir.parent / "photo_library_continually"
        if not photo_dir.exists():
            return 0
        idx_file = photo_dir / "index.yaml"
        if not idx_file.exists():
            return 0
        import yaml
        try:
            data = yaml.safe_load(idx_file.read_text(encoding="utf-8")) or {}
        except Exception:
            return 0
        pc = self.get_or_create_photo_collection()
        exist_ids = set(pc.get()["ids"]) if pc.count() > 0 else set()
        indexed = 0
        for p in data.get("photos", []):
            fname = p.get("file", "")
            desc = p.get("desc", "")
            if not desc or not fname:
                continue
            pid = f"photo_{fname}"
            import hashlib
            ph = hashlib.md5(desc[:200].encode()).hexdigest()[:8]
            chroma_id = f"{pid}__{ph}"
            if chroma_id not in exist_ids:
                for old in exist_ids:
                    if old.startswith(pid):
                        pc.delete(ids=[old]); break
                pc.add(ids=[chroma_id], documents=[f"{fname}: {desc}"[:1200]],
                       metadatas=[{"file": fname, "desc": desc[:400]}])
                indexed += 1
        return indexed

    def search_photos(self, query: str, n: int = 5) -> list[dict]:
        """搜摄影库"""
        return self._search_col(self.get_or_create_photo_collection(), query, n)

    # ══════════════════════════════════════════════════
    # 语义路由：按用户意图分三档
    #
    # 档位 1：对话记忆 → 永远搜（基础上下文）
    # 档位 2：影库 + 乐库 → 创作模式（镜头参考、配乐灵感）
    #         影库和乐库绑定 —— 创作时画面和音乐都是灵感来源
    # 档位 3：摄影库 → 仅当用户明确在找自己拍过的照片
    #         （个人索引，不参与创作）
    # ══════════════════════════════════════════════════

    # ── 创作模式关键词（触发影库 + 乐库搜索）──
    _CREATIVE_KW = [
        # 电影本体
        "电影", "影片", "影评", "看电影", "观影", "看什么", "看了",
        "胶片", "上映", "票房", "演员", "编剧", "导演", "剪辑",
        "片尾", "片头", "影库", "film", "movie", "cinema",
        # 镜头语言 / 视觉参考
        "镜头", "构图", "分镜", "画面", "调色", "色调", "光线",
        "景别", "运镜", "长镜头", "特写", "远景", "近景",
        # 配乐 / 音乐参考
        "配乐", "原声", "soundtrack", "旋律", "编曲", "插曲",
        "主题曲", "背景音乐", "bgm", "ost", "音乐", "歌曲",
        "乐库", "听什么", "专辑", "唱片", "音乐人", "歌手",
        # 创作行为
        "学习", "分析", "参考", "借鉴", "致敬", "灵感",
        "旁白", "文案", "台词", "创作", "写", "脚本", "稿子",
    ]

    # ── 个人照片查找关键词（仅当用户明确在翻自己的相册）──
    #     注意：刻意不用 "构图""色调" 等泛词，否则日常聊天也会触发
    _PHOTO_FIND_KW = [
        "我拍", "我的照片", "我相册", "我摄影",
        "拍过", "拍的照片", "拍的图片",
        "找照片", "找一下照片", "找张照片", "找图片",
        "翻翻", "翻一下", "翻相册",
        "之前拍", "上次拍", "以前拍",
        "照片库", "摄影库", "相册里",
    ]

    @classmethod
    def _match_any(cls, query: str, keywords: list[str]) -> bool:
        """query 中是否包含任一关键词（大小写不敏感）"""
        q = query.lower()
        return any(kw.lower() in q for kw in keywords)

    def _should_search_creative(self, query: str) -> bool:
        """创作模式：触发影库 + 乐库搜索"""
        return self._match_any(query, self._CREATIVE_KW)

    def _should_search_photos(self, query: str) -> bool:
        """个人照片查找模式：仅当用户明确在翻自己的相册"""
        return self._match_any(query, self._PHOTO_FIND_KW)

    def search_all(self, query: str, n: int = 5) -> list[dict]:
        """跨库语义路由 —— 按用户意图只搜相关库"""
        results = []

        # ── 档位 1：对话记忆（永远搜）──
        results.extend(self.search(query, n * 2))

        # ── 档位 2：创作模式 → 影库 + 乐库（绑定）──
        if self._should_search_creative(query):
            try:
                results.extend(self.search_films(query, n))
            except Exception:
                pass
            try:
                results.extend(self.search_music(query, n))
            except Exception:
                pass

        # ── 档位 3：个人照片查找 → 仅当明确在翻相册 ──
        if self._should_search_photos(query):
            try:
                results.extend(self.search_photos(query, n))
            except Exception:
                pass

        # 去重：同一个 source 只保留第一条
        seen_sources = set()
        deduped = []
        for r in results:
            src = r.get("source", r.get("title", ""))
            if src not in seen_sources:
                seen_sources.add(src)
                deduped.append(r)
        return deduped[:n]

    @property
    def count(self) -> int:
        return self.collection.count()
