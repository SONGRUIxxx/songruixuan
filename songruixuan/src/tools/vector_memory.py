"""🧠 向量记忆 — 内容单元 ID 检测 + 增量索引"""

import hashlib
import yaml
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions


_ef = None


def _get_ef():
    """延迟加载模型"""
    global _ef
    if _ef is None:
        _ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="BAAI/bge-small-zh-v1.5",
        )
    return _ef


class VectorMemory:
    """语义记忆索引 — 按内容单元检测，只索引新增部分"""

    CS, OL = 800, 200  # 分块大小 & 重叠（800 字 ≈ BGE 512 token 上限）

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
        if self._film_collection is None:
            self._film_collection = self.client.get_or_create_collection(
                name="film_library_daily",
                embedding_function=_get_ef(),
                metadata={"hnsw:space": "cosine"},
            )
        return self._film_collection

    def get_or_create_music_collection(self):
        if self._music_collection is None:
            self._music_collection = self.client.get_or_create_collection(
                name="music_library_weekly",
                embedding_function=_get_ef(),
                metadata={"hnsw:space": "cosine"},
            )
        return self._music_collection

    def get_or_create_photo_collection(self):
        if self._photo_collection is None:
            self._photo_collection = self.client.get_or_create_collection(
                name="photo_library_continually",
                embedding_function=_get_ef(),
                metadata={"hnsw:space": "cosine"},
            )
        return self._photo_collection

    # ── 核心：分块 + 检测 ──────────────────────────

    def _chunk_exists(self, collection, prefix: str) -> bool:
        """检查集合中是否存在以 prefix 开头的条目（前缀匹配）"""
        try:
            all_ids = collection.get(include=[])["ids"]
            return any(rid.startswith(f"{prefix}_chunk_") for rid in all_ids)
        except Exception:
            return False

    def _index_chunks(self, collection, text: str, prefix: str, metadata: dict,
                      source: str = "", title: str = "") -> int:
        """将文本分块嵌入到 collection，ID 格式 {prefix}_chunk_{n}__{hash}"""
        cs, ol = self.CS, self.OL
        step = cs - ol
        indexed = 0

        # 清除该 prefix 的旧记录
        exist_ids = collection.get()["ids"] if collection.count() > 0 else []
        for oid in list(exist_ids):
            if oid.startswith(f"{prefix}_chunk_"):
                collection.delete(ids=[oid])

        if len(text) <= cs:
            chunk_hash = hashlib.md5(text[:200].encode()).hexdigest()[:8]
            cid = f"{prefix}_chunk_0__{chunk_hash}"
            meta = dict(metadata)
            meta["chunk_index"] = 0
            meta["preview"] = text
            if source:
                meta["source"] = source
            if title:
                meta["title"] = title
            collection.add(ids=[cid], documents=[text], metadatas=[meta])
            indexed += 1
        else:
            for i, p in enumerate(range(0, len(text), step)):
                chunk = text[p:p + cs]
                if not chunk.strip():
                    continue
                chunk_hash = hashlib.md5(chunk[:200].encode()).hexdigest()[:8]
                cid = f"{prefix}_chunk_{i}__{chunk_hash}"
                meta = dict(metadata)
                meta["chunk_index"] = i
                meta["preview"] = chunk
                if source:
                    meta["source"] = source
                if title:
                    meta["title"] = title
                collection.add(ids=[cid], documents=[chunk], metadatas=[meta])
                indexed += 1
        return indexed

    # ── 搜索 ──────────────────────────────────────

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
                preview = meta.get("preview", meta.get("desc", ""))
                source_path = meta.get("source", "")
                memories.append({
                    "title": meta.get("title", meta.get("film", meta.get("album", ""))),
                    "preview": preview,
                    "full_content": preview,  # yaml source: preview is enough
                    "source": source_path,
                })
        return memories

    # ── 对话记忆索引 ──────────────────────────────

    def index_conversations(self) -> int:
        """读 conversations.yaml，按 session ID 检测，新增则分块嵌入"""
        yaml_path = self.memory_dir / "conversations.yaml"
        if not yaml_path.exists():
            return 0

        try:
            data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
        except Exception:
            return 0

        indexed = 0
        source = str(yaml_path)
        for session in data.get("sessions", []):
            sid = session.get("id", 0)
            prefix = f"session_{sid}"
            if self._chunk_exists(self.collection, prefix):
                continue
            transcript = session.get("transcript", "")
            if not transcript.strip():
                continue
            indexed += self._index_chunks(
                self.collection, transcript, prefix=prefix,
                metadata={"session_id": sid},
                source=source, title=f"第{sid}次对话",
            )
        return indexed

    # ── 电影笔记索引 ──────────────────────────────

    def _notes_yaml_path(self) -> Path | None:
        p = self.memory_dir.parent / "film_library_daily" / "notes.yaml"
        return p if p.exists() else None

    def index_notes(self) -> int:
        """读 notes.yaml，按电影名检测，新增则分块嵌入"""
        yp = self._notes_yaml_path()
        if yp is None:
            return 0

        try:
            data = yaml.safe_load(yp.read_text(encoding="utf-8")) or {}
        except Exception:
            return 0

        indexed = 0
        source = str(yp)
        # 用 conversations collection（和对话记忆放一起搜）
        for film in data.get("films", []):
            name = film.get("name", "")
            content = film.get("content", "")
            if not name or not content.strip():
                continue
            prefix = f"film_note_{name}"
            if self._chunk_exists(self.collection, prefix):
                continue
            indexed += self._index_chunks(
                self.collection, content, prefix=prefix,
                metadata={"film": name, "type": "film_note"},
                source=source, title=f"《{name}》笔记",
            )
        return indexed

    def search_notes(self, query: str, n: int = 5) -> list[dict]:
        """搜电影笔记（存在 conversations collection 里）"""
        # 过滤只返回 type=film_note 的结果
        if self.collection.count() == 0:
            return []
        results = self.collection.query(
            query_texts=[query],
            n_results=min(n * 3, self.collection.count()),
            where={"type": "film_note"},
        )
        memories = []
        if results["metadatas"] and results["metadatas"][0]:
            for meta in results["metadatas"][0]:
                memories.append({
                    "title": meta.get("title", meta.get("film", "")),
                    "preview": meta.get("preview", ""),
                    "full_content": meta.get("preview", ""),
                    "source": meta.get("source", ""),
                })
        return memories[:n]

    def search(self, query: str, n: int = 5) -> list[dict]:
        """搜对话记忆（排除 film_note 类型）"""
        if self.collection.count() == 0:
            return []
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=min(n * 2, self.collection.count()),
                where={"type": {"$ne": "film_note"}},
            )
        except Exception:
            # 旧 ChromaDB 可能不支持 $ne 过滤，fallback
            return self._search_col(self.collection, query, n)

        memories = []
        if results["metadatas"] and results["metadatas"][0]:
            for meta in results["metadatas"][0]:
                memories.append({
                    "title": meta.get("title", ""),
                    "preview": meta.get("preview", ""),
                    "full_content": meta.get("preview", ""),
                    "source": meta.get("source", ""),
                })
        return memories[:n]

    def add_memory(self, title: str, content: str):
        """添加记忆到 conversations collection"""
        full_text = f"# {title}\n\n{content}"
        prefix = f"mem_{title}"
        self._index_chunks(
            self.collection, full_text, prefix=prefix,
            metadata={"title": title}, title=title,
        )

    # ── 影库索引 ──────────────────────────────────

    def _index_yaml_items(self, collection, yaml_path: Path, items_key: str,
                           prefix_fn, meta_fn, source: str) -> int:
        """通用的 yaml 条目索引：逐条检测 ChromaDB ID，新增则分块嵌入"""
        if not yaml_path.exists():
            return 0
        try:
            data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
        except Exception:
            return 0

        exist_ids = set(collection.get()["ids"]) if collection.count() > 0 else set()
        indexed = 0

        for item in data.get(items_key, []):
            prefix = prefix_fn(item)
            full_text = meta_fn(item)
            if not full_text:
                continue

            # 检查是否存在（前缀匹配）
            already = any(oid.startswith(prefix) for oid in exist_ids)
            if already:
                continue

            # 清除旧记录
            for old_id in list(exist_ids):
                if old_id.startswith(prefix):
                    collection.delete(ids=[old_id])
                    exist_ids.discard(old_id)

            indexed += self._index_chunks(
                collection, full_text, prefix=prefix,
                metadata={}, source=source,
            )
        return indexed

    def index_film_library(self) -> int:
        idx_file = self.memory_dir.parent / "film_library_daily" / "index.yaml"
        if not idx_file.exists():
            return 0

        fc = self.get_or_create_film_collection()

        def _prefix(fr):
            return f"{fr.get('film','')}__{fr.get('file','')}"

        def _text(fr):
            film = fr.get("film", "")
            fname = fr.get("file", "")
            desc = fr.get("desc", "")
            return f"《{film}》{fname}: {desc}" if desc else ""

        return self._index_yaml_items(fc, idx_file, "frames", _prefix, _text, str(idx_file))

    def search_films(self, query: str, n: int = 5) -> list[dict]:
        return self._search_col(self.get_or_create_film_collection(), query, n)

    # ── 乐库索引 ──────────────────────────────────

    def index_music_library(self) -> int:
        idx_file = self.memory_dir.parent / "music_library_weekly" / "index.yaml"
        if not idx_file.exists():
            return 0

        mc = self.get_or_create_music_collection()

        def _prefix(tr):
            tname = tr.get("file", tr.get("title", ""))
            return f"track_{tname}"

        def _text(tr):
            tname = tr.get("file", tr.get("title", ""))
            desc = tr.get("desc", "")
            return f"{tname}: {desc}" if desc else ""

        return self._index_yaml_items(mc, idx_file, "tracks", _prefix, _text, str(idx_file))

    def search_music(self, query: str, n: int = 5) -> list[dict]:
        return self._search_col(self.get_or_create_music_collection(), query, n)

    # ── 摄影库索引 ────────────────────────────────

    def index_photo_library(self) -> int:
        idx_file = self.memory_dir.parent / "photo_library_continually" / "index.yaml"
        if not idx_file.exists():
            return 0

        pc = self.get_or_create_photo_collection()

        def _prefix(p):
            return f"photo_{p.get('file','')}"

        def _text(p):
            fname = p.get("file", "")
            desc = p.get("desc", "")
            return f"{fname}: {desc}" if desc else ""

        return self._index_yaml_items(pc, idx_file, "photos", _prefix, _text, str(idx_file))

    def search_photos(self, query: str, n: int = 5) -> list[dict]:
        return self._search_col(self.get_or_create_photo_collection(), query, n)

    # ══════════════════════════════════════════════════
    # 语义路由
    # ══════════════════════════════════════════════════

    _CREATIVE_KW = [
        "电影", "影片", "影评", "看电影", "观影", "看什么", "看了",
        "胶片", "上映", "票房", "演员", "编剧", "导演", "剪辑",
        "片尾", "片头", "影库", "film", "movie", "cinema",
        "镜头", "构图", "分镜", "画面", "调色", "色调", "光线",
        "景别", "运镜", "长镜头", "特写", "远景", "近景",
        "配乐", "原声", "soundtrack", "旋律", "编曲", "插曲",
        "主题曲", "背景音乐", "bgm", "ost", "音乐", "歌曲",
        "乐库", "听什么", "专辑", "唱片", "音乐人", "歌手",
        "学习", "分析", "参考", "借鉴", "致敬", "灵感",
        "旁白", "文案", "台词", "创作", "写", "脚本", "稿子",
    ]

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
        q = query.lower()
        return any(kw.lower() in q for kw in keywords)

    def _should_search_creative(self, query: str) -> bool:
        return self._match_any(query, self._CREATIVE_KW)

    def _should_search_photos(self, query: str) -> bool:
        return self._match_any(query, self._PHOTO_FIND_KW)

    def search_all(self, query: str, n: int = 5) -> list[dict]:
        """跨库语义路由"""
        results = []

        # 档位 1：对话记忆（永远搜）
        results.extend(self.search(query, n * 2))

        # 档位 2：创作模式 → 影库 + 乐库 + 电影笔记
        if self._should_search_creative(query):
            try:
                results.extend(self.search_notes(query, n))
            except Exception:
                pass
            try:
                results.extend(self.search_films(query, n))
            except Exception:
                pass
            try:
                results.extend(self.search_music(query, n))
            except Exception:
                pass

        # 档位 3：个人照片
        if self._should_search_photos(query):
            try:
                results.extend(self.search_photos(query, n))
            except Exception:
                pass

        # 去重
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
