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
                name="music_library_daily",
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
                    "full_content": preview,
                    "source": source_path,
                    "session_id": meta.get("session_id", 0),
                    "chunk_index": meta.get("chunk_index", 0),
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

    # ── 音乐笔记索引 ──────────────────────────────

    def index_music_notes(self) -> int:
        """读 music_library_daily/notes.yaml，按曲目名检测，新增则分块嵌入"""
        yp = self.memory_dir.parent / "music_library_daily" / "notes.yaml"
        if not yp.exists():
            return 0

        try:
            data = yaml.safe_load(yp.read_text(encoding="utf-8")) or {}
        except Exception:
            return 0

        indexed = 0
        source = str(yp)
        for track in data.get("tracks", []):
            name = track.get("name", "")
            content = track.get("content", "")
            if not name or not content.strip():
                continue
            prefix = f"music_note_{name}"
            if self._chunk_exists(self.collection, prefix):
                continue
            indexed += self._index_chunks(
                self.collection, content, prefix=prefix,
                metadata={"music": name, "type": "music_note"},
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
                    "chunk_index": meta.get("chunk_index", 0),
                    "group_id": f"film_note_{meta.get('film', '')}",
                })
        return memories[:n]

    def search_music_notes(self, query: str, n: int = 5) -> list[dict]:
        """搜音乐笔记（存在 conversations collection 里）"""
        if self.collection.count() == 0:
            return []
        results = self.collection.query(
            query_texts=[query],
            n_results=min(n * 3, self.collection.count()),
            where={"type": "music_note"},
        )
        memories = []
        if results["metadatas"] and results["metadatas"][0]:
            for meta in results["metadatas"][0]:
                memories.append({
                    "title": meta.get("title", meta.get("music", "")),
                    "preview": meta.get("preview", ""),
                    "full_content": meta.get("preview", ""),
                    "source": meta.get("source", ""),
                    "chunk_index": meta.get("chunk_index", 0),
                    "group_id": f"music_note_{meta.get('music', '')}",
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
                where={"$and": [{"type": {"$ne": "film_note"}}, {"type": {"$ne": "music_note"}}]},
            )
        except Exception:
            # 旧 ChromaDB 可能不支持 $ne 过滤，fallback
            return self._search_col(self.collection, query, n)

        memories = []
        if results["metadatas"] and results["metadatas"][0]:
            for meta in results["metadatas"][0]:
                sid = meta.get("session_id", 0)
                memories.append({
                    "title": meta.get("title", ""),
                    "preview": meta.get("preview", ""),
                    "full_content": meta.get("preview", ""),
                    "source": meta.get("source", ""),
                    "session_id": sid,
                    "chunk_index": meta.get("chunk_index", 0),
                    "group_id": f"session_{sid}" if sid else "",
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
        idx_file = self.memory_dir.parent / "music_library_daily" / "index.yaml"
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

    _FILM_KW = ["影库"]

    _MUSIC_KW = ["乐库"]

    _PHOTO_FIND_KW = ["摄影库"]

    @classmethod
    def _match_any(cls, query: str, keywords: list[str]) -> bool:
        q = query.lower()
        return any(kw.lower() in q for kw in keywords)

    def search_all(self, query: str, n: int = 5) -> list[dict]:
        """按库独立路由：提到哪个库就搜哪个"""
        results = []

        # ── 独占路由：提到库则只搜库，没提才搜对话 ──
        has_photo = self._match_any(query, self._PHOTO_FIND_KW)
        has_music = self._match_any(query, self._MUSIC_KW)
        # 摄影库包含"影库"子串 → 互斥，摄影库优先
        has_film = self._match_any(query, self._FILM_KW) and not has_photo
        any_lib = has_film or has_music or has_photo

        # ── 分池配额：各子库内去重后按配额取，避免长文本挤压短条目 ──
        HALF = max(n // 2, 5)  # 每池配额

        def _dedup_pool(items: list[dict], limit: int) -> list[dict]:
            """池内按分块粒度去重，取前 limit 条"""
            s = set()
            out = []
            for r in items:
                sid = r.get("session_id", 0)
                ci = r.get("chunk_index", -1)
                gid = r.get("group_id", "")
                if (sid and ci >= 0) or (gid and ci >= 0):
                    tag = f"session_{sid}" if sid else gid
                    key = (r.get("source", ""), f"{tag}_chunk_{ci}")
                else:
                    key = (r.get("source", ""), r.get("title", ""))
                if key not in s:
                    s.add(key)
                    out.append(r)
            return out[:limit]

        if not any_lib:
            results = _dedup_pool(self.search(query, n * 2), 10)

        if has_film:
            frame_pool = []
            note_pool = []
            cross_notes = []
            try:
                frame_pool = self.search_films(query, n)
                film_names = set()
                for fr in frame_pool:
                    name = fr.get("title", "").split("__")[0]
                    if name:
                        film_names.add(name)
                for fname in film_names:
                    try:
                        cross_notes.extend(self.search_notes(fname, 2))
                    except Exception:
                        pass
            except Exception:
                pass
            try:
                note_pool = self.search_notes(query, n)
            except Exception:
                pass
            # 配额：帧 5 + 笔记 5，交叉关联笔记额外追加
            results = _dedup_pool(frame_pool, HALF) + _dedup_pool(note_pool + cross_notes, HALF)

        if has_music:
            track_pool = []
            note_pool = []
            try:
                track_pool = self.search_music(query, n)
            except Exception:
                pass
            try:
                note_pool = self.search_music_notes(query, n)
            except Exception:
                pass
            results = _dedup_pool(track_pool, HALF) + _dedup_pool(note_pool, HALF)

        if has_photo:
            results = _dedup_pool(self.search_photos(query, n), 10)

        top = results[:10]

        # ── 第二步：对精选的 10 条做上下文扩展（±2 相邻分块）──
        expanded = list(top)

        # 构建所有 _chunk_ 前缀的分块索引（对话 + 影库笔记 + 乐库笔记）
        all_ids = self.collection.get(include=[])["ids"]
        chunk_map: dict[str, list[tuple[int, str]]] = {}
        for rid in all_ids:
            if "_chunk_" not in rid:
                continue
            prefix, rest = rid.split("_chunk_", 1)
            try:
                ci = int(rest.split("__")[0])
            except (ValueError, IndexError):
                continue
            chunk_map.setdefault(prefix, []).append((ci, rid))

        for r in top:
            ci = r.get("chunk_index", -1)
            if ci < 0:
                continue
            prefix = None
            sid = r.get("session_id", 0)
            gid = r.get("group_id", "")
            if sid:
                prefix = f"session_{sid}"
            elif gid:
                prefix = gid
            if not prefix:
                continue

            for nci, nrid in chunk_map.get(prefix, []):
                if 1 <= abs(nci - ci) <= 2:  # ±2
                    try:
                        n_data = self.collection.get(ids=[nrid], include=["metadatas", "documents"])
                        if n_data["metadatas"]:
                            nm = n_data["metadatas"][0]
                            ndoc = n_data["documents"][0] if n_data["documents"] else ""
                            expanded.append({
                                "title": nm.get("title", r["title"]),
                                "preview": ndoc or nm.get("preview", ""),
                                "full_content": ndoc or nm.get("preview", ""),
                                "source": nm.get("source", r["source"]),
                                "session_id": nm.get("session_id", sid),
                                "chunk_index": nci,
                                "group_id": gid,
                            })
                    except Exception:
                        pass

        # ── 第三步：最终去重（按分块粒度），不截断，全部返回 ──
        seen2 = set()
        final = []
        for r in expanded:
            sid = r.get("session_id", 0)
            ci = r.get("chunk_index", -1)
            gid = r.get("group_id", "")
            if (sid and ci >= 0) or (gid and ci >= 0):
                tag = f"session_{sid}" if sid else gid
                key = (r.get("source", ""), f"{tag}_chunk_{ci}")
            else:
                key = (r.get("source", ""), r.get("title", ""))
            if key not in seen2:
                seen2.add(key)
                final.append(r)
        return final

    @property
    def count(self) -> int:
        return self.collection.count()
