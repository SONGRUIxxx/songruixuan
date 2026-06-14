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
                source_path = meta.get("source", "")
                preview = meta.get("preview", meta.get("desc", ""))
                # 尝试回读源文件获取完整内容
                full_content = ""
                if source_path:
                    sp = Path(source_path)
                    if sp.exists() and sp.suffix == ".md":
                        # 聚合文件（如 journal.md）可能巨大，只用匹配到的分块；
                        # 独立小文件（单次会话）完整读取
                        if sp.stat().st_size < 10000:
                            try:
                                full_content = sp.read_text(encoding="utf-8")
                            except Exception:
                                full_content = preview
                        else:
                            full_content = preview
                    elif sp.suffix in (".yaml", ".yml"):
                        full_content = preview
                    else:
                        full_content = preview
                else:
                    full_content = preview

                memories.append({
                    "title": meta.get("title", meta.get("film", meta.get("album", ""))),
                    "preview": preview,
                    "full_content": full_content,
                    "source": source_path,
                })
        return memories

    def _file_changed(self, filepath: Path) -> bool:
        """检查文件自上次索引后是否有变更（用 mtime+size 摘要）"""
        key = f"__meta__{filepath.name}"
        current = self._file_hash(filepath)
        # 从任意 collection 读 meta（用 conversations 兜底）
        results = self.collection.get(ids=[key])
        if results["ids"] and results["metadatas"]:
            stored = results["metadatas"][0].get("hash", "")
            return stored != current
        return True  # 没存过 → 视为变更

    def _mark_indexed(self, filepath: Path):
        """记录文件已被索引"""
        key = f"__meta__{filepath.name}"
        current = self._file_hash(filepath)
        self.collection.upsert(
            ids=[key],
            documents=[f"index meta: {filepath.name}"],
            metadatas=[{"hash": current, "file": str(filepath)}],
        )

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

        # 2. 索引 journal.md：增量追加（文件变大只处理新增部分）
        journal = self.memory_dir / "journal.md"
        if journal.exists():
            key = "__meta__journal.md"
            cs, ol = 2000, 500
            step = cs - ol  # 1500

            content = journal.read_text(encoding="utf-8")
            current_size = len(content)

            # 读取上次索引时的文件大小
            results = self.collection.get(ids=[key])
            last_size = 0
            if results["ids"] and results["metadatas"]:
                last_size = results["metadatas"][0].get("last_size", 0)

            if current_size == last_size:
                pass  # 没变，跳过
            elif current_size < last_size:
                # 文件缩小（手动删了内容）→ 全量重建
                for oid in list(existing_ids):
                    if oid.startswith("j_"):
                        self.collection.delete(ids=[oid])
                start_pos = 0
                start_chunk = 0
            else:
                # 文件增大 → 增量：从 last_size - cs 处开始，覆盖可能断裂的最后一个分块
                start_chunk = max(0, (last_size - cs) // step)
                start_pos = start_chunk * step
                # 删除受影响的旧分块
                for oid in list(existing_ids):
                    if oid.startswith("j_"):
                        try:
                            idx_str = oid.split("_")[1]
                            idx_str = idx_str.split("__")[0]
                            if int(idx_str) >= start_chunk:
                                self.collection.delete(ids=[oid])
                        except (ValueError, IndexError):
                            pass

            if current_size != last_size:
                for i, p in enumerate(range(start_pos, current_size, step)):
                    chunk = content[p:p + cs]
                    if not chunk.strip():
                        continue
                    chunk_idx = start_chunk + i
                    chroma_id = f"j_{chunk_idx}__{hashlib.md5(chunk[:200].encode()).hexdigest()[:8]}"
                    self.collection.add(
                        ids=[chroma_id],
                        documents=[chunk[:1200]],
                        metadatas=[{"source": str(journal), "title": f"journal-{chunk_idx+1}", "preview": chunk[:800]}],
                    )
                    indexed += 1

                # 记录本次索引状态
                self.collection.upsert(
                    ids=[key],
                    documents=["index meta: journal.md"],
                    metadatas=[{"hash": self._file_hash(journal), "file": str(journal), "last_size": current_size}],
                )

        return indexed

    def _index_one_file(self, md_file, doc_id, existing_ids):
        """索引单个 .md 文件：内容短则存一条，长则 2000/500 滑动窗口分块"""
        current_hash = self._file_hash(md_file)
        # 清除旧记录（可能之前是单条，现在要分块）
        for old_id in list(existing_ids):
            if old_id.startswith(f"{doc_id}__") or old_id.startswith(f"{doc_id}_chunk_"):
                self.collection.delete(ids=[old_id])
                existing_ids.discard(old_id)
        content = md_file.read_text(encoding="utf-8")
        cs, ol = 2000, 500
        if len(content) <= cs:
            chroma_id = f"{doc_id}__{current_hash}"
            self.collection.add(
                ids=[chroma_id],
                documents=[content[:1200]],
                metadatas=[{"source": str(md_file), "title": doc_id, "preview": content[:800]}],
            )
        else:
            for i, p in enumerate(range(0, len(content), cs - ol)):
                chunk = content[p:p + cs]
                if not chunk.strip():
                    continue
                c_hash = hashlib.md5(chunk[:200].encode()).hexdigest()[:8]
                chroma_id = f"{doc_id}_chunk_{i}__{c_hash}"
                self.collection.add(
                    ids=[chroma_id],
                    documents=[chunk[:1200]],
                    metadatas=[{"source": str(md_file), "title": f"{doc_id}[{i}]", "preview": chunk[:800], "chunk_index": i}],
                )

    def search(self, query: str, n: int = 5) -> list[dict]:
        """搜对话记忆"""
        return self._search_col(self.collection, query, n)

    def add_memory(self, title: str, content: str):
        """添加记忆：内容短则存一条，长则 2000/500 滑动窗口分块"""
        full_text = f"# {title}\n\n{content}"
        cs, ol = 2000, 500
        if len(full_text) <= cs:
            chroma_id = f"{title}__{self._text_hash(content)}"
            self.collection.add(
                ids=[chroma_id],
                documents=[full_text[:1200]],
                metadatas=[{"title": title, "preview": content[:800]}],
            )
        else:
            for i, p in enumerate(range(0, len(full_text), cs - ol)):
                chunk = full_text[p:p + cs]
                if not chunk.strip():
                    continue
                c_hash = hashlib.md5(chunk[:200].encode()).hexdigest()[:8]
                chroma_id = f"{title}_chunk_{i}__{c_hash}"
                self.collection.add(
                    ids=[chroma_id],
                    documents=[chunk[:1200]],
                    metadatas=[{"title": f"{title}[{i}]", "preview": chunk[:800], "chunk_index": i}],
                )

    # ── 影库索引 ──────────────────────────────────

    def index_film_library(self) -> int:
        """扫描扁平影库：读根目录 index.yaml，长 desc 分块索引"""
        film_dir = self.memory_dir.parent / "film_library_daily"
        if not film_dir.exists():
            return 0

        idx_file = film_dir / "index.yaml"
        if not idx_file.exists():
            return 0

        # 文件未变则跳过全量重建
        if not self._file_changed(idx_file):
            return 0

        import yaml
        try:
            data = yaml.safe_load(idx_file.read_text(encoding="utf-8")) or {}
        except Exception:
            return 0

        fc = self.get_or_create_film_collection()
        exist_ids = set(fc.get()["ids"]) if fc.count() > 0 else set()
        indexed = 0
        cs, ol = 2000, 500

        for fr in data.get("frames", []):
            fname = fr.get("file", "")
            film = fr.get("film", "")
            desc = fr.get("desc", "")
            if not desc or not fname:
                continue
            doc_id = f"{film}__{fname}"
            full_text = f"《{film}》{fname}: {desc}"

            # 清除旧记录
            for old_id in list(exist_ids):
                if old_id.startswith(doc_id):
                    fc.delete(ids=[old_id])
                    exist_ids.discard(old_id)

            if len(full_text) <= cs:
                f_hash = hashlib.md5(desc[:200].encode()).hexdigest()[:12]
                chroma_id = f"{doc_id}__{f_hash}"
                fc.add(
                    ids=[chroma_id],
                    documents=[full_text[:1200]],
                    metadatas=[{"film": film, "frame": fname, "desc": desc[:800], "source": str(idx_file)}],
                )
                indexed += 1
            else:
                for i, p in enumerate(range(0, len(full_text), cs - ol)):
                    chunk = full_text[p:p + cs]
                    if not chunk.strip():
                        continue
                    c_hash = hashlib.md5(chunk[:200].encode()).hexdigest()[:8]
                    chroma_id = f"{doc_id}_chunk_{i}__{c_hash}"
                    fc.add(
                        ids=[chroma_id],
                        documents=[chunk[:1200]],
                        metadatas=[{"film": film, "frame": fname, "desc": chunk[:800], "source": str(idx_file), "chunk_index": i}],
                    )
                    indexed += 1

        self._mark_indexed(idx_file)
        return indexed

    def search_films(self, query: str, n: int = 5) -> list[dict]:
        """搜影库"""
        return self._search_col(self.get_or_create_film_collection(), query, n)

    # ── 乐库索引 ──────────────────────────────────

    def index_music_library(self) -> int:
        """扫描扁平乐库：读根目录 index.yaml，长 desc 分块索引"""
        music_dir = self.memory_dir.parent / "music_library_weekly"
        if not music_dir.exists():
            return 0

        idx_file = music_dir / "index.yaml"
        if not idx_file.exists():
            return 0

        # 文件未变则跳过全量重建
        if not self._file_changed(idx_file):
            return 0

        import yaml
        try:
            data = yaml.safe_load(idx_file.read_text(encoding="utf-8")) or {}
        except Exception:
            return 0

        mc = self.get_or_create_music_collection()
        exist_ids = set(mc.get()["ids"]) if mc.count() > 0 else set()
        indexed = 0
        cs, ol = 2000, 500

        for tr in data.get("tracks", []):
            tname = tr.get("file", tr.get("title", ""))
            album = tr.get("album", "")
            desc = tr.get("desc", "")
            if not desc or not tname:
                continue
            doc_id = f"{album}__{tname}"
            full_text = f"《{album}》{tname}: {desc}"

            # 清除旧记录
            for old_id in list(exist_ids):
                if old_id.startswith(doc_id):
                    mc.delete(ids=[old_id])
                    exist_ids.discard(old_id)

            if len(full_text) <= cs:
                t_hash = hashlib.md5(desc[:200].encode()).hexdigest()[:12]
                chroma_id = f"{doc_id}__{t_hash}"
                mc.add(
                    ids=[chroma_id],
                    documents=[full_text[:1200]],
                    metadatas=[{"album": album, "track": tname, "desc": desc[:800], "source": str(idx_file)}],
                )
                indexed += 1
            else:
                for i, p in enumerate(range(0, len(full_text), cs - ol)):
                    chunk = full_text[p:p + cs]
                    if not chunk.strip():
                        continue
                    c_hash = hashlib.md5(chunk[:200].encode()).hexdigest()[:8]
                    chroma_id = f"{doc_id}_chunk_{i}__{c_hash}"
                    mc.add(
                        ids=[chroma_id],
                        documents=[chunk[:1200]],
                        metadatas=[{"album": album, "track": tname, "desc": chunk[:800], "source": str(idx_file), "chunk_index": i}],
                    )
                    indexed += 1

        self._mark_indexed(idx_file)
        return indexed

    def search_music(self, query: str, n: int = 5) -> list[dict]:
        """搜乐库"""
        return self._search_col(self.get_or_create_music_collection(), query, n)

    # ── 摄影库索引 ────────────────────────────────

    def index_photo_library(self) -> int:
        """扫描扁平摄影库：读根目录 index.yaml，长 desc 分块索引"""
        photo_dir = self.memory_dir.parent / "photo_library_continually"
        if not photo_dir.exists():
            return 0
        idx_file = photo_dir / "index.yaml"
        if not idx_file.exists():
            return 0
        # 文件未变则跳过全量重建
        if not self._file_changed(idx_file):
            return 0
        import yaml
        try:
            data = yaml.safe_load(idx_file.read_text(encoding="utf-8")) or {}
        except Exception:
            return 0
        pc = self.get_or_create_photo_collection()
        exist_ids = set(pc.get()["ids"]) if pc.count() > 0 else set()
        indexed = 0
        cs, ol = 2000, 500
        for p in data.get("photos", []):
            fname = p.get("file", "")
            desc = p.get("desc", "")
            if not desc or not fname:
                continue
            pid = f"photo_{fname}"
            full_text = f"{fname}: {desc}"

            # 清除旧记录
            for old_id in list(exist_ids):
                if old_id.startswith(pid):
                    pc.delete(ids=[old_id])
                    exist_ids.discard(old_id)

            if len(full_text) <= cs:
                ph = hashlib.md5(desc[:200].encode()).hexdigest()[:8]
                chroma_id = f"{pid}__{ph}"
                pc.add(ids=[chroma_id], documents=[full_text[:1200]],
                       metadatas=[{"file": fname, "desc": desc[:800], "source": str(idx_file)}])
                indexed += 1
            else:
                for i, p_off in enumerate(range(0, len(full_text), cs - ol)):
                    chunk = full_text[p_off:p_off + cs]
                    if not chunk.strip():
                        continue
                    c_hash = hashlib.md5(chunk[:200].encode()).hexdigest()[:8]
                    chroma_id = f"{pid}_chunk_{i}__{c_hash}"
                    pc.add(ids=[chroma_id], documents=[chunk[:1200]],
                           metadatas=[{"file": fname, "desc": chunk[:800], "source": str(idx_file), "chunk_index": i}])
                    indexed += 1
        self._mark_indexed(idx_file)
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
