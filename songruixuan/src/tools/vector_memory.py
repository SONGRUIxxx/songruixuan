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
        self._material_collection = None
        self._photo_collection = None

    def get_or_create_photo_collection(self):
        """摄影库 collection"""
        if self._photo_collection is None:
            self._photo_collection = self.client.get_or_create_collection(
                name="photo_library_continually",
                embedding_function=_get_ef(),
                metadata={"hnsw:space": "cosine"},
            )
        return self._photo_collection

    def get_or_create_film_collection(self):
        if self._film_collection is None:
            self._film_collection = self.client.get_or_create_collection(
                name="film_library_daily",
                embedding_function=_get_ef(),
                metadata={"hnsw:space": "cosine"},
            )
        return self._film_collection

    def get_or_create_material_collection(self):
        """语料库 collection（懒加载）"""
        if self._material_collection is None:
            self._material_collection = self.client.get_or_create_collection(
                name="material_library_monthly",
                embedding_function=_get_ef(),
                metadata={"hnsw:space": "cosine"},
            )
        return self._material_collection
        """影库 collection（懒加载）"""
        if self._film_collection is None:
            self._film_collection = self.client.get_or_create_collection(
                name="film_library_daily",
                embedding_function=_get_ef(),
                metadata={"hnsw:space": "cosine"},
            )
        return self._film_collection

    def _file_hash(self, filepath: Path) -> str:
        stat = filepath.stat()
        raw = f"{stat.st_mtime}_{stat.st_size}".encode()
        return hashlib.md5(raw).hexdigest()[:12]

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

        # 2. 索引 journal.md：每次启动清空旧索引，从头重建
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

    def search_films(self, query: str, n: int = 5) -> list[dict]:
        """搜影库"""
        fc = self.get_or_create_film_collection()
        return self._search_col(fc, query, n)

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
                    "title": meta.get("title", meta.get("film", "")),
                    "preview": meta.get("preview", meta.get("desc", "")),
                    "source": meta.get("source", meta.get("film", "")),
                })
        return memories

    def add_memory(self, title: str, content: str):
        chroma_id = f"{title}__{self._file_hash_from_text(content)}"
        self.collection.add(
            ids=[chroma_id],
            documents=[f"# {title}\n\n{content}"[:1200]],
            metadatas=[{"title": title, "preview": content[:400]}],
        )

    def _file_hash_from_text(self, text: str) -> str:
        return hashlib.md5(text[:200].encode()).hexdigest()[:12]

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

    def index_material_library(self) -> int:
        """扫描 material_library_monthly 目录，索引 card.md"""
        mat_dir = self.memory_dir.parent / "material_library_monthly"
        if not mat_dir.exists():
            return 0

        mc = self.get_or_create_material_collection()
        exist_ids = set(mc.get()["ids"]) if mc.count() > 0 else set()
        indexed = 0

        for mat_path in sorted(mat_dir.iterdir()):
            if not mat_path.is_dir():
                continue
            card = mat_path / "card.md"
            if not card.exists():
                continue
            card_id = mat_path.name
            card_hash = self._file_hash(card)
            chroma_id = f"{card_id}__{card_hash}"
            if chroma_id in exist_ids:
                continue
            for old in exist_ids:
                if old.startswith(card_id):
                    mc.delete(ids=[old]); break
            content = card.read_text(encoding="utf-8")
            mc.add(ids=[chroma_id], documents=[content[:1200]],
                   metadatas=[{"source": str(card), "title": card_id, "preview": content[:400]}])
            indexed += 1

        return indexed

    def search_materials(self, query: str, n: int = 5) -> list[dict]:
        """搜语料库"""
        mc = self.get_or_create_material_collection()
        return self._search_col(mc, query, n)

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
        pc = self.get_or_create_photo_collection()
        return self._search_col(pc, query, n)

    def search_all(self, query: str, n: int = 5) -> list[dict]:
        """跨库搜索，保证结果多样性（不完全重复同一条记忆）"""
        results = []
        results.extend(self.search(query, n * 2))
        try:
            results.extend(self.search_films(query, n))
        except Exception:
            pass
        try:
            results.extend(self.search_materials(query, n))
        except Exception:
            pass
        try:
            results.extend(self.search_photos(query, n))
        except Exception:
            pass

        # 去重：同一个 source 文件只保留第一条
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
