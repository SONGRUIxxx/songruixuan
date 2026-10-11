"""🧠 向量记忆 — 内容单元 ID 检测 + 增量索引"""

import hashlib
import yaml
from pathlib import Path
from typing import List

import chromadb
from chromadb.api.types import EmbeddingFunction as ChromaEmbeddingFunction
import httpx


# ══════════════════════════════════════════════════
# 火山引擎多模态 Embedding Function
# ══════════════════════════════════════════════════

class VolcanoMultimodalEmbedding(ChromaEmbeddingFunction):
    """适配火山引擎 /v3/embeddings/multimodal 端点"""

    def __init__(self, api_key: str, model_name: str, dimensions: int = 1024):
        self.api_key = api_key
        self.model_name = model_name
        self.dimensions = dimensions
        self.endpoint = "https://ark.cn-beijing.volces.com/api/v3/embeddings/multimodal"

    def __call__(self, input: List[str]) -> List[List[float]]:
        multimodal_input = [{"type": "text", "text": text} for text in input]
        last_err = None

        for attempt in range(3):
            try:
                resp = httpx.post(
                    self.endpoint,
                    json={
                        "model": self.model_name,
                        "input": multimodal_input,
                        "dimensions": self.dimensions,
                        "encoding_format": "float",
                    },
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    timeout=60,
                )
                resp.raise_for_status()
                data = resp.json()
                break
            except Exception as e:
                last_err = e
                if attempt < 2:
                    import time
                    time.sleep(2 * (attempt + 1))
        else:
            raise RuntimeError(f"火山 embedding 请求失败（重试3次）: {last_err}")

        # 单条返回 {embedding, object}，多条返回 [{index, embedding}, ...]
        inner = data.get("data", [])
        if isinstance(inner, dict):
            return [inner["embedding"]]
        if isinstance(inner, list) and inner and isinstance(inner[0], dict):
            items = sorted(inner, key=lambda d: d.get("index", 0))
            return [item["embedding"] for item in items]
        if isinstance(inner, list) and inner and isinstance(inner[0], list):
            return inner
        raise RuntimeError(f"未知的 embedding 返回格式: {type(inner).__name__}")


# ══════════════════════════════════════════════════
# Embedding 配置
# ══════════════════════════════════════════════════

_ef = None
_cloud_ef = None
_cloud_config: dict = {}


def configure_cloud_embedding(api_key: str = "", model_name: str = ""):
    """配置云端 embedding（火山引擎豆包），不传参则回退到本地 BGE"""
    global _cloud_config, _ef, _cloud_ef
    _cloud_config = {"api_key": api_key, "model_name": model_name} if api_key else {}
    _ef = None
    _cloud_ef = None


def _get_ef():
    """获取 embedding function：优先云端豆包，fallback 本地 BGE"""
    global _ef, _cloud_ef
    if _ef is not None:
        return _ef
    if _cloud_config.get("api_key"):
        if _cloud_ef is None:
            _cloud_ef = VolcanoMultimodalEmbedding(
                api_key=_cloud_config["api_key"],
                model_name=_cloud_config["model_name"],
            )
        _ef = _cloud_ef
    else:
        _ef = chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction(
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
        self._photo_collection = None

    # ── Collection 懒加载 ──────────────────────────

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
                    "title": meta.get("title", ""),
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

    def search(self, query: str, n: int = 5) -> list[dict]:
        """搜对话记忆"""
        if self.collection.count() == 0:
            return []
        results = self.collection.query(
            query_texts=[query],
            n_results=min(n * 2, self.collection.count()),
        )

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

    # ── 通用条目索引 ──────────────────────────────

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

    # ── 摄影库索引 ────────────────────────────────

    def index_photo_library(self) -> int:
        idx_file = self.memory_dir.parent / "lens_library" / "index.yaml"
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
    # 按库路由搜索（Web UI 驱动，不走关键词）
    # ══════════════════════════════════════════════════

    def search_libs(self, query: str, libs: list[str], n: int = 20) -> list[dict]:
        """按用户选择的库搜索：分块粒度去重 → 取前 20 → ±2 扩展 → 最终去重 → 全部返回"""
        results = []

        if not libs:
            results = self.search(query, n * 2)
        else:
            for lib in libs:
                if lib == "对话库":
                    results.extend(self.search(query, n))
                elif lib == "镜头库":
                    try: results.extend(self.search_photos(query, n))
                    except: pass

        # 第一步：按分块粒度去重（同文档不同分块都保留），取最贴合的前 20
        seen = set()
        top = []
        for r in results:
            sid = r.get("session_id", 0)
            ci = r.get("chunk_index", -1)
            gid = r.get("group_id", "")
            if (sid and ci >= 0) or (gid and ci >= 0):
                tag = f"session_{sid}" if sid else gid
                key = (r.get("source", ""), f"{tag}_chunk_{ci}")
            else:
                key = (r.get("source", ""), r.get("title", ""))
            if key not in seen:
                seen.add(key)
                top.append(r)
        top = top[:20]  # 只取最贴合的 20 条

        # 第二步：上下文扩展（±2 相邻分块）
        expanded = list(top)
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
                if 1 <= abs(nci - ci) <= 1:
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

        # 第三步：最终去重，全部返回
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

    # ══════════════════════════════════════════════════
    # 语义路由
    # ══════════════════════════════════════════════════

    _PHOTO_FIND_KW = ["图库", "镜头库"]

    @classmethod
    def _match_any(cls, query: str, keywords: list[str]) -> bool:
        q = query.lower()
        return any(kw.lower() in q for kw in keywords)

    def search_all(self, query: str, n: int = 5) -> list[dict]:
        """按库独立路由：提到哪个库就搜哪个"""
        results = []

        # ── 独占路由：提到库则只搜库，没提才搜对话 ──
        has_photo = self._match_any(query, self._PHOTO_FIND_KW)
        any_lib = has_photo

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

        if has_photo:
            results = _dedup_pool(self.search_photos(query, n), 10)

        top = results[:10]

        # ── 第二步：对精选的 10 条做上下文扩展（±2 相邻分块）──
        expanded = list(top)

        # 构建所有 _chunk_ 前缀的分块索引（对话 + 摄影库笔记）
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
                if 1 <= abs(nci - ci) <= 1:  # ±2
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
