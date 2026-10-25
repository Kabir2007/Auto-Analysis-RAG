"""In-memory retrieval for shared guidance and user-scoped dataset context."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional

import chromadb

# Dataset schemas and conversational memory are deliberately process-local. Each
# dataset collection is keyed by the authenticated owner and dataset UUID.
_CLIENT = chromadb.EphemeralClient()
KB_COLLECTION_NAME = "kb_guide"
_KB_INDEXED = False


def _get_client():
    return _CLIENT


def _safe_collection_name(name: str) -> str:
    cleaned = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in name)
    cleaned = cleaned.strip("_-")
    if len(cleaned) < 3:
        cleaned = f"col_{cleaned}"
    return (cleaned if len(cleaned) >= 3 else "col_data")[:63]


def _dataset_collection_name(dataset_id: str, user_id: str) -> str:
    key = hashlib.sha256(f"{user_id}:{dataset_id}".encode()).hexdigest()[:32]
    return f"ds_{key}"


def has_dataset(dataset_id: str, *, user_id: str) -> bool:
    try:
        _CLIENT.get_collection(name=_dataset_collection_name(dataset_id, user_id))
        return True
    except Exception:
        return False


def index_kb(path: str = "kb/stats_guide.md") -> int:
    """Index shared, non-user-specific statistical guidance once per process."""
    global _KB_INDEXED
    if _KB_INDEXED:
        return 0
    kb_path = Path(path)
    if not kb_path.exists():
        return 0
    sections = kb_path.read_text(encoding="utf-8").split("\n## ")
    docs, ids, metadatas = [], [], []
    for i, sec in enumerate(sections):
        if not sec.strip():
            continue
        lines = sec.strip().split("\n", 1)
        title = lines[0].replace("#", "").strip()
        body = lines[1].strip() if len(lines) > 1 else title
        chunk = f"## {title}\n{body}"
        docs.append(chunk)
        ids.append(hashlib.sha1(chunk.encode()).hexdigest()[:16])
        metadatas.append({"topic": title, "source": "kb"})
    if docs:
        _CLIENT.get_or_create_collection(name=KB_COLLECTION_NAME).upsert(
            documents=docs, ids=ids, metadatas=metadatas
        )
    _KB_INDEXED = True
    return len(docs)


def index_dataset(
    dataset_id: str,
    profile: dict,
    cleaning_log: Optional[list[dict]] = None,
    *,
    user_id: str,
) -> int:
    """Index user-specific dataset structure and cleaning audit information."""
    name = _dataset_collection_name(dataset_id, user_id)
    try:
        _CLIENT.delete_collection(name=name)
    except Exception:
        pass
    collection = _CLIENT.get_or_create_collection(name=name)
    docs, ids, metadata = [], [], []
    shape = profile.get("shape", [0, 0])
    flags = profile.get("flags", [])
    overview = (
        f"Dataset ID: {dataset_id}\nShape: {shape[0]:,} rows, {shape[1]} columns\n"
        f"Duplicate Rows: {profile.get('duplicate_rows', 0)}\nQuality Flags: {len(flags)} detected\n"
    )
    if flags:
        overview += "Flags: " + "; ".join(
            f"{f['column']} ({f['flag']}: {f['detail']})" for f in flags
        )
    docs.append(overview)
    ids.append("overview")
    metadata.append({"source": "dataset", "dataset_id": dataset_id, "chunk_type": "overview"})
    for col, info in profile.get("columns", {}).items():
        text = (
            f"Column: {col}\nDtype: {info.get('dtype')}\n"
            f"Semantic Type: {info.get('semantic_type')}\n"
            f"Missing Share: {info.get('missing_pct', 0)*100:.1f}% ({info.get('missing_count', 0)} missing)\n"
            f"Unique Values: {info.get('n_unique', 0)}\n"
        )
        if info.get("semantic_type") == "numeric":
            text += f"Numeric Stats: min={info.get('min')}, mean={info.get('mean')}, max={info.get('max')}, std={info.get('std')}\n"
        elif info.get("semantic_type") == "categorical":
            text += f"Top Values: {info.get('top_values')}\n"
        elif info.get("semantic_type") == "datetime":
            text += f"Date Range: {info.get('min_date')} to {info.get('max_date')}\n"
        docs.append(text)
        ids.append(f"column_{hashlib.sha1(str(col).encode()).hexdigest()[:16]}")
        metadata.append({"source": "dataset", "dataset_id": dataset_id, "chunk_type": "column", "column": str(col)})
    if cleaning_log:
        log_text = "Cleaning Log:\n" + "\n".join(
            f"- [{e.get('action')}] {e.get('column')}: {e.get('detail')} ({e.get('rows_affected')} rows affected)"
            for e in cleaning_log
        )
        docs.append(log_text)
        ids.append("cleaning_log")
        metadata.append({"source": "dataset", "dataset_id": dataset_id, "chunk_type": "cleaning_log"})
    collection.add(documents=docs, ids=ids, metadatas=metadata)
    return len(docs)


def add_memory(dataset_id: str, question: str, summary: str, *, user_id: str) -> None:
    collection = _CLIENT.get_or_create_collection(name=_dataset_collection_name(dataset_id, user_id))
    mem_id = "mem_" + hashlib.sha1((question + summary).encode()).hexdigest()[:24]
    collection.upsert(
        documents=[f"Past Query: {question}\nFinding Summary: {summary[:400]}"],
        ids=[mem_id],
        metadatas=[{"source": "memory", "dataset_id": dataset_id}],
    )


def retrieve(
    dataset_id: str,
    query: str,
    k_ds: int = 4,
    k_kb: int = 2,
    k_mem: int = 2,
    *,
    user_id: str,
) -> list[dict]:
    results: list[dict] = []
    try:
        collection = _CLIENT.get_collection(name=_dataset_collection_name(dataset_id, user_id))
        for source, limit in (("dataset", k_ds), ("memory", k_mem)):
            if limit <= 0:
                continue
            found = collection.query(query_texts=[query], n_results=limit, where={"source": source})
            if found and found.get("documents") and found["documents"][0]:
                for doc, distance in zip(found["documents"][0], found["distances"][0]):
                    results.append({"source": source, "text": doc, "distance": float(distance)})
    except Exception:
        pass
    try:
        collection = _CLIENT.get_collection(name=KB_COLLECTION_NAME)
        found = collection.query(query_texts=[query], n_results=k_kb)
        if found and found.get("documents") and found["documents"][0]:
            for doc, distance in zip(found["documents"][0], found["distances"][0]):
                results.append({"source": "kb", "text": doc, "distance": float(distance)})
    except Exception:
        pass
    return results


def delete_dataset(dataset_id: str, *, user_id: str) -> None:
    """Remove only the authenticated user's in-memory context for this dataset."""
    try:
        _CLIENT.delete_collection(name=_dataset_collection_name(dataset_id, user_id))
    except Exception:
        pass
