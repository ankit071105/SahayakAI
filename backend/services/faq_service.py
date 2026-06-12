"""
FAQ Cache Service — Semantic Question Deduplication.

How it works:
  1. Every answered question gets embedded + stored in a separate ChromaDB collection.
  2. Before calling Gemini, we check if a semantically similar question was already answered
     (cosine similarity > 0.92 threshold).
  3. Cache hit → return cached answer instantly (zero Gemini tokens used).
  4. Cache miss → call Gemini, store the new Q&A pair.

This saves ~60-80% of Gemini API calls for repeated/similar questions,
which is crucial for SIH demos (same questions get asked by multiple judges).

Each cache entry stores:
  - question text
  - answer text
  - source_filter (per-document cache isolation)
  - language
  - hit_count (how many times this was served from cache)
  - created_at
"""

import uuid
import json
import datetime
import chromadb
from chromadb.config import Settings
from typing import Optional
import config

_faq_collection = None
FAQ_COLLECTION_NAME = "faq_cache"
SIMILARITY_THRESHOLD = 0.92   # cosine similarity — tune this


def _get_faq_collection():
    global _faq_collection
    if _faq_collection is None:
        client = chromadb.PersistentClient(
            path=config.CHROMA_PERSIST_DIR,
            settings=Settings(anonymized_telemetry=False),
        )
        _faq_collection = client.get_or_create_collection(
            name=FAQ_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _faq_collection


def check_faq_cache(
    query_embedding: list[float],
    source_filter: Optional[str] = None,
    language: str = "en",
) -> Optional[dict]:
    """
    Check if a semantically similar question exists in the FAQ cache.
    Returns the cached answer dict or None.
    """
    col = _get_faq_collection()
    if col.count() == 0:
        return None

    try:
        # ChromaDB requires $and for multiple conditions
        if source_filter:
            where_clause = {"$and": [
                {"language": language},
                {"source_filter": source_filter},
            ]}
        else:
            where_clause = {"language": language}

        results = col.query(
            query_embeddings=[query_embedding],
            n_results=1,
            include=["documents", "metadatas", "distances"],
            where=where_clause,
        )

        if not results["distances"][0]:
            return None

        similarity = 1 - results["distances"][0][0]
        if similarity >= SIMILARITY_THRESHOLD:
            meta = results["metadatas"][0][0]
            doc_id = results["ids"][0][0]

            # Increment hit count
            hit_count = meta.get("hit_count", 0) + 1
            meta["hit_count"] = hit_count
            col.update(ids=[doc_id], metadatas=[meta])

            return {
                "answer":      meta.get("answer", ""),
                "sources":     json.loads(meta.get("sources", "[]")),
                "language":    meta.get("language", language),
                "chunks_used": meta.get("chunks_used", 0),
                "mode":        meta.get("mode", "rag"),
                "from_cache":  True,
                "cache_similarity": round(similarity, 3),
                "cache_hits":  hit_count,
            }
    except Exception as e:
        print(f"⚠️  FAQ cache query error: {e}")
    return None


def store_faq_cache(
    question: str,
    question_embedding: list[float],
    answer: str,
    sources: list[str],
    language: str,
    mode: str,
    chunks_used: int,
    source_filter: Optional[str] = None,
) -> None:
    """Store a new Q&A pair in the FAQ cache."""
    col = _get_faq_collection()
    doc_id = str(uuid.uuid4())
    now = datetime.datetime.utcnow().isoformat()

    meta = {
        "question":      question[:500],
        "answer":        answer[:2000],
        "sources":       json.dumps(sources),
        "language":      language,
        "mode":          mode,
        "chunks_used":   chunks_used,
        "source_filter": source_filter or "",
        "hit_count":     0,
        "created_at":    now,
    }

    try:
        col.add(
            documents=[question],
            embeddings=[question_embedding],
            metadatas=[meta],
            ids=[doc_id],
        )
    except Exception as e:
        print(f"⚠️  FAQ cache store error: {e}")


def get_popular_faqs(limit: int = 10) -> list[dict]:
    """Return the most-hit cached questions."""
    col = _get_faq_collection()
    if col.count() == 0:
        return []

    all_data = col.get(include=["metadatas"])
    items = []
    for meta in all_data["metadatas"]:
        items.append({
            "question":   meta.get("question", ""),
            "answer":     meta.get("answer", "")[:300],
            "hit_count":  meta.get("hit_count", 0),
            "language":   meta.get("language", "en"),
            "mode":       meta.get("mode", ""),
            "created_at": meta.get("created_at", ""),
        })

    items.sort(key=lambda x: x["hit_count"], reverse=True)
    return items[:limit]


def get_faq_stats() -> dict:
    col = _get_faq_collection()
    count = col.count()
    if count == 0:
        return {"total_cached": 0, "total_hits": 0, "tokens_saved_estimate": 0}

    all_meta = col.get(include=["metadatas"])["metadatas"]
    total_hits = sum(m.get("hit_count", 0) for m in all_meta)
    # Rough estimate: avg ~500 tokens per Gemini call saved
    tokens_saved = total_hits * 500

    return {
        "total_cached":          count,
        "total_hits":            total_hits,
        "tokens_saved_estimate": tokens_saved,
    }


def clear_faq_cache() -> int:
    """Clear all FAQ cache entries. Returns count deleted."""
    col = _get_faq_collection()
    count = col.count()
    if count > 0:
        all_ids = col.get()["ids"]
        if all_ids:
            col.delete(ids=all_ids)
    return count


def delete_faq_entry(question_preview: str) -> bool:
    """Delete a specific FAQ entry by question preview match."""
    col = _get_faq_collection()
    all_data = col.get(include=["metadatas"])
    for i, meta in enumerate(all_data["metadatas"]):
        if meta.get("question", "").startswith(question_preview[:50]):
            col.delete(ids=[all_data["ids"][i]])
            return True
    return False
