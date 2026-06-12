"""
ChromaDB service.
- Uses Gemini embeddings (via gemini_service) instead of the default SentenceTransformer
  so we don't need a heavy ML model on Render's free tier.
- Persists to disk at CHROMA_PERSIST_DIR so documents survive redeploys
  (mount a Render Disk at /data and set CHROMA_PERSIST_DIR=/data/chroma_db).
"""

import uuid
import chromadb
from chromadb.config import Settings
from typing import Optional
import config

_client: Optional[chromadb.PersistentClient] = None
_collection = None


def init_chroma():
    global _client, _collection
    _client = chromadb.PersistentClient(
        path=config.CHROMA_PERSIST_DIR,
        settings=Settings(anonymized_telemetry=False),
    )
    _collection = _client.get_or_create_collection(
        name=config.CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )
    print(f"✅ ChromaDB ready — {_collection.count()} documents in store")


def get_collection():
    if _collection is None:
        init_chroma()
    return _collection


def add_documents(
    texts: list[str],
    embeddings: list[list[float]],
    metadatas: list[dict],
    ids: Optional[list[str]] = None,
) -> list[str]:
    """
    Insert chunked text + precomputed Gemini embeddings into ChromaDB.
    Returns the list of inserted IDs.
    """
    col = get_collection()
    if ids is None:
        ids = [str(uuid.uuid4()) for _ in texts]

    col.add(
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids,
    )
    return ids


def query_documents(
    query_embedding: list[float],
    top_k: int = None,
    filter_meta: Optional[dict] = None,
) -> list[dict]:
    """
    Semantic search against ChromaDB.
    Returns list of {text, metadata, distance} dicts sorted by relevance.
    """
    col = get_collection()
    top_k = top_k or config.TOP_K_RESULTS

    kwargs = dict(
        query_embeddings=[query_embedding],
        n_results=min(top_k, max(col.count(), 1)),
        include=["documents", "metadatas", "distances"],
    )
    if filter_meta:
        kwargs["where"] = filter_meta

    results = col.query(**kwargs)

    output = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        output.append({"text": doc, "metadata": meta, "score": 1 - dist})

    return output


def delete_document_by_source(source: str) -> int:
    """Delete all chunks belonging to a document (matched by metadata.source)."""
    col = get_collection()
    results = col.get(where={"source": source}, include=["metadatas"])
    ids_to_delete = results["ids"]
    if ids_to_delete:
        col.delete(ids=ids_to_delete)
    return len(ids_to_delete)


def list_documents() -> list[dict]:
    """Return unique document sources with their chunk count."""
    col = get_collection()
    all_meta = col.get(include=["metadatas"])["metadatas"]
    counts: dict[str, dict] = {}
    for m in all_meta:
        src = m.get("source", "unknown")
        if src not in counts:
            counts[src] = {
                "source": src,
                "language": m.get("language", "unknown"),
                "chunks": 0,
                "uploaded_at": m.get("uploaded_at", ""),
            }
        counts[src]["chunks"] += 1
    return list(counts.values())


def collection_count() -> int:
    return get_collection().count()
