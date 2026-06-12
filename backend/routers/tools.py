"""
Tools router — power features for the Smart Tools page + FAQ cache management.

  POST /api/tools/summarize       → Document summary
  POST /api/tools/quiz            → Quiz generation
  POST /api/tools/translate       → Translate text
  POST /api/tools/followups       → Suggest follow-up questions
  GET  /api/tools/analytics       → Usage analytics
  GET  /api/tools/search?q=...    → Search document names
  GET  /api/tools/preview/{src}   → First chars of a document
  GET  /api/tools/faqs            → Popular cached questions
  GET  /api/tools/faq-stats       → Cache stats (tokens saved)
  DELETE /api/tools/faqs          → Clear FAQ cache
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from services.rag_service import (
    summarize_document, generate_quiz, translate_text, get_followup_suggestions
)
from services.chroma_service import get_collection, list_documents, collection_count
from services.faq_service import (
    get_popular_faqs, get_faq_stats, clear_faq_cache
)

router = APIRouter()


# ── Summarize ──────────────────────────────────────────────────────────────────
class SummarizeRequest(BaseModel):
    source: str
    language: str = "en"

@router.post("/tools/summarize")
async def summarize(req: SummarizeRequest):
    try:
        return summarize_document(req.source, req.language)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Quiz ───────────────────────────────────────────────────────────────────────
class QuizRequest(BaseModel):
    source: str
    num_questions: int = Field(5, ge=2, le=10)
    language: str = "en"

@router.post("/tools/quiz")
async def quiz(req: QuizRequest):
    try:
        return generate_quiz(req.source, req.num_questions, req.language)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Translate ──────────────────────────────────────────────────────────────────
class TranslateRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=3000)
    target_lang: str = Field("hi", pattern="^(hi|en)$")

@router.post("/tools/translate")
async def translate(req: TranslateRequest):
    try:
        return translate_text(req.text, req.target_lang)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Follow-up suggestions ──────────────────────────────────────────────────────
class FollowupRequest(BaseModel):
    query: str
    answer: str
    language: str = "en"

@router.post("/tools/followups")
async def followups(req: FollowupRequest):
    try:
        return {"suggestions": get_followup_suggestions(req.query, req.answer, req.language)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Analytics ──────────────────────────────────────────────────────────────────
@router.get("/tools/analytics")
async def analytics():
    try:
        docs         = list_documents()
        total_chunks = collection_count()
        lang_counts  = {"en": 0, "hi": 0, "unknown": 0}
        type_counts: dict[str, int] = {}

        for doc in docs:
            lang = doc.get("language", "unknown")
            lang_counts[lang] = lang_counts.get(lang, 0) + 1
            src = doc.get("source", "")
            ft = ".txt"
            for ext in [".pdf", ".docx", ".txt", ".md", ".csv"]:
                if src.lower().endswith(ext):
                    ft = ext
                    break
            type_counts[ft] = type_counts.get(ft, 0) + 1

        faq = get_faq_stats()

        return {
            "total_documents":       len(docs),
            "total_chunks":          total_chunks,
            "avg_chunks_per_doc":    round(total_chunks / max(len(docs), 1), 1),
            "language_distribution": lang_counts,
            "file_type_distribution": type_counts,
            "cache_stats":           faq,
            "documents": [
                {"source": d["source"], "chunks": d["chunks"], "language": d["language"]}
                for d in sorted(docs, key=lambda x: x["chunks"], reverse=True)
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Search documents ───────────────────────────────────────────────────────────
@router.get("/tools/search")
async def search_docs(q: str = Query(..., min_length=1)):
    try:
        docs    = list_documents()
        q_lower = q.lower()
        matches = [d for d in docs if q_lower in d["source"].lower()]
        return {"query": q, "results": matches, "count": len(matches)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Document preview ───────────────────────────────────────────────────────────
@router.get("/tools/preview/{source:path}")
async def preview_doc(source: str):
    try:
        col    = get_collection()
        result = col.get(where={"source": source}, include=["documents", "metadatas"])
        if not result["documents"]:
            raise HTTPException(status_code=404, detail="Document not found")
        return {
            "source":    source,
            "preview":   result["documents"][0][:600],
            "language":  result["metadatas"][0].get("language", "en"),
            "file_type": result["metadatas"][0].get("file_type", ""),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── FAQ cache ──────────────────────────────────────────────────────────────────
@router.get("/tools/faqs")
async def get_faqs(limit: int = Query(10, ge=1, le=50)):
    try:
        return {"faqs": get_popular_faqs(limit)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tools/faq-stats")
async def faq_stats():
    try:
        return get_faq_stats()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/tools/faqs")
async def clear_faqs():
    try:
        count = clear_faq_cache()
        return {"message": f"Cleared {count} cached entries", "cleared": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
