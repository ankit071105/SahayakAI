from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from services.rag_service import rag_query

router = APIRouter()


class ChatMessage(BaseModel):
    role: str       # "user" or "model"
    content: str


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    history: list[ChatMessage] = []
    language: str = "auto"
    source_filter: Optional[str] = None
    top_k: Optional[int] = Field(None, ge=1, le=15)


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]
    language: str
    chunks_used: int
    mode: str                       # rag | general | general_fallback | blocked
    from_cache: bool = False        # was this served from FAQ cache?
    intent: Optional[str] = None    # detected intent
    sentiment: Optional[str] = None # detected sentiment
    cache_similarity: Optional[float] = None
    cache_hits: Optional[int] = None


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    try:
        history = [{"role": m.role, "content": m.content} for m in req.history]
        result = rag_query(
            query=req.query,
            history=history,
            top_k=req.top_k,
            language=req.language,
            source_filter=req.source_filter,
        )
        return ChatResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
