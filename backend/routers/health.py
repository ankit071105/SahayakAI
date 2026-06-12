from fastapi import APIRouter
from services.chroma_service import collection_count
import config

router = APIRouter()

@router.get("/health")
async def health():
    return {
        "status":            "ok",
        "chat_provider":     "Groq",
        "chat_model":        config.GROQ_CHAT_MODEL,
        "fallback_model":    config.GROQ_FALLBACK_MODEL,
        "embed_provider":    "Cohere",
        "embed_model":       config.COHERE_EMBED_MODEL,
        "embed_dimension":   config.EMBED_DIMENSION,
        "groq_keys_loaded":  len(config.GROQ_API_KEYS),
        "total_chunks":      collection_count(),
        "supported_languages": config.SUPPORTED_LANGUAGES,
    }
