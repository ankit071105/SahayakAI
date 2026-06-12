import os
from dotenv import load_dotenv

load_dotenv()

# ════════════════════════════════════════════════════════════════════════════
#  GROQ — Chat / RAG Answers
#  Free tier: 30 req/min, 14,400 req/day per key. No credit card.
#  Get keys: https://console.groq.com → API Keys
# ════════════════════════════════════════════════════════════════════════════
_groq_raw = [
    os.getenv("GROQ_API_KEY_1", ""),
    os.getenv("GROQ_API_KEY_2", ""),
    os.getenv("GROQ_API_KEY_3", ""),
    os.getenv("GROQ_API_KEY_4", ""),
    os.getenv("GROQ_API_KEY_5", ""),
]
GROQ_API_KEYS = [k.strip() for k in _groq_raw if k.strip()]

if not GROQ_API_KEYS:
    raise RuntimeError(
        "No Groq API keys found!\n"
        "Set GROQ_API_KEY_1 in your .env or Render environment variables.\n"
        "Get free keys at: https://console.groq.com"
    )

# Primary model — fast, great quality, 30k context
GROQ_CHAT_MODEL    = os.getenv("GROQ_CHAT_MODEL",    "llama-3.1-8b-instant")
# Fallback if primary hits quota — even faster, lighter
GROQ_FALLBACK_MODEL= os.getenv("GROQ_FALLBACK_MODEL","llama3-8b-8192")

# ════════════════════════════════════════════════════════════════════════════
#  COHERE — Embeddings
#  embed-multilingual-v3.0 handles Hindi + English in same vector space.
#  Free trial: 1000 calls/month. Production key: unlimited embed calls.
#  Get key: https://dashboard.cohere.com → API Keys
# ════════════════════════════════════════════════════════════════════════════
COHERE_API_KEY   = os.getenv("COHERE_API_KEY", "")
COHERE_EMBED_MODEL = "embed-multilingual-v3.0"  # 1024-dim, Hindi+English native
EMBED_DIMENSION  = 1024  # must match ChromaDB collection dimension

if not COHERE_API_KEY:
    raise RuntimeError(
        "No Cohere API key found!\n"
        "Set COHERE_API_KEY in your .env or Render environment variables.\n"
        "Get free key at: https://dashboard.cohere.com"
    )

# ════════════════════════════════════════════════════════════════════════════
#  CHROMADB
# ════════════════════════════════════════════════════════════════════════════
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
CHROMA_COLLECTION  = "rag_documents"

# ════════════════════════════════════════════════════════════════════════════
#  RAG SETTINGS
# ════════════════════════════════════════════════════════════════════════════
CHUNK_SIZE    = int(os.getenv("CHUNK_SIZE",    "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "5"))

# ════════════════════════════════════════════════════════════════════════════
#  UPLOAD
# ════════════════════════════════════════════════════════════════════════════
MAX_FILE_SIZE_MB   = int(os.getenv("MAX_FILE_SIZE_MB", "20"))
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md", ".docx", ".csv"}

# ════════════════════════════════════════════════════════════════════════════
#  RETRY
# ════════════════════════════════════════════════════════════════════════════
RETRY_ATTEMPTS      = int(os.getenv("RETRY_ATTEMPTS",      "3"))
RETRY_DELAY_SECONDS = float(os.getenv("RETRY_DELAY_SECONDS","1"))

# ════════════════════════════════════════════════════════════════════════════
#  LANGUAGE
# ════════════════════════════════════════════════════════════════════════════
SUPPORTED_LANGUAGES = ["en", "hi", "auto"]
