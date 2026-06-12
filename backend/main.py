import os
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("CHROMA_TELEMETRY_IMPL", "none")
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
import os

from routers import chat, documents, health, tools
from services.chroma_service import init_chroma

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: initialize ChromaDB
    init_chroma()
    print("✅ ChromaDB initialized")
    yield
    # Shutdown
    print("👋 Shutting down")

app = FastAPI(
    title="NeuraVault — Multilingual RAG API",
    description="RAG pipeline with Gemini Flash, ChromaDB, Hindi/English support",
    version="1.0.0",
    lifespan=lifespan
)

# CORS — allow all origins for Render deployment
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API routers
app.include_router(health.router, prefix="/api", tags=["Health"])
app.include_router(chat.router, prefix="/api", tags=["Chat"])
app.include_router(documents.router, prefix="/api", tags=["Documents"])
app.include_router(tools.router, prefix="/api", tags=["Tools"])

# Serve frontend static files (built React or plain HTML)
frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.exists(frontend_path):
    app.mount("/static", StaticFiles(directory=frontend_path), name="static")

    @app.get("/", response_class=FileResponse)
    async def serve_frontend():
        return FileResponse(os.path.join(frontend_path, "index.html"))

    @app.get("/{full_path:path}", response_class=FileResponse)
    async def serve_spa(full_path: str):
        file_path = os.path.join(frontend_path, full_path)
        if os.path.exists(file_path):
            return FileResponse(file_path)
        return FileResponse(os.path.join(frontend_path, "index.html"))
