#  SahayakAI — Multilingual AI Knowledge Vault

**Your Neural Knowledge Vault** · Hindi + English · RAG Chatbot with FAQ Cache & Smart Tools

**Stack:** FastAPI · Gemini Flash · ChromaDB · Python 3.11+
**Deploy:** Render (free-tier friendly with 4–5 Gemini API keys)

---

## ✨ Key Features

- 💬 **Multilingual RAG Chat** — answers from your documents in Hindi & English
- ⚡ **FAQ Token-Saver Cache** — similar questions answered instantly, zero Gemini tokens
- 🧠 **Query Intelligence** — sentiment, intent detection & harmful-query blocking
- 🛠️ **Smart Tools** — document summarizer, quiz generator, translator, analytics
- 🌐 **General + RAG modes** — chat works even with no documents uploaded
- 📄 **Multi-format** — PDF, DOCX, TXT, MD, CSV
- 🔑 **Multi-key rotation** — spreads load across 4–5 free Gemini keys

---

## 📁 Project Structure

```
NeuraVault/
├── backend/
│   ├── main.py                       # FastAPI entry point
│   ├── config.py                     # Settings + API key pool
│   ├── requirements.txt
│   ├── .env.example
│   ├── routers/
│   │   ├── chat.py                   # POST /api/chat
│   │   ├── documents.py              # Upload / list / delete
│   │   ├── tools.py                  # Summarize / quiz / translate / FAQ / analytics
│   │   └── health.py                 # GET /api/health
│   └── services/
│       ├── gemini_service.py         # Key rotation + chat + embeddings
│       ├── chroma_service.py         # Vector store CRUD
│       ├── ingest_service.py         # Parse → chunk → embed → store
│       ├── rag_service.py            # Retrieve → prompt → answer (+ tools)
│       ├── faq_service.py            # Semantic FAQ cache (token saver)
│       └── intelligence_service.py   # Sentiment / intent / harm detection
├── frontend/
│   └── index.html                    # Chat + Smart Tools UI (served by FastAPI)
├── render.yaml                       # Render deploy config
└── WHATS_NEW.md                      # Feature changelog + SIH demo script
```

---

## 🚀 Local Setup

```bash
cd backend
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# Add your Gemini keys to .env (get free keys: https://aistudio.google.com/app/apikey)

uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Open http://localhost:8000

---

## ☁️ Deploy on Render

1. Push this folder to GitHub
2. Render → **New → Web Service** → connect repo
   - Root Directory: `backend`
   - Build: `pip install -r requirements.txt`
   - Start: `uvicorn main:app --host 0.0.0.0 --port $PORT`
3. Add environment variables (`GEMINI_API_KEY_1`…`_5`, `CHROMA_PERSIST_DIR=/data/chroma_db`)
4. Add a Disk → mount `/data` (1GB) so ChromaDB + FAQ cache persist
5. Deploy → live at `https://neuravault-xxxx.onrender.com`

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Health + key count |
| POST | `/api/chat` | Ask a question (RAG + cache) |
| POST | `/api/documents/upload` | Upload a document |
| GET | `/api/documents` | List documents |
| DELETE | `/api/documents/{file}` | Delete a document |
| POST | `/api/tools/summarize` | Summarize a document |
| POST | `/api/tools/quiz` | Generate quiz |
| POST | `/api/tools/translate` | Translate text |
| POST | `/api/tools/followups` | Suggest follow-up questions |
| GET | `/api/tools/analytics` | Vault analytics |
| GET | `/api/tools/faqs` | Popular cached questions |
| GET | `/api/tools/faq-stats` | Cache stats (tokens saved) |

---


