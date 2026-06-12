"""
Document ingestion pipeline:
  1. Parse uploaded file (PDF, TXT, MD, DOCX, CSV)
  2. Detect or accept user-specified language
  3. Chunk with overlap
  4. Embed via Gemini
  5. Store in ChromaDB
"""

import io
import re
import datetime
from typing import Optional

import config
from services.llm_service import cohere_embed as gemini_embed
from services.chroma_service import add_documents, delete_document_by_source

# ── Language detection ─────────────────────────────────────────────────────────
_DEVANAGARI_RE = re.compile(r'[\u0900-\u097F]')

# Common Hinglish words — Hindi meaning written in Roman/English script.
# These are the most frequent short words that pure English never uses.
_HINGLISH_WORDS = {
    # pronouns / verbs
    "kya","kyun","kaise","kaisa","kaisi","kab","kahan","kaun","kitna","kitne",
    "aap","app","tum","main","hum","woh","yeh","ye","vo",
    "kar","karo","kare","karta","karti","karte","krta","krte","krti",
    "hain","hai","tha","thi","hoga","hogi","honge",
    "kr","kro","kre","krna","karna",
    # common words
    "nahi","nhi","nai","mat","mujhe","muje","tumhe","use","inhe","unhe",
    "ky","kyu","kyun","kyuki","kyonki",
    "bhi","toh","to","par","pe","se","me","mein","ko","ka","ki","ke",
    "ek","teen","char","aur","ya","lekin","magar","phir","ab",
    "accha","acha","thik","theek","sahi","galat",
    "bata","batao","bataye","samjho","samjhao","samajh",
    "help","madad","batao","puchna","poochna","jaanna",
    "kuch","koi","sab","sabhi","bahut","zyada","thoda","sirf","bas",
    "stke","skte","skte","skta","skti","sakta","sakti","sakte",
    "liye","liye","chahiye","chahye","chaiye",
    "please","plz","pls","yaar","bhai","dost",
}

def detect_language(text: str) -> str:
    """
    Detect language of a short query.

    Priority order:
    1. If >5% Devanagari chars → Hindi (देवनागरी)
    2. If multiple Hinglish keywords found → Hindi (Romanized/Hinglish)
    3. Else → English

    This handles all three common user input styles:
      - Pure Hindi:    "आप क्या कर सकते हो?"        → hi
      - Hinglish:      "app ky kr stke ho"            → hi
      - Mixed:         "Please mujhe RTI ke baare mein batao" → hi
      - Pure English:  "What can you do?"             → en
    """
    if not text:
        return "en"

    # 1. Devanagari script check
    deva_count = len(_DEVANAGARI_RE.findall(text))
    if deva_count / max(len(text), 1) > 0.05:
        return "hi"

    # 2. Hinglish word check — tokenize and count matches
    tokens = re.findall(r'[a-zA-Z]+', text.lower())
    if not tokens:
        return "en"

    matches = sum(1 for t in tokens if t in _HINGLISH_WORDS)
    # If ≥1 match in short text (≤5 tokens) or ≥2 matches in longer text → Hinglish
    threshold = 1 if len(tokens) <= 4 else 2
    if matches >= threshold:
        return "hi"

    return "en"


# ── File parsers ───────────────────────────────────────────────────────────────
def _parse_txt(data: bytes) -> str:
    return data.decode("utf-8", errors="replace")

def _parse_md(data: bytes) -> str:
    return data.decode("utf-8", errors="replace")

def _parse_csv(data: bytes) -> str:
    import csv
    text_rows = []
    reader = csv.reader(io.StringIO(data.decode("utf-8", errors="replace")))
    for row in reader:
        text_rows.append(", ".join(row))
    return "\n".join(text_rows)

def _parse_pdf(data: bytes) -> str:
    try:
        import pdfplumber
        text_parts = []
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    text_parts.append(t)
        return "\n".join(text_parts)
    except ImportError:
        raise RuntimeError("pdfplumber not installed. Add it to requirements.txt")

def _parse_docx(data: bytes) -> str:
    try:
        from docx import Document
        doc = Document(io.BytesIO(data))
        return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
    except ImportError:
        raise RuntimeError("python-docx not installed. Add it to requirements.txt")

PARSERS = {
    ".txt":  _parse_txt,
    ".md":   _parse_md,
    ".csv":  _parse_csv,
    ".pdf":  _parse_pdf,
    ".docx": _parse_docx,
}


# ── Chunking ───────────────────────────────────────────────────────────────────
def chunk_text(
    text: str,
    chunk_size: int = None,
    overlap: int = None,
) -> list[str]:
    """
    Split text into overlapping word-based chunks.
    Preserves sentence boundaries where possible.
    """
    chunk_size = chunk_size or config.CHUNK_SIZE
    overlap    = overlap    or config.CHUNK_OVERLAP

    # Split into sentences first (handles both Hindi and English)
    sentences = re.split(r'(?<=[।?.!])\s+', text.strip())
    chunks    = []
    current   = []
    current_len = 0

    for sent in sentences:
        words = sent.split()
        if current_len + len(words) > chunk_size and current:
            chunks.append(" ".join(current))
            # Keep overlap words
            current = current[-overlap:] if overlap > 0 else []
            current_len = len(current)
        current.extend(words)
        current_len += len(words)

    if current:
        chunks.append(" ".join(current))

    return [c for c in chunks if len(c.strip()) > 20]  # filter tiny chunks


# ── Main ingestion function ────────────────────────────────────────────────────
def ingest_document(
    file_data: bytes,
    filename: str,
    language: str = "auto",
    overwrite: bool = True,
) -> dict:
    """
    Full ingestion pipeline. Returns a summary dict.
    """
    import os
    ext = os.path.splitext(filename)[-1].lower()

    if ext not in PARSERS:
        raise ValueError(f"Unsupported file type: {ext}. Allowed: {list(PARSERS)}")

    file_size_mb = len(file_data) / (1024 * 1024)
    if file_size_mb > config.MAX_FILE_SIZE_MB:
        raise ValueError(f"File too large: {file_size_mb:.1f}MB (max {config.MAX_FILE_SIZE_MB}MB)")

    # 1. Parse
    raw_text = PARSERS[ext](file_data)
    if not raw_text.strip():
        raise ValueError("Document appears to be empty or unreadable.")

    # 2. Detect language
    detected_lang = detect_language(raw_text) if language == "auto" else language

    # 3. Chunk
    chunks = chunk_text(raw_text)
    if not chunks:
        raise ValueError("No usable text chunks found in document.")

    # 4. Delete old version if overwriting
    if overwrite:
        deleted = delete_document_by_source(filename)
        if deleted:
            print(f"🗑️  Removed {deleted} old chunks for '{filename}'")

    # 5. Embed all chunks (batched by gemini_service)
    embeddings = gemini_embed(chunks)

    # 6. Build metadata per chunk
    now = datetime.datetime.utcnow().isoformat()
    metadatas = [
        {
            "source":      filename,
            "language":    detected_lang,
            "chunk_index": i,
            "total_chunks": len(chunks),
            "uploaded_at": now,
            "file_type":   ext,
        }
        for i in range(len(chunks))
    ]

    # 7. Store in ChromaDB
    ids = add_documents(
        texts=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    return {
        "filename":       filename,
        "language":       detected_lang,
        "chunks_created": len(chunks),
        "characters":     len(raw_text),
        "ids":            ids[:3],   # return first 3 IDs as sample
        "message":        f"✅ '{filename}' ingested successfully ({len(chunks)} chunks, lang={detected_lang})",
    }