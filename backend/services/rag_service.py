"""
RAG Service — Full pipeline with:
  - FAQ semantic cache (saves Gemini tokens on repeated questions)
  - Query intelligence (sentiment, intent, harm detection)
  - 3 answer modes: RAG / general-fallback / pure-general
  - Follow-up suggestion generation
  - Document summarization
  - Quiz generation
  - Translation
"""

import json
import re
from services.llm_service import cohere_embed_query as gemini_embed_query, groq_chat as gemini_chat
from services.chroma_service import query_documents, collection_count
from services.ingest_service import detect_language
from services.faq_service import check_faq_cache, store_faq_cache
from services.intelligence_service import analyze_query, get_sentiment_system_addition
import config

# ══════════════════════════════════════════════════════════════════════════════
#  SYSTEM PROMPTS
#
#  KEY FIX: Language is injected as the very FIRST line of every system prompt
#  so LLaMA cannot ignore it. Vague instructions like "match the user's language"
#  are replaced with a hard explicit command e.g.:
#  "RESPOND ONLY IN ENGLISH. Do NOT use Hindi or any other language."
#  This is the only reliable way to enforce language with instruction-tuned LLMs.
# ══════════════════════════════════════════════════════════════════════════════

def _lang_instruction(lang: str) -> str:
    """Return an unambiguous first-line language command for the system prompt."""
    if lang == "hi":
        return (
            "IMPORTANT: आपको केवल हिंदी (देवनागरी लिपि) में उत्तर देना है। "
            "अंग्रेज़ी में उत्तर देना सख्त मना है।\n"
            "CRITICAL RULE: RESPOND ONLY IN HINDI (Devanagari script). "
            "Do NOT write even a single sentence in English."
        )
    else:
        return (
            "CRITICAL RULE: RESPOND ONLY IN ENGLISH. "
            "Do NOT use Hindi, Hinglish, or any other language, even if the user wrote in Hindi. "
            "Your entire response must be in English only."
        )

_SYSTEM_RAG = """You are SahayakAI, a precise and helpful AI assistant.
{lang_instruction}

Answer using the provided document context below. Supplement with general knowledge only when needed, and clearly say so.

Additional rules:
- Cite the source document name when using document content.
- Use bullet points for lists/steps. Use markdown for code blocks.
- Never fabricate facts from documents that are not present.
{style_hint}
{sentiment_hint}

Document context:
{context}
"""

_SYSTEM_GENERAL = """You are SahayakAI, a knowledgeable and helpful AI assistant.
{lang_instruction}

Answer the user's question using your general knowledge.
- Be concise but complete. Use bullet points where helpful.
- You may suggest uploading relevant documents for more precise sourced answers.
- NEVER assist with illegal activities, violence, or harmful content.
{style_hint}
{sentiment_hint}
"""

_STYLE_HINTS = {
    "brief":         "Keep the response brief — 2-3 sentences max.",
    "detailed":      "Provide a thorough, detailed explanation.",
    "bullet_points": "Structure the response as clear bullet points or a numbered list.",
}

_STYLE_HINTS_HI = {
    "brief":         "उत्तर संक्षिप्त रखें — अधिकतम 2-3 वाक्य।",
    "detailed":      "विस्तृत और संपूर्ण उत्तर दें।",
    "bullet_points": "उत्तर को स्पष्ट बुलेट पॉइंट में दें।",
}


def _build_context_str(chunks: list[dict]) -> str:
    parts = []
    for i, c in enumerate(chunks, 1):
        src   = c["metadata"].get("source", "unknown")
        score = c.get("score", 0)
        parts.append(f"[{i}] Source: {src} (relevance: {score:.2f})\n{c['text']}")
    return "\n\n---\n\n".join(parts)


def _build_system(template: str, context: str, style: str, sentiment: str, lang: str) -> str:
    hints = _STYLE_HINTS_HI if lang == "hi" else _STYLE_HINTS
    style_hint     = hints.get(style, "")
    sentiment_hint = get_sentiment_system_addition(sentiment, lang)
    lang_instruction = _lang_instruction(lang)
    return template.format(
        lang_instruction=lang_instruction,
        context=context,
        style_hint=style_hint,
        sentiment_hint=sentiment_hint,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN RAG PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def rag_query(
    query: str,
    history: list = None,
    top_k: int = None,
    language: str = "auto",
    source_filter: str = None,
) -> dict:
    """
    Full RAG pipeline with FAQ cache + query intelligence.
    Modes: rag | general_fallback | general
    """

    # ── 1. Intelligence analysis ──────────────────────────────────────────────
    intel = analyze_query(query)

    # Block harmful queries immediately
    if intel["is_harmful"]:
        harm_msg = (
            "यह अनुरोध संसाधित नहीं किया जा सकता। कृपया एक उचित प्रश्न पूछें।"
            if detect_language(query) == "hi"
            else "This request cannot be processed. Please ask an appropriate question."
        )
        return {
            "answer": harm_msg,
            "sources": [], "language": "en",
            "chunks_used": 0, "mode": "blocked",
            "from_cache": False, "intent": "blocked",
        }

    # ── 2. Language detection ─────────────────────────────────────────────────
    query_lang = detect_language(query) if language == "auto" else language

    # ── 3. Embed query ────────────────────────────────────────────────────────
    query_embedding = gemini_embed_query(query)

    # ── 4. FAQ cache check ────────────────────────────────────────────────────
    cached = check_faq_cache(
        query_embedding=query_embedding,
        source_filter=source_filter,
        language=query_lang,
    )
    if cached:
        cached["intent"] = intel["intent"]
        cached["sentiment"] = intel["sentiment"]
        return cached

    # ── 5. Document retrieval ─────────────────────────────────────────────────
    doc_count = collection_count()
    style     = intel["response_style"]
    sentiment = intel["sentiment"]

    # ── MODE: No documents → pure general chat ────────────────────────────────
    if doc_count == 0:
        system = _build_system(_SYSTEM_GENERAL, "", style, sentiment, query_lang)
        answer = gemini_chat(system_prompt=system, user_message=query, history=history or [])

        result = {
            "answer": answer, "sources": [], "language": query_lang,
            "chunks_used": 0, "mode": "general", "from_cache": False,
            "intent": intel["intent"], "sentiment": intel["sentiment"],
        }
        store_faq_cache(query, query_embedding, answer, [], query_lang, "general", 0, source_filter)
        return result

    # Retrieve top-K chunks
    filter_meta = {"source": source_filter} if source_filter else None
    chunks      = query_documents(
        query_embedding=query_embedding,
        top_k=top_k or config.TOP_K_RESULTS,
        filter_meta=filter_meta,
    )
    relevant = [c for c in chunks if c.get("score", 0) > 0.3]

    # ── MODE: Docs exist but no relevant match → general fallback ─────────────
    if not relevant:
        system = _build_system(_SYSTEM_GENERAL, "", style, sentiment, query_lang)
        answer = gemini_chat(system_prompt=system, user_message=query, history=history or [])

        hint = (
            "\n\n💡 *आपके दस्तावेज़ों में इस प्रश्न से मेल खाने वाली सामग्री नहीं मिली। सामान्य ज्ञान से उत्तर दिया।*"
            if query_lang == "hi" else
            "\n\n💡 *No matching content found in your documents. Answered from general knowledge.*"
        )
        answer += hint

        result = {
            "answer": answer, "sources": [], "language": query_lang,
            "chunks_used": 0, "mode": "general_fallback", "from_cache": False,
            "intent": intel["intent"], "sentiment": intel["sentiment"],
        }
        store_faq_cache(query, query_embedding, answer, [], query_lang, "general_fallback", 0, source_filter)
        return result

    # ── MODE: Full RAG ────────────────────────────────────────────────────────
    context_str = _build_context_str(relevant)
    system      = _build_system(_SYSTEM_RAG, context_str, style, sentiment, query_lang)
    answer      = gemini_chat(system_prompt=system, user_message=query, history=history or [])
    sources     = list({c["metadata"]["source"] for c in relevant})

    result = {
        "answer": answer, "sources": sources, "language": query_lang,
        "chunks_used": len(relevant), "mode": "rag", "from_cache": False,
        "intent": intel["intent"], "sentiment": intel["sentiment"],
    }
    store_faq_cache(query, query_embedding, answer, sources, query_lang, "rag", len(relevant), source_filter)
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  TOOLS
# ══════════════════════════════════════════════════════════════════════════════

def summarize_document(source: str, lang: str = "en") -> dict:
    from services.chroma_service import get_collection
    col    = get_collection()
    result = col.get(where={"source": source}, include=["documents"])
    chunks = result.get("documents", [])
    if not chunks:
        raise ValueError(f"No chunks found for '{source}'")

    full_text = "\n\n".join(chunks[:30])

    if lang == "hi":
        prompt = f"""निम्नलिखित दस्तावेज़ का विश्लेषण करें और केवल JSON लौटाएं:
{{"title":"शीर्षक","summary":"2-3 वाक्यों में सारांश","key_points":["बिंदु1","बिंदु2","बिंदु3","बिंदु4","बिंदु5"],"topics":["विषय1","विषय2","विषय3"],"language":"hi"}}

दस्तावेज़:\n{full_text[:4000]}"""
    else:
        prompt = f"""Analyze this document and return ONLY valid JSON:
{{"title":"document title","summary":"2-3 sentence summary","key_points":["point1","point2","point3","point4","point5"],"topics":["topic1","topic2","topic3"],"language":"en"}}

Document:\n{full_text[:4000]}"""

    raw  = gemini_chat("Return ONLY valid JSON. No markdown, no extra text.", prompt)
    raw  = re.sub(r"```json|```", "", raw).strip()
    try:
        data = json.loads(raw)
    except Exception:
        data = {"title": source.split("/")[-1], "summary": raw[:300], "key_points": [], "topics": [], "language": lang}

    data["source"]      = source
    data["chunk_count"] = len(chunks)
    data["word_count"]  = sum(len(c.split()) for c in chunks)
    return data


def generate_quiz(source: str, num_questions: int = 5, lang: str = "en") -> dict:
    from services.chroma_service import get_collection
    col    = get_collection()
    result = col.get(where={"source": source}, include=["documents"])
    chunks = result.get("documents", [])
    if not chunks:
        raise ValueError(f"No chunks found for '{source}'")

    full_text = "\n\n".join(chunks[:20])
    n = min(num_questions, 10)

    if lang == "hi":
        prompt = f"""{n} MCQ प्रश्न बनाएं। केवल JSON array:
[{{"question":"प्रश्न?","options":["A. option1","B. option2","C. option3","D. option4"],"answer":"A","explanation":"स्पष्टीकरण"}}]

दस्तावेज़:\n{full_text[:3500]}"""
    else:
        prompt = f"""Generate {n} MCQ questions. Return ONLY JSON array:
[{{"question":"Question?","options":["A. opt1","B. opt2","C. opt3","D. opt4"],"answer":"A","explanation":"Why correct"}}]

Document:\n{full_text[:3500]}"""

    raw  = gemini_chat("Return ONLY a valid JSON array. No markdown, no extra text.", prompt)
    raw  = re.sub(r"```json|```", "", raw).strip()
    try:
        questions = json.loads(raw)
        if not isinstance(questions, list):
            questions = []
    except Exception:
        questions = []

    return {"source": source, "questions": questions, "count": len(questions), "language": lang}


def translate_text(text: str, target_lang: str) -> dict:
    if target_lang == "hi":
        prompt = f"Translate to Hindi (Devanagari). Return ONLY the translation:\n\n{text}"
    else:
        prompt = f"Translate to English. Return ONLY the translation:\n\n{text}"
    translated = gemini_chat("You are a translator. Return only the translated text.", prompt)
    return {"original": text, "translated": translated.strip(), "target_lang": target_lang}


def get_followup_suggestions(query: str, answer: str, lang: str = "en") -> list[str]:
    if lang == "hi":
        prompt = f"""3 अनुवर्ती प्रश्न सुझाएं। केवल JSON array:
["प्रश्न1?","प्रश्न2?","प्रश्न3?"]
प्रश्न: {query}\nउत्तर: {answer[:400]}"""
    else:
        prompt = f"""Suggest 3 follow-up questions. Return ONLY JSON array:
["Question1?","Question2?","Question3?"]
Question: {query}\nAnswer: {answer[:400]}"""

    raw = gemini_chat("Return ONLY a JSON array of 3 question strings. No markdown.", prompt)
    raw = re.sub(r"```json|```", "", raw).strip()
    try:
        s = json.loads(raw)
        return s[:3] if isinstance(s, list) else []
    except Exception:
        return []