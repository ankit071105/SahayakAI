"""
LLM Service — Groq (chat) + Cohere (embeddings).

Groq:
  - Free tier, no credit card needed
  - 30 req/min, 14,400 req/day per key
  - OpenAI-compatible API
  - Supports multiple keys for round-robin rotation
  - Models: llama-3.1-8b-instant (primary), llama3-8b-8192 (fallback)

Cohere:
  - embed-multilingual-v3.0: handles Hindi + English in ONE vector space
  - A Hindi question will find English document chunks and vice versa
  - 1024-dim vectors
  - Free trial key works immediately, no credit card
"""

import time
import threading
from typing import Optional
import groq as groq_sdk
import cohere
import config

# ── Thread-safe Groq key rotation ─────────────────────────────────────────────
_lock          = threading.Lock()
_key_index     = 0
_blocked_keys: set[str] = set()   # rate-limited/invalid keys for this session


def _next_groq_key(skip: Optional[set] = None) -> Optional[str]:
    """Return next available Groq key in round-robin order."""
    global _key_index
    skip      = (skip or set()) | _blocked_keys
    available = [k for k in config.GROQ_API_KEYS if k not in skip]
    if not available:
        return None
    with _lock:
        key = available[_key_index % len(available)]
        _key_index += 1
    return key


def _classify_error(err: str) -> str:
    e = err.lower()
    if "401" in e or "invalid api key" in e or "authentication" in e:
        return "invalid_key"
    if "429" in e or "rate_limit" in e or "rate limit" in e:
        return "ratelimit"
    if "quota" in e or "exhausted" in e or "daily" in e:
        return "quota"
    if "503" in e or "overloaded" in e or "unavailable" in e:
        return "overloaded"
    return "other"


# ════════════════════════════════════════════════════════════════════════════
#  GROQ CHAT
# ════════════════════════════════════════════════════════════════════════════
def groq_chat(
    system_prompt: str,
    user_message:  str,
    history:       Optional[list] = None,
    _model:        Optional[str]  = None,
    _is_fallback:  bool = False,
) -> str:
    """
    Call Groq LLM. Rotates API keys on rate-limit.
    Falls back to llama3-8b-8192 if primary model is overloaded.
    """
    model      = _model or config.GROQ_CHAT_MODEL
    tried_keys: set[str] = set()
    last_err   = None
    max_attempts = config.RETRY_ATTEMPTS * max(len(config.GROQ_API_KEYS), 1)

    for attempt in range(max_attempts):
        key = _next_groq_key(skip=tried_keys)
        if key is None:
            # All keys tried — attempt fallback model once
            if not _is_fallback:
                print(f"⚠️  All Groq keys exhausted on {model}. Trying fallback model…")
                return groq_chat(
                    system_prompt, user_message, history,
                    _model=config.GROQ_FALLBACK_MODEL, _is_fallback=True,
                )
            break

        try:
            client = groq_sdk.Groq(api_key=key)

            # Build messages
            messages = [{"role": "system", "content": system_prompt}]
            for turn in (history or []):
                messages.append({
                    "role": "user" if turn["role"] == "user" else "assistant",
                    "content": turn["content"],
                })
            messages.append({"role": "user", "content": user_message})

            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.4,
                max_tokens=2048,
            )
            return response.choices[0].message.content

        except Exception as e:
            last_err = e
            kind     = _classify_error(str(e))

            if kind == "invalid_key":
                print(f"🚫 Groq key invalid — skipping: …{key[-6:]}")
                _blocked_keys.add(key)
                tried_keys.add(key)
                continue

            if kind in ("ratelimit", "quota"):
                print(f"⏳ Groq rate limited (attempt {attempt+1}). Rotating key…")
                tried_keys.add(key)
                time.sleep(config.RETRY_DELAY_SECONDS)
                continue

            if kind == "overloaded" and not _is_fallback:
                print(f"⚠️  Groq overloaded on {model}. Trying fallback…")
                return groq_chat(
                    system_prompt, user_message, history,
                    _model=config.GROQ_FALLBACK_MODEL, _is_fallback=True,
                )

            # Unknown error — don't retry
            raise

    raise RuntimeError(
        f"All Groq attempts failed after {attempt+1} tries "
        f"(model: {model}). Last error: {last_err}"
    )


# ════════════════════════════════════════════════════════════════════════════
#  COHERE EMBEDDINGS
# ════════════════════════════════════════════════════════════════════════════

# One shared Cohere client (single API key, stateless)
_cohere_client: Optional[cohere.ClientV2] = None

def _get_cohere() -> cohere.ClientV2:
    global _cohere_client
    if _cohere_client is None:
        _cohere_client = cohere.ClientV2(api_key=config.COHERE_API_KEY)
    return _cohere_client


def cohere_embed(texts: list[str]) -> list[list[float]]:
    """
    Embed a list of document texts for storage.
    Uses embed-multilingual-v3.0: Hindi + English share the same vector space,
    so a Hindi question retrieves English doc chunks and vice versa.
    Returns list of 1024-dim float vectors.
    """
    all_vecs: list[list[float]] = []
    batch_size = 96  # Cohere max batch

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        for attempt in range(config.RETRY_ATTEMPTS):
            try:
                client = _get_cohere()
                res = client.embed(
                    model=config.COHERE_EMBED_MODEL,
                    texts=batch,
                    input_type="search_document",
                    embedding_types=["float"],
                )
                vecs = res.embeddings.float_
                all_vecs.extend(vecs)
                break
            except Exception as e:
                if attempt < config.RETRY_ATTEMPTS - 1:
                    print(f"⏳ Cohere embed retry {attempt+1}: {e}")
                    time.sleep(config.RETRY_DELAY_SECONDS * (attempt + 1))
                else:
                    raise RuntimeError(f"Cohere embedding failed: {e}") from e

    return all_vecs


def cohere_embed_query(query: str) -> list[float]:
    """
    Embed a single search query.
    Uses input_type='search_query' (different from document embedding —
    this is important for retrieval quality).
    """
    for attempt in range(config.RETRY_ATTEMPTS):
        try:
            client = _get_cohere()
            res = client.embed(
                model=config.COHERE_EMBED_MODEL,
                texts=[query],
                input_type="search_query",
                embedding_types=["float"],
            )
            return list(res.embeddings.float_[0])
        except Exception as e:
            if attempt < config.RETRY_ATTEMPTS - 1:
                print(f"⏳ Cohere query embed retry {attempt+1}: {e}")
                time.sleep(config.RETRY_DELAY_SECONDS * (attempt + 1))
            else:
                raise RuntimeError(f"Cohere query embedding failed: {e}") from e
