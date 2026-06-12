"""
Smart Query Intelligence Service

Features:
  1. Sentiment analysis of user queries (positive/negative/neutral/frustrated)
  2. Intent classification (question / summarize / compare / translate / quiz / factual)
  3. Complexity scoring (simple → complex) — adjusts response depth
  4. Toxicity/harm detection — block harmful queries before hitting Gemini

All done with lightweight pattern matching + optional Gemini call for complex cases.
Zero extra dependencies — pure Python regex + Gemini when needed.
"""

import re
from typing import Optional

# ── Hindi + English harmful content patterns ──────────────────────────────────
_HARMFUL_PATTERNS = [
    # Violence
    r'\b(murder|kill|bomb|weapon|explosiv|poison|attack|harm|hurt|suicide|self.harm)\b',
    r'\b(मार|हत्या|बम|हथियार|जहर|हमला|नुकसान|आत्महत्या)\b',
    # Illegal
    r'\b(hack|crack|steal|fraud|scam|illegal|bypass|exploit)\b',
    r'\b(हैक|चोरी|धोखा|अवैध|घोटाला)\b',
    # Explicit
    r'\b(porn|xxx|nude|explicit|sexual.content)\b',
]

_HARMFUL_RE = re.compile('|'.join(_HARMFUL_PATTERNS), re.IGNORECASE)

# ── Intent patterns (stems, no trailing \b so 'summar' matches 'summarize') ────
_INTENT_PATTERNS = {
    "summarize":  [r'\b(summar|tldr|brief|overview|gist|संक्षेप|सारांश)'],
    "compare":    [r'\b(compar|differ|versus|\bvs\b|better|तुलना|अंतर)'],
    "translate":  [r'\b(translat|हिंदी में|अंग्रेज़ी में|in hindi|in english)'],
    "quiz":       [r'\b(quiz|mcq|test me|practice question|परीक्षा|प्रश्नोत्तर)'],
    "explain":    [r'\b(explain|what is|how does|how do|why|कैसे|क्यों|बताइए|समझाइए)'],
    "list":       [r'\b(list|enumerate|give me all|सभी|सूची)'],
    "factual":    [r'\b(who is|when did|where is|which|कौन|कब|कहाँ)'],
}

# ── Frustration/sentiment signals ─────────────────────────────────────────────
_FRUSTRATION_PATTERNS = [
    r'\b(not working|useless|wrong|bad|terrible|stupid|idiot|frustrat)\b',
    r'\b(काम नहीं|बेकार|गलत|खराब|निराश)\b',
    r'[!?]{2,}',   # multiple ! or ?
]
_FRUSTRATION_RE = re.compile('|'.join(_FRUSTRATION_PATTERNS), re.IGNORECASE)

_POSITIVE_PATTERNS = [
    r'\b(thanks|thank you|great|awesome|perfect|excellent|helpful|amazing)\b',
    r'\b(धन्यवाद|शुक्रिया|बढ़िया|अच्छा|परफेक्ट)\b',
]
_POSITIVE_RE = re.compile('|'.join(_POSITIVE_PATTERNS), re.IGNORECASE)


def analyze_query(query: str) -> dict:
    """
    Analyze a query for:
    - is_harmful: block before Gemini
    - intent: what the user wants
    - sentiment: positive / negative / frustrated / neutral
    - complexity: 1-5 score
    - suggested_response_style: brief / detailed / bullet_points
    """
    q = query.strip()

    # 1. Harm check
    is_harmful = bool(_HARMFUL_RE.search(q))

    # 2. Intent — check specific intents first, generic last
    detected_intent = "question"  # default
    _intent_priority = ["translate", "quiz", "summarize", "compare", "list", "factual", "explain"]
    for intent in _intent_priority:
        patterns = _INTENT_PATTERNS.get(intent, [])
        if any(re.search(p, q, re.IGNORECASE) for p in patterns):
            detected_intent = intent
            break

    # 3. Sentiment
    if _FRUSTRATION_RE.search(q):
        sentiment = "frustrated"
    elif _POSITIVE_RE.search(q):
        sentiment = "positive"
    elif q.endswith('?') or q.endswith('?'):
        sentiment = "curious"
    else:
        sentiment = "neutral"

    # 4. Complexity (word count + question depth heuristic)
    word_count = len(q.split())
    has_multiple_q = q.count('?') > 1 or ' and ' in q.lower() or ' और ' in q
    complexity = min(5, max(1, word_count // 8 + (2 if has_multiple_q else 0)))

    # 5. Suggested style
    if complexity <= 2:
        style = "brief"
    elif detected_intent in ("list", "compare"):
        style = "bullet_points"
    else:
        style = "detailed"

    return {
        "is_harmful":      is_harmful,
        "intent":          detected_intent,
        "sentiment":       sentiment,
        "complexity":      complexity,
        "response_style":  style,
        "word_count":      word_count,
    }


def get_sentiment_system_addition(sentiment: str, lang: str) -> str:
    """Append tone adjustment to system prompt based on user sentiment."""
    if sentiment == "frustrated":
        if lang == "hi":
            return "\n\nनोट: उपयोगकर्ता थोड़ा निराश लग रहा है। अत्यंत धैर्यपूर्ण और सहायक स्वर में उत्तर दें।"
        return "\n\nNote: The user seems frustrated. Be extra patient, empathetic, and clear in your response."
    if sentiment == "positive":
        if lang == "hi":
            return "\n\nनोट: उपयोगकर्ता का मूड अच्छा है। मित्रवत और उत्साही स्वर में उत्तर दें।"
        return "\n\nNote: The user is in a positive mood. Keep the tone friendly and engaging."
    return ""
