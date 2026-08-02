"""
Lightweight, dependency-free sentiment analysis for movie reviews/comments.

This is a compact lexicon + rule-based scorer (negation handling, intensifiers,
punctuation emphasis). It is intentionally self-contained so the whole engine
runs offline with zero extra installs.

For production-grade sentiment, swap `score_text` for VADER
(`pip install vaderSentiment`) or a transformer model — the rest of the engine
only depends on the returned float in [-1, 1], so it's a drop-in replacement.
"""

from __future__ import annotations

import re
from functools import lru_cache

# --- Compact opinion lexicon (weights roughly in [-3, 3]) -------------------
_POSITIVE = {
    "masterpiece": 3, "brilliant": 3, "phenomenal": 3, "outstanding": 3,
    "excellent": 3, "superb": 3, "magnificent": 3, "flawless": 3,
    "amazing": 2.5, "fantastic": 2.5, "wonderful": 2.5, "incredible": 2.5,
    "loved": 2.5, "love": 2, "beautiful": 2, "gripping": 2, "captivating": 2,
    "compelling": 2, "stunning": 2, "great": 2, "riveting": 2, "unforgettable": 2,
    "moving": 1.5, "clever": 1.5, "enjoyable": 1.5, "solid": 1.5, "fun": 1.5,
    "engaging": 1.5, "charming": 1.5, "impressive": 1.5, "good": 1.5,
    "liked": 1.5, "like": 1, "decent": 1, "nice": 1, "worth": 1, "recommend": 2,
    "favorite": 2, "perfect": 3, "best": 2.5, "delightful": 2, "thrilling": 2,
}
_NEGATIVE = {
    "atrocious": -3, "abysmal": -3, "terrible": -3, "horrible": -3,
    "awful": -3, "garbage": -3, "trash": -3, "disaster": -3, "unwatchable": -3,
    "worst": -3, "hate": -2.5, "hated": -2.5, "dreadful": -2.5, "pathetic": -2.5,
    "boring": -2, "dull": -2, "bland": -2, "disappointing": -2, "disappointed": -2,
    "mediocre": -1.5, "forgettable": -1.5, "weak": -1.5, "flat": -1.5,
    "predictable": -1.5, "messy": -1.5, "clunky": -1.5, "tedious": -2,
    "bad": -1.5, "poor": -1.5, "lame": -1.5, "annoying": -1.5, "mess": -2,
    "waste": -2.5, "avoid": -2, "overrated": -1.5, "ridiculous": -1.5,
    "confusing": -1.5, "slow": -1, "meh": -1, "nonsense": -2, "cringe": -2,
}
_LEXICON = {**_POSITIVE, **_NEGATIVE}

_NEGATIONS = {
    "not", "no", "never", "n't", "cannot", "cant", "without", "hardly",
    "barely", "rarely", "neither", "nor", "isnt", "wasnt", "dont", "didnt",
}
_INTENSIFIERS = {
    "very": 1.5, "really": 1.4, "so": 1.3, "extremely": 1.8, "incredibly": 1.7,
    "absolutely": 1.6, "totally": 1.4, "utterly": 1.7, "quite": 1.2,
    "remarkably": 1.5, "insanely": 1.7, "super": 1.4,
}
_DAMPENERS = {"slightly": 0.6, "somewhat": 0.7, "kinda": 0.7, "kind": 0.7, "bit": 0.7}

_TOKEN_RE = re.compile(r"[a-z']+|[!?]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@lru_cache(maxsize=8192)
def score_text(text: str) -> float:
    """Return a sentiment score in [-1, 1] for a piece of text.

    0 means neutral / no opinion words found. Positive is favorable.
    """
    if not text or not isinstance(text, str):
        return 0.0
    tokens = _tokenize(text)
    if not tokens:
        return 0.0

    total = 0.0
    hits = 0
    exclaim_boost = 1.0

    for i, tok in enumerate(tokens):
        if tok and set(tok) <= {"!"}:
            exclaim_boost = min(1.0 + 0.15 * len(tok), 1.6)
            continue
        if tok in _LEXICON:
            weight = _LEXICON[tok]
            # Look back up to 2 tokens for negation / intensity modifiers.
            window = tokens[max(0, i - 2):i]
            if any(w in _NEGATIONS for w in window):
                weight *= -0.75  # negation flips + slightly softens
            for w in window:
                if w in _INTENSIFIERS:
                    weight *= _INTENSIFIERS[w]
                if w in _DAMPENERS:
                    weight *= _DAMPENERS[w]
            total += weight
            hits += 1

    if hits == 0:
        return 0.0

    # Average opinion weight, apply punctuation emphasis, squash to [-1, 1].
    avg = (total / hits) * exclaim_boost
    return max(-1.0, min(1.0, avg / 3.0))


def label(score: float) -> str:
    """Human-readable bucket for a sentiment score."""
    if score >= 0.35:
        return "positive"
    if score <= -0.35:
        return "negative"
    return "neutral"


if __name__ == "__main__":
    samples = [
        "An absolute masterpiece, brilliant from start to finish!",
        "Not bad, but somewhat predictable and a bit slow.",
        "Utterly boring and a complete waste of time.",
        "It was fine, nothing special.",
        "I really loved this — incredibly moving and beautiful.",
    ]
    for s in samples:
        sc = score_text(s)
        print(f"{sc:+.2f}  {label(sc):8s}  {s}")
