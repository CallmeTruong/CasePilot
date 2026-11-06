"""Offline English/Vietnamese fallbacks; model replies can use any user language."""

import json
import re
from functools import lru_cache
from pathlib import Path

VIETNAMESE = re.compile(r"[ăâđêôơưĂÂĐÊÔƠƯ\u1ea0-\u1ef9]")
ENGLISH = re.compile(
    r"\b(?:please|hello|hi|help|order|cancel|return|exchange|calculate|show|check|"
    r"ignore|instructions|print|system|prompt|payment|price|thanks)\b",
    re.I,
)


def fallback_language(text, previous="en"):
    # Record attachments are English metadata, not a user's language preference.
    text = text.split("\n\nSelected Retail record:", 1)[0]
    if re.search(r"(?:reply|respond|answer|speak).*\b(?:in )?English\b", text, re.I) or re.search(
        r"(?:trả lời|nói).*tiếng (?:Anh|anh)", text
    ):
        return "en"
    if re.search(r"(?:reply|respond|answer|speak).*\bVietnamese\b", text, re.I) or re.search(
        r"(?:trả lời|nói).*tiếng Việt", text, re.I
    ):
        return "vi"
    if VIETNAMESE.search(text) or re.search(
        r"\b(?:hãy|tính|xin chào|cam on|don hang)\b", text, re.I
    ):
        return "vi"
    if ENGLISH.search(text):
        return "en"
    # IDs, emails and short confirmations retain the preceding language.
    return previous


@lru_cache(maxsize=1)
def _vietnamese():
    return json.loads(Path(__file__).with_name("translations.json").read_text(encoding="utf-8"))


def localize(text, language):
    return _vietnamese().get(text, text) if language == "vi" else text
