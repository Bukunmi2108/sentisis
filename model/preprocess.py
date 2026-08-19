"""Text normalization shared by training and inference."""

import html
import re
import unicodedata

import emoji

ESCAPE_RE = re.compile(r"\\u([0-9a-fA-F]{4})")
URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
MENTION_RE = re.compile(r"@\w{1,15}\b")
WS_RE = re.compile(r"\s+")
NON_ALPHA_RE = re.compile(r"[^a-z ]")


def _escaped_char(match: re.Match[str]) -> str:
    return chr(int(match.group(1), 16))


def expand_escapes(text: str) -> str:
    """Expand escaped unicode and HTML entities repeatedly until the text stops changing.

    Every expansion shortens the string, so the loop always terminates. Running to a fixed
    point is what keeps normalize idempotent on double-encoded input such as "&amp;amp;".
    """
    while True:
        expanded = html.unescape(ESCAPE_RE.sub(_escaped_char, text))
        if expanded == text:
            return text
        text = expanded


def normalize(text: str) -> str:
    """Normalize text while preserving sentiment-bearing case and punctuation."""
    normalized = unicodedata.normalize("NFKC", expand_escapes(text))
    normalized = URL_RE.sub("[url]", normalized)
    normalized = MENTION_RE.sub("[user]", normalized)
    normalized = emoji.demojize(normalized, delimiters=("[", "]"))
    return WS_RE.sub(" ", normalized).strip()


def to_bow(text: str) -> str:
    """Convert normalized text to lowercase letters and spaces for TF-IDF."""
    normalized = normalize(text)
    bow = NON_ALPHA_RE.sub(" ", normalized.lower())
    return WS_RE.sub(" ", bow).strip()
