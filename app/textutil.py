"""Tokenisation + chunking shared by the index builder and the retriever."""
from __future__ import annotations

import re

STOPWORDS = frozenset(
    ["a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have", "in", "is", "it", "its", "of", "on", "or", "that", "the", "this", "to", "was", "were", "will", "with", "i", "me", "my", "we", "our", "you", "your", "he", "she", "they", "them", "what", "which", "who", "whom", "how", "when", "where", "why", "do", "does", "did", "can", "could", "should", "would", "any", "some", "all", "about", "into", "than", "then", "there", "these", "those", "so", "such", "not", "no", "yes", "please", "tell", "show", "give", "find", "list"]
)
TOKEN_RE = re.compile(r"[a-z0-9\u0600-\u06ff]+")
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS and len(t) > 1 or t.isdigit()]


def chunk_text(text: str, max_chars: int = 1500, overlap: float = 0.15) -> list[str]:
    """Paragraph-aware chunking with ~15% character overlap between consecutive chunks."""
    paras = [p.strip() for p in re.split(r"\n\s*\n|\n(?=[A-Z][A-Z &'/-]{3,}\n)", text) if p.strip()]
    # hard-split giant paragraphs on sentence boundaries
    pieces: list[str] = []
    for p in paras:
        if len(p) <= max_chars:
            pieces.append(p)
            continue
        sent = re.split(r"(?<=[.!?])\s+|\n", p)
        buf = ""
        for s in sent:
            if len(buf) + len(s) + 1 > max_chars and buf:
                pieces.append(buf.strip())
                buf = ""
            buf += s + " "
        if buf.strip():
            pieces.append(buf.strip()[: max_chars * 2])
    chunks: list[str] = []
    cur = ""
    for p in pieces:
        if cur and len(cur) + len(p) + 2 > max_chars:
            chunks.append(cur.strip())
            tail = cur[-int(max_chars * overlap):]
            tail = tail[tail.find(" ") + 1:] if " " in tail else tail
            cur = tail + "\n" + p
        else:
            cur = f"{cur}\n\n{p}" if cur else p
    if cur.strip():
        chunks.append(cur.strip())
    return chunks or [text[:max_chars]]
