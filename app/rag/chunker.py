"""Sentence-aware chunking. Same idea as LedgerLens's section-aware chunking,
adapted for narrative Wikipedia text instead of regulatory filings.
"""
import re
from dataclasses import dataclass


@dataclass
class Chunk:
    text: str
    source_title: str
    source_url: str


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def chunk_text(
    text: str,
    source_title: str,
    source_url: str,
    target_chars: int = 600,
    overlap_chars: int = 80,
) -> list[Chunk]:
    """Greedy sentence packing into ~target_chars chunks with small overlap.

    Why sentence-aware: avoids cutting mid-sentence, which embeddings handle
    poorly. Why overlap: a fact straddling a chunk boundary still appears
    in one chunk wholly.
    """
    sentences = _SENTENCE_SPLIT.split(text)
    chunks: list[Chunk] = []
    buffer = ""

    for sentence in sentences:
        if not sentence.strip():
            continue
        candidate = f"{buffer} {sentence}".strip() if buffer else sentence
        if len(candidate) <= target_chars:
            buffer = candidate
        else:
            if buffer:
                chunks.append(Chunk(text=buffer, source_title=source_title, source_url=source_url))
            # Start new buffer with overlap tail of previous + this sentence.
            tail = buffer[-overlap_chars:] if buffer else ""
            buffer = f"{tail} {sentence}".strip()

    if buffer:
        chunks.append(Chunk(text=buffer, source_title=source_title, source_url=source_url))

    return chunks