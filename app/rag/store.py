"""Per-city FAISS index with disk persistence.

Layout on disk:
  data/faiss/
    Tokyo.faiss        # raw FAISS index
    Tokyo.meta.json    # parallel array of chunk metadata
    Singapore.faiss
    Singapore.meta.json
    ...

Index type: IndexFlatIP (inner product on normalised vectors == cosine).
At our scale (a few hundred chunks per city) brute-force is fine and saves
the complexity of training IVF/HNSW. We can swap later without touching callers.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import faiss
import numpy as np

from app.rag.chunker import Chunk, chunk_text
from app.rag.embedder import embed_texts
from app.rag.wiki_fetcher import WikiFetcher

logger = logging.getLogger(__name__)

_INDEX_DIR = Path("data/faiss")
_INDEX_DIR.mkdir(parents=True, exist_ok=True)
_EMBED_DIM = 384  # bge-small-en-v1.5


@dataclass
class RetrievedChunk:
    text: str
    source_title: str
    source_url: str
    score: float


def _safe_filename(city: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", city.strip())


class CityVectorStore:
    """One FAISS index per city, lazily built and persisted."""

    def __init__(self) -> None:
        self._wiki = WikiFetcher()

    # ---------- public API ----------

    def retrieve(self, city: str, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        """Return the top_k most relevant chunks for the query within the city."""
        index, metadata = self._load_or_build(city)
        if index is None or index.ntotal == 0:
            return []

        query_vec = embed_texts([query])  # (1, dim)
        scores, indices = index.search(query_vec, top_k)

        results: list[RetrievedChunk] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(metadata):
                continue
            meta = metadata[idx]
            results.append(
                RetrievedChunk(
                    text=meta["text"],
                    source_title=meta["source_title"],
                    source_url=meta["source_url"],
                    score=float(score),
                )
            )
        return results

    # ---------- internals ----------

    def _index_path(self, city: str) -> Path:
        return _INDEX_DIR / f"{_safe_filename(city)}.faiss"

    def _meta_path(self, city: str) -> Path:
        return _INDEX_DIR / f"{_safe_filename(city)}.meta.json"

    def _load_or_build(self, city: str) -> tuple[faiss.Index | None, list[dict]]:
        idx_path = self._index_path(city)
        meta_path = self._meta_path(city)

        if idx_path.exists() and meta_path.exists():
            logger.info("Loading cached FAISS index for %s", city)
            index = faiss.read_index(str(idx_path))
            metadata = json.loads(meta_path.read_text())
            return index, metadata

        logger.info("Building FAISS index for %s (first time)", city)
        return self._build_and_persist(city)

    def _build_and_persist(self, city: str) -> tuple[faiss.Index | None, list[dict]]:
        pages = self._wiki.fetch_for_city(city)
        if not pages:
            logger.warning("No Wikipedia pages found for %s", city)
            return None, []

        chunks: list[Chunk] = []
        for page in pages:
            chunks.extend(
                chunk_text(
                    text=page.text,
                    source_title=page.title,
                    source_url=page.url,
                )
            )
        if not chunks:
            return None, []

        logger.info("Embedding %d chunks for %s", len(chunks), city)
        vectors = embed_texts([c.text for c in chunks])

        index = faiss.IndexFlatIP(_EMBED_DIM)
        index.add(vectors)

        metadata = [asdict(c) for c in chunks]

        # Persist.
        faiss.write_index(index, str(self._index_path(city)))
        self._meta_path(city).write_text(json.dumps(metadata))
        logger.info("Saved index for %s (%d vectors)", city, index.ntotal)

        return index, metadata