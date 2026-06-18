"""Long-term per-user preference memory backed by FAISS.

Each user gets their own small FAISS index containing extracted preferences.
On every assistant turn we ask the LLM to extract any new stable preferences
from the latest exchange and append them. On every new turn we retrieve the
top-k preferences most relevant to the user's current message and inject them
into the system prompt.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import faiss
import numpy as np

from app.rag.embedder import embed_texts

logger = logging.getLogger(__name__)

_MEM_DIR = Path("data/memory")
_MEM_DIR.mkdir(parents=True, exist_ok=True)
_EMBED_DIM = 384


def _safe(user_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", user_id.strip()) or "anon"


class UserPreferenceStore:
    """One FAISS index + parallel JSON list of preference strings per user."""

    def __init__(self, user_id: str) -> None:
        self._user_id = _safe(user_id)
        self._idx_path = _MEM_DIR / f"{self._user_id}.faiss"
        self._meta_path = _MEM_DIR / f"{self._user_id}.meta.json"
        self._index, self._preferences = self._load()

    # ---------- public ----------

    def add(self, preferences: list[str]) -> None:
        """Append new preferences. De-duplicates against exact existing strings."""
        new = [p.strip() for p in preferences if p.strip() and p.strip() not in self._preferences]
        if not new:
            return

        vectors = embed_texts(new)
        self._index.add(vectors)
        self._preferences.extend(new)
        self._save()
        logger.info("Stored %d new preferences for user %s", len(new), self._user_id)

    def retrieve(self, query: str, top_k: int = 3) -> list[str]:
        if self._index.ntotal == 0:
            return []
        q_vec = embed_texts([query])
        scores, indices = self._index.search(q_vec, min(top_k, self._index.ntotal))
        # Only keep moderately relevant matches (cosine > 0.3).
        return [
            self._preferences[i]
            for s, i in zip(scores[0], indices[0])
            if i >= 0 and s > 0.3
        ]

    def all(self) -> list[str]:
        return list(self._preferences)

    # ---------- internals ----------

    def _load(self) -> tuple[faiss.Index, list[str]]:
        if self._idx_path.exists() and self._meta_path.exists():
            index = faiss.read_index(str(self._idx_path))
            prefs = json.loads(self._meta_path.read_text())
            return index, prefs
        return faiss.IndexFlatIP(_EMBED_DIM), []

    def _save(self) -> None:
        faiss.write_index(self._index, str(self._idx_path))
        self._meta_path.write_text(json.dumps(self._preferences))