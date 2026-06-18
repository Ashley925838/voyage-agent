"""Embedding wrapper around sentence-transformers bge-small-en-v1.5.

Loaded once and cached (model is ~130MB; reloading per request is wasteful).
"""
import logging
from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

_MODEL_NAME = "BAAI/bge-small-en-v1.5"


@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    logger.info("Loading embedding model %s (first call only)", _MODEL_NAME)
    return SentenceTransformer(_MODEL_NAME)


def embed_texts(texts: list[str]) -> np.ndarray:
    """Returns a (N, dim) numpy array of L2-normalised embeddings.

    bge models recommend prepending an instruction to *queries* (not passages).
    For simplicity we treat both the same here; quality is still good.
    """
    model = get_embedder()
    vectors = model.encode(
        texts,
        normalize_embeddings=True,  # so inner product == cosine similarity
        show_progress_bar=False,
    )
    return np.asarray(vectors, dtype="float32")