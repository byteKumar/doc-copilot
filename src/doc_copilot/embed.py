"""Embedders behind one small interface.

- LocalEmbedder: sentence-transformers all-MiniLM-L6-v2 (384 dims), runs on CPU, free.
- FakeEmbedder: deterministic feature hashing of words and word bigrams. No model download,
  so tests and CI are fast and repeatable. It only captures word overlap, not meaning.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol

from .config import Settings


class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


_WORD = re.compile(r"[a-z0-9]+")


class FakeEmbedder:
    def __init__(self, dim: int = 384):
        self.dim = dim

    def _bucket(self, token: str) -> tuple[int, float]:
        h = hashlib.blake2b(token.encode(), digest_size=8).digest()
        idx = int.from_bytes(h[:4], "little") % self.dim
        sign = 1.0 if h[4] & 1 else -1.0
        return idx, sign

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for text in texts:
            vec = [0.0] * self.dim
            words = _WORD.findall(text.lower())
            feats = words + [f"{a}_{b}" for a, b in zip(words, words[1:], strict=False)]
            for f in feats:
                idx, sign = self._bucket(f)
                vec[idx] += sign
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            out.append([v / norm for v in vec])
        return out


class LocalEmbedder:
    def __init__(self, model_name: str, dim: int):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:  # pragma: no cover - depends on optional extra
            raise RuntimeError(
                "EMBEDDER=local needs the optional extra: uv sync --extra local-embeddings"
            ) from e
        self._model = SentenceTransformer(model_name, device="cpu")
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        vecs = self._model.encode(
            texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False
        )
        return [v.tolist() for v in vecs]


def make_embedder(settings: Settings) -> Embedder:
    if settings.embedder == "fake":
        return FakeEmbedder(settings.embedding_dim)
    if settings.embedder == "local":
        return LocalEmbedder(settings.embedding_model, settings.embedding_dim)
    raise ValueError(f"Unknown EMBEDDER={settings.embedder!r} (use 'local' or 'fake')")
