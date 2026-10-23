"""
Postgres-only embedding layer: local dense embeddings via
sentence-transformers `all-MiniLM-L6-v2` (384-d, CPU/GPU, no API key, free).
First use downloads weights (~90MB) to the HF cache; afterwards offline.
`fit()` is a no-op (pretrained) kept for API compatibility.
"""
from __future__ import annotations
import logging
from typing import Protocol

logger = logging.getLogger(__name__)


class Embedder(Protocol):
    def fit(self, texts: list[str]) -> None: ...
    def embed(self, texts: list[str]) -> list[list[float]]: ...
    def embed_one(self, text: str) -> list[float]: ...


class MiniLMEmbedder:
    MODEL_NAME = "all-MiniLM-L6-v2"
    DIM = 384

    def __init__(self):
        self._model = None
        self._warned_truncation = False

    def _ensure(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.MODEL_NAME)
        return self._model

    def _warn_if_truncating(self, texts: list[str], model) -> None:
        """Belt-and-braces observability: log (once per process) when an input
        exceeds max_seq_length -- its tail would be silently dropped."""
        if self._warned_truncation:
            return
        try:
            limit = getattr(model, "max_seq_length", None)
            if not limit:
                return
            enc = model.tokenizer(texts, add_special_tokens=True,
                                  truncation=False, padding=False)
            over = sum(1 for ids in enc["input_ids"] if len(ids) > limit)
            if over:
                self._warned_truncation = True
                logger.warning(
                    "%d/%d inputs exceed embedder max_seq_length=%d; tails truncated",
                    over, len(texts), limit,
                )
        except Exception:  # tokenizer hiccups must never break embedding
            pass

    def fit(self, texts: list[str]) -> None:
        self._ensure()

    def embed(self, texts: list[str]) -> list[list[float]]:
        model = self._ensure()
        self._warn_if_truncating(texts, model)
        vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return [list(map(float, v)) for v in vecs]

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]


_embedder: MiniLMEmbedder | None = None


def get_embedder() -> Embedder:
    """Process-wide singleton. Loading SentenceTransformer from disk on every
    query added ~1s of dead weight per request; the weights are immutable, so
    one shared instance is safe and much faster."""
    global _embedder
    if _embedder is None:
        _embedder = MiniLMEmbedder()
    return _embedder
