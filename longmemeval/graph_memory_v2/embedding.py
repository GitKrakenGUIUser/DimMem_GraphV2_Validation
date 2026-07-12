from __future__ import annotations

import hashlib
import math
import threading
from typing import Iterable, List, Protocol, Sequence

from .io_utils import tokenize


class Embedder(Protocol):
    def encode(self, texts: Sequence[str]) -> List[List[float]]: ...


class HashEmbedder:
    """Dependency-free deterministic semantic fallback.

    This is for reproducible smoke tests and ablations. Benchmark runs should use
    SentenceTransformerEmbedder or the same embedding model as the DimMem baseline.
    """

    def __init__(self, dim: int = 512) -> None:
        self.dim = max(64, int(dim))

    def encode(self, texts: Sequence[str]) -> List[List[float]]:
        outputs: List[List[float]] = []
        for text in texts:
            vector = [0.0] * self.dim
            tokens = tokenize(text)
            grams = tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]
            for token in grams:
                digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
                index = int.from_bytes(digest[:4], "big") % self.dim
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                vector[index] += sign
            norm = math.sqrt(sum(value * value for value in vector)) or 1.0
            outputs.append([value / norm for value in vector])
        return outputs


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str, device: str = "cpu") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is not installed; install requirements-optional.txt "
                "or use --embedder hash"
            ) from exc
        self.model = SentenceTransformer(model_name, device=device)
        self._encode_lock = threading.RLock()

    def encode(self, texts: Sequence[str]) -> List[List[float]]:
        # One shared model is reused across case workers. Most PyTorch backends are
        # not reliably re-entrant for concurrent encode() calls on one GPU model,
        # so the small critical section prevents duplicate model loads and OOMs.
        with self._encode_lock:
            vectors = self.model.encode(
                list(texts),
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        return [list(map(float, row)) for row in vectors]


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right))
