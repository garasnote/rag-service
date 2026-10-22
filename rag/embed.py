"""Dense retrieval with no third-party dependency.

Feature hashing over word unigrams + bigrams, sublinear tf, L2 normalised.
It is a *lexical* embedding: cheap, deterministic and offline. It is NOT a
semantic model -- for that, swap in MiniLM (see requirements.txt) and keep the
same interface: vector(text) -> list[float] must be L2-normalised.
"""
import hashlib
import math
from collections import Counter

from .core import tokenize


def _hash(feature: str) -> int:
    return int.from_bytes(hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest(), "big")


class HashingEmbedder:
    name = "hashing-512"

    def __init__(self, dim: int = 512):
        self.dim = dim

    def vector(self, text: str) -> list:
        toks = tokenize(text)
        feats = Counter(toks)
        for a, b in zip(toks, toks[1:]):
            feats[f"{a}_{b}"] += 1
        v = [0.0] * self.dim
        for feature, count in feats.items():
            h = _hash(feature)
            idx = h % self.dim
            sign = 1.0 if (h >> 63) & 1 else -1.0
            v[idx] += sign * (1.0 + math.log(count))
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]


def cosine(a: list, b: list) -> float:
    return sum(x * y for x, y in zip(a, b))


class VectorIndex:
    """Brute-force cosine search: 400 documents, so an index would be vanity."""

    def __init__(self, docs, embedder=None):
        self.docs = docs
        self.embedder = embedder or HashingEmbedder()
        self.vectors = [self.embedder.vector(d["text"]) for d in docs]

    def search_text(self, text: str, k: int = 5) -> list:
        return self.search_vector(self.embedder.vector(text), k)

    def search_vector(self, qvec: list, k: int = 5) -> list:
        scored = [(cosine(qvec, v), i) for i, v in enumerate(self.vectors)]
        scored = [s for s in scored if s[0] > 0]
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [{"score": round(s, 4), **self.docs[i]} for s, i in scored[:k]]
