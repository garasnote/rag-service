"""Optional approximate nearest-neighbour backend for dense retrieval.

``VectorIndex`` compares the query against every document vector. On a corpus of
a few hundred records that is microseconds per query, so an ANN index buys
nothing today -- this module exists to *measure* what approximation would cost
if the corpus grew, not to make today's search faster.

    uv pip install --python <venv> hnswlib          # or: pip install 'job-market-rag[ann]'
    python3 eval/bench_ann.py
    python3 eval/run_intent.py --index ann --ef 50

hnswlib is imported lazily, inside the constructor, so the core library stays
stdlib-only: importing ``rag.ann`` costs nothing and the tests skip when the
dependency is missing.
"""
from __future__ import annotations

import time

from .embed import VectorIndex  # noqa: F401  (re-exported for callers)

SPACES = ("cosine", "ip")


def _require_hnswlib():
    """Import hnswlib or fail with an actionable message."""
    try:
        import hnswlib
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise RuntimeError(
            "HNSWIndex needs the optional dependency hnswlib: "
            "uv pip install --python <venv> hnswlib  (or pip install 'job-market-rag[ann]')"
        ) from exc
    return hnswlib


class HNSWIndex(VectorIndex):
    """The ``VectorIndex`` surface (``search_text``, ``search_vector``, ``vectors``)
    backed by an hnswlib graph instead of a full scan.

    ``space="cosine"`` keeps the same score meaning as ``VectorIndex`` -- cosine
    similarity -- so hits from the two are directly comparable. ``ef`` is the
    query-time breadth: it can be changed later, and only affects recall and
    latency, never the graph.
    """

    def __init__(self, docs, embedder=None, space: str = "cosine", ef: int = 50,
                 M: int = 16, ef_construction: int = 200, cache_dir=".cache",
                 seed: int = 42, build_threads: int = 1):
        if space not in SPACES:
            raise ValueError(f"space must be one of {SPACES}, got {space!r}")
        super().__init__(docs, embedder, cache_dir)
        if not self.vectors:
            raise ValueError("HNSWIndex needs at least one document")
        hnswlib = _require_hnswlib()
        import numpy as np

        self.space, self.M, self.ef_construction = space, M, ef_construction
        matrix = np.asarray(self.vectors, dtype=np.float32)
        self.dim = int(matrix.shape[1])

        started = time.perf_counter()
        self._index = hnswlib.Index(space=space, dim=self.dim)
        self._index.init_index(max_elements=len(self.vectors), ef_construction=ef_construction,
                               M=M, random_seed=seed)
        # build_threads=1 on purpose: hnswlib builds with every core by default, which makes
        # the graph (and therefore recall at a given ef) differ between runs. Reproducible
        # numbers are worth more than a faster build at this corpus size.
        self._index.add_items(matrix, np.arange(len(self.vectors)), num_threads=build_threads)
        self.build_s = time.perf_counter() - started  # graph construction only, embeddings excluded
        self.ef = ef

    @property
    def ef(self) -> int:
        return self._ef

    @ef.setter
    def ef(self, value: int) -> None:
        self._ef = max(int(value), 1)
        self._index.set_ef(self._ef)

    def search_vector(self, qvec: list, k: int = 5) -> list:
        import numpy as np

        k = min(int(k), len(self.vectors))
        if k <= 0:
            return []
        labels, distances = self._index.knn_query(np.asarray([qvec], dtype=np.float32), k=k)
        hits = []
        for label, distance in zip(labels[0], distances[0]):
            i = int(label)
            if i < 0:  # hnswlib pads the result with -1 if the graph has fewer than k items
                continue
            score = 1.0 - float(distance)  # both "cosine" and "ip" report 1 - similarity
            if score <= 0:                 # same filter as the exact search, so top-k sets stay comparable
                continue
            hits.append({"score": round(score, 4), **self.docs[i]})
        return hits
