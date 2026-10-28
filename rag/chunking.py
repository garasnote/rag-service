"""Passage-level citations: window a document so a citation can point at a passage.

One document stays one search result -- the UI and the API rely on that. The
windows exist only so a hit can cite ``text[start:end]`` instead of the whole
posting, which matters when the sentence that matched sits 400 characters into a
long posting.

Standard library only, like the rest of the core.

    from rag.chunking import chunk_document
    chunk_document("tracker:12", text)
    # [{"doc_id": "tracker:12", "chunk_index": 0, "start": 0, "end": 600, "text": ...}]

Retrieval wrappers score a document by its best window and still return the
document:

    index = ChunkedBM25(docs)            # or ChunkedVectorIndex(docs, embedder)
    hits[0]["text"]              # the full document, unchanged
    hits[0]["chunk"]["start"]    # where the matching passage starts
"""
from __future__ import annotations

from .core import BM25
from .embed import VectorIndex

DEFAULT_SIZE = 600
DEFAULT_OVERLAP = 100
DEFAULT_THRESHOLD = 600
SNIPPET_CHARS = 240
SNAP = 80  # how far back a window may move to end on a word boundary


def _window_end(text: str, start: int, ideal: int) -> int:
    """Back the window off to the last space so a chunk does not cut a word in half."""
    if ideal >= len(text):
        return len(text)
    floor = max(start + 1, ideal - SNAP)
    cut = text.rfind(" ", floor, ideal)
    return cut + 1 if cut >= floor else ideal


def chunk_document(doc_id, text, size: int = DEFAULT_SIZE, overlap: int = DEFAULT_OVERLAP,
                   threshold: int = DEFAULT_THRESHOLD) -> list:
    """Overlapping windows over one document.

    A document at or below ``threshold`` characters yields exactly one window
    whose text is the document text, so short records behave as they always did.
    Offsets always index the original string: ``text[start:end] == chunk["text"]``.
    """
    if not 0 <= overlap < size:
        raise ValueError(f"overlap must be in [0, size): size={size} overlap={overlap}")
    text = text or ""
    if not text:
        return []
    if len(text) <= max(threshold, size):
        return [{"doc_id": doc_id, "chunk_index": 0, "start": 0, "end": len(text), "text": text}]

    out, start = [], 0
    while True:
        end = _window_end(text, start, min(start + size, len(text)))
        out.append({"doc_id": doc_id, "chunk_index": len(out), "start": start, "end": end,
                    "text": text[start:end]})
        if end >= len(text):
            break
        nxt = max(end - overlap, start + 1)
        if len(text) - nxt <= overlap:  # the tail is already inside the window we just took
            break
        start = nxt
    return out


def chunk_corpus(docs, size: int = DEFAULT_SIZE, overlap: int = DEFAULT_OVERLAP,
                 threshold: int = DEFAULT_THRESHOLD) -> list:
    """Flatten a corpus into windows, document by document."""
    out = []
    for d in docs:
        out.extend(chunk_document(d["id"], d.get("text", ""), size, overlap, threshold))
    return out


def _best_per_doc(hits, docs_by_id, k: int) -> list:
    """Keep the first (best scoring) window per document and return once per document."""
    out, seen = [], set()
    for h in hits:
        doc_id = h["doc_id"]
        if doc_id in seen:
            continue
        seen.add(doc_id)
        hit = dict(docs_by_id[doc_id])
        hit["score"] = h["score"]
        hit["chunk"] = {"index": h["chunk_index"], "start": h["start"], "end": h["end"],
                        "text": h["text"]}
        out.append(hit)
        if len(out) >= k:
            break
    return out


class _Windows:
    """Bookkeeping shared by the two chunked indexes: windows, and each one's parent."""

    def __init__(self, docs, size=DEFAULT_SIZE, overlap=DEFAULT_OVERLAP,
                 threshold=DEFAULT_THRESHOLD):
        self.docs = list(docs)
        self.by_id = {d["id"]: d for d in self.docs}
        self.windows = chunk_corpus(self.docs, size, overlap, threshold)
        self.rows = [{"id": f"{w['doc_id']}#{w['chunk_index']}", "doc_id": w["doc_id"],
                      "chunk_index": w["chunk_index"], "start": w["start"], "end": w["end"],
                      "text": w["text"]} for w in self.windows]

    @property
    def chunks_per_doc(self) -> float:
        return len(self.rows) / len(self.docs) if self.docs else 0.0


class ChunkedBM25(_Windows):
    """BM25 over windows. A document scores as its best window and is returned once."""

    def __init__(self, docs, size=DEFAULT_SIZE, overlap=DEFAULT_OVERLAP,
                 threshold=DEFAULT_THRESHOLD, stem=False):
        super().__init__(docs, size, overlap, threshold)
        self.index = BM25(self.rows, stem=stem)

    def search(self, query: str, k: int = 5) -> list:
        hits = self.index.search(query, k=len(self.rows)) if self.rows else []
        return _best_per_doc(hits, self.by_id, k)


class ChunkedVectorIndex(_Windows):
    """Dense search over windows, same contract as ``VectorIndex``."""

    def __init__(self, docs, embedder=None, size=DEFAULT_SIZE, overlap=DEFAULT_OVERLAP,
                 threshold=DEFAULT_THRESHOLD, cache_dir=".cache"):
        super().__init__(docs, size, overlap, threshold)
        self.index = VectorIndex(self.rows, embedder=embedder, cache_dir=cache_dir)

    @property
    def embedder(self):
        return self.index.embedder

    def search_text(self, text: str, k: int = 5) -> list:
        return self._search(text, k, self.index.search_text)

    def search_vector(self, qvec: list, k: int = 5) -> list:
        return self._search(qvec, k, self.index.search_vector)

    def _search(self, query, k: int, fn) -> list:
        hits = fn(query, len(self.rows)) if self.rows else []
        return _best_per_doc(hits, self.by_id, k)


def snippet(text: str, limit: int = SNIPPET_CHARS) -> str:
    """One-line excerpt of a passage, cut on a word boundary."""
    flat = " ".join((text or "").split())
    if len(flat) <= limit:
        return flat
    cut = flat.rfind(" ", 0, limit)
    return flat[:cut if cut > 0 else limit].rstrip() + "..."


def citation_fields(hit: dict, limit: int = SNIPPET_CHARS) -> dict:
    """Additive citation keys for an API source.

    A hit from a plain (unchunked) index has no ``chunk``, so the whole document
    is cited: index 0, offsets 0..len(text). Same shape either way.
    """
    text = hit.get("text") or ""
    chunk = hit.get("chunk") or {"index": 0, "start": 0, "end": len(text), "text": text}
    return {"chunk_index": chunk["index"], "start": chunk["start"], "end": chunk["end"],
            "snippet": snippet(chunk["text"], limit)}
