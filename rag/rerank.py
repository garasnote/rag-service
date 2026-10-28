"""Reranking: a second, more expensive look at a shortlist.

Stage one (BM25, dense, RRF) is built for recall, and it is cheap because it
scores query and document apart. A reranker scores the *pair*, so it can tell
"junior" from "senior" in the same document, or notice that ``power bi`` is one
constraint and not two independent words. The price is a forward pass per
candidate, and the ceiling is whatever stage one already retrieved: a reranker
can reorder a shortlist, never extend it.

It is worth the pass here -- measured, not assumed: at pool 40 the lexical
reranker lifts satisfied@5 on the intent benchmark from 55% to 75% over hybrid
retrieval, and the cross-encoder lifts it to 75% while also lifting hit@1 on the
entity benchmark from 60% to 80%. Both numbers, and the costs behind them, are
in ``NOTES-rerank.md``. It still stays opt-in and off by default: nothing in the
retrieval path calls it unless asked.

Two implementations behind one protocol:

    LexicalReranker       standard library only. Idf-weighted overlap over the
                          candidate pool plus a bonus for query phrases that
                          appear verbatim and in order.
    CrossEncoderReranker  the ms-marco MiniLM cross-encoder on onnxruntime.
                          Optional: needs the ``rerank`` extra and a local model
                          directory (``RERANK_DIR``).

Getting a model (about 90 MB, Apache-2.0, no torch required):

    huggingface-cli download cross-encoder/ms-marco-MiniLM-L6-v2 \\
        --include "onnx/model.onnx" "tokenizer.json" "tokenizer_config.json" \\
        --local-dir ~/.cache/models/ms-marco-MiniLM-L6-v2
    export RERANK_DIR=~/.cache/models/ms-marco-MiniLM-L6-v2
"""
from __future__ import annotations

import math
import os
import pathlib
from collections import Counter
from typing import Protocol, runtime_checkable

from .core import tokenize


class RerankerUnavailable(RuntimeError):
    """Raised when an optional reranker cannot run on this machine."""


@runtime_checkable
class Reranker(Protocol):
    """Reorder a shortlist. The pool comes from stage one, in stage-one order."""

    name: str

    def rerank(self, query: str, docs: list, top: int | None = None) -> list:
        """Return ``docs`` reordered best-first, each keeping its original keys
        plus ``rerank_score``. Stable: ties keep the incoming order."""
        ...


def _idf(df: int, n: int) -> float:
    """Same shape as BM25's idf, so a term rare in the pool weighs more."""
    return math.log(1 + (n - df + 0.5) / (df + 0.5))


class LexicalReranker:
    """Idf-weighted term overlap with a phrase bonus. No dependencies.

    Three things it can see that BM25 over a whole corpus cannot:

      * constraint conjunction. ``junior`` and ``barcelona`` are both required
        by the question, so a document holding both beats one that repeats the
        rarer half five times. ``coverage_weight`` pays for that.
      * order. ``power bi`` as two adjacent tokens is a different claim from
        ``power`` here and ``bi`` there; ``phrase_weight`` pays for that.
      * idf inside the pool. Stage one scores against every document in the
        corpus; a term that is common corpus-wide but decisive inside a
        forty-document shortlist gets its weight back here.

    ``df`` comes from the candidate pool, not the corpus: computing it over the
    corpus would need the whole index, and a reranker that needs the index is
    just a second stage-one. The pool is usually 10-40 documents, so the cost is
    negligible and it makes the reranker stateless.
    """

    name = "lexical-idf"

    def __init__(self, phrase_weight: float = 0.5, coverage_weight: float = 0.5):
        self.phrase_weight = phrase_weight
        self.coverage_weight = coverage_weight

    def scores(self, query: str, docs: list) -> list:
        q_terms = list(dict.fromkeys(tokenize(query)))
        n = len(docs)
        doc_tokens = [tokenize(d.get("text", "")) for d in docs]
        if not q_terms or not n:
            return [0.0] * n

        wanted = set(q_terms)
        df = Counter()
        for toks in doc_tokens:
            df.update(wanted.intersection(toks))
        idf = {t: _idf(df[t], n) for t in q_terms}
        total_idf = sum(idf.values()) or 1.0

        bigrams = [(a, b) for a, b in zip(q_terms, q_terms[1:])]
        out = []
        for toks in doc_tokens:
            tf = Counter(toks)
            matched = [t for t in q_terms if tf.get(t)]
            base = sum(idf[t] * (1.0 + math.log(tf[t])) for t in matched)
            coverage = sum(idf[t] for t in matched) / total_idf
            pairs = set(zip(toks, toks[1:]))
            phrase = sum(idf[a] + idf[b] for a, b in bigrams if (a, b) in pairs)
            out.append(base * (1.0 + self.coverage_weight * coverage)
                       + self.phrase_weight * phrase)
        return out

    def rerank(self, query: str, docs: list, top: int | None = None) -> list:
        docs = list(docs)
        scored = self.scores(query, docs)
        order = sorted(range(len(docs)), key=lambda i: (-scored[i], i))
        out = [{**docs[i], "rerank_score": round(scored[i], 4)} for i in order]
        return out[:top] if top is not None else out


# The candidate file names, in order of preference: full precision first, then
# the quantised export, so an int8 download still works.
MODEL_FILES = ("model.onnx", "model_quantized.onnx")
TOKENIZER_FILE = "tokenizer.json"
DEFAULT_MODEL_LABEL = "cross-encoder/ms-marco-MiniLM-L6-v2"


def find_model_file(root):
    """Look in the directory and in its ``onnx/`` subdirectory, as HF exports it."""
    for folder in (root, root / "onnx"):
        for name in MODEL_FILES:
            if (folder / name).exists():
                return folder / name
    return None


class CrossEncoderReranker:
    """ms-marco MiniLM-L6 cross-encoder over onnxruntime.

    The model scores ``[CLS] query [SEP] document [SEP]`` and returns one logit:
    higher is more relevant, no calibration needed, monotone in relevance. It is
    English-trained -- a Spanish question against a Spanish record is outside
    its distribution and the numbers in NOTES-rerank.md show it.
    """

    name = "cross-encoder-ms-marco-minilm-l6"

    def __init__(self, model_dir=None, max_len: int = 256, batch_size: int = 32):
        # Path checks first: a wrong RERANK_DIR is worth reporting on a machine
        # that has none of the optional dependencies installed either.
        raw = str(model_dir or os.getenv("RERANK_DIR") or "").strip()
        root = pathlib.Path(raw) if raw else None
        model_path = find_model_file(root) if root else None
        if root is None or not (root / TOKENIZER_FILE).exists():
            raise RerankerUnavailable(
                f"no cross-encoder in {raw or 'RERANK_DIR'}: set RERANK_DIR to a "
                f"directory holding {TOKENIZER_FILE} and one of {', '.join(MODEL_FILES)} "
                f"(export {DEFAULT_MODEL_LABEL} to ONNX, or --rerank lexical to "
                "use the stdlib reranker instead)")
        if model_path is None:
            raise RerankerUnavailable(
                f"{root} has {TOKENIZER_FILE} but no {', '.join(MODEL_FILES)}, "
                f"not in it and not in {root / 'onnx'}; this needs an ONNX export "
                f"of {DEFAULT_MODEL_LABEL}")

        try:
            import numpy as np
            import onnxruntime as ort
            from tokenizers import Tokenizer
        except ImportError as exc:  # pragma: no cover - depends on the machine
            raise RerankerUnavailable(
                "CrossEncoderReranker needs numpy, onnxruntime and tokenizers: "
                'pip install -e ".[rerank]"'
            ) from exc

        self.np = np
        self.batch_size = batch_size
        self.model_path = model_path
        self.tok = Tokenizer.from_file(str(root / TOKENIZER_FILE))
        self.tok.enable_truncation(max_length=max_len)
        self.tok.enable_padding()
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = int(os.getenv("ORT_THREADS", "8"))
        self.sess = ort.InferenceSession(str(model_path), opts,
                                         providers=["CPUExecutionProvider"])
        self.inputs = {i.name for i in self.sess.get_inputs()}

    def scores(self, query: str, docs: list) -> list:
        np = self.np
        texts = [doc.get("text", "") for doc in docs]
        out = []
        for i in range(0, len(texts), self.batch_size):
            encs = self.tok.encode_batch([(query, t) for t in texts[i:i + self.batch_size]])
            ids = np.array([e.ids for e in encs], dtype=np.int64)
            mask = np.array([e.attention_mask for e in encs], dtype=np.int64)
            feed = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in self.inputs:
                feed["token_type_ids"] = np.array([e.type_ids for e in encs], dtype=np.int64)
            logits = self.sess.run(None, feed)[0]
            out.extend(float(v) for v in logits.reshape(-1))
        return out

    def rerank(self, query: str, docs: list, top: int | None = None) -> list:
        docs = list(docs)
        scored = self.scores(query, docs)
        order = sorted(range(len(docs)), key=lambda i: (-scored[i], i))
        out = [{**docs[i], "rerank_score": round(scored[i], 4)} for i in order]
        return out[:top] if top is not None else out


NAMES = ("none", "lexical", "cross")


def get_reranker(name: str = "none", **kwargs):
    """``none`` returns None -- the default everywhere, see the module docstring."""
    name = (name or "none").lower()
    if name in ("none", "off", ""):
        return None
    if name in ("lexical", "idf"):
        return LexicalReranker(**kwargs)
    if name in ("cross", "cross-encoder", "onnx"):
        return CrossEncoderReranker(**kwargs)
    raise SystemExit(f"unknown reranker: {name} (none | lexical | cross)")


def rerank_hits(reranker, query: str, hits: list, top: int | None = None) -> list:
    """Convenience for callers that may pass ``None`` to mean "do not rerank"."""
    return list(hits[:top] if top is not None else hits) if reranker is None \
        else reranker.rerank(query, hits, top=top)
