#!/usr/bin/env python3
"""Command-line retrieval: rank documents for a query, print them with citations.

    python3 -m rag.cli "junior computer vision in Spain" --corpus fixtures/corpus.json
    python3 -m rag.cli "remote LLM roles" -k 5 --embedder multi

Reranking is opt-in and off by default: it costs a forward pass per candidate
and, on this corpus, it pays for itself only if you ask for it. See
NOTES-rerank.md for the measurement before flipping the flag.

    python3 -m rag.cli "¿algo de BI junior?" --rerank lexical
    python3 -m rag.cli "junior data jobs in Barcelona" --rerank cross --pool 20

Answer generation lives in api.py / rag.llm; this is the retrieval half, which is
the half that can be evaluated.
"""
from __future__ import annotations

import argparse
import sys


def main(argv=None) -> int:
    from rag.core import BM25, load_corpus
    from rag.embed import VectorIndex, get_embedder
    from rag.fusion import rrf
    from rag.rerank import NAMES, RerankerUnavailable, get_reranker, rerank_hits

    ap = argparse.ArgumentParser(prog="rag.cli", description="rank documents for a query")
    ap.add_argument("query")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--corpus", default=None, help="standalone JSON corpus")
    ap.add_argument("--parquet", default=None, help="harvested parquet snapshot (needs the parquet extra)")
    ap.add_argument("--limit", type=int, default=0, help="cap documents read from --parquet")
    ap.add_argument("--embedder", default="hash", help="hash | minilm | multi")
    ap.add_argument("--mode", default="bm25", choices=["bm25", "dense", "hybrid"])
    ap.add_argument("--rerank", default="none", choices=list(NAMES),
                    help="opt-in second stage over the shortlist (default: none)")
    ap.add_argument("--pool", type=int, default=20,
                    help="shortlist size handed to the reranker (-k when <= 0)")
    args = ap.parse_args(argv)

    if args.parquet:
        from rag.parquet import load as load_parquet
        docs = load_parquet(args.parquet, limit=args.limit or None)
    else:
        docs = load_corpus(corpus=args.corpus) if args.corpus else load_corpus()
    if not docs:
        print("no documents: pass --corpus or point TRACKER/WAREHOUSE at real data",
              file=sys.stderr)
        return 1

    try:
        reranker = get_reranker(args.rerank)
    except RerankerUnavailable as exc:
        print(f"reranker unavailable: {exc}", file=sys.stderr)
        return 1
    pool = args.pool if args.pool > 0 else args.k
    depth = max(args.k, pool) if reranker is not None else args.k

    bm25 = BM25(docs)
    if args.mode == "bm25":
        hits = bm25.search(args.query, k=depth)
    else:
        index = VectorIndex(docs, get_embedder(args.embedder))
        dense = index.search_text(args.query, k=depth)
        hits = dense if args.mode == "dense" else rrf([bm25.search(args.query, k=depth), dense],
                                                      top=depth)
    if reranker is not None:
        hits = rerank_hits(reranker, args.query, hits, top=args.k)

    for n, doc in enumerate(hits[: args.k], 1):
        print(f"[{n}] {doc.get('title', '')}")
        if doc.get("url"):
            print(f"    {doc['url']}")
        print(f"    {doc.get('text', '')[:160]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
