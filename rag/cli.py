#!/usr/bin/env python3
"""Command-line retrieval: rank documents for a query, print them with citations.

    python3 -m rag.cli "junior computer vision in Spain" --corpus fixtures/corpus.json
    python3 -m rag.cli "remote LLM roles" -k 5 --embedder multi

Answer generation lives in api.py / rag.llm; this is the retrieval half, which is
the half that can be evaluated.
"""
from __future__ import annotations

import argparse
import sys


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="rag.cli", description="rank documents for a query")
    ap.add_argument("query")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--corpus", default=None, help="standalone JSON corpus")
    ap.add_argument("--embedder", default="hash", help="hash | minilm | multi")
    ap.add_argument("--mode", default="bm25", choices=["bm25", "dense", "hybrid"])
    args = ap.parse_args(argv)

    from rag.core import BM25, load_corpus
    from rag.embed import VectorIndex, get_embedder
    from rag.fusion import rrf

    docs = load_corpus(corpus=args.corpus) if args.corpus else load_corpus()
    if not docs:
        print("no documents: pass --corpus or point TRACKER/WAREHOUSE at real data",
              file=sys.stderr)
        return 1

    bm25 = BM25(docs)
    if args.mode == "bm25":
        hits = bm25.search(args.query, k=args.k)
    else:
        index = VectorIndex(docs, get_embedder(args.embedder))
        dense = index.search_text(args.query, k=args.k)
        hits = dense if args.mode == "dense" else rrf([bm25.search(args.query, k=args.k), dense])

    for n, doc in enumerate(hits[: args.k], 1):
        print(f"[{n}] {doc.get('title', '')}")
        if doc.get("url"):
            print(f"    {doc['url']}")
        print(f"    {doc.get('text', '')[:160]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
