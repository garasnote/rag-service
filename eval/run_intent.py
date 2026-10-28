#!/usr/bin/env python3
"""Intent evaluation: does the top result actually satisfy what was asked?

The entity benchmark measures "did it find the record I named", which rewards the
ability to match a company name. Real job-search queries do not name anything.
They are constraint bundles — junior, remote, Madrid, Power BI — and the right
answer is any document that meets the constraints, not one specific row.

So this harness scores satisfaction instead of recall-at-k:

    satisfied@k = share of questions where at least one of the top k documents
                  contains a term from every constraint group.

A question carries ``must``: a list of groups, OR inside a group, AND across
groups. No single expected document, because for a desperate search there isn't
one.

    python3 eval/run_intent.py
    python3 eval/run_intent.py --embedder multi --lang en
    python3 eval/run_intent.py --index ann --ef 50    # same questions, approximate search
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag.core import BM25, load_corpus, strip_accents  # noqa: E402
from rag.embed import VectorIndex, get_embedder  # noqa: E402
from rag.fusion import rrf  # noqa: E402


def satisfies(doc: dict, must: list[list[str]]) -> bool:
    text = strip_accents(doc.get("text", "").lower())
    return all(any(strip_accents(t.lower()) in text for t in group) for group in must)


def satisfied_at(hits: list[dict], must: list[list[str]], k: int) -> bool:
    return any(satisfies(d, must) for d in hits[:k])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default=str(ROOT / "eval" / "questions_intent.json"))
    ap.add_argument("--corpus", default=None)
    ap.add_argument("--embedder", default="hash", help="hash | minilm | multi")
    ap.add_argument("--lang", default="all", choices=["all", "es", "en"])
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--index", default="exact", choices=["exact", "ann"],
                    help="exact cosine (default) or the hnswlib ANN backend from rag/ann.py")
    ap.add_argument("--ef", type=int, default=50, help="query-time ef for --index ann")
    ap.add_argument("--tracker", default=None,
                    help="explicit job_tracker data.json, for running from a worktree where the default path is not visible")
    args = ap.parse_args()

    if args.corpus:
        docs = load_corpus(corpus=args.corpus)
    elif args.tracker:
        docs = load_corpus(tracker=pathlib.Path(args.tracker).expanduser())
    else:
        docs = load_corpus()
    questions = json.loads(pathlib.Path(args.questions).read_text(encoding="utf-8"))
    if args.lang != "all":
        questions = [q for q in questions if q.get("lang", args.lang) == args.lang]
    if not questions:
        print("no questions")
        return 1

    bm25 = BM25(docs)
    if args.index == "ann":
        from rag.ann import HNSWIndex
        index = HNSWIndex(docs, get_embedder(args.embedder), ef=args.ef)
        backend = f"ann(ef={args.ef})"
    else:
        index = VectorIndex(docs, get_embedder(args.embedder))
        backend = "exact"

    def rankings(q: str):
        lex = bm25.search(q, k=args.k)
        den = index.search_text(q, k=args.k)
        return {"bm25": lex, "dense": den, "hybrid": rrf([lex, den])}

    modes = ["bm25", "dense", "hybrid"]
    scores = {m: {k: 0 for k in (1, 3, args.k)} for m in modes}
    misses = {m: [] for m in modes}

    for item in questions:
        for mode, hits in rankings(item["q"]).items():
            for k in (1, 3, args.k):
                if satisfied_at(hits, item["must"], k):
                    scores[mode][k] += 1
            if not satisfied_at(hits, item["must"], args.k):
                misses[mode].append(item["q"])

    n = len(questions)
    print(f"corpus: {len(docs)} documents | questions: {n} | embedder: {index.embedder.name} | dense: {backend}")
    print(f"\n{'mode':8} {'sat@1':>7} {'sat@3':>7} {'sat@' + str(args.k):>7}")
    for mode in modes:
        row = " ".join(f"{scores[mode][k] / n:>7.0%}" for k in (1, 3, args.k))
        print(f"{mode:8} {row}")

    for mode in modes:
        if misses[mode]:
            print(f"\nunsatisfied for {mode} ({len(misses[mode])}/{n}):")
            for q in misses[mode][:6]:
                print("  -", q)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
