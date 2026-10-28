#!/usr/bin/env python3
"""Reranking eval: is a second pass over the shortlist worth its cost?

Stage one returns a shortlist; a reranker reorders it. So the only honest
question is: after reranking the top ``p``, does a question get satisfied at
rank 1, 3 or 5 more often than before?

The ``none`` row is the unreranked shortlist truncated to five -- the thing the
reranker has to beat. It is identical in every pool row on purpose: a reranker
that loses to it at p=40 lost, and no amount of extra pool changes that.

Two question sets, because they measure different things and a reranker can
improve one while wrecking the other:

  --metric sat  eval/questions_intent.json, constraint satisfaction (default)
  --metric hit  eval/questions.json, did the specifically named record surface

    python3 eval/run_rerank.py                       # stdlib reranker, no model
    python3 eval/run_rerank.py --rerank cross        # needs RERANK_DIR + extra
    python3 eval/run_rerank.py --metric hit --rerank cross

The scores are imported from eval/run_intent.py and eval/run_eval.py rather than
copied: two implementations of "satisfied" would drift, and then the tables
would stop being comparable.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

from rag.core import BM25, load_corpus  # noqa: E402
from rag.embed import VectorIndex, get_embedder  # noqa: E402
from rag.fusion import rrf  # noqa: E402
from rag.rerank import RerankerUnavailable, get_reranker  # noqa: E402
from run_eval import matches  # noqa: E402
from run_intent import satisfied_at  # noqa: E402

KS = (1, 3, 5)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default=None,
                    help="default: eval/questions_intent.json for --metric sat, "
                         "eval/questions.json for --metric hit")
    ap.add_argument("--metric", default="sat", choices=["sat", "hit"],
                    help="sat: constraint satisfaction (questions_intent.json); "
                         "hit: did the named record surface (questions.json)")
    ap.add_argument("--corpus", default=None)
    ap.add_argument("--embedder", default="hash", help="hash | minilm | multi")
    ap.add_argument("--lang", default="all", choices=["all", "es", "en"])
    ap.add_argument("--rerank", default="lexical", choices=["none", "lexical", "cross"])
    ap.add_argument("--pools", default="10,20,40", help="comma-separated pool sizes")
    ap.add_argument("--phrase-weight", type=float, default=0.5,
                    help="lexical reranker: weight of a verbatim query phrase")
    ap.add_argument("--coverage-weight", type=float, default=0.5,
                    help="lexical reranker: weight of matching every query term")
    ap.add_argument("--timing", action="store_true",
                    help="also report median milliseconds per query, stage one vs rerank")
    args = ap.parse_args()

    if args.questions is None:
        name = "questions_intent.json" if args.metric == "sat" else "questions.json"
        args.questions = str(ROOT / "eval" / name)

    docs = load_corpus(corpus=args.corpus) if args.corpus else load_corpus()
    if not docs:
        print("no documents: pass --corpus, or run from a tree where the private "
              "tracker and warehouse resolve", file=sys.stderr)
        return 1
    questions = json.loads(pathlib.Path(args.questions).read_text(encoding="utf-8"))
    key = "must" if args.metric == "sat" else "expect"
    questions = [q for q in questions if key in q]
    if args.lang != "all":
        questions = [q for q in questions if q.get("lang", args.lang) == args.lang]
    if not questions:
        print(f"no {key!r} questions in {args.questions}")
        return 1

    def is_satisfied(ranked, item, k):
        if args.metric == "sat":
            return satisfied_at(ranked, item["must"], k)
        return any(matches(d, item["expect"]) for d in ranked[:k])

    pools = [int(p) for p in args.pools.split(",") if p.strip()]

    try:
        kwargs = {"phrase_weight": args.phrase_weight,
                  "coverage_weight": args.coverage_weight} if args.rerank == "lexical" else {}
        reranker = get_reranker(args.rerank, **kwargs)
    except RerankerUnavailable as exc:
        print(f"reranker unavailable: {exc}", file=sys.stderr)
        return 2

    bm25 = BM25(docs)
    index = VectorIndex(docs, get_embedder(args.embedder))

    def base_hits(item, p):
        """Stage one at width p: each retriever returns p, RRF fuses, keep p.

        For bm25 the first five are the same at every width. For hybrid they are
        not: a wider fusion lets a document that only one retriever found into
        the head, so the ``none`` row is the baseline *at that width*, and the
        comparison that matters is inside a pool row.
        """
        lex = bm25.search(item["q"], k=p)
        den = index.search_text(item["q"], k=p)
        return {"bm25": lex, "hybrid": rrf([lex, den], top=p)}

    # scores[(mode, pool, which)][k] = number of satisfied questions
    scores, changed = {}, {}
    stage_ms, rerank_ms = {}, {}
    for item in questions:
        for p in pools:
            t0 = time.perf_counter()
            bases = base_hits(item, p)
            stage_ms.setdefault(p, []).append((time.perf_counter() - t0) * 1000)
            for mode, hits in bases.items():
                pool = hits[:p]
                rows = {"none": pool[:5]}
                if reranker is not None:
                    t1 = time.perf_counter()
                    rows[reranker.name] = reranker.rerank(item["q"], pool, top=5)
                    rerank_ms.setdefault(p, []).append((time.perf_counter() - t1) * 1000)
                    if [d["id"] for d in rows[reranker.name]] != [d["id"] for d in pool[:5]]:
                        changed[(mode, p)] = changed.get((mode, p), 0) + 1
                for which, ranked in rows.items():
                    for k in KS:
                        cell_key = (mode, p, which)
                        scores.setdefault(cell_key, {kk: 0 for kk in KS})
                        if is_satisfied(ranked, item, k):
                            scores[cell_key][k] += 1

    n = len(questions)
    label = reranker.name if reranker is not None else "none"
    width = max(len(label), len("none"), 10) + 1
    header = " ".join(f"{args.metric + '@' + str(k):>7}" for k in KS)
    print(f"corpus: {len(docs)} documents | questions: {n} | embedder: {index.embedder.name}")
    print(f"reranker: {label}" + (f" (phrase={args.phrase_weight} coverage={args.coverage_weight})"
                                  if args.rerank == "lexical" else ""))
    print("pools widen stage one, so 'none' is the unreranked baseline at that width; "
          "compare inside a pool row")
    print()
    print(f"{'base':7} {'pool':>5} {'reranker':{width}} {header}")
    for mode in ("bm25", "hybrid"):
        for p in pools:
            for which in (["none", label] if args.rerank != "none" else ["none"]):
                row = scores[(mode, p, which)]
                cells = " ".join(f"{row[k] / n:>7.0%}" for k in KS)
                print(f"{mode:7} {p:>5} {which:{width}} {cells}")
            if reranker is not None:
                print(f"{'':7} {'':>5} {'top-5 changed':{width}} "
                      f"{changed.get((mode, p), 0)}/{n} shortlists")
    if args.timing:
        print(f"\nmedian milliseconds per query (n={n}):")
        print(f"{'pool':>5} {'stage one':>10} {label:>{width}}")
        for p in pools:
            reranked = statistics.median(rerank_ms[p]) if p in rerank_ms else 0.0
            print(f"{p:>5} {statistics.median(stage_ms[p]):>10.2f} "
                  f"{('%.2f' % reranked) if reranker is not None else 'off':>{width}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
