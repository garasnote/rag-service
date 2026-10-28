#!/usr/bin/env python3
"""Retrieval eval: hit@1, hit@3 and MRR over hand-written questions with known answers.

Compares four configurations on the same questions:
  bm25        lexical ranking (Okapi BM25)
  dense       hashed bag-of-words embeddings + cosine
  hybrid      reciprocal rank fusion of the two
  hyde*       hybrid, but the query is replaced by a hypothetical passage

The hyde row without an LLM key is an *oracle* upper bound: it stands in for a
perfect hypothetical passage, meaning "this is the ceiling HyDE could reach".
Run with --hyde and an API key to measure the real thing.
"""
import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rag.core import BM25, load_corpus, strip_accents  # noqa: E402
from rag.embed import VectorIndex, get_embedder  # noqa: E402
from rag.fusion import rrf  # noqa: E402


def matches(doc, expect):
    text = strip_accents(doc["text"].lower())
    return any(strip_accents(e.lower()) in text for e in expect)


def score(rankings, questions, k=5):
    h1 = h3 = 0
    rr = 0.0
    misses = []
    for item, hits in zip(questions, rankings):
        rank = next((i for i, h in enumerate(hits[:k], 1) if matches(h, item["expect"])), None)
        if rank == 1:
            h1 += 1
        if rank and rank <= 3:
            h3 += 1
        if rank:
            rr += 1 / rank
        else:
            misses.append(item["q"])
    n = len(questions)
    return {"hit1": h1 / n, "hit3": h3 / n, "mrr": rr / n, "misses": misses}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hyde", action="store_true", help="use the LLM instead of the oracle stand-in")
    ap.add_argument("--embedder", default="hash", help="hash | minilm | api")
    ap.add_argument("--lang", default="all", choices=["all", "es", "en"],
                    help="restrict to one language when the question set is tagged")
    ap.add_argument("--questions", default=None,
                    help="JSON question set (e.g. fixtures/questions.json)")
    ap.add_argument("--corpus", default=None,
                    help="standalone JSON corpus (e.g. fixtures/corpus.json) instead of the private tracker")
    args = ap.parse_args()

    docs = load_corpus(corpus=args.corpus) if args.corpus else load_corpus()
    bm25 = BM25(docs)
    dense = VectorIndex(docs, embedder=get_embedder(args.embedder))
    questions = json.loads(open(args.questions, encoding="utf-8").read()) if getattr(args, "questions", None) else json.loads((ROOT / "eval" / "questions.json").read_text(encoding="utf-8"))
    if getattr(args, "lang", "all") != "all":
        questions = [q for q in questions if q.get("lang", args.lang) == args.lang]

    client = None
    if args.hyde:
        from rag.llm import OpenAICompat, LLMError
        try:
            client = OpenAICompat()
        except LLMError as exc:
            sys.exit(f"--hyde requires an LLM: {exc}")

    def hyde_text(item):
        if client:
            from rag.hyde import hypothetical
            return hypothetical(item["q"], client)
        # oracle: the text of the document that should be found, truncated
        for d in docs:
            if matches(d, item["expect"]):
                return d["text"][:600]
        return item["q"]

    conf = {"bm25": [], "dense": [], "hybrid": [], "hybrid3:1": [], "hyde": []}
    for item in questions:
        lex = bm25.search(item["q"], k=10)
        den = dense.search_text(item["q"], k=10)
        conf["bm25"].append(lex[:5])
        conf["dense"].append(den[:5])
        conf["hybrid"].append(rrf([lex, den], top=5))
        conf["hybrid3:1"].append(rrf([lex, den], top=5, weights=[3.0, 1.0]))
        hq = hyde_text(item)
        conf["hyde"].append(rrf([bm25.search(hq, k=10), dense.search_text(hq, k=10)], top=5))

    print(f"corpus: {len(docs)} documents | questions: {len(questions)} | embedder: {dense.embedder.name}")
    print(f"hyde    : {'LLM (' + client.model + ')' if client else 'oracle stand-in (ceiling, not a measurement)'}\n")
    print(f"{'config':8} {'hit@1':>6} {'hit@3':>6} {'MRR':>6}")
    results = {}
    for name, rankings in conf.items():
        r = score(rankings, questions)
        results[name] = r
        print(f"{name:8} {r['hit1']:>6.0%} {r['hit3']:>6.0%} {r['mrr']:>6.3f}")
    best = max(results, key=lambda n: results[n]["mrr"])
    print(f"\nbest by MRR: {best}")
    for name in ("bm25", "hybrid"):
        if results[name]["misses"]:
            print(f"\nfailures for {name}:")
            for m in results[name]["misses"]:
                print("  -", m)


if __name__ == "__main__":
    main()
