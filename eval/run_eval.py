#!/usr/bin/env python3
"""Retrieval eval: hit@1, hit@3 and MRR over hand-written questions with known answers."""
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rag.core import BM25, load_corpus, strip_accents  # noqa: E402


def matches(doc, expect):
    text = strip_accents(doc["text"].lower())
    return any(strip_accents(e.lower()) in text for e in expect)


def main():
    docs = load_corpus()
    index = BM25(docs)
    questions = json.loads((ROOT / "eval" / "questions.json").read_text(encoding="utf-8"))
    h1 = h3 = 0
    rr = 0.0
    misses = []
    for item in questions:
        hits = index.search(item["q"], k=5)
        rank = next((i for i, h in enumerate(hits, 1) if matches(h, item["expect"])), None)
        if rank == 1:
            h1 += 1
        if rank and rank <= 3:
            h3 += 1
        if rank:
            rr += 1 / rank
        else:
            misses.append(item["q"])
    n = len(questions)
    print(f"corpus: {len(docs)} documentos | preguntas: {n}")
    print(f"hit@1 : {h1}/{n} = {h1/n:.0%}")
    print(f"hit@3 : {h3}/{n} = {h3/n:.0%}")
    print(f"MRR   : {rr/n:.3f}")
    if misses:
        print("\nfallos:")
        for m in misses:
            print("  -", m)


if __name__ == "__main__":
    main()
