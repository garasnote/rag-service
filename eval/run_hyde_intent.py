#!/usr/bin/env python3
"""HyDE on intent queries, scored the way intent queries should be scored.

Each question is turned into N hypothetical *records* (or prose passages, with
--style prose), each sample retrieves, and the N rankings are fused with RRF.
A hit is any document in the top-k that satisfies the question's constraints.

    DEEPSEEK_API_KEY=... python3 eval/run_hyde_intent.py --embedder multi --samples 6
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag.core import BM25, load_corpus  # noqa: E402
from rag.embed import VectorIndex, get_embedder  # noqa: E402
from rag.fusion import rrf  # noqa: E402
from rag.hyde import build_prompt  # noqa: E402
from rag.llm import OpenAICompat  # noqa: E402
from run_intent import satisfies  # noqa: E402

CACHE = ROOT / ".cache"


def samples_for(client, questions, style, n, temperature):
    """Generate n hypotheticals per question, cached so a re-run costs nothing."""
    CACHE.mkdir(exist_ok=True)
    path = CACHE / f"hyde-{style}-{client.model}-{n}.json"
    cache = json.loads(path.read_text()) if path.exists() else {}

    errors = []

    def one(args):
        q, _ = args
        out = []
        for _ in range(n):
            for attempt in range(4):
                try:
                    out.append(client.chat([{"role": "user", "content": build_prompt(q, style)}],
                                           temperature=temperature).strip())
                    break
                except Exception as exc:                       # noqa: BLE001
                    errors.append(f"{type(exc).__name__}: {exc}")
                    time.sleep(1.5 * (attempt + 1))
        return q, out

    todo = [(q, None) for q in questions if len(cache.get(q, [])) < n]
    if todo:
        with ThreadPoolExecutor(max_workers=3) as pool:
            for q, out in pool.map(one, todo):
                if out:
                    cache[q] = out
        path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
        if errors:
            print(f"{len(errors)} failed calls, last: {errors[-1]}", file=sys.stderr)
    return cache


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default=str(ROOT / "eval" / "questions_intent.json"))
    ap.add_argument("--corpus", default=None)
    ap.add_argument("--embedder", default="multi")
    ap.add_argument("--style", default="record", choices=["record", "prose"])
    ap.add_argument("--samples", type=int, default=6)
    ap.add_argument("--hypotheticals", default=None,
                    help="pre-generated {question: [samples]} JSON; skips the LLM entirely")
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--model", default=os.getenv("LLM_MODEL", "deepseek-chat"))
    ap.add_argument("-k", type=int, default=5)
    args = ap.parse_args()

    key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("LLM_API_KEY") or ""
    if not key:
        print("no DEEPSEEK_API_KEY / LLM_API_KEY in the environment")
        return 1
    client = OpenAICompat(model=args.model, api_key=key)
    client.model = args.model

    docs = load_corpus(corpus=args.corpus) if args.corpus else load_corpus()
    questions = json.loads(pathlib.Path(args.questions).read_text(encoding="utf-8"))
    bm25 = BM25(docs)
    index = VectorIndex(docs, get_embedder(args.embedder))
    if args.hypotheticals:
        cache = json.loads(pathlib.Path(args.hypotheticals).read_text(encoding="utf-8"))
    else:
        cache = samples_for(client, [q["q"] for q in questions], args.style, args.samples, args.temperature)

    def rate(mode, k):
        ok = 0
        for item in questions:
            q = item["q"]
            if mode == "bm25":
                hits = bm25.search(q, k=k)
            elif mode == "hybrid":
                hits = rrf([bm25.search(q, k=k), index.search_text(q, k=k)])
            elif mode == "hyde_fused":
                rankings = [index.search_text(s, k=k) for s in cache.get(q, []) if s]
                hits = rrf(rankings) if rankings else []
            else:  # single best sample
                first = (cache.get(q) or [""])[0]
                hits = index.search_text(first, k=k) if first else []
            ok += 1 if any(satisfies(d, item["must"]) for d in hits[:k]) else 0
        return ok / len(questions)

    n = len(questions)
    rows = []
    for mode in ("bm25", "hybrid", "hyde_fused", "hyde_1x"):
        rows.append((mode, rate(mode, 1), rate(mode, 3), rate(mode, args.k)))

    print(f"corpus: {len(docs)} documents | questions: {n} | embedder: {index.embedder.name} "
          f"| style: {args.style} | samples: {args.samples}")
    print(f"\n{'mode':12} {'sat@1':>7} {'sat@3':>7} {'sat@' + str(args.k):>7}")
    for mode, a, b, c in rows:
        print(f"{mode:12} {a:>7.0%} {b:>7.0%} {c:>7.0%}")

    print("\nsample hypotheticals:")
    for q in [i["q"] for i in questions[:3]]:
        got = (cache.get(q) or [""])[0]
        print(f"  {q}\n    -> {got[:110]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
