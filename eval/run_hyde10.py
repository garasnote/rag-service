#!/usr/bin/env python3
"""HyDE measured for real: N samples per question, fused, scored.

The eval in run_eval.py falls back to an oracle passage when there is no LLM
key, which makes the HyDE row a ceiling. This script does the honest thing:

  for each question:
      generate N hypothetical passages with the LLM (temperature 0.9)
      retrieve with BM25 + dense on each passage, fuse with RRF
      fuse the N rankings again, and score hit@1 / hit@3 / MRR

Both numbers are reported: the fused result (what a user would get) and the
mean over single samples (what one call buys). Samples are cached so re-runs
cost nothing.

    LLM_API_KEY=... python3 eval/run_hyde10.py --samples 10
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import statistics
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rag.core import BM25, load_corpus  # noqa: E402
from rag.embed import VectorIndex, get_embedder  # noqa: E402
from rag.fusion import rrf  # noqa: E402
from rag.hyde import build_prompt  # noqa: E402
from rag.llm import LLMError, OpenAICompat  # noqa: E402
from run_eval import matches, score  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=None, help="standalone JSON corpus instead of the private tracker")
    ap.add_argument("--samples", type=int, default=10)
    ap.add_argument("--embedder", default="hash")
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--model", default=os.getenv("LLM_MODEL", "deepseek-chat"))
    ap.add_argument("--base-url", default=os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1"))
    args = ap.parse_args()

    key = os.getenv("LLM_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
    if not key:
        sys.exit("no DEEPSEEK_API_KEY / LLM_API_KEY in the environment")

    docs = load_corpus(corpus=args.corpus) if getattr(args,"corpus",None) else load_corpus()
    questions = json.loads((ROOT / "eval" / "questions.json").read_text(encoding="utf-8"))
    bm25 = BM25(docs)
    dense = VectorIndex(docs, embedder=get_embedder(args.embedder))
    client = OpenAICompat(base_url=args.base_url, api_key=key, model=args.model)

    cache = ROOT / ".cache" / f"hyde-{args.model}-{args.samples}.json"
    cache.parent.mkdir(exist_ok=True)
    samples = json.loads(cache.read_text()) if cache.exists() else {}

    for item in questions:
        if item["q"] in samples and len(samples[item["q"]]) >= args.samples:
            continue
        passages = []
        for i in range(args.samples):
            try:
                passages.append(client.chat([{"role": "user", "content": build_prompt(item["q"])}],
                                            temperature=args.temperature))
            except LLMError as exc:
                print(f"  llm error on {item['q'][:40]}: {exc}")
                break
        samples[item["q"]] = passages
        print(f"  {len(passages)}/{args.samples} passages for: {item['q'][:60]}")
        cache.write_text(json.dumps(samples, ensure_ascii=False, indent=1))

    fused, per_sample = [], []
    for item in questions:
        rankings = []
        for passage in samples.get(item["q"], [])[:args.samples]:
            rankings.append(rrf([bm25.search(passage, k=10), dense.search_text(passage, k=10)], top=10))
        if not rankings:
            fused.append([])
            per_sample.append(0.0)
            continue
        fused.append(rrf(rankings, top=5))
        hits = [score([r[:5]], [item]) for r in rankings]
        per_sample.append(statistics.mean(h["mrr"] for h in hits))

    base = {
        "bm25": [bm25.search(i["q"], k=5) for i in questions],
        "dense": [dense.search_text(i["q"], k=5) for i in questions],
        "hybrid": [rrf([bm25.search(i["q"], k=10), dense.search_text(i["q"], k=10)], top=5) for i in questions],
    }

    print(f"\ncorpus: {len(docs)} documents | questions: {len(questions)} | "
          f"embedder: {dense.embedder.name} | samples: {args.samples} | model: {client.model}")
    print(f"{'config':22} {'hit@1':>6} {'hit@3':>6} {'MRR':>6}")
    for name, rankings in base.items():
        r = score(rankings, questions)
        print(f"{name:22} {r['hit1']:>6.0%} {r['hit3']:>6.0%} {r['mrr']:>6.3f}")
    r = score(fused, questions)
    print(f"{'hyde fused (' + str(args.samples) + 'x)':22} {r['hit1']:>6.0%} {r['hit3']:>6.0%} {r['mrr']:>6.3f}")
    print(f"{'hyde single-call mean':22} {'':>6} {'':>6} {statistics.mean(per_sample):>6.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
