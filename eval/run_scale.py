#!/usr/bin/env python3
"""What brute-force retrieval costs at scale: index build, memory, latency.

    python3 eval/run_scale.py --parquet /path/to/postings_all.parquet

Needs the parquet extra. Nothing here is tuned — it is the same BM25 the rest of
the repo uses, on a corpus five times the size of the fixture one.
"""
import argparse
import json
import os
import pathlib
import random
import resource
import statistics
import sys
import time

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from rag.core import BM25     # noqa: E402
from rag.parquet import load  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument(
    "--parquet",
    default=os.environ.get("POSTINGS_PARQUET")
    or str(pathlib.Path.home() / "Documents/JobApp/data/market/postings_all.parquet"),
)
ap.add_argument("--chars", type=int, default=600)
ap.add_argument("--queries", type=int, default=40)
args = ap.parse_args()
random.seed(5)

t0 = time.time()
docs = load(args.parquet, max_chars=args.chars)
print(f"documents {len(docs)} in {time.time()-t0:.0f}s", flush=True)
print(f"median text {int(statistics.median(len(d['text']) for d in docs))} chars", flush=True)

t1 = time.time()
bm25 = BM25(docs)
build = time.time() - t1
rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024
print(f"bm25 built in {build:.0f}s · peak RSS {rss:.1f} GB", flush=True)

sample = random.sample(range(len(docs)), min(args.queries, len(docs)))
lat, hits = [], {1: 0, 5: 0}
for i in sample:
    q = f'{docs[i]["title"]} {docs[i]["company"]}'.strip()
    t = time.time()
    res = bm25.search(q, 5)
    lat.append(time.time() - t)
    rank = next((n + 1 for n, h in enumerate(res) if h["id"] == docs[i]["id"]), None)
    if rank:
        hits[1] += rank == 1
        hits[5] += rank <= 5
n = len(sample)
print(f"\nquery latency: median {statistics.median(lat):.2f}s  p90 {sorted(lat)[int(0.9*n)-1]:.2f}s  max {max(lat):.2f}s")
print(f"hit@1 {hits[1]/n:.2f} · hit@5 {hits[5]/n:.2f} · n={n}")
json.dump({"query": "title + company", "documents": len(docs), "build_s": round(build, 1), "peak_rss_gb": round(rss, 2),
           "median_latency_s": round(statistics.median(lat), 2), "p90_latency_s": round(sorted(lat)[int(0.9*n)-1], 2),
           "hit@1": round(hits[1]/n, 3), "hit@5": round(hits[5]/n, 3), "n": n},
          open("/tmp/run_scale.json", "w"), indent=1)
print("wrote /tmp/run_scale.json")
