#!/usr/bin/env python3
"""What does approximation cost? Measured recall against exact cosine, and latency.

Two dense backends, same corpus, same queries:

    exact   rag.embed.VectorIndex   brute-force cosine over every document, stdlib loop
    numpy   the same exhaustive scan with matrix @ query, no approximation (recall 1.0)
    ann     rag.ann.HNSWIndex       hnswlib graph, swept over query-time ef

The numpy row exists because "ANN is 100x faster" is only meaningful against a
baseline that is not a Python loop: it separates "approximation buys speed" from
"a C kernel buys speed".

recall@k is measured, not assumed: the exact index produces the reference top-k
for every query and each other ranking is credited |hits & exact_top-k| /
|exact_top-k|. The mean over queries is the recall column.

Timing: every backend is sampled inside the same repeat (repeat is the outer
loop, backends the inner one), so drift in machine load hits all rows instead of
whipping whichever row happened to be measured last. Mean, median and p95 are
all printed because a mean over a busy box is dominated by scheduler preemption.

Build time is the time until the backend can answer a query, embeddings from the
on-disk cache included, so both rows mean the same thing. The ANN row also shows
how much of that is the graph itself.

    python3 eval/bench_ann.py
    python3 eval/bench_ann.py --embedder multi
    python3 eval/bench_ann.py --corpus fixtures/corpus.json
    python3 eval/bench_ann.py --tracker ~/Documents/JobApp/job_tracker/data.json

Needs numpy + hnswlib: run it with the venv python, not the stdlib one.
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

from rag.core import load_corpus  # noqa: E402
from rag.embed import VectorIndex, get_embedder  # noqa: E402

DEFAULT_QUESTIONS = ("eval/questions_intent.json", "eval/questions.json")


def resolve_corpus(args) -> tuple:
    """Real tracker when it is visible, fixtures otherwise. Returns (docs, source label)."""
    if args.corpus:
        path = pathlib.Path(args.corpus).expanduser()
        return load_corpus(corpus=path), str(path)
    if args.tracker:
        path = pathlib.Path(args.tracker).expanduser()
        return load_corpus(tracker=path), str(path)
    docs = load_corpus()
    if docs:
        return docs, "tracker/warehouse defaults"
    fixture = ROOT / "fixtures" / "corpus.json"
    return load_corpus(corpus=fixture), f"fixtures fallback ({fixture})"


def load_queries(paths) -> list:
    queries, seen = [], set()
    for path in paths:
        file = pathlib.Path(path)
        if not file.is_absolute():
            file = ROOT / file
        if not file.exists():
            continue
        for item in json.loads(file.read_text(encoding="utf-8")):
            q = item.get("q") if isinstance(item, dict) else None
            if q and q not in seen:
                seen.add(q)
                queries.append(q)
    return queries


def measure(backends, queries, k, repeats) -> dict:
    """Interleaved sampling. backends: [(label, prepare, search)]; prepare may be None."""
    for _, prepare, search in backends:  # warm the graph, the embedder and the caches
        if prepare:
            prepare()
        for q in queries:
            search(q, k)
    samples = {label: [] for label, _, _ in backends}
    for _ in range(repeats):
        for label, prepare, search in backends:
            if prepare:
                prepare()
            bucket = samples[label]
            for q in queries:
                started = time.perf_counter()
                search(q, k)
                bucket.append((time.perf_counter() - started) * 1000.0)
    out = {}
    for label, values in samples.items():
        values.sort()
        p95 = values[min(len(values) - 1, int(round(0.95 * (len(values) - 1))))]
        out[label] = (statistics.fmean(values), statistics.median(values), p95)
    return out


def recall_against(ann_hits, exact_hits) -> tuple:
    """recall@k for one query, plus the reference size so a short one stays visible."""
    exact_ids = {d["id"] for d in exact_hits}
    if not exact_ids:
        return 0.0, 0
    ann_ids = {d["id"] for d in ann_hits}
    return len(exact_ids & ann_ids) / len(exact_ids), len(exact_ids)


def load_average() -> str:
    try:
        return pathlib.Path("/proc/loadavg").read_text().split()[0]
    except OSError:
        return "?"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=None, help="standalone JSON list of {id,text} rows")
    ap.add_argument("--tracker", default=None, help="path to a job_tracker data.json")
    ap.add_argument("--embedder", default="hash", help="hash | minilm | multi")
    ap.add_argument("--questions", default=None,
                    help="comma-separated question files (default: %s)" % ",".join(DEFAULT_QUESTIONS))
    ap.add_argument("-k", type=int, default=10, help="cutoff for recall and latency")
    ap.add_argument("--efs", default="10,25,50,100", help="query-time ef values to sweep")
    ap.add_argument("--repeats", type=int, default=20, help="passes over the query set for the timing")
    args = ap.parse_args()

    try:
        from rag.ann import HNSWIndex
    except RuntimeError as exc:
        print(f"ann benchmark unavailable: {exc}")
        return 1

    docs, source = resolve_corpus(args)
    if not docs:
        print("no corpus: pass --corpus or --tracker")
        return 1
    questions = args.questions.split(",") if args.questions else list(DEFAULT_QUESTIONS)
    queries = load_queries(questions)
    if not queries:
        print("no queries: pass --questions")
        return 1
    efs = [int(x) for x in args.efs.split(",") if x.strip()]

    try:  # a process cost, not a per-index cost, and the ANN needs it anyway
        import numpy as np
    except ImportError:
        np = None

    embedder = get_embedder(args.embedder)
    started = time.perf_counter()
    exact = VectorIndex(docs, embedder)
    exact_build_ms = (time.perf_counter() - started) * 1000.0
    started = time.perf_counter()
    ann = HNSWIndex(docs, embedder, ef=max(efs))
    ann_build_ms = (time.perf_counter() - started) * 1000.0

    reference = [exact.search_text(q, k=args.k) for q in queries]

    # A third backend: the same exhaustive scan as VectorIndex, done with numpy instead of a
    # Python loop. Not approximation, so it is the honest yardstick for "is HNSW fast?" --
    # benchmarking against a Python loop flatters any C implementation.
    if np is not None:
        matrix = np.asarray(exact.vectors, dtype=np.float32)

        def numpy_search(text, k=5):
            scores = matrix @ np.asarray(embedder.vector(text), dtype=np.float32)
            cut = min(int(k), len(docs))
            top = np.argpartition(-scores, cut - 1)[:cut]
            top = top[np.argsort(-scores[top])]
            hits = []
            for i in top:
                score = float(scores[i])
                if score <= 0:  # same filter as VectorIndex, so top-k sets are comparable
                    continue
                hits.append({"score": round(score, 4), **docs[int(i)]})
            return hits

    rows: list = []
    if np is not None:
        rows.append({"label": "numpy", "ef": "-", "key": "numpy", "search": numpy_search,
                     "prepare": None, "build": exact_build_ms})
    rows.insert(0, {"label": "exact", "ef": "-", "key": "exact", "search": exact.search_text,
                    "prepare": None, "build": exact_build_ms})
    for ef in efs:

        def prepare(ef=ef):
            ann.ef = ef

        rows.append({"label": "ann", "ef": str(ef), "key": f"ann@{ef}", "search": ann.search_text,
                     "prepare": prepare, "build": ann_build_ms})
    timings = measure([(r["key"], r["prepare"], r["search"]) for r in rows],
                      queries, args.k, args.repeats)

    print(f"corpus: {len(docs)} documents from {source}")
    print(f"embedder: {embedder.name} | queries: {len(queries)} from {','.join(questions)}")
    print(f"ann graph: {ann.dim} dimensions, M={ann.M}, ef_construction={ann.ef_construction}, "
          f"graph {ann.build_s * 1000:.1f} ms of the {ann_build_ms:.1f} ms total")
    print(f"timing: {args.repeats} interleaved passes over the query set | load average: {load_average()}")
    print()
    header = (f"{'backend':8} {'ef':>4} {'recall@' + str(args.k):>10} {'mean ms':>9} "
              f"{'med ms':>8} {'p95 ms':>8} {'build ms':>9}")
    print(header)
    print("-" * len(header))

    totals, last_mean, exact_mean = {"short": 0, "pairs": 0}, None, timings["exact"][0]
    for row in rows:
        if row["prepare"]:
            row["prepare"]()  # so the recall pass below uses the same ef as the timing pass
        recalls = []
        for query, ref in zip(queries, reference):
            recall, size = recall_against(row["search"](query, args.k), ref)
            recalls.append(recall)
            totals["short"] += size < args.k
            totals["pairs"] += 1
        mean_ms, med_ms, p95_ms = timings[row["key"]]
        last_mean = mean_ms
        print(f"{row['label']:8} {row['ef']:>4} {statistics.fmean(recalls):>10.2f} {mean_ms:>9.3f} "
              f"{med_ms:>8.3f} {p95_ms:>8.3f} {row['build']:>9.3f}")

    if totals["short"]:
        print(f"\nnote: {totals['short']}/{totals['pairs']} row/query pairs have fewer than "
              f"{args.k} exact hits (cosine <= 0 filtered), so those recalls are relative to "
              f"the available set.")
    ratio = last_mean / exact_mean if exact_mean else 0.0
    print(f"ann(ef={efs[-1]}) mean latency is {ratio:.2f}x the exact mean")
    print(f"load average after: {load_average()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
