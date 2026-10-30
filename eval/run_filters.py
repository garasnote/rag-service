#!/usr/bin/env python3
"""Filter-derived ground truth: metadata constraints as gold sets, no hand labels.

    python3 eval/run_filters.py --parquet /path/to/postings_all.parquet

A query is a sentence a person would type. Its gold set is every posting that
satisfies the constraints the sentence names — location, role family, seniority —
so relevance is decided by a rule, not by a human. Two systems answer each query:

  text      BM25 over the whole corpus, no filter
  filtered  BM25 inside the location pool, which is what a product would do
"""
import argparse
import json
import os
import pathlib
import re
import statistics
import time

import rag  # noqa: F401
from rag.core import BM25
from rag.parquet import load

PQ = os.environ.get("POSTINGS_PARQUET") or str(
    pathlib.Path.home() / "Documents/JobApp/data/market/postings_all.parquet"
)

JUNIOR = r"\b(junior|jr|entry|graduate|trainee|intern)\b"

# ordered common -> rare -> absurd; the pool column is filled in at runtime
SPECS = [
    ("junior data analyst jobs in Spain",            r"spain|espa|madrid|barcelona|valencia|sevilla", r"analy|data", JUNIOR),
    ("nurse jobs in Germany",                        r"german",  r"\b(nurse|rn|pflege)", None),
    ("welder jobs in Poland",                        r"poland|polska|warsaw|krak", r"weld", None),
    ("remote customer support jobs",                 r"remote|anywhere", r"customer (support|service)|support (agent|specialist)", None),
    ("nurses in Congo",                              r"congo",   r"\b(nurse|rn|infirmier)", None),
    ("data analyst jobs in Bolivia",                 r"bolivia", r"analy|data", None),
    ("software engineer jobs in Nepal",              r"nepal",   r"engineer|developer", None),
    ("machine learning jobs in Paraguay",            r"paraguay", r"machine learning|\bml\b|ai ", None),
    ("remote banana peeler with ten years in the DRC", r"congo", r"banana", None),
]


def p_at(labels, hits, k):
    return sum(1 for h in hits[:k] if h in labels) / k


def recall_at(labels, hits, k):
    if not labels:
        return None
    return sum(1 for h in hits[:k] if h in labels) / len(labels)


def mrr(labels, hits):
    for n, h in enumerate(hits, 1):
        if h in labels:
            return round(1 / n, 3)
    return 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default=PQ)
    ap.add_argument("--out", default="/tmp/bench_filters.json")
    args = ap.parse_args()
    t = time.time()
    docs = load(pathlib.Path(args.parquet), max_chars=600)
    print(f"docs {len(docs)} in {time.time()-t:.0f}s", flush=True)

    t = time.time()
    full = BM25(docs)
    print(f"full index {time.time()-t:.0f}s", flush=True)

    out = []
    for text, loc_re, role_re, level_re in SPECS:
        loc = re.compile(loc_re, re.I)
        role = re.compile(role_re, re.I)
        lvl = re.compile(level_re, re.I) if level_re else None
        pool = [d for d in docs if loc.search(d["location"] or "")]
        labels = {d["id"] for d in pool
                  if role.search(d["title"] or "") and (not lvl or lvl.search(d["title"] or ""))}
        row = {"query": text, "pool": len(pool), "labels": len(labels)}

        if labels:
            hits_text = [h["id"] for h in full.search(text, 10)]
            row["text_p10"] = round(p_at(labels, hits_text, 10), 3)
            row["text_r10"] = round(recall_at(labels, hits_text, 10), 3)
            row["text_mrr"] = mrr(labels, hits_text)

            if pool:
                sub = BM25(pool)
                hits_f = [h["id"] for h in sub.search(text, 10)]
                row["filtered_p10"] = round(p_at(labels, hits_f, 10), 3)
                row["filtered_r10"] = round(recall_at(labels, hits_f, 10), 3)
                row["filtered_mrr"] = mrr(labels, hits_f)
                row["filtered_empty"] = len(hits_f) == 0
            else:
                row["filtered_p10"] = None
                row["filtered_empty"] = True
        else:
            # nothing satisfies the constraints: correct behaviour is an empty answer
            hits_text = [h["id"] for h in full.search(text, 10)]
            row["text_p10"] = 0.0
            row["text_returned"] = len(hits_text)
            row["filtered_p10"] = None
            row["filtered_empty"] = True

        print(json.dumps(row), flush=True)
        out.append(row)

    graded = [r for r in out if r.get("labels", 0) >= 3]
    if graded:
        print("\nmean P@10 text    ", round(statistics.mean(r["text_p10"] for r in graded), 3))
        print("mean P@10 filtered", round(statistics.mean(r["filtered_p10"] or 0 for r in graded), 3))
        print("mean R@10 text    ", round(statistics.mean(r["text_r10"] or 0 for r in graded), 3))
        print("mean R@10 filtered", round(statistics.mean(r["filtered_r10"] or 0 for r in graded), 3))
        print("mean MRR text     ", round(statistics.mean(r["text_mrr"] or 0 for r in graded), 3))
        print("mean MRR filtered ", round(statistics.mean(r["filtered_mrr"] or 0 for r in graded), 3))
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"wrote {args.out}")


main()
