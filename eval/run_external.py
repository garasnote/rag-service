#!/usr/bin/env python3
"""Independent-corpus evaluation: the same pipeline on postings it has never seen.

Corpus: edwarddgao/open-apply-jobs (MIT) — real ATS postings, one Lever day.
Download one parquet, then:

    python3 eval/run_external.py --parquet /tmp/lever.parquet

Nothing from the job_tracker corpus is used here, which is the point: this file
exists so the retrieval numbers can be reproduced on someone else's data.
"""
import argparse
import html
import json
import random
import re
import statistics
import sys
import time

import numpy as np
import pyarrow.parquet as pq
from snowballstemmer import stemmer as make_stemmer

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from rag.core import BM25, tokenize          # noqa: E402
from rag.embed import HashingEmbedder        # noqa: E402
from rag.fusion import rrf                   # noqa: E402

random.seed(11)
DEFAULT = "/tmp/lever.parquet"
TAG = re.compile(r"<[^>]+>")
WS = re.compile(r"\s+")
EN = make_stemmer("english")
CACHE = {}


def stem(w):
    s = CACHE.get(w)
    if s is None:
        s = CACHE[w] = EN.stemWord(w)
    return s


def plain(h):
    return WS.sub(" ", html.unescape(TAG.sub(" ", h or ""))).strip()


def stem_text(t):
    return " ".join(stem(w) for w in tokenize(t))


ap = argparse.ArgumentParser()
ap.add_argument("--parquet", default=DEFAULT, help="one open-apply-jobs parquet slice")
ap.add_argument("--out", default="/tmp/bench_external.json")
args = ap.parse_args()
OUT = args.out

t0 = time.time()
tbl = pq.read_table(args.parquet, columns=["id", "title", "source_slug", "description_html",
                                                   "department", "locations", "remote", "salary_min",
                                                   "employment_type"])
c = {k: tbl.column(k).to_pylist() for k in tbl.column_names}
docs, meta = [], []
for i in range(len(c["title"])):
    body = plain(c["description_html"][i])
    if len(body) < 200:
        continue
    docs.append({"id": c["id"][i], "text": (c["title"][i] + ". " + body)[:3000]})
    meta.append({
        "title": (c["title"][i] or "").lower(),
        "dept": (c["department"][i] or "").lower(),
        "loc": " ".join(c["locations"][i] or []).lower(),
        "remote": bool(c["remote"][i]),
        "salary_min": c["salary_min"][i],
        "etype": (c["employment_type"][i] or "").lower(),
    })
print(f"documents {len(docs)} · median {int(statistics.median(len(d['text']) for d in docs))} chars", flush=True)

stemmed_docs = [{"id": d["id"], "text": stem_text(d["text"])} for d in docs]
print(f"stemmed in {time.time()-t0:.0f}s · vocab {len(CACHE)}", flush=True)

sample = random.sample(range(len(docs)), 120)
entity = [{"q": meta[i]["title"], "gold": i} for i in sample]

TEXT_QUERIES = [
    ("data engineering jobs", lambda m: "data" in m["title"] and "engineer" in m["title"]),
    ("product designer roles", lambda m: "designer" in m["title"]),
    ("senior backend engineer jobs", lambda m: "backend" in m["title"] and "engineer" in m["title"]),
    ("customer support specialist", lambda m: "support" in m["title"]),
    ("marketing manager positions", lambda m: "marketing" in m["title"]),
    ("machine learning engineer", lambda m: "machine learning" in m["title"] or "ml engineer" in m["title"]),
]
STRUCT_QUERIES = [
    ("fully remote engineering jobs", lambda m: m["remote"] and "engineer" in m["dept"]),
    ("senior backend engineer jobs paying more than 150000",
     lambda m: "backend" in m["title"] and (m["salary_min"] or 0) >= 150000),
    ("sales roles based in london", lambda m: "sales" in m["dept"] and "london" in m["loc"]),
    ("part time roles", lambda m: "part" in m["etype"]),
]
text_q = [{"q": q, "gold": [i for i, m in enumerate(meta) if fn(m)]} for q, fn in TEXT_QUERIES]
struct_q = [{"q": q, "gold": [i for i, m in enumerate(meta) if fn(m)]} for q, fn in STRUCT_QUERIES]
print(f"entity {len(entity)} · text-intent {len(text_q)} · structured {len(struct_q)}", flush=True)

t1 = time.time()
bm25 = BM25(docs)
bm25s = BM25(stemmed_docs)
print(f"indexes built in {time.time()-t1:.1f}s", flush=True)


def entity_eval(search, label, queries, pool):
    hits = {1: 0, 3: 0, 5: 0, 10: 0}
    rr = []
    for item in queries:
        ids = [d["id"] for d in search(item["q"], 10)]
        gold = pool[item["gold"]]["id"]
        rank = ids.index(gold) + 1 if gold in ids else None
        rr.append(1.0 / rank if rank else 0.0)
        for k in hits:
            if rank and rank <= k:
                hits[k] += 1
    n = len(queries)
    return {"config": label, "n": n, **{f"hit@{k}": round(hits[k] / n, 3) for k in hits},
            "mrr@10": round(sum(rr) / n, 3)}


def intent_eval(search, label, queries):
    rows = []
    for item in queries:
        ids = {d["id"] for d in search(item["q"], 10)}
        gold = {docs[i]["id"] for i in item["gold"]}
        found = len(ids & gold)
        rows.append({"q": item["q"], "relevant": len(gold), "found@10": found,
                     "P@10": round(found / 10, 2), "R@10": round(found / min(10, len(gold)), 2)})
    return {"config": label, "mean_P@10": round(sum(r["P@10"] for r in rows) / len(rows), 3),
            "mean_R@10": round(sum(r["R@10"] for r in rows) / len(rows), 3), "rows": rows}


ent = [entity_eval(lambda q, k: bm25.search(q, k), "bm25 · no stem · 46k", entity, docs),
       entity_eval(lambda q, k: bm25s.search(stem_text(q), k), "bm25 · en stem · 46k", entity, stemmed_docs)]

# dense + hybrid on a 12k slice
t2 = time.time()
idx = random.sample(range(len(docs)), 12000)
sub, sub_s = [docs[i] for i in idx], [stemmed_docs[i] for i in idx]
pos = {o: j for j, o in enumerate(idx)}
emb = HashingEmbedder()
mat = np.zeros((len(sub), len(emb.vector("x"))), dtype=np.float32)
for j, d in enumerate(sub_s):
    mat[j] = emb.vector(d["text"])
mat /= np.maximum(np.linalg.norm(mat, axis=1, keepdims=True), 1e-9)
bm25_sub, bm25s_sub = BM25(sub), BM25(sub_s)
print(f"dense {mat.shape} in {time.time()-t2:.0f}s", flush=True)


def dense(q, k):
    v = np.asarray(emb.vector(stem_text(q)), dtype=np.float32)
    v /= max(np.linalg.norm(v), 1e-9)
    sc = mat @ v
    top = np.argsort(-sc)[:k]
    return [{"score": float(sc[j]), **sub[j]} for j in top if sc[j] > 0]


def hybrid(q, k):
    return rrf([bm25s_sub.search(stem_text(q), 20), dense(q, 20)], top=k)


ent_sub = [{"q": i["q"], "gold": pos[i["gold"]]} for i in entity if i["gold"] in pos]
saved = docs
docs = sub
ent += [entity_eval(lambda q, k: bm25s_sub.search(stem_text(q), k), "bm25 · en stem · 12k", ent_sub, sub),
        entity_eval(dense, "dense · 12k", ent_sub, sub),
        entity_eval(hybrid, "hybrid · 12k", ent_sub, sub)]
docs = saved

intent = [intent_eval(lambda q, k: bm25.search(q, k), "bm25 · no stem · 46k", text_q),
          intent_eval(lambda q, k: bm25s.search(stem_text(q), k), "bm25 · en stem · 46k", text_q),
          intent_eval(lambda q, k: bm25.search(q, k), "bm25 · no stem · 46k", struct_q),
          intent_eval(lambda q, k: bm25s.search(stem_text(q), k), "bm25 · en stem · 46k", struct_q)]

result = {"corpus": {"source": "edwarddgao/open-apply-jobs", "license": "MIT",
                     "slice": "data/date=2026-04-17/source=lever", "documents": len(docs)},
          "entity": ent, "text_intent": intent[:2], "structured_intent": intent[2:]}
json.dump(result, open(OUT, "w"), indent=1)

print("\n== entity (posting described by its own title)")
for r in ent:
    print(f"  {r['config']:<22} hit@1 {r['hit@1']:.2f}  hit@3 {r['hit@3']:.2f}  hit@5 {r['hit@5']:.2f}  mrr {r['mrr@10']:.3f}  n={r['n']}")
print("\n== intent resolvable from text")
for r in intent[:2]:
    print(f"  {r['config']:<22} P@10 {r['mean_P@10']:.3f}  R@10 {r['mean_R@10']:.3f}")
    for row in r["rows"]:
        print(f"      {row['q'][:40]:<42} rel {row['relevant']:>4}  found {row['found@10']:>2}")
print("\n== structured constraints (remote / salary / city / contract)")
for r in intent[2:]:
    print(f"  {r['config']:<22} P@10 {r['mean_P@10']:.3f}  R@10 {r['mean_R@10']:.3f}")
print(f"\nwrote {OUT} · {time.time()-t0:.0f}s")
