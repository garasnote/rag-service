# Stream: optional ANN index (rag/ann.py) — built, measured, and it does not earn its dependency

Branch `wt/ann`, worktree `~/Documents/JobApp/portfolio/.wt-rag/ann`.
Deliverables are all in place. The result is negative and I am reporting it as such: the HNSW
backend works and is fast, but an exhaustive numpy scan is just as fast, exact, and needs no new
dependency. Default behaviour is unchanged (`exact`); `rag/ann.py` is opt-in only.

## What changed

- `rag/ann.py` (new): `HNSWIndex(docs, embedder, space="cosine", ef=50, M=16,
  ef_construction=200, build_threads=1)`. Subclasses `VectorIndex`, so it exposes the same surface
  (`search_text`, `search_vector`, `vectors`) and reuses the same on-disk embedding cache.
  `import hnswlib` happens inside `__init__`; without it you get
  `RuntimeError: HNSWIndex needs the optional dependency hnswlib: ...` and the core library still
  imports nothing but the stdlib. `space` accepts `cosine` and `ip`, both reporting
  `1 - similarity`, so scores mean what `VectorIndex` scores mean, including its "drop score <= 0"
  filter — that filter is kept on purpose so top-k sets stay comparable. `ef` is a property mapped
  onto `set_ef`, so recall/latency can be swept without rebuilding. `HNSWIndex.build_s` records
  graph construction time alone, excluding embeddings.
- `eval/bench_ann.py` (new): the measurement. Three backends on one corpus and one query set —
  `exact` (`VectorIndex`, stdlib loop), `numpy` (the same exhaustive scan via `matrix @ q`, no
  approximation), and `ann` at ef in {10, 25, 50, 100}. recall@10 is computed against the exact
  index's top-10 (intersection / reference size, averaged over queries), never assumed. Real corpus
  when visible, `fixtures/corpus.json` otherwise. Latency uses the repeat loop as the outer loop and
  every backend inside it, because a contiguous block per backend on a loaded box measures load.
- `eval/run_intent.py`: `--index exact|ann` (default stays `exact`) and `--ef`. Also `--tracker`,
  because `load_corpus()`'s default `TRACKER` is `ROOT.parents[1]/job_tracker/data.json`, which from
  a worktree resolves to `.wt-rag/job_tracker/data.json` — without `--tracker` the harness silently
  ran on a 0-document corpus from here.
- `tests/test_ann.py` (new): 7 tests. 2 always run — missing dependency must raise an actionable
  `RuntimeError` (faked through `sys.modules`), and a bad `space` must raise `ValueError`. 5 skip
  unless `hnswlib` imports, and one of them asserts recall is exactly 1.0 against the exact index on
  a tiny corpus at `ef=200`.
- `pyproject.toml`: `ann` extra (`hnswlib>=0.8`, `numpy>=1.24`).

Nothing else touched. `README.md` untouched. `rag/embed.py` untouched — see "why it loses".

## Numbers

Corpus: the live tracker, 485 documents, `md5 28f0cc98968413df0ad4689eec600d09` before and after
every run below. Queries: 35 (`eval/questions_intent.json` + `eval/questions.json`, deduped).
The box was running six other agents' work throughout (load average 12-17), which is why median and
p95 are printed next to the mean.

### Hash embedder (512-d)

```
$ ~/.venvs/rag/bin/python eval/bench_ann.py --tracker ~/Documents/JobApp/job_tracker/data.json --repeats 20
corpus: 485 documents from ~/Documents/JobApp/job_tracker/data.json
embedder: hashing-512 | queries: 35 from eval/questions_intent.json,eval/questions.json
ann graph: 512 dimensions, M=16, ef_construction=200, graph 42.5 ms of the 86.5 ms total
timing: 20 interleaved passes over the query set | load average: 14.83

backend    ef  recall@10   mean ms   med ms   p95 ms  build ms
--------------------------------------------------------------
exact       -       1.00    16.666   16.531   18.075    35.208
numpy       -       0.99     0.170    0.159    0.221    35.208
ann        10       0.80     0.136    0.133    0.166    86.532
ann        25       0.92     0.140    0.138    0.160    86.532
ann        50       0.97     0.153    0.152    0.174    86.532
ann       100       0.99     0.171    0.170    0.194    86.532
ann(ef=100) mean latency is 0.01x the exact mean
```

`numpy` recalls 0.99, not 1.00: it runs in float32 while `VectorIndex` compares float64, and one of
the 35 queries has a tie at the k=10 boundary that the two orders break differently. Measured, so
reported as measured.

### Multilingual MiniLM embedder (384-d)

```
$ ~/.venvs/rag/bin/python eval/bench_ann.py --tracker .../data.json --embedder multi --repeats 10
corpus: 485 documents from ~/Documents/JobApp/job_tracker/data.json
embedder: multilingual-minilm-l12-384 | queries: 35 from eval/questions_intent.json,eval/questions.json
ann graph: 384 dimensions, M=16, ef_construction=200, graph 34.7 ms of the 100.3 ms total
timing: 10 interleaved passes over the query set | load average: 12.21

backend    ef  recall@10   mean ms   med ms   p95 ms  build ms
--------------------------------------------------------------
exact       -       1.00    23.800   23.101   30.786    59.520
numpy       -       1.00     4.695    4.447    7.090    59.520
ann        10       0.92     4.602    4.519    6.198   100.287
ann        25       0.99     4.713    4.532    6.732   100.287
ann        50       1.00     4.948    4.695    7.184   100.287
ann       100       1.00     4.874    4.709    6.810   100.287
```

Here everything but `exact` is dominated by the MiniLM query embedding (~4.4 ms), so the three fast
rows are the same number twice over.

### Fixtures fallback (16 docs, for the fresh-clone path only)

```
$ ~/.venvs/rag/bin/python eval/bench_ann.py --repeats 20
corpus: 16 documents from fixtures fallback (.../fixtures/corpus.json)
exact       -       1.00     0.625    0.623    0.688    12.356
numpy       -       1.00     0.127    0.125    0.153    12.356
ann        10       1.00     0.118    0.116    0.137     4.432
ann        50       1.00     0.115    0.114    0.133     4.432
```

16 documents is too few for recall to distinguish anything (every ef gets 1.00). This table only
proves the fallback path runs; the real corpus is where the tradeoff shows.

### Intent satisfaction, exact vs ANN at ef=50 — identical everywhere

```
$ ... eval/run_intent.py --tracker .../data.json --index exact        (hash embedder)
corpus: 485 documents | questions: 20 | embedder: hashing-512 | dense: exact
mode       sat@1   sat@3   sat@5
bm25         50%     85%     85%
dense        35%     70%     80%
hybrid       50%     80%     90%

$ ... eval/run_intent.py --tracker .../data.json --index ann --ef 50  (hash embedder)
corpus: 485 documents | questions: 20 | embedder: hashing-512 | dense: ann(ef=50)
bm25         50%     85%     85%
dense        35%     70%     80%
hybrid       50%     80%     90%

$ ... --embedder multi --index exact          $ ... --embedder multi --index ann --ef 50
dense        50%     85%     90%              dense        50%     85%     90%
hybrid       65%     90%     95%              hybrid       65%     90%     95%
```

Every cell matches, on both embedders. At `ef=50` the approximation is invisible to the satisfaction
metric: recall@10 = 0.97 (hash) / 1.00 (multi) is high enough that the satisfying document is already
in the top 5 either way.

**The ef I would ship: 50**, if this were shipped at all. `ef=25` costs 0.05 recall@10 on the hash
embedder for 0.014 ms; `ef=100` buys 0.02 recall for 0.018 ms; `ef=10` is clearly too lossy (0.80).

Latency difference at ef=50, median per query on the hash embedder: **16.5 ms exact -> 0.15 ms ANN**.
On the MiniLM embedder: **23.1 ms -> 4.7 ms**, of which 4.4 ms is the embedding, not the search.

## Why it loses anyway

That 100x is not approximation paying off. It is a C++/numpy kernel versus a Python loop. The `numpy`
row does the identical exhaustive scan with `matrix @ q` + `argpartition`: on the hash embedder it
lands at **0.159 ms median against ANN's 0.152 ms at ef=50** — the same speed, at recall 0.99 instead
of 0.97, with no graph to build (35 ms vs 87 ms) and no new dependency (numpy is already in the
`embeddings` extra). On the MiniLM embedder the two are again tied (4.45 vs 4.70 ms), both swamped by
the embedding.

So the honest accounting on this corpus:

| | exact (stdlib) | numpy | ann @50 |
|---|---|---|---|
| recall@10, hash | 1.00 | 0.99 | 0.97 |
| median ms, hash | 16.5 | 0.159 | 0.152 |
| median ms, multi | 23.1 | 4.45 | 4.70 |
| build ms, hash | 35.2 | 35.2 | 86.5 |
| extra dependency | none | none (already in `embeddings`) | hnswlib (C++ build) |
| knob to tune | none | none | `ef` |

HNSW only starts winning when the scan stops being cache-resident, i.e. orders of magnitude more
documents than 485. It is also the only row that loses recall.

**Recommendation: keep `exact` as the default and keep `ann.py` as the documented opt-in for a much
larger corpus. Do not advertise the 100x without the numpy footnote.** If dense latency actually
matters, the cheap fix is `matrix @ qvec` inside `VectorIndex.search_vector`: one line, no new
dependency, no recall loss. I did not make that change — `rag/embed.py` is shared with other streams
and the brief scoped me to an optional backend — but it is the change the numbers point at.

## What I tried that failed

- **Recall on the fixtures corpus.** 16 documents: recall is 1.00 at every ef. Zero information.
- **`load_corpus()` from inside the worktree.** 0 documents, because `TRACKER` resolves to
  `.wt-rag/job_tracker/data.json`. Hence `--tracker` on both scripts.
- **One contiguous timing block per backend.** First runs produced nonsense (`ann@10` slower than
  `ann@50`; `ann` at 41 ms in one block and 4.5 ms in another) because six other agents were
  compiling on the same 16 cores, load average 18-30. Fixed by interleaving backends inside the
  repeat loop. Even so, absolute latencies move by 2x between runs on this box; only the ratios
  inside one table are trustworthy.
- **A separate scratch script for the numpy baseline.** Measuring it minutes apart from the bench
  gave numbers that could not be compared (0.174 ms vs 0.345 ms for the same work), and it produced
  3/35 disagreements with the exact top-10 because I had forgotten the `score > 0` filter. The numpy
  row now lives inside `bench_ann.py`, shares the interleaved pass, and applies the same filter.
- **Multithreaded graph construction.** hnswlib builds with all cores by default, so the graph — and
  therefore recall at a fixed `ef` — changed between runs (ef=10 recall came out 0.77, then 0.80,
  then 0.81 from identical code). `build_threads=1` is now the default (2 consecutive runs then give
  identical recall columns: 0.80/0.92/0.97/0.99). Reproducibility over build speed at this size.
- **Counting numpy's import in the ANN build time.** It added ~150 ms and made the build column
  meaningless; the bench warms the import before timing anything.

## Tests

Before this branch: `15 passed, 2 skipped` (`/usr/bin/python3 -m pytest -q`, clean checkout).
After:
```
$ /usr/bin/python3 -m pytest -q
17 passed, 7 skipped in 0.93s      # the 5 hnswlib tests skip, as required
$ ~/.venvs/rag/bin/python -m pytest -q
22 passed, 2 skipped in 1.19s      # venv has hnswlib 0.8.0 and pytest 9.2.0
```

## Reproducing

Installed into the venv only, never system python:
```
uv pip install --python ~/.venvs/rag/bin/python hnswlib pytest
```
From the worktree:
```
~/.venvs/rag/bin/python eval/bench_ann.py --tracker ~/Documents/JobApp/job_tracker/data.json
~/.venvs/rag/bin/python eval/bench_ann.py --tracker ~/Documents/JobApp/job_tracker/data.json --embedder multi
~/.venvs/rag/bin/python eval/bench_ann.py            # fixtures fallback, no tracker
~/.venvs/rag/bin/python eval/run_intent.py --tracker ~/Documents/JobApp/job_tracker/data.json --index exact
~/.venvs/rag/bin/python eval/run_intent.py --tracker ~/Documents/JobApp/job_tracker/data.json --index ann --ef 50
```
`bench_ann.py` needs numpy + hnswlib, so it runs under the venv python. Everything else, including
`run_intent.py --index exact`, still runs on stdlib python.

## Caveat on the corpus

The tracker is Garas's live file and it changed while I worked: 493 documents at first, 485 for every
number above. The exploratory 493-document runs gave recall@10 0.81/0.94/0.97/0.99 at ef
10/25/50/100 — the same shape. Each table states the document count that was actually loaded, so the
numbers are not silently mixed. The corpus will keep drifting; the ratios are what matter.
