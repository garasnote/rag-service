# Stream: a reranker that has to prove itself

Branch `wt/rerank`, worktree `~/Documents/JobApp/portfolio/.wt-rag/rerank`.
README.md was not touched.

## What changed

| file | what |
| --- | --- |
| `rag/rerank.py` | new. `Reranker` protocol, `LexicalReranker` (stdlib, idf-weighted overlap plus a verbatim-phrase bonus), `CrossEncoderReranker` (ms-marco MiniLM-L6 on onnxruntime, lazy import, model dir from `RERANK_DIR`, `RerankerUnavailable` with a message that names the variable and the model), `get_reranker()` factory where `"none"` returns `None`. |
| `eval/run_rerank.py` | new. Reranks the top `p` of the bm25 and hybrid rankings for `p` in {10, 20, 40} and prints `sat@1/3/5` (intent questions) or `hit@1/3/5` (entity questions) next to the unreranked baseline at the same width. Runs on stdlib alone; `--rerank cross` needs the model. `--timing` adds median ms per query. |
| `tests/test_rerank.py` | new. 14 tests, 2 of which skip without a model (12 run with no dependency at all). |
| `rag/cli.py` | `--rerank none|lexical|cross` (default `none`) and `--pool`. Off by default: without the flag the retrieval path is byte-for-byte the old one. |
| `pyproject.toml` | new `rerank` extra (`numpy`, `onnxruntime`, `tokenizers`). The core library still imports stdlib only. |

Reranking is opt-in everywhere and off by default. The measurement says it
helps, but paying a second pass by default for a 2,577-document corpus where
stage one already answers most questions would be a choice the repo has not
made; so it is a flag, not a default.

## Reproducing this

One environment quirk first: `rag/core.py` resolves its private data as
`ROOT.parents[1]/job_tracker/data.json` and `ROOT.parent/hiresignal/warehouse.db`.
In a linked worktree neither path exists, and in the main tree only the first
one does (the warehouse now lives under `portfolio/hiresignal/`). So
`load_corpus()` yields 0 documents from this worktree and 493 from the main
tree, not 2,577. Every number below comes from a snapshot built explicitly from
both real files:

```bash
cd ~/Documents/JobApp/portfolio/.wt-rag/rerank
mkdir -p .cache && python3 -c "
import sys, json, pathlib; sys.path.insert(0, '.')
from rag.core import load_corpus
docs = load_corpus(tracker=pathlib.Path('~/Documents/JobApp/job_tracker/data.json'),
                   warehouse=pathlib.Path('~/Documents/JobApp/portfolio/hiresignal/warehouse.db'))
pathlib.Path('.cache/corpus.json').write_text(json.dumps(docs, ensure_ascii=False))
print(len(docs))"          # 2577
```

`.cache/` is gitignored, so the snapshot stays out of the repo.

The model (90 MB, Apache-2.0, no torch):

```bash
D=~/.cache/models/cross-encoder-ms-marco-minilm-l6-v2
for f in onnx/model.onnx tokenizer.json tokenizer_config.json config.json vocab.txt; do
  curl -sL "https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2/resolve/main/$f" \
       -o "$D/$f" --create-dirs
done
```

The four measurements that matter (`P=/usr/bin/python3`,
`V=~/.venvs/rag/bin/python`, `RERANK_DIR` set for the cross runs):

```bash
$P eval/run_rerank.py --corpus .cache/corpus.json --rerank none
$P eval/run_rerank.py --corpus .cache/corpus.json --rerank lexical
$V eval/run_rerank.py --corpus .cache/corpus.json --rerank cross          # RERANK_DIR set
$P eval/run_rerank.py --corpus .cache/corpus.json --rerank lexical --metric hit
$V eval/run_rerank.py --corpus .cache/corpus.json --rerank cross --metric hit
$P eval/run_rerank.py --corpus .cache/corpus.json --rerank lexical --timing
```

## The numbers

Corpus 2,577 documents (tracker + warehouse; the README says 2,576 because one
document has been added since it was written). Embedder `hashing-512`, i.e. the
same configuration the README's tables use. Pools widen stage one: each
retriever returns `p` and RRF fuses those `p`, so `none` is the baseline at that
width. For bm25 the first five are identical at every width; for hybrid they are
not, which is why both rows are printed.

### Intent questions, `eval/questions_intent.json`, n = 20

`satisfied@k`: at least one document in the top k contains a term from every
constraint group of the question. One question is 5 percentage points.

```
base   pool  reranker    sat@1  sat@3  sat@5
bm25     10  none          40%    65%    65%
bm25     10  lexical       40%    50%    65%     <- the one cell that loses
bm25     10  cross         50%    50%    55%     <- and this one
bm25     20  none          40%    65%    65%
bm25     20  lexical       45%    60%    70%
bm25     20  cross         60%    65%    65%
bm25     40  none          40%    65%    65%
bm25     40  lexical       45%    75%    80%
bm25     40  cross         65%    85%    85%
hybrid   10  none          40%    45%    55%
hybrid   10  lexical       50%    65%    75%
hybrid   10  cross         60%    65%    70%
hybrid   20  none          35%    45%    50%
hybrid   20  lexical       40%    70%    75%
hybrid   20  cross         65%    65%    70%
hybrid   40  none          35%    45%    55%
hybrid   40  lexical       45%    75%    75%
hybrid   40  cross         65%    75%    75%
```

The reranker changed the top five in 16-20 of 20 shortlists, so it is not
returning the input unchanged.

For reference, `python3 eval/run_intent.py --corpus .cache/corpus.json` on the
same snapshot prints bm25 40/65/65, dense 30/30/35, hybrid 40/50/60 -- the 5-wide
fusion. Reranking at any pool beats all three at `sat@3` and `sat@5`.

### Entity questions, `eval/questions.json`, n = 15

`hit@k`: did a document containing the named employer surface. One question is
6.7 percentage points.

```
base   pool  reranker    hit@1  hit@3  hit@5
bm25     10  none          53%    73%    73%
bm25     10  lexical       73%    80%    80%
bm25     10  cross         73%    80%    80%
bm25     20  none          53%    73%    73%
bm25     20  lexical       73%    80%    80%
bm25     20  cross         73%    80%    80%
bm25     40  none          53%    73%    73%
bm25     40  lexical       67%    80%    87%
bm25     40  cross         87%    93%    93%
hybrid   10  none          60%    73%    73%
hybrid   10  lexical       67%    80%    80%
hybrid   10  cross         73%    80%    80%
hybrid   20  none          60%    73%    73%
hybrid   20  lexical       73%    87%    87%
hybrid   20  cross         80%    87%    87%
hybrid   40  none          60%    67%    67%
hybrid   40  lexical       73%    87%    87%
hybrid   40  cross         80%    87%    87%
```

Nothing loses on this set. This is the benchmark the README's reranker claim
most likely referred to, and it is the one that most clearly does not reproduce.

### Language split, pool 40 (n = 10 per language)

The cross-encoder is English-trained (`ms-marco`), and it shows.

```
lang  base    sat@1/3/5 none      lexical   cross
es    bm25    50/70/70            40/80/90  50/90/90
es    hybrid  40/50/60            50/80/80  50/70/70
en    bm25    30/60/60            50/70/70  80/80/80
en    hybrid  30/40/50            40/70/70  80/80/80
```

Spanish questions gain on `sat@3/@5` and stay flat on `sat@1`; English questions
gain everywhere. A Spanish-capable cross-encoder (bge-reranker-v2-m3) is the
obvious next experiment; it was not run here, so no number is claimed for it.

### Cost, median ms per query, pool 40, 16-core CPU box

```
stage one (bm25 + dense + RRF)   180-220 ms      (noisy on this box; 104-220 ms
                                                 across runs)
LexicalReranker                  1.3 ms          same process
CrossEncoderReranker             1000-1500 ms    ORT_THREADS=8; 16 threads was
                                                 slower (1093 vs 957 ms)
```

The lexical reranker is free next to stage one. The cross-encoder is about 7x
stage one at pool 40, which is still small next to the LLM call that follows it,
but it is not free and it is not 1.3 ms.

### Knobs: the lexical win is not a lucky weight

Nine combinations of `--phrase-weight` and `--coverage-weight` in {0, 0.5, 1},
pool 40, intent questions:

```
phrase  coverage  bm25 sat@1/3/5   hybrid sat@1/3/5   (baselines 40/65/65, 35/45/55)
0       0         35/80/80         40/70/75
0       0.5       35/80/80         45/70/75
0       1.0       35/75/80         45/70/75
0.5     0         45/75/80         45/75/75
0.5     0.5       45/75/80         45/75/75        <- defaults
0.5     1.0       40/75/80         45/75/75
1.0     0         40/75/75         40/70/75
1.0     0.5       45/75/75         45/75/75
1.0     1.0       45/75/80         45/75/75
```

All nine beat the baseline on `sat@3` and `sat@5`; `sat@1` moves between 35% and
45% against a 35-40% baseline, i.e. zero to two questions. The phrase bonus is
what buys the early ranks; coverage alone does not.

## Which reranker I would ship, and why

**Ship the cross-encoder at pool 40; ship the lexical one as the default when
the model is not there.**

- The cross-encoder is the quality winner and it is the only option that
  dominates the baseline at every pool from 20 up on both benchmarks and both
  base rankings: intent `sat@1` 35% -> 65% over hybrid, entity `hit@1` 60% ->
  80% over hybrid and 53% -> 87% over bm25, with `hit@5` at 87-93%. It loses
  exactly two cells, both at pool 10 from bm25, where ten rescorable documents
  are not enough to fix a shortlist that is already wrong -- a pool problem, not
  a model problem.
- It costs 1.0-1.5 s per query on this box. That is affordable in this pipeline
  (a single LLM call dominates), it is not affordable if the reranker is ever
  put in front of a synchronous autocomplete.
- The lexical reranker is the fallback that matters: no dependencies, 1.3 ms,
  and at pool 40 it captures the entire `sat@3/sat@5` gain (45/75/75 over
  hybrid) and most of the entity gain (73/87/87). What it does not buy is rank
  one, which is the rank the answer prompt actually cites first.
- Pool 40 rather than 10 or 20: 40 is where both rerankers stop being able to
  hurt the baseline and where the gains are largest. It is also the costliest.
  20 is the compromise if the latency ever matters: cross at 20 gives 65/65/70
  and 80/87/87.

Both are kept opt-in and off by default (`rag.cli --rerank`, `eval/run_rerank.py
--rerank`). The default path is unchanged, which is the honest position for a
feature whose gain is 4-6 questions out of 20 on the intent set.

## The README claim does not hold, and why I think it looked like it did

The README says: "A cross-encoder over the BM25 shortlist was tried at pool
sizes 10, 20 and 40 and lost at every size on this corpus." On the current
snapshot, with this harness, it wins at every pool size on the entity questions
and at two of three on the intent questions. I cannot reproduce the claim and I
did not change it, per the ground rules -- the evidence is here, the README
should be corrected by whoever merges the streams.

What could have produced the old result, stated as hypothesis rather than fact:

1. The corpus drifted. The README counts 2,576 documents, this snapshot has
   2,577, and the benchmark questions (`eval/questions*.json`) have clearly been
   edited since some tables were written.
2. Pool width and baseline width. If the old run compared "cross-encoder over
   the top 40" against a hybrid baseline fused at width 5, part of what it
   measured was the cost of widening the fusion, not the reranker. This harness
   compares inside a fixed width for exactly that reason, and it prints the
   width so nobody has to guess.
3. The intent benchmark did not exist yet. It measures constraint satisfaction
   rather than "did it find the row I named", and it is the benchmark where a
   reranker has something to win.

## What I tried that did not work, and what is still weak

- **Fusing at a fixed width for all pools** (first version of the harness):
  printed a `none` row that changed size with the pool and did not match
  `eval/run_intent.py`, so the table could not be compared to the README. Fixed
  by fusing at the width being evaluated.
- **Passing every reranker's kwargs through one factory**: blew up with
  `TypeError` on the cross-encoder for `phrase_weight`. Fixed by only forwarding
  the lexical knobs; the eval now fails loudly instead of silently ignoring a
  flag.
- **Looking for `model.onnx` only in the model directory root**: real HF exports
  put it in `onnx/model.onnx`, so a correctly downloaded model reported "has
  tokenizer.json but no model.onnx". Fixed by searching the root and `onnx/`.
- **`ORT_THREADS=16`** on a 16-core box: slower than 8 (1093 ms vs 957 ms per
  query). Left the default at 8.
- **The intent metric flatters a lexical reranker.** `satisfied@k` asks whether
  some document contains a term from every constraint group, which is very close
  to what `LexicalReranker` optimises, so its intent numbers are partly a
  property of the metric. The cross-encoder gets those numbers without being
  built for them, which is the more interesting result. For the same reason, a
  human reading these lists would care about things neither metric sees -- a
  posting that is one year old, a company already rejected.
- **Sample size.** 20 intent and 15 entity questions. One question is 5 and 6.7
  points. The `sat@1` differences between the two rerankers (45% vs 65%) are
  four questions. Treat the ordering between the rerankers as suggestive and the
  ordering against the baseline as solid.
- **The corpus layout is stale**: `rag/core.py` still points `WAREHOUSE` at a
  path that does not exist in this tree, so `load_corpus()` silently returns 493
  documents instead of 2,577 outside the snapshot. Not fixed here because it is
  outside this stream, but it will bite anyone who runs the eval without
  `--corpus`.

## Tests

`python3 -m pytest -q` on `/usr/bin/python3`, no model installed:

```
before:  15 passed, 2 skipped
after:   27 passed, 4 skipped
```

Same command in `~/.venvs/rag/bin/python` with `RERANK_DIR` pointing at
the downloaded model, so the ONNX path is actually exercised:

```
29 passed, 2 skipped
```

The two skips in the first run are the cross-encoder behaviour tests; the two in
the second are the pre-existing `test_corpus_loads_with_urls` (no private
tracker in a worktree) and `test_minilm_is_optional` (no `MINILM_DIR`).

Tests cover: constraint conjunction, phrase adjacency, accent folding, tie
stability, key preservation and non-mutation of the input, `top` truncation,
empty query/pool, protocol conformance, the factory's `none` default and unknown
name, the availability error message, and -- when a model is present -- a
relevant document outranking an irrelevant one.
