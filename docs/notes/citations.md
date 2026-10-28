# Stream: chunk-level citations for long postings

Branch `wt/citations`, worktree `.wt-rag/citations`. Commit `f950e70` (code + tests) and the commit
that adds this file.

## What changed

**`rag/chunking.py` (new, stdlib only).** `chunk_document(doc_id, text, size=600, overlap=100,
threshold=600)` returns windows as `{doc_id, chunk_index, start, end, text}`.

- A document of at most `threshold` characters returns exactly one window with
  `start=0, end=len(text), text=text` — the chunk text *is* the document text, so short records
  behave exactly as before.
- Longer documents are cut at `size` characters with `overlap` between consecutive windows. Each
  window end is pulled back to the last space within 80 characters, so a chunk does not end mid-word;
  offsets are always relative to the original string, so `text[start:end] == chunk["text"]` holds for
  every window (that is the property the citation relies on).
- The tail is dropped as a separate window when it is already inside the previous window (a 601-char
  document gives one window, not a one-character second one).
- Empty text returns `[]`; `overlap >= size` raises `ValueError` rather than looping forever.

**Retrieval scores windows, returns documents.** `ChunkedBM25` and `ChunkedVectorIndex` wrap the
existing `BM25` and `VectorIndex` (no second scorer, no duplicated BM25 maths). They index the
windows, then keep the highest-scoring window per document: one hit per document, as before, with

```
hit["text"]              # still the whole posting
hit["chunk"] = {"index": 2, "start": 997, "end": 1555, "text": "..."}
```

**`api.py` / `rag/llm.py`.** `/search` and `/ask` sources gain `chunk_index`, `start`, `end` and
`snippet` (240 characters, flattened whitespace, cut on a word boundary). Nothing is removed and
`text` keeps its old meaning. A hit without a chunk (plain index) is cited as the whole document:
index 0, offsets 0..len(text).

The API builds the windowed index by default — that is the point of the stream — with `RAG_CHUNK=0`
as the escape hatch back to `BM25` over whole documents. The eval harnesses default to **off**;
`--chunk` turns chunking on for either one.

**`eval/run_eval.py`, `eval/run_intent.py`.** New `--chunk` flag; the header line now says which mode
ran (`whole document` or `chunked (N windows/doc)`).

## Numbers

Private corpus: 2577 documents (tracker + warehouse), 20 intent questions, 15 entity questions,
hashing embedder, defaults (600/100/600).

Windows: 2616 windows for 2577 documents = **1.015 windows per document**; only 33 documents exceed
600 characters (longest is 1555), so 33 documents get split. Chunk-level scoring is therefore barely
exercised by this corpus — see "what this does not show" below.

Intent benchmark, `sat@5` chunked vs unchunked:

```
$ python3 eval/run_intent.py --corpus $CORPUS | head -6
corpus: 2577 documents | questions: 20 | embedder: hashing-512 | whole document

mode       sat@1   sat@3   sat@5
bm25         40%     65%     65%
dense        30%     30%     35%
hybrid       40%     50%     60%

$ python3 eval/run_intent.py --corpus $CORPUS --chunk | head -6
corpus: 2577 documents | questions: 20 | embedder: hashing-512 | chunked (1.02 windows/doc)

mode       sat@1   sat@3   sat@5
bm25         40%     70%     70%
dense        30%     30%     40%
hybrid       40%     50%     60%
```

Difference at sat@5: **bm25 +5 points (65% -> 70%), dense +5 points (35% -> 40%), hybrid 0**.
Per-question flip check (nothing lost, one question gained per mode):

```
$ python3 - <<'EOF'
import json, sys
sys.path.insert(0, ".")
from rag.chunking import ChunkedBM25, ChunkedVectorIndex
from rag.core import BM25, load_corpus
from rag.embed import VectorIndex, get_embedder
from rag.fusion import rrf
from eval.run_intent import satisfied_at

docs = load_corpus(corpus="/tmp/rag-citations/corpus_private.json")
questions = json.loads(open("eval/questions_intent.json", encoding="utf-8").read())
plain = (BM25(docs), VectorIndex(docs, get_embedder("hash")))
cut = (ChunkedBM25(docs), ChunkedVectorIndex(docs, get_embedder("hash")))
def rank(idx, q):
    lex, den = idx[0].search(q, k=5), idx[1].search_text(q, k=5)
    return {"bm25": lex, "dense": den, "hybrid": rrf([lex, den])}
rows = [{"q": it["q"], **{m: (satisfied_at(rank(plain, it["q"])[m], it["must"], 5),
                              satisfied_at(rank(cut, it["q"])[m], it["must"], 5))
                          for m in ("bm25", "dense", "hybrid")}} for it in questions]
for m in ("bm25", "dense", "hybrid"):
    off = sum(r[m][0] for r in rows) / len(rows)
    on = sum(r[m][1] for r in rows) / len(rows)
    print(f"{m:7} sat@5 off {off:.0%} on {on:.0%} flips {sum(1 for r in rows if r[m][0] != r[m][1])}")
    for r in rows:
        if r[m][0] != r[m][1]:
            print(f"   {'+' if r[m][1] else '-'} {r['q']}")
print("windows/doc:", (len(cut[0].rows) / len(docs)))
EOF
bm25    sat@5 off 65% on 70% flips 1
   + junior data jobs fully remote
dense   sat@5 off 35% on 40% flips 1
   + trabajo junior de visión por computador
hybrid  sat@5 off 60% on 60% flips 0
windows/doc: 1.0151338766006985
```

One question is 5 points on a 20-question set. **This is not evidence that chunking retrieves
better.** It is a positive direction on a small sample, with no question lost in any mode.

Entity benchmark (secondary, same corpus, 15 questions) — same direction:

```
$ python3 eval/run_eval.py --corpus $CORPUS | head -10     (whole document)
bm25        53%    73%  0.622
dense       40%    53%  0.467
hybrid      60%    73%  0.656

$ python3 eval/run_eval.py --corpus $CORPUS --chunk | head -10
bm25        67%    73%  0.689
dense       40%    60%  0.489
hybrid      67%    73%  0.700
```

`hit@1` 53% -> 67% for bm25, MRR 0.622 -> 0.689. Same caveat: 15 questions.

Sanity check at `size=300` (more documents split, 83 of 2577, 1.049 windows/doc): bm25 65% -> 75%,
dense 35% -> 40%, hybrid 60% -> 65%, again no question lost. Reported for completeness, not adopted;
the brief asks for ~600.

Worked example of the citation itself, on the longest posting (1555 characters):

```
$ python3 - <<'EOF'
import sys; sys.path.insert(0, ".")
from rag.core import load_corpus
from rag.chunking import ChunkedBM25, citation_fields
docs = load_corpus(corpus="/tmp/rag-citations/corpus_private.json")
q = " ".join(w for w in max(docs, key=lambda d: len(d["text"]))["text"].split()[-12:] if len(w) > 3)
print("query:", q)
for h in ChunkedBM25(docs).search(q, k=1):
    f = citation_fields(h)
    print(h["id"], "| doclen", len(h["text"]), "| score", h["score"], "| chunk", f["chunk_index"], (f["start"], f["end"]))
    print("slice ok:", h["text"][f["start"]:f["end"]] == h["chunk"]["text"])
    print("snippet:", f["snippet"])
EOF
query: Power Power Automate, OLAP/multidimensional. Empleador identificado anuncio. Target
tracker:8b2d15db-8dc6-4b5c-acc1-e4fb473aadb7 | doclen 1555 | score 24.4643 | chunk 2 (997, 1555)
slice ok: True
snippet: Python o R. Condiciones: contrato indefinido, jornada completa, PRESENCIAL en Taco (Tenerife), dos turnos posibles (8-13/14-17 y viernes hasta 16:00, o 9-14/15-18 y viernes hasta 17:00), becas formativas y plan de carrera. Preguntas del...
```

Whole-document results returned chunk 0 `(0, len)` and cited the full text, so the short records are
untouched.

Tests: **17 collected (15 passed, 2 skipped) before -> 30 collected (27 passed, 3 skipped) after**,
`python3 -m pytest -q`. The one new skip is the `/search` source-shape test, which needs fastapi
(absent from system python); the other two are pre-existing (no `MINILM_DIR`, no private tracker).
With the `api` extra installed (`~/.venvs/rag/bin/python -m pytest -q`): 28 passed, 2
skipped — so that test does run and pass where fastapi exists, per the brief.

## What this does not show, and what I tried that did not work

- **The corpus is the wrong shape for this feature.** 2544 of 2577 documents are shorter than 600
  characters, so chunking is a no-op for 99% of the corpus and the benchmark can only see it through
  the 33 long records. Whatever it buys on real long postings (scraped descriptions rather than the
  composed two-line records) is unmeasured here. Do not quote the +5 points as a retrieval win.
- I did **not** find a configuration where chunking loses: at 600 and at 300, no question and no
  entity moved from satisfied to unsatisfied. The cost is index size (2616 BM25 rows, 2616 dense
  vectors instead of 2577) and one extra pass over the corpus at startup.
- I left the LLM prompt alone. `rag.llm.build_prompt` still feeds `hit["text"][:800]` to the model,
  which for a long posting is the *head*, not the matched passage. Feeding `hit["chunk"]["text"]`
  instead is the obvious follow-up, but it changes generated answers and this stream did not measure
  that, so it is out of scope and unmeasured.
- I did not chunk the `rag.cli` output or the README: `python3 -m rag.cli` still prints whole
  documents. The brief forbids editing `README.md`.

## How to reproduce

The private tracker and warehouse do not resolve from this worktree
(`rag.core.TRACKER` = `portfolio/job_tracker/data.json`, `rag.core.WAREHOUSE` =
`.wt-rag/hiresignal/warehouse.db`; neither exists), so the corpus is dumped once to a file outside
git:

```
mkdir -p /tmp/rag-citations
python3 - <<'EOF'
import json, pathlib, sys
sys.path.insert(0, "~/Documents/JobApp/portfolio/.wt-rag/citations")
from rag.core import load_corpus
docs = load_corpus(tracker=pathlib.Path("~/Documents/JobApp/job_tracker/data.json"),
                   warehouse=pathlib.Path("~/Documents/JobApp/portfolio/hiresignal/warehouse.db"))
pathlib.Path("/tmp/rag-citations/corpus_private.json").write_text(json.dumps(docs, ensure_ascii=False))
print(len(docs))
EOF
CORPUS=/tmp/rag-citations/corpus_private.json
```

Then, from `~/Documents/JobApp/portfolio/.wt-rag/citations`:

```
python3 -m pytest -q
python3 eval/run_intent.py --corpus $CORPUS
python3 eval/run_intent.py --corpus $CORPUS --chunk
python3 eval/run_eval.py   --corpus $CORPUS
python3 eval/run_eval.py   --corpus $CORPUS --chunk

# no private data, no API key, what CI runs:
python3 eval/run_eval.py   --corpus fixtures/corpus.json --questions fixtures/questions.json
python3 eval/run_intent.py --corpus fixtures/corpus.json --questions fixtures/questions_intent.json
python3 eval/run_intent.py --corpus fixtures/corpus.json --questions fixtures/questions_intent.json --chunk

# the /search shape test needs the api extra:
~/.venvs/rag/bin/python -m pytest -q
```

The fixture runs give identical scores with and without `--chunk` (all 16 fixture documents are
short), which is the check that the flag off-by-default changes nothing.

The per-question flip check above is inline; it only needs the modules the repo already has in
addition to the dumped corpus. Nothing in these notes needs an API key or a network call, except the
`--hyde` paths, which are unchanged by this stream.

## Commands the orchestrator needs

- `python3 -m pytest -q` — must stay green (27 passed, 3 skipped here).
- `python3 eval/run_intent.py --corpus <corpus.json> [--chunk]` — the headline: sat@5 65% -> 70%
  (bm25), 35% -> 40% (dense), 60% -> 60% (hybrid).
- 1.015 windows per document on this corpus.
