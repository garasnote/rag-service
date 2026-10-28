# job-market-rag

**Ask questions about your own job hunt in plain language, and get answers with citations.**

![python](https://img.shields.io/badge/python-3.9%2B-3776AB?logo=python&logoColor=white)
![core](https://img.shields.io/badge/core-stdlib%20only-2ea44f)
![tests](https://img.shields.io/badge/tests-15%20passed-2ea44f)
![license](https://img.shields.io/badge/license-MIT-blue)
![corpus](https://img.shields.io/badge/corpus-2%2C576%20documents-6f42c1)

Retrieval over **2,576 job postings and application records** — BM25 and a dense
index, fused with Reciprocal Rank Fusion, answered by any OpenAI-compatible LLM,
with a citation on every claim. Every configuration was measured on the same
benchmark, and the ones that lost are still in the tables.

```
$ python3 -m rag.cli "which junior jobs are in Barcelona?"
[1] Datavant — Junior Data Scientist
    https://…/datavant-junior
    Datavant junior data scientist Barcelona linked health records python sql
[2] Metyis — Data Engineering Analyst
    https://…/metyis-dea
    Metyis Barcelona data engineering analyst junior sql python etl
[3] Buynomics — Junior Data Scientist
    https://…/buynomics-js
    Buynomics junior data scientist Barcelona pricing analytics python pandas
```

---

## Quickstart

Thirty seconds, no API key, no network, no private data — a sixteen-document
corpus and eight answered questions are checked in.

```bash
git clone https://github.com/garasnote/rag-service && cd rag-service
pip install -e .
python3 eval/run_eval.py --corpus fixtures/corpus.json --questions fixtures/questions.json
python3 -m pytest -q
```

```
corpus: 16 documents | questions: 8 | embedder: hashing-512

config        hit@1  hit@3    MRR
bm25            88%   100%  0.917
dense           88%   100%  0.938
hybrid          88%   100%  0.938
hybrid 3:1      88%   100%  0.917
```

Point it at your own records — a JSON list of `{id, title, text, url}` — and ask
for rankings:

```bash
python3 -m rag.cli "remote LLM roles in Spain" --corpus my_postings.json -k 5
```

---

## How it works

```
        ┌──────────────┐   ┌────────────────┐      one document per posting,
        │ job tracker  │   │ ATS warehouse  │      every one keeps its URL
        │   (JSON)     │   │    (SQLite)    │
        └──────┬───────┘   └───────┬────────┘
               └────────┬──────────┘
                        ▼
                 ┌─────────────┐
                 │   corpus    │   2,576 documents, deduplicated by URL
                 └──────┬──────┘
            ┌───────────┴────────────┐
            ▼                        ▼
     ┌─────────────┐          ┌─────────────┐
     │    BM25     │          │    dense    │   brute-force cosine over
     │  Okapi, k1  │          │    index    │   L2-normalised vectors
     └──────┬──────┘          └──────┬──────┘
            └───────────┬────────────┘
                        ▼
                 ┌─────────────┐
                 │     RRF     │   rank fusion, optional weighting
                 └──────┬──────┘
                        ▼
                 ┌─────────────┐
                 │  LLM answer │   OpenAI-compatible, forced to cite [n]
                 │  + sources  │   every claim maps to a numbered document
                 └─────────────┘
```

**No framework.** BM25 is fifty lines of Okapi, the vector index is a cosine
loop, RRF is five lines, the eval harness is a list of questions. A framework
would hide the parts worth reading.

---

## Results

2,576 documents, 30 questions (15 Spanish, 15 English), identical questions
across every configuration.

### Entity questions, same language and cross-language

| embedder | config | hit@1 | hit@3 | MRR |
|---|---|---|---|---|
| hashing-512 | bm25 | **50%** | **70%** | **0.594** |
| hashing-512 | dense | 33% | 47% | 0.394 |
| hashing-512 | hybrid | 53% | 63% | 0.572 |
| hashing-512 | hybrid 3:1 | 50% | 63% | 0.568 |
| multilingual-MiniLM | bm25 | **50%** | **70%** | **0.594** |
| multilingual-MiniLM | dense | 30% | 40% | 0.340 |
| multilingual-MiniLM | hybrid | 43% | 63% | 0.533 |

Per language with the multilingual encoder: Spanish **0.667** MRR, English
**0.522** against a corpus that is mostly Spanish — the same split BM25 shows,
which locates the loss in the documents rather than the encoder.

### Paraphrase questions, no entity names

Eight queries that describe a role instead of naming it — the case embeddings
are supposed to win:

| config | hit@1 | hit@3 | MRR |
|---|---|---|---|
| bm25 | 0% | 0% | 0.000 |
| dense (multilingual) | 0% | 12% | 0.042 |
| dense (hashing) | 12% | 12% | 0.125 |
| hybrid | 0% | 0% | 0.000 |

### HyDE, measured instead of assumed

Ten DeepSeek passages per question, each retrieved, fused with RRF:

| config | MRR |
|---|---|
| bm25 | **0.622** |
| hyde, 10 samples fused | 0.419 |
| hyde, single sample | 0.309 |

---

## What the numbers say

**BM25 wins, and the reason is the data.** A job record is two lines: company,
title, city, stack. What separates 2,576 junior data roles from each other is
*which words are in them* — the company, the town, the one tool that matters.
That is a lexical problem. Embeddings put "Datavant junior in Barcelona" and
"Buynomics junior in Barcelona" almost on top of each other, which is fatal when
the difference is the whole query.

**A 113 MB multilingual model loses to 512 dimensions of hashed bag-of-words**
(0.340 vs 0.394). Not a bug in the model — a statement about the task. Hashing
keeps exact tokens; the transformer keeps meaning, and the meaning here is "a
junior data job", shared by a thousand documents.

**Cross-lingual retrieval does not rescue it.** English questions reach Spanish
records at 0.522, but they arrive on the same company names. Remove the names
and both languages fail together, which is the paraphrase table.

**HyDE hurts by a third.** Hypothetical passages are fluent prose; the documents
are two-line records. Invented specifics — plausible companies, plausible
cities — steer retrieval toward the wrong rows, and RRF over ten noisy rankings
fuses the noise instead of cancelling it. Implemented, off by default, and the
table above is the reason.

**The questions that fail, fail for data reasons.** "Company that asked for FastAPI and
PostgreSQL in Madrid" is missed by several configurations because that record never mentions
FastAPI; "Datavant junior in Barcelona" is drowned by a corpus full of similar Barcelona junior
records. The eval prints the failure list per configuration instead of averaging it away.

---

## Design decisions

**Standard library core.** Retrieval, fusion and evaluation run on
`pip install -e .` and nothing else. Heavy pieces are extras: `.[embeddings]` for
ONNX encoders, `.[api]` for HTTP. A reviewer can reproduce the benchmark on a
fresh machine in under a minute.

**Deterministic evaluation.** Thirty questions with known answers, scored by hit
rate and MRR, no LLM in the loop. A judge model would make the numbers
unreproducible and the comparison worthless.

**RRF instead of score normalisation.** BM25 scores and cosine similarities live
on different scales; fusing ranks avoids a weighted sum whose weights nobody can
defend. The 3:1 variant exists to test whether BM25 deserves more weight — it
does, marginally, and the table shows it.

**Citations are structural.** The prompt forces `[n]` markers and the API returns
a `sources` array beside the answer, so a client renders links without parsing
prose. Retrieval always carries URLs, never bare text.

**Fixtures, not private data.** The corpus is a personal job tracker and is not
in the repo. `fixtures/` holds a small representative corpus plus questions,
which is what CI runs.

---

## API

```bash
pip install -e ".[api]"
uvicorn api:app --port 8100

curl -s localhost:8100/search -H 'content-type: application/json' \
     -d '{"q":"junior computer vision Spain","k":5}'

curl -s localhost:8100/ask -H 'content-type: application/json' \
     -d '{"question":"which junior roles mention PyTorch?","k":5}'
```

`POST /search` returns ranked documents; `POST /ask` returns
`{"answer": "...", "sources": [{"n": 1, "title": "...", "url": "..."}]}`.

Set `OPENAI_BASE_URL`, `OPENAI_API_KEY` and `OPENAI_MODEL` to any
OpenAI-compatible endpoint. Without a key the retrieval half still works, and
`hyde` falls back to an oracle stand-in that is labelled as such in every output
it touches.

---

## Limitations

- **Thirty questions is a small benchmark.** Enough to rank configurations
  against each other, not enough to claim absolute quality.
- **No ANN index.** Brute-force cosine over 2,576 documents takes milliseconds;
  at a hundred thousand it would need one, and the eval would need to prove the
  index did not cost recall.
- **No reranker that helps.** A cross-encoder over the BM25 shortlist was tried
  at pool sizes 10, 20 and 40 and lost at every size on this corpus.
- **Spanish morphology is unhandled.** "analistas" does not match "analista".
  A naive suffix stripper made things worse (0.917 → 0.667 on the fixture)
  because it also shreds company names.
- **Answers are only as good as retrieval.** The model is constrained to the
  retrieved context, so a miss stays a miss — it will say so rather than invent.

## Roadmap

- [ ] ANN index (hnswlib) with a recall-versus-latency measurement
- [ ] A reranker that is measured to help, not assumed to
- [ ] Spanish lemmatisation with a real stemmer, ablation kept in `eval/`
- [ ] Chunk-level citations for long postings
- [ ] Streamlit front end over `/ask`

## License

MIT — see [LICENSE](LICENSE).
