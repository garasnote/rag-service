# job-market-rag

Ask questions about 2,576 job postings and application records you already own. Retrieval is
BM25 plus a dense index written from scratch and fused with RRF; the answer comes from any
OpenAI-compatible LLM, with citations. HyDE is implemented and measured — and on this corpus it
loses, which is the most useful thing in the repo.

## Why this exists

Every GenAI job posting in Spain asks for the same three things: RAG, embeddings or retrieval,
and evaluation. This repo is those three things on a corpus nobody else has — my own job hunt.

## Measured, not claimed

`python3 eval/run_eval.py` — 15 hand-written questions with known answers, against the live
corpus: **2,576 documents** (my job tracker plus every posting in the warehouse), hashing
embedder, run 28 Oct 2026.

```
config        hit@1  hit@3    MRR
bm25            53%    73%   0.622
dense           40%    53%   0.467
hybrid          60%    73%   0.656
hybrid 3:1      53%    73%   0.622
hyde 10x        27%    53%   0.419
hyde 1x          —      —     0.309
```

Read that honestly:

- **On 400 documents BM25 won at 0.850 and the README said so.** The corpus then grew sixfold
  (job tracker + warehouse), the questions did not change, and the ranking inverted: dense
  contributions now help, hybrid leads at 0.656, BM25 falls to 0.622. Five times more documents
  means five times more lexical near-misses. Scaling the corpus is a change to the problem, not
  to the model, and the eval has to be re-run every time it happens.
- **HyDE makes it much worse here, and that is a real measurement, not an assumption.** Ten
  DeepSeek passages per question, each retrieved and fused with RRF: 0.419 MRR fused, 0.309 for a
  single call, against 0.622 for plain BM25. The hypothetical passages are four to six sentences
  of fluent prose; the documents are two-line records ("Datavant junior", "PySpark remoto"). The
  embedder cannot place the two in the same region, and the invented specifics — plausible
  company names, plausible cities — pull retrieval toward the wrong records. RRF over ten noisy
  rankings fuses the noise rather than cancelling it. HyDE is off by default, and this is why.
- **Why BM25 is strong on this data:** short, entity-heavy documents, and questions that name
  the entity. Rare exact tokens are exactly BM25's strength.
- **Dense retrieval earns its place on vocabulary mismatch** — paraphrase queries over long
  prose — which this corpus does not have.
- **Reranking a BM25 shortlist with the embedder made it worse** at every pool size tried
  (10, 20, 40).
- **One question fails in every configuration** ("empresa que pedía FastAPI y PostgreSQL en
  Madrid") because the record's notes never mention FastAPI — a retrieval miss caused by the
  data.

The earlier all-MiniLM-L6-v2 table (dense 0.600, hybrid 0.802 on 400 documents) has not been
re-measured since the corpus grew; treat those numbers as historical, not current.

## Retrieval modes

| mode | what it does | when to use it |
|---|---|---|
| `BM25` | Okapi BM25, k1=1.5 b=0.75 | default; best on this corpus |
| `VectorIndex` | dense vectors + cosine; `hash` (512-d) or MiniLM (384-d) or any /embeddings API | when vocabulary mismatch is the problem |
| `rrf` | reciprocal rank fusion of any rankings, optional weights | combining lexical + dense |
| `hyde` | LLM writes a hypothetical passage, retrieval uses that | queries phrased as questions, not keywords |

HyDE with a real model:

```bash
export LLM_API_KEY=... LLM_BASE_URL=https://api.deepseek.com/v1 LLM_MODEL=deepseek-chat
python3 eval/run_eval.py --hyde    # measures HyDE instead of the oracle
```

Cost: one extra LLM call per query. Failure mode: the model invents a plausible posting that
does not exist, and you retrieve documents similar to the hallucination. That is why it is off
by default and why it has to beat 0.85 MRR to be worth switching on.

## How it works

```
job_tracker/data.json + warehouse.db
        │
        ▼
  load_corpus()   → one document per posting/record (title + text + url)
        │
        ├── BM25 index          → lexical ranking
        └── VectorIndex         → hashed embeddings + cosine
        │
        ▼
  rrf fusion → top-k passages → prompt with [n] citations → LLM → answer + sources
```

No vector database on purpose: at 400 documents brute-force cosine is honest and a FAISS
dependency would be vanity. The interface (`vector(text) -> list[float]`, L2-normalised) is the
seam where a real embedding model drops in.

## Install

The core is standard library only — clone and run, nothing to install:

```bash
python3 eval/run_eval.py                       # BM25 + hashed dense + fusion
python3 -m unittest discover -s tests -v       # 16 tests, offline
```

The real embedder and the HTTP API are opt-in extras. Pick one environment manager:

```bash
# uv
uv venv && uv pip install -e ".[embeddings,api]"

# venv + pip
python3 -m venv .venv && . .venv/bin/activate && pip install -e ".[embeddings,api]"

# micromamba
micromamba create -n rag python=3.11 -c conda-forge -y
micromamba activate rag && pip install -e ".[embeddings,api]"
```

## Model

```bash
mkdir -p ~/models/minilm && cd ~/models/minilm
curl -LO https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/onnx/model.onnx
curl -LO https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/tokenizer.json
```

## Run

```bash
MINILM_DIR=~/models/minilm python3 eval/run_eval.py --embedder minilm
export LLM_API_KEY=... LLM_BASE_URL=... LLM_MODEL=...
uvicorn api:app --port 8100
curl -X POST localhost:8100/ask -H 'Content-Type: application/json' \
  -d '{"question":"ofertas junior de computer vision en Espana"}'
```

## What it does not do yet

- A real embedding model. The harness is ready for one; the hashed embedder is a placeholder
  that is cheap and offline, and the numbers above show exactly what that costs.
- Reranking, streaming, conversation memory.
- Auth. `/ask` is open; if this ever faces the internet it gets a key and a rate limit first.

## What I say about it in an interview

"I built the lexical baseline first and measured it, so anything I add has to beat 0.85 MRR to
justify itself. I implemented dense retrieval and HyDE, and both lost on my corpus — the dense
index because a hashed bag of words is not semantic, HyDE because I would only trust it with a
real LLM in the loop. The eval table is the point: I would rather know that than ship the
architecture diagram."
