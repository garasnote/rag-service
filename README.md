# job-market-rag

Ask questions about 400 job postings and application records you already own. Retrieval is
BM25 plus a dense index written from scratch and fused with RRF; the answer comes from any
OpenAI-compatible LLM, with citations. HyDE is implemented and measured, not assumed.

## Why this exists

Every GenAI job posting in Spain asks for the same three things: RAG, embeddings or retrieval,
and evaluation. This repo is those three things on a corpus nobody else has — my own job hunt.

## Measured, not claimed

`python3 eval/run_eval.py` over 15 hand-written questions with known answers:

```
corpus: 400 documentos | preguntas: 15 | embedder: hashing-512

config    hit@1  hit@3    MRR
bm25        80%    87%  0.850
dense       53%    53%  0.567
hybrid      60%    73%  0.697
hybrid3:1   67%    73%  0.733
hyde       100%   100%  1.000   <- oracle ceiling, not a measurement
```

Read that table honestly, because it is the most useful thing in the repo:

- **BM25 wins on this corpus.** At 400 short, keyword-dense documents, lexical matching is hard
  to beat.
- **The dense index loses** because it is a hashed bag of words, not a semantic model. It has no
  idea that "analista" and "analytics" are related. Swap in MiniLM or an embedding API and the
  ranking changes; the harness is how I would find out, instead of trusting the vendor.
- **Fusion hurt.** Blending a weak dense run into a strong lexical one with equal weights cost
  0.15 MRR; weighting BM25 3:1 clawed some back, still below BM25 alone. RRF is not free.
- **The HyDE row is a ceiling, not a result.** Without an API key the eval substitutes the
  target document's own text for the hypothetical passage, so of course it scores 100%. It
  proves the shape of the idea (retrieve with answer-shaped text and the ranking collapses to
  the right document) and says nothing about what an LLM would actually write.

One question fails ("empresa que pedía FastAPI y PostgreSQL en Madrid") because the record's
notes never mention FastAPI — a retrieval miss caused by the data, which is the honest kind.

## Retrieval modes

| mode | what it does | when to use it |
|---|---|---|
| `BM25` | Okapi BM25, k1=1.5 b=0.75 | default; best on this corpus |
| `VectorIndex` | hashed 512-dim embeddings + cosine | when a semantic embedder is plugged in |
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

## Run

```bash
python3 eval/run_eval.py                   # four configurations, no API key needed
python3 eval/run_eval.py --hyde            # HyDE with a real LLM
python3 -m unittest discover -s tests -v   # 14 tests, all offline
pip install -r requirements.txt
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
