# job-market-rag

Ask questions about 400 job postings and application records you already own. Retrieval is
BM25 plus a dense index written from scratch and fused with RRF; the answer comes from any
OpenAI-compatible LLM, with citations. HyDE is implemented and measured, not assumed.

## Why this exists

Every GenAI job posting in Spain asks for the same three things: RAG, embeddings or retrieval,
and evaluation. This repo is those three things on a corpus nobody else has — my own job hunt.

## Measured, not claimed

`python3 eval/run_eval.py` over 15 hand-written questions with known answers, 400 documents.

**Hashed bag-of-words embedder** (zero dependencies, the default):

```
config     hit@1  hit@3    MRR
bm25         80%    87%   0.850
dense        53%    53%   0.567
hybrid       60%    73%   0.697
hybrid 3:1   67%    73%   0.733
```

**all-MiniLM-L6-v2, 384-d, ONNX** (`--embedder minilm`):

```
config        hit@1  hit@3    MRR
bm25            80%    87%   0.850
dense           60%    60%   0.600
hybrid          67%    80%   0.763
hybrid 3:1      73%    87%   0.802
cascade@10      67%    80%   0.722
```

Read that honestly, because it is the most useful thing in the repo:

- **BM25 still wins.** With a real semantic model the dense side improves from 0.567 to 0.600
  MRR, and fusion with BM25 climbs from 0.697 to 0.802 — and still loses to lexical alone.
- **Why:** these are 400 short, entity-heavy documents ("Datavant junior", "PySpark remoto") and
  the questions name the entity. Rare exact tokens are exactly BM25's strength; MiniLM blurs them.
  Embeddings earn their place on *vocabulary mismatch* — paraphrase queries over long prose —
  which this corpus does not have. That is a property of the data, not of the model.
- **Reranking a BM25 shortlist with the embedder made it worse** at every pool size tried
  (10, 20, 40), because the first stage rarely loses the right document and cosine adds noise.
- **The HyDE row is a ceiling, not a result.** Without an API key the eval substitutes the target
  document's own text for the hypothetical passage, so of course it scores 100%.

One question fails in every configuration ("empresa que pedía FastAPI y PostgreSQL en Madrid")
because the record's notes never mention FastAPI — a retrieval miss caused by the data.

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

## Run

```bash
python3 eval/run_eval.py                   # hashed embedder, no dependencies
pip install numpy onnxruntime tokenizers    # for the real model
export MINILM_DIR=~/models/minilm           # model.onnx + tokenizer.json
python3 eval/run_eval.py --embedder minilm
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
