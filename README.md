# job-market-rag

Ask questions about 382 job postings and application records you already own. Retrieval is
BM25 written from scratch; the answer comes from any OpenAI-compatible LLM, with citations.

## Why this exists

Every GenAI job posting in Spain asks for the same three things: RAG, embeddings or retrieval,
and evaluation. This repo is those three things on a corpus nobody else has — my own job hunt.

## Measured, not claimed

`python3 eval/run_eval.py` over 15 hand-written questions with known answers:

```
corpus: 382 documentos | preguntas: 15
hit@1 : 12/15 = 80%
hit@3 : 13/15 = 87%
MRR   : 0.850
```

One question fails ("empresa que pedía FastAPI y PostgreSQL en Madrid") because the record's
notes never mention FastAPI — a retrieval miss caused by the data, which is the honest kind.

## How it works

```
job_tracker/data.json + warehouse.db
        │
        ▼
  load_corpus()  → one document per posting/record (title + text + url)
        │
        ▼
  BM25 index     → idf, tf saturation, length normalisation, k1=1.5 b=0.75
        │
        ▼
  top-k passages → prompt with [n] citations → LLM → answer + sources
```

No vector database on purpose: at 382 documents BM25 is faster, needs no embeddings bill, and
gives an honest baseline to measure any embedding model against.

## Run

```bash
python3 eval/run_eval.py            # retrieval quality, no API key needed
python3 -m unittest discover -s tests -v   # 8 tests, all offline
pip install -r requirements.txt
export LLM_API_KEY=... LLM_BASE_URL=... LLM_MODEL=...
uvicorn api:app --port 8100
curl -X POST localhost:8100/ask -H 'Content-Type: application/json'      -d '{"question":"ofertas junior de computer vision en Espana"}'
```

## What it does not do yet

- Embeddings and a vector store. That is the next step, and the eval harness is what will tell
  me whether it actually beats BM25 or just costs more.
- Reranking, streaming, conversation memory.
- Auth. `/ask` is open; if this ever faces the internet it gets a key and a rate limit first.

## What I say about it in an interview

"I built the retrieval baseline first and measured it, so that any embedding model I add has to
beat 0.85 MRR to justify itself. The one failure in the eval comes from missing data, not from
the ranking, and I know that because I looked at the failing question instead of the average."
