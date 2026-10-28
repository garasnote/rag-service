"""HTTP layer. Needs the api extra:

    pip install -e ".[api]"

    uvicorn api:app --reload --port 8100
    curl -X POST localhost:8100/ask -H 'Content-Type: application/json' \
         -d '{"question":"ofertas junior de computer vision en Espana"}'

Sources carry a passage citation: ``chunk_index``, ``start``, ``end`` and
``snippet``. ``text`` is still the whole posting, so callers that use it keep
working. Set ``RAG_CHUNK=0`` to index whole documents instead of windows.
"""
import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from rag.chunking import ChunkedBM25, citation_fields
from rag.core import BM25, load_corpus
from rag.llm import LLMError, answer as rag_answer

app = FastAPI(title="job-market-rag", version="0.1.0")
_index = None


def get_index():  # BM25, or ChunkedBM25 when passage-level citations are on
    global _index
    if _index is None:
        corpus = os.getenv("RAG_CORPUS")
        docs = load_corpus(corpus=corpus) if corpus else load_corpus()
        _index = BM25(docs) if os.getenv("RAG_CHUNK") == "0" else ChunkedBM25(docs)
    return _index


class Ask(BaseModel):
    question: str
    k: int = 5


@app.get("/health")
def health():
    return {"status": "ok", "documents": len(get_index().docs)}


@app.get("/search")
def search(q: str, k: int = 5):
    if not q.strip():
        raise HTTPException(status_code=422, detail="q is empty")
    hits = get_index().search(q, k=min(k, 20))
    return {"results": [{**h, **citation_fields(h)} for h in hits]}


@app.post("/ask")
def ask(req: Ask):
    if not req.question.strip():
        raise HTTPException(status_code=422, detail="question is empty")
    try:
        return rag_answer(req.question, get_index(), k=req.k)
    except LLMError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
