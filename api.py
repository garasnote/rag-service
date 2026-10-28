"""HTTP layer. Needs the api extra:

    pip install -e ".[api]"

    uvicorn api:app --reload --port 8100
    curl -X POST localhost:8100/ask -H 'Content-Type: application/json' \
         -d '{"question":"ofertas junior de computer vision en Espana"}'
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from rag.core import BM25, load_corpus
from rag.llm import LLMError, answer as rag_answer

app = FastAPI(title="job-market-rag", version="0.1.0")
_index = None


def get_index() -> BM25:
    global _index
    if _index is None:
        _index = BM25(load_corpus())
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
    return {"results": get_index().search(q, k=min(k, 20))}


@app.post("/ask")
def ask(req: Ask):
    if not req.question.strip():
        raise HTTPException(status_code=422, detail="question is empty")
    try:
        return rag_answer(req.question, get_index(), k=req.k)
    except LLMError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
