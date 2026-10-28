"""Dense retrieval: two embedders behind one interface.

vector(text) -> L2-normalised list[float]

  HashingEmbedder  zero dependencies, lexical, offline. A baseline, not a model.
  MiniLMEmbedder   all-MiniLM-L6-v2 (384-d, 22M params) on onnxruntime.
                   Needs numpy + onnxruntime + tokenizers and a local model dir:
                     MINILM_DIR=/path/to/model  python3 eval/run_eval.py --embedder minilm
                   Files: model.onnx, tokenizer.json (from
                   hf.co/sentence-transformers/all-MiniLM-L6-v2).

Why ONNX and not sentence-transformers: 90 MB of weights and no torch. The same
interface also accepts an HTTP embedding endpoint (see ApiEmbedder).
"""
import hashlib
import json
import math
import os
import pathlib
import re
import urllib.request
from collections import Counter

from .core import tokenize


def _hash(feature: str) -> int:
    return int.from_bytes(hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest(), "big")


class HashingEmbedder:
    name = "hashing-512"

    def __init__(self, dim: int = 512):
        self.dim = dim

    def vector(self, text: str) -> list:
        toks = tokenize(text)
        feats = Counter(toks)
        for a, b in zip(toks, toks[1:]):
            feats[f"{a}_{b}"] += 1
        v = [0.0] * self.dim
        for feature, count in feats.items():
            h = _hash(feature)
            idx = h % self.dim
            sign = 1.0 if (h >> 63) & 1 else -1.0
            v[idx] += sign * (1.0 + math.log(count))
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]


class MiniLMEmbedder:
    """all-MiniLM-L6-v2 via onnxruntime: mean pooling over the last hidden state."""

    name = "minilm-l6-v2-384"

    def __init__(self, model_dir=None, max_len: int = 256):
        import numpy as np
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self.np = np
        model_dir = pathlib.Path(model_dir or os.getenv("MINILM_DIR") or "")
        if not (model_dir / "model.onnx").exists():
            raise FileNotFoundError(
                f"MiniLM model not found in {model_dir or 'MINILM_DIR'}: needs model.onnx and tokenizer.json")
        self.tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self.tok.enable_truncation(max_length=max_len)
        self.tok.enable_padding()
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = int(os.getenv("ORT_THREADS", "8"))
        self.sess = ort.InferenceSession(str(model_dir / "model.onnx"), opts,
                                         providers=["CPUExecutionProvider"])
        self.inputs = {i.name for i in self.sess.get_inputs()}

    def vector(self, text: str) -> list:
        np = self.np
        enc = self.tok.encode(text or "")
        ids = np.array([enc.ids], dtype=np.int64)
        mask = np.array([enc.attention_mask], dtype=np.int64)
        feed = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self.inputs:
            feed["token_type_ids"] = np.zeros_like(ids)
        hidden = self.sess.run(None, feed)[0]          # (1, seq, 384)
        m = mask[..., None].astype(hidden.dtype)
        pooled = (hidden * m).sum(axis=1) / np.clip(m.sum(axis=1), 1e-9, None)
        pooled = pooled[0]
        norm = float(np.linalg.norm(pooled)) or 1.0
        return (pooled / norm).astype("float32").tolist()

    def vectors(self, texts, batch_size: int = 32) -> list:
        """Batched: one ONNX run per 32 documents instead of one per document."""
        np = self.np
        out = []
        texts = list(texts)
        for i in range(0, len(texts), batch_size):
            encs = self.tok.encode_batch(texts[i:i + batch_size])
            ids = np.array([e.ids for e in encs], dtype=np.int64)
            mask = np.array([e.attention_mask for e in encs], dtype=np.int64)
            feed = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in self.inputs:
                feed["token_type_ids"] = np.zeros_like(ids)
            hidden = self.sess.run(None, feed)[0]
            m = mask[..., None].astype(hidden.dtype)
            pooled = (hidden * m).sum(axis=1) / np.clip(m.sum(axis=1), 1e-9, None)
            norms = np.linalg.norm(pooled, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            out.extend((pooled / norms).astype("float32").tolist())
        return out


class ApiEmbedder:
    """Any OpenAI-compatible /embeddings endpoint (OpenAI, Voyage via gateway, vLLM)."""

    def __init__(self, base_url=None, api_key=None, model=None):
        self.base_url = (base_url or os.getenv("EMBED_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self.api_key = api_key or os.getenv("EMBED_API_KEY") or os.getenv("LLM_API_KEY")
        self.model = model or os.getenv("EMBED_MODEL") or "text-embedding-3-small"
        self.name = f"api:{self.model}"

    def vector(self, text: str) -> list:
        if not self.api_key:
            raise RuntimeError("ApiEmbedder needs EMBED_API_KEY (or LLM_API_KEY)")
        body = json.dumps({"model": self.model, "input": text}).encode()
        req = urllib.request.Request(
            f"{self.base_url}/embeddings", data=body,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"})
        with urllib.request.urlopen(req, timeout=60) as r:
            v = json.loads(r.read().decode())["data"][0]["embedding"]
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]


def cosine(a: list, b: list) -> float:
    return sum(x * y for x, y in zip(a, b))


def get_embedder(name: str = "hash"):
    name = (name or "hash").lower()
    if name in ("hash", "hashing", "bow"):
        return HashingEmbedder()
    if name in ("minilm", "sentence-transformers", "st"):
        return MiniLMEmbedder()
    if name in ("api", "openai"):
        return ApiEmbedder()
    raise SystemExit(f"unknown embedder: {name} (hash | minilm | api)")


class VectorIndex:
    """Brute-force cosine search: a few thousand documents, so an ANN index would be vanity.

    Embeddings are cached on disk keyed by embedder name + document ids, because a
    CPU-only box re-embeds 2,000 documents for every eval run otherwise.
    """

    def __init__(self, docs, embedder=None, cache_dir=".cache"):
        self.docs = docs
        self.embedder = embedder or HashingEmbedder()
        self.cache_dir = pathlib.Path(__file__).resolve().parents[1] / cache_dir
        self.vectors = self._vectors()

    def _cache_path(self):
        ids = [d["id"] for d in self.docs]
        digest = hashlib.sha1(json.dumps(ids).encode("utf-8")).hexdigest()[:16]
        safe = re.sub(r"[^a-z0-9]+", "-", self.embedder.name.lower())
        return self.cache_dir / f"{safe}-{len(ids)}-{digest}.json"

    def _vectors(self) -> list:
        path = self._cache_path()
        if path.exists():
            try:
                blob = json.loads(path.read_text())
                if len(blob["vectors"]) == len(self.docs):
                    return blob["vectors"]
            except Exception:
                pass
        vecs = getattr(self.embedder, "vectors", None)
        vectors = vecs([d["text"] for d in self.docs]) if vecs else [self.embedder.vector(d["text"]) for d in self.docs]
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"name": self.embedder.name, "vectors": vectors}))
        except OSError:
            pass
        return vectors

    def search_text(self, text: str, k: int = 5) -> list:
        return self.search_vector(self.embedder.vector(text), k)

    def search_vector(self, qvec: list, k: int = 5) -> list:
        scored = [(cosine(qvec, v), i) for i, v in enumerate(self.vectors)]
        scored = [s for s in scored if s[0] > 0]
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [{"score": round(s, 4), **self.docs[i]} for s, i in scored[:k]]
