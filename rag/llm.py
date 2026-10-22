"""LLM layer: OpenAI-compatible chat client + prompt with citations. Stdlib only."""
import json
import os
import urllib.request

SYSTEM = ("Answer only from the numbered sources. If they do not contain the answer, say so "
          "plainly. Cite sources as [n]. Be brief: three sentences at most.")


class LLMError(RuntimeError):
    pass


class OpenAICompat:
    """Works with OpenAI, DeepSeek, Azure OpenAI-compatible gateways, local vLLM."""
    def __init__(self, base_url=None, api_key=None, model=None, timeout=60):
        self.base_url = (base_url or os.getenv("LLM_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self.api_key = api_key or os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.model = model or os.getenv("LLM_MODEL") or "gpt-4o-mini"
        self.timeout = timeout

    def chat(self, messages: list, temperature: float = 0.2) -> str:
        if not self.api_key:
            raise LLMError("no API key: set LLM_API_KEY (or OPENAI_API_KEY)")
        body = json.dumps({"model": self.model, "messages": messages, "temperature": temperature}).encode()
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=body,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                payload = json.loads(r.read().decode())
        except Exception as exc:  # network, auth, quota
            raise LLMError(f"llm request failed: {exc}") from exc
        try:
            return payload["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"unexpected response shape: {payload}") from exc


def build_prompt(question: str, hits: list) -> list:
    sources = "\n\n".join(
        f"[{i}] {h['title']}\n{h['text'][:800]}" for i, h in enumerate(hits, 1))
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Sources:\n{sources}\n\nQuestion: {question}"},
    ]


def answer(question, index, k=5, client=None):
    hits = index.search(question, k=k)
    if not hits:
        return {"answer": "No matching postings.", "sources": []}
    client = client or OpenAICompat()
    text = client.chat(build_prompt(question, hits))
    return {"answer": text, "sources": [{"n": i, "title": h["title"], "url": h["url"]}
                                        for i, h in enumerate(hits, 1)]}
