"""python3 -m unittest discover -s tests -v   (stdlib only, no network, no API key)"""
import json
import pathlib
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag.core import BM25, chunk, load_corpus, strip_accents, tokenize  # noqa: E402
from rag.llm import LLMError, OpenAICompat, build_prompt  # noqa: E402


class TestCore(unittest.TestCase):
    def test_accents_and_stopwords(self):
        self.assertEqual(strip_accents("Cámara"), "Camara")
        self.assertNotIn("de", tokenize("Cámara de Comercio de Bogotá"))
        self.assertIn("camara", tokenize("Cámara de Comercio"))

    def test_bm25_ranks_rare_term_first(self):
        docs = [{"text": "junior data analyst power bi"}, {"text": "snow segmentation webcams pytorch"},
                {"text": "backend python fastapi postgres"}]
        hits = BM25(docs).search("pytorch webcams", k=3)
        self.assertTrue(hits and "pytorch" in hits[0]["text"])

    def test_chunk_overlap(self):
        parts = chunk("x" * 1500, size=600, overlap=80)
        self.assertGreater(len(parts), 1)
        self.assertLessEqual(max(len(p) for p in parts), 600)

    def test_corpus_loads_with_urls(self):
        docs = load_corpus()
        self.assertGreater(len(docs), 200, "corpus should include tracker + warehouse")
        self.assertTrue(all(d["id"] and d["text"] for d in docs))

    def test_retrieval_finds_known_employer(self):
        docs = load_corpus()
        hits = BM25(docs).search("Datavant junior data scientist Barcelona", k=3)
        self.assertTrue(any("datavant" in h["text"].lower() for h in hits))


class _Mock(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)
        body = json.dumps({"choices": [{"message": {"content": "respuesta simulada [1]"}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


class TestLLM(unittest.TestCase):
    def test_prompt_contains_sources_and_question(self):
        msgs = build_prompt("¿qué ofertas hay?", [{"title": "A — B", "text": "texto"}])
        self.assertIn("Question: ¿qué ofertas hay?", msgs[-1]["content"])
        self.assertIn("[1] A — B", msgs[-1]["content"])

    def test_missing_key_raises(self):
        with self.assertRaises(LLMError):
            OpenAICompat(api_key=None, base_url="http://localhost:1").chat([{"role": "user", "content": "x"}])

    def test_client_parses_mock_response(self):
        srv = HTTPServer(("127.0.0.1", 0), _Mock)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            client = OpenAICompat(base_url=f"http://127.0.0.1:{srv.server_port}", api_key="test", model="m")
            self.assertEqual(client.chat([{"role": "user", "content": "hola"}]), "respuesta simulada [1]")
        finally:
            srv.shutdown()
            srv.server_close()


if __name__ == "__main__":
    unittest.main()
