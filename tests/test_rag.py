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


class EmbedTest(unittest.TestCase):
    def test_deterministic_and_normalised(self):
        from rag.embed import HashingEmbedder, cosine
        e = HashingEmbedder(128)
        a = e.vector("junior data scientist barcelona")
        b = e.vector("junior data scientist barcelona")
        self.assertEqual(a, b)
        self.assertAlmostEqual(cosine(a, a), 1.0, places=6)
        self.assertAlmostEqual(sum(x * x for x in a) ** 0.5, 1.0, places=6)
        self.assertLess(cosine(a, e.vector("camión de reparto en valladolid")), 0.5)

    def test_vector_index_ranks_the_right_doc_first(self):
        from rag.embed import VectorIndex
        docs = [
            {"id": "a", "text": "analista de datos power bi etl excel sql remoto"},
            {"id": "b", "text": "cocinero restaurante jornada completa madrid"},
            {"id": "c", "text": "ingeniero de machine learning pytorch vision"},
        ]
        idx = VectorIndex(docs)
        self.assertEqual(idx.search_text("power bi etl sql analista", k=1)[0]["id"], "a")
        self.assertEqual(idx.search_text("pytorch vision machine learning", k=1)[0]["id"], "c")


class FusionTest(unittest.TestCase):
    def _doc(self, i):
        return {"id": i, "text": i}

    def test_doc_in_both_rankings_wins(self):
        from rag.fusion import rrf
        lex = [self._doc("x"), self._doc("y"), self._doc("z")]
        den = [self._doc("y"), self._doc("w"), self._doc("x")]
        fused = rrf([lex, den], top=3)
        self.assertEqual(fused[0]["id"], "y")  # 2nd + 1st (0.0325) beats 1st + 3rd (0.0323)
        self.assertEqual({d["id"] for d in fused}, {"x", "y", "w"})

    def test_weights_shift_the_order(self):
        from rag.fusion import rrf
        lex = [self._doc("x"), self._doc("y")]
        den = [self._doc("y"), self._doc("x")]
        self.assertEqual(rrf([lex, den], top=2, weights=[5.0, 1.0])[0]["id"], "x")
        self.assertEqual(rrf([lex, den], top=2, weights=[1.0, 5.0])[0]["id"], "y")


class HydeTest(unittest.TestCase):
    def test_prompt_contains_question_and_forbids_invention(self):
        from rag.hyde import build_prompt
        p = build_prompt("¿Hay ofertas junior de RAG en Barcelona?")
        self.assertIn("RAG en Barcelona", p)
        self.assertIn("Invent no names", p)

    def test_hyde_search_uses_the_generated_passage(self):
        from rag.embed import VectorIndex
        from rag.hyde import hyde_search
        class Stub:
            model = "stub"
            def chat(self, messages):
                self.seen = messages
                return "oferta junior de power bi y sql en barcelona"
        docs = [
            {"id": "a", "text": "junior power bi sql barcelona analista"},
            {"id": "b", "text": "cocinero madrid"},
        ]
        idx = VectorIndex(docs)
        client = Stub()
        out = hyde_search("¿algo de BI junior?", client, idx, k=1)
        self.assertIn("power bi", out["hypothetical"])
        self.assertEqual(out["hits"][0]["id"], "a")
        self.assertIn("¿algo de BI junior?", client.seen[0]["content"])


class EmbedderFactoryTest(unittest.TestCase):
    def test_hash_is_the_default_and_unknown_names_fail(self):
        from rag.embed import get_embedder
        self.assertEqual(get_embedder().name, "hashing-512")
        self.assertEqual(get_embedder("bow").name, "hashing-512")
        with self.assertRaises(SystemExit):
            get_embedder("does-not-exist")

    def test_minilm_is_optional(self):
        import os
        from rag.embed import get_embedder
        if not os.getenv("MINILM_DIR"):
            self.skipTest("MINILM_DIR not set")
        e = get_embedder("minilm")
        v = e.vector("junior data scientist barcelona")
        self.assertEqual(len(v), 384)
        self.assertAlmostEqual(sum(x * x for x in v) ** 0.5, 1.0, places=5)
