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

FIXTURE = ROOT / "fixtures" / "corpus.json"


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
        if len(docs) < 200:
            self.skipTest("private tracker/warehouse not present")
        self.assertGreater(len(docs), 200, "corpus should include tracker + warehouse")
        self.assertTrue(all(d["id"] and d["text"] for d in docs))

    def test_retrieval_finds_known_employer(self):
        docs = load_corpus(corpus=FIXTURE)
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
    def test_prompt_contains_question_and_asks_for_a_record(self):
        from rag.hyde import build_prompt
        p = build_prompt("¿Hay ofertas junior de RAG en Barcelona?")
        self.assertIn("RAG en Barcelona", p)
        # the default prompt mirrors the corpus shape: records, not prose
        self.assertIn("Company", p)
        self.assertIn("Record", p)

    def test_prose_prompt_is_still_available_for_the_ablation(self):
        from rag.hyde import build_prompt
        p = build_prompt("¿Hay ofertas junior de RAG en Barcelona?", style="prose")
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


class ChunkingTest(unittest.TestCase):
    """Passage-level windows: one document stays one result, offsets point at the passage."""

    SHORT = "Junior data analyst, power bi, sql, Madrid, hibrido, jornada completa."

    def _long(self, marker="kubernetes", filler="python sql pandas remoto junior", n=40):
        return (filler + " ") * n + marker + " " + (filler + " ") * n

    def test_short_document_is_exactly_one_chunk_identical_to_the_document(self):
        from rag.chunking import chunk_document
        out = chunk_document("d1", self.SHORT)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["doc_id"], "d1")
        self.assertEqual(out[0]["chunk_index"], 0)
        self.assertEqual((out[0]["start"], out[0]["end"]), (0, len(self.SHORT)))
        self.assertEqual(out[0]["text"], self.SHORT)

    def test_empty_document_has_no_chunks(self):
        from rag.chunking import chunk_document
        self.assertEqual(chunk_document("d1", ""), [])
        self.assertEqual(chunk_document("d1", None), [])

    def test_long_document_is_split_and_still_covered(self):
        from rag.chunking import chunk_document
        text = self._long(n=60)
        out = chunk_document("d1", text)
        self.assertGreater(len(out), 1)
        self.assertLessEqual(max(len(c["text"]) for c in out), 600)
        self.assertEqual(out[0]["start"], 0)
        self.assertEqual(out[-1]["end"], len(text))
        self.assertEqual([c["chunk_index"] for c in out], list(range(len(out))))

    def test_every_offset_slices_back_to_its_chunk_text(self):
        from rag.chunking import chunk_document
        text = self._long(n=60)
        for c in chunk_document("d1", text):
            self.assertEqual(text[c["start"]:c["end"]], c["text"])

    def test_consecutive_windows_overlap_by_the_configured_amount(self):
        from rag.chunking import chunk_document
        out = chunk_document("d1", self._long(n=60), size=600, overlap=100)
        for prev, nxt in zip(out, out[1:]):
            self.assertEqual(nxt["start"], prev["end"] - 100)

    def test_window_boundaries_do_not_cut_words_in_half(self):
        from rag.chunking import chunk_document
        text = self._long(n=60)
        for c in chunk_document("d1", text)[:-1]:
            # a chunk ends either on a space or at the very end of the document
            self.assertTrue(c["text"].endswith(" ") or text[c["end"]] in " ")

    def test_degenerate_overlap_is_rejected(self):
        from rag.chunking import chunk_document
        with self.assertRaises(ValueError):
            chunk_document("d1", "x" * 2000, size=100, overlap=100)

    def test_chunked_bm25_returns_the_document_once_with_its_best_window(self):
        from rag.chunking import ChunkedBM25
        docs = [{"id": "long", "url": "u1", "title": "long", "text": self._long()},
                {"id": "short", "url": "u2", "title": "short", "text": "java spring madrid"}]
        hits = ChunkedBM25(docs).search("kubernetes", k=5)
        self.assertEqual(len(hits), 1, "one document, one result")
        self.assertEqual(hits[0]["id"], "long")
        self.assertEqual(hits[0]["text"], docs[0]["text"], "text stays the whole posting")
        self.assertEqual(hits[0]["url"], "u1")
        chunk = hits[0]["chunk"]
        self.assertGreater(chunk["index"], 0)
        self.assertIn("kubernetes", chunk["text"])
        self.assertEqual(docs[0]["text"][chunk["start"]:chunk["end"]], chunk["text"])

    def test_short_documents_are_cited_as_whole_documents(self):
        from rag.chunking import ChunkedBM25
        docs = [{"id": "a", "text": self.SHORT}, {"id": "b", "text": "cocinero madrid"}]
        hit = ChunkedBM25(docs).search("power bi sql", k=1)[0]
        self.assertEqual(hit["chunk"], {"index": 0, "start": 0, "end": len(self.SHORT),
                                        "text": self.SHORT})
        self.assertEqual(hit["text"], self.SHORT)

    def test_chunked_dense_index_matches_the_plain_one_on_short_documents(self):
        from rag.chunking import ChunkedVectorIndex
        docs = [{"id": "a", "text": "analista de datos power bi etl excel sql remoto"},
                {"id": "b", "text": "cocinero restaurante jornada completa madrid"},
                {"id": "c", "text": "ingeniero de machine learning pytorch vision"}]
        idx = ChunkedVectorIndex(docs)
        self.assertEqual(idx.search_text("power bi etl sql", k=1)[0]["id"], "a")
        self.assertEqual(idx.search_text("pytorch vision", k=1)[0]["id"], "c")
        self.assertEqual(idx.chunks_per_doc, 1.0)

    def test_citation_fields_fall_back_to_the_whole_document(self):
        from rag.chunking import citation_fields, snippet
        hit = {"id": "a", "text": "junior data analyst " * 40}
        fields = citation_fields(hit)
        self.assertEqual(fields["chunk_index"], 0)
        self.assertEqual((fields["start"], fields["end"]), (0, len(hit["text"])))
        self.assertLessEqual(len(fields["snippet"]), 243)
        self.assertEqual(snippet("  hola \n\n mundo  "), "hola mundo")


class ApiSourceTest(unittest.TestCase):
    """/search and /ask sources gain chunk_index, start, end, snippet. No server needed."""

    DOCS = [{"id": "long", "url": "https://x/1", "title": "Encora — Junior ML Engineer",
             "text": ("junior data python sql remoto " * 40) + "kubernetes" + (" relleno" * 40)}]

    def test_ask_sources_carry_the_passage_citation(self):
        from rag.chunking import ChunkedBM25
        from rag.llm import answer
        class Stub:
            model = "stub"
            def chat(self, messages):
                return "una respuesta [1]"
        out = answer("kubernetes", ChunkedBM25(self.DOCS), k=1, client=Stub())
        src = out["sources"][0]
        self.assertEqual((src["n"], src["title"], src["url"]), (1, self.DOCS[0]["title"], "https://x/1"))
        for key in ("chunk_index", "start", "end", "snippet"):
            self.assertIn(key, src)
        self.assertIn("kubernetes", src["snippet"])

    def test_search_sources_carry_the_passage_citation(self):
        try:
            import api  # needs the api extra (fastapi)
        except ImportError:
            self.skipTest("fastapi not installed")
        from rag.chunking import ChunkedBM25
        api._index = ChunkedBM25(self.DOCS)
        try:
            out = api.search(q="kubernetes", k=1)
        finally:
            api._index = None
        hit = out["results"][0]
        for key in ("chunk_index", "start", "end", "snippet"):
            self.assertIn(key, hit)
        self.assertIn("kubernetes", hit["snippet"])
        self.assertEqual(hit["text"], self.DOCS[0]["text"], "text stays the whole document")


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
