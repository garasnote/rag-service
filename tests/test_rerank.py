"""Reranker tests: ordering behaviour on tiny fixtures, no model required.

    python3 -m pytest -q

The cross-encoder half skips itself unless numpy + onnxruntime + tokenizers are
installed and RERANK_DIR points at a real export -- the core library still has
to pass with no optional dependency at all.
"""
import importlib.util
import os
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag.rerank import (  # noqa: E402
    CrossEncoderReranker,
    LexicalReranker,
    Reranker,
    RerankerUnavailable,
    get_reranker,
    rerank_hits,
)


def doc(did, text):
    return {"id": did, "title": did, "text": text}


class LexicalRerankerTest(unittest.TestCase):
    def setUp(self):
        self.rr = LexicalReranker()

    def test_constraint_conjunction_beats_term_repetition(self):
        """Two of two query terms outrank one term hammered three times."""
        docs = [doc("repeat", "junior junior junior"),
                doc("both", "junior analytics barcelona")]
        hits = self.rr.rerank("junior analytics barcelona", docs)
        self.assertEqual(hits[0]["id"], "both")

    def test_verbatim_phrase_beats_scattered_terms(self):
        """Same three terms, same counts: adjacency is the only signal left."""
        docs = [doc("scattered", "analista power sql bi remoto"),
                doc("adjacent", "analista power bi remoto")]
        hits = self.rr.rerank("power bi analista", docs)
        self.assertEqual(hits[0]["id"], "adjacent")

    def test_accents_do_not_matter(self):
        docs = [doc("plain", "analisis de datos en medellin"),
                doc("accented", "análisis de datos en Medellín"),
                doc("other", "cocinero en madrid")]
        hits = self.rr.rerank("análisis medellín", docs)
        self.assertEqual({h["id"] for h in hits[:2]}, {"plain", "accented"})
        self.assertEqual(hits[2]["id"], "other")

    def test_ties_keep_the_incoming_order(self):
        docs = [doc("first", "data engineering"), doc("second", "data engineering")]
        hits = self.rr.rerank("data engineering", docs)
        self.assertEqual([h["id"] for h in hits], ["first", "second"])

    def test_documents_keep_their_keys_and_gain_a_score(self):
        docs = [{"id": "a", "text": "junior data analyst", "url": "u", "score": 9.5}]
        hit = self.rr.rerank("junior data", docs)[0]
        self.assertEqual(hit["url"], "u")
        self.assertEqual(hit["score"], 9.5)          # stage-one score survives
        self.assertIsInstance(hit["rerank_score"], float)
        self.assertEqual(docs[0], {"id": "a", "text": "junior data analyst",
                                   "url": "u", "score": 9.5})  # input not mutated

    def test_top_truncates_after_ordering(self):
        docs = [doc("a", "cocinero madrid"), doc("b", "junior data analyst")]
        hits = self.rr.rerank("junior data analyst", docs, top=1)
        self.assertEqual([h["id"] for h in hits], ["b"])

    def test_empty_query_and_empty_pool_are_safe(self):
        docs = [doc("a", "texto"), doc("b", "otro texto")]
        self.assertEqual([h["id"] for h in self.rr.rerank("", docs)], ["a", "b"])
        self.assertEqual(self.rr.rerank("algo", []), [])

    def test_it_satisfies_the_protocol(self):
        self.assertIsInstance(self.rr, Reranker)
        self.assertEqual(self.rr.name, "lexical-idf")


class FactoryTest(unittest.TestCase):
    def test_none_is_the_default_and_means_off(self):
        self.assertIsNone(get_reranker())
        self.assertIsNone(get_reranker("none"))
        lex = get_reranker("lexical")
        self.assertIsInstance(lex, LexicalReranker)
        self.assertEqual(lex.coverage_weight, 0.5)

    def test_unknown_name_fails_loudly(self):
        with self.assertRaises(SystemExit):
            get_reranker("magic")

    def test_rerank_hits_passes_through_when_off(self):
        docs = [doc("a", "x"), doc("b", "y")]
        self.assertEqual(rerank_hits(None, "query", docs), docs)


class CrossEncoderAvailabilityTest(unittest.TestCase):
    def test_missing_directory_names_the_variable_and_the_model(self):
        with self.assertRaises(RerankerUnavailable) as ctx:
            CrossEncoderReranker(model_dir="/nonexistent/reranker")
        message = str(ctx.exception)
        self.assertIn("/nonexistent/reranker", message)
        self.assertIn("RERANK_DIR", message)
        self.assertIn("ms-marco-MiniLM-L6-v2", message)


def _deps_note():
    missing = [m for m in ("numpy", "onnxruntime", "tokenizers")
               if importlib.util.find_spec(m) is None]
    if missing:
        return "optional dependencies missing: " + ", ".join(missing)
    model_dir = pathlib.Path(os.getenv("RERANK_DIR") or "")
    if not (model_dir / "tokenizer.json").exists():
        return "RERANK_DIR is not set to an exported cross-encoder"
    return None


class CrossEncoderModelTest(unittest.TestCase):
    """Runs only on a machine that actually has the model."""

    @classmethod
    def setUpClass(cls):
        reason = _deps_note()
        if reason:
            raise unittest.SkipTest(reason)
        cls.rr = CrossEncoderReranker()

    def test_relevant_document_scores_above_an_irrelevant_one(self):
        docs = [doc("chef", "Head chef for a restaurant in Madrid, kitchen team lead"),
                doc("analyst", "Junior data analyst, Barcelona, python sql power bi")]
        hits = self.rr.rerank("junior data analyst in Barcelona", docs)
        self.assertEqual([h["id"] for h in hits], ["analyst", "chef"])
        self.assertGreater(hits[0]["rerank_score"], hits[1]["rerank_score"])

    def test_it_satisfies_the_protocol(self):
        self.assertIsInstance(self.rr, Reranker)


if __name__ == "__main__":
    unittest.main()
