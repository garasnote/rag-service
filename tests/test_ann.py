"""rag/ann.py: the optional hnswlib backend.

The suite stays green with or without the dependency: the corpus tests skip when
hnswlib cannot be imported, and the "missing dependency" test always runs (it
fakes the absence with sys.modules).

    python3 -m pytest -q                                   # skips the ANN body
    /path/to/venv/bin/python -m pytest -q tests/test_ann.py  # runs everything
"""
import pathlib
import sys
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag.embed import HashingEmbedder, VectorIndex  # noqa: E402

DOCS = [
    {"id": "a", "text": "analista de datos power bi etl excel sql remoto junior"},
    {"id": "b", "text": "cocinero restaurante jornada completa madrid hosteleria"},
    {"id": "c", "text": "ingeniero de machine learning pytorch vision computacional"},
    {"id": "d", "text": "data scientist junior barcelona python pandas salud"},
    {"id": "e", "text": "devops kubernetes terraform aws remoto sre"},
    {"id": "f", "text": "analista power bi dax sql azure consultoria madrid"},
    {"id": "g", "text": "investigador nieve camaras web segmentation deep learning"},
    {"id": "h", "text": "becario marketing contenidos redes sociales barcelona"},
]
QUERIES = [
    "power bi sql analista",
    "junior data scientist barcelona",
    "pytorch vision computacional",
    "kubernetes remoto",
    "camaras web nieve",
    "cocinero hosteleria madrid",
]


def hnswlib_available() -> bool:
    try:
        import hnswlib  # noqa: F401
    except ImportError:
        return False
    return True


class MissingDependencyTest(unittest.TestCase):
    """Runs everywhere: the failure has to be an actionable RuntimeError, not ImportError."""

    def test_missing_hnswlib_raises_runtimeerror(self):
        from rag.ann import HNSWIndex
        with mock.patch.dict(sys.modules, {"hnswlib": None}):
            with self.assertRaises(RuntimeError) as ctx:
                HNSWIndex(DOCS, HashingEmbedder(128))
        self.assertIn("hnswlib", str(ctx.exception))

    def test_bad_space_is_rejected(self):
        from rag.ann import HNSWIndex
        with self.assertRaises(ValueError):
            HNSWIndex(DOCS, HashingEmbedder(128), space="dot-product-ish")


@unittest.skipUnless(hnswlib_available(), "hnswlib not installed (pip install 'job-market-rag[ann]')")
class HNSWIndexTest(unittest.TestCase):
    def build(self, **kwargs):
        from rag.ann import HNSWIndex
        kwargs.setdefault("ef", 200)
        return HNSWIndex(DOCS, HashingEmbedder(128), **kwargs)

    def test_same_surface_as_the_exact_index(self):
        index = self.build()
        self.assertEqual(len(index.vectors), len(DOCS))
        self.assertEqual(len(index.vectors[0]), 128)
        hits = index.search_text("power bi sql analista", k=3)
        self.assertTrue(hits)
        self.assertEqual(hits[0]["id"], "a")
        self.assertIn("score", hits[0])
        self.assertTrue(all(h["id"] in {d["id"] for d in DOCS} for h in hits))
        self.assertEqual(len(index.search_vector(index.vectors[0], k=2)), 2)

    def test_recall_is_one_at_high_ef_on_a_tiny_corpus(self):
        exact = VectorIndex(DOCS, HashingEmbedder(128))
        index = self.build(ef=200)
        for query in QUERIES:
            reference = {d["id"] for d in exact.search_text(query, k=5)}
            found = {d["id"] for d in index.search_text(query, k=5)}
            self.assertEqual(reference, found, f"ann disagreed with exact on: {query}")

    def test_scores_are_cosine_like_the_exact_index(self):
        exact = VectorIndex(DOCS, HashingEmbedder(128))
        index = self.build(ef=200)
        query = "power bi sql analista"
        best_exact = exact.search_text(query, k=1)[0]
        best_ann = index.search_text(query, k=1)[0]
        self.assertEqual(best_exact["id"], best_ann["id"])
        self.assertAlmostEqual(best_exact["score"], best_ann["score"], places=3)

    def test_ef_is_settable_and_build_time_is_recorded(self):
        index = self.build(ef=10)
        self.assertEqual(index.ef, 10)
        index.ef = 64
        self.assertEqual(index.ef, 64)
        self.assertGreater(index.build_s, 0.0)
        # fewer than k hits is correct here: the cosine > 0 filter is kept, like the exact index
        exact = VectorIndex(DOCS, HashingEmbedder(128)).search_text("kubernetes remoto", k=3)
        self.assertEqual([h["id"] for h in index.search_text("kubernetes remoto", k=3)],
                         [h["id"] for h in exact])

    def test_empty_corpus_is_refused(self):
        from rag.ann import HNSWIndex
        with self.assertRaises(ValueError):
            HNSWIndex([], HashingEmbedder(128))


if __name__ == "__main__":
    unittest.main()
