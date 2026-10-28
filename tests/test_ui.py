"""The page's pure layer: no browser, no server, no API key.

    python3 -m pytest -q tests/test_ui.py

Every test here needs the optional ui extra (streamlit), so a stdlib-only checkout
skips the class instead of failing. Nothing in this file starts streamlit.
"""
import importlib.util
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import streamlit_app as ui  # noqa: E402

HAS_STREAMLIT = importlib.util.find_spec("streamlit") is not None

FIXTURE_DOC = {
    "id": "fx:01",
    "url": "https://example.com/jobs/01",
    "title": "Datavant \u2014 Junior Data Scientist",
    "text": "Datavant junior data scientist Barcelona linked health records python sql junior",
    "score": 8.4213,
}


@unittest.skipUnless(HAS_STREAMLIT, "streamlit absent: pip install -e '.[ui]'")
class TestLabels(unittest.TestCase):
    def test_toggle_only_switches_labels(self):
        en, es = ui.labels("en"), ui.labels("es")
        self.assertEqual(set(en), set(es))
        self.assertNotEqual(en["query"], es["query"])

    def test_unknown_language_falls_back_to_english(self):
        self.assertIs(ui.labels("fr"), ui.labels("en"))
        self.assertEqual(ui.labels(None)["results"], "Ranked postings")


@unittest.skipUnless(HAS_STREAMLIT, "streamlit absent: pip install -e '.[ui]'")
class TestResultFormatting(unittest.TestCase):
    def test_split_title_uses_the_separator(self):
        self.assertEqual(ui.split_title("Datavant \u2014 Junior Data Scientist"),
                         ("Datavant", "Junior Data Scientist"))
        self.assertEqual(ui.split_title("Data Scientist"), ("", "Data Scientist"))
        self.assertEqual(ui.split_title(""), ("", ""))

    def test_location_is_the_city_when_the_text_names_one(self):
        self.assertEqual(ui.infer_location(FIXTURE_DOC["text"]), "Barcelona")
        self.assertEqual(ui.infer_location("KPMG Madrid junior data engineer sql"), "Madrid")

    def test_location_falls_back_to_mode_and_country(self):
        self.assertEqual(ui.infer_location("Cinte remoto Colombia analista"), "Remote / Colombia")
        self.assertEqual(ui.infer_location("Quental remoto programador"), "Remote")
        self.assertIsNone(ui.infer_location("junior power bi etl sql"))

    def test_location_matches_whole_words_and_multi_word_cities(self):
        self.assertEqual(ui.infer_location("Cofidis Cornell\u00e0 data scientist"), "Cornell\u00e0")
        self.assertEqual(ui.infer_location("Las Palmas remote data analyst"), "Las Palmas / Remote")
        self.assertIsNone(ui.infer_location("notmadrid data scientist"))

    def test_snippet_truncates_on_a_word_boundary(self):
        self.assertEqual(ui.snippet("  short   text "), "short text")
        long = "palabra " * 60
        cut = ui.snippet(long, limit=50)
        self.assertLessEqual(len(cut), 53)
        self.assertTrue(cut.endswith("..."))
        self.assertNotIn("  ", cut)

    def test_score_chip_renders_three_decimals(self):
        self.assertEqual(ui.score_chip(8.4213), '<span class="chip">8.421</span>')
        self.assertEqual(ui.score_chip("0.5"), '<span class="chip">0.500</span>')
        self.assertEqual(ui.score_chip(None), "")

    def test_format_result_carries_every_field_the_card_draws(self):
        r = ui.format_result(FIXTURE_DOC, rank=2)
        self.assertEqual(r["rank"], 2)
        self.assertEqual(r["title"], "Junior Data Scientist")
        self.assertEqual(r["company"], "Datavant")
        self.assertEqual(r["location"], "Barcelona")
        self.assertEqual(r["url"], "https://example.com/jobs/01")
        self.assertEqual(r["score"], 8.4213)
        self.assertEqual(r["chip"], '<span class="chip">8.421</span>')
        self.assertIn("linked health records", r["snippet"])

    def test_format_result_drops_a_dangerous_url(self):
        doc = dict(FIXTURE_DOC, url="javascript:alert(1)")
        self.assertEqual(ui.format_result(doc)["url"], "")

    def test_result_html_escapes_the_text_and_links_the_title(self):
        doc = dict(FIXTURE_DOC, title="Acme \u2014 <script>x</script>")
        card = ui.result_html(ui.format_result(doc))
        self.assertIn('href="https://example.com/jobs/01"', card)
        self.assertIn("&lt;script&gt;", card)
        self.assertNotIn("<script>", card)
        self.assertIn('class="chip"', card)

    def test_format_results_numbers_the_hits_it_is_given(self):
        hits = ui.format_results([FIXTURE_DOC, dict(FIXTURE_DOC, id="fx:02")])
        self.assertEqual([r["rank"] for r in hits], [1, 2])
        self.assertEqual(ui.format_results([]), [])


@unittest.skipUnless(HAS_STREAMLIT, "streamlit absent: pip install -e '.[ui]'")
class TestCitations(unittest.TestCase):
    SOURCES = [{"n": 1, "title": "Datavant", "url": "https://example.com/jobs/01"},
               {"n": 2, "title": "KPMG", "url": "https://example.com/jobs/09"}]

    def test_known_markers_become_links(self):
        out = ui.render_answer("Two roles fit [1][2].", self.SOURCES)
        self.assertIn('<a class="cite" href="https://example.com/jobs/01"', out)
        self.assertIn(">[1]</a>", out)
        self.assertIn(">[2]</a>", out)

    def test_unknown_marker_stays_visible_and_is_reported(self):
        out = ui.render_answer("Fit [1], maybe [7].", self.SOURCES)
        self.assertIn('<span class="cite-missing"', out)
        self.assertIn("[7]</span>", out)
        self.assertEqual(ui.unresolved_citations("Fit [1], maybe [7].", self.SOURCES), [7])

    def test_answer_text_is_escaped_before_it_is_rendered(self):
        out = ui.render_answer("<b>bold</b> [1]", self.SOURCES)
        self.assertIn("&lt;b&gt;bold&lt;/b&gt;", out)
        self.assertIn('href="https://example.com/jobs/01"', out)

    def test_newlines_survive_and_empty_answer_renders_empty(self):
        self.assertIn("<br>", ui.render_answer("line one\nline two", self.SOURCES))
        self.assertEqual(ui.render_answer("", self.SOURCES), "")

    def test_duplicate_markers_are_all_linked(self):
        out = ui.render_answer("[1] and again [1]", self.SOURCES)
        self.assertEqual(out.count('class="cite"'), 2)


@unittest.skipUnless(HAS_STREAMLIT, "streamlit absent: pip install -e '.[ui]'")
class TestPageAgainstTheFixture(unittest.TestCase):
    def test_fixture_runs_with_no_key_and_renders_five_cards(self):
        docs, source = ui.load_documents(ROOT / "fixtures" / "corpus.json")
        self.assertEqual(source, "corpus.json")
        hits = ui.search_local("junior data scientist Barcelona", 5,
                               ui.BM25(docs))
        self.assertEqual(len(hits), 5)
        cards = ui.format_results(hits)
        self.assertTrue(all(c["title"] and c["chip"] and c["snippet"] for c in cards))
        self.assertIn("Datavant", {c["company"] for c in cards})
        self.assertTrue(all(c["url"].startswith("https://") for c in cards))

    def test_a_fresh_clone_falls_back_to_the_fixture(self):
        import os
        if os.getenv("CORPUS"):
            self.skipTest("CORPUS points somewhere else")
        docs, source = ui.load_documents()
        self.assertTrue(docs)
        self.assertIn(source, ("corpus.json", "tracker + warehouse"))


if __name__ == "__main__":
    unittest.main()
