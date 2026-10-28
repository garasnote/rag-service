"""The console entry point advertised in pyproject.toml, tested like a public API.

`pip install -e .` must produce a working `rag` command, so the CLI is part of the
package surface. Standard library only, no network.
"""
import contextlib
import io
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag import cli  # noqa: E402

FIXTURE = ROOT / "fixtures" / "corpus.json"


class CliTest(unittest.TestCase):
    def test_ranks_documents_from_a_fixture_corpus(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = cli.main(["junior data analyst", "--corpus", str(FIXTURE), "-k", "2"])
        self.assertEqual(code, 0)
        printed = out.getvalue().splitlines()
        self.assertTrue(printed[0].startswith("[1]"), printed[:1])
        self.assertIn("[2]", out.getvalue())
        # every hit is printed with its citation line indented underneath
        self.assertTrue(any(line.strip().startswith("http") for line in printed))

    def test_k_limits_the_number_of_hits(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            cli.main(["junior data analyst", "--corpus", str(FIXTURE), "-k", "1"])
        self.assertEqual(out.getvalue().count("[1]"), 1)
        self.assertNotIn("[2]", out.getvalue())


class EntryPointTest(unittest.TestCase):
    def test_console_script_points_at_rag_cli_main(self):
        from importlib import metadata

        eps = metadata.entry_points()
        # 3.9 returns a dict of groups, 3.10+ returns an EntryPoints object.
        if hasattr(eps, "select"):
            scripts = list(eps.select(group="console_scripts", name="rag"))
        else:  # importlib.metadata without the selectable API: a plain dict of groups
            scripts = [e for e in eps.get("console_scripts", []) if e.name == "rag"]
        if not scripts:
            self.skipTest("job-market-rag is not installed in this interpreter")
        self.assertEqual(scripts[0].value, "rag.cli:main")
        self.assertTrue(callable(cli.main))


if __name__ == "__main__":
    unittest.main()
