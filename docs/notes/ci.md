# CI stream notes

Branch `wt/ci`. Worktree `~/Documents/JobApp/portfolio/.wt-rag/ci`.

Headline: the repo claimed `requires-python = ">=3.9"` and no CI job had ever run on
3.9. When I added the 3.9 job it crashed on import with
`TypeError: unsupported operand type(s) for |: 'type' and 'NoneType'`. One line in
`rag/core.py` fixed it; CI is now green on 3.9 and 3.12 with **18 passed, 2 skipped**
(was 15 passed, 2 skipped).

## 1. What CI does now

`.github/workflows/tests.yml`

| job | what it proves |
| --- | --- |
| `pytest - Python 3.9` / `pytest - Python 3.12` | matrix, `fail-fast: true`, `setup-python` pip cache keyed on `pyproject.toml`. Each matrix leg: `pip install -e ".[dev]"` from the checkout, `python -m pytest -q`, the fixture retrieval eval (`eval/run_eval.py --corpus fixtures/corpus.json --questions fixtures/questions.json`), the fixture intent eval (`eval/run_intent.py --corpus fixtures/corpus.json --questions fixtures/questions_intent.json`). No secrets, no network beyond pip. |
| `installs from a clean checkout` | `pip wheel --no-deps` builds `job_market_rag-0.1.0-py3-none-any.whl`, that wheel is installed into a throwaway venv, the package is imported **from outside the source tree** (so a module missing from `packages = ["rag"]` fails here), and the `rag` console script answers a query against the fixture corpus. |

`.github/workflows/lint.yml`

| job | what it proves |
| --- | --- |
| `compileall + ruff` | `python -m compileall -q rag eval api.py` byte-compiles the library, the eval harness and `api.py` without importing them (the eval scripts are not imported by any test, so a syntax error there used to be invisible), then `python -m ruff check rag eval api.py`. |

Both workflows: `permissions: contents: read`, no secrets, `concurrency` cancels a
superseded run on the same ref, trigger unchanged (`push` to `main`, plus every
`pull_request`).

Why ruff rides in `.[dev]`: `dev` is already the extras group CI installs and the
core library stays stdlib only — `ruff` is a build-time tool, never a runtime
import. Verified (`uv pip list` in the 3.9 venv after `pip install -e ".[dev]"`):
`job-market-rag 0.1.0 (editable)`, `pytest 8.4.2`, `ruff 0.17.3`, `pluggy 1.6.0`,
`iniconfig 2.1.0`, `packaging 26.3`, `pygments 2.21.0`, `tomli 2.5.0`,
`exceptiongroup 1.3.1`, `typing-extensions 4.16.0` — no numpy, no onnxruntime.

## 2. Packaging metadata (`pyproject.toml`)

- `license = "WTFPL"` + `license-files = ["LICENSE"]` (PEP 639). `build-system`
  floor moved to `setuptools>=77`, which is the version that supports the SPDX
  field; the old `license = { text = "WTFPL" }` form is deprecated. The built wheel
  now carries `License-Expression: WTFPL` and `License-File: LICENSE`.
- `keywords` (rag, retrieval, bm25, reciprocal-rank-fusion, hyde, evaluation,
  job-search) and 13 trove `classifiers` (status, audiences, Python 3.9-3.13, the
  two topics).
- `project.urls`: `Homepage`, `Repository`, `Issues`, all pointing at the real
  remote `git@github.com:garasnote/rag-service.git`. Nothing invented.
- `[project.scripts] rag = "rag.cli:main"`. `rag/cli.py` already had a `main()`
  and no way to reach it; now `pip install -e .` produces a `rag` command.
- `requires-python = ">=3.9"` kept, and it is now *true* and *tested*: the matrix
  runs the floor and a modern interpreter. The 3.10/3.11/3.13 classifiers are a
  claim about compatibility, not about CI coverage — only 3.9 and 3.12 are executed.
- Every non-stdlib dependency is still inside an extra (`embeddings`, `api`,
  `dev`); `dependencies = []` is untouched.
- `[tool.ruff]`: `target-version = "py39"` (ruff parses the sources as 3.9),
  `line-length = 100`, `src = ["."]`, and
  `[tool.ruff.lint] select = ["E4", "E7", "E9", "F"]`.

Why that narrow rule set: with ruff 0.17.3's own defaults the tree has 33 findings
(18 `RUF100` unused-`noqa`, 5 `EXE001`, 4 `I001`, 1 each of `BLE001`, `SIM115`,
`SIM905`, `S110`, `F401`, `RUF007`), and with `E,F,W,I,B,UP` it has 58. Fixing
those means reformatting files other streams are editing in parallel — so the lint
job fails only on real defects: syntax errors, undefined names, unused imports,
broken comparisons. Everything the wider set found is listed in section 6.

## 3. Repo hygiene

Audit result: the tree was already free of ship-blockers. No `*.egg-info/`,
no `out/`, no scratch files, no oversized fixtures (largest tracked file:
`eval/hyde_intent_samples.json`, 8 KB; `fixtures/` is 12 KB total). Nothing was
deleted.

What I did change:

- `.gitignore` extended from 8 to ~20 rules: `*.py[cod]`, `.pytest_cache/`,
  `.ruff_cache/`, `.mypy_cache/`, `.venv/`, `venv/`, `wheelhouse/`, `out/`,
  `*.log`. `.pytest_cache/` and `.ruff_cache/` were only accidentally ignored
  before (`git check-ignore -v .pytest_cache` found the repo `.gitignore` did not
  match it, while `git status --ignored` still hid the directory). The dense index
  cache (`VectorIndex` writes to `<repo>/.cache/` by default) was already covered.
- `eval/run_hyde10.py`: removed an unused `matches` import (`ruff F401`).
- `tests/test_cli.py` (new, 3 tests): the CLI is now a declared entry point, so it
  is tested like a public API — ranking against `fixtures/corpus.json`, `-k`
  limiting the hit count, and the installed console-script metadata pointing at
  `rag.cli:main` (skips itself when the package is not installed).
- `rag/core.py`: one line, `from __future__ import annotations` — see section 4.

Not deleted, on purpose: `eval/questions*.json` (4 sets) and `fixtures/` per the
brief; the two extra question sets (`questions_paraphrase.json`,
`questions_bilingual.json`) are not referenced by any script but they are eval
data reached through `--questions`, so they stay.

## 4. The bug the 3.9 leg found

```
$ ~/.hermes/cache/scratch/ci/v39/bin/python eval/run_eval.py --corpus fixtures/corpus.json --questions fixtures/questions.json
Traceback (most recent call last):
  File ".../eval/run_eval.py", line 21, in <module>
    from rag.core import BM25, load_corpus, strip_accents  # noqa: E402
  File ".../rag/core.py", line 33, in <module>
    corpus: pathlib.Path | None = None) -> list:
TypeError: unsupported operand type(s) for |: 'type' and 'NoneType'
```

`rag/core.py` used a PEP 604 union (`pathlib.Path | None`) in a signature and has
no `from __future__ import annotations`, so the union is evaluated at import time.
On 3.9 that is a hard error: the whole library was unimportable, which means
`requires-python = ">=3.9"` was false and the installable package did not work on
two of the classifiers it advertised. `rag/cli.py` and the eval harnesses already
had the future import; `rag/core.py` was the only module missing it
(`grep`-ed: it is the only `X | None` in the tree). Fixed with the import, not by
raising the floor, because the floor is what the project documents.

## 5. Numbers, from commands run in this worktree

Test count, before (checked out clean, no changes):

```
$ /usr/bin/python3 -m pytest -q
15 passed, 2 skipped in 0.65s
$ /usr/bin/python3 -m unittest discover -s tests
Ran 17 tests in 0.522s
OK (skipped=2)
```

Test count, after (`pip install -e ".[dev]"` in the venv, which is what CI does):

```
$ $S/v39/bin/python -m pytest -q -rs
SKIPPED [1] tests/test_rag.py:35: private tracker/warehouse not present
SKIPPED [1] tests/test_rag.py:173: MINILM_DIR not set
18 passed, 2 skipped in 0.67s          # Python 3.9.25

$ $S/v312/bin/python -m pytest -q -rs
SKIPPED [1] tests/test_rag.py:35: private tracker/warehouse not present
SKIPPED [1] tests/test_rag.py:173: MINILM_DIR not set
18 passed, 2 skipped in 0.65s          # Python 3.12.14
```

On a bare interpreter without the package installed the new entry-point test skips
itself instead of failing (`/usr/bin/python3 -m pytest -q` -> `17 passed,
3 skipped`, the third skip being `tests/test_cli.py:41: job-market-rag is not
installed in this interpreter`).

(17 collected -> 20 collected, +3 from `tests/test_cli.py`.)

Fixture retrieval eval, identical on 3.9.25 and 3.12.14 (16 documents, 8 questions,
`hashing-512`):

```
config    hit@1  hit@3    MRR
bm25        88%   100%  0.917
dense       88%   100%  0.938
hybrid      88%   100%  0.938
hybrid3:1   88%   100%  0.917
hyde       100%   100%  1.000      <- oracle stand-in, a ceiling, not a measurement
```

Fixture intent eval, identical on both (16 documents, 6 questions):

```
mode       sat@1   sat@3   sat@5
bm25         83%     83%     83%
dense        67%     83%     83%
hybrid       83%     83%     83%
```

Lint, before the F401 fix: `ruff check rag eval api.py` with the shipped config
reports `Found 1 error` (`eval/run_hyde10.py:34 F401`). After: `All checks passed!`

Package job, run locally as the workflow runs it:

```
$ python -m pip wheel --no-deps --wheel-dir $S/wheelhouse .
Created wheel for job-market-rag: filename=job_market_rag-0.1.0-py3-none-any.whl size=17531 sha256=8a499fb44399b3...
$ $S/clean/bin/python -m pip install --no-index $S/wheelhouse/*.whl
Successfully installed job-market-rag-0.1.0
$ cd $S && $S/clean/bin/python -c "import rag, rag.core, ... , rag.cli; print(rag.__file__)"
~/.hermes/cache/scratch/ci/clean/lib/python3.12/site-packages/rag/__init__.py
$ $S/clean/bin/rag "junior data analyst" --corpus fixtures/corpus.json -k 2
[1] Metyis — Data Engineering Analyst
    https://example.com/jobs/05
[2] Keepler Data Tech — Junior Data Analyst (Power BI)
    https://example.com/jobs/03
```

Wheel contents: `rag/` (6 modules) + `dist-info` only, 13 files, 37 KB.

## 6. Found and deliberately not touched

1. `tests/test_rag.py` has `if __name__ == "__main__": unittest.main()` at line 84,
   *before* `EmbedTest`, `FusionTest`, `HydeTest` and `EmbedderFactoryTest` are
   defined (line 88 onwards). `pytest` and `unittest discover` collect all 20 tests,
   so CI is unaffected, but `python3 tests/test_rag.py` silently runs 8 of 20:
   ```
   $ /usr/bin/python3 tests/test_rag.py
   ...s....
   ----------------------------------------------------------------------
   Ran 8 tests in 0.588s

   OK (skipped=1)
   ```
   Fix is to move the block to the end of the file. Left alone because `tests/` is
   shared with the other streams and the diff would not be mine to own.
2. `api.py` is not in the wheel (`packages = ["rag"]` only). It is run from the
   repo root with `uvicorn api:app`, so shipping it would change how the service is
   started. Documented, not changed.
3. Wider ruff findings, none of them bugs, all left as-is: 18 `RUF100` (the
   `# noqa: E402` comments in the eval harnesses are "unused" once `E402` is not
   selected), 5 `EXE001` (shebanged `eval/*.py` without the executable bit),
   4 `I001` (import order, including `api.py`), 1 `S110`/`BLE001`
   (`rag/embed.py:178` swallows a corrupt cache file — intentional), 1 `SIM115`
   (bare `open()` in `eval/run_eval.py:64`), 1 `SIM905`, 1 `RUF007`.
4. `rag/embed.py` writes its vector cache to `<repo>/.cache/` — now gitignored, but
   it is a side effect in the working tree. Changing the default would be a
   behaviour change owned by the retrieval stream.
5. `README.md`: untouched, per the brief.

## 7. Reproduce

```sh
cd ~/Documents/JobApp/portfolio/.wt-rag/ci
S=~/.hermes/cache/scratch/ci

# interpreter matrix, same commands as the workflow
uv venv --python 3.9  $S/v39  && uv pip install --python $S/v39/bin/python  -e ".[dev]"
uv venv --python 3.12 $S/v312 && uv pip install --python $S/v312/bin/python -e ".[dev]"

for PY in $S/v39/bin/python $S/v312/bin/python; do
  $PY -V
  $PY -m pytest -q
  $PY eval/run_eval.py   --corpus fixtures/corpus.json --questions fixtures/questions.json
  $PY eval/run_intent.py --corpus fixtures/corpus.json --questions fixtures/questions_intent.json
  $PY -m compileall -q rag eval api.py && $PY -m ruff check rag eval api.py
done

# package job
uv venv --python 3.12 --seed $S/build
$S/build/bin/python -m pip wheel --no-deps --wheel-dir $S/wheelhouse .
uv venv --python 3.12 --seed $S/clean
$S/clean/bin/python -m pip install --no-index $S/wheelhouse/*.whl
cd $S && $S/clean/bin/python -c "import rag, rag.cli; print(rag.__file__)"
cd ~/Documents/JobApp/portfolio/.wt-rag/ci && $S/clean/bin/rag "junior data analyst" --corpus fixtures/corpus.json -k 2
```

Note: `pip install -e .` leaves `job_market_rag.egg-info/` in the tree and
`pip wheel` leaves `build/`; both are gitignored and were removed before the
commit.

## 8. Risks in the real GitHub run

- Everything above was executed locally on 3.9.25 / 3.12.14 / 3.14.8; the workflows
  themselves were validated by parsing them with PyYAML and by running each step
  command by hand. No GitHub runner was available here, so action versions
  (`checkout@v4`, `setup-python@v5`) and the `cache: pip` behaviour are standard,
  untested-by-me choices.
- The 3.9 leg depends on `setuptools>=77` being installable on 3.9; it installs
  cleanly there today (verified by the editable install in the 3.9 venv).
- Both workflows only trigger on `push` to `main` and on pull requests, as before.
  A push to `wt/*` will not run them until this lands on `main` or a PR is opened.
