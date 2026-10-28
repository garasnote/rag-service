# Stream UI: `streamlit_app.py`

Branch `wt/ui`. Files: `streamlit_app.py` (new), `tests/test_ui.py` (new),
`docs/ui.png` (screenshot), `pyproject.toml` (one new extra `ui`).

## What it is

One page: a query box, an EN/ES label toggle, the ranked postings (title, company,
location, URL, score chip, snippet) and, when a key is configured, a short answer whose
`[n]` markers are links to the postings they cite.

- Calls the library directly (`rag.core.BM25`, `rag.llm.answer`): no server, no key, no
  install beyond streamlit.
- If `API_URL` is set it calls `api.py` instead (`GET /search`, `POST /ask`) over stdlib
  `urllib`, so the page has no HTTP client dependency either.
- `CORPUS=/path/to/corpus.json` points it at another corpus. With neither that nor the
  private tracker/warehouse present it falls back to `fixtures/corpus.json` — the state a
  reviewer is in on a fresh clone. The corpus and its document count are printed under the
  title, so the page never pretends to be searching data it does not have.

Streamlit is optional: `streamlit_app.py` imports it in a `try/except ModuleNotFoundError`
and raises one clear error from `main()` when it is missing. Nothing else in the repo
imports it (`grep -rn streamlit --include=*.py .` hits only `streamlit_app.py` and
`tests/test_ui.py`).

## The parts that are testable are pure

Everything above `main()` is a pure function: no streamlit, no network, no file reads.
That is what `tests/test_ui.py` pins, so no browser and no server are needed:

| function | what it decides |
| --- | --- |
| `labels(lang)` | EN/ES strings; the toggle changes labels only, never the data |
| `split_title(title)` | `"Datavant — Junior Data Scientist"` -> company, posting title |
| `infer_location(text)` | city, else work mode, plus country; `None` when the text says nothing |
| `snippet(text, limit)` | one-line snippet, cut on a word boundary |
| `score_chip(score)` | the chip HTML, three decimals |
| `format_result(s)` | the dict the card draws, incl. `safe_url` dropping non-http URLs |
| `render_answer(text, sources)` | escapes the text, turns `[n]` into links, keeps unknown `[n]` visible and marked |
| `unresolved_citations` | the citation numbers the model invented |

Two things are **inferred, not structured**, and the page says so in a caption under the
heading ("Location inferred from the posting text"):

1. Company is split from the title on `" — "`, because that is how `load_corpus` builds it
   (`f"{company} — {role}"`).
2. The corpus has no city field; `load_corpus` folds city into `text`. `infer_location`
   matches a gazetteer of Spanish and Colombian cities on whole words, then falls back to
   remote/hybrid keywords and then to a country. On the fixture:

```
$ /usr/bin/python3 -c "...infer_location over fixtures/corpus.json..."
docs 16
location inferred 16 of 16
  fx:01 -> Barcelona        fx:02 -> Barcelona / Remote   fx:03 -> Madrid
  fx:04 -> Cornellà         fx:05 -> Barcelona            fx:06 -> Remote / Colombia
  fx:07 -> Villavicencio    fx:08 -> Madrid               fx:09 -> Madrid
  fx:10 -> Valencia         fx:11 -> Remote / Spain       fx:12 -> Madrid
  fx:13 -> Sevilla          fx:14 -> Madrid               fx:15 -> Remote
  fx:16 -> Alicante
```

13 of 16 name a city outright; the other three name a work mode. That is 100% coverage on
this corpus and would not be on a corpus that never names a place — in that case the card
shows company only, it does not invent a city.

## Tests

```
$ git log --oneline -1     # before my change
c67a66c Fix the HyDE test for the record prompt

$ /usr/bin/python3 -m pytest -q          # before, stdlib python, no streamlit
15 passed, 2 skipped in 0.17s

$ /usr/bin/python3 -m pytest -q          # after, stdlib python, streamlit absent
15 passed, 21 skipped in 0.86s

$ ~/.venvs/rag/bin/python -m pytest -q    # after, streamlit 1.66.1 present
34 passed, 2 skipped in 2.37s
```

19 new tests. They live in one file and every class carries
`@unittest.skipUnless(HAS_STREAMLIT, ...)`, so `python3 -m pytest -q` on a stdlib-only
checkout goes green with 21 skips (the 2 pre-existing skips are the private-corpus and
`MINILM_DIR` tests). CI installs only `pip install -e .`, so CI sees 15 passed / 21
skipped until someone installs the extra.

The last two tests are the one a reviewer cares about: they load
`fixtures/corpus.json`, run a real `BM25` search through `search_local`, and assert five
cards come out with a title, a chip, a snippet and an `https://` URL — no API key anywhere.

## Running it

```
uv pip install --python ~/.venvs/rag/bin/python streamlit
~/.venvs/rag/bin/python -m streamlit run streamlit_app.py
```

Or, from a fresh clone:

```
python3 -m venv .venv && . .venv/bin/activate
pip install -e ".[ui]"
streamlit run streamlit_app.py
```

The extra resolves (`uv pip install --dry-run -e ".[ui]"` -> "Resolved 36 packages").
Open http://localhost:8501, type `junior data scientist Barcelona`, the cards appear; the
Answer section says "No LLM key configured" until `LLM_API_KEY` is exported.

To exercise the answer path with citations and no real key, point it at any
OpenAI-compatible mock (`LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`). I did exactly that
with a 20-line mock server returning `"... fit [1] and [2] ..."`: the page rendered the
answer with two `<a class="cite">` links to `https://example.com/jobs/01` and `.../02`,
i.e. the citation markers resolve to the right postings. That mock is not committed.

### API mode

`API_URL=http://127.0.0.1:8100` makes the page call `api.py` instead of the library. Note
that `api.py` indexes `load_corpus()` with no fixture fallback, so it reports
`{"status":"ok","documents":0}` on a fresh clone until `TRACKER`/`WAREHOUSE` exist; that is
the API's behaviour, not the page's, and I did not change `api.py`. Verified against the
same fixture corpus served behind the API:

```
$ curl -s http://127.0.0.1:8100/health
{"status":"ok","documents":16}

$ /usr/bin/python3 -c "...ui.search_http('http://127.0.0.1:8100', ...)..."
http hits: 3
  1 Junior Data Scientist | Datavant | Barcelona | <span class="chip">3.767</span>
  2 Junior Data Scientist | Buynomics | Barcelona / Remote | <span class="chip">3.635</span>
  3 Data Engineering Analyst | Metyis | Barcelona | <span class="chip">2.393</span>
```

Error paths return a readable message instead of a traceback, and the page shows it in
`st.error`:

```
RuntimeError -> HTTP 502: {"detail":"no API key: set LLM_API_KEY (or OPENAI_API_KEY)"}
dead port -> <urlopen error [Errno 111] Connection refused>
```


Screenshot: `docs/ui.png` (1100x1500, EN, no key, `fixtures/corpus.json`, query
`junior data scientist Barcelona`). One screen holds the query, the slider, five cards and
the no-key note — it is what a reviewer sees on the first command.

## Numbers from commands I ran here

Index build and search cost, over the 16-document fixture:

```
$ /usr/bin/python3 -c "...load_documents, BM25, search_local..."
corpus=corpus.json docs=16
load=0.7ms index=4.3ms search=0.58ms hits=5
first hit: Datavant — Junior Data Scientist score=3.7672
```

So the page is idle-free at this size: the index is rebuilt in 4 ms on every rerun rather
than cached, which is why there is no `st.cache_resource` to go stale.

## What I tried that failed, or that I left out

1. **The language toggle dropped the query.** With unkeyed widgets, switching to ES reran
   the script and the text box came back empty. Fixed by giving the three widgets explicit
   `key`s (`lang`, `query`, `k`); re-checked in the browser: value survives the toggle.
   That is a real bug I would have shipped had I only run the unit tests.
2. **A hyphen is a title separator too** (`\\s+[—-–]\\s+`). Fine on the fixture; on a corpus
   with `"Data Analyst - O&M"` the dash will be read as company/title. I did not add a
   heuristic for it — a wrong split shows up as a wrong company, not as a crash, and
   inventing a rule from one example is worse than the plain separator.
3. **No dense/hybrid mode in the UI.** `--embedder minilm|multi` needs onnxruntime and a
   model directory; a reviewer without those would get an error instead of postings. BM25
   is what runs with nothing installed, so BM25 is what the page offers.
4. **No `st.cache_resource`.** Considered, measured 4 ms, dropped it: a cache is another
   thing to invalidate when `CORPUS` changes.
5. **Full-page screenshot via `Page.captureScreenshot(captureBeyondViewport=True)`**
   timed out against the browser daemon; the committed image is a plain viewport capture
   at 1100x1500, which is the same content.

## Caveats an orchestrator should know

- The page has never been run behind a real LLM key (none available here). The local
  answer path is `rag.llm.answer`, which the existing tests already cover with a mock
  server; I verified the page against the same kind of mock.
- `infer_location` is a gazetteer, so a posting whose city is not in the list loses its
  city silently, company and snippet stay correct. Extending the list is one line per city.
- The "legible in ten seconds" claim is not measured. The evidence is the screenshot and
  the field list, nothing more.
