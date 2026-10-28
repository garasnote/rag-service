# Spanish morphology: a real stemmer, and the guard that had to change

Branch `wt/stemmer`. The README claimed Spanish morphology was unhandled and that a
naive suffix stripper made retrieval worse (0.917 to 0.667 on the fixture) because it
shredded company names. This stream implements the real stemmer and measures it.

**Verdict: stemming earns its place, but not with the guard it was specified with.**
With a real Spanish Snowball stemmer and a guard that refuses digits, punctuation,
all-caps and a short employer allowlist, intent satisfaction at k=1 goes from 40% to
50% (bm25) and 40% to 55% (hybrid) on the 2,577-document corpus, and the fixture goes
from 88%/0.917 to 100%/1.000. The rule "refuse anything capitalised" is the part that
fails: it cost 15 points of sat@1 and turned the whole feature into a regression. That
rule is still in the code as the `strict` policy and is still measurable, one constant
away.

## What changed

| file | change |
| --- | --- |
| `rag/stem_es.py` (new) | `stem_es(word)`: the published Spanish Snowball stemmer, ported from `algorithms/spanish.sbl` in pure Python. `stemmable(token, sentence_initial=False, policy="")`: the entity guard, two policies. `GUARD_POLICY = "acronyms"`. No dependencies, no regex chains. |
| `rag/core.py` | `tokenize(text, stem=False)` and `BM25(docs, stem=...)`. With `stem=True` the tokenizer works on the *surface* text so the guard can see capitals, then stems only the tokens the guard allows. |
| `rag/embed.py` | `HashingEmbedder(dim, stem=False)` and `get_embedder(name, stem=False)`. The stemmed embedder reports the name `hashing-512+stem-es`, so the on-disk vector cache cannot hand stemmed vectors to an unstemmed run. |
| `eval/run_intent.py`, `eval/run_eval.py` | `--stem none|es` (default `none`), applied to documents and queries alike. |
| `tests/test_rag.py` | `StemmerTest`: 7 tests (known pairs, accents, guard under both policies, policy difference, tokenizer, BM25 through a plural, parity with the reference implementation). |

`README.md` untouched.

## The stemmer is checked, not assumed

A stemmer that is "roughly right" is worse than no stemmer, so `stem_es` is compared
word for word against the reference `snowballstemmer` package (Snowball 3.1.1):

```
$ ~/.venvs/rag/bin/python ~/.hermes/cache/scratch/check_stem2.py
words checked: 33593
mismatches:    0
```

plus 136,655 random strings over the Spanish alphabet and 92 hand-picked real words,
also 0 mismatches. That check found two real bugs in my first draft, both the same
mistake: Snowball's verb and y-verb steps use `setlimit tomark pV`, so a suffix that
would start before RV is not a candidate and a shorter suffix inside RV wins. I had
matched the longest suffix first and then rejected it, so `miremos` stayed `miremos`
instead of `mir` and `daras` stayed `daras` instead of `dar`. 52 of 33,593 synthetic
words were wrong before the fix, 0 after. The same comparison is in the test suite and
skips itself where `snowballstemmer` is not installed (it is not in CI).

Stems, for the record: `analistas`/`analista` -> `anal`, `corriendo`/`correr` ->
`corr`, `naciones`/`nacion` -> `nacion`, `aplicacion` -> `aplic`.

## The guard: two policies, and why the specified one loses

`stemmable(token)` refuses a token (it is indexed verbatim) when

- it is not purely alphabetic: digits, `+`, `#`, `-`, `.` inside;
- it is all-caps and longer than one letter: `NTT`, `BBVA`, `KPMG`, `SQL`;
- it is in the employer allowlist: `datavant`, `keepler`, `buynomics`, `cofidis`,
  `quental`, `metyis`, `cinte`, plus the obvious consultancies.

The `strict` policy adds "refuse anything capitalised, except a capital that opens a
sentence". Measured with the harnesses' own scoring functions, bm25 only:

```
fixture no stemming            entity hit@1= 88% hit@3=100% MRR=0.917 | intent sat@1= 83% sat@3= 83% sat@5= 83%
fixture guard=strict           entity hit@1= 88% hit@3=100% MRR=0.917 | intent sat@1= 83% sat@3= 83% sat@5= 83%
fixture guard=acronyms         entity hit@1=100% hit@3=100% MRR=1.000 | intent sat@1=100% sat@3=100% sat@5=100%

real    no stemming            entity hit@1= 53% hit@3= 73% MRR=0.622 | intent sat@1= 40% sat@3= 65% sat@5= 65%
real    guard=strict           entity hit@1= 53% hit@3= 73% MRR=0.633 | intent sat@1= 25% sat@3= 60% sat@5= 65%
real    guard=acronyms         entity hit@1= 60% hit@3= 73% MRR=0.667 | intent sat@1= 50% sat@3= 75% sat@5= 75%
```

The case rule is not just useless here, it is actively harmful, and the reason is
mechanical: **matching is symmetric, so mangling a name costs nothing as long as both
sides mangle it the same way. What costs is treating the same word differently in the
query and in the document.** Job records are title-cased lists; queries are typed in
lower case:

```
tracker:457  Mediaset España Analista de Datos Junior Madrid Spain auto: linkedin 2026-10-28 Target
  tokenize(stem=False)                 ['mediaset', 'espana', 'analista', 'datos', 'junior', 'madrid', 'spain', 'auto', ...]
  tokenize(stem=True, guard=acronyms)  ['mediaset', 'espan', 'anal', 'dat', 'junior', 'madr', 'spain', 'aut', ...]
  tokenize(stem=True, guard=strict)    ['mediaset', 'espana', 'analista', 'datos', 'junior', 'madrid', 'spain', 'aut', ...]
```

Under the strict guard the query `analista de datos` stems to `anal dat` while this
document keeps `analista datos`, so the two never meet. 21,513 of 36,650 tokens in the
corpus were blocked under `strict`, dominated by title-cased city and role words
(`madrid` 730, `spain` 593, `bogota` 564, `barcelona` 445, `data` 299, `engineer` 290).
Under `acronyms` only 7,232 (20%) are blocked, and those are genuinely strings an index
wants verbatim.

Is the guard worth having at all? It is not cosmetic: 247 of the 571 employer names in
the corpus change under the raw stemmer (`Accenture` -> `accentur`, `Alcampo` ->
`alcamp`, `Arelance` -> `arelanc`, `Bureau Veritas` -> `bureau verit`), and `Datavant`
is untouched only by luck. Dropping the guard entirely scored exactly the same as
`acronyms` on both corpora, so the cheap rules (all-caps, digits, allowlist) cost
nothing and keep the obvious cases safe. `acronyms` is the default because of that, and
because `stemmable("Datavant")`, `stemmable("NTT")` and `stemmable("BBVA")` are all
False either way.

## Numbers

`hashing-512`, the default embedder. Entity = `run_eval.py` (hit@1, hit@3, MRR over
questions with one expected document). Intent = `run_intent.py` (share of questions
where a top-k document satisfies every constraint group). `es` below means the shipped
`acronyms` guard.

### Fixture: 16 documents, 8 entity questions, 6 intent questions

```
entity   hit@1  hit@3    MRR       intent   sat@1  sat@3  sat@5
none bm25   88%   100%  0.917       none bm25   83%    83%    83%
es   bm25  100%   100%  1.000       es   bm25  100%   100%   100%
none dense  88%   100%  0.938       none dense  67%    83%    83%
es   dense  88%   100%  0.938       es   dense 100%   100%   100%
none hybrid 88%   100%  0.938       none hybrid 83%    83%    83%
es   hybrid 100%  100%  1.000       es   hybrid 100%  100%   100%
```

### The real corpus: 2,577 documents, 15 entity questions, 20 intent questions

```
entity   hit@1  hit@3    MRR       intent   sat@1  sat@3  sat@5
none bm25   53%    73%  0.622       none bm25   40%    65%    65%
es   bm25   60%    73%  0.667       es   bm25   50%    75%    75%
none dense  40%    53%  0.467       none dense  30%    30%    35%
es   dense  47%    60%  0.511       es   dense  35%    50%    65%
none hybrid 60%    73%  0.656       none hybrid 40%    50%    60%
es   hybrid 60%    67%  0.639       es   hybrid 55%    65%    75%
```

Reading it honestly:

- **Intent, the benchmark the brief is about: stemming wins both combinations.** bm25
  +10 points at every k, hybrid +15 points at every k.
- **Entity bm25 and dense improve** (+7 hit@1 and +0.045 MRR for bm25; +7 hit@1 and
  +0.044 MRR for dense). Entity hybrid is the one regression: hit@1 flat at 60%, hit@3
  73% -> 67%, MRR 0.656 -> 0.639. That is one question moving from rank 2 to rank 4 in
  a 15-question set, but it is a regression and it is the reason the flag default is
  not flipped for you (see the handoff note).
- **The Datavant question still fails in every configuration.** The corpus has exactly
  one Datavant record; many other Barcelona junior rows outrank it. Stemming cannot fix
  a ranking problem, and the guard is not the cause.
- The fixture is 16 documents and 6 intent questions. 100% there is a ceiling effect,
  not evidence. The 2,577-document numbers are the ones to trust.

Why the corpus changes at all: 36,650 token occurrences, 29,418 (80%) reach the
stemmer, 15,283 (42%) change, 334 stems merge two or more distinct words
(`aplicaciones`/`aplicar`/`aplicado` -> `aplic`, `programador`/`programacion`/`programa`
-> `program`, `estadistica`/`estadistico` -> `estadist`), and the distinct term count
falls from 4,475 to 4,093. The index pays for it: `BM25` build over 2,577 documents
goes from 0.14s to 1.45s.

## What I tried that failed

1. The guard as specified. Protecting every capitalised token turned a win into a loss
   (sat@1 40% -> 25%) for the reason above. This is the interesting failure: the rule
   was written to stop a stemmer shredding names, but names are shredded identically on
   both sides of the match, while title-cased prose is not.
2. Assuming the README's 0.917 -> 0.667 was the fate of any stemming at all. It is not:
   with the allowlist and acronym rules the fixture is bit-identical to no stemming,
   and with the case rule gone it is better.
3. Assuming "correct port" first: the version before the `setlimit` fix was wrong on 52
   of 33,593 words and I would not have noticed without the reference implementation.

## Handoff: the flag default

`--stem` still defaults to `none` on both eval scripts. Stemming won, so it should
probably be `es`, but flipping it now would silently invalidate the fixture table at the
top of the README, which this stream may not edit. When the prose is merged, change the
two argparse lines

```
eval/run_eval.py:   ap.add_argument("--stem", default="none", ...)   ->  default="es"
eval/run_intent.py: ap.add_argument("--stem", default="none", ...)   ->  default="es"
```

and quote the fixture table above (bm25 88%/0.917 -> 100%/1.000, intent 83% -> 100%)
instead of the current one. The library defaults (`tokenize`, `BM25`, `HashingEmbedder`)
stay `stem=False` either way, so nothing else moves.

## Commands

The evals default to the private tracker and warehouse, whose paths are one directory
off from inside a linked worktree, so export the corpus once and point `--corpus` at it:

```
cd ~/Documents/JobApp/portfolio/.wt-rag/stemmer
/usr/bin/python3 -c "
import json, pathlib, sys; sys.path.insert(0, '.')
from rag.core import load_corpus
docs = load_corpus(tracker=pathlib.Path('~/Documents/JobApp/job_tracker/data.json'),
                   warehouse=pathlib.Path('~/Documents/JobApp/portfolio/hiresignal/warehouse.db'))
pathlib.Path('/tmp/corpus_private.json').write_text(json.dumps(docs, ensure_ascii=False))"
```

The four combinations the brief asks for, fixture first:

```
/usr/bin/python3 eval/run_intent.py --corpus fixtures/corpus.json --questions fixtures/questions_intent.json --stem none
/usr/bin/python3 eval/run_intent.py --corpus fixtures/corpus.json --questions fixtures/questions_intent.json --stem es
/usr/bin/python3 eval/run_intent.py --corpus /tmp/corpus_private.json --stem none
/usr/bin/python3 eval/run_intent.py --corpus /tmp/corpus_private.json --stem es
```

and the entity benchmark, which also reports dense and the hyde ceiling:

```
/usr/bin/python3 eval/run_eval.py --corpus fixtures/corpus.json --questions fixtures/questions.json --stem none
/usr/bin/python3 eval/run_eval.py --corpus fixtures/corpus.json --questions fixtures/questions.json --stem es
/usr/bin/python3 eval/run_eval.py --corpus /tmp/corpus_private.json --stem none
/usr/bin/python3 eval/run_eval.py --corpus /tmp/corpus_private.json --stem es
```

The guard ablation is a one-word change: `GUARD_POLICY = "strict"` in `rag/stem_es.py`,
or `import rag.stem_es as se; se.GUARD_POLICY = "strict"` before building a `BM25`.
Clear `.cache/` before re-running a dense or hybrid eval, or reuse vectors from the
other policy.

Tests:

```
/usr/bin/python3 -m pytest -q                     # 21 passed, 3 skipped (no snowballstemmer)
~/.venvs/rag/bin/python -m pytest -q    # 22 passed, 2 skipped (parity test runs)
```

Before this branch: `python3 -m pytest -q` was 15 passed, 2 skipped. After: 21 passed,
3 skipped with the same interpreter, so 6 new tests plus the parity check that only runs
where `snowballstemmer` is installed (`uv pip install --python
~/.venvs/rag/bin/python snowballstemmer` -- a dev aid, not a project
dependency).

## The sentence that should replace the README limitation

Replace

> - **Spanish morphology is unhandled.** "analistas" does not match "analista".
>   A naive suffix stripper made things worse (0.917 → 0.667 on the fixture)
>   because it also shreds company names.

with

> - **Spanish morphology is handled, and the guard matters more than the stemmer.**
>   `--stem es` runs a real Spanish Snowball stemmer (`rag/stem_es.py`, checked word
>   for word against the reference implementation) behind a guard that keeps acronyms,
>   digits and an employer allowlist out of it. On the 2,577-document corpus intent
>   satisfaction at k=1 goes from 40% to 50% (bm25) and 40% to 55% (hybrid), and entity
>   MRR for bm25 from 0.622 to 0.667. The obvious guard — refuse anything capitalised —
>   is worse than no stemming at all (sat@1 25%), because postings are title-cased lists:
>   the query `analista` stems to `anal` while the record's `Analista` does not, and the
>   two never meet. It is kept as `GUARD_POLICY = "strict"` for the ablation.

The roadmap line "Spanish lemmatisation with a real stemmer, ablation kept in `eval/`"
can be ticked: the ablation is `--stem none` against `--stem es`, plus the `strict`
policy.
