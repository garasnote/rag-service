#!/usr/bin/env python3
"""Streamlit front end: one query box, ranked postings with citations.

    pip install -e ".[ui]"
    streamlit run streamlit_app.py

The page calls the library directly: no server, no API key, no install beyond
streamlit. Set ``API_URL`` (for example ``http://localhost:8100``) to use the HTTP
API from ``api.py`` instead, and ``LLM_API_KEY`` to get a cited answer.

Everything above ``main()`` is pure: no streamlit, no network, no I/O. That is the
part ``tests/test_ui.py`` exercises.
"""
from __future__ import annotations

import html
import json
import os
import pathlib
import re
import urllib.error
import urllib.parse
import urllib.request

try:  # optional extra, declared as [project.optional-dependencies].ui
    import streamlit as st
except ModuleNotFoundError:  # keep the pure helpers importable in a stdlib-only checkout
    st = None

from rag.core import BM25, load_corpus, strip_accents

ROOT = pathlib.Path(__file__).resolve().parent
FIXTURE = ROOT / "fixtures" / "corpus.json"

LABELS = {
    "en": {
        "title": "Job market RAG",
        "tagline": "BM25 retrieval over job postings, with cited answers.",
        "query": "Query",
        "k": "Results per query",
        "results": "Ranked postings",
        "answer": "Answer",
        "answer_btn": "Answer with citations",
        "empty": "Type a query to rank the postings.",
        "no_results": "No posting matches that query.",
        "no_key": "No LLM key configured: showing retrieval only. "
                  "Set LLM_API_KEY to get an answer with citation markers.",
        "corpus": "Corpus",
        "docs": "documents",
        "examples": "Examples: junior data scientist Barcelona / power BI Madrid / RAG engineer",
        "missing_source": "no source for citation",
        "api_error": "API error",
        "location_note": "location inferred from the posting text",
    },
    "es": {
        "title": "Buscador de ofertas",
        "tagline": "Recuperación BM25 sobre ofertas de empleo, con respuestas citadas.",
        "query": "Consulta",
        "k": "Resultados por consulta",
        "results": "Ofertas recuperadas",
        "answer": "Respuesta",
        "answer_btn": "Responder con citas",
        "empty": "Escribe una consulta para ordenar las ofertas.",
        "no_results": "Ninguna oferta coincide con esa consulta.",
        "no_key": "Sin clave de LLM: solo recuperación. "
                  "Define LLM_API_KEY para obtener una respuesta con citas.",
        "corpus": "Corpus",
        "docs": "documentos",
        "examples": "Ejemplos: junior data scientist Barcelona / power bi Madrid / ingeniero RAG",
        "missing_source": "cita sin fuente",
        "api_error": "Error de la API",
        "location_note": "ubicación inferida del texto de la oferta",
    },
}

# Postings name a city, a work mode, or a country; nothing is structured, so the
# location shown is inferred from the text and labelled as such.
CITY_PHRASES = {
    "a coruna": "A Coruña",
    "alcala": "Alcalá",
    "alicante": "Alicante",
    "almeria": "Almería",
    "badajoz": "Badajoz",
    "barcelona": "Barcelona",
    "bilbao": "Bilbao",
    "burgos": "Burgos",
    "caceres": "Cáceres",
    "cadiz": "Cádiz",
    "castellon": "Castellón",
    "castello": "Castelló",
    "cordoba": "Córdoba",
    "cornella": "Cornellà",
    "cuenca": "Cuenca",
    "donostia": "Donostia",
    "elche": "Elche",
    "gijon": "Gijón",
    "girona": "Girona",
    "granada": "Granada",
    "huelva": "Huelva",
    "huesca": "Huesca",
    "jaen": "Jaén",
    "jerez": "Jerez",
    "las palmas": "Las Palmas",
    "leganes": "Leganés",
    "leon": "León",
    "lleida": "Lleida",
    "logrono": "Logroño",
    "lugo": "Lugo",
    "madrid": "Madrid",
    "malaga": "Málaga",
    "murcia": "Murcia",
    "ourense": "Ourense",
    "oviedo": "Oviedo",
    "palencia": "Palencia",
    "pamplona": "Pamplona",
    "pontevedra": "Pontevedra",
    "salamanca": "Salamanca",
    "santander": "Santander",
    "santa cruz": "Santa Cruz",
    "segovia": "Segovia",
    "sevilla": "Sevilla",
    "tarragona": "Tarragona",
    "tenerife": "Tenerife",
    "teruel": "Teruel",
    "toledo": "Toledo",
    "valencia": "Valencia",
    "valladolid": "Valladolid",
    "vigo": "Vigo",
    "vitoria": "Vitoria",
    "zaragoza": "Zaragoza",
    "bogota": "Bogotá",
    "medellin": "Medellín",
    "cali": "Cali",
    "barranquilla": "Barranquilla",
    "bucaramanga": "Bucaramanga",
    "pereira": "Pereira",
    "manizales": "Manizales",
    "villavicencio": "Villavicencio",
    "ibague": "Ibagué",
    "santa marta": "Santa Marta",
    "neiva": "Neiva",
    "armenia": "Armenia",
    "monteria": "Montería",
    "popayan": "Popayán",
    "valledupar": "Valledupar",
    "sincelejo": "Sincelejo",
    "tunja": "Tunja",
    "pasto": "Pasto",
}
REMOTE = re.compile(r"\b(remoto|remota|remote|teletrabajo|hibrido|hybrid|fully remote)\b")
COUNTRIES = {
    "spain": "Spain", "espana": "España", "colombia": "Colombia", "portugal": "Portugal",
    "netherlands": "Netherlands", "paises bajos": "Países Bajos", "germany": "Germany",
    "france": "France", "ireland": "Ireland", "united kingdom": "United Kingdom",
}
TITLE_SPLIT = re.compile(r"\s+[\u2014\u2013-]\s+")
CITATION = re.compile(r"\[(\d+)\]")
SPACES = re.compile(r"\s+")
NON_ALNUM = re.compile(r"[^a-z0-9]+")

CHIP_CSS = """
<style>
.card { border-left: 3px solid #2f6f8f; padding: .5rem .8rem; margin: .5rem 0;
        background: rgba(47,111,143,.06); border-radius: 4px; }
.card .ttl { font-weight: 600; font-size: 1rem; text-decoration: none; color: inherit; }
.card .chip { float: right; font-family: ui-monospace, monospace; font-size: .78rem;
              background: #2f6f8f; color: #fff; padding: .05rem .45rem; border-radius: 999px; }
.card .meta { font-size: .82rem; opacity: .75; margin-top: .15rem; }
.card .snip { font-size: .88rem; margin-top: .3rem; }
.cite { text-decoration: none; font-weight: 600; }
.cite-missing { color: #b5452b; font-weight: 600; }
</style>
"""


def labels(lang: str | None = "en") -> dict:
    """UI strings only: the query, the postings and the answer stay untouched."""
    return LABELS.get((lang or "en").lower(), LABELS["en"])


def split_title(title: str) -> tuple:
    """'Datavant \u2014 Junior Data Scientist' -> ('Datavant', 'Junior Data Scientist')."""
    raw = (title or "").strip()
    parts = TITLE_SPLIT.split(raw, maxsplit=1)
    if len(parts) == 2 and parts[0].strip() and parts[1].strip():
        return parts[0].strip(), parts[1].strip()
    return "", raw


def _normalise(text: str) -> str:
    return " " + NON_ALNUM.sub(" ", strip_accents(text or "").lower()).strip() + " "


def infer_location(text: str):
    """Best-effort location: city, else work mode, plus the country when there is no city."""
    norm = _normalise(text)
    city = None
    for phrase in CITY_PHRASES:
        if f" {phrase} " in norm and (city is None or len(phrase) > len(city)):
            city = phrase
    remote = bool(REMOTE.search(norm))
    country = next((name for key, name in COUNTRIES.items() if f" {key} " in norm), None)
    parts = []
    if city:
        parts.append(CITY_PHRASES[city])
    if remote:
        parts.append("Remote")
    if country and (city is None or remote):
        parts.append(country)
    return " / ".join(parts) if parts else None


def snippet(text: str, limit: int = 220) -> str:
    clean = SPACES.sub(" ", text or "").strip()
    if len(clean) <= limit:
        return clean
    cut = clean[:limit]
    if " " in cut:
        cut = cut[: cut.rfind(" ")]
    return cut.rstrip(" ,;:.") + "..."


def safe_url(url: str) -> str:
    url = (url or "").strip()
    return url if url.startswith(("http://", "https://")) else ""


def score_chip(score) -> str:
    try:
        value = float(score)
    except (TypeError, ValueError):
        return ""
    return f'<span class="chip">{value:.3f}</span>'


def format_result(doc: dict, rank: int = 1) -> dict:
    """One raw hit -> the fields the page draws. Pure, so the tests can pin it."""
    company, title = split_title(doc.get("title", ""))
    text = doc.get("text", "")
    try:
        score = float(doc.get("score") or 0.0)
    except (TypeError, ValueError):
        score = 0.0
    return {
        "rank": int(rank),
        "title": title or company,
        "company": company,
        "location": infer_location(f"{doc.get('title', '')} {text}"),
        "url": safe_url(doc.get("url", "")),
        "score": round(score, 4),
        "chip": score_chip(score),
        "snippet": snippet(text),
    }


def format_results(hits: list) -> list:
    return [format_result(doc, i) for i, doc in enumerate(hits or [], 1)]


def result_html(result: dict) -> str:
    title = html.escape(result.get("title", ""))
    if result.get("url"):
        heading = (f'<a class="ttl" href="{html.escape(result["url"], quote=True)}" '
                   f'target="_blank" rel="noopener">{title}</a>')
    else:
        heading = f'<span class="ttl">{title}</span>'
    meta = " / ".join(x for x in (result.get("company"), result.get("location")) if x)
    return (
        f'<div class="card"><div>{heading}{result.get("chip", "")}</div>'
        f'<div class="meta">{html.escape(meta)}</div>'
        f'<div class="snip">{html.escape(result.get("snippet", ""))}</div></div>'
    )


def citation_map(sources: list) -> dict:
    """{n: url} for the sources the answer may cite."""
    out = {}
    for src in sources or []:
        try:
            n = int(src.get("n"))
        except (TypeError, ValueError, AttributeError):
            continue
        out[n] = safe_url(src.get("url", ""))
    return out


def unresolved_citations(text: str, sources: list) -> list:
    """Citation numbers the model emitted for which there is no source."""
    known = citation_map(sources)
    return sorted({int(m.group(1)) for m in CITATION.finditer(text or "")} - set(known))


def render_answer(text: str, sources: list) -> str:
    """Answer text with [n] markers turned into links; unknown markers stay visible."""
    if not text:
        return ""
    known = citation_map(sources)
    out, pos = [], 0
    for match in CITATION.finditer(text):
        out.append(html.escape(text[pos:match.start()]))
        n = int(match.group(1))
        url = known.get(n)
        if url:
            out.append(f'<a class="cite" href="{html.escape(url, quote=True)}" '
                       f'target="_blank" rel="noopener">[{n}]</a>')
        else:
            out.append(f'<span class="cite-missing" title="no source {n}">[{n}]</span>')
        pos = match.end()
    out.append(html.escape(text[pos:]))
    return "".join(out).replace("\n", "<br>")


def load_documents(corpus_path=None) -> tuple:
    """(documents, label). Falls back to the fixture so a fresh clone shows something."""
    explicit = corpus_path or os.getenv("CORPUS")
    if explicit:
        path = pathlib.Path(explicit)
        if path.exists():
            return load_corpus(corpus=path), path.name
        raise FileNotFoundError(f"corpus not found: {path}")
    docs = load_corpus()
    if docs:
        return docs, "tracker + warehouse"
    return load_corpus(corpus=FIXTURE), FIXTURE.name


def search_local(query: str, k: int, index: BM25) -> list:
    return index.search(query, k=k)


def get_json(url: str, timeout: int = 20) -> dict:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:200]
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise RuntimeError(str(exc)) from exc


def search_http(api_url: str, query: str, k: int) -> list:
    params = urllib.parse.urlencode({"q": query, "k": k})
    return get_json(f"{api_url.rstrip('/')}/search?{params}").get("results", [])


def answer_http(api_url: str, query: str, k: int) -> dict:
    body = json.dumps({"question": query, "k": k}).encode()
    req = urllib.request.Request(f"{api_url.rstrip('/')}/ask", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code}: {exc.read().decode(errors='replace')[:200]}") from exc
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise RuntimeError(str(exc)) from exc


def has_llm_key() -> bool:
    return bool(os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY"))


def main() -> None:
    if st is None:
        raise SystemExit('streamlit is not installed: pip install -e ".[ui]"')

    st.set_page_config(page_title="job-market-rag", page_icon=None, layout="centered")
    st.markdown(CHIP_CSS, unsafe_allow_html=True)

    lang = st.radio("Language", ["EN", "ES"], horizontal=True, index=0,
                    label_visibility="collapsed", key="lang")
    L = labels(lang.lower())
    st.title(L["title"])
    st.caption(L["tagline"])

    api_url = (os.getenv("API_URL") or "").rstrip("/")
    index = None
    if api_url:
        st.caption(f"{L['corpus']}: {api_url}")
    else:
        docs, source = load_documents()
        index = BM25(docs) if docs else None
        st.caption(f"{L['corpus']}: {source} / {len(docs)} {L['docs']}")

    # Widget keys: switching the language must not drop the query a reviewer typed.
    query = st.text_input(L["query"], value="", placeholder="junior data scientist Barcelona",
                          key="query")
    k = st.slider(L["k"], 1, 10, 5, key="k")

    if not query.strip():
        st.caption(L["examples"])
        return

    try:
        hits = search_http(api_url, query, k) if api_url else (
            search_local(query, k, index) if index else [])
    except RuntimeError as exc:
        st.error(f"{L['api_error']}: {exc}")
        return

    if not hits:
        st.info(L["no_results"])
        return

    st.subheader(L["results"])
    results = format_results(hits)
    st.caption(L["location_note"].capitalize())
    for result in results:
        st.markdown(result_html(result), unsafe_allow_html=True)

    st.subheader(L["answer"])
    if not api_url and not has_llm_key():
        st.caption(L["no_key"])
        return
    if not st.button(L["answer_btn"]):
        return
    try:
        payload = (answer_http(api_url, query, k) if api_url else _answer_local(query, index, k))
    except RuntimeError as exc:
        st.error(f"{L['api_error']}: {exc}")
        return
    st.markdown(render_answer(payload.get("answer", ""), payload.get("sources", [])),
                unsafe_allow_html=True)


def _answer_local(query: str, index, k: int) -> dict:
    from rag.llm import LLMError, answer

    if index is None:
        raise RuntimeError("no index: corpus is empty")
    try:
        return answer(query, index, k=k)
    except LLMError as exc:
        raise RuntimeError(str(exc)) from exc


if __name__ == "__main__":
    main()
