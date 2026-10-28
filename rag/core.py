"""Retrieval core: corpus loading, tokenising, BM25. Standard library only."""
import json
import math
import pathlib
import re
import sqlite3
import unicodedata
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parents[1]
TRACKER = ROOT.parents[1] / "job_tracker" / "data.json"
WAREHOUSE = ROOT.parent / "hiresignal" / "warehouse.db"

STOP = set("""a al algo algunas algunos ante antes como con contra cual cuando de del desde donde
dos el ella ellas ellos en entre era eres es esa esas ese eso esos esta estaba estan este esto estos
fue ha hay la las le les lo los mas me mi mis mucho muy nada ni no nos o para pero por porque que
quien se sin sobre su sus tambien tan te tiene todo tu tus un una uno unos y ya
the a an and are as at be by for from has have how in is it its of on or that the their this to was
were what when where which who why will with you your""".split())
WORD = re.compile(r"[a-z0-9][a-z0-9+#.-]*")
# Token shapes are found on the *surface* text when stemming, because the entity
# guard needs to see capitals before they are lowered away.
SURFACE = re.compile(r"[0-9A-Za-z\u00c0-\u024f][0-9A-Za-z\u00c0-\u024f+#.-]*")


def strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


def tokenize(text: str, stem: bool = False) -> list:
    """Lowercase, accent-free tokens. ``stem`` runs the Spanish stemmer over the
    tokens the entity guard allows and leaves the rest untouched."""
    if stem:
        return _tokenize_stemmed(text)
    text = strip_accents((text or "").lower())
    return [w for w in WORD.findall(text) if w not in STOP and len(w) > 1]


def _tokenize_stemmed(text: str) -> list:
    from .stem_es import stem_es, stemmable

    raw = text or ""
    out = []
    for match in SURFACE.finditer(raw):
        surface = match.group(0)
        token = strip_accents(surface.lower())
        if token in STOP or len(token) < 2:
            continue
        if stemmable(surface):
            token = stem_es(token)
        if token:
            out.append(token)
    return out


def load_corpus(tracker: pathlib.Path = TRACKER, warehouse: pathlib.Path = WAREHOUSE,
                corpus: pathlib.Path | None = None) -> list:
    """One document per job record / posting. Every doc keeps its url for citation.

    Pass ``corpus`` to read a standalone JSON list instead of the private tracker
    and warehouse — that is how a fresh clone runs the eval and the tests.
    """
    if corpus is not None:
        rows = json.loads(pathlib.Path(corpus).read_text(encoding="utf-8"))
        return [{"id": r.get("id") or f"fx:{i}", "url": r.get("url", ""),
                 "title": r.get("title", ""), "text": r.get("text", "")}
                for i, r in enumerate(rows)]

    docs, seen = [], set()

    if tracker.exists():
        data = json.loads(tracker.read_text(encoding="utf-8"))
        for c in data.get("companies", []):
            url = (c.get("url") or "").split("#")[0]
            key = url or f"{c.get('name')}|{c.get('role')}"
            if key in seen:
                continue
            seen.add(key)
            text = " ".join(str(x) for x in [
                c.get("name", ""), c.get("role", ""), c.get("city", ""), c.get("country", ""),
                c.get("type", ""), c.get("focus", ""), c.get("notes", ""), c.get("status", ""),
            ] if x)
            docs.append({"id": f"tracker:{c.get('id', len(docs))}", "url": url,
                         "title": f"{c.get('name','')} — {c.get('role','')}", "text": text})

    if warehouse.exists():
        con = sqlite3.connect(warehouse)
        con.row_factory = sqlite3.Row
        try:
            rows = con.execute("""
                SELECT f.url, f.title, f.company, f.city, f.source, f.salary_min_cop,
                       f.salary_max_cop, f.is_junior, r.family
                FROM fact_offer f LEFT JOIN dim_role r ON r.role_key = f.role_key
            """).fetchall()
        except sqlite3.Error:
            rows = []
        for r in rows:
            url = (r["url"] or "").split("#")[0]
            if url in seen:
                continue
            seen.add(url)
            pay = ""
            if r["salary_min_cop"]:
                pay = f"salario {r['salary_min_cop']} a {r['salary_max_cop'] or r['salary_min_cop']} COP"
            text = " ".join(str(x) for x in [
                r["title"], r["company"], r["city"], r["family"], r["source"], pay,
                "junior" if r["is_junior"] else "",
            ] if x)
            docs.append({"id": f"warehouse:{url}", "url": url, "title": f"{r['company'] or 'sin empresa'} — {r['title']}", "text": text})
        con.close()

    return docs


class BM25:
    """Okapi BM25. Small corpus, so scoring every doc per query is fine.

    ``stem=True`` indexes and queries Spanish stems instead of surface words;
    the entity guard in stem_es keeps company names out of the stemmer either way.
    """
    def __init__(self, docs, k1: float = 1.5, b: float = 0.75, stem: bool = False):
        self.docs = docs
        self.k1, self.b = k1, b
        self.stem = stem
        self.tokens = [tokenize(d["text"], stem=stem) for d in docs]
        self.tf = [Counter(t) for t in self.tokens]
        self.dl = [len(t) for t in self.tokens]
        self.avgdl = (sum(self.dl) / len(self.dl)) if self.dl else 0.0
        df = Counter()
        for t in self.tokens:
            df.update(set(t))
        n = len(docs)
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}

    def search(self, query: str, k: int = 5) -> list:
        q = tokenize(query, stem=self.stem)
        scored = []
        for i, tf in enumerate(self.tf):
            s = 0.0
            for w in q:
                f = tf.get(w)
                if not f:
                    continue
                dl = self.dl[i] or 1
                s += self.idf[w] * (f * (self.k1 + 1)) / (f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1)))
            if s > 0:
                scored.append((s, i))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [{"score": round(s, 4), **self.docs[i]} for s, i in scored[:k]]


def chunk(text: str, size: int = 600, overlap: int = 80) -> list:
    """Character chunks for the LLM prompt; retrieval stays at document level."""
    text = (text or "").strip()
    if not text:
        return []
    out, i = [], 0
    while i < len(text):
        out.append(text[i:i + size])
        if i + size >= len(text):
            break
        i += size - overlap
    return out
