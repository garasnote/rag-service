"""Parquet corpora: the harvest writes one file, the retriever reads it.

Kept out of ``core`` so the base install stays stdlib-only — ``pyarrow`` lives in
the ``parquet`` extra and is imported lazily. Rows are streamed in batches:
``read_table`` + ``to_pylist`` on a 111 MB snapshot took 221 s on this box, the
same file through ``iter_batches`` takes 1.4 s.
"""
from __future__ import annotations

import pathlib

BATCH = 20_000
WANTED = ("url", "title", "company", "location", "description", "ats")


def available() -> bool:
    try:
        import pyarrow  # noqa: F401
        return True
    except ImportError:
        return False


def _schema_names(path: pathlib.Path) -> set[str]:
    import pyarrow.parquet as pq
    return set(pq.read_schema(path).names)


def load(path: str | pathlib.Path, limit: int | None = None, min_chars: int = 200,
         max_chars: int = 1500) -> list[dict]:
    """Read a parquet snapshot into the document shape the pipeline expects.

    ``min_chars`` drops title-only rows, which cannot be retrieved on content;
    ``max_chars`` bounds memory per document.
    """
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise RuntimeError('parquet corpora need pyarrow: pip install -e ".[parquet]"') from exc

    path = pathlib.Path(path)
    cols = [c for c in WANTED if c in _schema_names(path)]
    docs: list[dict] = []
    seen = 0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=BATCH, columns=cols):
        data = {c: batch.column(c).to_pylist() for c in cols}
        for i in range(batch.num_rows):
            seen += 1
            body = (data.get("description", [""] * batch.num_rows)[i] or "")
            if len(body) < min_chars:
                continue
            title = (data.get("title", [""] * batch.num_rows)[i] or "").strip()
            url = (data.get("url", [""] * batch.num_rows)[i] or "").strip()
            docs.append({
                "id": url or f"pq:{seen}",
                "url": url,
                "title": title,
                "company": (data.get("company", [""] * batch.num_rows)[i] or ""),
                "location": (data.get("location", [""] * batch.num_rows)[i] or ""),
                "ats": (data.get("ats", [""] * batch.num_rows)[i] or ""),
                "text": f"{title}. {body}"[:max_chars],
            })
            if limit and len(docs) >= limit:
                return docs
    return docs
