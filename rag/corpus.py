"""One entry point for every corpus this repo can read.

Three sources exist and they used to be loaded in three different places:

- the private default: tracker records + warehouse postings (``core.load_corpus``)
- a standalone JSON list of documents (fixtures, checked-in samples)
- a harvested Parquet snapshot (``rag.parquet``, needs the ``parquet`` extra)

``load_source`` dispatches on the file extension so callers stop caring which
one they were handed. The CLI, the HTTP API and the Streamlit app all come here.
"""
from __future__ import annotations

import pathlib

from .core import load_corpus

PARQUET_SUFFIXES = {".parquet", ".pq"}


def load_source(spec: str | pathlib.Path | None = None, limit: int | None = None) -> list[dict]:
    """Load documents from ``spec``.

    ``None`` reads the default tracker + warehouse corpus. A ``.parquet`` file is
    streamed through :mod:`rag.parquet`; anything else is read as a JSON list by
    :func:`rag.core.load_corpus`.
    """
    if spec in (None, ""):
        return load_corpus()

    path = pathlib.Path(spec).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"corpus not found: {path}")

    if path.suffix.lower() in PARQUET_SUFFIXES:
        from .parquet import load as load_parquet
        return load_parquet(path, limit=limit)

    return load_corpus(corpus=path)
