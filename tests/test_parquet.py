import pytest


def test_parquet_loader_shape():
    pyarrow = pytest.importorskip("pyarrow")
    import pyarrow as pa
    import pyarrow.parquet as pq

    from rag.parquet import load

    rows = [
        {"url": "https://a/1", "title": "Junior Data Analyst", "company": "A",
         "description": "x" * 300, "location": "Madrid", "ats": "greenhouse"},
        {"url": "https://a/2", "title": "Title only", "company": "B",
         "description": "", "location": "", "ats": "lever"},
    ]
    path = "/tmp/_rag_test_corpus.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")

    docs = load(path)
    assert len(docs) == 1, "title-only rows must be dropped"
    d = docs[0]
    assert d["id"] == "https://a/1"
    assert d["title"] == "Junior Data Analyst"
    assert d["company"] == "A"
    assert d["ats"] == "greenhouse"
    assert d["text"].startswith("Junior Data Analyst. xxx")

    assert len(load(path, limit=1)) == 1
    assert len(load(path, max_chars=20)) == 1 and len(load(path, max_chars=20)[0]["text"]) <= 20
