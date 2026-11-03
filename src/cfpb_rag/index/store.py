"""Build and persist the retrieval index (dense matrix + BM25 + document table)."""
from __future__ import annotations

import json
import time
from pathlib import Path

import duckdb
import joblib
import numpy as np

from ..config import DEFAULT_EMBEDDER
from .bm25 import BM25
from .embedders import make_embedder


def doc_text(row, meta_in_text: bool) -> str:
    """Text that is embedded / BM25-indexed for one complaint."""
    if meta_in_text:
        return f"{row['product_group']}. {row['issue']}. {row['narrative']}"
    return row["narrative"]


def build_index(
    db_path: Path,
    index_dir: Path,
    embedder_spec: str = DEFAULT_EMBEDDER,
    meta_in_text: bool = True,
    batch_size: int = 64,
) -> dict:
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        docs = con.execute("SELECT * FROM complaints ORDER BY complaint_id").df()
    finally:
        con.close()
    if docs.empty:
        raise ValueError("The complaints table is empty; run ingest first.")

    t0 = time.perf_counter()
    texts = [doc_text(r, meta_in_text) for r in docs.to_dict("records")]
    embedder = make_embedder(embedder_spec)
    embedder.fit(texts)
    parts = []
    for i in range(0, len(texts), 1024):
        parts.append(embedder.encode(texts[i : i + 1024], batch_size=batch_size))
        done = min(i + 1024, len(texts))
        if (i // 1024) % 5 == 0 or done == len(texts):
            print(f"  embedded {done:,} / {len(texts):,} complaints "
                  f"({time.perf_counter() - t0:,.0f}s)", flush=True)
    matrix = np.vstack(parts)
    bm25 = BM25().fit(texts)

    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    np.save(index_dir / "matrix.npy", matrix)
    joblib.dump(docs, index_dir / "docs.joblib")
    joblib.dump(bm25, index_dir / "bm25.joblib")
    embedder.save(index_dir)
    meta = {
        "embedder": embedder.name,
        "meta_in_text": meta_in_text,
        "n_docs": int(len(docs)),
        "embedding_dim": int(matrix.shape[1]),
        "build_seconds": round(time.perf_counter() - t0, 1),
    }
    (index_dir / "index_meta.json").write_text(json.dumps(meta, indent=2))
    return meta
