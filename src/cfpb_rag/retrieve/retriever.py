"""Dense, BM25 and hybrid (reciprocal rank fusion) retrieval with metadata filters."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ..config import INDEX_DIR
from ..index.embedders import load_embedder

MODES = ("dense", "bm25", "hybrid")
RRF_K = 60


@dataclass
class Filters:
    product_group: str | None = None
    company: str | None = None
    state: str | None = None
    issue: str | None = None
    date_from: str | None = None
    date_to: str | None = None

    def is_empty(self) -> bool:
        return not any(self.__dict__.values())


@dataclass
class Hit:
    complaint_id: str
    score: float          # score used for ranking (RRF for hybrid, cosine for dense, BM25 for bm25)
    dense_score: float    # cosine similarity to the query, always computed
    rank: int
    text: str
    meta: dict


class Retriever:
    def __init__(self, index_dir: Path = INDEX_DIR):
        index_dir = Path(index_dir)
        self.matrix = np.load(index_dir / "matrix.npy")
        self.docs: pd.DataFrame = joblib.load(index_dir / "docs.joblib")
        self.bm25 = joblib.load(index_dir / "bm25.joblib")
        self.embedder = load_embedder(index_dir)
        self.info = json.loads((index_dir / "index_meta.json").read_text())
        self._cross_encoder = None

        d = self.docs
        self._ids = d["complaint_id"].to_numpy()
        self._lower = {
            c: d[c].fillna("").astype(str).str.lower().to_numpy()
            for c in ("product_group", "company", "state", "issue")
        }
        self._dates = pd.to_datetime(d["date_received"]).to_numpy("datetime64[ns]")

    def __len__(self) -> int:
        return len(self.docs)

    # ------------------------------------------------------------------ filters
    def mask(self, f: Filters | None) -> np.ndarray:
        m = np.ones(len(self.docs), dtype=bool)
        if f is None:
            return m
        for col in ("product_group", "company", "state", "issue"):
            val = getattr(f, col)
            if val:
                m &= self._lower[col] == val.lower()
        if f.date_from:
            m &= self._dates >= np.datetime64(f.date_from)
        if f.date_to:
            m &= self._dates <= np.datetime64(f.date_to)
        return m

    @staticmethod
    def _top(scores: np.ndarray, mask: np.ndarray, n: int) -> np.ndarray:
        s = np.where(mask, scores, -np.inf)
        n = int(min(n, mask.sum()))
        if n <= 0:
            return np.array([], dtype=int)
        idx = np.argpartition(-s, n - 1)[:n]
        return idx[np.argsort(-s[idx])]

    # ------------------------------------------------------------------ search
    def search(
        self,
        query: str,
        k: int = 5,
        mode: str = "hybrid",
        filters: Filters | None = None,
        rerank: bool = False,
        candidates: int = 50,
    ) -> list[Hit]:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        mask = self.mask(filters)
        if not mask.any():
            return []

        qvec = self.embedder.encode([query])[0]
        sims = self.matrix @ qvec
        n_pool = max(candidates, k) if (rerank or mode == "hybrid") else k

        if mode == "dense":
            order = self._top(sims, mask, n_pool)
            scores = {int(i): float(sims[i]) for i in order}
        elif mode == "bm25":
            bm = self.bm25.scores(query)
            order = [i for i in self._top(bm, mask, n_pool) if bm[i] > 0]
            scores = {int(i): float(bm[i]) for i in order}
        else:
            bm = self.bm25.scores(query)
            dense_order = self._top(sims, mask, n_pool)
            bm_order = [i for i in self._top(bm, mask, n_pool) if bm[i] > 0]
            fused: dict[int, float] = {}
            for ranking in (dense_order, bm_order):
                for r, i in enumerate(ranking):
                    fused[int(i)] = fused.get(int(i), 0.0) + 1.0 / (RRF_K + r + 1)
            scores = dict(sorted(fused.items(), key=lambda kv: -kv[1]))

        ranked = list(scores.items())
        if rerank and ranked:
            ranked = self._rerank(query, ranked[:candidates])
        ranked = ranked[:k]

        hits = []
        for rank, (i, sc) in enumerate(ranked, start=1):
            row = self.docs.iloc[i]
            hits.append(
                Hit(
                    complaint_id=str(row["complaint_id"]),
                    score=float(sc),
                    dense_score=float(sims[i]),
                    rank=rank,
                    text=str(row["narrative"]),
                    meta={
                        "product_group": row["product_group"],
                        "issue": row["issue"],
                        "sub_issue": row["sub_issue"],
                        "company": row["company"],
                        "state": row["state"],
                        "date": str(pd.Timestamp(row["date_received"]).date()),
                    },
                )
            )
        return hits

    def _rerank(self, query: str, ranked: list[tuple[int, float]]) -> list[tuple[int, float]]:
        if self._cross_encoder is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as e:  # pragma: no cover
                raise RuntimeError(
                    "Reranking needs sentence-transformers: pip install -e .[st]"
                ) from e
            self._cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        pairs = [(query, str(self.docs.iloc[i]["narrative"])[:1000]) for i, _ in ranked]
        ce = self._cross_encoder.predict(pairs)
        order = np.argsort(-ce)
        return [(ranked[j][0], float(ce[j])) for j in order]
