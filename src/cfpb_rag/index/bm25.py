"""Small BM25 implementation on a sparse term-frequency matrix."""
from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


class BM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b

    def fit(self, texts) -> "BM25":
        self.vec = CountVectorizer(
            lowercase=True, stop_words="english", token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b"
        )
        x = self.vec.fit_transform(texts).astype(np.float32).tocsc()
        n = x.shape[0]
        df = np.asarray((x > 0).sum(axis=0)).ravel()
        self.idf = np.log(1.0 + (n - df + 0.5) / (df + 0.5)).astype(np.float32)
        self.dl = np.asarray(x.sum(axis=1)).ravel().astype(np.float32)
        self.avgdl = float(self.dl.mean()) or 1.0
        self.x = x
        return self

    def scores(self, query: str) -> np.ndarray:
        out = np.zeros(self.x.shape[0], dtype=np.float32)
        q = self.vec.transform([query])
        for j in q.indices:
            start, end = self.x.indptr[j], self.x.indptr[j + 1]
            rows, tf = self.x.indices[start:end], self.x.data[start:end]
            denom = tf + self.k1 * (1 - self.b + self.b * self.dl[rows] / self.avgdl)
            out[rows] += self.idf[j] * tf * (self.k1 + 1) / denom
        return out
