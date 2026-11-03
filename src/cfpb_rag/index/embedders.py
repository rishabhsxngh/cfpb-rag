"""Embedding backends.

- `st:<model>`   sentence-transformers model (recommended for real results)
- `tfidf-lsa`    TF-IDF + truncated SVD; no downloads, used for tests/CI/offline runs
- `auto`         st if sentence-transformers is installed, else tfidf-lsa
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from ..config import DEFAULT_ST_MODEL


def _normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (m / norms).astype(np.float32)


class TfidfLsaEmbedder:
    kind = "tfidf-lsa"

    def __init__(self, n_components: int = 256):
        self.n_components = n_components
        self.vec: TfidfVectorizer | None = None
        self.svd: TruncatedSVD | None = None

    @property
    def name(self) -> str:
        return "tfidf-lsa"

    def fit(self, texts):
        self.vec = TfidfVectorizer(
            sublinear_tf=True, ngram_range=(1, 2), min_df=2,
            max_features=150_000, stop_words="english",
        )
        x = self.vec.fit_transform(texts)
        n = max(2, min(self.n_components, x.shape[1] - 1, x.shape[0] - 1))
        self.svd = TruncatedSVD(n_components=n, random_state=0).fit(x)
        return self

    def encode(self, texts, batch_size: int = 256) -> np.ndarray:
        return _normalize(self.svd.transform(self.vec.transform(list(texts))))

    def save(self, d: Path) -> None:
        joblib.dump({"vec": self.vec, "svd": self.svd}, d / "embedder.joblib")
        (d / "embedder.json").write_text(json.dumps({"kind": self.kind}))

    @classmethod
    def load(cls, d: Path) -> "TfidfLsaEmbedder":
        blob = joblib.load(d / "embedder.joblib")
        e = cls()
        e.vec, e.svd = blob["vec"], blob["svd"]
        return e


class SentenceTransformerEmbedder:
    kind = "st"

    def __init__(self, model: str = DEFAULT_ST_MODEL):
        self.model_name = model
        self._model = None

    @property
    def name(self) -> str:
        return f"st:{self.model_name}"

    def _m(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def fit(self, texts):
        return self

    def encode(self, texts, batch_size: int = 64) -> np.ndarray:
        texts = list(texts)
        vecs = self._m().encode(
            texts, batch_size=batch_size, normalize_embeddings=True,
            show_progress_bar=len(texts) > 2000,
        )
        return np.asarray(vecs, dtype=np.float32)

    def save(self, d: Path) -> None:
        (d / "embedder.json").write_text(json.dumps({"kind": self.kind, "model": self.model_name}))

    @classmethod
    def load(cls, d: Path) -> "SentenceTransformerEmbedder":
        meta = json.loads((d / "embedder.json").read_text())
        return cls(meta["model"])


def make_embedder(spec: str = "auto"):
    if spec == "tfidf-lsa":
        return TfidfLsaEmbedder()
    if spec.startswith("st:"):
        return SentenceTransformerEmbedder(spec[3:])
    if spec == "auto":
        try:
            import sentence_transformers  # noqa: F401

            return SentenceTransformerEmbedder()
        except ImportError:
            warnings.warn(
                "sentence-transformers not installed; falling back to tfidf-lsa. "
                "Install with `pip install -e .[st]` for real embedding results."
            )
            return TfidfLsaEmbedder()
    raise ValueError(f"Unknown embedder spec: {spec!r}")


def load_embedder(d: Path):
    kind = json.loads((d / "embedder.json").read_text())["kind"]
    return TfidfLsaEmbedder.load(d) if kind == "tfidf-lsa" else SentenceTransformerEmbedder.load(d)
