"""Retrieval helpers for EPR.

BM25-flavored sparse retrieval (TF-IDF cosine; close enough to canonical BM25 for candidate-set generation without
adding `rank_bm25`) and dense top-k via numpy or optional FAISS.
"""
from __future__ import annotations

import logging
import warnings

import numpy as np

logger = logging.getLogger(__name__)


def build_bm25_index(documents: list[str]) -> dict:
    """Build a TF-IDF/BM25-flavored sparse index using sklearn.

    Returns a dict containing the fitted `TfidfVectorizer` and the `[N, V]` sparse document matrix. The dict is
    joblib-serializable for `RetrievalMemory`.

    The paper uses canonical BM25 (Robertson & Zaragoza 2009); we use TF-IDF cosine as a close approximation to avoid
    adding the `rank_bm25` dependency. Users needing canonical BM25 can build their own index and inject it via the
    `RetrievalMemory.bm25_state` field.

    Args:
        documents: One string per document.

    Returns:
        Dict with keys `"vectorizer"` and `"doc_matrix"`.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize

    vectorizer = TfidfVectorizer()
    doc_matrix = vectorizer.fit_transform(documents)
    doc_matrix = normalize(doc_matrix, norm="l2", axis=1, copy=False)
    return {"vectorizer": vectorizer, "doc_matrix": doc_matrix}


def bm25_search(index: dict, query: str, k: int) -> list[int]:
    """Return indices of top-k documents by TF-IDF cosine similarity (descending)."""
    from sklearn.preprocessing import normalize

    vectorizer = index["vectorizer"]
    doc_matrix = index["doc_matrix"]
    q_vec = vectorizer.transform([query])
    q_vec = normalize(q_vec, norm="l2", axis=1, copy=False)
    scores = (doc_matrix @ q_vec.T).toarray().ravel()

    n = scores.shape[0]
    if k <= 0:
        return []
    if k >= n:
        return list(np.argsort(-scores).astype(int))
    top = np.argpartition(-scores, kth=k - 1)[:k]
    top_sorted = top[np.argsort(-scores[top])]
    return list(top_sorted.astype(int))


def dense_top_k(
    query_embedding: np.ndarray,
    corpus_embeddings: np.ndarray,
    k: int,
    use_faiss: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Return `(indices, scores)` of top-k by inner product, descending.

    If `use_faiss=True` and FAISS is importable, uses `IndexFlatIP`. Otherwise (or on import failure) falls back to a
    numpy matmul + `argpartition`.

    Args:
        query_embedding: `[dim]` array.
        corpus_embeddings: `[N, dim]` array.
        k: Number of results.
        use_faiss: Opt-in FAISS path.

    Returns:
        `(indices, scores)` as 1-D numpy arrays of length `min(k, N)`.
    """
    if use_faiss:
        try:
            import faiss  # type: ignore

            return _faiss_top_k(query_embedding, corpus_embeddings, k)
        except ImportError:
            warnings.warn(
                "use_faiss=True but faiss is not importable; falling back to numpy.",
                UserWarning,
                stacklevel=2,
            )
    return _numpy_top_k(query_embedding, corpus_embeddings, k)


def _numpy_top_k(
    query_embedding: np.ndarray,
    corpus_embeddings: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    if corpus_embeddings.size == 0:
        return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.float32)

    q = np.asarray(query_embedding, dtype=corpus_embeddings.dtype).reshape(-1)
    scores = corpus_embeddings @ q  # [N]
    n = scores.shape[0]

    if k <= 0:
        return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=scores.dtype)
    if k >= n:
        order = np.argsort(-scores)
        return order.astype(np.int64), scores[order]

    top = np.argpartition(-scores, kth=k - 1)[:k]
    top_sorted = top[np.argsort(-scores[top])]
    return top_sorted.astype(np.int64), scores[top_sorted]


def _faiss_top_k(
    query_embedding: np.ndarray,
    corpus_embeddings: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    import faiss  # type: ignore

    corpus = np.ascontiguousarray(corpus_embeddings, dtype=np.float32)
    query = np.ascontiguousarray(query_embedding, dtype=np.float32).reshape(1, -1)
    n, dim = corpus.shape
    if n == 0:
        return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.float32)
    effective_k = min(k, n)
    index = faiss.IndexFlatIP(dim)
    index.add(corpus)
    scores, indices = index.search(query, effective_k)
    return indices[0].astype(np.int64), scores[0].astype(np.float32)
