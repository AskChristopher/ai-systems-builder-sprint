"""vector_search.py

Purpose:
- Given a query vector, find the nearest vectors in a large collection. This is
  the "search" half of a RAG stack: embeddings.py produces vectors; this module
  indexes and queries them by similarity.

Real-world applications:
- Semantic search, recommendation ("more like this"), dedup, clustering — any
  time you need nearest-neighbor lookup over embeddings at scale.

When to use it:
- You have thousands to billions of embeddings and need fast top-k similarity.
  For a few dozen vectors, a plain numpy scan (exact search) is fine.

Simple example:
- Exact nearest-neighbor search with cosine similarity in numpy.

Production example:
- An approximate nearest neighbor (ANN) index with FAISS, with metadata kept
  alongside so results carry their source text.

Common mistakes:
- Comparing vectors from different embedding models (incompatible spaces).
- Forgetting to normalize before using inner-product as cosine.
- Treating ANN recall as exact — ANN trades a little accuracy for speed.
- A stale index after documents change.

Best practices:
- Normalize once at write time so inner product == cosine.
- Pick the index for your scale: flat (exact, small), IVF (billions, needs
  training), HNSW (great recall/latency, memory-heavy). Tune top-k + probes.
- Store row->metadata mapping so a hit returns usable text, not just an id.

Related concepts:
- embeddings (produces the vectors), retrieval (chunking + reranking), rag.

Augmented Method learning cycle:
- Observe → Understand → Imitate → Modify → Predict → Build → Reflect → Teach
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

try:
    import numpy as np
except ImportError:  # numpy is optional for reading this file
    np = None


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def simple_example() -> None:
    """Exact top-k with cosine over a tiny in-memory set (no ANN index)."""
    docs = [
        ("refunds take 5-7 business days", [0.9, 0.1, 0.0]),
        ("rate limit is 1000 rpm", [0.1, 0.9, 0.0]),
        ("reset password in settings", [0.0, 0.1, 0.9]),
    ]
    query_vec = [0.85, 0.15, 0.0]  # "how long for a refund?"
    ranked = sorted(
        ((_cosine(query_vec, vec), text) for text, vec in docs), reverse=True
    )
    for score, text in ranked[:2]:
        print(f"{score:.3f}  {text}")


def production_example() -> None:
    """ANN search with FAISS. Vectors are L2-normalized so inner product (IP)
    equals cosine similarity; metadata is kept in a parallel list."""
    if np is None:
        print("numpy/faiss not installed; skipping production_example().")
        return
    try:
        import faiss
    except ImportError:
        print("faiss not installed; skipping production_example().")
        return

    # In production these come from embeddings.py (voyage-3, normalized).
    texts = ["refunds take 5-7 business days", "rate limit is 1000 rpm",
             "reset password in settings"]
    vectors = np.array([[0.9, 0.1, 0.0], [0.1, 0.9, 0.0], [0.0, 0.1, 0.9]],
                       dtype="float32")
    faiss.normalize_L2(vectors)  # so IndexFlatIP == cosine

    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)  # exact; swap for IndexIVFFlat/HNSW at scale
    index.add(vectors)

    query = np.array([[0.85, 0.15, 0.0]], dtype="float32")
    faiss.normalize_L2(query)
    scores, ids = index.search(query, k=2)  # top-2
    for score, idx in zip(scores[0], ids[0]):
        print(f"{score:.3f}  {texts[idx]}")


def main() -> None:
    print("=== exact (numpy) ===")
    simple_example()
    print("\n=== ANN (faiss) ===")
    production_example()


if __name__ == "__main__":
    main()
