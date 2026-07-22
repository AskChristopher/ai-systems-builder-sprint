"""retrieval.py

Purpose:
- The retrieve step of RAG, broader than raw vector search: chunk documents,
  combine keyword (BM25-style) and vector signals (hybrid), filter by metadata,
  and rerank so only the most relevant, well-sized context reaches the model.

Real-world applications:
- Search assistants, support automation, doc Q&A — anywhere the quality of the
  final answer is bounded by the quality of what you retrieve.

When to use it:
- Before any grounded generation. Retrieval quality is usually the biggest lever
  on RAG accuracy — more than the generation prompt.

Simple example:
- Chunk a document and rank chunks against a query by keyword overlap.

Production example:
- Hybrid retrieval: blend a keyword score and a vector score, apply a metadata
  filter, then rerank and cap the number of chunks returned.

Common mistakes:
- Chunks too large (dilute relevance, cost) or too small (lose context).
- Pure vector search missing exact-match terms (ids, error codes) — hence hybrid.
- Returning too many chunks with no reranking; ignoring metadata filters.

Best practices:
- Chunk on semantic boundaries with modest overlap; keep source + metadata.
- Blend keyword + vector; rerank the shortlist; return a small, high-signal set.
- Evaluate retrieval separately (recall@k) from generation.

Related concepts:
- embeddings, vector_search, rag, document_qa.

Augmented Method learning cycle:
- Observe → Understand → Imitate → Modify → Predict → Build → Reflect → Teach
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import List


@dataclass
class Chunk:
    text: str
    source: str
    metadata: dict = field(default_factory=dict)
    vector: List[float] | None = None


def chunk_document(text: str, source: str, size: int = 40, overlap: int = 10) -> List[Chunk]:
    """Split text into overlapping word windows (a simple, robust default)."""
    words = text.split()
    chunks: List[Chunk] = []
    step = max(1, size - overlap)
    for start in range(0, len(words), step):
        window = words[start:start + size]
        if not window:
            break
        chunks.append(Chunk(text=" ".join(window), source=source,
                            metadata={"start": start}))
    return chunks


def _keyword_score(query: str, text: str) -> float:
    q = set(re.findall(r"\w+", query.lower()))
    t = re.findall(r"\w+", text.lower())
    if not q or not t:
        return 0.0
    hits = sum(1 for w in t if w in q)
    return hits / math.sqrt(len(t))  # length-normalized term frequency


def simple_example() -> None:
    doc = ("Refunds are processed within 5 to 7 business days. The API rate "
           "limit is 1000 requests per minute. Reset your password in settings.")
    chunks = chunk_document(doc, "faq.md", size=8, overlap=2)
    query = "how long do refunds take"
    ranked = sorted(chunks, key=lambda c: _keyword_score(query, c.text), reverse=True)
    for c in ranked[:2]:
        print(f"{_keyword_score(query, c.text):.3f}  {c.text}")


def _cosine(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)); nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def hybrid_retrieve(query: str, query_vec: List[float], chunks: List[Chunk],
                    k: int = 3, alpha: float = 0.5,
                    where: dict | None = None) -> List[Chunk]:
    """Blend keyword + vector scores, filter by metadata, return top-k.

    alpha weights vector vs keyword (0 = keyword only, 1 = vector only).
    """
    def passes(c: Chunk) -> bool:
        return all(c.metadata.get(k2) == v for k2, v in (where or {}).items())

    scored = []
    for c in chunks:
        if not passes(c):
            continue
        kw = _keyword_score(query, c.text)
        vec = _cosine(query_vec, c.vector) if c.vector else 0.0
        scored.append((alpha * vec + (1 - alpha) * kw, c))
    scored.sort(key=lambda t: t[0], reverse=True)
    return [c for _, c in scored[:k]]  # rerank + cap


def production_example() -> None:
    chunks = [
        Chunk("refunds take 5-7 business days", "faq.md",
              {"lang": "en"}, [0.9, 0.1, 0.0]),
        Chunk("rate limit is 1000 rpm", "faq.md", {"lang": "en"}, [0.1, 0.9, 0.0]),
        Chunk("mot de passe: page parametres", "faq.md",
              {"lang": "fr"}, [0.0, 0.1, 0.9]),
    ]
    results = hybrid_retrieve(
        query="refund timing", query_vec=[0.85, 0.15, 0.0], chunks=chunks,
        k=2, alpha=0.6, where={"lang": "en"},  # metadata filter
    )
    for c in results:
        print(f"[{c.source}] {c.text}")


def main() -> None:
    print("=== simple (keyword) ===")
    simple_example()
    print("\n=== production (hybrid + filter + rerank) ===")
    production_example()


if __name__ == "__main__":
    main()
