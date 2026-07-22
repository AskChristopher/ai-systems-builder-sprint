"""embeddings.py

Purpose:
- Turn text (or other modalities) into dense numeric vectors so that semantic
  similarity becomes geometric proximity. Embeddings are the encoding layer that
  everything else in a RAG stack sits on top of: they convert "meaning" into
  fixed-length float arrays you can compare with a dot product.

Real-world applications:
- Semantic search over a docs site, deduplication of support tickets, clustering
  of user feedback, recommendation ("more like this"), and building the index
  that a vector database queries. Any time you need "find things that mean the
  same thing" rather than "find things that share keywords."

When to use it:
- Reach for embeddings whenever you need meaning-based matching or grouping and
  exact keyword overlap is not enough (synonyms, paraphrases, cross-lingual).
  Do NOT use them when you only need exact lookups (use a hash/DB index) or when
  a small enumerated set of labels would be better served by a classifier.

Simple example:
- Embed a few sentences with Voyage AI and print the vector dimensionality plus
  a cosine similarity between two of them. See simple_example().

Production example:
- Batch-embed a document corpus with the correct input_type, L2-normalize the
  vectors once so cosine == dot product downstream, attach model+version metadata
  to every record, and embed the *query* with input_type="query". See
  production_example().

Common mistakes:
- Query/document asymmetry: embedding the query with input_type="document" (or
  vice versa) quietly degrades recall. Use "query" for queries, "document" for
  corpus text.
- Mismatched models between index build and query time — vectors from voyage-3
  and voyage-3-lite are NOT comparable. Pin one model per index.
- Forgetting to normalize when your similarity metric assumes unit vectors.
- Re-embedding on every request instead of caching; embeddings are deterministic
  per (text, model, input_type).

Best practices:
- Store the embedding model id and version alongside every vector so you can
  detect and migrate a stale index.
- Normalize once at write time; then cosine similarity is a plain dot product.
- Batch requests (Voyage accepts lists) to cut latency and cost.
- IMPORTANT FACT: Anthropic has NO first-party embeddings endpoint. Anthropic's
  docs recommend Voyage AI, so this module uses `voyageai`. Alternatives include
  Cohere and open-source sentence-transformers (all-MiniLM, bge, e5).

Related concepts:
- vector_search.py (indexing/querying these vectors), retrieval.py (chunking +
  hybrid ranking that produces the text you embed), rag.py (feeding retrieved
  text to Claude).

Augmented Method learning cycle:
Observe -> Understand -> Imitate -> Modify -> Predict -> Build -> Reflect -> Teach
"""

from __future__ import annotations

import math
from typing import List

# voyageai may not be installed in every environment; that is fine for reading
# this file. The functions below only need it at call time.
try:
    import voyageai
except ImportError:  # pragma: no cover - dependency is optional here
    voyageai = None


def _cosine(a: List[float], b: List[float]) -> float:
    """Cosine similarity between two equal-length vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def simple_example() -> None:
    """Embed a few sentences and compare two of them.

    Anthropic has no embeddings endpoint, so we call Voyage AI. VOYAGE_API_KEY
    is read from the environment by voyageai.Client() -- never hardcode a key.
    """
    if voyageai is None:
        print("voyageai not installed; skipping simple_example().")
        return

    vo = voyageai.Client()  # reads VOYAGE_API_KEY from the environment
    sentences = [
        "The cat sat on the mat.",
        "A feline rested on the rug.",
        "Quarterly revenue grew 12% year over year.",
    ]
    # Corpus/document text uses input_type="document".
    result = vo.embed(sentences, model="voyage-3", input_type="document")
    vectors = result.embeddings  # list[list[float]]

    print(f"Embedded {len(vectors)} sentences, dim={len(vectors[0])}")
    print(f"sim(cat, feline) = {_cosine(vectors[0], vectors[1]):.3f}")
    print(f"sim(cat, revenue) = {_cosine(vectors[0], vectors[2]):.3f}")


def _normalize(vec: List[float]) -> List[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec] if norm else vec


def production_example() -> None:
    """Batch-embed a corpus and run one semantic query end to end.

    Demonstrates the production-critical details: input_type asymmetry,
    normalization at write time, and model/version metadata on every record.
    """
    if voyageai is None:
        print("voyageai not installed; skipping production_example().")
        return

    vo = voyageai.Client()
    model = "voyage-3"

    corpus = [
        "Refunds are processed within 5-7 business days.",
        "Our API rate limit is 1000 requests per minute.",
        "Reset your password from the account settings page.",
    ]

    # Documents: input_type="document". Normalize so downstream cosine == dot.
    doc_result = vo.embed(corpus, model=model, input_type="document")
    records = [
        {
            "text": text,
            "vector": _normalize(vec),
            "embedding_model": model,  # metadata: detect a stale/mismatched index
            "embedding_version": "voyage-3@2024",
        }
        for text, vec in zip(corpus, doc_result.embeddings)
    ]

    # Query: input_type="query" -- asymmetric with documents on purpose.
    question = "How long do refunds take?"
    q_result = vo.embed([question], model=model, input_type="query")
    q_vec = _normalize(q_result.embeddings[0])

    scored = sorted(
        (( _cosine(q_vec, r["vector"]), r["text"]) for r in records),
        reverse=True,
    )
    print(f"Query: {question}")
    for score, text in scored:
        print(f"  {score:.3f}  {text}")


def main() -> None:
    print("=== embeddings: simple ===")
    simple_example()
    print("\n=== embeddings: production ===")
    production_example()


if __name__ == "__main__":
    main()
