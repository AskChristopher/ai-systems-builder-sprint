"""rag.py

Purpose:
- Retrieval-Augmented Generation: fetch relevant source text at query time and
  feed it to Claude so answers are grounded in your data, not just the model's
  training. This module wires the retrieve step into a generation call.

Real-world applications:
- Docs/knowledge-base Q&A, customer support over policies, internal wikis,
  research assistants that must cite specific sources.

When to use it:
- Answers must depend on private, large, or frequently-changing corpora that
  won't fit (or shouldn't be baked) into the prompt. If the whole corpus is
  small and static, just put it in the prompt (with prompt caching) instead.

Simple example:
- Retrieve top-k chunks by similarity and inject them into a grounded prompt.

Production example:
- A grounded prompt that answers ONLY from context and cites sources, plus
  notes on prompt caching a fixed corpus and Claude's native Citations feature.

Common mistakes:
- Not instructing the model to stick to the context (hallucination).
- Stuffing too much context (cost + lost-in-the-middle) with no reranking.
- No source attribution; stale index; mismatched embedding model.

Best practices:
- Ground explicitly: "Answer only from the context; if it's not there, say so."
- Keep k small and rerank; label each chunk with its source for citation.
- Cache a large fixed corpus with cache_control; consider native Citations for
  first-class, verifiable grounding.

Related concepts:
- retrieval (the retrieve step), embeddings, vector_search, document_qa.

Augmented Method learning cycle:
- Observe → Understand → Imitate → Modify → Predict → Build → Reflect → Teach
"""

from __future__ import annotations

import math
import os

import anthropic

MODEL = "claude-opus-4-8"

# Tiny in-memory corpus of (source, text). In production these come from your
# vector store / retriever (see retrieval.py, vector_search.py).
CORPUS = [
    ("refunds.md", "Refunds are processed within 5-7 business days after approval."),
    ("limits.md", "The API rate limit is 1000 requests per minute per key."),
    ("account.md", "Reset your password from the account settings page."),
]


def _score(query: str, text: str) -> float:
    """Cheap keyword overlap so the example is dependency-free. Swap for real
    vector similarity from embeddings.py + vector_search.py in production."""
    q = set(query.lower().split())
    t = set(text.lower().split())
    return len(q & t) / math.sqrt(len(q) * len(t) + 1)


def retrieve(query: str, k: int = 2) -> list[tuple[str, str]]:
    ranked = sorted(CORPUS, key=lambda row: _score(query, row[1]), reverse=True)
    return ranked[:k]


def build_grounded_prompt(chunks: list[tuple[str, str]]) -> str:
    context = "\n\n".join(f"[{src}] {text}" for src, text in chunks)
    return (
        "Answer the user's question using ONLY the context below. "
        "Cite the bracketed source for each claim. If the answer is not in the "
        "context, say you don't know.\n\n<context>\n" + context + "\n</context>"
    )


def simple_example() -> None:
    client = anthropic.Anthropic()
    question = "How long do refunds take?"
    chunks = retrieve(question)
    resp = client.messages.create(
        model=MODEL, max_tokens=16000,
        system=build_grounded_prompt(chunks),
        messages=[{"role": "user", "content": question}],
    )
    print(next((b.text for b in resp.content if b.type == "text"), ""))


def production_example() -> None:
    """Grounded answer with a cached corpus. For a large fixed corpus, mark the
    context block with cache_control so repeated questions reuse it (~0.1x cost).
    """
    client = anthropic.Anthropic()
    question = "What is the rate limit and how do I reset my password?"
    chunks = retrieve(question, k=3)
    context = "\n\n".join(f"[{src}] {text}" for src, text in chunks)
    resp = client.messages.create(
        model=MODEL, max_tokens=16000,
        system=[
            {"type": "text",
             "text": "Answer ONLY from the provided context and cite sources "
                     "with their bracketed filename. Say 'I don't know' if absent."},
            {"type": "text", "text": context,
             "cache_control": {"type": "ephemeral"}},  # cache the corpus prefix
        ],
        messages=[{"role": "user", "content": question}],
    )
    print(next((b.text for b in resp.content if b.type == "text"), ""))
    # Alternative to manual RAG: pass documents with {"citations": {"enabled": True}}
    # and Claude returns verifiable, span-level citations (see document_qa.py).


def main() -> None:
    print("Retrieved for 'refunds':", retrieve("refund time"))
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Set ANTHROPIC_API_KEY to run the live examples.")
        return
    print("\n=== simple ===")
    simple_example()
    print("\n=== production (cached corpus) ===")
    production_example()


if __name__ == "__main__":
    main()
