"""document_qa.py

Purpose:
- Answer questions over documents you hand to Claude directly. For docs that fit
  in context, pass the PDF/text as a document block (optionally with Citations)
  instead of building a full RAG pipeline.

Real-world applications:
- Contract/policy Q&A, research paper analysis, financial report review,
  knowledge-base assistants over a bounded set of documents.

When to use it:
- The relevant document(s) fit in the context window. If the corpus is too big
  to fit, retrieve first (see rag.py / retrieval.py) then answer over the top-k.

Simple example:
- Send a base64 PDF as a document block and ask a question about it.

Production example:
- Enable Citations for verifiable, span-level source references; upload once via
  the Files API to reuse across many questions; cache the doc for repeated Q&A.

Common mistakes:
- Placing the document block AFTER the text (put it before).
- base64 data containing newlines (must be one unbroken string).
- Combining Citations with output_config.format (incompatible — 400).
- Exceeding limits (32MB / up to ~600 pages for large-context models).

Best practices:
- Put document blocks before the question; enable Citations when you need
  verifiable grounding; upload once (Files API) for multi-question sessions;
  cache large docs; fall back to RAG when the corpus won't fit.

Related concepts:
- rag, retrieval, guardrails, evaluation.

Augmented Method learning cycle:
- Observe → Understand → Imitate → Modify → Predict → Build → Reflect → Teach
"""

from __future__ import annotations

import base64
import os

import anthropic

MODEL = "claude-opus-4-8"


def _pdf_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")  # no newlines


def simple_example(pdf_path: str, question: str) -> None:
    """Ask a question over a base64 PDF passed inline as a document block."""
    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=MODEL, max_tokens=16000,
        messages=[{
            "role": "user",
            "content": [
                # Document block goes BEFORE the text block.
                {"type": "document",
                 "source": {"type": "base64", "media_type": "application/pdf",
                            "data": _pdf_b64(pdf_path)}},
                {"type": "text", "text": question},
            ],
        }],
    )
    print(next((b.text for b in resp.content if b.type == "text"), ""))


def production_example(pdf_path: str, questions: list[str]) -> None:
    """Upload once via the Files API, then ask many questions with Citations on.
    Citations return verifiable, span-level references to the source document."""
    client = anthropic.Anthropic()
    uploaded = client.beta.files.upload(
        file=(os.path.basename(pdf_path), open(pdf_path, "rb"), "application/pdf"),
        betas=["files-api-2025-04-14"],
    )
    for q in questions:
        resp = client.beta.messages.create(
            model=MODEL, max_tokens=16000,
            betas=["files-api-2025-04-14"],
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": q},
                    {"type": "document",
                     "source": {"type": "file", "file_id": uploaded.id},
                     "citations": {"enabled": True}},  # NOT with output_config.format
                ],
            }],
        )
        # Cited answers split into text blocks; cited blocks carry a citations list.
        print(f"\nQ: {q}")
        for block in resp.content:
            if block.type == "text":
                print(block.text, end="")
                for cite in (getattr(block, "citations", None) or []):
                    print(f"  [cite: {getattr(cite, 'cited_text', '')[:40]}...]", end="")
        print()
    client.beta.files.delete(uploaded.id, betas=["files-api-2025-04-14"])


def main() -> None:
    print("document_qa.py — pass PDFs directly to Claude; enable Citations for")
    print("verifiable grounding. Use RAG (rag.py) when the corpus won't fit.")
    pdf = os.environ.get("SAMPLE_PDF")
    if not (os.environ.get("ANTHROPIC_API_KEY") and pdf and os.path.exists(pdf)):
        print("Set ANTHROPIC_API_KEY and SAMPLE_PDF=/path/to/file.pdf to run live.")
        return
    print("\n=== simple ===")
    simple_example(pdf, "Summarize this document in three bullet points.")
    print("\n=== production (Files API + Citations) ===")
    production_example(pdf, ["What is the main conclusion?", "List any dates mentioned."])


if __name__ == "__main__":
    main()
