"""workflows.py

Purpose:
- A workflow is a multi-step pipeline where YOUR CODE controls the flow — each
  step is a scoped LLM call (or plain function) and you decide the ordering,
  branching, and stopping. Start here before reaching for an agent.

Real-world applications:
- Content pipelines (draft -> edit -> format), ticket triage (classify ->
  route -> summarize), data extraction, moderation chains.

When to use it:
- The steps are known in advance and can be specified up front. Prefer a
  workflow over an agent whenever you can express the control flow yourself —
  it's cheaper, more predictable, and easier to test than a model-driven loop.

Simple example:
- A two-step chain: classify an input, then act on the classification.

Production example:
- classify -> route -> summarize, with each step a separate messages.create
  call and deterministic Python routing between them.

Common mistakes:
- Reaching for an autonomous agent when a fixed pipeline would do (more cost,
  less determinism).
- Hidden state between steps; no observability; no per-step validation.

Best practices:
- Keep each step single-purpose with a tight prompt; validate/parse between
  steps; log inputs and outputs of every stage; make steps idempotent.

Related concepts:
- agents (model-driven alternative), tool_calling, evaluation, structured_outputs.

Augmented Method learning cycle:
- Observe → Understand → Imitate → Modify → Predict → Build → Reflect → Teach
"""

from __future__ import annotations

import os

import anthropic

MODEL = "claude-opus-4-8"


def _text(resp) -> str:
    return next((b.text for b in resp.content if b.type == "text"), "").strip()


def classify(client: anthropic.Anthropic, ticket: str) -> str:
    """Step 1: a tightly-scoped classification call."""
    resp = client.messages.create(
        model=MODEL, max_tokens=16000,
        system="Classify the support ticket as exactly one of: BILLING, BUG, "
               "HOWTO. Reply with only the label.",
        messages=[{"role": "user", "content": ticket}],
    )
    return _text(resp).upper()


def simple_example() -> None:
    """Two steps, Python decides what happens after classification."""
    client = anthropic.Anthropic()
    ticket = "I was charged twice for my subscription this month."
    label = classify(client, ticket)
    print("Label:", label)
    if label == "BILLING":
        print("-> routed to the billing queue")
    else:
        print("-> routed to general support")


def summarize(client: anthropic.Anthropic, ticket: str, label: str) -> str:
    """Step 3: summarize with a label-specific instruction."""
    focus = {"BILLING": "the amount and billing period in question",
             "BUG": "steps to reproduce and expected vs actual behavior",
             "HOWTO": "what the user is trying to accomplish"}.get(label, "the request")
    resp = client.messages.create(
        model=MODEL, max_tokens=16000,
        system=f"Summarize the ticket in one sentence, emphasizing {focus}.",
        messages=[{"role": "user", "content": ticket}],
    )
    return _text(resp)


def production_example() -> None:
    """classify -> route -> summarize, all orchestrated by Python."""
    client = anthropic.Anthropic()
    queue = {"BILLING": "billing-team", "BUG": "engineering", "HOWTO": "support"}
    ticket = ("The export button throws a 500 error whenever I select the CSV "
              "format. It worked last week.")

    label = classify(client, ticket)                 # step 1
    route = queue.get(label, "support")              # step 2 (pure code)
    summary = summarize(client, ticket, label)       # step 3
    print(f"label={label} route={route}\nsummary={summary}")


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Set ANTHROPIC_API_KEY to run the workflow examples.")
        return
    print("=== simple (classify + route) ===")
    simple_example()
    print("\n=== production (classify -> route -> summarize) ===")
    production_example()


if __name__ == "__main__":
    main()
