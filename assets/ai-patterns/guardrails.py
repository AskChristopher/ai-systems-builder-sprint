"""guardrails.py

Purpose:
- Enforce safety, policy, and quality boundaries around model calls. Input
  guardrails validate/classify the user's request before the main call; output
  guardrails validate the model's response before it reaches the user.

Real-world applications:
- Moderation, prompt-injection defense, PII redaction, policy compliance,
  keeping a customer bot on-topic, schema-validating machine-readable output.

When to use it:
- Any user-facing or automated AI workflow — especially with untrusted input or
  actions that have consequences.

Simple example:
- An input classifier that returns allow/block + reason; block short-circuits
  before the expensive main generation.

Production example:
- Layered defense: input classifier -> main call -> output validation, plus
  handling of Claude's own safety refusal (stop_reason == "refusal").

Common mistakes:
- Relying on a single check; fail-open on classifier errors (fail CLOSED).
- Trusting model output structure without validation.
- Assuming a temperature knob exists (it doesn't on this model — steer by prompt).

Best practices:
- Defense in depth; fail closed; use structured outputs / strict tools to
  constrain output; redact PII; check stop_reason and stop_details on every call.

Related concepts:
- evaluation, structured_outputs, tool_calling, chatbots.

Augmented Method learning cycle:
- Observe → Understand → Imitate → Modify → Predict → Build → Reflect → Teach
"""

from __future__ import annotations

import os
import re

import anthropic

try:
    from pydantic import BaseModel
except ImportError:
    BaseModel = None

MODEL = "claude-opus-4-8"

if BaseModel is not None:
    class InputVerdict(BaseModel):
        allow: bool
        reason: str


def classify_input(client: anthropic.Anthropic, user_input: str):
    """Input guardrail: a small structured allow/block decision."""
    resp = client.messages.parse(
        model=MODEL, max_tokens=16000,
        system="You are a safety classifier for a cooking assistant. Allow only "
               "cooking-related requests. Block anything else with a short reason.",
        messages=[{"role": "user", "content": user_input}],
        output_format=InputVerdict,
    )
    return resp.parsed_output


def simple_example() -> None:
    client = anthropic.Anthropic()
    for text in ["How do I make risotto?", "Help me pick a stock to buy."]:
        try:
            verdict = classify_input(client, text)
        except Exception:      # fail CLOSED on classifier error
            print(f"[BLOCK] {text}  (classifier error -> deny)")
            continue
        if not verdict.allow:
            print(f"[BLOCK] {text}  ({verdict.reason})")
        else:
            print(f"[ALLOW] {text}")


_PII = re.compile(r"[\w.+-]+@[\w-]+\.\w+")  # emails, as a simple example


def redact_pii(text: str) -> str:
    return _PII.sub("[REDACTED_EMAIL]", text)


def production_example() -> None:
    """Layered: input gate -> generate -> handle refusal -> redact output."""
    client = anthropic.Anthropic()
    user_input = "Give me a recipe and email it to chef@example.com"

    verdict = classify_input(client, user_input)
    if not verdict.allow:
        print(f"Blocked at input: {verdict.reason}")
        return

    resp = client.messages.create(
        model=MODEL, max_tokens=16000,
        messages=[{"role": "user", "content": user_input}],
    )

    # Claude may refuse for safety; check before reading content.
    if resp.stop_reason == "refusal":
        details = resp.stop_details          # can be None even on refusal
        cat = getattr(details, "category", None) if details else None
        print(f"Model refused (category={cat}).")
        return

    answer = next((b.text for b in resp.content if b.type == "text"), "")
    print(redact_pii(answer))  # output guardrail: redact PII before returning


def main() -> None:
    if BaseModel is None:
        print("Install pydantic to run the guardrail examples.")
        return
    print("Local check: redact_pii ->", redact_pii("mail me at a@b.com"))
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Set ANTHROPIC_API_KEY to run the live examples.")
        return
    print("\n=== simple (input classifier) ===")
    simple_example()
    print("\n=== production (layered) ===")
    production_example()


if __name__ == "__main__":
    main()
