"""prompt_engineering.py

Purpose:
- Prompt engineering is the practice of shaping a language model's behavior
  through the text you send it: the system prompt (role, rules, tone), the
  user messages (the actual task), and the structure that surrounds them.
  With Claude the system prompt is the primary steering lever, because this
  model family (claude-opus-4-8) has removed temperature/top_p/top_k and
  assistant-message prefill — you steer with words, not sampling knobs.

Real-world applications:
- Support chatbots that must stay on-policy and cite an internal knowledge base.
- Coding copilots that must return diffs in a fixed format.
- Classification/extraction pipelines (sentiment, PII redaction, ticket routing).
- Report generators that adopt a house style and length budget.

When to use it:
- Whenever output quality, format, or safety depends on how you phrase the
  request — which is almost always the first thing to tune before reaching for
  fine-tuning, retrieval, or tools. Reach for a stronger technique only after a
  well-written prompt plateaus.

Simple example:
- A single call with a focused system prompt ("role + task + constraints +
  output shape") and one user turn. See simple_prompt() below.

Production example:
- A reusable, versioned template with named variables, explicit guardrails, an
  effort setting, and adaptive thinking for tasks that need reasoning. Log the
  stop_reason and token usage so you can catch truncation and drift.
  See production_prompt() below.

Common mistakes:
- Vague goals ("summarize this") with no length, audience, or format target.
- Conflicting instructions ("be brief" + "be exhaustive").
- Trying to force output shape with assistant-message prefill — that 400s on
  claude-opus-4-8. Use structured outputs (see structured_outputs.py) instead.
- Passing temperature/top_p/top_k — also a 400 on this model. Steer via prompt.
- Aggressive "CRITICAL: YOU MUST" tool language, which overtriggers on modern
  Claude models that already follow the system prompt closely.
- Hardcoding the API key in source instead of reading ANTHROPIC_API_KEY.

Best practices:
- Put stable instructions in `system`, the variable task in `messages`.
- Be explicit about audience, length, and output format; give a positive
  example of the shape you want rather than a list of "don'ts".
- Use adaptive thinking (thinking={"type": "adaptive"}) and an effort level
  when the task needs multi-step reasoning; leave it off for simple lookups.
- Always set max_tokens (16000 for non-streaming) and check stop_reason.

Related concepts:
- Structured outputs, JSON schema authoring, evaluation, guardrails, RAG.

Augmented Method learning cycle:
- Observe → Understand → Imitate → Modify → Predict → Build → Reflect → Teach
"""

import os

import anthropic

MODEL = "claude-opus-4-8"


def _text(response) -> str:
    """Concatenate the text blocks from a Messages API response."""
    return "".join(block.text for block in response.content if block.type == "text")


def simple_prompt(topic: str) -> str:
    """Minimal, well-structured single call.

    The system prompt carries the stable role + constraints; the user turn
    carries only the variable task. No sampling params, no prefill.
    """
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment
    response = client.messages.create(
        model=MODEL,
        max_tokens=16000,
        system=(
            "You are a precise technical explainer. "
            "Answer in at most 5 bullet points, each one sentence, no preamble."
        ),
        messages=[{"role": "user", "content": f"Explain: {topic}"}],
    )
    return _text(response)


PROMPT_TEMPLATE = (
    "You are a senior AI engineer reviewing a design for a {audience} audience.\n"
    "Task: {task}\n"
    "Constraints:\n"
    "- Lead with the single most important recommendation.\n"
    "- Keep the whole reply under {word_limit} words.\n"
    "- If information is missing, say so instead of guessing."
)


def production_prompt(task: str, audience: str = "engineering", word_limit: int = 150) -> str:
    """Templated, reasoning-enabled call suitable for a real pipeline.

    Uses a versionable template, adaptive thinking with a high effort level for
    the reasoning step, and defensive handling of truncation/refusal.
    """
    client = anthropic.Anthropic()
    system_prompt = PROMPT_TEMPLATE.format(
        audience=audience, task=task, word_limit=word_limit
    )
    response = client.messages.create(
        model=MODEL,
        max_tokens=16000,
        thinking={"type": "adaptive"},          # let Claude decide when to reason
        output_config={"effort": "high"},        # low | medium | high | xhigh | max
        system=system_prompt,
        messages=[{"role": "user", "content": task}],
    )

    if response.stop_reason == "max_tokens":
        return "[truncated: raise max_tokens]"
    if response.stop_reason == "refusal":
        return "[refused: request tripped a safety classifier]"
    return _text(response)


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Set ANTHROPIC_API_KEY to run live examples.")
        print("Simple prompt (system):")
        print("  You are a precise technical explainer...")
        print("Production template:")
        print(PROMPT_TEMPLATE.format(audience="engineering", task="<task>", word_limit=150))
        return

    print("=== simple_prompt ===")
    print(simple_prompt("What is prompt caching and when does it help?"))
    print("\n=== production_prompt ===")
    print(production_prompt("Review whether we should add a retry-with-backoff layer to our LLM client."))


if __name__ == "__main__":
    main()
