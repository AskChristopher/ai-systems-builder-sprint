"""agents.py

Purpose:
- An agent is a MODEL-DRIVEN tool-use loop: Claude decides its own next step,
  calls tools, observes results, and repeats until the goal is met. Unlike a
  workflow (you own the control flow), the model owns it here.

Real-world applications:
- Open-ended tasks: "investigate this failing test", support triage that may
  need several lookups, research that branches based on what it finds.

When to use it — check all four before building an agent:
- Complexity: the task is multi-step and hard to fully specify up front.
- Value: the outcome justifies higher cost/latency.
- Viability: Claude is actually capable at this task type.
- Cost of error: mistakes are catchable/recoverable (tests, review, rollback).
  If any answer is "no", use a workflow or a single call instead.

Simple example:
- A manual agentic loop with a stop-reason check and an iteration cap.

Production example:
- The SDK Tool Runner (recommended) plus adaptive thinking for hard tasks.

Common mistakes:
- Unbounded loops with no stop criteria — ALWAYS cap iterations.
- No telemetry; losing tool_use context; over-using an agent where a workflow
  would be cheaper and more predictable.

Best practices:
- Set explicit goals and limits; log every step; adaptive thinking +
  output_config effort for hard reasoning; gate destructive tools.

Related concepts:
- tool_calling, workflows, multi_agent, memory, guardrails, evaluation.

Augmented Method learning cycle:
- Observe → Understand → Imitate → Modify → Predict → Build → Reflect → Teach
"""

from __future__ import annotations

import os

import anthropic
from anthropic import beta_tool

MODEL = "claude-opus-4-8"

_KB = {"error 500 export": "A fix shipped in v2.3; ask the user to update."}


def _search_kb(query: str) -> str:
    for key, val in _KB.items():
        if all(w in query.lower() for w in key.split()):
            return val
    return "No knowledge-base article matched."


def simple_example() -> None:
    """Manual loop: model decides when to call the tool; we cap iterations."""
    client = anthropic.Anthropic()
    tools = [{
        "name": "search_kb",
        "description": "Search the internal knowledge base. Call this when you "
                       "need product facts you don't already know.",
        "input_schema": {"type": "object",
                         "properties": {"query": {"type": "string"}},
                         "required": ["query"]},
    }]
    messages = [{"role": "user",
                 "content": "A user reports error 500 when exporting. What do I tell them?"}]

    response = None
    for _ in range(6):  # hard iteration cap == stop criterion
        response = client.messages.create(
            model=MODEL, max_tokens=16000, tools=tools, messages=messages,
        )
        if response.stop_reason == "end_turn":
            break
        if response.stop_reason == "pause_turn":  # server-tool pause; resume
            messages.append({"role": "assistant", "content": response.content})
            continue
        messages.append({"role": "assistant", "content": response.content})
        results = []
        for block in response.content:
            if block.type == "tool_use":
                out = _search_kb(block.input.get("query", ""))
                results.append({"type": "tool_result", "tool_use_id": block.id,
                                "content": out})
        messages.append({"role": "user", "content": results})
    print(next((b.text for b in response.content if b.type == "text"), ""))


@beta_tool
def search_kb(query: str) -> str:
    """Search the internal knowledge base for product facts.

    Args:
        query: What to look up.
    """
    return _search_kb(query)


def production_example() -> None:
    """Tool Runner drives the loop; adaptive thinking for harder reasoning."""
    client = anthropic.Anthropic()
    runner = client.beta.messages.tool_runner(
        model=MODEL, max_tokens=16000,
        thinking={"type": "adaptive"}, output_config={"effort": "high"},
        tools=[search_kb],
        messages=[{"role": "user",
                   "content": "Diagnose the export 500 error and draft a reply."}],
    )
    for message in runner:
        for block in message.content:
            if block.type == "text":
                print(block.text)


def main() -> None:
    print("Local check: _search_kb('error 500 export') ->", _search_kb("error 500 export"))
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Set ANTHROPIC_API_KEY to run the agent examples.")
        return
    print("\n=== simple (manual loop) ===")
    simple_example()
    print("\n=== production (Tool Runner + thinking) ===")
    production_example()


if __name__ == "__main__":
    main()
