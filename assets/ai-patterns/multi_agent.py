"""multi_agent.py

Purpose:
- Decompose a complex task across specialized agents: an orchestrator (coordinator)
  plans subtasks, worker agents each handle one with their own system prompt/tools,
  and a final call synthesizes the results.

Real-world applications:
- Research (search + read + summarize), code (plan + implement + review),
  content (outline + draft sections + edit) — anything with separable subtasks.

When to use it:
- The task splits cleanly into independent or role-specialized pieces that
  benefit from focused prompts. If it's linear and known, a workflow is simpler;
  if it's a single open-ended goal, one agent may suffice.

Simple example:
- A planner call proposes subtasks; a worker call answers one of them.

Production example:
- Orchestrator -> N specialized workers (parallelizable) -> synthesizer, all
  on one model so the shared prompt prefix stays cache-friendly.

Common mistakes:
- Ambiguous responsibilities and context drift between agents.
- Assuming workers share memory — they don't; pass what each needs explicitly.
- Switching models mid-pipeline (invalidates prompt cache).

Best practices:
- Give each agent a crisp role + inputs + expected output shape; keep workers on
  one model; run independent subtasks in parallel; synthesize deliberately.

Related concepts:
- agents, workflows, memory, evaluation.

Augmented Method learning cycle:
- Observe → Understand → Imitate → Modify → Predict → Build → Reflect → Teach
"""

from __future__ import annotations

import json
import os

import anthropic

MODEL = "claude-opus-4-8"


def _text(resp) -> str:
    return next((b.text for b in resp.content if b.type == "text"), "").strip()


def worker(client: anthropic.Anthropic, role: str, task: str) -> str:
    """A specialized worker agent: its role lives entirely in the system prompt."""
    resp = client.messages.create(
        model=MODEL, max_tokens=16000,
        system=f"You are a {role}. Do only the assigned subtask; be concise.",
        messages=[{"role": "user", "content": task}],
    )
    return _text(resp)


def simple_example() -> None:
    """Planner proposes subtasks; one worker executes one of them."""
    client = anthropic.Anthropic()
    goal = "Write a short launch blurb for a new API rate-limiting feature."
    plan = worker(client, "project planner",
                  f"List 3 subtasks (one per line) to accomplish: {goal}")
    print("Plan:\n" + plan)
    first = plan.splitlines()[0] if plan else goal
    print("\nWorker output for subtask 1:\n" + worker(client, "technical writer", first))


def orchestrate(client: anthropic.Anthropic, goal: str) -> str:
    """Coordinator decides subtasks (as JSON), workers run them, synthesizer merges."""
    plan_resp = client.messages.create(
        model=MODEL, max_tokens=16000,
        system="You are an orchestrator. Break the goal into 2-3 subtasks. "
               "Reply as a JSON array of objects: [{\"role\": ..., \"task\": ...}].",
        output_config={"format": {"type": "json_schema", "schema": {
            "type": "array",
            "items": {"type": "object",
                      "properties": {"role": {"type": "string"},
                                     "task": {"type": "string"}},
                      "required": ["role", "task"],
                      "additionalProperties": False}}}},
        messages=[{"role": "user", "content": goal}],
    )
    subtasks = json.loads(_text(plan_resp))

    # Workers are independent -> parallelize in real code (threads/async).
    outputs = [f"[{s['role']}] {worker(client, s['role'], s['task'])}"
               for s in subtasks]

    synth = client.messages.create(
        model=MODEL, max_tokens=16000,
        system="You are an editor. Synthesize the worker outputs into one "
               "coherent final answer for the goal.",
        messages=[{"role": "user",
                   "content": f"Goal: {goal}\n\nWorker outputs:\n" + "\n\n".join(outputs)}],
    )
    return _text(synth)


def production_example() -> None:
    client = anthropic.Anthropic()
    print(orchestrate(client, "Draft a concise release note for API rate limiting."))


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Set ANTHROPIC_API_KEY to run the multi-agent examples.")
        return
    print("=== simple (planner + worker) ===")
    simple_example()
    print("\n=== production (orchestrator -> workers -> synthesizer) ===")
    production_example()


if __name__ == "__main__":
    main()
