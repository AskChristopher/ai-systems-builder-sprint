"""evaluation.py

Purpose:
- Measure whether an LLM system actually works: score outputs against criteria,
  catch regressions when you change a prompt or model, and track quality over
  time. The workhorse pattern is LLM-as-judge with a structured verdict.

Real-world applications:
- Regression suites before shipping a prompt change, A/B comparing prompts or
  models, online quality monitoring of production traffic.

When to use it:
- Before and after ANY meaningful change to prompts, tools, or model. If you
  can't measure it, you can't safely change it.

Simple example:
- An LLM judge scores one answer against a rubric and returns a structured
  pass/fail + reason via structured outputs.

Production example:
- A small eval harness: run the system under test over a dataset, judge each
  case, and aggregate a pass rate.

Common mistakes:
- Subjective scores with no rubric or baseline.
- Judge position/verbosity bias — mitigate by blinding and randomizing order.
- Testing only happy paths; no regression set.

Best practices:
- Keep a versioned golden dataset; force the judge to return structured output;
  separate offline eval sets from online monitoring; note that this model has no
  temperature knob, so variation comes from prompts, not sampling params.

Related concepts:
- guardrails, prompt_engineering, structured_outputs, workflows.

Augmented Method learning cycle:
- Observe → Understand → Imitate → Modify → Predict → Build → Reflect → Teach
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import anthropic

try:
    from pydantic import BaseModel
except ImportError:
    BaseModel = None

MODEL = "claude-opus-4-8"


if BaseModel is not None:
    class Verdict(BaseModel):
        passed: bool
        score: int          # 1-5
        reason: str


RUBRIC = ("A good answer is factually correct, directly addresses the question, "
          "and is concise. Score 1 (poor) to 5 (excellent).")


def judge(client: anthropic.Anthropic, question: str, answer: str):
    """LLM-as-judge returning a validated structured verdict."""
    resp = client.messages.parse(
        model=MODEL, max_tokens=16000,
        system=f"You are a strict grader. {RUBRIC} Return a structured verdict.",
        messages=[{"role": "user",
                   "content": f"Question: {question}\n\nAnswer: {answer}"}],
        output_format=Verdict,
    )
    return resp.parsed_output


def simple_example() -> None:
    client = anthropic.Anthropic()
    v = judge(client, "What is the capital of France?", "Paris.")
    print(f"passed={v.passed} score={v.score} reason={v.reason}")


@dataclass
class Case:
    question: str
    answer: str  # output from the system under test


def run_system_under_test(client: anthropic.Anthropic, question: str) -> str:
    resp = client.messages.create(
        model=MODEL, max_tokens=16000,
        messages=[{"role": "user", "content": question}],
    )
    return next((b.text for b in resp.content if b.type == "text"), "")


def production_example() -> None:
    """Run the system over a dataset, judge each output, aggregate pass rate."""
    client = anthropic.Anthropic()
    dataset = [
        "What is the capital of France?",
        "What is 12 * 12?",
        "Name the largest planet in the solar system.",
    ]
    passed = 0
    for q in dataset:
        answer = run_system_under_test(client, q)   # produce
        v = judge(client, q, answer)                # grade
        status = "PASS" if v.passed else "FAIL"
        print(f"[{status}] ({v.score}/5) {q} -> {answer[:60]}")
        passed += int(v.passed)
    print(f"\nPass rate: {passed}/{len(dataset)} = {passed / len(dataset):.0%}")


def main() -> None:
    if BaseModel is None:
        print("Install pydantic to run the judge examples.")
        return
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Set ANTHROPIC_API_KEY to run the evaluation examples.")
        return
    print("=== simple (single judge) ===")
    simple_example()
    print("\n=== production (eval harness) ===")
    production_example()


if __name__ == "__main__":
    main()
