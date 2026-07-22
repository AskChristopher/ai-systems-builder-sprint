"""function_calling.py

Purpose:
- "Function calling" in the Claude/Anthropic API IS tool use — the exact same
  mechanism (see tool_calling.py). You expose typed functions; Claude decides
  when to call them and with what arguments; you execute and return results.

Real-world applications:
- Turning natural language into precise, validated function calls: unit/currency
  conversion, arithmetic, record lookups, form filling, invoking microservices.

When to use it:
- You have well-defined functions with typed inputs and want the model to route
  a request to the right one with correctly-typed arguments.

Simple example:
- One calculator function via the SDK Tool Runner (schema derived from hints).

Production example:
- A small registry of typed functions dispatched automatically by the runner.

Common mistakes:
- Believing this is a different API from tool use — it is not.
- Under-describing functions so the model guesses arguments.
- Executing side-effecting functions without validating arguments first.

Best practices:
- One clear responsibility per function; enums for closed value sets; explicit
  required vs optional args; prescriptive descriptions.
- Prefer typed decorators / the Tool Runner so the JSON Schema stays in sync
  with the Python signature.

Related concepts:
- tool_calling (same feature), json_schema, structured_outputs, agents.

Augmented Method learning cycle:
- Observe → Understand → Imitate → Modify → Predict → Build → Reflect → Teach
"""

from __future__ import annotations

import os

import anthropic
from anthropic import beta_tool

MODEL = "claude-opus-4-8"


# ---- Simple: a single typed function via the Tool Runner ------------------

@beta_tool
def calculate(expression: str) -> str:
    """Evaluate a basic arithmetic expression and return the result.

    Call this when the user asks for an exact numeric computation.

    Args:
        expression: A safe arithmetic expression, e.g. '(3 + 4) * 5'.
    """
    allowed = set("0123456789+-*/(). ")
    if not set(expression) <= allowed:
        return "Error: expression contains unsupported characters."
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))  # noqa: S307
    except Exception as exc:
        return f"Error: {exc}"


def simple_example() -> None:
    """Claude routes the question to calculate(); the runner executes it."""
    client = anthropic.Anthropic()
    runner = client.beta.messages.tool_runner(
        model=MODEL, max_tokens=16000, tools=[calculate],
        messages=[{"role": "user", "content": "What is (3 + 4) * 5, then halved?"}],
    )
    for message in runner:
        for block in message.content:
            if block.type == "text":
                print(block.text)


# ---- Production: a small registry of typed functions ----------------------

_USERS = {"alice": {"plan": "Enterprise", "seats": 40}, "bob": {"plan": "Pro", "seats": 3}}


@beta_tool
def lookup_account(username: str) -> str:
    """Look up a customer's plan and seat count by username.

    Args:
        username: The account login name.
    """
    acct = _USERS.get(username.lower())
    return (f"{username}: {acct['plan']} plan, {acct['seats']} seats."
            if acct else f"No account found for '{username}'.")


@beta_tool
def convert_currency(amount: float, source: str, target: str) -> str:
    """Convert an amount between currencies using a fixed demo rate table.

    Args:
        amount: The amount to convert.
        source: ISO code to convert from, e.g. 'USD'.
        target: ISO code to convert to, e.g. 'EUR'.
    """
    rates = {("USD", "EUR"): 0.92, ("EUR", "USD"): 1.09}
    rate = rates.get((source.upper(), target.upper()))
    if rate is None:
        return f"No rate for {source}->{target}."
    return f"{amount} {source.upper()} = {round(amount * rate, 2)} {target.upper()}"


def production_example() -> None:
    """Claude chooses among several typed functions; the runner dispatches."""
    client = anthropic.Anthropic()
    runner = client.beta.messages.tool_runner(
        model=MODEL, max_tokens=16000,
        tools=[calculate, lookup_account, convert_currency],
        messages=[{"role": "user",
                   "content": "How many seats does alice have, and what is that "
                              "many times 120 USD in EUR?"}],
    )
    for message in runner:
        for block in message.content:
            if block.type == "text":
                print(block.text)


def main() -> None:
    # Local, no-API sanity check so the file is useful even without a key.
    print("calculate('(3 + 4) * 5') ->", calculate("(3 + 4) * 5"))
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Set ANTHROPIC_API_KEY to run the live examples.")
        return
    print("\n=== simple_example ===")
    simple_example()
    print("\n=== production_example ===")
    production_example()


if __name__ == "__main__":
    main()
