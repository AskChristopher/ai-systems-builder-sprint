"""structured_outputs.py

Purpose:
- Structured outputs constrain a model's response to a fixed shape so that
  downstream code can consume it without brittle string parsing. With Claude
  (claude-opus-4-8) the preferred path is Pydantic via
  `client.messages.parse(..., output_format=MyModel)`, which returns a
  validated instance on `response.parsed_output`. A raw-schema alternative uses
  `output_config={"format": {"type": "json_schema", "schema": {...}}}`.

Real-world applications:
- Extracting contacts, invoices, or lead details from free text.
- Turning a support message into a routed ticket (category + priority + fields).
- Producing typed API responses that a frontend can render directly.
- Any step that feeds a database, a queue, or another service.

When to use it:
- When a human-readable paragraph is not enough and a machine needs specific
  keys with specific types. If you were about to write a regex to pull JSON out
  of a model reply, use structured outputs instead.

Simple example:
- Extract a person's name/email/plan into a Pydantic model with messages.parse.
  See simple_structured() below.

Production example:
- A richer nested model with enums and lists, plus handling for the failure
  modes that matter in production: max_tokens truncation and safety refusals
  (on stop_reason == "refusal" the output may not match the schema).
  See production_structured() below.

Common mistakes:
- Trying to force JSON with assistant-message prefill — that 400s on
  claude-opus-4-8. Use output_format / output_config.format.
- Combining output_config.format with citations — that returns a 400.
- Forgetting that on stop_reason == "max_tokens" the JSON is truncated and
  json.loads will fail; raise max_tokens and retry.
- Over-constraining the schema with things Claude's structured outputs do not
  support (numeric/string length bounds) — see json_schema.py.

Best practices:
- Prefer Pydantic (messages.parse) — you get validation and typed access.
- Keep every object closed with additionalProperties: false and list every key
  in `required` (the SDK enforces this for the raw-schema path).
- New schemas pay a one-time compile latency, then cache for ~24h — reuse the
  same model/schema to stay on the fast path.
- Always set max_tokens (16000 non-streaming) and check stop_reason.

Related concepts:
- JSON schema authoring (json_schema.py), strict tool use, prompt engineering.

Augmented Method learning cycle:
- Observe → Understand → Imitate → Modify → Predict → Build → Reflect → Teach
"""

import os
from enum import Enum
from typing import List, Optional

import anthropic
from pydantic import BaseModel

MODEL = "claude-opus-4-8"


class Contact(BaseModel):
    name: str
    email: str
    plan: str


def simple_structured(text: str) -> Contact:
    """Extract a Contact using the preferred Pydantic path.

    `messages.parse` validates the model output against the Pydantic model and
    returns a typed instance on `response.parsed_output`.
    """
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment
    response = client.messages.parse(
        model=MODEL,
        max_tokens=16000,
        messages=[{"role": "user", "content": f"Extract the contact details:\n{text}"}],
        output_format=Contact,
    )
    return response.parsed_output


class Priority(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"
    urgent = "urgent"


class Ticket(BaseModel):
    subject: str
    category: str
    priority: Priority
    tags: List[str]
    needs_human: bool
    suggested_reply: Optional[str] = None


def production_structured(message: str) -> Ticket:
    """Classify a support message into a typed Ticket.

    Demonstrates enums, lists, and optional fields, plus production-grade
    handling of the two failure modes that break structured parsing.
    """
    client = anthropic.Anthropic()
    response = client.messages.parse(
        model=MODEL,
        max_tokens=16000,
        thinking={"type": "adaptive"},
        output_config={"effort": "high"},
        system=(
            "You triage inbound support messages. Choose the single best "
            "category and priority. Only set needs_human when the request "
            "requires a policy decision or an action you cannot take."
        ),
        messages=[{"role": "user", "content": message}],
        output_format=Ticket,
    )

    if response.stop_reason == "max_tokens":
        raise ValueError("Output truncated — raise max_tokens and retry.")
    if response.stop_reason == "refusal":
        raise ValueError("Model refused; parsed_output may not match the schema.")
    return response.parsed_output


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Set ANTHROPIC_API_KEY to run live examples.")
        print("Contact schema:", Contact.model_json_schema()["properties"].keys())
        print("Ticket schema:", Ticket.model_json_schema()["properties"].keys())
        return

    print("=== simple_structured ===")
    contact = simple_structured("Hi, I'm Jane Doe (jane@co.com) and I want the Enterprise plan.")
    print(contact.model_dump())

    print("\n=== production_structured ===")
    ticket = production_structured(
        "Your billing charged me twice this month and I need a refund today!"
    )
    print(ticket.model_dump())


if __name__ == "__main__":
    main()
