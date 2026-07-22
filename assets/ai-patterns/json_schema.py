"""json_schema.py

Purpose:
- This module is about *authoring* JSON Schema for two Claude features:
  `output_config.format` (constrain the whole response to a schema) and strict
  tool use (guarantee a tool's `input` validates exactly). The schema is the
  contract between your code and the model — get it right and parsing becomes
  trivial; get it wrong and Claude returns a 400 or unusable output.

Real-world applications:
- Defining the exact shape of an extraction result (invoice, resume, event).
- Declaring a tool's parameters so the model must fill them correctly.
- Enforcing enums/consts for routing decisions (category, status, region).
- Versioned data contracts shared across services.

When to use it:
- Whenever you use output_config.format or set strict: True on a tool. Reach for
  hand-written JSON Schema (rather than Pydantic) when you need fine control over
  the raw schema, are working across languages, or already have a schema on hand.

Simple example:
- A closed object schema with typed properties + a required list, passed via
  output_config.format. See simple_schema_call() below.

Production example:
- A strict tool definition whose input_schema uses enum, string formats, arrays,
  and $ref/$defs, wired into a tool call. See strict_tool_call() below.

Common mistakes:
- Omitting additionalProperties: false or leaving a key out of `required`
  (structured outputs require every object to be closed and fully required).
- Using unsupported constraints and expecting them to be enforced: minimum,
  maximum, multipleOf, minLength, maxLength, and complex array constraints are
  NOT supported — the Python/TS SDKs strip them and validate client-side.
- Recursive schemas ($ref pointing back into an ancestor) — not supported.
- Putting strict on tool_choice instead of on the tool definition itself.
- Combining output_config.format with citations (400).

Best practices:
- SUPPORTED building blocks: object / array / string / integer / number /
  boolean / null, enum, const, anyOf, allOf, $ref + $defs, and string formats
  (date-time, date, time, duration, email, hostname, uri, ipv4, ipv6, uuid).
- Model optionality with anyOf that includes "null" rather than min/max bounds.
- Factor shared shapes into $defs and reference them with $ref to stay DRY.
- New schemas compile once (one-time latency) then cache ~24h — keep them stable.

Related concepts:
- Structured outputs (structured_outputs.py), strict tool use, prompt engineering.

Augmented Method learning cycle:
- Observe → Understand → Imitate → Modify → Predict → Build → Reflect → Teach
"""

import json
import os

import anthropic

MODEL = "claude-opus-4-8"

# A closed object schema: every property typed, additionalProperties forbidden,
# and every key listed in `required`. This is the minimum valid shape for
# output_config.format.
EVENT_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "date": {"type": "string", "format": "date"},          # supported string format
        "status": {"type": "string", "enum": ["draft", "confirmed", "cancelled"]},
        "attendees": {"type": "integer"},
    },
    "required": ["title", "date", "status", "attendees"],
    "additionalProperties": False,
}


def simple_schema_call(text: str) -> dict:
    """Constrain the whole response to EVENT_SCHEMA and parse it as JSON.

    With output_config.format the first text block is guaranteed to be valid
    JSON matching the schema (unless truncated or refused).
    """
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment
    response = client.messages.create(
        model=MODEL,
        max_tokens=16000,
        messages=[{"role": "user", "content": f"Extract the event:\n{text}"}],
        output_config={"format": {"type": "json_schema", "schema": EVENT_SCHEMA}},
    )
    if response.stop_reason == "max_tokens":
        raise ValueError("Truncated JSON — raise max_tokens and retry.")
    text_block = next(b.text for b in response.content if b.type == "text")
    return json.loads(text_block)


# A strict tool. The input_schema uses $defs/$ref for a reusable sub-object, an
# enum, and a supported string format. `strict: True` + additionalProperties:
# false + required guarantee the tool input validates exactly.
BOOKING_TOOL = {
    "name": "book_trip",
    "description": "Book a trip for a traveler to a destination on a date.",
    "strict": True,
    "input_schema": {
        "type": "object",
        "$defs": {
            "traveler": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "email": {"type": "string", "format": "email"},
                },
                "required": ["name", "email"],
                "additionalProperties": False,
            }
        },
        "properties": {
            "destination": {"type": "string"},
            "date": {"type": "string", "format": "date"},
            "cabin": {"type": "string", "enum": ["economy", "premium", "business"]},
            "traveler": {"$ref": "#/$defs/traveler"},
        },
        "required": ["destination", "date", "cabin", "traveler"],
        "additionalProperties": False,
    },
}


def strict_tool_call(request: str) -> dict:
    """Force a strict tool call and return the validated tool input.

    `strict: True` lives on the tool definition (not tool_choice); tool_choice
    only forces *which* tool is used.
    """
    client = anthropic.Anthropic()
    response = client.messages.create(
        model=MODEL,
        max_tokens=16000,
        tools=[BOOKING_TOOL],
        tool_choice={"type": "tool", "name": "book_trip"},
        messages=[{"role": "user", "content": request}],
    )
    for block in response.content:
        if block.type == "tool_use" and block.name == "book_trip":
            return block.input  # already validated against input_schema
    raise ValueError("Model did not call book_trip.")


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Set ANTHROPIC_API_KEY to run live examples.")
        print("Event schema keys:", list(EVENT_SCHEMA["properties"]))
        print("Booking tool input keys:", list(BOOKING_TOOL["input_schema"]["properties"]))
        return

    print("=== simple_schema_call ===")
    event = simple_schema_call("Team offsite, confirmed, on 2026-09-14, 12 people attending.")
    print(event)

    print("\n=== strict_tool_call ===")
    booking = strict_tool_call(
        "Book business class to Tokyo on 2026-10-02 for Sam Lee (sam@co.com)."
    )
    print(booking)


if __name__ == "__main__":
    main()
