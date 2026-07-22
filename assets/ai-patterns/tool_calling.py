"""tool_calling.py

Purpose:
- Give Claude the ability to reach beyond text generation by exposing external
  "tools" (functions) it can request. In the Anthropic API, tool use is the
  single mechanism behind agents, retrieval, calculators, and API integrations:
  you describe tools, Claude decides when to call them, you execute and return
  results, and the loop continues until Claude produces a final answer.

Real-world applications:
- Live data lookups (weather, stock prices, inventory) that post-date training.
- CRM / database queries, ticket creation, sending email or Slack messages.
- Multi-step research agents that search, read, and synthesize.
- Code execution, unit conversion, and math where determinism matters.

When to use it:
- When the model needs a capability it cannot do reliably from parametric
  knowledge alone: current facts, private data, side effects, or exact compute.
- When you want the model to *decide* which action to take (tool_choice "auto"),
  or you want to *force* a specific action (tool_choice "tool").
- Prefer plain text generation when no external capability is required.

Simple example:
- A single get_weather tool wired through the beta Tool Runner. The runner drives
  the whole call -> execute -> loop cycle for you; you only write the typed
  function decorated with @beta_tool.

Production example:
- A manual agentic loop over a weather tool with a tool_runner-style dispatch
  table, real error handling (is_error results), and correct message threading:
  append the assistant content block, then return ALL tool_result blocks in ONE
  user message. Handles pause_turn for long server-tool turns.

Common mistakes:
- Splitting parallel tool_results across multiple user messages -> this trains
  Claude to stop parallelizing. Send every result in ONE user message.
- Dropping the assistant content block before returning results -> loses the
  tool_use context and the API rejects the follow-up.
- Registering too many tools -> the model gets confused; keep the set focused.
- Raw-string-matching serialized JSON instead of reading block.input (a dict).
- Not gating destructive tools behind human approval.

Best practices:
- Write PRESCRIPTIVE tool descriptions ("Call this when the user asks about
  current weather..."), not just what the tool does.
- Always set max_tokens (16000 for non-streaming). Never hardcode the API key;
  anthropic.Anthropic() reads ANTHROPIC_API_KEY.
- On tool failure, return a tool_result with "is_error": True so Claude adapts.
- Prefer the Tool Runner; drop to a manual loop only when you need control it
  does not expose.

Related concepts:
- function_calling.py (same mechanism, typed-function framing), mcp.py
  (protocol-based tools), guardrails, structured outputs, workflows, agents.

Augmented Method learning cycle:
- Observe -> Understand -> Imitate -> Modify -> Predict -> Build -> Reflect -> Teach
"""

import os

import anthropic

MODEL = "claude-opus-4-8"

# A tiny fake weather backend so the examples are self-contained and runnable
# (swap for a real API call in production).
_FAKE_WEATHER = {
    "paris": "18C, light rain",
    "san francisco": "18C, foggy",
    "tokyo": "24C, clear",
}


def _lookup_weather(location: str) -> str:
    return _FAKE_WEATHER.get(location.strip().lower(), "no data for that location")


def simple_example() -> None:
    """Tool Runner (RECOMMENDED, beta): decorate a typed function; the runner
    drives the call -> execute -> loop cycle automatically."""
    from anthropic import beta_tool

    @beta_tool
    def get_weather(location: str) -> str:
        """Get the current weather for a location.

        Call this whenever the user asks about current weather, temperature,
        or conditions in a named place.

        Args:
            location: City name, e.g. "Paris" or "San Francisco".
        """
        return _lookup_weather(location)

    client = anthropic.Anthropic()
    runner = client.beta.messages.tool_runner(
        model=MODEL,
        max_tokens=16000,
        tools=[get_weather],
        messages=[{"role": "user", "content": "What's the weather in Paris?"}],
    )
    # Each iteration yields a BetaMessage; iteration stops when Claude is done.
    for message in runner:
        for block in message.content:
            if block.type == "text":
                print(block.text)


def production_example() -> None:
    """Manual agentic loop with a tool_runner dispatch table, per-tool error
    handling, and correct message threading."""
    client = anthropic.Anthropic()

    # Manual tool definition: PRESCRIPTIVE description + JSON input_schema.
    tools = [
        {
            "name": "get_weather",
            "description": (
                "Get the current weather for a location. Call this when the user "
                "asks about current weather, temperature, or conditions."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"},
                },
                "required": ["location"],
            },
        }
    ]

    # A tool_runner: map tool name -> callable. Each returns (content, is_error).
    def run_get_weather(args: dict) -> tuple[str, bool]:
        location = args.get("location", "")
        if not location:
            return ("Error: 'location' is required.", True)
        return (_lookup_weather(location), False)

    tool_runner = {"get_weather": run_get_weather}

    user_input = "Compare the weather in Tokyo and San Francisco."
    messages = [{"role": "user", "content": user_input}]

    while True:
        response = client.messages.create(
            model=MODEL, max_tokens=16000, tools=tools, messages=messages,
        )

        if response.stop_reason == "end_turn":
            break

        # Long server-tool turns can pause; re-send to resume.
        if response.stop_reason == "pause_turn":
            messages.append({"role": "assistant", "content": response.content})
            continue

        # Preserve tool_use context: append the assistant content block first.
        messages.append({"role": "assistant", "content": response.content})

        # Collect every tool_use block and return ALL results in ONE user message.
        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            handler = tool_runner.get(block.name)
            if handler is None:
                content, is_error = (f"Unknown tool: {block.name}", True)
            else:
                content, is_error = handler(block.input)  # block.input is a dict
            result = {
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": content,
            }
            if is_error:
                result["is_error"] = True
            tool_results.append(result)

        messages.append({"role": "user", "content": tool_results})

    final = next((b.text for b in response.content if b.type == "text"), "")
    print(final)


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Set ANTHROPIC_API_KEY to run the live examples.")
        return
    print("=== simple_example (Tool Runner) ===")
    simple_example()
    print("\n=== production_example (manual loop) ===")
    production_example()


if __name__ == "__main__":
    main()
