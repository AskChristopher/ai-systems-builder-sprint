"""streaming.py

Purpose:
- Deliver output token-by-token as it's generated instead of waiting for the
  whole response. Improves perceived latency and is REQUIRED for large outputs
  (the SDK guards non-streaming requests with big max_tokens against timeouts).

Real-world applications:
- Chat UIs, live coding copilots, voice pipelines, any long generation where a
  progress feel matters or where max_tokens is high.

When to use it:
- The response is long or latency-sensitive, or max_tokens is above ~16000
  (128K output on this model is only reachable via streaming).

Simple example:
- Iterate stream.text_stream and print tokens as they arrive.

Production example:
- Handle individual event types (text vs thinking deltas), read usage, and get
  the accumulated final message; includes an async variant note.

Common mistakes:
- Not flushing output (tokens appear in bursts).
- Requesting large max_tokens without streaming (SDK raises / times out).
- Ignoring the final message / stop_reason.

Best practices:
- Always flush; use get_final_message() for the complete result + usage;
  set thinking display to "summarized" so reasoning streams instead of pausing.

Related concepts:
- chatbots, voice, agents.

Augmented Method learning cycle:
- Observe → Understand → Imitate → Modify → Predict → Build → Reflect → Teach
"""

from __future__ import annotations

import os

import anthropic

MODEL = "claude-opus-4-8"


def simple_example() -> None:
    """The high-level helper: iterate text_stream and print as it arrives."""
    client = anthropic.Anthropic()
    with client.messages.stream(
        model=MODEL, max_tokens=64000,
        messages=[{"role": "user", "content": "Write a haiku about streaming data."}],
    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)   # flush so tokens appear immediately
    print()


def production_example() -> None:
    """Handle event types (thinking vs text deltas), then read the final message."""
    client = anthropic.Anthropic()
    with client.messages.stream(
        model=MODEL, max_tokens=64000,
        thinking={"type": "adaptive", "display": "summarized"},  # stream reasoning
        messages=[{"role": "user",
                   "content": "Briefly reason, then answer: 27 * 453."}],
    ) as stream:
        for event in stream:
            if event.type == "content_block_delta":
                if event.delta.type == "thinking_delta":
                    print(event.delta.thinking, end="", flush=True)
                elif event.delta.type == "text_delta":
                    print(event.delta.text, end="", flush=True)
            elif event.type == "message_delta" and event.usage:
                pass  # usage.output_tokens accrues here if you want a live counter
        final = stream.get_final_message()      # accumulated Message
    print(f"\n[stop_reason={final.stop_reason} out_tokens={final.usage.output_tokens}]")


ASYNC_NOTE = """\
Async variant (for concurrent servers) — use anthropic.AsyncAnthropic:

    async_client = anthropic.AsyncAnthropic()
    async with async_client.messages.stream(
        model="claude-opus-4-8", max_tokens=64000,
        messages=[{"role": "user", "content": "Write a story"}],
    ) as stream:
        async for text in stream.text_stream:
            print(text, end="", flush=True)
"""


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Set ANTHROPIC_API_KEY to run the streaming examples.")
        print("\n" + ASYNC_NOTE)
        return
    print("=== simple (text_stream) ===")
    simple_example()
    print("\n=== production (event handling) ===")
    production_example()
    print("\n" + ASYNC_NOTE)


if __name__ == "__main__":
    main()
