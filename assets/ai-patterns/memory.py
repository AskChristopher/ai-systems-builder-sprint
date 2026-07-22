"""memory.py

Purpose:
- Persist useful state so the assistant has continuity. Two layers: short-term
  (the conversation history you resend each call — the API is stateless) and
  long-term (durable facts kept across sessions via the Memory tool).

Real-world applications:
- Personal assistants that remember preferences, long-running agents that keep
  notes across sessions, support bots that recall a customer's context.

When to use it:
- Short-term: any multi-turn conversation. Long-term: when state must survive
  process restarts and separate sessions.

Simple example:
- A ConversationManager that resends full history each turn (short-term memory).

Production example:
- The client-side Memory tool: Claude issues view/create/str_replace/insert/
  delete/rename commands against a /memories directory you implement.

Common mistakes:
- Forgetting the API is stateless — you MUST resend history each call.
- Storing secrets or PII in memory; not validating memory file paths.
- Letting history grow unbounded (cost) instead of caching or compacting.

Best practices:
- First message must be role 'user'; roles need not strictly alternate.
- Never store secrets/PII; confine memory paths to /memories and reject
  traversal (`..`, symlinks, absolute paths).
- For very long chats, cache a stable prefix or use server-side compaction.

Related concepts:
- chatbots, agents, workflows, guardrails.

Augmented Method learning cycle:
- Observe → Understand → Imitate → Modify → Predict → Build → Reflect → Teach
"""

from __future__ import annotations

import os

import anthropic

MODEL = "claude-opus-4-8"


class ConversationManager:
    """Short-term memory = resend the whole history each turn (stateless API)."""

    def __init__(self, system: str | None = None) -> None:
        self.client = anthropic.Anthropic()
        self.system = system
        self.messages: list[dict] = []  # first entry must be role 'user'

    def send(self, user_message: str) -> str:
        self.messages.append({"role": "user", "content": user_message})
        resp = self.client.messages.create(
            model=MODEL, max_tokens=16000, system=self.system, messages=self.messages,
        )
        reply = next((b.text for b in resp.content if b.type == "text"), "")
        self.messages.append({"role": "assistant", "content": reply})
        return reply


def simple_example() -> None:
    convo = ConversationManager(system="You are a helpful assistant.")
    print(convo.send("My name is Alice and I prefer metric units."))
    print(convo.send("What's my name and unit preference?"))  # recalls from history


def production_example() -> None:
    """Long-term memory via the client-side Memory tool. Claude issues file
    commands against /memories; YOU implement the backend (subclass
    BetaAbstractMemoryTool) and run it with the Tool Runner.

    SECURITY: never store secrets/PII; validate every path stays under /memories
    and reject traversal (`..`, symlinks, absolute paths outside the root).
    """
    client = anthropic.Anthropic()
    try:
        from anthropic.lib.tools import BetaAbstractMemoryTool
    except ImportError:
        print("Memory helper not available in this SDK version.")
        return

    class InMemoryStore(BetaAbstractMemoryTool):  # demo backend (use disk in prod)
        def __init__(self):
            super().__init__()
            self._files: dict[str, str] = {}

        def view(self, command):        return self._files.get(command.path, "")
        def create(self, command):      self._files[command.path] = command.file_text; return "ok"
        def str_replace(self, command): return "ok"
        def insert(self, command):      return "ok"
        def delete(self, command):      self._files.pop(command.path, None); return "ok"
        def rename(self, command):      return "ok"

    runner = client.beta.messages.tool_runner(
        model=MODEL, max_tokens=16000, tools=[InMemoryStore()],
        messages=[{"role": "user", "content": "Remember that I prefer Python."}],
    )
    for message in runner:
        for block in message.content:
            if block.type == "text":
                print(block.text)


COMPACTION_NOTE = """\
For very long conversations, enable server-side compaction so earlier context is
summarized automatically (beta 'compact-2026-01-12'):

    resp = client.beta.messages.create(
        betas=["compact-2026-01-12"],
        model="claude-opus-4-8", max_tokens=16000, messages=messages,
        context_management={"edits": [{"type": "compact_20260112"}]},
    )
    messages.append({"role": "assistant", "content": resp.content})  # keep FULL content
"""


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Set ANTHROPIC_API_KEY to run the memory examples.")
        print("\n" + COMPACTION_NOTE)
        return
    print("=== simple (short-term history) ===")
    simple_example()
    print("\n=== production (Memory tool) ===")
    production_example()
    print("\n" + COMPACTION_NOTE)


if __name__ == "__main__":
    main()
