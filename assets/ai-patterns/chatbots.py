"""chatbots.py

Purpose:
- Build a stateful, multi-turn conversational assistant on top of the stateless
  Messages API: you keep the history and resend it each turn, optionally with
  streaming, prompt caching, and compaction for long conversations.

Real-world applications:
- Support bots, internal assistants, tutors, onboarding guides.

When to use it:
- Users need interactive, contextful back-and-forth (as opposed to one-shot
  calls). Add retrieval/tools/guardrails as the use case grows.

Simple example:
- A ConversationManager that appends each user + assistant turn and resends.

Production example:
- Streaming replies + a cached system prompt to cut cost on every turn, with a
  note on server-side compaction for very long chats.

Common mistakes:
- Forgetting the API is stateless (must resend the full history each call).
- Unbounded history growth (cost) with no caching/compaction.
- First message not being role 'user'.

Best practices:
- Cache a large stable system prompt (cache_control) so repeated turns are cheap;
  stream for responsiveness; compact when nearing the context window; keep a
  clear escalation/fallback path.

Related concepts:
- memory, retrieval, guardrails, streaming.

Augmented Method learning cycle:
- Observe → Understand → Imitate → Modify → Predict → Build → Reflect → Teach
"""

from __future__ import annotations

import os

import anthropic

MODEL = "claude-opus-4-8"

SYSTEM_PROMPT = ("You are a friendly support assistant for a note-taking app. "
                 "Answer concisely and stay on topic.")


class ChatBot:
    """Stateless API -> we own the history and resend it every turn."""

    def __init__(self, system: str = SYSTEM_PROMPT) -> None:
        self.client = anthropic.Anthropic()
        self.system = system
        self.messages: list[dict] = []  # first message must be role 'user'

    def send(self, user_message: str) -> str:
        self.messages.append({"role": "user", "content": user_message})
        resp = self.client.messages.create(
            model=MODEL, max_tokens=16000, system=self.system, messages=self.messages,
        )
        reply = next((b.text for b in resp.content if b.type == "text"), "")
        self.messages.append({"role": "assistant", "content": reply})
        return reply

    def send_streaming(self, user_message: str) -> str:
        """Same, but stream tokens as they arrive; cache the system prompt."""
        self.messages.append({"role": "user", "content": user_message})
        chunks: list[str] = []
        with self.client.messages.stream(
            model=MODEL, max_tokens=16000,
            system=[{"type": "text", "text": self.system,
                     "cache_control": {"type": "ephemeral"}}],  # cache prefix
            messages=self.messages,
        ) as stream:
            for text in stream.text_stream:
                chunks.append(text)
                print(text, end="", flush=True)
        print()
        reply = "".join(chunks)
        self.messages.append({"role": "assistant", "content": reply})
        return reply


def simple_example() -> None:
    bot = ChatBot()
    print("bot:", bot.send("Hi! How do I export my notes?"))
    print("bot:", bot.send("And can I export just one notebook?"))  # keeps context


def production_example() -> None:
    bot = ChatBot()
    print("user: Do you support markdown?\nbot: ", end="")
    bot.send_streaming("Do you support markdown?")
    print("user: Give me a quick example.\nbot: ", end="")
    bot.send_streaming("Give me a quick example.")


COMPACTION_NOTE = """\
For very long conversations, enable server-side compaction (beta) so early
context is summarized automatically instead of growing unbounded:

    resp = client.beta.messages.create(
        betas=["compact-2026-01-12"],
        model="claude-opus-4-8", max_tokens=16000, messages=messages,
        context_management={"edits": [{"type": "compact_20260112"}]},
    )
    messages.append({"role": "assistant", "content": resp.content})  # keep FULL content
"""


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Set ANTHROPIC_API_KEY to run the chatbot examples.")
        print("\n" + COMPACTION_NOTE)
        return
    print("=== simple ===")
    simple_example()
    print("\n=== production (streaming + cached system) ===")
    production_example()
    print("\n" + COMPACTION_NOTE)


if __name__ == "__main__":
    main()
