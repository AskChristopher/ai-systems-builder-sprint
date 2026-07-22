"""voice.py

Purpose:
- Build a speech-enabled assistant. IMPORTANT: Claude is a text/vision model —
  it does NOT do speech-to-text or text-to-speech natively. Voice is a PIPELINE:
  STT (transcribe) -> Claude (understand + respond) -> TTS (synthesize).

Real-world applications:
- Voice assistants, phone/call automation, hands-free and accessibility tooling.

When to use it:
- Users interact by speaking. Claude sits in the middle doing the reasoning turn;
  you bring your own STT and TTS providers on either side.

Simple example:
- The three-stage pipeline with placeholder transcribe()/synthesize() and a
  Claude turn in the middle.

Production example:
- A streaming reply so audio synthesis can start before Claude finishes, plus
  notes on latency and barge-in (letting the user interrupt).

Common mistakes:
- Expecting Claude to accept audio or emit speech directly — it cannot.
- Ignoring transcription errors (confirm intent for high-stakes actions).
- Waiting for the full text before synthesizing (adds latency) — stream.

Best practices:
- Confirm ambiguous/high-stakes intents; keep replies short for TTS; stream so
  time-to-first-audio is low; support barge-in.

Related concepts:
- streaming, chatbots, guardrails.

Augmented Method learning cycle:
- Observe → Understand → Imitate → Modify → Predict → Build → Reflect → Teach
"""

from __future__ import annotations

import os

import anthropic

MODEL = "claude-opus-4-8"


# ---- Plug in a real STT/TTS provider here ---------------------------------

def transcribe(audio_path: str) -> str:
    """Placeholder STT. Replace with a real transcription service."""
    # e.g. return my_stt_provider.transcribe(audio_path)
    return "(pretend transcript) What's a good pasta for a weeknight dinner?"


def synthesize(text: str) -> bytes:
    """Placeholder TTS. Replace with a real speech synthesis service."""
    # e.g. return my_tts_provider.speak(text)
    return text.encode("utf-8")  # stand-in for audio bytes


# ---- Claude in the middle -------------------------------------------------

VOICE_SYSTEM = ("You are a voice assistant. Keep replies short and speakable — "
                "one or two sentences, no markdown, no lists.")


def simple_example() -> None:
    """transcribe -> Claude -> synthesize."""
    client = anthropic.Anthropic()
    user_text = transcribe("user_audio.wav")           # 1. STT
    resp = client.messages.create(                     # 2. Claude
        model=MODEL, max_tokens=16000, system=VOICE_SYSTEM,
        messages=[{"role": "user", "content": user_text}],
    )
    reply = next((b.text for b in resp.content if b.type == "text"), "")
    audio = synthesize(reply)                           # 3. TTS
    print(f"heard: {user_text}\nreply: {reply}\naudio bytes: {len(audio)}")


def production_example() -> None:
    """Stream Claude's reply so TTS can begin on the first sentence, cutting
    time-to-first-audio. For realtime voice, also support barge-in: if the user
    starts speaking, cancel the current synthesis and start a new STT turn."""
    client = anthropic.Anthropic()
    user_text = transcribe("user_audio.wav")
    buffer = ""
    with client.messages.stream(
        model=MODEL, max_tokens=16000, system=VOICE_SYSTEM,
        messages=[{"role": "user", "content": user_text}],
    ) as stream:
        for text in stream.text_stream:
            buffer += text
            # Flush a completed sentence to TTS as soon as it's ready.
            if buffer.endswith((".", "!", "?")):
                synthesize(buffer)  # speak this sentence now
                print(f"[speak] {buffer.strip()}")
                buffer = ""
    if buffer.strip():
        synthesize(buffer)
        print(f"[speak] {buffer.strip()}")


def main() -> None:
    print("voice.py — STT -> Claude -> TTS pipeline (Claude does the middle only)")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Set ANTHROPIC_API_KEY to run the Claude turn.")
        return
    print("\n=== simple ===")
    simple_example()
    print("\n=== production (streamed to TTS) ===")
    production_example()


if __name__ == "__main__":
    main()
