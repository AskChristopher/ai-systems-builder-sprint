"""image_generation.py

Purpose:
- IMPORTANT: Claude does NOT generate images. Claude's role in an image workflow
  is (1) writing/expanding rich image PROMPTS for a dedicated image model, and
  (2) analyzing/critiquing generated images via VISION input. The actual pixels
  come from a separate image-generation model you plug in.

Real-world applications:
- Creative tooling, marketing asset prototyping, iterative art direction where
  Claude drafts prompts and then evaluates results against a brief.

When to use it:
- You want strong prompt authoring and visual critique in the loop; pair it with
  a real image model for the generation step.

Simple example:
- Claude expands a short idea into a detailed image prompt.

Production example:
- Full loop: Claude drafts a prompt -> (placeholder) image model renders ->
  Claude evaluates the image via a vision message against the brief.

Common mistakes:
- Expecting Claude to return an image — it returns text/analysis only.
- Vague prompts with no style/composition constraints and no acceptance criteria.
- Sending base64 image data with newlines (must be a single unbroken string).

Best practices:
- Be explicit about subject, style, composition, lighting, and negatives in the
  generated prompt; define acceptance criteria and let Claude score the output.

Related concepts:
- prompt_engineering, guardrails, evaluation, document_qa (vision input).

Augmented Method learning cycle:
- Observe → Understand → Imitate → Modify → Predict → Build → Reflect → Teach
"""

from __future__ import annotations

import base64
import os

import anthropic

MODEL = "claude-opus-4-8"


def generate_image(prompt: str) -> bytes:
    """Placeholder for a real image-generation model. Returns PNG bytes."""
    # e.g. return my_image_model.render(prompt)
    return b"\x89PNG\r\n\x1a\n(pretend png bytes)"


def simple_example() -> None:
    """Claude expands a short idea into a detailed, model-ready image prompt."""
    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=MODEL, max_tokens=16000,
        system="You are an art director. Turn the idea into ONE detailed image "
               "prompt covering subject, style, composition, lighting, and "
               "negatives. Output only the prompt.",
        messages=[{"role": "user", "content": "a cozy reading nook, autumn mood"}],
    )
    print(next((b.text for b in resp.content if b.type == "text"), ""))


def evaluate_image(client: anthropic.Anthropic, png_bytes: bytes, brief: str) -> str:
    """Vision: send the generated image back to Claude for critique."""
    b64 = base64.standard_b64encode(png_bytes).decode("utf-8")  # no newlines
    resp = client.messages.create(
        model=MODEL, max_tokens=16000,
        messages=[{
            "role": "user",
            "content": [
                # Image/document blocks go BEFORE the text block.
                {"type": "image",
                 "source": {"type": "base64", "media_type": "image/png", "data": b64}},
                {"type": "text",
                 "text": f"Does this image satisfy the brief: '{brief}'? "
                         "Score 1-5 and list what to change."},
            ],
        }],
    )
    return next((b.text for b in resp.content if b.type == "text"), "")


def production_example() -> None:
    """Draft prompt -> render (placeholder) -> Claude evaluates via vision."""
    client = anthropic.Anthropic()
    brief = "a cozy reading nook, autumn mood, warm light"
    prompt_resp = client.messages.create(
        model=MODEL, max_tokens=16000,
        system="Output only a detailed image prompt.",
        messages=[{"role": "user", "content": brief}],
    )
    image_prompt = next((b.text for b in prompt_resp.content if b.type == "text"), "")
    png = generate_image(image_prompt)                 # your image model
    critique = evaluate_image(client, png, brief)      # Claude vision critique
    print(f"prompt: {image_prompt[:80]}...\n\ncritique:\n{critique}")


def main() -> None:
    print("image_generation.py — Claude authors prompts & critiques; it does NOT")
    print("generate images. Plug in a real image model at generate_image().")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Set ANTHROPIC_API_KEY to run the examples.")
        return
    print("\n=== simple (prompt authoring) ===")
    simple_example()
    print("\n=== production (author -> render -> critique) ===")
    production_example()


if __name__ == "__main__":
    main()
