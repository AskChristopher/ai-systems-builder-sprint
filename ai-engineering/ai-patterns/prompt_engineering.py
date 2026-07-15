"""prompt_engineering.py

Purpose:
- Design prompts that improve model reliability and relevance.

Real-world applications:
- Chat assistants, coding copilots, content generation.

When to use it:
- Whenever model behavior depends heavily on instructions.

Simple example:
- Provide clear role, task, format, and constraints.

Production example:
- Use templated prompts with variables, guard instructions, and evaluation.

Common mistakes:
- Vague goals, conflicting instructions, missing output format.

Best practices:
- Be explicit, concise, and test across edge cases.

Related concepts:
- Structured outputs, evaluation, guardrails.

Augmented Method:
- Observe → Understand → Imitate → Modify → Predict → Build → Reflect → Teach
"""


def build_prompt(task: str) -> str:
    return f"You are a helpful AI engineer. Complete the task: {task}. Return concise bullet points."


def main() -> None:
    print(build_prompt("Explain tool calling basics"))


if __name__ == "__main__":
    main()
