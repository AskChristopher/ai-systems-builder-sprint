"""function_calling.py

Purpose:
- Use structured function signatures for model-driven actions.

Real-world applications:
- Booking, form filling, backend operations.

When to use it:
- When actions must map to known functions and parameters.

Simple example:
- Parse arguments and call Python function.

Production example:
- Strict argument validation + retries + fallback.

Common mistakes:
- Missing schema constraints and poor error handling.

Best practices:
- Keep function contracts explicit and minimal.

Related concepts:
- Tool calling, JSON schema, structured outputs.
"""


def add(a: int, b: int) -> int:
    return a + b


def main() -> None:
    print(add(2, 3))


if __name__ == "__main__":
    main()
