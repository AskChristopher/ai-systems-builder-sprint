"""json_schema.py

Purpose:
- Define and validate structured data contracts.

Real-world applications:
- API payload validation, model output constraints.

When to use it:
- Any time shape, types, and required fields matter.

Simple example:
- Validate object has `title` and `status`.

Production example:
- Versioned schemas with compatibility checks.

Common mistakes:
- Under-specified required fields.

Best practices:
- Keep schemas versioned and test with fixtures.

Related concepts:
- Structured outputs, function calling.
"""


def main() -> None:
    schema = {"type": "object", "properties": {"title": {"type": "string"}}}
    print(schema)


if __name__ == "__main__":
    main()
