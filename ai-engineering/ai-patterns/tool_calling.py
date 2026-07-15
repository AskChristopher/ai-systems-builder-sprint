"""tool_calling.py

Purpose:
- Let models invoke external tools for grounded actions.

Real-world applications:
- Search, calculations, data retrieval.

When to use it:
- When model needs external capabilities beyond text generation.

Simple example:
- Route a user request to a mock tool handler.

Production example:
- Register tools with schemas and validate calls/results.

Common mistakes:
- Blindly trusting arguments/results.

Best practices:
- Validate inputs, enforce auth, log tool usage.

Related concepts:
- Function calling, guardrails, workflows.
"""


def route_tool(name: str) -> str:
    return f"Dispatching tool: {name}"


def main() -> None:
    print(route_tool("search"))


if __name__ == "__main__":
    main()
