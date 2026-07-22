# Python + AI Engineering Cheat Sheet

A quick-reference document that grows with your sprint.

---

## Variables

- Assignment: `name = "Chris"`
- Type inspection: `type(name)`
- Constants (convention): `UPPER_CASE`

## Collections

- List: `[]` (ordered, mutable)
- Tuple: `()` (ordered, immutable)
- Set: `{}` (unique values)
- Dict: `{key: value}`

## Functions

- Define: `def function_name(...):`
- Return values with `return`
- Document with docstrings

## Classes

- Define with `class`
- Constructor: `__init__`
- Instance methods include `self`

## Files

- Open safely with context manager:
  - `with open("file.txt", "r") as f:`
- Modes: `r`, `w`, `a`, `rb`, `wb`

## JSON

- Parse: `json.loads()` / `json.load()`
- Serialize: `json.dumps()` / `json.dump()`

## Exceptions

- `try` / `except` / `finally`
- Raise custom errors with `raise`

## Modules

- Import module: `import math`
- Import names: `from math import sqrt`
- Create reusable code in `.py` files

## HTTP / APIs

- Request methods: `GET`, `POST`, `PUT`, `DELETE`
- Status classes: `2xx`, `4xx`, `5xx`
- Parse JSON response bodies safely

## Async

- `async def` for coroutines
- `await` for async operations
- Event loop executes tasks

## Common AI Patterns

- Prompt templates
- Structured output with JSON schema
- Tool/function calling
- Retrieval augmented generation (RAG)
- Embeddings + vector search
- Guardrails + evaluation loops

---

## Additions Queue

Use this section to append shortcuts, patterns, and reminders as you learn.

- [ ] Add common debugging snippets
- [ ] Add testing quick references
- [ ] Add deployment checklists
