"""Exceptions examples."""


def divide(a: float, b: float) -> float:
    if b == 0:
        raise ValueError("b must not be zero")
    return a / b


def main() -> None:
    try:
        print(divide(10, 2))
    except ValueError as exc:
        print(f"Error: {exc}")


if __name__ == "__main__":
    main()
