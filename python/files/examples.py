"""Files examples."""

from pathlib import Path


def main() -> None:
    path = Path("example.txt")
    path.write_text("placeholder\n", encoding="utf-8")
    print(path.read_text(encoding="utf-8").strip())


if __name__ == "__main__":
    main()
