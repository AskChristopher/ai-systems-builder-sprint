"""Files solutions."""

from pathlib import Path


def main() -> None:
    path = Path("solution.txt")
    path.write_text("done", encoding="utf-8")
    print(path.exists())


if __name__ == "__main__":
    main()
