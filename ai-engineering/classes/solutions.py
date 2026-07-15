"""Classes solutions."""


class Item:
    def __init__(self, name: str) -> None:
        self.name = name


def main() -> None:
    print(Item("Notebook").name)


if __name__ == "__main__":
    main()
