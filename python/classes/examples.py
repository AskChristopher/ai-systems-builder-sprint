"""Classes examples."""


class Learner:
    def __init__(self, name: str) -> None:
        self.name = name

    def introduce(self) -> str:
        return f"I am {self.name}."


def main() -> None:
    learner = Learner("AI Builder")
    print(learner.introduce())


if __name__ == "__main__":
    main()
