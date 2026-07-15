"""Exceptions solutions."""


class ValidationError(Exception):
    """Custom validation error."""


def main() -> None:
    try:
        raise ValidationError("placeholder")
    except ValidationError as exc:
        print(exc)


if __name__ == "__main__":
    main()
