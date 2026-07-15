"""JSON examples."""

import json


def main() -> None:
    data = {"topic": "json", "status": "placeholder"}
    print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
